// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`LambdaUDF`] definitions for any_match function.

use arrow::{
    array::{Array, AsArray, BooleanArray, OffsetSizeTrait, new_null_array},
    datatypes::{DataType, Field, FieldRef},
};
use crate::array_transform::remove_list_null_values;
use datafusion_common::{
    Result, plan_err,
    utils::{list_values, take_function_args},
};
use datafusion_expr::{
    ColumnarValue, Documentation, LambdaFunctionArgs, LambdaReturnFieldArgs,
    LambdaSignature, LambdaUDF, ValueOrLambda, Volatility,
};
use datafusion_macros::user_doc;
use std::{any::Any, fmt::Debug, sync::Arc};

make_udlf_expr_and_func!(
    AnyMatch,
    any_match,
    array lambda,
    "returns true if any element in the array satisfies the predicate",
    any_match_udlf
);

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns true if any element in the array satisfies the predicate lambda.",
    syntax_example = "any_match(array, x -> x > 0)",
    sql_example = r#"```sql
> select any_match([1, 2, 3], x -> x > 2);
+-------------------------------------+
| any_match([1, 2, 3], x -> x > 2)   |
+-------------------------------------+
| true                                |
+-------------------------------------+
```"#,
    argument(
        name = "array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    ),
    argument(name = "lambda", description = "Lambda predicate returning a boolean")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct AnyMatch {
    signature: LambdaSignature,
}

impl Default for AnyMatch {
    fn default() -> Self {
        Self::new()
    }
}

impl AnyMatch {
    pub fn new() -> Self {
        Self {
            signature: LambdaSignature::user_defined(Volatility::Immutable),
        }
    }
}

impl LambdaUDF for AnyMatch {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "any_match"
    }

    fn signature(&self) -> &LambdaSignature {
        &self.signature
    }

    fn short_circuits(&self) -> bool {
        true
    }

    fn coerce_value_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let list = if arg_types.len() == 1 {
            &arg_types[0]
        } else {
            return plan_err!(
                "{} function requires 1 value arguments, got {}",
                self.name(),
                arg_types.len()
            );
        };

        let coerced = match list {
            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _) => list.clone(),
            DataType::ListView(field) => DataType::List(Arc::clone(field)),
            DataType::LargeListView(field) => DataType::LargeList(Arc::clone(field)),
            _ => {
                return plan_err!(
                    "{} expected a list as first argument, got {}",
                    self.name(),
                    list
                );
            }
        };

        Ok(vec![coerced])
    }

    fn lambdas_parameters(&self, value_fields: &[FieldRef]) -> Result<Vec<Vec<Field>>> {
        let list = if value_fields.len() == 1 {
            &value_fields[0]
        } else {
            return plan_err!(
                "{} function requires 1 value arguments, got {}",
                self.name(),
                value_fields.len()
            );
        };

        let field = match list.data_type() {
            DataType::List(field) => field,
            DataType::LargeList(field) => field,
            DataType::FixedSizeList(field, _) => field,
            _ => return plan_err!("expected list, got {list}"),
        };

        let value = Field::new("", field.data_type().clone(), field.is_nullable())
            .with_metadata(field.metadata().clone());

        Ok(vec![vec![value]])
    }

    fn return_field_from_args(&self, args: LambdaReturnFieldArgs) -> Result<Arc<Field>> {
        let (list, _lambda) = value_lambda_pair(self.name(), args.arg_fields)?;

        // result is nullable if the list itself is nullable
        Ok(Arc::new(Field::new("", DataType::Boolean, list.is_nullable())))
    }

    fn invoke_with_args(&self, args: LambdaFunctionArgs) -> Result<ColumnarValue> {
        let (list, lambda) = value_lambda_pair(self.name(), &args.args)?;

        let list_array = list.to_array(args.number_rows)?;

        // null list rows produce null output
        if list_array.null_count() == list_array.len() {
            return Ok(ColumnarValue::Array(new_null_array(
                &DataType::Boolean,
                list_array.len(),
            )));
        }

        // null sublists may contain values that cause the predicate to fail (e.g. 1/x with x=0)
        let list_array = remove_list_null_values(&list_array)?;

        let list_values = list_values(&list_array)?;

        // evaluate the predicate on all values at once (vectorized)
        let values_param = || Ok(Arc::clone(&list_values));
        let predicate_result = lambda
            .evaluate(&[&values_param])?
            .into_array(list_values.len())?;

        let predicate_bools = predicate_result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                datafusion_common::exec_datafusion_err!(
                    "{} lambda must return a boolean, got {}",
                    self.name(),
                    predicate_result.data_type()
                )
            })?;

        // for each row in the list, check if any element's predicate is true
        let result: BooleanArray = match list_array.data_type() {
            DataType::List(_) => {
                let list = list_array.as_list::<i32>();
                any_match_over_offsets(list.offsets(), predicate_bools, list.nulls())
            }
            DataType::LargeList(_) => {
                let list = list_array.as_list::<i64>();
                any_match_over_offsets(list.offsets(), predicate_bools, list.nulls())
            }
            DataType::FixedSizeList(_, size) => {
                let fsl = list_array.as_fixed_size_list();
                any_match_over_fixed_size(
                    fsl.len(),
                    *size,
                    predicate_bools,
                    fsl.nulls(),
                )
            }
            other => {
                return datafusion_common::exec_err!("expected list, got {other}");
            }
        };

        Ok(ColumnarValue::Array(Arc::new(result)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn value_lambda_pair<'a, V: Debug, L: Debug>(
    name: &str,
    args: &'a [ValueOrLambda<V, L>],
) -> Result<(&'a V, &'a L)> {
    let [value, lambda] = take_function_args(name, args)?;

    let (ValueOrLambda::Value(value), ValueOrLambda::Lambda(lambda)) = (value, lambda)
    else {
        return plan_err!(
            "{name} expects a value followed by a lambda, got {value:?} and {lambda:?}"
        );
    };

    Ok((value, lambda))
}

/// For each row in a List/LargeList, returns true if any element's predicate is true.
fn any_match_over_offsets<O: OffsetSizeTrait>(
    offsets: &arrow::buffer::OffsetBuffer<O>,
    predicate: &BooleanArray,
    nulls: Option<&arrow::buffer::NullBuffer>,
) -> BooleanArray {
    let mut builder = arrow::array::BooleanBuilder::new();

    for (i, window) in offsets.windows(2).enumerate() {
        if nulls.map_or(false, |n| n.is_null(i)) {
            builder.append_null();
            continue;
        }
        let start = window[0].as_usize();
        let end = window[1].as_usize();
        let any = (start..end).any(|j| predicate.value(j) && !predicate.is_null(j));
        builder.append_value(any);
    }

    builder.finish()
}

/// For each row in a FixedSizeList, returns true if any element's predicate is true.
fn any_match_over_fixed_size(
    len: usize,
    size: i32,
    predicate: &BooleanArray,
    nulls: Option<&arrow::buffer::NullBuffer>,
) -> BooleanArray {
    let size = size as usize;
    let mut builder = arrow::array::BooleanBuilder::new();

    for i in 0..len {
        if nulls.map_or(false, |n| n.is_null(i)) {
            builder.append_null();
            continue;
        }
        let start = i * size;
        let end = start + size;
        let any = (start..end).any(|j| predicate.value(j) && !predicate.is_null(j));
        builder.append_value(any);
    }

    builder.finish()
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow::{
        array::{Array, ArrayRef, AsArray, BooleanArray, Int32Array, ListArray},
        buffer::{NullBuffer, OffsetBuffer},
        datatypes::{DataType, Field},
    };
    use datafusion_common::{DFSchema, Result};
    use datafusion_expr::{
        Expr, col, execution_props::ExecutionProps, expr::LambdaFunction, lambda,
        lambda_var, lit,
    };
    use datafusion_physical_expr::create_physical_expr;

    use crate::array_any_match::any_match_udlf;

    fn create_i32_list(
        values: impl Into<Int32Array>,
        offsets: OffsetBuffer<i32>,
        nulls: Option<NullBuffer>,
    ) -> ListArray {
        let list_field = Arc::new(Field::new_list_field(DataType::Int32, true));
        ListArray::new(list_field, offsets, Arc::new(values.into()), nulls)
    }

    fn any_greater_than_zero(list: impl Array + Clone + 'static) -> Result<ArrayRef> {
        let any_match = any_match_udlf();

        let schema = DFSchema::from_unqualified_fields(
            vec![Field::new("list", list.data_type().clone(), list.is_nullable())].into(),
            HashMap::new(),
        )?;

        create_physical_expr(
            &Expr::LambdaFunction(LambdaFunction::new(
                any_match,
                vec![
                    col("list"),
                    lambda(
                        ["v"],
                        lambda_var("v", Arc::new(Field::new("v", DataType::Int32, true)))
                            .gt(lit(0i32)),
                    ),
                ],
            )),
            &schema,
            &ExecutionProps::new(),
        )?
        .evaluate(&arrow::array::RecordBatch::try_new(
            Arc::clone(schema.inner()),
            vec![Arc::new(list.clone())],
        )?)?
        .into_array(list.len())
    }

    #[test]
    fn any_match_basic() {
        let list = create_i32_list(
            vec![-1, -2, 3],
            OffsetBuffer::<i32>::from_lengths(vec![3]),
            None,
        );

        let res = any_greater_than_zero(list).unwrap();
        let actual = res.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(actual, &BooleanArray::from(vec![true]));
    }

    #[test]
    fn any_match_all_negative() {
        let list = create_i32_list(
            vec![-1, -2, -3],
            OffsetBuffer::<i32>::from_lengths(vec![3]),
            None,
        );

        let res = any_greater_than_zero(list).unwrap();
        let actual = res.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(actual, &BooleanArray::from(vec![false]));
    }

    #[test]
    fn any_match_null_list_row_returns_null() {
        let list = create_i32_list(
            vec![1, 2, 3, 4],
            OffsetBuffer::<i32>::from_lengths(vec![2, 2]),
            Some(NullBuffer::from(vec![false, true])),
        );

        let res = any_greater_than_zero(list).unwrap();
        let actual = res.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(
            actual,
            &BooleanArray::from(vec![None, Some(true)])
        );
    }

    #[test]
    fn any_match_multiple_rows() {
        // row 0: [-1, -2] -> false
        // row 1: [-1, 5]  -> true
        // row 2: [0]      -> false
        let list = create_i32_list(
            vec![-1, -2, -1, 5, 0],
            OffsetBuffer::<i32>::from_lengths(vec![2, 2, 1]),
            None,
        );

        let res = any_greater_than_zero(list).unwrap();
        let actual = res.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(
            actual,
            &BooleanArray::from(vec![false, true, false])
        );
    }

    #[test]
    fn any_match_empty_sublist() {
        let list = create_i32_list(
            vec![0i32; 0],
            OffsetBuffer::<i32>::from_lengths(vec![0]),
            None,
        );

        let res = any_greater_than_zero(list).unwrap();
        let actual = res.as_any().downcast_ref::<BooleanArray>().unwrap();

        // no elements -> no match
        assert_eq!(actual, &BooleanArray::from(vec![false]));
    }
}

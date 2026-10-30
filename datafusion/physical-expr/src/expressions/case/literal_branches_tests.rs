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

use super::*;
use crate::expressions::BinaryExpr;
use datafusion_expr::Operator;

fn batch(keys: ArrayRef, values: ArrayRef) -> RecordBatch {
    RecordBatch::try_from_iter(vec![
        ("unused", Arc::clone(&keys)),
        ("key", keys),
        ("value", values),
    ])
    .unwrap()
}

fn expression(
    whens: Vec<ScalarValue>,
    otherwise: Option<Arc<dyn PhysicalExpr>>,
) -> CaseExpr {
    CaseExpr::try_new(
        Some(Arc::new(Column::new("key", 1))),
        whens
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                (
                    lit(value),
                    Arc::new(BinaryExpr::new(
                        Arc::new(Column::new("value", 2)),
                        Operator::Plus,
                        lit(i as i64),
                    )) as Arc<dyn PhysicalExpr>,
                )
            })
            .collect(),
        otherwise,
    )
    .unwrap()
}

fn compare(expr: &CaseExpr, batch: &RecordBatch) {
    let mut branches = expr.body.when_then_expr.clone();
    while branches.len() < 9 {
        branches.push(branches[0].clone());
    }
    let routed = CaseExpr::try_new(
        expr.body.expr.clone(),
        branches,
        expr.body.else_expr.clone(),
    )
    .unwrap();
    assert!(matches!(
        routed.eval_method,
        EvalMethod::WithLiteralWhen(..)
    ));
    let expected = expr
        .case_when_with_expr(batch, &expr.body.project().unwrap(), None)
        .unwrap()
        .into_array(batch.num_rows())
        .unwrap();
    for actual in [expr.evaluate(batch), routed.evaluate(batch)] {
        assert_eq!(
            actual
                .unwrap()
                .into_array(batch.num_rows())
                .unwrap()
                .as_ref(),
            expected.as_ref()
        );
    }
}

#[test]
fn routes_null_duplicate_sparse_and_missing_keys() {
    let values = Arc::new(Int64Array::from(vec![
        Some(10),
        None,
        Some(30),
        Some(40),
        Some(50),
        Some(60),
    ]));
    let input = batch(
        Arc::new(Int64Array::from(vec![
            Some(-12),
            Some(400),
            Some(-12),
            None,
            Some(999),
            Some(400),
        ])),
        values,
    );
    for otherwise in [
        None,
        Some(lit(-1_i64)),
        Some(lit(-1_i32)),
        Some(Arc::new(Column::new("value", 2)) as Arc<dyn PhysicalExpr>),
    ] {
        let expr = expression(
            vec![
                ScalarValue::Int64(None),
                ScalarValue::Int64(Some(-12)),
                ScalarValue::Int64(Some(-12)),
                ScalarValue::Int64(Some(400)),
            ],
            otherwise,
        );
        compare(&expr, &input);
        compare(&expr, &input.slice(0, 0));
        compare(&expr, &input.slice(2, 3));
    }
}

#[test]
fn routes_strings_booleans_and_dictionary_keys() {
    let values: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40]));
    for keys in [
        Arc::new(StringArray::from(vec![
            Some("red"),
            None,
            Some("blue"),
            Some("other"),
        ])) as ArrayRef,
        Arc::new(StringViewArray::from(vec![
            Some("red"),
            None,
            Some("blue"),
            Some("other"),
        ])) as ArrayRef,
    ] {
        let kind = keys.data_type().clone();
        let whens = ["red", "blue"]
            .into_iter()
            .map(|v| ScalarValue::Utf8(Some(v.into())).cast_to(&kind).unwrap())
            .collect();
        compare(
            &expression(whens, Some(lit(-1_i64))),
            &batch(keys, Arc::clone(&values)),
        );
    }
    let keys: ArrayRef = Arc::new(BooleanArray::from(vec![
        Some(true),
        Some(false),
        None,
        Some(true),
    ]));
    compare(
        &expression(
            vec![
                ScalarValue::Boolean(Some(true)),
                ScalarValue::Boolean(Some(false)),
            ],
            None,
        ),
        &batch(keys, Arc::clone(&values)),
    );
    let mut builder = StringDictionaryBuilder::<arrow::datatypes::Int8Type>::new();
    builder.append("red").unwrap();
    builder.append_null();
    builder.append("blue").unwrap();
    builder.append("other").unwrap();
    compare(
        &expression(
            vec![
                ScalarValue::Utf8(Some("red".into())),
                ScalarValue::Utf8(Some("blue".into())),
            ],
            None,
        ),
        &batch(Arc::new(builder.finish()), values),
    );
}

#[test]
fn evaluates_fallible_branches_only_for_matching_rows() {
    let divide = Arc::new(BinaryExpr::new(
        lit(100_i64),
        Operator::Divide,
        Arc::new(Column::new("value", 2)),
    )) as Arc<dyn PhysicalExpr>;
    let input = batch(
        Arc::new(Int64Array::from(vec![1, 2, 3, 2])),
        Arc::new(Int64Array::from(vec![10, 0, 5, 0])),
    );
    let expr = CaseExpr::try_new(
        Some(Arc::new(Column::new("key", 1))),
        vec![
            (lit(-2_i64), Arc::clone(&divide)),
            (lit(-1_i64), Arc::clone(&divide)),
            (lit(1_i64), Arc::clone(&divide)),
            (lit(2_i64), lit(5_i64)),
            (lit(1_i64), Arc::clone(&divide)),
            (lit(2_i64), lit(5_i64)),
            (lit(2_i64), lit(5_i64)),
        ],
        Some(divide),
    )
    .unwrap();
    compare(&expr, &input);
    assert_eq!(
        expr.evaluate(&input)
            .unwrap()
            .into_array(4)
            .unwrap()
            .as_primitive::<arrow::datatypes::Int64Type>()
            .values()
            .as_ref(),
        &[10, 5, 20, 5]
    );
}

#[test]
fn nonliteral_when_preserves_sequential_evaluation() {
    let mut branches: Vec<WhenThen> =
        (1..9).map(|i| (lit(i as i64), lit(i as i64))).collect();
    branches.push((Arc::new(Column::new("key", 1)), lit(9_i64)));
    let expr = CaseExpr::try_new(Some(lit(1_i64)), branches, None).unwrap();
    assert!(matches!(expr.eval_method, EvalMethod::WithExpression(..)));
}

#[test]
fn routes_expression_results_with_many_branches() {
    for branches in [2, 4, 8, 32, 64, 128] {
        let keys = Int64Array::from_iter((0..8192).map(|i| {
            if i % 17 == 0 {
                None
            } else {
                Some(i % (branches + 3))
            }
        }));
        let input = batch(
            Arc::new(keys),
            Arc::new(Int64Array::from_iter_values(0..8192)),
        );
        let expr = expression(
            (0..branches).map(|i| ScalarValue::Int64(Some(i))).collect(),
            None,
        );
        compare(&expr, &input);
    }
}

#[test]
fn routes_constant_null_and_floating_keys() {
    for keys in [
        vec![None; 8],
        vec![Some(0.0); 8],
        vec![
            Some(-0.0),
            Some(0.0),
            Some(f64::NAN),
            Some(f64::from_bits(0x7ff8_0000_0000_0001)),
            None,
            Some(10.0),
            Some(1.0),
            Some(2.0),
        ],
    ] {
        let input = batch(
            Arc::new(Float64Array::from(keys)),
            Arc::new(Int64Array::from_iter_values(0..8)),
        );
        let expr = expression(
            vec![
                ScalarValue::Float64(Some(-0.0)),
                ScalarValue::Float64(Some(0.0)),
                ScalarValue::Float64(Some(f64::NAN)),
                ScalarValue::Float64(None),
            ],
            Some(Arc::new(Column::new("value", 2))),
        );
        compare(&expr, &input);
    }
}

#[test]
fn short_literal_case_uses_sequential_comparisons() {
    for branches in 1..=8 {
        let expr = expression(
            (0..branches).map(|i| ScalarValue::Int64(Some(i))).collect(),
            None,
        );
        assert!(matches!(expr.eval_method, EvalMethod::WithExpression(..)));
    }
}

#[test]
fn routes_constant_dictionary_and_scalar_inputs() {
    let values: ArrayRef = Arc::new(Int64Array::from_iter_values(0..8));
    for value in [Some("red"), None] {
        let keys = (0..8)
            .map(|_| value)
            .collect::<DictionaryArray<arrow::datatypes::Int8Type>>();
        let input = batch(Arc::new(keys), Arc::clone(&values));
        let expr = expression(
            vec![
                ScalarValue::Utf8(Some("red".into())),
                ScalarValue::Utf8(Some("blue".into())),
            ],
            None,
        );
        compare(&expr, &input);
    }
    let input = batch(Arc::new(Int64Array::from_iter_values(0..8)), values);
    let expr = expression((0..12).map(|i| ScalarValue::Int64(Some(i))).collect(), None);
    let scalar =
        CaseExpr::try_new(Some(lit(3_i64)), expr.body.when_then_expr, None).unwrap();
    compare(&scalar, &input);
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct GeneratedKeys;

impl std::fmt::Display for GeneratedKeys {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("generated_keys")
    }
}

impl PhysicalExpr for GeneratedKeys {
    fn data_type(&self, _: &Schema) -> Result<DataType> {
        Ok(DataType::Int64)
    }
    fn nullable(&self, _: &Schema) -> Result<bool> {
        Ok(false)
    }
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Array(Arc::new(
            Int64Array::from_iter_values((0..batch.num_rows()).map(|i| (i % 11) as i64)),
        )))
    }
    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![]
    }
    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(self)
    }
    fn fmt_sql(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

#[test]
fn preserves_row_count_without_input_columns() {
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(32)),
    )
    .unwrap();
    let expr = CaseExpr::try_new(
        Some(Arc::new(GeneratedKeys)),
        (0..9)
            .map(|i| {
                (
                    lit(i as i64),
                    Arc::new(BinaryExpr::new(
                        Arc::new(GeneratedKeys),
                        Operator::Plus,
                        lit(10_i64),
                    )) as Arc<dyn PhysicalExpr>,
                )
            })
            .collect(),
        Some(lit(-1_i64)),
    )
    .unwrap();
    compare(&expr, &batch);
    let actual = expr.evaluate(&batch).unwrap().into_array(32).unwrap();
    let expected = Int64Array::from_iter_values(
        (0..32).map(|i| if i % 11 < 9 { i / 11 + 10 } else { -1 }),
    );
    assert_eq!(actual.as_ref(), &expected as &dyn Array);
}

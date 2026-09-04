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

//! Defines physical expressions that can evaluated at runtime during query execution

use crate::hyperloglog::{HLL_HASH_STATE, HyperLogLog};
use arrow::array::{Array, BinaryArray, StringViewArray};
use arrow::array::{
    GenericBinaryArray, GenericStringArray, OffsetSizeTrait, PrimitiveArray,
};
use arrow::datatypes::{
    ArrowPrimitiveType, Date32Type, Date64Type, FieldRef, Int32Type, Int64Type,
    Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType,
    TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt32Type, UInt64Type,
};
use arrow::{array::ArrayRef, datatypes::DataType, datatypes::Field};
use datafusion_common::ScalarValue;
use datafusion_common::{
    DataFusionError, Result, downcast_value, internal_datafusion_err, internal_err,
    not_impl_err,
};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, Signature, Volatility,
};
use datafusion_functions_aggregate_common::aggregate::count_distinct::{
    Bitmap65536DistinctCountAccumulator, Bitmap65536DistinctCountAccumulatorI16,
    BoolArray256DistinctCountAccumulator, BoolArray256DistinctCountAccumulatorI8,
};
use datafusion_functions_aggregate_common::noop_accumulator::NoopAccumulator;
use datafusion_macros::user_doc;
use std::fmt::{Debug, Formatter};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

make_udaf_expr_and_func!(
    ApproxDistinct,
    approx_distinct,
    expression,
    "approximate number of distinct input values",
    approx_distinct_udaf
);

impl<T: Hash + ?Sized> From<&HyperLogLog<T>> for ScalarValue {
    fn from(v: &HyperLogLog<T>) -> ScalarValue {
        let values = v.as_ref().to_vec();
        ScalarValue::Binary(Some(values))
    }
}

impl<T: Hash + ?Sized> TryFrom<&[u8]> for HyperLogLog<T> {
    type Error = DataFusionError;
    fn try_from(v: &[u8]) -> Result<HyperLogLog<T>> {
        let arr: [u8; 16384] = v.try_into().map_err(|_| {
            internal_datafusion_err!("Impossibly got invalid binary array from states")
        })?;
        Ok(HyperLogLog::<T>::new_with_registers(arr))
    }
}

impl<T: Hash + ?Sized> TryFrom<&ScalarValue> for HyperLogLog<T> {
    type Error = DataFusionError;
    fn try_from(v: &ScalarValue) -> Result<HyperLogLog<T>> {
        if let ScalarValue::Binary(Some(slice)) = v {
            slice.as_slice().try_into()
        } else {
            internal_err!(
                "Impossibly got invalid scalar value while converting to HyperLogLog"
            )
        }
    }
}

#[derive(Debug)]
struct ApproxDistinctBitmapWrapper<A: Accumulator> {
    inner: A,
}

impl<A: Accumulator> Accumulator for ApproxDistinctBitmapWrapper<A> {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.inner.update_batch(values)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        match self.inner.evaluate()? {
            ScalarValue::Int64(Some(v)) => Ok(ScalarValue::UInt64(Some(v as u64))),
            other => internal_err!("unexpected: {other}"),
        }
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.inner.state()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.inner.merge_batch(states)
    }
}

#[derive(Debug)]
struct NumericHLLAccumulator<T>
where
    T: ArrowPrimitiveType,
    T::Native: Hash,
{
    hll: HyperLogLog<T::Native>,
}

impl<T> NumericHLLAccumulator<T>
where
    T: ArrowPrimitiveType,
    T::Native: Hash,
{
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
        }
    }
}

#[derive(Debug)]
struct StringHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    hll: HyperLogLog<str>,
    phantom_data: PhantomData<T>,
}

impl<T> StringHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
            phantom_data: PhantomData,
        }
    }
}

#[derive(Debug)]
struct StringViewHLLAccumulator {
    hll: HyperLogLog<str>,
}

impl StringViewHLLAccumulator {
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
        }
    }
}

#[derive(Debug)]
struct BinaryHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    hll: HyperLogLog<[u8]>,
    phantom_data: PhantomData<T>,
}

impl<T> BinaryHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
            phantom_data: PhantomData,
        }
    }
}

macro_rules! default_accumulator_impl {
    () => {
        fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
            assert_eq!(1, states.len(), "expect only 1 element in the states");
            let binary_array = downcast_value!(states[0], BinaryArray);
            for v in binary_array.iter() {
                let v = v.ok_or_else(|| {
                    internal_datafusion_err!(
                        "Impossibly got empty binary array from states"
                    )
                })?;
                let other = v.try_into()?;
                self.hll.merge(&other);
            }
            Ok(())
        }

        fn state(&mut self) -> Result<Vec<ScalarValue>> {
            let value = ScalarValue::from(&self.hll);
            Ok(vec![value])
        }

        fn evaluate(&mut self) -> Result<ScalarValue> {
            Ok(ScalarValue::UInt64(Some(self.hll.count() as u64)))
        }

        fn size(&self) -> usize {
            // HLL has static size
            std::mem::size_of_val(self)
        }
    };
}

impl<T> Accumulator for BinaryHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &GenericBinaryArray<T> =
            downcast_value!(values[0], GenericBinaryArray, T);
        // flatten because we would skip nulls
        self.hll.extend(array.into_iter().flatten());
        Ok(())
    }

    default_accumulator_impl!();
}

impl Accumulator for StringViewHLLAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &StringViewArray = downcast_value!(values[0], StringViewArray);

        // When all strings are stored inline in the StringView (≤ 12 bytes),
        // hash the raw u128 view directly instead of materializing a &str.
        if array.data_buffers().is_empty() {
            for (i, &view) in array.views().iter().enumerate() {
                if !array.is_null(i) {
                    self.hll.add_hashed(HLL_HASH_STATE.hash_one(view));
                }
            }
        } else {
            self.hll.extend(array.iter().flatten());
        }

        Ok(())
    }

    default_accumulator_impl!();
}

impl<T> Accumulator for StringHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &GenericStringArray<T> =
            downcast_value!(values[0], GenericStringArray, T);
        // flatten because we would skip nulls
        self.hll.extend(array.into_iter().flatten());
        Ok(())
    }

    default_accumulator_impl!();
}

impl<T> Accumulator for NumericHLLAccumulator<T>
where
    T: ArrowPrimitiveType + Debug,
    T::Native: Hash,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &PrimitiveArray<T> = downcast_value!(values[0], PrimitiveArray, T);
        // flatten because we would skip nulls
        self.hll.extend(array.into_iter().flatten());
        Ok(())
    }

    default_accumulator_impl!();
}

impl Debug for ApproxDistinct {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApproxDistinct")
            .field("name", &self.name())
            .field("signature", &self.signature)
            .finish()
    }
}

impl Default for ApproxDistinct {
    fn default() -> Self {
        Self::new()
    }
}

#[user_doc(
    doc_section(label = "Approximate Functions"),
    description = "Returns the approximate number of distinct input values calculated using the HyperLogLog algorithm.",
    syntax_example = "approx_distinct(expression)",
    sql_example = r#"```sql
> SELECT approx_distinct(column_name) FROM table_name;
+-----------------------------------+
| approx_distinct(column_name)      |
+-----------------------------------+
| 42                                |
+-----------------------------------+
```"#,
    standard_argument(name = "expression",)
)]
#[derive(PartialEq, Eq, Hash)]
pub struct ApproxDistinct {
    signature: Signature,
}

impl ApproxDistinct {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

#[cold]
fn get_small_int_approx_accumulator(
    data_type: &DataType,
) -> Result<Box<dyn Accumulator>> {
    match data_type {
        DataType::UInt8 => Ok(Box::new(ApproxDistinctBitmapWrapper {
            inner: BoolArray256DistinctCountAccumulator::new(),
        })),
        DataType::Int8 => Ok(Box::new(ApproxDistinctBitmapWrapper {
            inner: BoolArray256DistinctCountAccumulatorI8::new(),
        })),
        DataType::UInt16 => Ok(Box::new(ApproxDistinctBitmapWrapper {
            inner: Bitmap65536DistinctCountAccumulator::new(),
        })),
        DataType::Int16 => Ok(Box::new(ApproxDistinctBitmapWrapper {
            inner: Bitmap65536DistinctCountAccumulatorI16::new(),
        })),
        _ => internal_err!("unsupported small int type: {}", data_type),
    }
}

#[cold]
fn get_small_int_state_field(name: &str, data_type: &DataType) -> Result<Vec<FieldRef>> {
    Ok(vec![
        Field::new_list(
            format_state_name(name, "approx_distinct"),
            Field::new_list_field(data_type.clone(), true),
            false,
        )
        .into(),
    ])
}

impl AggregateUDFImpl for ApproxDistinct {
    fn name(&self) -> &str {
        "approx_distinct"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt64)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let data_type = args.input_fields[0].data_type();
        match data_type {
            DataType::Null => Ok(vec![
                Field::new(
                    format_state_name(args.name, self.name()),
                    DataType::Null,
                    true,
                )
                .into(),
            ]),
            DataType::UInt8 | DataType::Int8 | DataType::UInt16 | DataType::Int16 => {
                get_small_int_state_field(args.name, data_type)
            }
            DataType::Dictionary(_, _) if is_supported_type(data_type) => {
                let value_type = dictionary_value_type(data_type);
                if is_fixed_domain_type(value_type) {
                    get_small_int_state_field(args.name, value_type)
                } else {
                    Ok(vec![
                        Field::new(
                            format_state_name(args.name, "hll_registers"),
                            DataType::Binary,
                            false,
                        )
                        .into(),
                    ])
                }
            }
            _ => Ok(vec![
                Field::new(
                    format_state_name(args.name, "hll_registers"),
                    DataType::Binary,
                    false,
                )
                .into(),
            ]),
        }
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let data_type = acc_args.expr_fields[0].data_type();

        let accumulator: Box<dyn Accumulator> = match data_type {
            DataType::UInt8 | DataType::Int8 | DataType::UInt16 | DataType::Int16 => {
                return get_small_int_approx_accumulator(data_type);
            }
            DataType::UInt32 => Box::new(NumericHLLAccumulator::<UInt32Type>::new()),
            DataType::UInt64 => Box::new(NumericHLLAccumulator::<UInt64Type>::new()),
            DataType::Int32 => Box::new(NumericHLLAccumulator::<Int32Type>::new()),
            DataType::Int64 => Box::new(NumericHLLAccumulator::<Int64Type>::new()),
            DataType::Date32 => Box::new(NumericHLLAccumulator::<Date32Type>::new()),
            DataType::Date64 => Box::new(NumericHLLAccumulator::<Date64Type>::new()),
            DataType::Time32(TimeUnit::Second) => {
                Box::new(NumericHLLAccumulator::<Time32SecondType>::new())
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                Box::new(NumericHLLAccumulator::<Time32MillisecondType>::new())
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                Box::new(NumericHLLAccumulator::<Time64MicrosecondType>::new())
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                Box::new(NumericHLLAccumulator::<Time64NanosecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                Box::new(NumericHLLAccumulator::<TimestampSecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                Box::new(NumericHLLAccumulator::<TimestampMillisecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                Box::new(NumericHLLAccumulator::<TimestampMicrosecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                Box::new(NumericHLLAccumulator::<TimestampNanosecondType>::new())
            }
            DataType::Utf8 => Box::new(StringHLLAccumulator::<i32>::new()),
            DataType::LargeUtf8 => Box::new(StringHLLAccumulator::<i64>::new()),
            DataType::Utf8View => Box::new(StringViewHLLAccumulator::new()),
            DataType::Binary => Box::new(BinaryHLLAccumulator::<i32>::new()),
            DataType::LargeBinary => Box::new(BinaryHLLAccumulator::<i64>::new()),
            DataType::Dictionary(_, _) if is_supported_type(data_type) => {
                let value_type = dictionary_value_type(data_type).clone();
                let inner = make_approx_distinct_accumulator(&value_type)?;
                Box::new(DictionaryAccumulator { inner, value_type })
            }
            DataType::Null => {
                Box::new(NoopAccumulator::new(ScalarValue::UInt64(Some(0))))
            }
            other => {
                return not_impl_err!(
                    "Support for 'approx_distinct' for data type {other} is not implemented"
                );
            }
        };
        Ok(accumulator)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[derive(Debug)]
struct DictionaryAccumulator {
    inner: Box<dyn Accumulator>,
    value_type: DataType,
}

impl Accumulator for DictionaryAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let plain = arrow::compute::cast(&values[0], &self.value_type)?;
        self.inner.update_batch(&[plain])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        self.inner.evaluate()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.inner.state()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.inner.merge_batch(states)
    }
}

fn make_approx_distinct_accumulator(
    data_type: &DataType,
) -> Result<Box<dyn Accumulator>> {
    match data_type {
        DataType::UInt8 | DataType::Int8 | DataType::UInt16 | DataType::Int16 => {
            get_small_int_approx_accumulator(data_type)
        }
        DataType::UInt32 => Ok(Box::new(NumericHLLAccumulator::<UInt32Type>::new())),
        DataType::UInt64 => Ok(Box::new(NumericHLLAccumulator::<UInt64Type>::new())),
        DataType::Int32 => Ok(Box::new(NumericHLLAccumulator::<Int32Type>::new())),
        DataType::Int64 => Ok(Box::new(NumericHLLAccumulator::<Int64Type>::new())),
        DataType::Date32 => Ok(Box::new(NumericHLLAccumulator::<Date32Type>::new())),
        DataType::Date64 => Ok(Box::new(NumericHLLAccumulator::<Date64Type>::new())),
        DataType::Time32(TimeUnit::Second) => {
            Ok(Box::new(NumericHLLAccumulator::<Time32SecondType>::new()))
        }
        DataType::Time32(TimeUnit::Millisecond) => Ok(Box::new(NumericHLLAccumulator::<
            Time32MillisecondType,
        >::new())),
        DataType::Time64(TimeUnit::Microsecond) => Ok(Box::new(NumericHLLAccumulator::<
            Time64MicrosecondType,
        >::new())),
        DataType::Time64(TimeUnit::Nanosecond) => Ok(Box::new(NumericHLLAccumulator::<
            Time64NanosecondType,
        >::new())),
        DataType::Timestamp(TimeUnit::Second, _) => {
            Ok(Box::new(NumericHLLAccumulator::<TimestampSecondType>::new()))
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => Ok(Box::new(
            NumericHLLAccumulator::<TimestampMillisecondType>::new(),
        )),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Ok(Box::new(
            NumericHLLAccumulator::<TimestampMicrosecondType>::new(),
        )),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => Ok(Box::new(
            NumericHLLAccumulator::<TimestampNanosecondType>::new(),
        )),
        DataType::Utf8 => Ok(Box::new(StringHLLAccumulator::<i32>::new())),
        DataType::LargeUtf8 => Ok(Box::new(StringHLLAccumulator::<i64>::new())),
        DataType::Utf8View => Ok(Box::new(StringViewHLLAccumulator::new())),
        DataType::Binary => Ok(Box::new(BinaryHLLAccumulator::<i32>::new())),
        DataType::LargeBinary => Ok(Box::new(BinaryHLLAccumulator::<i64>::new())),
        DataType::Null => {
            Ok(Box::new(NoopAccumulator::new(ScalarValue::UInt64(Some(0)))))
        }
        other => not_impl_err!(
            "Support for 'approx_distinct' for data type {other} is not implemented"
        ),
    }
}

fn is_fixed_domain_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::UInt8 | DataType::Int8 | DataType::UInt16 | DataType::Int16
    )
}

fn is_supported_type(data_type: &DataType) -> bool {
    let value_type = dictionary_value_type(data_type);
    matches!(value_type, DataType::Null)
        || is_fixed_domain_type(value_type)
        || is_hll_groups_type(value_type)
}

fn dictionary_value_type(data_type: &DataType) -> &DataType {
    let mut value_type = data_type;
    while let DataType::Dictionary(_, inner) = value_type {
        value_type = inner;
    }
    value_type
}

fn is_hll_groups_type(data_type: &DataType) -> bool {
    if matches!(data_type, DataType::Dictionary(_, _)) {
        return is_supported_type(data_type);
    }

    matches!(
        data_type,
        DataType::UInt32
            | DataType::UInt64
            | DataType::Int32
            | DataType::Int64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(TimeUnit::Second)
            | DataType::Time32(TimeUnit::Millisecond)
            | DataType::Time64(TimeUnit::Microsecond)
            | DataType::Time64(TimeUnit::Nanosecond)
            | DataType::Timestamp(TimeUnit::Second, _)
            | DataType::Timestamp(TimeUnit::Millisecond, _)
            | DataType::Timestamp(TimeUnit::Microsecond, _)
            | DataType::Timestamp(TimeUnit::Nanosecond, _)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_support() {
        for value_type in [
            DataType::UInt8,
            DataType::Int8,
            DataType::UInt16,
            DataType::Int16,
            DataType::Int64,
            DataType::Null,
            DataType::Utf8,
            DataType::Binary,
        ] {
            let dict_type = DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(value_type.clone()),
            );
            assert!(is_hll_groups_type(&dict_type));
        }

        // Nested dictionaries resolve to the innermost value
        assert!(is_hll_groups_type(&DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(DataType::Utf8)
            ))
        )));

        // Unsupported value types are rejected
        for value_type in [DataType::Float16, DataType::Float32, DataType::Float64] {
            let dict_type = DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(value_type.clone()),
            );
            let nested_dict_type = DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(dict_type.clone()),
            );
            assert!(!is_hll_groups_type(&value_type));
            assert!(!is_supported_type(&dict_type));
            assert!(!is_hll_groups_type(&dict_type));
            assert!(!is_hll_groups_type(&nested_dict_type));
        }
    }
}

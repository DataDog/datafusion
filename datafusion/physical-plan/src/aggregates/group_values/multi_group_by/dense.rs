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

//! Direct-address grouping for a small observed integer × dictionary domain.
//!
//! Admission depends only on input values. Sparse or changing domains fall back to the
//! ordinary column hash table, preserving every group ID and accumulator state.

use std::mem::{self, size_of};
use std::sync::Arc;

use super::{GroupColumn, GroupValuesColumn};
use crate::aggregates::AGGREGATION_HASH_SEED;
use crate::aggregates::group_values::GroupValues;
use arrow::array::{Array, ArrayRef, AsArray, DictionaryArray};
use arrow::datatypes::{
    DataType, Int32Type, Int64Type, Schema, SchemaRef, TimeUnit,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType,
};
use datafusion_common::hash_utils::RandomState;
use datafusion_common::{Result, internal_err, not_impl_err, resources_datafusion_err};
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::EmitTo;
use hashbrown::HashMap;

// Admission limits bound speculative routing/canonicalization, independently of query values.
// The enclosing native aggregate accounts for retained allocations through size() and spills.
const MAX_ROUTING_BYTES: usize = 8 * 1024 * 1024;
const EMPTY: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IntegerDomain {
    base: i64,
    stride: i64,
    width: usize,
}

impl IntegerDomain {
    fn observe(values: &[i64], array: &ArrayRef, old: Option<Self>) -> Option<Self> {
        let mut min = old.map(|domain| domain.base);
        let mut max = match old {
            Some(domain) => Some(domain.last()?),
            None => None,
        };
        for (row, &value) in values.iter().enumerate() {
            if !array.is_null(row) {
                min = Some(min.map_or(value, |min| min.min(value)));
                max = Some(max.map_or(value, |max| max.max(value)));
            }
        }
        let (min, max) = (min?, max?);
        let mut stride = old.map_or(0, |domain| domain.stride);
        if let Some(old) = old {
            stride = gcd(stride, old.base.checked_sub(min)?);
        }
        for (row, &value) in values.iter().enumerate() {
            if !array.is_null(row) {
                stride = gcd(stride, value.checked_sub(min)?);
            }
        }
        // A constant/all-null first batch provides no evidence for a useful lattice.
        if stride == 0 {
            return None;
        }
        let span = max.checked_sub(min)?;
        let width = usize::try_from(span / stride).ok()?.checked_add(1)?;
        let width = width.checked_next_power_of_two()?;
        let distance = i64::try_from(width.checked_sub(1)?)
            .ok()?
            .checked_mul(stride)?;
        // Leave geometric growth room in the direction which expanded the domain.
        let base = if old.is_some_and(|old| min < old.base) {
            max.checked_sub(distance)?
        } else {
            min
        };
        base.checked_add(distance)?;
        Some(Self {
            base,
            stride,
            width,
        })
    }

    fn last(self) -> Option<i64> {
        self.base.checked_add(
            i64::try_from(self.width.checked_sub(1)?)
                .ok()?
                .checked_mul(self.stride)?,
        )
    }

    fn ordinal(self, value: i64) -> Option<usize> {
        let difference = value.checked_sub(self.base)?;
        if difference < 0 || difference % self.stride != 0 {
            return None;
        }
        let ordinal = usize::try_from(difference / self.stride).ok()?;
        // Ordinal zero is reserved for the SQL NULL group.
        (ordinal < self.width).then(|| ordinal + 1)
    }
}

// Sampling only rejects: its smaller span and coarser GCD cannot overestimate
// the full integer width. Accepted inputs still require complete observation.
fn sample_exceeds_limit(
    values: &[i64],
    array: &ArrayRef,
    dictionary_len: usize,
    limit: usize,
) -> bool {
    const SAMPLES: usize = 64;
    if values.len() <= SAMPLES {
        return false;
    }
    let step = (values.len() - 1) / (SAMPLES - 1);
    let remainder = (values.len() - 1) % (SAMPLES - 1);
    let rows = (0..SAMPLES)
        .map(|sample| sample * step + sample * remainder / (SAMPLES - 1))
        .filter(|&row| !array.is_null(row));
    let (mut min, mut max) = (i64::MAX, i64::MIN);
    for row in rows.clone() {
        min = min.min(values[row]);
        max = max.max(values[row]);
    }
    if min >= max {
        return false; // A constant/all-null sample says nothing about unsampled rows.
    }
    let Some(span) = max.checked_sub(min) else {
        return true; // The full span cannot fit the supported i64 domain either.
    };
    let stride = rows.fold(0, |stride, row| gcd(stride, values[row] - min));
    // Include one ordinal for the first integer and one for NULL. Use the same
    // conservative physical-dictionary dimension as full initial admission.
    !usize::try_from(span / stride)
        .ok()
        .and_then(|width| width.checked_add(2))
        .and_then(|width| {
            dictionary_len
                .checked_add(1)
                .and_then(|tags| width.checked_mul(tags))
        })
        .is_some_and(|cells| cells <= limit / size_of::<u32>())
}

fn gcd(mut left: i64, mut right: i64) -> i64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

/// Value IDs, rather than physical dictionary keys, define SQL grouping identity.
struct DictionaryIds {
    values: HashMap<Vec<u8>, u32, RandomState>,
    payload_bytes: usize,
    cached_values: Option<ArrayRef>,
    cached_ids: Vec<u32>,
}

impl Default for DictionaryIds {
    fn default() -> Self {
        Self {
            values: HashMap::with_hasher(AGGREGATION_HASH_SEED),
            payload_bytes: 0,
            cached_values: None,
            cached_ids: Vec::new(),
        }
    }
}

impl DictionaryIds {
    fn prepare(&mut self, dictionary: &DictionaryArray<Int32Type>, limit: usize) -> bool {
        let values = dictionary.values();
        if self.cached_values.as_ref().is_some_and(|old| {
            // Coalescing can rebuild the Array wrapper while retaining its immutable data.
            Arc::ptr_eq(old, values) || old.to_data().ptr_eq(&values.to_data())
        }) {
            return true;
        }
        let value_bytes = values.get_array_memory_size();
        let Some(mapping_bytes) = values.len().checked_mul(size_of::<u32>()) else {
            return false;
        };
        if value_bytes.saturating_add(mapping_bytes) > limit {
            return false;
        }
        self.cached_values = Some(Arc::clone(values));
        self.cached_ids.clear();
        if self.cached_ids.try_reserve(values.len()).is_err() {
            return false;
        }
        for row in 0..values.len() {
            let id = if values.is_null(row) {
                0
            } else {
                let bytes = match values.data_type() {
                    DataType::Binary => values.as_binary::<i32>().value(row),
                    DataType::Utf8 => values.as_string::<i32>().value(row).as_bytes(),
                    _ => unreachable!("schema checked at construction"),
                };
                if let Some(id) = self.values.get(bytes) {
                    *id
                } else {
                    let Ok(id) = u32::try_from(self.values.len() + 1) else {
                        return false;
                    };
                    let owned = bytes.to_vec();
                    self.payload_bytes += owned.capacity();
                    self.values.insert(owned, id);
                    id
                }
            };
            self.cached_ids.push(id);
            if self.size() > limit {
                return false;
            }
        }
        true
    }

    fn len(&self) -> usize {
        self.values.len() + 1 // null keys and null dictionary values both use zero
    }

    fn size(&self) -> usize {
        self.values.allocation_size()
            + self.payload_bytes
            + self.cached_ids.allocated_size()
            + self
                .cached_values
                .as_ref()
                .map_or(0, |values| values.get_array_memory_size())
    }
}

/// An optional native grouping strategy. Only unordered grouping uses this wrapper.
pub(crate) struct GroupValuesDense {
    schema: SchemaRef,
    integer_index: usize,
    dictionary_index: usize,
    columns: Vec<Box<dyn GroupColumn>>,
    domain: Option<IntegerDomain>,
    dictionary: DictionaryIds,
    table: Vec<u32>,
    tag_width: usize,
    ordinals: Vec<usize>,
    new_groups: Vec<usize>,
    fallback: Option<GroupValuesColumn<false>>,
    limit: usize,
}

impl GroupValuesDense {
    pub(crate) fn supports_schema(schema: &Schema) -> bool {
        if schema.fields().len() != 2 {
            return false;
        }
        let integer = |data_type: &DataType| {
            matches!(data_type, DataType::Int64 | DataType::Timestamp(_, _))
        };
        let dictionary = |data_type: &DataType| {
            matches!(
                data_type,
                DataType::Dictionary(key, value)
                    if key.as_ref() == &DataType::Int32
                        && matches!(value.as_ref(), DataType::Binary | DataType::Utf8)
            )
        };
        (integer(schema.field(0).data_type()) && dictionary(schema.field(1).data_type()))
            || (integer(schema.field(1).data_type())
                && dictionary(schema.field(0).data_type()))
    }

    pub(crate) fn try_new(schema: SchemaRef) -> Result<Self> {
        if !Self::supports_schema(&schema) {
            return not_impl_err!("Unsupported direct-address group schema: {schema}");
        }
        let integer_index = usize::from(matches!(
            schema.field(0).data_type(),
            DataType::Dictionary(_, _)
        ));
        Ok(Self {
            columns: GroupValuesColumn::<false>::build_group_columns(&schema)?,
            schema,
            integer_index,
            dictionary_index: 1 - integer_index,
            domain: None,
            dictionary: DictionaryIds::default(),
            table: Vec::new(),
            tag_width: 0,
            ordinals: Vec::new(),
            new_groups: Vec::new(),
            fallback: None,
            limit: MAX_ROUTING_BYTES,
        })
    }

    fn migrate(&mut self) -> Result<()> {
        if self.fallback.is_some() {
            return Ok(());
        }
        let columns: Vec<_> = mem::take(&mut self.columns)
            .into_iter()
            .map(|column| column.build())
            .collect();
        self.table = Vec::new();
        self.dictionary = DictionaryIds::default();
        self.domain = None;
        self.ordinals = Vec::new();
        self.new_groups = Vec::new();
        self.fallback = Some(GroupValuesColumn::<false>::try_new_from_distinct(
            Arc::clone(&self.schema),
            &columns,
        )?);
        Ok(())
    }

    fn fallback_intern(
        &mut self,
        cols: &[ArrayRef],
        groups: &mut Vec<usize>,
    ) -> Result<()> {
        self.migrate()?;
        self.fallback
            .as_mut()
            .expect("migration installs fallback")
            .intern(cols, groups)
    }

    fn resize_table(&mut self, domain: IntegerDomain) -> bool {
        let Some(tag_width) = self.dictionary.len().checked_next_power_of_two() else {
            return false;
        };
        if self.domain == Some(domain) && self.tag_width == tag_width {
            return true;
        }
        let Some(cells) = domain
            .width
            .checked_add(1)
            .and_then(|width| width.checked_mul(tag_width))
        else {
            return false;
        };
        if cells > self.limit / size_of::<u32>() || cells >= EMPTY as usize {
            return false;
        }
        let mut table = vec![EMPTY; cells];
        if let Some(old) = self.domain {
            for (slot, &group) in self
                .table
                .iter()
                .enumerate()
                .filter(|(_, group)| **group != EMPTY)
            {
                let old_ordinal = slot / self.tag_width;
                let tag = slot % self.tag_width;
                let ordinal = if old_ordinal == 0 {
                    0
                } else {
                    let Some(value) = i64::try_from(old_ordinal - 1)
                        .ok()
                        .and_then(|ordinal| ordinal.checked_mul(old.stride))
                        .and_then(|distance| old.base.checked_add(distance))
                    else {
                        return false;
                    };
                    let Some(ordinal) = domain.ordinal(value) else {
                        return false;
                    };
                    ordinal
                };
                table[ordinal * tag_width + tag] = group;
            }
        }
        self.table = table;
        self.domain = Some(domain);
        self.tag_width = tag_width;
        true
    }

    fn map_ordinals(
        &mut self,
        domain: IntegerDomain,
        array: &ArrayRef,
        values: &[i64],
    ) -> bool {
        self.ordinals.clear();
        if self.ordinals.try_reserve(values.len()).is_err() {
            return false;
        }
        for (row, &value) in values.iter().enumerate() {
            let ordinal = if array.is_null(row) {
                Some(0)
            } else {
                domain.ordinal(value)
            };
            let Some(ordinal) = ordinal else {
                return false;
            };
            self.ordinals.push(ordinal);
        }
        true
    }
}

impl GroupValues for GroupValuesDense {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        if let Some(fallback) = &mut self.fallback {
            return fallback.intern(cols, groups);
        }
        if cols.len() != 2 || cols[0].len() != cols[1].len() {
            return internal_err!(
                "Direct-address grouping requires two equally sized columns"
            );
        }
        groups.clear();
        if cols[0].is_empty() {
            return Ok(());
        }
        let integers = &cols[self.integer_index];
        let values = integer_values(integers);
        let dictionary = cols[self.dictionary_index].as_dictionary::<Int32Type>();
        if self.domain.is_none()
            && sample_exceeds_limit(
                values,
                integers,
                dictionary.values().len(),
                self.limit,
            )
        {
            return self.fallback_intern(cols, groups);
        }
        let mapped = self
            .domain
            .is_some_and(|domain| self.map_ordinals(domain, integers, values));
        let domain = if mapped {
            self.domain
        } else {
            IntegerDomain::observe(values, integers, self.domain)
        };
        let Some(domain) = domain else {
            return self.fallback_intern(cols, groups);
        };
        // Avoid copying/canonicalizing a large initial dictionary for a product which
        // cannot fit even its physical dictionary domain. This conservative admission
        // may decline dictionaries with many duplicates; generic grouping remains exact.
        if self.domain.is_none()
            && !domain
                .width
                .checked_add(1)
                .and_then(|width| {
                    width.checked_mul(dictionary.values().len().saturating_add(1))
                })
                .is_some_and(|cells| cells <= self.limit / size_of::<u32>())
        {
            return self.fallback_intern(cols, groups);
        }
        if !self.dictionary.prepare(dictionary, self.limit)
            || !self.resize_table(domain)
            || (!mapped && !self.map_ordinals(domain, integers, values))
        {
            return self.fallback_intern(cols, groups);
        }
        self.new_groups.clear();
        let old_len = self.len();
        groups.try_reserve(values.len()).map_err(|e| {
            resources_datafusion_err!("failed to reserve {} group IDs: {e}", values.len())
        })?;
        for (row, &ordinal) in self.ordinals.iter().enumerate() {
            let tag = dictionary
                .key(row)
                .map_or(0, |key| self.dictionary.cached_ids[key])
                as usize;
            let entry = &mut self.table[ordinal * self.tag_width + tag];
            if *entry == EMPTY {
                *entry = (old_len + self.new_groups.len()) as u32;
                self.new_groups.push(row);
            }
            groups.push(*entry as usize);
        }
        if !self.new_groups.is_empty() {
            for (column, input) in self.columns.iter_mut().zip(cols) {
                column.vectorized_append(input, &self.new_groups)?;
            }
        }
        Ok(())
    }

    fn size(&self) -> usize {
        self.fallback.as_ref().map_or_else(
            || {
                self.columns
                    .iter()
                    .map(|column| column.size())
                    .sum::<usize>()
                    + self.table.allocated_size()
                    + self.dictionary.size()
                    + self.ordinals.allocated_size()
                    + self.new_groups.allocated_size()
            },
            |fallback| {
                fallback.size()
                    + fallback.group_index_lists.allocated_size()
                    + fallback
                        .group_index_lists
                        .iter()
                        .map(VecAllocExt::allocated_size)
                        .sum::<usize>()
            },
        )
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.fallback
            .as_ref()
            .map_or_else(|| self.columns[0].len(), GroupValues::len)
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        if matches!(emit_to, EmitTo::First(_)) {
            self.migrate()?;
        }
        if let Some(fallback) = &mut self.fallback {
            let output = fallback.emit(emit_to)?;
            if matches!(emit_to, EmitTo::All) {
                fallback.clear_shrink(0);
            }
            return Ok(output);
        }
        let fresh = GroupValuesColumn::<false>::build_group_columns(&self.schema)?;
        let output = mem::replace(&mut self.columns, fresh)
            .into_iter()
            .map(|column| column.build())
            .collect();
        self.table = Vec::new();
        self.dictionary = DictionaryIds::default();
        self.domain = None;
        self.tag_width = 0;
        Ok(output)
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        if let Some(fallback) = &mut self.fallback {
            fallback.clear_shrink(num_rows);
        } else {
            self.columns = GroupValuesColumn::<false>::build_group_columns(&self.schema)
                .expect("schema validated by direct-address grouping");
            self.table = Vec::new();
            self.dictionary = DictionaryIds::default();
            self.domain = None;
            self.tag_width = 0;
        }
        self.ordinals.clear();
        self.ordinals.shrink_to(num_rows);
        self.new_groups.clear();
        self.new_groups.shrink_to(num_rows);
    }
}

fn integer_values(array: &ArrayRef) -> &[i64] {
    match array.data_type() {
        DataType::Int64 => array.as_primitive::<Int64Type>().values(),
        DataType::Timestamp(TimeUnit::Second, _) => {
            array.as_primitive::<TimestampSecondType>().values()
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            array.as_primitive::<TimestampMillisecondType>().values()
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            array.as_primitive::<TimestampMicrosecondType>().values()
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            array.as_primitive::<TimestampNanosecondType>().values()
        }
        _ => unreachable!("schema checked at construction"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coalesce::LimitedBatchCoalescer;
    use arrow::array::{Int32Array, Int64Array, StringArray};
    use arrow::compute::cast;
    use arrow::datatypes::Field;
    use arrow::record_batch::RecordBatch;
    use datafusion_common::ScalarValue;
    use datafusion_expr::GroupsAccumulator;
    use datafusion_functions_aggregate_common::aggregate::groups_accumulator::prim_op::PrimitiveGroupsAccumulator;

    fn schema(integer_type: DataType, binary: bool, reverse: bool) -> SchemaRef {
        let mut fields = vec![
            Field::new("integer", integer_type, true),
            Field::new(
                "dictionary",
                DataType::Dictionary(
                    Box::new(DataType::Int32),
                    Box::new(if binary {
                        DataType::Binary
                    } else {
                        DataType::Utf8
                    }),
                ),
                true,
            ),
        ];
        if reverse {
            fields.reverse();
        }
        Arc::new(Schema::new(fields))
    }

    fn batch(
        schema: &SchemaRef,
        integers: &[Option<i64>],
        keys: &[Option<i32>],
        values: &[Option<&str>],
    ) -> Result<Vec<ArrayRef>> {
        let integer_index = usize::from(schema.field(0).name() == "dictionary");
        let integer = cast(
            &Int64Array::from(integers.to_vec()),
            schema.field(integer_index).data_type(),
        )?;
        let DataType::Dictionary(_, value_type) =
            schema.field(1 - integer_index).data_type()
        else {
            unreachable!()
        };
        let values = cast(&StringArray::from(values.to_vec()), value_type)?;
        let dictionary: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(keys.to_vec()),
            values,
        )?);
        Ok(if integer_index == 0 {
            vec![integer, dictionary]
        } else {
            vec![dictionary, integer]
        })
    }

    fn logical_rows(columns: &[ArrayRef]) -> Result<Vec<Vec<ScalarValue>>> {
        (0..columns[0].len())
            .map(|row| {
                columns
                    .iter()
                    .map(|column| {
                        let value = ScalarValue::try_from_array(column, row)?;
                        Ok(match value {
                            ScalarValue::Dictionary(_, value) => *value,
                            value => value,
                        })
                    })
                    .collect()
            })
            .collect()
    }

    fn check_batch(
        dense: &mut GroupValuesDense,
        reference: &mut GroupValuesColumn<true>,
        columns: &[ArrayRef],
    ) -> Result<()> {
        let mut actual = Vec::new();
        let mut expected = Vec::new();
        dense.intern(columns, &mut actual)?;
        reference.intern(columns, &mut expected)?;
        assert_eq!(actual, expected);
        assert_eq!(dense.len(), reference.len());
        Ok(())
    }

    fn check_emit(
        dense: &mut GroupValuesDense,
        reference: &mut GroupValuesColumn<true>,
        emit: EmitTo,
    ) -> Result<()> {
        let actual = dense.emit(emit)?;
        let expected = reference.emit(emit)?;
        assert_eq!(logical_rows(&actual)?, logical_rows(&expected)?);
        assert_eq!(
            actual
                .iter()
                .map(|array| array.data_type())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|array| array.data_type())
                .collect::<Vec<_>>()
        );
        if matches!(emit, EmitTo::All) {
            reference.clear_shrink(0);
        }
        Ok(())
    }

    #[test]
    fn domain_growth_dictionary_identity_and_nulls() -> Result<()> {
        for reverse in [false, true] {
            for binary in [false, true] {
                for integer_type in [
                    DataType::Int64,
                    DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                ] {
                    let schema = schema(integer_type, binary, reverse);
                    let mut dense = GroupValuesDense::try_new(Arc::clone(&schema))?;
                    let mut reference =
                        GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
                    let first = batch(
                        &schema,
                        &[
                            Some(0),
                            Some(30),
                            Some(60),
                            Some(90),
                            None,
                            Some(0),
                            None,
                            Some(30),
                        ],
                        &[
                            Some(0),
                            Some(2),
                            Some(1),
                            Some(1),
                            Some(0),
                            Some(3),
                            None,
                            Some(1),
                        ],
                        &[Some("a"), Some("b"), Some("a"), None],
                    )?;
                    check_batch(&mut dense, &mut reference, &first)?;
                    check_batch(&mut dense, &mut reference, &first)?; // shared values Arc
                    let second = batch(
                        &schema,
                        &[Some(-60), Some(15), Some(300), Some(0), Some(0), None],
                        &[Some(3), Some(1), Some(0), None, Some(2), Some(4)],
                        &[Some("b"), Some("a"), None, Some("c"), Some("a")],
                    )?;
                    check_batch(&mut dense, &mut reference, &second)?;
                    assert!(dense.fallback.is_none());
                    assert_eq!(dense.domain.unwrap().stride, 15);
                    assert!(
                        dense.size()
                            >= dense.table.allocated_size() + dense.dictionary.size()
                    );
                    check_emit(&mut dense, &mut reference, EmitTo::All)?;
                    assert!(dense.is_empty());
                    check_batch(&mut dense, &mut reference, &second)?;
                    dense.clear_shrink(0);
                    reference.clear_shrink(0);
                    check_batch(&mut dense, &mut reference, &first)?;
                    check_emit(&mut dense, &mut reference, EmitTo::All)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn coalesced_dictionary_wrappers_reuse_mapping() -> Result<()> {
        for binary in [false, true] {
            let schema = schema(DataType::Int64, binary, false);
            let mut dense = GroupValuesDense::try_new(Arc::clone(&schema))?;
            let mut reference = GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
            let input = batch(
                &schema,
                &[Some(0), Some(30), Some(0), Some(30)],
                &[Some(0), Some(1), Some(2), Some(3)],
                &[Some("a"), Some("b"), Some("a"), None],
            )?;
            check_batch(&mut dense, &mut reference, &input)?;
            let original_values = input[1].as_dictionary::<Int32Type>().values();
            let record_batch = RecordBatch::try_new(Arc::clone(&schema), input.clone())?;
            let mut coalescer = LimitedBatchCoalescer::new(Arc::clone(&schema), 16, None);
            coalescer.push_batch(record_batch.clone())?;
            coalescer.push_batch(record_batch)?;
            coalescer.finish()?;
            let merged = coalescer.next_completed_batch().unwrap();
            assert_eq!(merged.num_rows(), 8);
            let merged_values = merged.column(1).as_dictionary::<Int32Type>().values();
            assert!(!Arc::ptr_eq(original_values, merged_values));
            assert!(original_values.to_data().ptr_eq(&merged_values.to_data()));
            check_batch(&mut dense, &mut reference, merged.columns())?;
            assert!(dense.fallback.is_none());
            // A cache miss replaces this Arc; retained identity proves the mapping was reused.
            assert!(Arc::ptr_eq(
                dense.dictionary.cached_values.as_ref().unwrap(),
                original_values,
            ));
            // The same value buffers with a different slice must still remap dictionary keys.
            let sliced: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![Some(0), Some(1), Some(2), None]),
                original_values.slice(1, 3),
            )?);
            check_batch(&mut dense, &mut reference, &[Arc::clone(&input[0]), sliced])?;
            check_emit(&mut dense, &mut reference, EmitTo::All)?;
        }
        Ok(())
    }

    // Also run with datafusion-common/force_hash_collisions: seed >=3 colliding
    // groups, migrate, then revisit every old group and add another group.
    #[test]
    fn sparse_fallback_preserves_existing_accumulator_indices() -> Result<()> {
        let schema = schema(DataType::Int64, false, false);
        let mut dense = GroupValuesDense::try_new(Arc::clone(&schema))?;
        let mut reference = GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
        let first = batch(
            &schema,
            &[Some(0), Some(30), Some(60)],
            &[Some(0), Some(1), Some(0)],
            &[Some("a"), Some("b")],
        )?;
        let second = batch(
            &schema,
            &[Some(60), Some(1_000_003), Some(0), Some(30)],
            &[Some(1), Some(0), Some(1), Some(0)],
            &[Some("b"), Some("a")],
        )?;
        let third = batch(
            &schema,
            &[Some(0), Some(30), Some(60), Some(90)],
            &[Some(0), Some(1), Some(0), Some(1)],
            &[Some("a"), Some("b")],
        )?;
        let mut actual_sums =
            PrimitiveGroupsAccumulator::<Int64Type, _>::new(&DataType::Int64, |a, b| {
                *a += b
            });
        let mut expected_sums =
            PrimitiveGroupsAccumulator::<Int64Type, _>::new(&DataType::Int64, |a, b| {
                *a += b
            });
        for (index, input) in [&first, &second, &third].into_iter().enumerate() {
            let mut actual = Vec::new();
            let mut expected = Vec::new();
            dense.intern(input, &mut actual)?;
            reference.intern(input, &mut expected)?;
            assert_eq!(actual, expected);
            let weights: ArrayRef = Arc::new(Int64Array::from_iter_values(
                (0..input[0].len()).map(|row| (index * 100 + row + 1) as i64),
            ));
            actual_sums.update_batch(
                std::slice::from_ref(&weights),
                &actual,
                None,
                dense.len(),
            )?;
            expected_sums.update_batch(&[weights], &expected, None, reference.len())?;
            assert_eq!(dense.fallback.is_some(), index != 0);
        }
        let actual_sums = actual_sums.evaluate(EmitTo::All)?;
        let expected_sums = expected_sums.evaluate(EmitTo::All)?;
        assert_eq!(actual_sums.as_ref(), expected_sums.as_ref());
        check_emit(&mut dense, &mut reference, EmitTo::All)?;
        check_batch(&mut dense, &mut reference, &second)?;
        check_emit(&mut dense, &mut reference, EmitTo::All)?;
        Ok(())
    }

    #[test]
    fn prefix_emission_and_reappearing_keys() -> Result<()> {
        for emit_count in [0, 1, 3] {
            let schema = schema(DataType::Int64, false, false);
            let mut dense = GroupValuesDense::try_new(Arc::clone(&schema))?;
            let mut reference = GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
            let input = batch(
                &schema,
                &[Some(0), Some(30), Some(60)],
                &[Some(0), Some(1), Some(0)],
                &[Some("a"), Some("b")],
            )?;
            check_batch(&mut dense, &mut reference, &input)?;
            check_emit(&mut dense, &mut reference, EmitTo::First(emit_count))?;
            assert!(dense.fallback.is_some());
            check_batch(&mut dense, &mut reference, &input)?;
            check_emit(&mut dense, &mut reference, EmitTo::All)?;
        }
        Ok(())
    }

    #[test]
    fn unproven_or_overflowing_domains_fall_back() -> Result<()> {
        for integers in [
            vec![None, None],
            vec![Some(7), Some(7)],
            vec![Some(i64::MIN), Some(i64::MAX)],
        ] {
            let schema = schema(DataType::Int64, false, false);
            let mut dense = GroupValuesDense::try_new(Arc::clone(&schema))?;
            let mut reference = GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
            let input = batch(&schema, &integers, &[None, None], &[])?;
            check_batch(&mut dense, &mut reference, &input)?;
            assert!(dense.fallback.is_some());
            check_emit(&mut dense, &mut reference, EmitTo::All)?;
        }
        Ok(())
    }

    #[test]
    fn dictionary_admission_is_bounded_before_remapping_unused_values() -> Result<()> {
        let schema = schema(DataType::Int64, false, false);
        let mut dense = GroupValuesDense::try_new(Arc::clone(&schema))?;
        dense.limit = 32;
        let mut reference = GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
        let input = batch(
            &schema,
            &[Some(0), Some(30)],
            &[None, None],
            &[Some(
                "unreferenced payload which exceeds the admission limit",
            )],
        )?;
        check_batch(&mut dense, &mut reference, &input)?;
        assert!(dense.fallback.is_some());
        assert!(dense.dictionary.cached_ids.is_empty());
        check_emit(&mut dense, &mut reference, EmitTo::All)?;
        Ok(())
    }
    #[test]
    fn sampled_rejection_preserves_nullable_sliced_groups() -> Result<()> {
        for samples in [[0, 1, 1_000_000_000], [i64::MIN, 0, i64::MAX]] {
            let schema = schema(DataType::Int64, false, false);
            let mut dense = GroupValuesDense::try_new(Arc::clone(&schema))?;
            let mut reference = GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
            let mut integers = vec![None; 131];
            integers[0] = Some(99);
            integers[130] = Some(99);
            for (row, value) in [1, 3, 129].into_iter().zip(samples) {
                integers[row] = Some(value);
            }
            let input = batch(&schema, &integers, &[Some(0); 131], &[Some("a")])?;
            let input: Vec<_> = input.iter().map(|array| array.slice(1, 129)).collect();
            assert!(sample_exceeds_limit(
                integer_values(&input[0]),
                &input[0],
                1,
                dense.limit,
            ));
            check_batch(&mut dense, &mut reference, &input)?;
            assert!(dense.fallback.is_some());
            assert_eq!(dense.len(), 4); // Three distinct integers and the NULL group.
            check_emit(&mut dense, &mut reference, EmitTo::All)?;
        }
        Ok(())
    }

    #[test]
    fn compact_sample_still_checks_unsampled_stride_changes() -> Result<()> {
        for (step, remains_dense) in [(1_000i64, true), (1_000_000, false)] {
            let schema = schema(DataType::Int64, false, false);
            let mut dense = GroupValuesDense::try_new(Arc::clone(&schema))?;
            let mut reference = GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
            let mut integers: Vec<_> = (0..129).map(|index| Some(index * step)).collect();
            // The 64 evenly spaced samples skip row 1. It changes the actual GCD
            // to one, either widening a valid domain or forcing generic fallback.
            integers[1] = Some(step + 1);
            let input = batch(&schema, &integers, &[Some(0); 129], &[Some("a")])?;
            assert!(!sample_exceeds_limit(
                integer_values(&input[0]),
                &input[0],
                1,
                dense.limit,
            ));
            check_batch(&mut dense, &mut reference, &input)?;
            assert_eq!(dense.fallback.is_none(), remains_dense);
            if remains_dense {
                assert_eq!(dense.domain.unwrap().stride, 1);
            }
            check_emit(&mut dense, &mut reference, EmitTo::All)?;
        }
        Ok(())
    }

    #[test]
    fn high_cardinality_product_declines_dense_routing() -> Result<()> {
        let schema = schema(DataType::Int64, false, false);
        let mut dense = GroupValuesDense::try_new(Arc::clone(&schema))?;
        let mut reference = GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
        let integers: Vec<_> = (0..4096).map(|value| Some(i64::from(value))).collect();
        let keys: Vec<_> = (0..4096).map(Some).collect();
        let owned: Vec<_> = (0..4096).map(|value| value.to_string()).collect();
        let values: Vec<_> = owned.iter().map(|value| Some(value.as_str())).collect();
        let input = batch(&schema, &integers, &keys, &values)?;
        check_batch(&mut dense, &mut reference, &input)?;
        assert!(dense.fallback.is_some());
        assert!(dense.table.is_empty());
        check_emit(&mut dense, &mut reference, EmitTo::All)?;
        Ok(())
    }

    #[test]
    fn varied_small_batches_and_slices_match_native_grouping() -> Result<()> {
        let schema = schema(DataType::Int64, true, true);
        let mut dense = GroupValuesDense::try_new(Arc::clone(&schema))?;
        let mut reference = GroupValuesColumn::<true>::try_new(Arc::clone(&schema))?;
        let mut seed = 13u64;
        for batch_index in 0..12 {
            let mut integers = Vec::new();
            let mut keys = Vec::new();
            for row in 0..19 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                integers
                    .push((row % 7 != 0).then_some(((seed >> 32) % 41) as i64 * 6 - 120));
                keys.push((row % 5 != 0).then_some((seed % 4) as i32));
            }
            let values = if batch_index % 2 == 0 {
                [Some("a"), Some("b"), None, Some("a")]
            } else {
                [Some("b"), Some("a"), Some("c"), None]
            };
            let input = batch(&schema, &integers, &keys, &values)?;
            let input: Vec<_> = input.iter().map(|array| array.slice(1, 17)).collect();
            check_batch(&mut dense, &mut reference, &input)?;
            if batch_index == 5 {
                check_emit(&mut dense, &mut reference, EmitTo::First(3))?;
            }
        }
        check_emit(&mut dense, &mut reference, EmitTo::All)?;
        Ok(())
    }
}

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

use crate::aggregates::group_values::multi_group_by::{
    GroupColumn, Nulls, nulls_equal_to,
};
use ahash::RandomState;
use datafusion_common::hash_utils::{HashValue, combine_hashes};
use crate::aggregates::group_values::null_builder::MaybeNullBufferBuilder;
use arrow::array::{Array, ArrayRef, FixedSizeBinaryArray};
use arrow::array::cast::AsArray;
use arrow::buffer::Buffer;
use datafusion_common::Result;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use itertools::izip;
use std::iter;
use std::sync::Arc;

/// An implementation of [`GroupColumn`] for [`FixedSizeBinaryArray`]
///
/// Stores values in a flat `Vec<u8>` buffer with a fixed stride (`value_size`),
/// enabling fast fixed-size `memcmp`-style equality checks.
///
/// # Template parameters
///
/// `NULLABLE`: if the data can contain any nulls
#[derive(Debug)]
pub struct FixedSizeBinaryGroupValueBuilder<const NULLABLE: bool> {
    /// Flat buffer storing all values, each `value_size` bytes.
    /// Row `i` occupies `values[i*value_size .. (i+1)*value_size]`.
    values: Vec<u8>,
    /// Size of each FixedSizeBinary value in bytes (e.g. 16)
    value_size: usize,
    /// Number of stored rows
    len: usize,
    /// Null tracking (only meaningful if NULLABLE)
    nulls: MaybeNullBufferBuilder,
}

impl<const NULLABLE: bool> FixedSizeBinaryGroupValueBuilder<NULLABLE> {
    pub fn new(value_size: usize) -> Self {
        Self {
            values: Vec::new(),
            value_size,
            len: 0,
            nulls: MaybeNullBufferBuilder::new(),
        }
    }

    /// Return the stored value bytes for row `row`
    #[inline]
    fn value(&self, row: usize) -> &[u8] {
        let start = row * self.value_size;
        &self.values[start..start + self.value_size]
    }

    fn vectorized_equal_to_non_nullable(
        &self,
        lhs_rows: &[usize],
        array: &ArrayRef,
        rhs_rows: &[usize],
        equal_to_results: &mut [bool],
    ) {
        let array = array.as_fixed_size_binary();
        let rhs_values = array.value_data();
        let size = self.value_size;

        let iter = izip!(
            lhs_rows.iter(),
            rhs_rows.iter(),
            equal_to_results.iter_mut(),
        );

        for (&lhs_row, &rhs_row, equal_to_result) in iter {
            let lhs_start = lhs_row * size;
            let rhs_start = rhs_row * size;
            let result = self.values[lhs_start..lhs_start + size]
                == rhs_values[rhs_start..rhs_start + size];
            *equal_to_result = result && *equal_to_result;
        }
    }

    fn vectorized_equal_to_nullable(
        &self,
        lhs_rows: &[usize],
        array: &ArrayRef,
        rhs_rows: &[usize],
        equal_to_results: &mut [bool],
    ) {
        let array = array.as_fixed_size_binary();

        let iter = izip!(
            lhs_rows.iter(),
            rhs_rows.iter(),
            equal_to_results.iter_mut(),
        );

        for (&lhs_row, &rhs_row, equal_to_result) in iter {
            if !*equal_to_result {
                continue;
            }

            let exist_null = self.nulls.is_null(lhs_row);
            let input_null = array.is_null(rhs_row);
            if let Some(result) = nulls_equal_to(exist_null, input_null) {
                *equal_to_result = result;
                continue;
            }

            *equal_to_result = self.value(lhs_row) == array.value(rhs_row);
        }
    }
}

impl<const NULLABLE: bool> GroupColumn for FixedSizeBinaryGroupValueBuilder<NULLABLE> {
    fn equal_to(&self, lhs_row: usize, array: &ArrayRef, rhs_row: usize) -> bool {
        let array = array.as_fixed_size_binary();

        if NULLABLE {
            let exist_null = self.nulls.is_null(lhs_row);
            let input_null = array.is_null(rhs_row);
            if let Some(result) = nulls_equal_to(exist_null, input_null) {
                return result;
            }
        }

        self.value(lhs_row) == array.value(rhs_row)
    }

    fn append_val(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
        let array = array.as_fixed_size_binary();

        if NULLABLE {
            if array.is_null(row) {
                self.nulls.append(true);
                self.values
                    .extend(iter::repeat_n(0u8, self.value_size));
            } else {
                self.nulls.append(false);
                self.values.extend_from_slice(array.value(row));
            }
        } else {
            self.values.extend_from_slice(array.value(row));
        }

        self.len += 1;
        Ok(())
    }

    fn vectorized_equal_to(
        &self,
        lhs_rows: &[usize],
        array: &ArrayRef,
        rhs_rows: &[usize],
        equal_to_results: &mut [bool],
    ) {
        if !NULLABLE || (array.null_count() == 0 && !self.nulls.might_have_nulls()) {
            self.vectorized_equal_to_non_nullable(
                lhs_rows,
                array,
                rhs_rows,
                equal_to_results,
            );
        } else {
            self.vectorized_equal_to_nullable(
                lhs_rows,
                array,
                rhs_rows,
                equal_to_results,
            );
        }
    }

    fn vectorized_append(&mut self, array: &ArrayRef, rows: &[usize]) -> Result<()> {
        let arr = array.as_fixed_size_binary();
        let size = self.value_size;

        let null_count = array.null_count();
        let num_rows = array.len();
        let all_null_or_non_null = if null_count == 0 {
            Nulls::None
        } else if null_count == num_rows {
            Nulls::All
        } else {
            Nulls::Some
        };

        match (NULLABLE, all_null_or_non_null) {
            (true, Nulls::Some) => {
                for &row in rows {
                    if array.is_null(row) {
                        self.nulls.append(true);
                        self.values
                            .extend(iter::repeat_n(0u8, size));
                    } else {
                        self.nulls.append(false);
                        self.values.extend_from_slice(arr.value(row));
                    }
                }
            }

            (true, Nulls::None) => {
                self.nulls.append_n(rows.len(), false);
                let value_data = arr.value_data();
                for &row in rows {
                    let start = row * size;
                    self.values
                        .extend_from_slice(&value_data[start..start + size]);
                }
            }

            (true, Nulls::All) => {
                self.nulls.append_n(rows.len(), true);
                self.values
                    .extend(iter::repeat_n(0u8, rows.len() * size));
            }

            (false, _) => {
                let value_data = arr.value_data();
                for &row in rows {
                    let start = row * size;
                    self.values
                        .extend_from_slice(&value_data[start..start + size]);
                }
            }
        }

        self.len += rows.len();
        Ok(())
    }

    fn len(&self) -> usize {
        self.len
    }

    fn size(&self) -> usize {
        self.values.allocated_size() + self.nulls.allocated_size()
    }

    fn build(self: Box<Self>) -> ArrayRef {
        let Self {
            values,
            value_size,
            len: _,
            nulls,
        } = *self;

        let nulls = nulls.build();
        if !NULLABLE {
            assert!(nulls.is_none(), "unexpected nulls in non nullable input");
        }

        let array = FixedSizeBinaryArray::new(
            value_size as i32,
            Buffer::from_vec(values),
            nulls,
        );
        Arc::new(array)
    }

    fn take_n(&mut self, n: usize) -> ArrayRef {
        let size = self.value_size;
        let byte_count = n * size;

        let first_n_values: Vec<u8> = self.values.drain(0..byte_count).collect();

        let first_n_nulls = if NULLABLE { self.nulls.take_n(n) } else { None };

        self.len -= n;

        Arc::new(FixedSizeBinaryArray::new(
            size as i32,
            Buffer::from_vec(first_n_values),
            first_n_nulls,
        ))
    }

    fn input_rows_equal(&self, array: &ArrayRef, row_a: usize, row_b: usize) -> bool {
        if NULLABLE {
            let a_null = array.is_null(row_a);
            let b_null = array.is_null(row_b);
            if a_null || b_null {
                return a_null && b_null;
            }
        }
        let arr = array.as_fixed_size_binary();
        arr.value(row_a) == arr.value(row_b)
    }

    fn hash_input_row(
        &self,
        array: &ArrayRef,
        row: usize,
        random_state: &RandomState,
        rehash: bool,
        current_hash: u64,
    ) -> u64 {
        if NULLABLE && array.is_null(row) {
            return current_hash;
        }
        let value = array.as_fixed_size_binary().value(row);
        let h = value.hash_one(random_state);
        if rehash {
            combine_hashes(h, current_hash)
        } else {
            h
        }
    }

    fn compute_boundaries(&self, array: &ArrayRef, boundaries: &mut [bool]) {
        let arr = array.as_fixed_size_binary();
        let value_data = arr.value_data();
        let size = self.value_size;
        if NULLABLE && array.null_count() > 0 {
            for row in 1..array.len() {
                if boundaries[row] {
                    continue;
                }
                let prev_null = array.is_null(row - 1);
                let curr_null = array.is_null(row);
                if prev_null != curr_null {
                    boundaries[row] = true;
                } else if !prev_null {
                    let prev_start = (row - 1) * size;
                    let curr_start = row * size;
                    if value_data[prev_start..prev_start + size] != value_data[curr_start..curr_start + size] {
                        boundaries[row] = true;
                    }
                }
            }
        } else {
            for row in 1..array.len() {
                if !boundaries[row] {
                    let prev_start = (row - 1) * size;
                    let curr_start = row * size;
                    if value_data[prev_start..prev_start + size] != value_data[curr_start..curr_start + size] {
                        boundaries[row] = true;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow::array::{ArrayRef, FixedSizeBinaryArray};

    /// Helper to build a FixedSizeBinaryArray from an iterator of Option<&[u8]>
    fn make_fsb(items: Vec<Option<&[u8]>>) -> ArrayRef {
        Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter(items.into_iter()).unwrap(),
        ) as ArrayRef
    }

    #[test]
    fn test_nullable_fixed_size_binary_equal_to() {
        let mut builder = FixedSizeBinaryGroupValueBuilder::<true>::new(4);

        let array = make_fsb(vec![
            Some(b"abcd" as &[u8]),
            Some(b"efgh"),
            None,
            Some(b"ijkl"),
        ]);

        // Append rows: null (row 2), b"abcd" (row 0), null (row 2), b"efgh" (row 1)
        builder.append_val(&array, 2).unwrap(); // null
        builder.append_val(&array, 0).unwrap(); // b"abcd"
        builder.append_val(&array, 2).unwrap(); // null
        builder.append_val(&array, 1).unwrap(); // b"efgh"

        let test_array = make_fsb(vec![
            None,
            Some(b"abcd" as &[u8]),
            Some(b"xxxx"),
            Some(b"efgh"),
        ]);

        // null == null -> true
        assert!(builder.equal_to(0, &test_array, 0));
        // b"abcd" == b"abcd" -> true
        assert!(builder.equal_to(1, &test_array, 1));
        // null != b"xxxx" -> false
        assert!(!builder.equal_to(2, &test_array, 2));
        // b"efgh" == b"efgh" -> true
        assert!(builder.equal_to(3, &test_array, 3));
        // b"abcd" != b"xxxx" -> false
        assert!(!builder.equal_to(1, &test_array, 2));
    }

    #[test]
    fn test_non_nullable_fixed_size_binary_equal_to() {
        let mut builder = FixedSizeBinaryGroupValueBuilder::<false>::new(4);

        let array = make_fsb(vec![
            Some(b"abcd" as &[u8]),
            Some(b"efgh"),
        ]);

        builder.append_val(&array, 0).unwrap();
        builder.append_val(&array, 1).unwrap();

        let test_array = make_fsb(vec![
            Some(b"abcd" as &[u8]),
            Some(b"xxxx"),
        ]);

        assert!(builder.equal_to(0, &test_array, 0));
        assert!(!builder.equal_to(1, &test_array, 1));
    }

    #[test]
    fn test_vectorized_operations() {
        let mut builder = FixedSizeBinaryGroupValueBuilder::<false>::new(4);

        let array = make_fsb(vec![
            Some(b"aaaa" as &[u8]),
            Some(b"bbbb"),
            Some(b"cccc"),
            Some(b"dddd"),
        ]);

        builder
            .vectorized_append(&array, &[0, 1, 2, 3])
            .unwrap();
        assert_eq!(builder.len(), 4);

        let mut results = vec![true; 4];
        builder.vectorized_equal_to(&[0, 1, 2, 3], &array, &[0, 1, 2, 3], &mut results);
        assert!(results.iter().all(|&r| r));

        // Check inequality
        let other = make_fsb(vec![
            Some(b"aaaa" as &[u8]),
            Some(b"xxxx"),
            Some(b"cccc"),
            Some(b"yyyy"),
        ]);

        let mut results = vec![true; 4];
        builder.vectorized_equal_to(&[0, 1, 2, 3], &other, &[0, 1, 2, 3], &mut results);
        assert!(results[0]);
        assert!(!results[1]);
        assert!(results[2]);
        assert!(!results[3]);
    }

    #[test]
    fn test_build_and_take_n() {
        let mut builder = FixedSizeBinaryGroupValueBuilder::<false>::new(4);

        let array = make_fsb(vec![
            Some(b"aaaa" as &[u8]),
            Some(b"bbbb"),
            Some(b"cccc"),
        ]);

        builder
            .vectorized_append(&array, &[0, 1, 2])
            .unwrap();

        // take_n(2) should return first two and leave one
        let taken = builder.take_n(2);
        let taken = taken.as_fixed_size_binary();
        assert_eq!(taken.len(), 2);
        assert_eq!(taken.value(0), b"aaaa");
        assert_eq!(taken.value(1), b"bbbb");
        assert_eq!(builder.len(), 1);

        // build should return the remaining
        let remaining = Box::new(builder).build();
        let remaining = remaining.as_fixed_size_binary();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining.value(0), b"cccc");
    }
}

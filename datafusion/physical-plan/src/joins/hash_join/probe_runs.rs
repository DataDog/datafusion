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

//! Probe adjacent equal keys once, then expand their matches in probe-row order.

use std::ops::Range;

use arrow::array::{Array, ArrayRef, AsArray, UInt32Array, UInt64Array};
use arrow::buffer::BooleanBuffer;
use arrow::compute::take;
use arrow::datatypes::DataType;
use arrow_ord::cmp::distinct;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::{NullEquality, Result};

use super::stream::lookup_join_hashmap;
use crate::joins::MapOffset;
use crate::joins::utils::{JoinHashMapType, matchable_join_keys};

/// Bounded matches for the representatives of adjacent equal probe keys.
///
/// Only inner joins without residual filters use this path. Runs need not be
/// globally sorted: exact adjacent equality is sufficient. The cached matches
/// never exceed one output batch; higher fanout falls back to ordinary probing.
#[derive(Debug, Clone)]
pub(super) struct ProbeRuns {
    starts: Vec<u32>,
    build_indices: UInt64Array,
    run_indices: UInt32Array,
    match_start: usize,
    match_end: usize,
    next_match: usize,
    probe_row: u32,
}

impl ProbeRuns {
    #[expect(
        clippy::too_many_arguments,
        reason = "Borrow the existing join key arrays and scratch buffers without a second owning context"
    )]
    pub(super) fn try_new(
        values: &[ArrayRef],
        build_values: &[ArrayRef],
        map: &dyn JoinHashMapType,
        null_equality: NullEquality,
        random_state: &RandomState,
        batch_size: usize,
        hashes: &mut Vec<u64>,
        probe_indices: &mut Vec<u32>,
        build_indices: &mut Vec<u64>,
    ) -> Result<Option<Self>> {
        let Some(first) = values.first() else {
            return Ok(None);
        };
        let rows = first.len();
        // Float, encoded and nested keys retain their existing equality path.
        if rows < 2
            || rows > u32::MAX as usize
            || values.iter().any(|value| !supports_type(value.data_type()))
        {
            return Ok(None);
        }
        // A small prefix rejects mostly unique batches before scanning all key
        // bytes. This can only decline the optimization: accepted inputs still
        // undergo exact full-batch comparison and the ordinary collision check.
        const SAMPLE_ROWS: usize = 64;
        if rows > SAMPLE_ROWS && run_boundaries(values, SAMPLE_ROWS)?.is_none() {
            return Ok(None);
        }
        let Some(boundaries) = run_boundaries(values, rows)? else {
            return Ok(None);
        };
        let runs = boundaries.count_set_bits() + 1;
        // Bound the representative lookup as well as the retained matches.
        if runs > batch_size {
            return Ok(None);
        }
        let mut starts = Vec::with_capacity(runs + 1);
        starts.push(0);
        starts.extend(boundaries.set_indices().map(|index| (index + 1) as u32));
        let representatives = UInt32Array::from(starts.clone());
        starts.push(rows as u32);
        let values = values
            .iter()
            .map(|value| take(value.as_ref(), &representatives, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        hashes.clear();
        hashes.resize(runs, 0);
        create_hashes(&values, random_state, hashes)?;
        let valid_keys = matchable_join_keys(&values, null_equality);
        let (build_indices, run_indices, next_offset) = lookup_join_hashmap(
            map,
            build_values,
            &values,
            null_equality,
            hashes,
            valid_keys.as_ref(),
            batch_size,
            (0, None),
            probe_indices,
            build_indices,
        )?;
        if next_offset.is_some() {
            // Do not retain an unbounded duplicate-key match list, or change
            // probe-major output order by expanding a partial hash chain.
            return Ok(None);
        }
        Ok(Some(Self {
            starts,
            build_indices,
            run_indices,
            match_start: 0,
            match_end: 0,
            next_match: 0,
            probe_row: 0,
        }))
    }

    pub(super) fn run_count(&self) -> usize {
        self.starts.len() - 1
    }

    pub(super) fn next_indices(
        &mut self,
        limit: usize,
    ) -> (
        UInt64Array,
        UInt32Array,
        Option<MapOffset>,
        Option<Range<usize>>,
    ) {
        let mut build = Vec::with_capacity(limit);
        let mut probe = Vec::with_capacity(limit);
        let mut only_single_matches = true;
        while self.match_start < self.build_indices.len() && build.len() < limit {
            let run = self.run_indices.value(self.match_start);
            if self.match_end == self.match_start {
                while self.match_end < self.run_indices.len()
                    && self.run_indices.value(self.match_end) == run
                {
                    self.match_end += 1;
                }
                self.probe_row = self.starts[run as usize];
                self.next_match = self.match_start;
            }
            if self.match_end - self.match_start == 1 {
                // One verified build match: fill the complete remaining run,
                // bounded by this output batch, without per-probe-row control.
                let run_end = self.starts[run as usize + 1];
                let count =
                    (limit - build.len()).min((run_end - self.probe_row) as usize);
                build.resize(
                    build.len() + count,
                    self.build_indices.value(self.match_start),
                );
                let next_probe_row = self.probe_row + count as u32;
                probe.extend(self.probe_row..next_probe_row);
                self.probe_row = next_probe_row;
                if self.probe_row == run_end {
                    self.match_start = self.match_end;
                }
                continue;
            }
            only_single_matches = false;
            let count = (limit - build.len()).min(self.match_end - self.next_match);
            build.extend_from_slice(
                &self.build_indices.values()[self.next_match..self.next_match + count],
            );
            probe.resize(probe.len() + count, self.probe_row);
            self.next_match += count;
            if self.next_match == self.match_end {
                self.probe_row += 1;
                self.next_match = self.match_start;
                if self.probe_row == self.starts[run as usize + 1] {
                    self.match_start = self.match_end;
                }
            }
        }
        let next = (self.match_start < self.build_indices.len())
            .then_some((self.probe_row as usize, None));
        // Singleton spans are strictly increasing. Their endpoints prove a
        // contiguous range in O(1); unmatched gaps make the span wider than len.
        let contiguous_probe_range = match (probe.first(), probe.last()) {
            (Some(&first), Some(&last))
                if only_single_matches
                    && first as usize + probe.len() == last as usize + 1 =>
            {
                Some(first as usize..last as usize + 1)
            }
            _ => None,
        };
        (build.into(), probe.into(), next, contiguous_probe_range)
    }
}

fn supports_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_)
            | DataType::Timestamp(_, _)
    )
}

/// OR-ing another key's boundaries can only increase the run count. Stop
/// scanning composite keys as soon as they cannot remove half the key work.
fn run_boundaries(values: &[ArrayRef], rows: usize) -> Result<Option<BooleanBuffer>> {
    let mut boundaries = adjacent_distinct(&values[0], rows)?;
    if boundaries.count_set_bits() + 1 > rows / 2 {
        return Ok(None);
    }
    for value in &values[1..] {
        boundaries = &boundaries | &adjacent_distinct(value, rows)?;
        if boundaries.count_set_bits() + 1 > rows / 2 {
            return Ok(None);
        }
    }
    Ok(Some(boundaries))
}

fn adjacent_distinct(column: &ArrayRef, rows: usize) -> Result<BooleanBuffer> {
    let length = rows - 1;
    if column.null_count() == 0
        && matches!(column.data_type(), DataType::FixedSizeBinary(16))
    {
        let (values, _) = column.as_fixed_size_binary().value_data().as_chunks::<16>();
        return Ok(BooleanBuffer::collect_bool(length, |index| {
            values[index] != values[index + 1]
        }));
    }
    let left = column.slice(0, length);
    let right = column.slice(1, length);
    Ok(distinct(&left, &right)?.values().clone())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{FixedSizeBinaryArray, Int32Array, StringArray};

    use super::*;
    use crate::joins::join_hash_map::JoinHashMapU32;

    fn prepare_runs(
        build: &[ArrayRef],
        probe: &[ArrayRef],
        null_equality: NullEquality,
        batch_size: usize,
        force_collision: bool,
    ) -> Result<Option<ProbeRuns>> {
        let random_state = RandomState::default();
        let mut hashes = vec![0; build[0].len()];
        create_hashes(build, &random_state, &mut hashes)?;
        if force_collision {
            // The middle build key differs but shares the candidate hash chain.
            hashes[1] = hashes[0];
        }
        let mut map = JoinHashMapU32::with_capacity(hashes.len());
        map.update_from_iter(Box::new(hashes.iter().enumerate().rev()), 0);
        ProbeRuns::try_new(
            probe,
            build,
            &map,
            null_equality,
            &random_state,
            batch_size,
            &mut vec![],
            &mut vec![],
            &mut vec![],
        )
    }

    fn collect_pairs(mut runs: ProbeRuns, limit: usize) -> Vec<(u64, u32)> {
        let mut result = vec![];
        loop {
            let (build, probe, next, _) = runs.next_indices(limit);
            assert!(build.len() <= limit);
            result.extend(
                build
                    .values()
                    .iter()
                    .copied()
                    .zip(probe.values().iter().copied()),
            );
            if next.is_none() {
                return result;
            }
        }
    }

    #[test]
    fn run_probe_preserves_duplicate_order_and_collision_checks() -> Result<()> {
        let build: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![7, 8, 7]))];
        let probe: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![7; 6]))];
        let runs =
            prepare_runs(&build, &probe, NullEquality::NullEqualsNothing, 4, true)?
                .expect("one run with bounded build fanout");
        let expected = (0..6)
            .flat_map(|row| [(0, row), (2, row)])
            .collect::<Vec<_>>();
        // An output limit of three splits both a run and one row's build matches.
        assert_eq!(collect_pairs(runs, 3), expected);
        Ok(())
    }

    #[test]
    fn run_probe_handles_nulls_and_all_key_columns() -> Result<()> {
        let build: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![None, Some(1), Some(1), Some(2)])),
            Arc::new(Int32Array::from(vec![5, 7, 8, 9])),
        ];
        let probe: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![
                None,
                None,
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(3),
                Some(3),
            ])),
            Arc::new(Int32Array::from(vec![5, 5, 7, 7, 8, 8, 9, 9, 0, 0])),
        ];
        for null_equality in [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ] {
            let runs = prepare_runs(&build, &probe, null_equality, 8, false)?
                .expect("five exact-key runs");
            let mut expected = vec![(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)];
            if null_equality == NullEquality::NullEqualsNull {
                expected.splice(0..0, [(0, 0), (0, 1)]);
            }
            assert_eq!(collect_pairs(runs, 3), expected);
        }
        Ok(())
    }

    #[test]
    fn run_probe_fixed_binary_slices_compare_both_halves() -> Result<()> {
        let zero = [0_u8; 16];
        let mut low = zero;
        low[15] = 1;
        let mut high = low;
        high[0] = 1;
        let build: Vec<ArrayRef> = vec![Arc::new(FixedSizeBinaryArray::try_from_iter(
            [zero, low, high].into_iter(),
        )?)];
        let probe: ArrayRef = Arc::new(FixedSizeBinaryArray::try_from_iter(
            [high, zero, zero, low, low, high, high, zero].into_iter(),
        )?);
        let runs = prepare_runs(
            &build,
            &[probe.slice(1, 6)],
            NullEquality::NullEqualsNothing,
            4,
            false,
        )?
        .expect("three sliced binary-key runs");
        assert_eq!(
            collect_pairs(runs, 4),
            vec![(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5)],
        );
        Ok(())
    }

    #[test]
    fn run_probe_falls_back_for_sparse_runs_or_excess_fanout() -> Result<()> {
        let build: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![1; 5]))];
        let repeated: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![1; 8]))];
        assert!(
            prepare_runs(&build, &repeated, NullEquality::NullEqualsNothing, 4, false)?
                .is_none()
        );
        let sparse: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))];
        assert!(
            prepare_runs(&build, &sparse, NullEquality::NullEqualsNothing, 8, false)?
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn run_probe_bulk_single_matches_preserve_gaps_and_run_boundaries() -> Result<()> {
        let build: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![1, 2]))];
        let probe: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(
            [vec![1; 70], vec![9; 4], vec![2; 17], vec![1; 5]].concat(),
        ))];
        let runs =
            prepare_runs(&build, &probe, NullEquality::NullEqualsNothing, 4, false)?
                .expect("three matched runs and an unmatched gap");
        let expected = (0..70)
            .map(|row| (0, row))
            .chain((74..91).map(|row| (1, row)))
            .chain((91..96).map(|row| (0, row)))
            .collect::<Vec<_>>();
        for limit in [1, 4, 8] {
            assert_eq!(collect_pairs(runs.clone(), limit), expected);
        }
        Ok(())
    }

    #[test]
    fn run_probe_prefix_admission_still_checks_the_full_composite_key() -> Result<()> {
        let build: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["same"])),
        ];
        // A dense prefix must not hide a mostly unique second key in the tail.
        let dense_prefix: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![0; 128])),
            Arc::new(StringArray::from(
                (0..128)
                    .map(|row| {
                        if row < 64 {
                            "same".to_owned()
                        } else {
                            row.to_string()
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
        ];
        assert!(
            prepare_runs(
                &build,
                &dense_prefix,
                NullEquality::NullEqualsNothing,
                256,
                false,
            )?
            .is_none()
        );

        // A sparse prefix may decline even if the full batch later becomes
        // dense; the caller then performs the unchanged generic join.
        let sparse_prefix: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![0; 256])),
            Arc::new(StringArray::from(
                (0..256)
                    .map(|row| {
                        if row < 64 {
                            row.to_string()
                        } else {
                            "same".to_owned()
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
        ];
        assert!(
            prepare_runs(
                &build,
                &sparse_prefix,
                NullEquality::NullEqualsNothing,
                256,
                false,
            )?
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn run_probe_materialization_reuses_only_contiguous_probe_buffers() -> Result<()> {
        use arrow::datatypes::{Field, Int32Type, Schema};
        use arrow::record_batch::RecordBatch;
        use datafusion_common::{JoinSide, JoinType};

        use crate::joins::utils::{
            ColumnIndex, build_batch_from_indices_with_probe_range,
        };

        let input_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("payload", DataType::Int32, true),
        ]));
        let output_schema = Schema::new(vec![
            Field::new("build_payload", DataType::Int32, true),
            Field::new("probe_payload", DataType::Int32, true),
        ]);
        let columns = [
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
        ];
        for (build_keys, probe_keys, limit, shares_probe, expected) in [
            (
                vec![1, 2],
                vec![1, 1, 1, 1, 2, 2, 2, 2],
                3,
                true,
                vec![
                    (10, Some(100)),
                    (10, None),
                    (10, Some(102)),
                    (10, Some(103)),
                    (11, Some(104)),
                    (11, None),
                    (11, Some(106)),
                    (11, Some(107)),
                ],
            ),
            (
                vec![1, 2],
                vec![1, 1, 9, 9, 2, 2, 2, 2],
                16,
                false,
                vec![
                    (10, Some(100)),
                    (10, None),
                    (11, Some(104)),
                    (11, None),
                    (11, Some(106)),
                    (11, Some(107)),
                ],
            ),
            (
                vec![1, 1, 2],
                vec![1, 1, 1, 1, 2, 2, 2, 2],
                16,
                false,
                vec![
                    (10, Some(100)),
                    (11, Some(100)),
                    (10, None),
                    (11, None),
                    (10, Some(102)),
                    (11, Some(102)),
                    (10, Some(103)),
                    (11, Some(103)),
                    (12, Some(104)),
                    (12, None),
                    (12, Some(106)),
                    (12, Some(107)),
                ],
            ),
        ] {
            let build_payload = Int32Array::from_iter_values(
                (0..build_keys.len()).map(|row| 10 + row as i32),
            );
            let build_keys: ArrayRef = Arc::new(Int32Array::from(build_keys));
            let probe_keys: ArrayRef = Arc::new(Int32Array::from(probe_keys));
            let mut runs = prepare_runs(
                &[Arc::clone(&build_keys)],
                &[Arc::clone(&probe_keys)],
                NullEquality::NullEqualsNothing,
                16,
                false,
            )?
            .expect("bounded representative matches");
            let build = RecordBatch::try_new(
                Arc::clone(&input_schema),
                vec![build_keys, Arc::new(build_payload)],
            )?;
            // Start with a sliced payload so sharing also checks nonzero offsets.
            let probe_payload = Int32Array::from(vec![
                Some(-1),
                Some(100),
                None,
                Some(102),
                Some(103),
                Some(104),
                None,
                Some(106),
                Some(107),
                Some(-1),
            ])
            .slice(1, 8);
            let probe = RecordBatch::try_new(
                Arc::clone(&input_schema),
                vec![probe_keys, Arc::new(probe_payload)],
            )?;
            let source_values = probe.column(1).as_primitive::<Int32Type>();
            let mut actual = vec![];
            loop {
                let (build_indices, probe_indices, next, range) =
                    runs.next_indices(limit);
                assert_eq!(range.is_some(), shares_probe);
                let output = build_batch_from_indices_with_probe_range(
                    &output_schema,
                    &build,
                    &probe,
                    &build_indices,
                    &probe_indices,
                    &columns,
                    JoinSide::Left,
                    JoinType::Inner,
                    range.as_ref(),
                )?;
                assert!(output.num_rows() <= limit);
                let build_values = output.column(0).as_primitive::<Int32Type>();
                let probe_values = output.column(1).as_primitive::<Int32Type>();
                assert_eq!(
                    source_values.values().inner().data_ptr()
                        == probe_values.values().inner().data_ptr(),
                    shares_probe,
                );
                actual.extend(
                    build_values
                        .values()
                        .iter()
                        .copied()
                        .zip(probe_values.iter()),
                );
                if next.is_none() {
                    break;
                }
            }
            assert_eq!(actual, expected);
        }
        Ok(())
    }
}

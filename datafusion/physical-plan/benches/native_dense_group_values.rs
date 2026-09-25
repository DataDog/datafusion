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

//! Native grouping factory versus the existing generic column hash table.
//! Generation and coalescing are outside timing. Each sample starts with empty
//! state; first-batch and complete-input cases expose admission and steady costs.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryArray, DictionaryArray, Int32Array, Int64Array,
    TimestampNanosecondArray,
};
use arrow::datatypes::{
    DataType, Field, Int32Type, Int64Type, Schema, SchemaRef, TimeUnit,
    TimestampNanosecondType,
};
use arrow::record_batch::RecordBatch;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use datafusion_expr::{EmitTo, GroupsAccumulator};
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::prim_op::PrimitiveGroupsAccumulator;
use datafusion_physical_plan::aggregates::group_values::multi_group_by::GroupValuesColumn;
use datafusion_physical_plan::aggregates::group_values::{GroupValues, new_group_values};
use datafusion_physical_plan::aggregates::order::GroupOrdering;
use datafusion_physical_plan::coalesce::LimitedBatchCoalescer;

const BATCH_SIZE: usize = 8192;
const ORIGIN: i64 = 1_700_000_000_000_000_000;
const INTERVAL: i64 = 30_000_000_000;

#[derive(Clone, Copy)]
enum Pattern {
    Cartesian,
    Diagonal,
    Irregular,
}

struct Case {
    name: &'static str,
    bins: usize,
    tags: usize,
    rows: usize,
    pattern: Pattern,
    coalesced: bool,
    // Literal independent expectations: (group count, sum of tag-index + 1).
    full_expected: (usize, i64),
    first_expected: (usize, i64),
}

fn input(case: &Case) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "bin",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new(
            "tag",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Binary)),
            false,
        ),
    ]));
    let values: ArrayRef = Arc::new(BinaryArray::from_iter_values(
        (0..case.tags).map(|tag| format!("metric-tag-value-{tag:04}").into_bytes()),
    ));
    let timestamps =
        TimestampNanosecondArray::from_iter_values((0..case.rows).map(|row| {
            let bin = (row % case.bins) as i64;
            let ordinal = if matches!(case.pattern, Pattern::Irregular) {
                bin * bin
            } else {
                bin
            };
            ORIGIN + ordinal * INTERVAL
        }));
    let keys = Int32Array::from_iter_values((0..case.rows).map(|row| {
        (match case.pattern {
            Pattern::Cartesian => (row / case.bins) % case.tags,
            Pattern::Diagonal | Pattern::Irregular => row % case.tags,
        }) as i32
    }));
    let dictionary =
        DictionaryArray::<Int32Type>::try_new(keys, Arc::clone(&values)).unwrap();
    let all = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(timestamps), Arc::new(dictionary)],
    )
    .unwrap();
    if !case.coalesced {
        return (0..case.rows)
            .step_by(BATCH_SIZE)
            .map(|offset| all.slice(offset, BATCH_SIZE.min(case.rows - offset)))
            .collect();
    }
    let mut coalescer = LimitedBatchCoalescer::new(schema, BATCH_SIZE, None);
    let mut batches = Vec::new();
    for offset in (0..case.rows).step_by(BATCH_SIZE / 4) {
        coalescer
            .push_batch(all.slice(offset, (BATCH_SIZE / 4).min(case.rows - offset)))
            .unwrap();
        while let Some(batch) = coalescer.next_completed_batch() {
            batches.push(batch);
        }
    }
    coalescer.finish().unwrap();
    while let Some(batch) = coalescer.next_completed_batch() {
        batches.push(batch);
    }
    for batch in &batches {
        let wrapped = batch.column(1).as_dictionary::<Int32Type>().values();
        assert!(!Arc::ptr_eq(&values, wrapped));
        assert!(values.to_data().ptr_eq(&wrapped.to_data()));
    }
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        case.rows
    );
    batches
}

fn make_groups(schema: SchemaRef, factory: bool) -> Box<dyn GroupValues> {
    if factory {
        // Exercise the production strategy choice, not a direct dense constructor.
        new_group_values(schema, &GroupOrdering::None).unwrap()
    } else {
        Box::new(GroupValuesColumn::<false>::try_new(schema).unwrap())
    }
}

fn preflight(
    batches: &[RecordBatch],
    factory: bool,
    literal: (usize, i64),
) -> (usize, usize) {
    let mut groups = make_groups(batches[0].schema(), factory);
    let mut indices = Vec::new();
    let mut accumulator = PrimitiveGroupsAccumulator::<Int64Type, _>::new(
        &DataType::Int64,
        |sum, value| *sum += value,
    );
    let mut expected = BTreeMap::new();
    let mut reported_peak_bytes = 0;
    for batch in batches {
        groups.intern(batch.columns(), &mut indices).unwrap();
        assert_eq!(indices.len(), batch.num_rows());
        let times = batch.column(0).as_primitive::<TimestampNanosecondType>();
        let tags = batch.column(1).as_dictionary::<Int32Type>();
        let values = tags.values().as_binary::<i32>();
        let weights: ArrayRef = Arc::new(Int64Array::from_iter_values(
            (0..batch.num_rows()).map(|row| {
                let key = tags.key(row).unwrap();
                let weight = key as i64 + 1;
                *expected
                    .entry((times.value(row), values.value(key).to_vec()))
                    .or_insert(0i64) += weight;
                weight
            }),
        ));
        accumulator
            .update_batch(&[weights], &indices, None, groups.len())
            .unwrap();
        reported_peak_bytes = reported_peak_bytes.max(groups.size());
    }
    assert_eq!((expected.len(), expected.values().sum::<i64>()), literal);
    assert_eq!(groups.len(), literal.0);
    let retained_bytes = groups.size();
    let keys = groups.emit(EmitTo::All).unwrap();
    let sums = accumulator.evaluate(EmitTo::All).unwrap();
    let sums = sums.as_primitive::<Int64Type>();
    let times = keys[0].as_primitive::<TimestampNanosecondType>();
    let tags = keys[1].as_dictionary::<Int32Type>();
    let values = tags.values().as_binary::<i32>();
    let actual: BTreeMap<_, _> = (0..sums.len())
        .map(|row| {
            (
                (
                    times.value(row),
                    values.value(tags.key(row).unwrap()).to_vec(),
                ),
                sums.value(row),
            )
        })
        .collect();
    assert_eq!(actual, expected);
    assert_eq!(sums.values().iter().copied().sum::<i64>(), literal.1);
    (retained_bytes, reported_peak_bytes)
}

fn bench_native_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("native_dense_group_values");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for case in [
        Case {
            name: "dense_reused",
            bins: 480,
            tags: 400,
            rows: 1_536_000,
            pattern: Pattern::Cartesian,
            coalesced: false,
            full_expected: (192_000, 307_968_000),
            first_expected: (8192, 74_016),
        },
        Case {
            name: "dense_coalesced",
            bins: 480,
            tags: 400,
            rows: 1_536_000,
            pattern: Pattern::Cartesian,
            coalesced: true,
            full_expected: (192_000, 307_968_000),
            first_expected: (8192, 74_016),
        },
        Case {
            name: "sparse_diagonal",
            bins: 256,
            tags: 256,
            rows: 524_288,
            pattern: Pattern::Diagonal,
            coalesced: false,
            full_expected: (256, 67_371_008),
            first_expected: (256, 1_052_672),
        },
        Case {
            name: "sparse_irregular",
            bins: 8192,
            tags: 32,
            rows: 524_288,
            pattern: Pattern::Irregular,
            coalesced: false,
            full_expected: (8192, 8_650_752),
            first_expected: (8192, 135_168),
        },
        Case {
            name: "high_cardinality_product",
            bins: 4096,
            tags: 4096,
            rows: 524_288,
            pattern: Pattern::Diagonal,
            coalesced: false,
            full_expected: (4096, 1_074_003_968),
            first_expected: (4096, 16_781_312),
        },
    ] {
        let batches = input(&case);
        for (phase, selected, expected) in [
            ("first_batch", &batches[..1], case.first_expected),
            ("full_input", &batches[..], case.full_expected),
        ] {
            group.throughput(Throughput::Elements(
                selected.iter().map(RecordBatch::num_rows).sum::<usize>() as u64,
            ));
            for factory in [false, true] {
                let strategy = if factory { "factory" } else { "generic" };
                let (retained_bytes, reported_peak_bytes) =
                    preflight(selected, factory, expected);
                eprintln!(
                    "{}/{phase}/{strategy}: groups={} sum={} retained_bytes={retained_bytes} reported_peak_bytes={reported_peak_bytes}",
                    case.name, expected.0, expected.1
                );
                group.bench_function(format!("{}/{phase}/{strategy}", case.name), |b| {
                    b.iter_batched_ref(
                        || {
                            (
                                make_groups(selected[0].schema(), factory),
                                Vec::with_capacity(BATCH_SIZE),
                            )
                        },
                        |(groups, indices)| {
                            for batch in selected {
                                groups.intern(batch.columns(), indices).unwrap();
                                black_box(indices.as_slice());
                            }
                            black_box(groups.len());
                        },
                        BatchSize::LargeInput,
                    );
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_native_dense);
criterion_main!(benches);

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

//! Full native HashJoin A/B workload. Copy unchanged into both source snapshots.
//! Input generation is outside timing; every timed iteration rebuilds the join,
//! probes all batches, materializes both payload columns and drains the output.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, FixedSizeBinaryArray, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use datafusion_common::{JoinType, NullEquality};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode, utils::JoinOn};
use datafusion_physical_plan::test::TestMemoryExec;
use futures::StreamExt;
use tokio::runtime::Builder;

const BUILD_ROWS: usize = 65_536;
const PROBE_ROWS: usize = 524_288;
const BATCH_SIZE: usize = 8192;

#[derive(Clone, Copy)]
enum Keys {
    Binary,
    Utf8,
    Composite,
}

fn input(keys: &[usize], kind: Keys) -> Arc<dyn ExecutionPlan> {
    let mut fields = vec![];
    let mut columns: Vec<ArrayRef> = vec![];
    match kind {
        Keys::Binary => {
            fields.push(Field::new("key", DataType::FixedSizeBinary(16), false));
            columns.push(Arc::new(
                FixedSizeBinaryArray::try_from_iter(
                    keys.iter().map(|key| (*key as u128).to_be_bytes()),
                )
                .unwrap(),
            ));
        }
        Keys::Utf8 | Keys::Composite => {
            if matches!(kind, Keys::Composite) {
                fields.push(Field::new("group_key", DataType::Int64, false));
                columns.push(Arc::new(Int64Array::from(
                    keys.iter()
                        .map(|key| ((key % BUILD_ROWS) / 64) as i64)
                        .collect::<Vec<_>>(),
                )));
            }
            fields.push(Field::new("key", DataType::Utf8, false));
            columns.push(Arc::new(StringArray::from(
                keys.iter()
                    .map(|key| format!("common-prefix-for-join-key-{key:016x}"))
                    .collect::<Vec<_>>(),
            )));
        }
    }
    fields.push(Field::new("payload", DataType::Int64, false));
    columns.push(Arc::new(Int64Array::from_iter_values(
        (0..keys.len()).map(|row| row as i64),
    )));
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    let batches = (0..batch.num_rows())
        .step_by(BATCH_SIZE)
        .map(|offset| batch.slice(offset, BATCH_SIZE.min(batch.num_rows() - offset)))
        .collect::<Vec<_>>();
    TestMemoryExec::try_new_exec(&[batches], schema, None).unwrap()
}

fn bench_probe_runs(c: &mut Criterion) {
    let rt = Builder::new_current_thread().enable_all().build().unwrap();
    let mut config = SessionConfig::default()
        .with_batch_size(BATCH_SIZE)
        .with_target_partitions(1);
    // Exercise HashMap probing in both binaries, never the independent ArrayMap.
    config
        .options_mut()
        .execution
        .perfect_hash_join_small_build_threshold = 0;
    config
        .options_mut()
        .execution
        .perfect_hash_join_min_key_density = f64::INFINITY;
    let context = Arc::new(TaskContext::default().with_session_config(config));
    let mut group = c.benchmark_group("hash_join_probe_runs");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));
    group.throughput(Throughput::Elements(PROBE_ROWS as u64));

    for (name, kind, repeated, misses) in [
        ("repeated_binary_hits", Keys::Binary, true, false),
        ("mostly_unique_binary_hits", Keys::Binary, false, false),
        ("mostly_unique_utf8_misses", Keys::Utf8, false, true),
        (
            "mostly_unique_composite_misses",
            Keys::Composite,
            false,
            true,
        ),
    ] {
        let build_keys = (0..BUILD_ROWS).collect::<Vec<_>>();
        let probe_keys = (0..PROBE_ROWS)
            .map(|row| {
                // The control retains one adjacent repetition per 64 rows.
                let key = if repeated { row / 8 } else { row - row / 64 };
                key % BUILD_ROWS + if misses { BUILD_ROWS } else { 0 }
            })
            .collect::<Vec<_>>();
        let left = input(&build_keys, kind);
        let right = input(&probe_keys, kind);
        let key_names: &[&str] = if matches!(kind, Keys::Composite) {
            &["group_key", "key"]
        } else {
            &["key"]
        };
        let on: JoinOn = key_names
            .iter()
            .map(|name| {
                (
                    col(name, &left.schema()).unwrap(),
                    col(name, &right.schema()).unwrap(),
                )
            })
            .collect();
        let payload = key_names.len();
        let projection = vec![payload, left.schema().fields().len() + payload];
        let make_join = || {
            HashJoinExec::try_new(
                Arc::clone(&left),
                Arc::clone(&right),
                on.clone(),
                None,
                &JoinType::Inner,
                Some(projection.clone()),
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap()
        };
        let run = |join: &HashJoinExec| {
            rt.block_on(async {
                let mut stream = join.execute(0, Arc::clone(&context)).unwrap();
                let mut rows = 0;
                while let Some(batch) = stream.next().await {
                    rows += black_box(batch.unwrap()).num_rows();
                }
                rows
            })
        };

        let preflight = make_join();
        assert_eq!(run(&preflight), if misses { 0 } else { PROBE_ROWS });
        let metrics = preflight.metrics().unwrap();
        let count = |name| metrics.sum_by_name(name).map(|value| value.as_usize());
        eprintln!(
            "{name}: run_probe_rows={:?}, run_probe_keys={:?}",
            count("run_probe_rows"),
            count("run_probe_keys")
        );
        group.bench_function(name, |b| {
            b.iter(|| {
                let join = make_join();
                black_box(run(&join));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_probe_runs);
criterion_main!(benches);

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

use std::sync::Arc;

use datafusion::arrow::array::{
    FixedSizeBinaryArray, Int64Array, UInt32Array, UInt64Array,
};
use datafusion::arrow::compute::SortOptions;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::utils::memory::estimate_memory_size;
use datafusion::common::{NullEquality, Result, ScalarValue};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::{ColumnarValue, JoinType};
use datafusion::physical_expr::expressions::{DynamicFilterPhysicalExpr, col, lit};
use datafusion::physical_expr::{
    LexOrdering, PhysicalExpr, PhysicalSortExpr, RangePartitioning, SplitPoint,
};
use datafusion::physical_plan::filter::FilterExecBuilder;
use datafusion::physical_plan::joins::join_hash_map::JoinHashMapU32;
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion::physical_plan::repartition::RangeExpr;
use datafusion::physical_plan::{ExecutionPlan, common};
use datafusion::prelude::SessionConfig;
use datafusion_datasource::memory::MemorySourceConfig;

#[test]
fn compound_range_filter_matches_scalar_boundaries() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("rpid", DataType::UInt32, false),
        Field::new("bhandle", DataType::FixedSizeBinary(16), false),
    ]));
    let keys = vec![col("rpid", &schema)?, col("bhandle", &schema)?];
    let ordering = LexOrdering::new(
        keys.iter()
            .map(|k| PhysicalSortExpr::new(Arc::clone(k), SortOptions::default())),
    )
    .unwrap();
    let bounds = [(1_u32, 256_u128), (1, 65_536), (2, 1)];
    let points = bounds
        .iter()
        .map(|(id, handle)| {
            SplitPoint::new(vec![
                ScalarValue::UInt32(Some(*id)),
                ScalarValue::FixedSizeBinary(16, Some(handle.to_be_bytes().to_vec())),
            ])
        })
        .collect();
    let partitioning = RangePartitioning::try_new(ordering, points)?;
    let expr = RangeExpr::try_new_with_schema(keys, &partitioning, &schema)?;
    for rows in [
        vec![(2, 1), (1, 255), (1, 256), (1, 257), (0, u128::MAX)],
        vec![
            (1, 65_535),
            (1, 65_536),
            (1, 65_536),
            (1, 65_537),
            (2, 0),
            (2, 2),
            (u32::MAX, u128::MAX),
        ],
    ] {
        let partitions = UInt32Array::from_iter_values(rows.iter().map(|(id, _)| *id));
        let handles = FixedSizeBinaryArray::try_from_iter(
            rows.iter().map(|(_, h)| h.to_be_bytes()),
        )?;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(partitions), Arc::new(handles)],
        )?;
        let ColumnarValue::Array(result) = expr.evaluate(&batch)? else {
            panic!("expected a partition for each row")
        };
        let routed = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let expected: Vec<_> = rows
            .iter()
            .map(|row| bounds.iter().filter(|bound| *bound <= row).count() as u64)
            .collect();
        assert_eq!(routed.values().as_ref(), expected.as_slice());
    }
    Ok(())
}

async fn assert_map_charge(mode: PartitionMode, dense: bool) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let input = |keys: Vec<i64>| -> Result<Arc<dyn ExecutionPlan>> {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(keys))],
        )?;
        Ok(MemorySourceConfig::try_new_exec(
            &[vec![batch]],
            Arc::clone(&schema),
            None,
        )?)
    };
    let count = if dense { 20_000 } else { 200 };
    let stride = if dense { 1 } else { 10_000 };
    let build = input((0..count as i64).map(|k| k * stride).collect())?;
    let key = col("k", &schema)?;
    let filter = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![Arc::clone(&key)],
        lit(true),
    ));
    let probe = input(vec![0, 5 * stride, 15 * stride])?;
    let probe = Arc::new(FilterExecBuilder::new(filter.clone(), probe).build()?);
    let join = HashJoinExec::try_new(
        build,
        probe,
        vec![(Arc::clone(&key), key)],
        None,
        &JoinType::Inner,
        None,
        mode,
        NullEquality::NullEqualsNothing,
        false,
    )?
    .with_dynamic_filter_expr(Arc::clone(&filter))?;

    let limit = 4 << 20;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let mut config = SessionConfig::new();
    config
        .options_mut()
        .optimizer
        .enable_join_dynamic_filter_pushdown = true;
    config
        .options_mut()
        .optimizer
        .hash_join_inlist_pushdown_max_size = 0;
    let ctx = Arc::new(
        TaskContext::default()
            .with_runtime(runtime)
            .with_session_config(config),
    );
    let batches = common::collect(join.execute(0, ctx)?).await?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    let metrics = join.metrics().unwrap();
    assert_eq!(
        metrics
            .sum_by_name("array_map_created_count")
            .unwrap()
            .as_usize(),
        usize::from(dense)
    );
    let map_bytes = if dense {
        2 * count * size_of::<u32>()
    } else {
        estimate_memory_size::<(u32, u64)>(count, size_of::<JoinHashMapU32>())?
    };
    let expected = match mode {
        PartitionMode::Partitioned => map_bytes,
        // CollectLeft also retains the single non-null i64 build buffer.
        PartitionMode::CollectLeft => map_bytes + count * size_of::<i64>(),
        PartitionMode::Auto => unreachable!(),
    };
    assert_eq!(pool.reserved(), expected);
    drop(join);
    assert_eq!(pool.reserved(), map_bytes);
    let next = MemoryConsumer::new("next operator").register(&pool);
    assert!(next.try_grow(limit - map_bytes + 1).is_err());
    drop(filter);
    assert_eq!(pool.reserved(), 0);
    next.try_grow(limit)?;
    Ok(())
}

#[tokio::test]
async fn partitioned_build_maps_stay_reserved() -> Result<()> {
    assert_map_charge(PartitionMode::Partitioned, false).await?;
    assert_map_charge(PartitionMode::Partitioned, true).await
}

#[tokio::test]
async fn collected_build_maps_stay_reserved() -> Result<()> {
    assert_map_charge(PartitionMode::CollectLeft, false).await?;
    assert_map_charge(PartitionMode::CollectLeft, true).await
}

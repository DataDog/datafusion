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

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use arrow::array::{ArrayRef, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::{Result, ScalarValue, assert_batches_eq};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, ColumnarValue, Signature, Volatility,
    create_udf,
};
use datafusion_functions_aggregate::count::count_udaf;

#[derive(Debug, PartialEq, Eq, Hash)]
struct FieldDefault;

impl AggregateUDFImpl for FieldDefault {
    fn name(&self) -> &str {
        "field_default"
    }
    fn signature(&self) -> &Signature {
        static SIGNATURE: std::sync::LazyLock<Signature> =
            std::sync::LazyLock::new(|| {
                Signature::exact(vec![DataType::UInt64], Volatility::Immutable)
            });
        &SIGNATURE
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }
    fn return_field(&self, args: &[FieldRef]) -> Result<FieldRef> {
        Ok(Arc::new(Field::new(
            self.name(),
            DataType::Int64,
            args[0].is_nullable(),
        )))
    }
    fn default_value_for_field(&self, field: &Field) -> Result<ScalarValue> {
        // A sentinel makes it clear these paths used the field-aware hook.
        Ok(ScalarValue::Int64((!field.is_nullable()).then_some(42)))
    }
    fn accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        count_udaf().accumulator(args)
    }
    fn create_sliding_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        count_udaf().create_sliding_accumulator(args)
    }
    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        count_udaf().state_fields(args)
    }
}

#[tokio::test]
async fn aggregate_field_defaults_in_empty_windows_and_subqueries() -> Result<()> {
    let ctx = SessionContext::new();
    ctx.register_udaf(AggregateUDF::from(FieldDefault).with_aliases(["field_alias"]));
    ctx.register_batch(
        "t",
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("v", DataType::UInt64, false),
                Field::new("n", DataType::UInt64, true),
            ])),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2])),
                Arc::new(UInt64Array::from(vec![Some(1), None])),
            ],
        )?,
    )?;
    for frame in [
        "ROWS BETWEEN 1 FOLLOWING AND 1 FOLLOWING",
        "ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING",
        "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING",
    ] {
        let batches = ctx.sql(&format!("SELECT field_alias(v) OVER (ORDER BY v {frame}) AS v, field_alias(n) OVER (ORDER BY v {frame}) AS n, count(*) OVER (ORDER BY v {frame}) AS c FROM t")).await?.collect().await?;
        let defaults = batches
            .iter()
            .flat_map(|b| (0..b.num_rows()).map(move |i| (b, i)))
            .filter(|(b, i)| {
                ScalarValue::try_from_array(b.column(2), *i).unwrap()
                    == ScalarValue::Int64(Some(0))
            })
            .collect::<Vec<_>>();
        assert_eq!(defaults.len(), 1);
        for (batch, row) in defaults {
            assert_eq!(
                ScalarValue::try_from_array(batch.column(0), row)?,
                ScalarValue::Int64(Some(42))
            );
            assert_eq!(
                ScalarValue::try_from_array(batch.column(1), row)?,
                ScalarValue::Int64(None)
            );
        }
    }
    let batches = ctx.sql("SELECT (SELECT field_alias(inner_t.v) FROM t inner_t WHERE inner_t.v = outer_t.v + 10) AS v, (SELECT field_alias(inner_t.n) FROM t inner_t WHERE inner_t.v = outer_t.v + 10) AS n FROM t outer_t").await?.collect().await?;
    assert_batches_eq!(
        [
            "+----+---+",
            "| v  | n |",
            "+----+---+",
            "| 42 |   |",
            "| 42 |   |",
            "+----+---+"
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn aggregate_filters_evaluate_only_matching_rows() -> Result<()> {
    for (partitions, migrated, skip_partial) in [
        (1, true, false),
        (2, true, false),
        (2, false, false),
        (2, true, true),
    ] {
        let mut config = SessionConfig::new()
            .with_target_partitions(partitions)
            .with_batch_size(8);
        config.options_mut().execution.enable_migration_aggregate = migrated;
        if skip_partial {
            config
                .options_mut()
                .execution
                .skip_partial_aggregation_probe_rows_threshold = 1;
            config
                .options_mut()
                .execution
                .skip_partial_aggregation_probe_ratio_threshold = 0.0;
        }
        let ctx = SessionContext::new_with_config(config);
        let evaluated = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&evaluated);
        ctx.register_udf(create_udf(
            "probe",
            vec![DataType::Int64],
            DataType::Int64,
            Volatility::Immutable,
            Arc::new(move |args| {
                let ColumnarValue::Array(values) = &args[0] else {
                    panic!("expected array")
                };
                counter.fetch_add(values.len(), Ordering::Relaxed);
                Ok(args[0].clone())
            }),
        ));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values(0..32)),
            Arc::new(Int64Array::from_iter_values((0..32).map(|v| v % 4))),
        ];
        ctx.register_batch(
            "t",
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("v", DataType::Int64, false),
                    Field::new("g", DataType::Int64, false),
                ])),
                columns,
            )?,
        )?;
        for group in ["", "GROUP BY g", "GROUP BY GROUPING SETS ((g), ())"] {
            for (filter, expected) in
                [("v % 4 = 0", 8), ("v < 0", 0), ("NULL", 0), ("v >= 0", 32)]
            {
                evaluated.store(0, Ordering::Relaxed);
                let batches = ctx.sql(&format!("SELECT sum(probe(v)) FILTER (WHERE {filter}), avg(probe(v)) FILTER (WHERE {filter}), min(probe(v)) FILTER (WHERE {filter}), max(probe(v)) FILTER (WHERE {filter}) FROM t {group}")).await?.collect().await?;
                assert_eq!(
                    evaluated.load(Ordering::Relaxed),
                    expected,
                    "{group} {filter}"
                );
                let expected_groups = if group.is_empty() {
                    1
                } else if group.contains("SETS") {
                    5
                } else {
                    4
                };
                assert_eq!(
                    batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
                    expected_groups
                );
                let empty_groups = if expected == 0 {
                    expected_groups
                } else if expected == 8 && !group.is_empty() {
                    3
                } else {
                    0
                };
                for column in 0..4 {
                    assert_eq!(
                        batches
                            .iter()
                            .map(|batch| batch.column(column).null_count())
                            .sum::<usize>(),
                        empty_groups,
                        "empty groups for {group} {filter}"
                    );
                }
            }
        }
        // An all-true later batch must not mark earlier, excluded groups as seen.
        let batches = ctx.sql("SELECT g, sum(v) FILTER (WHERE v >= 16) AS s FROM (SELECT v, v / 16 AS g FROM t) GROUP BY g ORDER BY g").await?.collect().await?;
        assert_batches_eq!(
            [
                "+---+-----+",
                "| g | s   |",
                "+---+-----+",
                "| 0 |     |",
                "| 1 | 376 |",
                "+---+-----+"
            ],
            &batches
        );
        // percentile_cont's existing convert_to_state only accepts one argument;
        // its ordered-set form also supplies the percentile and ordering value.
        if !skip_partial {
            evaluated.store(0, Ordering::Relaxed);
            ctx.sql("SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY probe(v)) FILTER (WHERE v % 4 = 0) FROM t GROUP BY g").await?.collect().await?;
            assert_eq!(evaluated.load(Ordering::Relaxed), 8);
        }
        evaluated.store(0, Ordering::Relaxed);
        ctx.sql("SELECT sum(probe(v)) FILTER (WHERE v % 4 = 0), sum(probe(v)) FILTER (WHERE v % 4 = 1) FROM t GROUP BY g").await?.collect().await?;
        assert_eq!(evaluated.load(Ordering::Relaxed), 16);
        ctx.sql("SELECT sum(32 / (v - 1)) FILTER (WHERE v <> 1) FROM t GROUP BY g")
            .await?
            .collect()
            .await?;
        ctx.sql("SELECT sum(DISTINCT 32 / (v - 1)) FILTER (WHERE v <> 1) FROM (SELECT * FROM t ORDER BY g) GROUP BY g").await?.collect().await?;
    }
    Ok(())
}

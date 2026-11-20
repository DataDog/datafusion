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

//! Stop coalescing [`FilterExec`] output that feeds a TopK which can finish
//! early.
//!
//! A `SortExec` with a fetch (TopK) whose input is already sorted on a prefix
//! of its sort expressions stops reading its input once it receives a row that
//! sorts after its current top `fetch` rows on that prefix. A `FilterExec`
//! between the input and the TopK normally buffers its output until it has
//! `batch_size` matching rows (or its input ends). With a selective predicate
//! this delays the TopK's first rows until most, or all, of the input has been
//! read, so the TopK cannot stop the scan.
//!
//! This rule disables output coalescing on such `FilterExec`s so that each
//! input batch's matching rows reach the TopK immediately.
//!
//! ## Example
//!
//! ```text
//! SortExec: TopK(fetch=10), expr=[a DESC, b ASC], sort_prefix=[a DESC]
//!   ProjectionExec: expr=[a, b]
//!     FilterExec: c = 1
//!       DataSourceExec: output_ordering=[a DESC]
//! ```
//!
//! becomes
//!
//! ```text
//! SortExec: TopK(fetch=10), expr=[a DESC, b ASC], sort_prefix=[a DESC]
//!   ProjectionExec: expr=[a, b]
//!     FilterExec: c = 1, coalesce_batches=false
//!       DataSourceExec: output_ordering=[a DESC]
//! ```

use std::sync::Arc;

use crate::PhysicalOptimizerRule;
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::coop::CooperativeExec;
use datafusion_physical_plan::execution_plan::replace_children_if_necessary;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::sorts::sort::SortExec;

/// A physical optimizer rule that disables output coalescing on the
/// [`FilterExec`]s feeding a TopK that can finish early on a sort prefix.
///
/// See module-level documentation for details.
#[derive(Debug, Clone, Default)]
pub struct TopKFilterCoalescing;

impl TopKFilterCoalescing {
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for TopKFilterCoalescing {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            let Some(sort_exec) = node.downcast_ref::<SortExec>() else {
                return Ok(Transformed::no(node));
            };
            if sort_exec.fetch().is_none() || sort_exec.common_sort_prefix().is_empty() {
                return Ok(Transformed::no(node));
            }
            match disable_filter_coalescing(sort_exec.input())? {
                Some(new_input) => Ok(Transformed::yes(replace_children_if_necessary(
                    node,
                    vec![new_input],
                )?)),
                None => Ok(Transformed::no(node)),
            }
        })
        .data()
    }

    fn name(&self) -> &str {
        "TopKFilterCoalescing"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Disables coalescing on every `FilterExec` reachable from `plan` through
/// nodes that stream their input in order. Returns `None` if nothing changed.
fn disable_filter_coalescing(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        let new_input = disable_filter_coalescing(filter.input())?;
        if new_input.is_none() && !filter.coalesce_batches() {
            return Ok(None);
        }
        let filter: Arc<dyn ExecutionPlan> =
            Arc::new(filter.with_coalesce_batches(false));
        return match new_input {
            Some(new_input) => {
                replace_children_if_necessary(filter, vec![new_input]).map(Some)
            }
            None => Ok(Some(filter)),
        };
    }
    if plan.is::<ProjectionExec>() || plan.is::<CooperativeExec>() {
        let Some(new_input) = disable_filter_coalescing(plan.children()[0])? else {
            return Ok(None);
        };
        return replace_children_if_necessary(Arc::clone(plan), vec![new_input])
            .map(Some);
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion_common::ScalarValue;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{binary, col, lit};
    use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::displayable;
    use datafusion_physical_plan::test::TestMemoryExec;
    use insta::assert_snapshot;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Int64, false),
        ]))
    }

    fn scan(sorted_on_a: bool) -> Arc<dyn ExecutionPlan> {
        let s = schema();
        let batch = RecordBatch::try_new(
            Arc::clone(&s),
            vec![
                Arc::new(Int64Array::from(vec![3, 2, 1])),
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![1, 0, 1])),
            ],
        )
        .unwrap();
        let mut source = TestMemoryExec::try_new(&[vec![batch]], s, None).unwrap();
        if sorted_on_a {
            source = source
                .try_with_sort_information(vec![a_desc_ordering()])
                .unwrap();
        }
        Arc::new(TestMemoryExec::update_cache(&Arc::new(source)))
    }

    fn a_desc_ordering() -> LexOrdering {
        let s = schema();
        LexOrdering::new(vec![
            PhysicalSortExpr::new_default(col("a", &s).unwrap()).desc(),
        ])
        .unwrap()
    }

    fn topk_ordering() -> LexOrdering {
        let s = schema();
        LexOrdering::new(vec![
            PhysicalSortExpr::new_default(col("a", &s).unwrap()).desc(),
            PhysicalSortExpr::new_default(col("b", &s).unwrap()).asc(),
        ])
        .unwrap()
    }

    fn filter(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let predicate = binary(
            col("c", &schema()).unwrap(),
            Operator::Eq,
            lit(ScalarValue::Int64(Some(1))),
            &schema(),
        )
        .unwrap();
        Arc::new(FilterExec::try_new(predicate, input).unwrap())
    }

    fn projection(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let s = input.schema();
        let exprs = ["a", "b"].map(|name| (col(name, &s).unwrap(), name.to_string()));
        Arc::new(ProjectionExec::try_new(exprs, input).unwrap())
    }

    fn optimize(plan: Arc<dyn ExecutionPlan>) -> String {
        let optimized = TopKFilterCoalescing::new()
            .optimize(plan, &ConfigOptions::new())
            .unwrap();
        displayable(optimized.as_ref()).indent(true).to_string()
    }

    #[test]
    fn disables_coalescing_below_topk_with_sort_prefix() {
        let plan = filter(filter(scan(true)));
        let plan = Arc::new(CooperativeExec::new(projection(plan)));
        let sort = Arc::new(SortExec::new(topk_ordering(), plan).with_fetch(Some(2)));
        assert_snapshot!(optimize(sort), @r"
        SortExec: TopK(fetch=2), expr=[a@0 DESC, b@1 ASC], preserve_partitioning=[false], sort_prefix=[a@0 DESC]
          CooperativeExec
            ProjectionExec: expr=[a@0 as a, b@1 as b]
              FilterExec: c@2 = 1, coalesce_batches=false
                FilterExec: c@2 = 1, coalesce_batches=false
                  DataSourceExec: partitions=1, partition_sizes=[1], output_ordering=a@0 DESC
        ");
    }

    #[test]
    fn keeps_coalescing_without_sort_prefix() {
        let plan = projection(filter(scan(false)));
        let sort = Arc::new(SortExec::new(topk_ordering(), plan).with_fetch(Some(2)));
        assert_snapshot!(optimize(sort), @r"
        SortExec: TopK(fetch=2), expr=[a@0 DESC, b@1 ASC], preserve_partitioning=[false]
          ProjectionExec: expr=[a@0 as a, b@1 as b]
            FilterExec: c@2 = 1
              DataSourceExec: partitions=1, partition_sizes=[1]
        ");
    }

    #[test]
    fn keeps_coalescing_without_fetch() {
        let plan = projection(filter(scan(true)));
        let sort = Arc::new(SortExec::new(topk_ordering(), plan));
        assert_snapshot!(optimize(sort), @r"
        SortExec: expr=[a@0 DESC, b@1 ASC], preserve_partitioning=[false]
          ProjectionExec: expr=[a@0 as a, b@1 as b]
            FilterExec: c@2 = 1
              DataSourceExec: partitions=1, partition_sizes=[1], output_ordering=a@0 DESC
        ");
    }
}

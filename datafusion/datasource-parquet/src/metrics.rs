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

use datafusion_physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder,
    MetricCategory, MetricType, PruningMetrics, RatioMergeStrategy, RatioMetrics, Time,
};

/// Stores Parquet metric handles for one selected scope.
///
/// The set is shared by all files in an execution partition in compact mode,
/// and dedicated to one file in verbose mode. This component is subject to
/// change and is exposed for low-level integrations through
/// [`ParquetFileReaderFactory`].
///
/// [`ParquetFileReaderFactory`]: super::ParquetFileReaderFactory
#[derive(Debug, Clone)]
pub struct ParquetMetricSet {
    /// Number of file **ranges** pruned or matched by partition or file level statistics.
    /// Pruning of files often happens at planning time but may happen at execution time
    /// if dynamic filters (e.g. from a join) result in additional pruning.
    ///
    /// This does **not** necessarily equal the number of files pruned:
    /// files may be scanned in sub-ranges to increase parallelism,
    /// in which case this will represent the number of sub-ranges pruned, not the number of files.
    /// The number of files pruned will always be less than or equal to this number.
    ///
    /// A single file may have some ranges that are not pruned and some that are pruned.
    /// For example, with a query like `ORDER BY col LIMIT 10`, the TopK dynamic filter
    /// pushdown optimization may fill up the TopK heap when reading the first part of a file,
    /// then skip the second part if file statistics indicate it cannot contain rows
    /// that would be in the TopK.
    pub files_ranges_pruned_statistics: PruningMetrics,
    /// Number of times the predicate could not be evaluated
    pub predicate_evaluation_errors: Count,
    /// Number of row groups pruned by bloom filters
    pub row_groups_pruned_bloom_filter: PruningMetrics,
    /// Number of row groups pruned due to limit pruning.
    pub limit_pruned_row_groups: PruningMetrics,
    /// Number of row groups pruned by statistics
    pub row_groups_pruned_statistics: PruningMetrics,
    /// Total number of bytes scanned
    pub bytes_scanned: Count,
    /// Total rows filtered out by predicates pushed into parquet scan
    pub pushdown_rows_pruned: Count,
    /// Total rows passed predicates pushed into parquet scan
    pub pushdown_rows_matched: Count,
    /// Total time spent evaluating row-level pushdown filters
    pub row_pushdown_eval_time: Time,
    /// Total time spent evaluating row group-level statistics filters
    pub statistics_eval_time: Time,
    /// Total time spent evaluating row group Bloom Filters
    pub bloom_filter_eval_time: Time,
    /// Total rows filtered or matched by parquet page index
    pub page_index_rows_pruned: PruningMetrics,
    /// Total pages filtered or matched by parquet page index
    pub page_index_pages_pruned: PruningMetrics,
    /// Total time spent evaluating parquet page index filters
    pub page_index_eval_time: Time,
    /// Total time spent reading and parsing metadata from the footer
    pub metadata_load_time: Time,
    /// Scan Efficiency Ratio, calculated as bytes_scanned / total_file_size
    pub scan_efficiency_ratio: RatioMetrics,
    /// Predicate Cache: Total number of rows physically read and decoded from the Parquet file.
    ///
    /// This metric tracks "cache misses" in the predicate pushdown optimization.
    /// When the specialized predicate reader cannot find the requested data in its cache,
    /// it must fall back to the "inner reader" to physically decode the data from the
    /// Parquet.
    ///
    /// This is the expensive path (IO + Decompression + Decoding).
    ///
    /// We use a Gauge here as arrow-rs reports absolute numbers rather
    /// than incremental readings, we want a `set` operation here rather
    /// than `add`. Earlier it was `Count`, which led to this issue:
    /// github.com/apache/datafusion/issues/19334
    pub predicate_cache_inner_records: Gauge,
    /// Predicate Cache: number of records read from the cache. This is the
    /// number of rows that were stored in the cache after evaluating predicates
    /// reused for the output.
    pub predicate_cache_records: Gauge,
    /// Number of errors constructing pruning predicates.
    pub predicate_creation_errors: Count,
    /// Pages skipped because row-group statistics proved a full match.
    pub page_index_pages_skipped_by_fully_matched: Count,
    baseline_metrics: Arc<BaselineMetrics>,
}

impl ParquetMetricSet {
    /// Create a filename-labelled metric set for verbose mode.
    pub fn new(
        partition: usize,
        filename: &str,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        Self::new_for_file(partition, Some(filename), metrics)
    }

    /// Create metrics shared by all files in an execution partition.
    pub(crate) fn new_compact(
        partition: usize,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        Self::new_for_file(partition, None, metrics)
    }

    fn new_for_file(
        partition: usize,
        filename: Option<&str>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        let builder = || {
            let builder = MetricBuilder::new(metrics);
            match filename {
                Some(filename) => {
                    builder.with_new_label("filename", filename.to_string())
                }
                None => builder,
            }
        };

        // -----------------------
        // 'summary' level metrics
        // -----------------------
        let row_groups_pruned_bloom_filter = builder()
            .with_type(MetricType::Summary)
            .pruning_metrics("row_groups_pruned_bloom_filter", partition);

        let limit_pruned_row_groups = builder()
            .with_type(MetricType::Summary)
            .pruning_metrics("limit_pruned_row_groups", partition);

        let row_groups_pruned_statistics = builder()
            .with_type(MetricType::Summary)
            .pruning_metrics("row_groups_pruned_statistics", partition);

        let page_index_pages_pruned = builder()
            .with_type(MetricType::Summary)
            .pruning_metrics("page_index_pages_pruned", partition);

        let bytes_scanned = builder()
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Bytes)
            .counter("bytes_scanned", partition);

        let metadata_load_time = builder()
            .with_type(MetricType::Summary)
            .subset_time("metadata_load_time", partition);

        let files_ranges_pruned_statistics = MetricBuilder::new(metrics)
            .with_type(MetricType::Summary)
            .pruning_metrics("files_ranges_pruned_statistics", partition);

        let scan_efficiency_ratio = builder()
            .with_type(MetricType::Summary)
            .ratio_metrics_with_strategy(
                "scan_efficiency_ratio",
                partition,
                RatioMergeStrategy::AddPartSetTotal,
            );

        // -----------------------
        // 'dev' level metrics
        // -----------------------
        let predicate_evaluation_errors = builder()
            .with_category(MetricCategory::Rows)
            .counter("predicate_evaluation_errors", partition);

        let pushdown_rows_pruned = builder()
            .with_category(MetricCategory::Rows)
            .counter("pushdown_rows_pruned", partition);
        let pushdown_rows_matched = builder()
            .with_category(MetricCategory::Rows)
            .counter("pushdown_rows_matched", partition);

        let row_pushdown_eval_time =
            builder().subset_time("row_pushdown_eval_time", partition);
        let statistics_eval_time =
            builder().subset_time("statistics_eval_time", partition);
        let bloom_filter_eval_time =
            builder().subset_time("bloom_filter_eval_time", partition);

        let page_index_eval_time =
            builder().subset_time("page_index_eval_time", partition);

        let page_index_rows_pruned =
            builder().pruning_metrics("page_index_rows_pruned", partition);

        let predicate_cache_inner_records = builder()
            .with_category(MetricCategory::Rows)
            .gauge("predicate_cache_inner_records", partition);

        let predicate_cache_records = builder()
            .with_category(MetricCategory::Rows)
            .gauge("predicate_cache_records", partition);

        let predicate_creation_errors = MetricBuilder::new(metrics)
            .with_category(MetricCategory::Rows)
            .global_counter("num_predicate_creation_errors");

        let page_index_pages_skipped_by_fully_matched = builder()
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Rows)
            .counter("page_index_pages_skipped_by_fully_matched", partition);

        let baseline_metrics = Arc::new(BaselineMetrics::new(metrics, partition));

        Self {
            files_ranges_pruned_statistics,
            predicate_evaluation_errors,
            row_groups_pruned_bloom_filter,
            row_groups_pruned_statistics,
            limit_pruned_row_groups,
            bytes_scanned,
            pushdown_rows_pruned,
            pushdown_rows_matched,
            row_pushdown_eval_time,
            page_index_rows_pruned,
            page_index_pages_pruned,
            statistics_eval_time,
            bloom_filter_eval_time,
            page_index_eval_time,
            metadata_load_time,
            scan_efficiency_ratio,
            predicate_cache_inner_records,
            predicate_cache_records,
            predicate_creation_errors,
            page_index_pages_skipped_by_fully_matched,
            baseline_metrics,
        }
    }

    /// Record bytes scanned and update the scan-efficiency numerator.
    pub fn add_bytes_scanned(&self, bytes: usize) {
        self.bytes_scanned.add(bytes);
        self.scan_efficiency_ratio.add_part(bytes);
    }

    /// Baseline metrics for decoder compute in this scope.
    pub(crate) fn baseline_metrics(&self) -> Arc<BaselineMetrics> {
        Arc::clone(&self.baseline_metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_scanned_is_scan_efficiency_numerator() {
        let metrics = ExecutionPlanMetricsSet::new();
        let file_metrics = ParquetMetricSet::new(0, "test.parquet", &metrics);

        file_metrics.add_bytes_scanned(42);
        assert_eq!(file_metrics.bytes_scanned.value(), 42);
        assert_eq!(file_metrics.scan_efficiency_ratio.part(), 42);
    }
}

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

//! Compact repeated dictionary values once on the materialized join build side.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, DictionaryArray, downcast_dictionary_array,
};
use arrow::compute::{cast, take};
use arrow::datatypes::{ArrowDictionaryKeyType, ArrowNativeType, DataType};
use arrow::record_batch::RecordBatch;
use datafusion_common::utils::memory::estimate_memory_size;
use datafusion_common::{Result, exec_datafusion_err};
use datafusion_execution::memory_pool::MemoryReservation;

use crate::joins::utils::BuildProbeJoinMetrics;

pub(super) fn compact_dictionary_payloads(
    batch: RecordBatch,
    reservation: &MemoryReservation,
    metrics: &BuildProbeJoinMetrics,
) -> Result<RecordBatch> {
    let mut changed = false;
    let columns = batch
        .columns()
        .iter()
        .map(|column| {
            let compacted = match column.data_type() {
                DataType::Dictionary(_, value_type)
                    if matches!(
                        value_type.as_ref(),
                        DataType::Binary | DataType::Utf8
                    ) =>
                {
                    downcast_dictionary_array!(
                        column => compact_dictionary(column, reservation, metrics),
                        _ => unreachable!("matched dictionary type")
                    )?
                }
                _ => None,
            };
            changed |= compacted.is_some();
            Ok(compacted.unwrap_or_else(|| Arc::clone(column)))
        })
        .collect::<Result<Vec<_>>>()?;
    if !changed {
        return Ok(batch);
    }
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

fn compact_dictionary<K: ArrowDictionaryKeyType>(
    dictionary: &DictionaryArray<K>,
    reservation: &MemoryReservation,
    metrics: &BuildProbeJoinMetrics,
) -> Result<Option<ArrayRef>> {
    let values = dictionary.values();
    if dictionary.is_empty()
        || values.len() < 2
        || values.null_count() != 0
        || K::Native::from_usize(values.len() - 1).is_none()
    {
        // A null value becoming a null key can violate a non-nullable field. Also,
        // valid dictionaries may have more unreferenced values than their key type fits.
        return Ok(None);
    }

    // Account for the extra live values, remap keys, final row keys and Arrow's
    // temporary interner. The cast starts with at least 1024 dictionary slots.
    // The input batch remains covered by the join's existing build reservation.
    let bytes = dictionary
        .get_array_memory_size()
        .checked_mul(2)
        .ok_or_else(|| exec_datafusion_err!("dictionary compaction size overflow"))?;
    let scratch_bytes =
        estimate_memory_size::<(usize, usize)>(values.len().max(1024), bytes)?;
    let scratch = reservation.new_empty();
    if scratch.try_grow(scratch_bytes).is_err() {
        // Compaction is optional; a tight memory limit must not prevent this join.
        return Ok(None);
    }

    // Deduplicate only the values array, never expand it to one value per build row.
    let encoded = cast(values.as_ref(), dictionary.data_type())?;
    let encoded = encoded.as_dictionary::<K>();
    if encoded.values().len() == values.len() {
        return Ok(None);
    }
    let keys = take(encoded.keys(), dictionary.keys(), None)?;
    let compacted: ArrayRef = Arc::new(DictionaryArray::<K>::try_new(
        keys.as_primitive::<K>().clone(),
        Arc::clone(encoded.values()),
    )?);

    // Arrow builders retain spare capacity, so even fewer values can occasionally
    // require more retained bytes. Preserve the existing join accounting plus that delta.
    let extra = compacted
        .get_array_memory_size()
        .saturating_sub(dictionary.get_array_memory_size());
    if reservation.try_grow(extra).is_err() {
        return Ok(None);
    }
    metrics.build_mem_used.add(extra);
    metrics
        .build_dictionary_values_deduplicated
        .add(values.len() - encoded.values().len());
    Ok(Some(compacted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Int8Array, Int32Array, StringArray};
    use arrow::datatypes::{Field, Int8Type, Int32Type, Schema};
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool,
    };

    use crate::metrics::ExecutionPlanMetricsSet;

    fn reservation(limit: usize) -> MemoryReservation {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
        MemoryConsumer::new("dictionary test").register(&pool)
    }

    #[test]
    fn compact_dictionary_preserves_slices_and_nullable_keys() -> Result<()> {
        for value_type in [DataType::Utf8, DataType::Binary] {
            let values = StringArray::from(vec!["unused", "red", "blue", "red", "blue"]);
            let values = cast(&values, &value_type)?;
            let dictionary = DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![Some(3), Some(0), None, Some(2), Some(1)]),
                values.slice(1, 4),
            )?
            .slice(1, 4);
            let schema = Arc::new(Schema::new(vec![Field::new(
                "tag",
                dictionary.data_type().clone(),
                true,
            )]));
            let batch =
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(dictionary)])?;
            let metrics = BuildProbeJoinMetrics::new(0, &ExecutionPlanMetricsSet::new());
            let batch =
                compact_dictionary_payloads(batch, &reservation(1 << 20), &metrics)?;
            assert_eq!(batch.schema(), schema);
            assert_eq!(
                batch.column(0).as_dictionary::<Int32Type>().values().len(),
                2
            );
            assert_eq!(
                cast(batch.column(0), &DataType::Utf8)?.as_string::<i32>(),
                &StringArray::from(vec![Some("red"), None, Some("red"), Some("blue")])
            );
            assert_eq!(metrics.build_dictionary_values_deduplicated.value(), 2);
        }
        Ok(())
    }

    #[test]
    fn compact_dictionary_keeps_valid_fallbacks() -> Result<()> {
        let metrics = BuildProbeJoinMetrics::new(0, &ExecutionPlanMetricsSet::new());
        let memory = reservation(1 << 20);
        let empty = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(Vec::<i32>::new()),
            Arc::new(BinaryArray::from(vec![b"a".as_slice(), b"a".as_slice()])),
        )?;
        assert!(compact_dictionary(&empty, &memory, &metrics)?.is_none());
        let nullable = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![0, 1, 0]),
            Arc::new(StringArray::from(vec![Some("a"), None])),
        )?;
        let field = Field::new("tag", nullable.data_type().clone(), false);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![field])),
            vec![Arc::new(nullable)],
        )?;
        let output = compact_dictionary_payloads(batch.clone(), &memory, &metrics)?;
        assert!(Arc::ptr_eq(output.column(0), batch.column(0)));
        assert_eq!(
            cast(output.column(0), &DataType::Utf8)?.as_string::<i32>(),
            &StringArray::from(vec![Some("a"), None, Some("a")])
        );
        let oversized = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![0]),
            Arc::new(StringArray::from_iter_values(
                (0..129).map(|i| i.to_string()),
            )),
        )?;
        assert!(compact_dictionary(&oversized, &memory, &metrics)?.is_none());
        let duplicate = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![0, 1]),
            Arc::new(BinaryArray::from(vec![b"a".as_slice(), b"a".as_slice()])),
        )?;
        assert!(compact_dictionary(&duplicate, &reservation(0), &metrics)?.is_none());
        Ok(())
    }
}

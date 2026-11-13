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

//! Evaluate aggregate arguments only on rows admitted by their FILTER.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, new_empty_array};
use arrow::buffer::BooleanBuffer;
use arrow::compute::{filter_record_batch, prep_null_mask_filter};
use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::CastExpr;
use datafusion_physical_expr_common::physical_expr::is_volatile;
use datafusion_physical_expr_common::utils::{evaluate_expressions_to_arrays, scatter};

/// Arguments retain only matching rows. Group keys must still be interned from
/// the original batch so a group with no matches retains its empty result.
pub(super) struct FilteredInput {
    pub(super) arguments: Vec<ArrayRef>,
    selection: Option<Arc<Selection>>,
    filter: Option<BooleanArray>,
}

struct Selection {
    mask: BooleanArray,
    rows: Vec<usize>,
}

impl FilteredInput {
    pub(super) fn empty(arguments: Vec<ArrayRef>) -> Self {
        Self {
            arguments,
            selection: Some(Arc::new(Selection {
                mask: BooleanArray::new(BooleanBuffer::new_unset(0), None),
                rows: vec![],
            })),
            filter: Some(BooleanArray::new(BooleanBuffer::new_unset(0), None)),
        }
    }

    pub(super) fn filter(&self) -> Option<&BooleanArray> {
        // Compacted batches omit interned groups. Keep an all-true filter for
        // those batches so accumulators track groups that have seen no values.
        self.filter.as_ref()
    }

    pub(super) fn group_indices<'a>(&self, indices: &'a [usize]) -> Cow<'a, [usize]> {
        match &self.selection {
            Some(selection) => {
                Cow::Owned(selection.rows.iter().map(|&i| indices[i]).collect())
            }
            None => Cow::Borrowed(indices),
        }
    }

    /// Partial-aggregation bypass emits one state per original row. Restore
    /// positions and keep the filter so excluded rows produce empty states.
    pub(super) fn expanded(&self) -> Result<(Vec<ArrayRef>, Option<&BooleanArray>)> {
        let Some(selection) = &self.selection else {
            return Ok((self.arguments.clone(), self.filter.as_ref()));
        };
        let arguments = self
            .arguments
            .iter()
            .map(|array| scatter(&selection.mask, array.as_ref()))
            .collect::<Result<_>>()?;
        Ok((arguments, Some(&selection.mask)))
    }
}

struct FilterScope {
    batch: RecordBatch,
    selection: Option<Arc<Selection>>,
    filter: Option<BooleanArray>,
    values: HashMap<Arc<dyn PhysicalExpr>, ColumnarValue>,
}

impl FilterScope {
    fn evaluate(&mut self, expr: &Arc<dyn PhysicalExpr>) -> Result<ColumnarValue> {
        let reusable = !is_volatile(expr);
        if reusable && let Some(value) = self.values.get(expr) {
            return Ok(value.clone());
        }
        let value = if self.batch.num_rows() == 0 {
            ColumnarValue::Array(new_empty_array(
                &expr.data_type(self.batch.schema().as_ref())?,
            ))
        } else if let Some(cast) = expr.as_ref().downcast_ref::<CastExpr>() {
            // Coercions such as AVG's Float64 cast must not prevent sharing the
            // underlying expression with SUM under the same filter.
            self.evaluate(cast.expr())?
                .cast_to(cast.cast_type(), Some(cast.cast_options()))?
        } else {
            expr.evaluate(&self.batch)?
        };
        if reusable {
            self.values.insert(Arc::clone(expr), value.clone());
        }
        Ok(value)
    }
}

/// Batch-local sharing is restricted to identical non-volatile filters.
/// Expressions under different filters must never reuse each other's values.
pub(super) struct FilteredInputEvaluator<'a> {
    batch: &'a RecordBatch,
    scopes: HashMap<Arc<dyn PhysicalExpr>, FilterScope>,
}

impl<'a> FilteredInputEvaluator<'a> {
    pub(super) fn new(batch: &'a RecordBatch) -> Self {
        Self {
            batch,
            scopes: HashMap::new(),
        }
    }

    pub(super) fn evaluate(
        &mut self,
        expressions: &[Arc<dyn PhysicalExpr>],
        filter: Option<&Arc<dyn PhysicalExpr>>,
    ) -> Result<FilteredInput> {
        let Some(filter) = filter else {
            return Ok(FilteredInput {
                arguments: evaluate_expressions_to_arrays(expressions, self.batch)?,
                selection: None,
                filter: None,
            });
        };
        let reusable = !is_volatile(filter);
        if !reusable || !self.scopes.contains_key(filter) {
            let values = filter
                .evaluate(self.batch)?
                .into_array(self.batch.num_rows())?;
            let predicate = values.as_boolean();
            let mask = if predicate.null_count() == 0 {
                predicate.clone()
            } else {
                prep_null_mask_filter(predicate)
            };
            let (batch, selection, selected) =
                if mask.true_count() == self.batch.num_rows() {
                    (self.batch.clone(), None, None)
                } else {
                    let rows: Vec<usize> = mask.values().set_indices().collect();
                    let selected =
                        BooleanArray::new(BooleanBuffer::new_set(rows.len()), None);
                    (
                        filter_record_batch(self.batch, &mask)?,
                        Some(Arc::new(Selection { mask, rows })),
                        Some(selected),
                    )
                };
            self.scopes.insert(
                Arc::clone(filter),
                FilterScope {
                    batch,
                    selection,
                    filter: selected,
                    values: HashMap::new(),
                },
            );
        }
        let scope = self
            .scopes
            .get_mut(filter)
            .expect("filter scope was inserted");
        let arguments = expressions
            .iter()
            .map(|expr| scope.evaluate(expr)?.into_array(scope.batch.num_rows()))
            .collect::<Result<_>>()?;
        Ok(FilteredInput {
            arguments,
            selection: scope.selection.clone(),
            filter: scope.filter.clone(),
        })
    }
}

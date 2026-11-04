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

use crate::logical_plan::consumer::SubstraitConsumer;
use crate::logical_plan::consumer::expr::{
    contains_struct, is_literal, name_struct_literal_like,
};
use datafusion::common::{not_impl_err, substrait_err};
use datafusion::logical_expr::{LogicalPlan, LogicalPlanBuilder, Projection};
use substrait::proto::SetRel;
use substrait::proto::set_rel::SetOp;

pub async fn from_set_rel(
    consumer: &impl SubstraitConsumer,
    set: &SetRel,
) -> datafusion::common::Result<LogicalPlan> {
    if set.inputs.len() < 2 {
        return substrait_err!("Set operation requires at least two inputs");
    }
    let mut inputs = Vec::with_capacity(set.inputs.len());
    for input in &set.inputs {
        inputs.push(consumer.consume_rel(input).await?);
    }
    let mut inputs = name_struct_literal_columns(inputs)?;

    match set.op() {
        SetOp::UnionAll => union_plans(inputs, true),
        SetOp::UnionDistinct => union_plans(inputs, false),
        SetOp::IntersectionPrimary => {
            let secondary = inputs.split_off(1);
            LogicalPlanBuilder::intersect(
                inputs.remove(0),
                union_plans(secondary, true)?,
                false,
            )
        }
        SetOp::IntersectionMultiset => intersect_plans(inputs, false),
        SetOp::IntersectionMultisetAll => intersect_plans(inputs, true),
        SetOp::MinusPrimary => except_plans(inputs, false),
        SetOp::MinusPrimaryAll => except_plans(inputs, true),
        set_op => not_impl_err!("Unsupported set operator: {set_op:?}"),
    }
}

/// Gives a column that is a struct literal in some inputs the field names that column
/// has in the other inputs.
///
/// Substrait types carry no struct field names, so a struct literal is consumed with
/// generated names (`c0`, `c1`, ...). A query padding one input with `NULL` for a struct
/// column the other input reads from a table, as in
/// `SELECT s FROM t UNION ALL SELECT NULL FROM u`, would otherwise combine `Struct{a, b}`
/// with `Struct{c0, c1}`, which DataFusion can't cast between as it matches struct fields
/// by name.
fn name_struct_literal_columns(
    inputs: Vec<LogicalPlan>,
) -> datafusion::common::Result<Vec<LogicalPlan>> {
    let width = inputs[0].schema().fields().len();
    if inputs
        .iter()
        .any(|input| input.schema().fields().len() != width)
    {
        // let the set operation report the mismatch
        return Ok(inputs);
    }

    // for each column, the type of the first input that doesn't produce it with a literal
    let named_types = (0..width)
        .map(|column| {
            inputs
                .iter()
                .filter(|input| !projects_literal(input, column))
                .map(|input| input.schema().field(column).data_type())
                .find(|data_type| contains_struct(data_type))
                .cloned()
        })
        .collect::<Vec<_>>();
    if named_types.iter().all(Option::is_none) {
        return Ok(inputs);
    }

    inputs
        .into_iter()
        .map(|input| {
            let LogicalPlan::Projection(projection) = input else {
                return Ok(input);
            };
            let mut exprs = projection.expr;
            for (expr, named_type) in exprs.iter_mut().zip(&named_types) {
                if let Some(named_type) = named_type {
                    name_struct_literal_like(expr, named_type);
                }
            }
            Ok(LogicalPlan::Projection(Projection::try_new(
                exprs,
                projection.input,
            )?))
        })
        .collect()
}

/// Returns true if `input` is a projection whose `column` is a literal.
fn projects_literal(input: &LogicalPlan, column: usize) -> bool {
    matches!(input, LogicalPlan::Projection(projection)
        if projection.expr.get(column).is_some_and(is_literal))
}

fn union_plans(
    inputs: Vec<LogicalPlan>,
    is_all: bool,
) -> datafusion::common::Result<LogicalPlan> {
    let mut inputs = inputs.into_iter();
    let Some(first) = inputs.next() else {
        return substrait_err!("Set operation requires at least two inputs");
    };
    let mut union_builder = Ok(LogicalPlanBuilder::from(first));
    for rel_plan in inputs {
        union_builder = if is_all {
            union_builder?.union(rel_plan)
        } else {
            union_builder?.union_distinct(rel_plan)
        };
    }
    union_builder?.build()
}

fn intersect_plans(
    inputs: Vec<LogicalPlan>,
    is_all: bool,
) -> datafusion::common::Result<LogicalPlan> {
    let mut inputs = inputs.into_iter();
    let Some(mut rel) = inputs.next() else {
        return substrait_err!("Set operation requires at least two inputs");
    };
    for input in inputs {
        rel = LogicalPlanBuilder::intersect(rel, input, is_all)?;
    }
    Ok(rel)
}

fn except_plans(
    inputs: Vec<LogicalPlan>,
    is_all: bool,
) -> datafusion::common::Result<LogicalPlan> {
    let mut inputs = inputs.into_iter();
    let Some(mut rel) = inputs.next() else {
        return substrait_err!("Set operation requires at least two inputs");
    };
    for input in inputs {
        rel = LogicalPlanBuilder::except(rel, input, is_all)?;
    }
    Ok(rel)
}

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
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{not_impl_err, substrait_err};
use datafusion::logical_expr::{
    Aggregate, Distinct, Expr, Limit, LogicalPlan, LogicalPlanBuilder, Projection, Sort,
    SubqueryAlias, Union,
};
use std::sync::Arc;
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
        .map(|input| name_projected_struct_literals(input, &named_types))
        .collect()
}

/// Gives the struct literals in the projection producing `input`'s columns the names in
/// `named_types`, or in each input's projection for a nested set operation. The plans
/// above them (`LIMIT`, `ORDER BY`, ...) are rebuilt with the new types.
fn name_projected_struct_literals(
    input: LogicalPlan,
    named_types: &[Option<DataType>],
) -> datafusion::common::Result<LogicalPlan> {
    match input {
        LogicalPlan::Projection(projection) => {
            let exprs = projection
                .expr
                .into_iter()
                .zip(named_types)
                .map(|(mut expr, named_type)| {
                    let Some(named_type) = named_type else {
                        return Ok(expr);
                    };
                    // keep the column's name, which the plans above may refer to: an
                    // unaliased literal is named after its value, field names included
                    let name = expr.schema_name().to_string();
                    name_struct_literal_like(&mut expr, named_type);
                    expr.alias_if_changed(name)
                })
                .collect::<datafusion::common::Result<_>>()?;
            Ok(LogicalPlan::Projection(Projection::try_new(
                exprs,
                projection.input,
            )?))
        }
        LogicalPlan::Union(union) => {
            let inputs = union
                .inputs
                .into_iter()
                .map(|input| {
                    name_projected_struct_literals(
                        Arc::unwrap_or_clone(input),
                        named_types,
                    )
                    .map(Arc::new)
                })
                .collect::<datafusion::common::Result<_>>()?;
            // derive the schema again, as rebuilding a union keeps its old one
            Ok(LogicalPlan::Union(Union::try_new_with_loose_types(inputs)?))
        }
        _ => {
            let Some(child) = columns_input(&input) else {
                return Ok(input);
            };
            let child = name_projected_struct_literals(child.clone(), named_types)?;
            input.with_new_exprs(input.expressions(), vec![child])
        }
    }
}

/// The input of `plan` if `plan` passes its input's columns through unchanged, in the
/// same positions: a `LIMIT`, an `ORDER BY`, a `DISTINCT`, a subquery alias, or an
/// aggregate that only removes duplicate rows.
fn columns_input(plan: &LogicalPlan) -> Option<&LogicalPlan> {
    match plan {
        LogicalPlan::Limit(Limit { input, .. })
        | LogicalPlan::Sort(Sort { input, .. })
        | LogicalPlan::Distinct(Distinct::All(input))
        | LogicalPlan::SubqueryAlias(SubqueryAlias { input, .. }) => Some(input),
        LogicalPlan::Aggregate(aggregate) if removes_duplicates(aggregate) => {
            Some(&aggregate.input)
        }
        _ => None,
    }
}

/// Returns true if `aggregate` only removes duplicate rows, keeping its input's columns:
/// it groups by each input column in order and computes nothing else. This is how a
/// `UNION` (without `ALL`) arrives from Substrait.
fn removes_duplicates(aggregate: &Aggregate) -> bool {
    let input_columns = aggregate.input.schema().columns();
    aggregate.aggr_expr.is_empty()
        && aggregate.group_expr.len() == input_columns.len()
        && aggregate
            .group_expr
            .iter()
            .zip(&input_columns)
            .all(|(expr, column)| matches!(expr, Expr::Column(c) if c == column))
}

/// Returns true if `input`'s `column` is produced by literals only: a literal in its
/// projection, or in every input of a nested set operation.
fn projects_literal(input: &LogicalPlan, column: usize) -> bool {
    match input {
        LogicalPlan::Projection(projection) => {
            projection.expr.get(column).is_some_and(is_literal)
        }
        LogicalPlan::Union(union) => union
            .inputs
            .iter()
            .all(|input| projects_literal(input, column)),
        _ => columns_input(input).is_some_and(|input| projects_literal(input, column)),
    }
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
    for input in inputs {
        union_builder = if is_all {
            union_builder?.union(input)
        } else {
            union_builder?.union_distinct(input)
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

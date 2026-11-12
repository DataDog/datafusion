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

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::{DFSchema, Result, ScalarValue};
use datafusion_expr::planner::{
    ExprPlanner, InsertPlanner, InsertPlannerContext, InsertPlanning, PlannerResult,
    RawCastExpr, ResolvedInsert,
};
use datafusion_expr::{LogicalPlan, TableSource};
use datafusion_sql::parser::DFParser;
use datafusion_sql::planner::SqlToRel;

use crate::common::{MockContextProvider, MockSessionState};

struct Destination;
impl TableSource for Destination {
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, false).with_metadata(HashMap::from([(
                "example:logical_type".into(),
                "counter".into(),
            )])),
            Field::new("s", DataType::Utf8, false),
            Field::new("omitted", DataType::Int64, false),
        ]))
    }
}

#[derive(Debug)]
struct SourcePlanner {
    calls: Arc<AtomicUsize>,
    fallback: bool,
}
impl InsertPlanner for SourcePlanner {
    fn plan_input(
        &self,
        insert: ResolvedInsert,
        context: &mut dyn InsertPlannerContext,
    ) -> Result<InsertPlanning> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.fallback {
            return Ok(InsertPlanning::Original(Box::new(insert)));
        }
        // This test provider accepts sparse input. A production extension must
        // prepare input according to its target provider's write contract.
        assert_eq!(insert.column_indices, vec![Some(1), Some(0), None]);
        assert!(!insert.fields[0].is_nullable());
        Ok(InsertPlanning::Planned(Box::new(
            context.plan_source(*insert.source, insert.fields)?,
        )))
    }
}

fn context(planner: Option<Arc<dyn InsertPlanner>>) -> MockContextProvider {
    let mut state =
        MockSessionState::default().with_table_source("dest", Arc::new(Destination));
    if let Some(planner) = planner {
        state = state.with_insert_planner(planner);
    }
    MockContextProvider { state }
}

fn plan(ctx: &MockContextProvider, sql: &str) -> Result<LogicalPlan> {
    let statement = DFParser::parse_sql(sql)?.pop_front().unwrap();
    SqlToRel::new(ctx).statement_to_plan(statement)
}

#[test]
fn insert_source_hints_preserve_metadata_and_infer_nullability() -> Result<()> {
    let ctx = context(Some(Arc::new(SourcePlanner {
        calls: Arc::default(),
        fallback: false,
    })));
    for (values, nullable) in [
        ("(NULL, NULL)", true),
        ("('one', 1), (NULL, NULL)", true),
        ("($1, $2)", true),
        ("('one', 1)", false),
    ] {
        let mut p = plan(&ctx, &format!("INSERT INTO dest(s, n) VALUES {values}"))?;
        if values.contains('$') {
            p = p.with_param_values(vec![
                ScalarValue::Utf8(None),
                ScalarValue::Int64(None),
            ])?;
        }
        let LogicalPlan::Dml(dml) = p else {
            panic!("expected DML")
        };
        let schema = dml.input.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
        assert_eq!(schema.field(1).data_type(), &DataType::Int64);
        assert_eq!(
            schema.field(1).metadata().get("example:logical_type"),
            Some(&"counter".into())
        );
        assert_eq!(schema.field(1).is_nullable(), nullable, "{values}");
    }
    Ok(())
}

#[test]
fn insert_source_select_keeps_the_normal_query_plan() -> Result<()> {
    let ctx = context(Some(Arc::new(SourcePlanner {
        calls: Arc::default(),
        fallback: false,
    })));
    let source = "WITH x AS (SELECT s, n FROM dest WHERE n > 1) SELECT s, n FROM x ORDER BY n LIMIT 2";
    let expected = plan(&ctx, source)?;
    let LogicalPlan::Dml(dml) = plan(&ctx, &format!("INSERT INTO dest(s, n) {source}"))?
    else {
        panic!("expected DML")
    };
    assert_eq!(dml.input.as_ref(), &expected);
    // Nested VALUES in SELECT uses its own shape, not destination hints.
    plan(
        &ctx,
        "INSERT INTO dest(s, n) SELECT 'one', x FROM (VALUES (1)) AS v(x)",
    )?;
    plan(
        &ctx,
        "INSERT INTO dest(s,n) VALUES ((SELECT 'one' FROM (VALUES (1)) AS v(x)), NULL)",
    )?;
    plan(
        &ctx,
        "INSERT INTO dest(s,n) WITH x AS (VALUES (1)) VALUES ('one', (SELECT * FROM x))",
    )?;
    Ok(())
}

#[test]
fn insert_extension_fallback_preserves_builtin_planning() -> Result<()> {
    let calls = Arc::default();
    let ctx = context(Some(Arc::new(SourcePlanner {
        calls: Arc::clone(&calls),
        fallback: true,
    })));
    for sql in [
        "INSERT INTO dest(n) VALUES (1)",
        "INSERT INTO dest(s,n) VALUES ('one', 1)",
        "INSERT INTO dest(n) SELECT n FROM dest",
    ] {
        assert_eq!(plan(&ctx, sql)?, plan(&context(None), sql)?);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    Ok(())
}

#[test]
fn insert_validation_and_source_errors_are_retained() {
    let calls = Arc::default();
    let ctx = context(Some(Arc::new(SourcePlanner {
        calls: Arc::clone(&calls),
        fallback: false,
    })));
    for sql in [
        "INSERT INTO dest(n,n) VALUES (1,2)",
        "INSERT INTO dest(unknown) VALUES (1)",
    ] {
        assert!(plan(&ctx, sql).is_err());
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    for sql in [
        "INSERT INTO dest(s,n) VALUES ('one')",
        "INSERT INTO dest(s,n) VALUES ('one',1,2)",
        "INSERT INTO dest(s,n) SELECT 1",
        "INSERT INTO dest(s,n) VALUES ($0,1)",
    ] {
        assert!(plan(&ctx, sql).is_err(), "{sql}");
    }
    assert_eq!(calls.load(Ordering::Relaxed), 4);
}

#[derive(Debug)]
struct AssignmentPlanner(Arc<AtomicUsize>);
impl ExprPlanner for AssignmentPlanner {
    fn plan_cast(
        &self,
        raw: RawCastExpr,
        _schema: &DFSchema,
    ) -> Result<PlannerResult<RawCastExpr>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(PlannerResult::Original(raw))
    }
}

#[test]
fn insert_source_uses_assignment_cast_extensions() -> Result<()> {
    let calls = Arc::default();
    let mut ctx = context(Some(Arc::new(SourcePlanner {
        calls: Arc::default(),
        fallback: false,
    })));
    ctx.state = ctx
        .state
        .with_expr_planner(Arc::new(AssignmentPlanner(Arc::clone(&calls))));
    plan(&ctx, "INSERT INTO dest(s,n) VALUES ('one',1)")?;
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    Ok(())
}

#[derive(Debug)]
struct AssignmentInput;
impl InsertPlanner for AssignmentInput {
    fn plan_input(
        &self,
        insert: ResolvedInsert,
        context: &mut dyn InsertPlannerContext,
    ) -> Result<InsertPlanning> {
        let field = Arc::clone(&insert.fields[0]);
        let source = context.plan_source(*insert.source, insert.fields)?;
        let expr = datafusion_expr::Expr::Column(datafusion_common::Column::from(
            source.schema().qualified_field(0),
        ));
        let assigned = context.plan_assignment(expr, field, source.schema())?;
        Ok(InsertPlanning::Planned(Box::new(
            datafusion_expr::logical_plan::builder::project(source, vec![assigned])?,
        )))
    }
}

#[test]
fn insert_select_assignments_use_cast_extensions() -> Result<()> {
    let calls = Arc::default();
    let mut ctx = context(Some(Arc::new(AssignmentInput)));
    ctx.state = ctx
        .state
        .with_expr_planner(Arc::new(AssignmentPlanner(Arc::clone(&calls))));
    let LogicalPlan::Dml(dml) = plan(&ctx, "INSERT INTO dest(n) SELECT NULL")? else {
        panic!("expected DML")
    };
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(dml.input.schema().field(0).is_nullable());
    assert_eq!(dml.input.schema().field(0).data_type(), &DataType::Int64);
    assert_eq!(
        dml.input
            .schema()
            .field(0)
            .metadata()
            .get("example:logical_type"),
        Some(&"counter".into())
    );
    Ok(())
}

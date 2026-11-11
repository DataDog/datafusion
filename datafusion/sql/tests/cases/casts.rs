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

use std::sync::{Arc, LazyLock};

use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion_common::{DFSchema, Result, ScalarValue, not_impl_err, plan_err};
use datafusion_expr::planner::{
    CastKind, ExprPlanner, PlannerResult, RawCastExpr, TypePlanner,
};
use datafusion_expr::{
    ColumnarValue, Expr, ExprSchemable, LogicalPlan, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDF, ScalarUDFImpl, Signature, TableSource, Volatility, lit,
};
use datafusion_sql::{parser::DFParser, planner::SqlToRel};
use sqlparser::ast::DataType as SqlType;

use crate::common::{MockContextProvider, MockSessionState};

fn token_field(name: &str) -> Field {
    Field::new(name, DataType::FixedSizeBinary(8), true).with_metadata(
        [
            ("ARROW:extension:name".into(), "custom.token".into()),
            ("ARROW:extension:metadata".into(), "version=1".into()),
        ]
        .into(),
    )
}

fn contains_token(field: &Field) -> bool {
    field.extension_type_name() == Some("custom.token")
        || matches!(field.data_type(), DataType::List(child) if contains_token(child))
}

#[derive(Debug)]
struct TokenPlanner;

impl TypePlanner for TokenPlanner {
    fn plan_type_field(&self, sql_type: &SqlType) -> Result<Option<FieldRef>> {
        Ok(match sql_type {
            SqlType::Custom(name, _)
                if name.to_string().eq_ignore_ascii_case("token") =>
            {
                Some(Arc::new(token_field("")))
            }
            _ => None,
        })
    }
}

impl ExprPlanner for TokenPlanner {
    fn plan_cast(
        &self,
        raw: RawCastExpr,
        schema: &DFSchema,
    ) -> Result<PlannerResult<RawCastExpr>> {
        if !contains_token(&raw.field) {
            return Ok(PlannerResult::Original(raw));
        }
        if raw.expr == lit("rejected") {
            return plan_err!("custom token rejected");
        }
        // The source field is available even for inferred INSERT parameters.
        let _source = raw.expr.to_field(schema)?;
        Ok(PlannerResult::Planned(
            ScalarUDF::from(TokenCast {
                target: raw.field,
                kind: raw.kind,
            })
            .call(vec![raw.expr]),
        ))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TokenCast {
    target: FieldRef,
    kind: CastKind,
}

impl ScalarUDFImpl for TokenCast {
    fn name(&self) -> &str {
        match self.kind {
            CastKind::Explicit => "token_cast",
            CastKind::Try => "token_try_cast",
            CastKind::Assignment => "token_assignment",
        }
    }

    fn signature(&self) -> &Signature {
        static SIGNATURE: LazyLock<Signature> =
            LazyLock::new(|| Signature::any(1, Volatility::Immutable));
        &SIGNATURE
    }

    fn return_type(&self, _types: &[DataType]) -> Result<DataType> {
        Ok(self.target.data_type().clone())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(Arc::new(self.target.as_ref().clone().with_nullable(
            self.kind == CastKind::Try || args.arg_fields[0].is_nullable(),
        )))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        not_impl_err!("planning-only test function")
    }
}

struct TokenTable {
    schema: SchemaRef,
    default: Expr,
}

impl TableSource for TokenTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn get_column_default(&self, name: &str) -> Option<&Expr> {
        (name == "id").then_some(&self.default)
    }
}

fn context(custom: bool) -> MockContextProvider {
    let table = TokenTable {
        schema: Arc::new(Schema::new(vec![
            token_field("id"),
            Field::new("n", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new(
                "tokens",
                DataType::List(Arc::new(token_field("item"))),
                true,
            ),
        ])),
        default: lit("fallback"),
    };
    let source = TokenTable {
        schema: Arc::new(Schema::new(vec![
            Field::new("raw", DataType::FixedSizeBinary(8), false),
            Field::new("text", DataType::Utf8, true),
            Field::new(
                "texts",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
        ])),
        default: Expr::Literal(ScalarValue::Null, None),
    };
    let mut state = MockSessionState::default()
        .with_table_source("tokens", Arc::new(table))
        .with_table_source("source", Arc::new(source));
    if custom {
        state = state
            .with_expr_planner(Arc::new(TokenPlanner))
            .with_type_planner(Arc::new(TokenPlanner));
    }
    MockContextProvider { state }
}

fn plan(sql: &str, custom: bool) -> Result<LogicalPlan> {
    let context = context(custom);
    let mut statements = DFParser::parse_sql(sql)?;
    SqlToRel::new(&context).statement_to_plan(statements.pop_front().unwrap())
}

#[test]
fn custom_cast_kinds_and_output_nullability() -> Result<()> {
    for (sql, function, nullable) in [
        ("SELECT CAST('abc' AS TOKEN)", "token_cast", false),
        ("SELECT 'abc'::TOKEN", "token_cast", false),
        ("SELECT TRY_CAST('abc' AS TOKEN)", "token_try_cast", true),
        ("SELECT CAST(NULL AS TOKEN)", "token_cast", true),
    ] {
        let plan = plan(sql, true)?;
        assert!(
            plan.display_indent().to_string().contains(function),
            "{sql}: {plan}"
        );
        let field = plan.schema().field(0);
        assert_eq!(field.metadata(), token_field("").metadata());
        assert_eq!(field.is_nullable(), nullable);
    }
    Ok(())
}

#[test]
fn custom_assignments_cover_values_select_parameters_and_defaults() -> Result<()> {
    for sql in [
        "INSERT INTO tokens (id, n, s) VALUES ('abc', 1, 'one'), ('def', 2, 'two')",
        "INSERT INTO tokens (id, n) VALUES (NULL, 1), (NULL, 2)",
        "INSERT INTO tokens (id, n) VALUES ($1, $2)",
        "INSERT INTO tokens (id) SELECT text FROM source",
        "INSERT INTO tokens (id) SELECT raw FROM source",
        "INSERT INTO tokens (tokens) SELECT texts FROM source",
        "INSERT INTO tokens (n) VALUES (1)",
    ] {
        let plan = plan(sql, true)?;
        let LogicalPlan::Dml(dml) = &plan else {
            panic!("{sql}: {plan}");
        };
        assert_eq!(
            dml.input.schema().field(0).metadata(),
            token_field("").metadata()
        );
        assert!(contains_token(dml.input.schema().field(3)));
        let display = plan.display_indent().to_string();
        assert!(display.contains("token_assignment"), "{sql}: {display}");
        if sql.contains("$1") {
            let fields = plan.get_parameter_fields()?;
            assert_eq!(
                fields["$1"].as_ref().unwrap().metadata(),
                token_field("").metadata()
            );
            assert_eq!(fields["$2"].as_ref().unwrap().data_type(), &DataType::Int64);
        }
        if sql.contains("(n) VALUES") {
            assert!(display.contains("fallback"), "{display}");
        }
        if let LogicalPlan::Projection(projection) = dml.input.as_ref()
            && let LogicalPlan::Values(values) = projection.input.as_ref()
            && sql.contains("(id")
        {
            assert_eq!(
                values.schema.field(0).metadata(),
                token_field("").metadata()
            );
        }
    }
    Ok(())
}

#[test]
fn custom_conversion_errors_and_default_fallback() -> Result<()> {
    for sql in [
        "SELECT CAST('rejected' AS TOKEN)",
        "INSERT INTO tokens (id) VALUES ('rejected')",
    ] {
        assert!(
            plan(sql, true)
                .unwrap_err()
                .to_string()
                .contains("custom token rejected")
        );
    }
    for sql in [
        "SELECT DATE '2026-01-01'",
        "SELECT CAST(1 AS BIGINT)",
        "SELECT TRY_CAST('bad' AS BIGINT)",
        "INSERT INTO person (id, first_name) VALUES (1, 'a')",
        "INSERT INTO person (id, first_name) SELECT 1, 'a'",
    ] {
        assert_eq!(
            plan(sql, true)?.display_indent().to_string(),
            plan(sql, false)?.display_indent().to_string()
        );
    }
    for sql in [
        "INSERT INTO test (t_date32) VALUES (true)",
        "INSERT INTO test (t_date32) SELECT true",
    ] {
        assert_eq!(
            plan(sql, true).unwrap_err().strip_backtrace(),
            plan(sql, false).unwrap_err().strip_backtrace()
        );
    }
    assert!(plan("INSERT INTO tokens (id) VALUES ('abc')", false).is_err());
    Ok(())
}

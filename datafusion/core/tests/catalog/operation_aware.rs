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

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::prelude::SessionContext;
use datafusion_catalog::{
    CatalogProvider, MemTable, MemoryCatalogProvider, MemorySchemaProvider,
    SchemaProvider, Session, TableProvider,
};
use datafusion_common::test_util::batches_to_string;
use datafusion_common::{Result, assert_contains, plan_err};
use datafusion_expr::{Expr, TableIntent, TableType, dml::InsertOp};
use datafusion_physical_plan::ExecutionPlan;

#[derive(Debug)]
struct ObservedTable {
    inner: MemTable,
    schema_calls: AtomicUsize,
}

impl ObservedTable {
    fn new() -> Self {
        let schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![42]))],
        )
        .unwrap();
        Self {
            inner: MemTable::try_new(schema, vec![vec![batch]]).unwrap(),
            schema_calls: AtomicUsize::new(0),
        }
    }

    fn schema_calls(&self) -> usize {
        self.schema_calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl TableProvider for ObservedTable {
    fn schema(&self) -> SchemaRef {
        self.schema_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.scan(state, projection, filters, limit).await
    }

    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.insert_into(state, input, insert_op).await
    }
}

#[derive(Debug)]
struct AuthorizedSchema {
    tables: HashMap<String, Arc<ObservedTable>>,
    allowed: Mutex<HashSet<(String, TableIntent)>>,
    calls: Mutex<Vec<(String, TableIntent)>>,
}

impl AuthorizedSchema {
    fn new() -> Self {
        Self {
            tables: ["target", "source"]
                .into_iter()
                .map(|name| (name.to_owned(), Arc::new(ObservedTable::new())))
                .collect(),
            allowed: Mutex::new(HashSet::new()),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn grant(&self, name: &str, intent: TableIntent) {
        self.allowed
            .lock()
            .unwrap()
            .insert((name.to_owned(), intent));
    }

    fn revoke(&self, name: &str, intent: TableIntent) {
        self.allowed
            .lock()
            .unwrap()
            .remove(&(name.to_owned(), intent));
    }

    fn intents(&self) -> HashSet<(String, TableIntent)> {
        self.calls.lock().unwrap().iter().cloned().collect()
    }
}

#[async_trait]
impl SchemaProvider for AuthorizedSchema {
    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    async fn table(
        &self,
        name: &str,
        intent: TableIntent,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        self.calls.lock().unwrap().push((name.to_owned(), intent));
        if !self
            .allowed
            .lock()
            .unwrap()
            .contains(&(name.to_owned(), intent))
        {
            return plan_err!("Access denied to {name} for {intent:?}");
        }
        Ok(self
            .tables
            .get(name)
            .map(|table| Arc::clone(table) as Arc<dyn TableProvider>))
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }
}

fn context(schema: Arc<dyn SchemaProvider>) -> SessionContext {
    let catalog = MemoryCatalogProvider::new();
    catalog.register_schema("public", schema).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("datafusion", Arc::new(catalog));
    ctx
}

fn authorized_context() -> (SessionContext, Arc<AuthorizedSchema>) {
    let schema = Arc::new(AuthorizedSchema::new());
    let ctx = context(Arc::clone(&schema) as Arc<dyn SchemaProvider>);
    (ctx, schema)
}

#[tokio::test]
async fn write_only_insert_executes() {
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableIntent::Insert(InsertOp::Append));
    schema.grant("source", TableIntent::Read);

    for sql in [
        "INSERT INTO target VALUES (1)",
        "INSERT INTO target SELECT id FROM source",
    ] {
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(
            batches_to_string(&batches),
            "+-------+\n| count |\n+-------+\n| 1     |\n+-------+"
        );
    }
    assert_eq!(
        schema.intents(),
        HashSet::from([
            ("target".to_owned(), TableIntent::Insert(InsertOp::Append)),
            ("source".to_owned(), TableIntent::Read),
        ])
    );
}

#[tokio::test]
async fn dataframe_write_table_preserves_insert_intent() {
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableIntent::Insert(InsertOp::Append));
    schema.grant("source", TableIntent::Read);

    let batches = ctx
        .table("source")
        .await
        .unwrap()
        .write_table("target", DataFrameWriteOptions::new())
        .await
        .unwrap();
    assert_eq!(
        batches_to_string(&batches),
        "+-------+\n| count |\n+-------+\n| 1     |\n+-------+"
    );
    assert_eq!(
        schema.intents(),
        HashSet::from([
            ("target".to_owned(), TableIntent::Insert(InsertOp::Append)),
            ("source".to_owned(), TableIntent::Read),
        ])
    );

    for operation in [InsertOp::Overwrite, InsertOp::Replace] {
        let previous_schema_calls = schema.tables["target"].schema_calls();
        let err = ctx
            .table("source")
            .await
            .unwrap()
            .write_table(
                "target",
                DataFrameWriteOptions::new().with_insert_operation(operation),
            )
            .await
            .unwrap_err();
        assert_contains!(
            err.to_string(),
            format!("Access denied to target for Insert({operation:?})")
        );
        assert_eq!(
            schema.tables["target"].schema_calls(),
            previous_schema_calls
        );
    }
}

#[tokio::test]
async fn unauthorized_source_fails_before_schema() {
    for sql in [
        "INSERT INTO target SELECT id FROM source",
        "INSERT INTO target WITH input AS (SELECT id FROM source) SELECT id FROM input",
        "INSERT INTO target SELECT (SELECT id FROM source)",
    ] {
        let (ctx, schema) = authorized_context();
        schema.grant("target", TableIntent::Insert(InsertOp::Append));

        let err = ctx.sql(sql).await.unwrap_err();
        assert_contains!(err.to_string(), "Access denied to source for Read");
        assert_eq!(schema.tables["source"].schema_calls(), 0, "{sql}");
    }
}

#[tokio::test]
async fn unauthorized_target_fails_before_schema() {
    let (ctx, schema) = authorized_context();
    schema.grant("source", TableIntent::Read);

    let err = ctx
        .sql("INSERT INTO target SELECT id FROM source")
        .await
        .unwrap_err();
    assert_contains!(
        err.to_string(),
        "Access denied to target for Insert(Append)"
    );
    assert_eq!(schema.tables["target"].schema_calls(), 0);
}

#[tokio::test]
async fn self_insert_requires_read_and_insert_for_all_name_forms() {
    for sql in [
        "INSERT INTO target SELECT id FROM target",
        "INSERT INTO target SELECT s.id FROM target AS s",
        "INSERT INTO target SELECT s.id FROM public.target AS s",
        "INSERT INTO public.target SELECT s.id FROM target AS s",
        "INSERT INTO datafusion.public.target SELECT s.id FROM public.target AS s",
        "INSERT INTO target WITH input AS (SELECT id FROM target) SELECT id FROM input",
    ] {
        let (ctx, schema) = authorized_context();
        schema.grant("target", TableIntent::Insert(InsertOp::Append));
        let err = ctx.sql(sql).await.unwrap_err();
        assert_contains!(err.to_string(), "Access denied to target for Read");

        schema.grant("target", TableIntent::Read);
        ctx.sql(sql).await.unwrap();
        assert_eq!(
            schema.intents(),
            HashSet::from([
                ("target".to_owned(), TableIntent::Insert(InsertOp::Append)),
                ("target".to_owned(), TableIntent::Read),
            ]),
            "{sql}"
        );

        schema.revoke("target", TableIntent::Insert(InsertOp::Append));
        let err = ctx.sql(sql).await.unwrap_err();
        assert_contains!(
            err.to_string(),
            "Access denied to target for Insert(Append)"
        );
    }
}

#[tokio::test]
async fn later_queries_do_not_reuse_previous_authorization() {
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableIntent::Insert(InsertOp::Append));
    ctx.sql("INSERT INTO target VALUES (1)").await.unwrap();
    let before_read = schema.tables["target"].schema_calls();

    let err = ctx.sql("SELECT * FROM target").await.unwrap_err();
    assert_contains!(err.to_string(), "Access denied to target for Read");
    assert_eq!(schema.tables["target"].schema_calls(), before_read);

    schema.grant("target", TableIntent::Read);
    ctx.sql("SELECT * FROM target").await.unwrap();
    schema.revoke("target", TableIntent::Read);
    let before_revoked_read = schema.tables["target"].schema_calls();

    let err = ctx.sql("SELECT * FROM target").await.unwrap_err();
    assert_contains!(err.to_string(), "Access denied to target for Read");
    assert_eq!(schema.tables["target"].schema_calls(), before_revoked_read);
}

#[tokio::test]
async fn write_operations_preserve_target_intent() {
    for (sql, intent) in [
        (
            "INSERT INTO target VALUES (1)",
            TableIntent::Insert(InsertOp::Append),
        ),
        (
            "INSERT OVERWRITE target VALUES (1)",
            TableIntent::Insert(InsertOp::Overwrite),
        ),
        (
            "REPLACE INTO target VALUES (1)",
            TableIntent::Insert(InsertOp::Replace),
        ),
        (
            "INSERT OR REPLACE INTO target VALUES (1)",
            TableIntent::Insert(InsertOp::Replace),
        ),
        (
            "UPDATE target SET id = id + 1 WHERE id = 42",
            TableIntent::Update,
        ),
        ("DELETE FROM target WHERE id = 42", TableIntent::Delete),
        ("TRUNCATE TABLE target", TableIntent::Truncate),
        (
            "MERGE INTO target AS t USING source AS s ON t.id = s.id WHEN MATCHED THEN DELETE",
            TableIntent::Merge {
                insert: false,
                update: false,
                delete: true,
            },
        ),
    ] {
        let (ctx, schema) = authorized_context();
        schema.grant("target", intent);
        schema.grant("source", TableIntent::Read);

        ctx.state().create_logical_plan(sql).await.unwrap();
        assert!(
            schema.intents().contains(&("target".to_owned(), intent)),
            "{sql}"
        );
        assert!(
            !schema
                .intents()
                .contains(&("target".to_owned(), TableIntent::Read)),
            "{sql}"
        );
        if matches!(intent, TableIntent::Merge { .. }) {
            assert!(
                schema
                    .intents()
                    .contains(&("source".to_owned(), TableIntent::Read))
            );
        }
    }
}

#[tokio::test]
async fn append_permission_does_not_authorize_overwrite_or_replace() {
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableIntent::Insert(InsertOp::Append));
    for (sql, operation) in [
        ("INSERT OVERWRITE target VALUES (1)", "Overwrite"),
        ("REPLACE INTO target VALUES (1)", "Replace"),
        ("INSERT OR REPLACE INTO target VALUES (1)", "Replace"),
    ] {
        let err = ctx.sql(sql).await.unwrap_err();
        assert_contains!(
            err.to_string(),
            format!("Access denied to target for Insert({operation})")
        );
        assert_eq!(schema.tables["target"].schema_calls(), 0, "{sql}");
    }
}

#[tokio::test]
async fn write_target_subqueries_need_independent_read_access() {
    for (sql, intent) in [
        (
            "UPDATE target SET id = (SELECT max(id) FROM target)",
            TableIntent::Update,
        ),
        (
            "DELETE FROM target WHERE id IN (SELECT id FROM target)",
            TableIntent::Delete,
        ),
        (
            "MERGE INTO target AS t USING public.target AS s ON t.id = s.id WHEN MATCHED THEN DELETE",
            TableIntent::Merge {
                insert: false,
                update: false,
                delete: true,
            },
        ),
    ] {
        let (ctx, schema) = authorized_context();
        schema.grant("target", intent);
        let err = ctx.sql(sql).await.unwrap_err();
        assert_contains!(err.to_string(), "Access denied to target for Read");

        schema.grant("target", TableIntent::Read);
        ctx.state().create_logical_plan(sql).await.unwrap();
        assert_eq!(
            schema.intents(),
            HashSet::from([
                ("target".to_owned(), intent),
                ("target".to_owned(), TableIntent::Read),
            ]),
            "{sql}"
        );
    }
}

#[tokio::test]
async fn read_lookup_requires_read_permission() {
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableIntent::Insert(InsertOp::Append));

    let err = schema.table("target", TableIntent::Read).await.unwrap_err();
    assert_contains!(err.to_string(), "Access denied to target for Read");
    let err = ctx.table("target").await.unwrap_err();
    assert_contains!(err.to_string(), "Access denied to target for Read");
    assert_eq!(schema.tables["target"].schema_calls(), 0);
}

#[tokio::test]
async fn existing_schema_implementations_keep_working() {
    let schema = Arc::new(MemorySchemaProvider::new());
    schema
        .register_table("target".to_owned(), Arc::new(ObservedTable::new()))
        .unwrap();

    for intent in [
        TableIntent::Read,
        TableIntent::Insert(InsertOp::Append),
        TableIntent::Insert(InsertOp::Overwrite),
        TableIntent::Insert(InsertOp::Replace),
        TableIntent::Update,
        TableIntent::Delete,
        TableIntent::Truncate,
        TableIntent::Merge {
            insert: false,
            update: false,
            delete: true,
        },
    ] {
        assert!(schema.table("target", intent).await.unwrap().is_some());
        assert!(schema.table("missing", intent).await.unwrap().is_none());
    }

    let ctx = context(schema);
    ctx.sql("INSERT INTO target SELECT id FROM target")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batches = ctx
        .sql("SELECT id FROM target")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        batches_to_string(&batches),
        "+----+\n| id |\n+----+\n| 42 |\n| 42 |\n+----+"
    );
}

#[tokio::test]
async fn describe_permission_does_not_allow_select() {
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableIntent::Describe);
    ctx.sql("DESCRIBE target")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        schema.intents(),
        HashSet::from([("target".into(), TableIntent::Describe)])
    );
    let calls = schema.tables["target"].schema_calls();
    let error = ctx.sql("SELECT * FROM target").await.unwrap_err();
    assert_contains!(error.to_string(), "Access denied to target for Read");
    assert_eq!(schema.tables["target"].schema_calls(), calls);
}

#[tokio::test]
async fn ddl_needs_its_own_intent_without_a_describe_prerequisite() {
    for (sql, intent) in [
        ("CREATE INDEX idx ON target(id)", TableIntent::CreateIndex),
        (
            "CREATE TABLE IF NOT EXISTS target(id BIGINT)",
            TableIntent::CreateTable { or_replace: false },
        ),
        (
            "CREATE OR REPLACE TABLE target AS SELECT 1 AS id",
            TableIntent::CreateTable { or_replace: true },
        ),
        (
            "CREATE OR REPLACE VIEW target AS SELECT 1 AS id",
            TableIntent::CreateView { or_replace: true },
        ),
        ("DROP TABLE target", TableIntent::DropTable),
        ("DROP TABLE IF EXISTS target", TableIntent::DropTable),
        ("DROP VIEW IF EXISTS target", TableIntent::DropView),
        (
            "CREATE EXTERNAL TABLE IF NOT EXISTS target STORED AS CSV LOCATION 'missing.csv'",
            TableIntent::CreateTable { or_replace: false },
        ),
    ] {
        let (ctx, schema) = authorized_context();
        schema.grant("target", TableIntent::Describe);
        let error = ctx.sql(sql).await.unwrap_err();
        assert_contains!(
            error.to_string(),
            format!("Access denied to target for {intent:?}")
        );
        assert_eq!(schema.tables["target"].schema_calls(), 0, "{sql}");

        let (ctx, schema) = authorized_context();
        schema.grant("target", intent);
        ctx.state().create_logical_plan(sql).await.unwrap();
        assert_eq!(
            schema.intents(),
            HashSet::from([("target".into(), intent)]),
            "{sql}"
        );
        // The default schema disallows mutations. Reaching that error is distinct
        // from requiring an unrelated metadata grant during execution.
        if let Err(error) = ctx.sql(sql).await {
            assert!(
                !error.to_string().contains("Access denied"),
                "{sql}: {error}"
            );
        }
        assert_eq!(
            schema.intents(),
            HashSet::from([("target".into(), intent)]),
            "{sql}"
        );
    }
}

#[tokio::test]
async fn direct_ddl_execution_checks_absent_names() {
    for (sql, intent) in [
        (
            "CREATE TABLE absent(id BIGINT)",
            TableIntent::CreateTable { or_replace: false },
        ),
        (
            "CREATE VIEW absent AS SELECT 1 AS id",
            TableIntent::CreateView { or_replace: false },
        ),
        (
            "CREATE EXTERNAL TABLE absent STORED AS CSV LOCATION 'missing.csv'",
            TableIntent::CreateTable { or_replace: false },
        ),
        ("DROP TABLE IF EXISTS absent", TableIntent::DropTable),
        ("DROP VIEW IF EXISTS absent", TableIntent::DropView),
    ] {
        let (ctx, schema) = authorized_context();
        schema.grant("absent", intent);
        let plan = ctx.state().create_logical_plan(sql).await.unwrap();
        assert_eq!(schema.intents(), HashSet::from([("absent".into(), intent)]));
        schema.revoke("absent", intent);
        let error = ctx.execute_logical_plan(plan.clone()).await.unwrap_err();
        assert_contains!(
            error.to_string(),
            format!("Access denied to absent for {intent:?}")
        );
        schema.grant("absent", intent);
        if let Err(error) = ctx.execute_logical_plan(plan).await {
            assert!(
                !error.to_string().contains("Access denied"),
                "{sql}: {error}"
            );
        }
    }
}

#[tokio::test]
async fn file_resolution_requires_explicit_opt_in() {
    let ctx = SessionContext::new();
    for name in [
        "tests/data/example.csv",
        "https://example.com/secret.csv",
        "s3://private/secret.parquet",
    ] {
        assert!(ctx.table_provider(name, TableIntent::Read).await.is_err());
        assert!(ctx.sql(&format!("SELECT * FROM '{name}'")).await.is_err());
    }
    assert!(
        ctx.runtime_env()
            .object_store(
                datafusion::datasource::listing::ListingTableUrl::parse(
                    "https://example.com/"
                )
                .unwrap()
            )
            .is_err()
    );
    assert!(
        ctx.runtime_env()
            .object_store(
                datafusion::datasource::listing::ListingTableUrl::parse("s3://private/")
                    .unwrap()
            )
            .is_err()
    );
    ctx.register_csv(
        "explicit_file",
        "tests/data/example.csv",
        Default::default(),
    )
    .await
    .unwrap();
    assert!(
        !ctx.sql("SELECT * FROM explicit_file")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .is_empty()
    );
}

#[derive(Debug)]
struct ForwardingSchema(Arc<dyn SchemaProvider>);

#[async_trait]
impl SchemaProvider for ForwardingSchema {
    fn table_names(&self) -> Vec<String> {
        self.0.table_names()
    }
    fn table_exist(&self, name: &str) -> bool {
        self.0.table_exist(name)
    }
    async fn table(
        &self,
        name: &str,
        intent: TableIntent,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        self.0.table(name, intent).await
    }
}

#[tokio::test]
async fn forwarding_wrapper_preserves_mutation_intent() {
    let schema = Arc::new(AuthorizedSchema::new());
    schema.grant("target", TableIntent::Read);
    let ctx = context(Arc::new(ForwardingSchema(schema.clone())));
    let error = ctx.sql("INSERT INTO target VALUES (1)").await.unwrap_err();
    assert_contains!(
        error.to_string(),
        "Access denied to target for Insert(Append)"
    );
    assert_eq!(schema.tables["target"].schema_calls(), 0);
}

#[tokio::test]
async fn merge_prefetch_and_planning_preserve_every_clause_action() {
    for mask in 1..8 {
        let intent = TableIntent::Merge {
            insert: mask & 1 != 0,
            update: mask & 2 != 0,
            delete: mask & 4 != 0,
        };
        let mut sql =
            "MERGE INTO target AS t USING source AS s ON t.id = s.id".to_owned();
        if mask & 2 != 0 {
            sql.push_str(" WHEN MATCHED AND s.id = 1 THEN UPDATE SET id = s.id");
        }
        if mask & 4 != 0 {
            sql.push_str(" WHEN MATCHED THEN DELETE");
        }
        if mask & 1 != 0 {
            sql.push_str(" WHEN NOT MATCHED THEN INSERT (id) VALUES (s.id)");
        }
        let (ctx, schema) = authorized_context();
        schema.grant("source", TableIntent::Read);
        let error = ctx.state().create_logical_plan(&sql).await.unwrap_err();
        assert_contains!(
            error.to_string(),
            format!("Access denied to target for {intent:?}")
        );
        assert_eq!(schema.tables["target"].schema_calls(), 0);
        schema.grant("target", intent);
        ctx.state().create_logical_plan(&sql).await.unwrap();
        assert_eq!(
            schema.intents(),
            HashSet::from([
                ("target".into(), intent),
                ("source".into(), TableIntent::Read)
            ])
        );
    }
}

#[tokio::test]
async fn show_and_information_schema_request_describe() {
    let schema = Arc::new(AuthorizedSchema::new());
    let ctx = SessionContext::new_with_config(
        datafusion::execution::context::SessionConfig::new()
            .with_information_schema(true),
    );
    ctx.catalog("datafusion")
        .unwrap()
        .register_schema("public", Arc::clone(&schema) as Arc<dyn SchemaProvider>)
        .unwrap();
    for table in ["target", "source"] {
        schema.grant(table, TableIntent::Describe);
    }
    for sql in [
        "SHOW COLUMNS FROM target",
        "SHOW CREATE TABLE target",
        "SELECT * FROM information_schema.columns",
        "SELECT * FROM information_schema.views",
        "SELECT * FROM information_schema.tables",
    ] {
        ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert!(
            schema
                .intents()
                .iter()
                .all(|(_, intent)| *intent == TableIntent::Describe),
            "{sql}"
        );
    }
    assert!(
        schema
            .intents()
            .contains(&("target".into(), TableIntent::Describe))
    );
}

#[derive(Debug)]
struct DdlOnlySchema {
    inner: MemorySchemaProvider,
    allowed: TableIntent,
}

#[async_trait]
impl SchemaProvider for DdlOnlySchema {
    fn table_names(&self) -> Vec<String> {
        self.inner.table_names()
    }
    fn table_exist(&self, name: &str) -> bool {
        self.inner.table_exist(name)
    }
    async fn table(
        &self,
        name: &str,
        intent: TableIntent,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        if intent != self.allowed {
            return plan_err!("unexpected prerequisite: {intent:?}");
        }
        self.inner.table(name, intent).await
    }
    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        self.inner.register_table(name, table)
    }
    fn deregister_table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        self.inner.deregister_table(name)
    }
}

#[tokio::test]
async fn ddl_execution_succeeds_with_only_the_requested_operation() {
    for (sql, allowed, existing) in [
        (
            "CREATE TABLE target(id BIGINT)",
            TableIntent::CreateTable { or_replace: false },
            false,
        ),
        (
            "CREATE TABLE IF NOT EXISTS target(id BIGINT)",
            TableIntent::CreateTable { or_replace: false },
            true,
        ),
        (
            "CREATE OR REPLACE TABLE target AS SELECT 1 AS id",
            TableIntent::CreateTable { or_replace: true },
            true,
        ),
        (
            "SELECT 1 AS id INTO target",
            TableIntent::CreateTable { or_replace: false },
            false,
        ),
        (
            "CREATE VIEW target AS SELECT 1 AS id",
            TableIntent::CreateView { or_replace: false },
            false,
        ),
        (
            "CREATE OR REPLACE VIEW target AS SELECT 1 AS id",
            TableIntent::CreateView { or_replace: true },
            true,
        ),
        (
            "CREATE OR REPLACE EXTERNAL TABLE target STORED AS CSV LOCATION 'tests/data/example.csv'",
            TableIntent::CreateTable { or_replace: true },
            true,
        ),
        ("DROP TABLE target", TableIntent::DropTable, true),
        ("DROP VIEW target", TableIntent::DropView, true),
        ("DROP TABLE IF EXISTS target", TableIntent::DropTable, false),
        ("DROP VIEW IF EXISTS target", TableIntent::DropView, false),
    ] {
        let schema = Arc::new(DdlOnlySchema {
            inner: MemorySchemaProvider::new(),
            allowed,
        });
        if existing {
            let provider: Arc<dyn TableProvider> = if allowed == TableIntent::DropView {
                Arc::new(datafusion::datasource::ViewTable::new(
                    datafusion_expr::LogicalPlanBuilder::empty(false)
                        .build()
                        .unwrap(),
                    None,
                ))
            } else {
                Arc::new(ObservedTable::new())
            };
            schema.register_table("target".into(), provider).unwrap();
        }
        let ctx = context(schema.clone());
        ctx.sql(sql)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .collect()
            .await
            .unwrap();
        assert_eq!(
            schema.table_exist("target"),
            !matches!(allowed, TableIntent::DropTable | TableIntent::DropView),
            "{sql}"
        );
    }
}

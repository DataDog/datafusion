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
    CatalogProvider, DynamicFileSchemaProvider, MemTable, MemoryCatalogProvider,
    MemorySchemaProvider, SchemaProvider, Session, TableProvider, UrlTableFactory,
};
use datafusion_common::test_util::batches_to_string;
use datafusion_common::{Result, assert_contains, plan_err};
use datafusion_expr::{Expr, TableAccess, TableType, dml::InsertOp};
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
    allowed: Mutex<HashSet<(String, TableAccess)>>,
    calls: Mutex<Vec<(String, TableAccess)>>,
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

    fn grant(&self, name: &str, access: TableAccess) {
        self.allowed
            .lock()
            .unwrap()
            .insert((name.to_owned(), access));
    }

    fn revoke(&self, name: &str, access: TableAccess) {
        self.allowed
            .lock()
            .unwrap()
            .remove(&(name.to_owned(), access));
    }

    fn accesses(&self) -> HashSet<(String, TableAccess)> {
        self.calls.lock().unwrap().iter().cloned().collect()
    }
}

#[async_trait]
impl SchemaProvider for AuthorizedSchema {
    fn table_names(&self) -> Vec<String> {
        self.tables.keys().cloned().collect()
    }

    async fn table_with_access(
        &self,
        name: &str,
        access: TableAccess,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        self.calls.lock().unwrap().push((name.to_owned(), access));
        if !self
            .allowed
            .lock()
            .unwrap()
            .contains(&(name.to_owned(), access))
        {
            return plan_err!("Access denied to {name} for {access:?}");
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
    schema.grant("target", TableAccess::Insert(InsertOp::Append));
    schema.grant("source", TableAccess::Read);

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
        schema.accesses(),
        HashSet::from([
            ("target".to_owned(), TableAccess::Insert(InsertOp::Append)),
            ("source".to_owned(), TableAccess::Read),
        ])
    );
}

#[tokio::test]
async fn dataframe_write_table_preserves_insert_intent() {
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableAccess::Insert(InsertOp::Append));
    schema.grant("source", TableAccess::Read);

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
        schema.accesses(),
        HashSet::from([
            ("target".to_owned(), TableAccess::Insert(InsertOp::Append)),
            ("source".to_owned(), TableAccess::Read),
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
        schema.grant("target", TableAccess::Insert(InsertOp::Append));

        let err = ctx.sql(sql).await.unwrap_err();
        assert_contains!(err.to_string(), "Access denied to source for Read");
        assert_eq!(schema.tables["source"].schema_calls(), 0, "{sql}");
    }
}

#[tokio::test]
async fn unauthorized_target_fails_before_schema() {
    let (ctx, schema) = authorized_context();
    schema.grant("source", TableAccess::Read);

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
        schema.grant("target", TableAccess::Insert(InsertOp::Append));
        let err = ctx.sql(sql).await.unwrap_err();
        assert_contains!(err.to_string(), "Access denied to target for Read");

        schema.grant("target", TableAccess::Read);
        ctx.sql(sql).await.unwrap();
        assert_eq!(
            schema.accesses(),
            HashSet::from([
                ("target".to_owned(), TableAccess::Insert(InsertOp::Append)),
                ("target".to_owned(), TableAccess::Read),
            ]),
            "{sql}"
        );

        schema.revoke("target", TableAccess::Insert(InsertOp::Append));
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
    schema.grant("target", TableAccess::Insert(InsertOp::Append));
    ctx.sql("INSERT INTO target VALUES (1)").await.unwrap();
    let before_read = schema.tables["target"].schema_calls();

    let err = ctx.sql("SELECT * FROM target").await.unwrap_err();
    assert_contains!(err.to_string(), "Access denied to target for Read");
    assert_eq!(schema.tables["target"].schema_calls(), before_read);

    schema.grant("target", TableAccess::Read);
    ctx.sql("SELECT * FROM target").await.unwrap();
    schema.revoke("target", TableAccess::Read);
    let before_revoked_read = schema.tables["target"].schema_calls();

    let err = ctx.sql("SELECT * FROM target").await.unwrap_err();
    assert_contains!(err.to_string(), "Access denied to target for Read");
    assert_eq!(schema.tables["target"].schema_calls(), before_revoked_read);
}

#[tokio::test]
async fn write_operations_preserve_target_intent() {
    for (sql, access) in [
        (
            "INSERT INTO target VALUES (1)",
            TableAccess::Insert(InsertOp::Append),
        ),
        (
            "INSERT OVERWRITE target VALUES (1)",
            TableAccess::Insert(InsertOp::Overwrite),
        ),
        (
            "REPLACE INTO target VALUES (1)",
            TableAccess::Insert(InsertOp::Replace),
        ),
        (
            "INSERT OR REPLACE INTO target VALUES (1)",
            TableAccess::Insert(InsertOp::Replace),
        ),
        (
            "UPDATE target SET id = id + 1 WHERE id = 42",
            TableAccess::Update,
        ),
        ("DELETE FROM target WHERE id = 42", TableAccess::Delete),
        ("TRUNCATE TABLE target", TableAccess::Truncate),
        (
            "MERGE INTO target AS t USING source AS s ON t.id = s.id WHEN MATCHED THEN DELETE",
            TableAccess::Merge {
                insert: false,
                update: false,
                delete: true,
            },
        ),
    ] {
        let (ctx, schema) = authorized_context();
        schema.grant("target", access);
        schema.grant("source", TableAccess::Read);

        ctx.state().create_logical_plan(sql).await.unwrap();
        assert!(
            schema.accesses().contains(&("target".to_owned(), access)),
            "{sql}"
        );
        assert!(
            !schema
                .accesses()
                .contains(&("target".to_owned(), TableAccess::Read)),
            "{sql}"
        );
        if matches!(access, TableAccess::Merge { .. }) {
            assert!(
                schema
                    .accesses()
                    .contains(&("source".to_owned(), TableAccess::Read))
            );
        }
    }
}

#[tokio::test]
async fn append_permission_does_not_authorize_overwrite_or_replace() {
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableAccess::Insert(InsertOp::Append));
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
    for (sql, access) in [
        (
            "UPDATE target SET id = (SELECT max(id) FROM target)",
            TableAccess::Update,
        ),
        (
            "DELETE FROM target WHERE id IN (SELECT id FROM target)",
            TableAccess::Delete,
        ),
        (
            "MERGE INTO target AS t USING public.target AS s ON t.id = s.id WHEN MATCHED THEN DELETE",
            TableAccess::Merge {
                insert: false,
                update: false,
                delete: true,
            },
        ),
    ] {
        let (ctx, schema) = authorized_context();
        schema.grant("target", access);
        let err = ctx.sql(sql).await.unwrap_err();
        assert_contains!(err.to_string(), "Access denied to target for Read");

        schema.grant("target", TableAccess::Read);
        ctx.state().create_logical_plan(sql).await.unwrap();
        assert_eq!(
            schema.accesses(),
            HashSet::from([
                ("target".to_owned(), access),
                ("target".to_owned(), TableAccess::Read),
            ]),
            "{sql}"
        );
    }
}

#[tokio::test]
async fn legacy_lookup_remains_conservative_for_protected_catalogs() {
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableAccess::Insert(InsertOp::Append));

    let err = schema.table("target").await.unwrap_err();
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

    for access in [
        TableAccess::Read,
        TableAccess::Insert(InsertOp::Append),
        TableAccess::Insert(InsertOp::Overwrite),
        TableAccess::Insert(InsertOp::Replace),
        TableAccess::Update,
        TableAccess::Delete,
        TableAccess::Truncate,
        TableAccess::Merge {
            insert: false,
            update: false,
            delete: true,
        },
    ] {
        assert!(
            schema
                .table_with_access("target", access)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            schema
                .table_with_access("missing", access)
                .await
                .unwrap()
                .is_none()
        );
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

#[derive(Debug, Default)]
struct ObservedUrlFactory {
    calls: AtomicUsize,
}

#[async_trait]
impl UrlTableFactory for ObservedUrlFactory {
    async fn try_new(&self, _url: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Arc::new(ObservedTable::new())))
    }
}

#[tokio::test]
async fn dynamic_file_schema_forwards_access_and_only_falls_back_for_missing_tables() {
    let schema = Arc::new(AuthorizedSchema::new());
    let factory = Arc::new(ObservedUrlFactory::default());
    let wrapped = DynamicFileSchemaProvider::new(
        Arc::clone(&schema) as Arc<dyn SchemaProvider>,
        Arc::clone(&factory) as Arc<dyn UrlTableFactory>,
    );
    let access = TableAccess::Insert(InsertOp::Append);
    schema.grant("target", access);
    assert!(
        wrapped
            .table_with_access("target", access)
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(factory.calls.load(Ordering::Relaxed), 0);

    let err = wrapped
        .table_with_access("target", TableAccess::Read)
        .await
        .unwrap_err();
    assert_contains!(err.to_string(), "Access denied to target for Read");
    assert_eq!(factory.calls.load(Ordering::Relaxed), 0);

    schema.grant("file.csv", access);
    assert!(
        wrapped
            .table_with_access("file.csv", access)
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(factory.calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        schema.accesses(),
        HashSet::from([
            ("target".to_owned(), access),
            ("target".to_owned(), TableAccess::Read),
            ("file.csv".to_owned(), access),
        ])
    );
}

#[tokio::test]
async fn forwarding_wrapper_cannot_turn_read_permission_into_insert_permission() {
    let schema = Arc::new(AuthorizedSchema::new());
    schema.grant("target", TableAccess::Read);
    let factory = Arc::new(ObservedUrlFactory::default());
    let wrapped = Arc::new(DynamicFileSchemaProvider::new(
        Arc::clone(&schema) as Arc<dyn SchemaProvider>,
        Arc::clone(&factory) as Arc<dyn UrlTableFactory>,
    ));
    let ctx = context(wrapped);
    let error = ctx.sql("INSERT INTO target VALUES (1)").await.unwrap_err();
    assert_contains!(
        error.to_string(),
        "Access denied to target for Insert(Append)"
    );
    assert_eq!(schema.tables["target"].schema_calls(), 0);
    assert_eq!(factory.calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        schema.accesses(),
        HashSet::from([("target".into(), TableAccess::Insert(InsertOp::Append))])
    );
}

#[tokio::test]
async fn metadata_permission_allows_describe_but_not_select() {
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableAccess::Metadata);
    ctx.sql("DESCRIBE target")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        schema.accesses(),
        HashSet::from([("target".into(), TableAccess::Metadata)])
    );
    let calls = schema.tables["target"].schema_calls();
    let error = ctx.sql("SELECT * FROM target").await.unwrap_err();
    assert_contains!(error.to_string(), "Access denied to target for Read");
    assert_eq!(schema.tables["target"].schema_calls(), calls);
}

#[tokio::test]
async fn denied_metadata_fails_before_schema_and_does_not_authorize_ddl() {
    for sql in [
        "DESCRIBE target",
        "CREATE INDEX idx ON target(id)",
        "CREATE TABLE IF NOT EXISTS target(id BIGINT)",
        "CREATE OR REPLACE VIEW target AS SELECT 1 AS id",
        "DROP TABLE target",
        "DROP TABLE IF EXISTS target",
        "CREATE EXTERNAL TABLE IF NOT EXISTS target STORED AS CSV LOCATION 'missing.csv'",
    ] {
        let (ctx, schema) = authorized_context();
        schema.grant("target", TableAccess::Read);
        let error = ctx.sql(sql).await.unwrap_err();
        assert_contains!(error.to_string(), "Access denied to target for Metadata");
        assert_eq!(schema.tables["target"].schema_calls(), 0, "{sql}");
    }
    // Metadata lookup succeeds, but the separate catalog mutation remains denied.
    let (ctx, schema) = authorized_context();
    schema.grant("target", TableAccess::Metadata);
    let error = ctx.sql("DROP TABLE target").await.unwrap_err();
    assert_contains!(error.to_string(), "does not support deregistering tables");
}

#[tokio::test]
async fn merge_prefetch_and_planning_preserve_every_clause_action() {
    for mask in 1..8 {
        let access = TableAccess::Merge {
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
        schema.grant("source", TableAccess::Read);
        let error = ctx.state().create_logical_plan(&sql).await.unwrap_err();
        assert_contains!(
            error.to_string(),
            format!("Access denied to target for {access:?}")
        );
        assert_eq!(schema.tables["target"].schema_calls(), 0);
        schema.grant("target", access);
        ctx.state().create_logical_plan(&sql).await.unwrap();
        assert_eq!(
            schema.accesses(),
            HashSet::from([
                ("target".into(), access),
                ("source".into(), TableAccess::Read)
            ])
        );
    }
}

#[tokio::test]
async fn show_and_information_schema_use_metadata_access() {
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
        schema.grant(table, TableAccess::Metadata);
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
                .accesses()
                .iter()
                .all(|(_, access)| *access == TableAccess::Metadata),
            "{sql}"
        );
    }
    assert!(
        schema
            .accesses()
            .contains(&("target".into(), TableAccess::Metadata))
    );
}

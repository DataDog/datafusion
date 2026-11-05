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

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::{Result, TableReference, config::ConfigOptions, plan_err};
use datafusion_expr::{
    AggregateUDF, HigherOrderUDF, LogicalPlan, ScalarUDF, TableAccess, TableSource,
    WindowUDF, dml::InsertOp, planner::ContextProvider,
};
use datafusion_sql::{parser::DFParser, planner::SqlToRel};

#[derive(Default)]
struct AccessContext {
    options: ConfigOptions,
    allowed: HashSet<(String, TableAccess)>,
    tables: HashMap<String, Arc<TrackedSource>>,
    requests: RefCell<Vec<(String, TableAccess)>>,
}

impl AccessContext {
    fn with_access(allowed: &[(&str, TableAccess)]) -> Self {
        Self {
            allowed: allowed
                .iter()
                .map(|(name, access)| (name.to_string(), *access))
                .collect(),
            tables: ["target", "source"]
                .into_iter()
                .map(|name| (name.to_string(), Arc::new(TrackedSource::default())))
                .collect(),
            ..Default::default()
        }
    }

    fn plan(&self, sql: &str) -> Result<LogicalPlan> {
        let statement = DFParser::parse_sql(sql)?.pop_back().unwrap();
        SqlToRel::new(self).statement_to_plan(statement)
    }
}

#[derive(Default)]
struct TrackedSource {
    schema_calls: AtomicUsize,
}

impl TableSource for TrackedSource {
    fn schema(&self) -> SchemaRef {
        self.schema_calls.fetch_add(1, Ordering::SeqCst);
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
    }
}

impl ContextProvider for AccessContext {
    fn get_table_source(
        &self,
        name: TableReference,
        access: TableAccess,
    ) -> Result<Arc<dyn TableSource>> {
        let key = (name.to_string(), access);
        self.requests.borrow_mut().push(key.clone());
        if !self.allowed.contains(&key) {
            return plan_err!("access denied: {name} {access:?}");
        }
        Ok(Arc::clone(self.tables.get(name.table()).unwrap()) as Arc<dyn TableSource>)
    }

    fn get_function_meta(&self, _: &str) -> Option<Arc<ScalarUDF>> {
        None
    }
    fn get_higher_order_meta(&self, _: &str) -> Option<Arc<HigherOrderUDF>> {
        None
    }
    fn get_aggregate_meta(&self, _: &str) -> Option<Arc<AggregateUDF>> {
        None
    }
    fn get_window_meta(&self, _: &str) -> Option<Arc<WindowUDF>> {
        None
    }
    fn get_variable_type(&self, _: &[String]) -> Option<DataType> {
        None
    }
    fn options(&self) -> &ConfigOptions {
        &self.options
    }
    fn udf_names(&self) -> Vec<String> {
        vec![]
    }
    fn higher_order_function_names(&self) -> Vec<String> {
        vec![]
    }
    fn udaf_names(&self) -> Vec<String> {
        vec![]
    }
    fn udwf_names(&self) -> Vec<String> {
        vec![]
    }
}

#[test]
fn write_only_targets_use_their_operation() {
    let cases = [
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
        ("UPDATE target SET id = 1 WHERE id = 0", TableAccess::Update),
        ("DELETE FROM target WHERE id = 0", TableAccess::Delete),
        ("TRUNCATE TABLE target", TableAccess::Truncate),
    ];
    for (sql, access) in cases {
        let context = AccessContext::with_access(&[("target", access)]);
        context
            .plan(sql)
            .unwrap_or_else(|error| panic!("{sql}: {error}"));
        assert!(
            context
                .requests
                .borrow()
                .iter()
                .all(|(_, actual)| *actual == access)
        );
    }
}

#[test]
fn insert_source_denied_before_its_schema_is_used() {
    let context =
        AccessContext::with_access(&[("target", TableAccess::Insert(InsertOp::Append))]);
    let error = context
        .plan("INSERT INTO target SELECT * FROM source")
        .unwrap_err();
    assert!(
        error.to_string().contains("access denied: source Read"),
        "{error}"
    );
    assert_eq!(
        context.tables["source"].schema_calls.load(Ordering::SeqCst),
        0
    );
}

#[test]
fn self_insert_checks_read_and_insert() {
    let write = TableAccess::Insert(InsertOp::Append);
    let context = AccessContext::with_access(&[("target", write)]);
    assert!(
        context
            .plan("INSERT INTO target SELECT * FROM target")
            .is_err()
    );
    assert_eq!(
        context.requests.into_inner(),
        vec![
            ("target".to_string(), write),
            ("target".to_string(), TableAccess::Read),
        ]
    );
    let context =
        AccessContext::with_access(&[("target", write), ("target", TableAccess::Read)]);
    context
        .plan("INSERT INTO target SELECT * FROM target")
        .unwrap();
}

#[test]
fn update_self_subquery_needs_separate_read_access() {
    let sql = "UPDATE target SET id = (SELECT id FROM target)";
    let context = AccessContext::with_access(&[("target", TableAccess::Update)]);
    assert!(
        context
            .plan(sql)
            .unwrap_err()
            .to_string()
            .contains("access denied: target Read")
    );
    let context = AccessContext::with_access(&[
        ("target", TableAccess::Update),
        ("target", TableAccess::Read),
    ]);
    context.plan(sql).unwrap();
}

#[test]
fn merge_target_and_source_use_distinct_operations() {
    let context = AccessContext::with_access(&[
        (
            "target",
            TableAccess::Merge {
                insert: false,
                update: false,
                delete: true,
            },
        ),
        ("source", TableAccess::Read),
    ]);
    context.plan("MERGE INTO target USING source ON target.id = source.id WHEN MATCHED THEN DELETE").unwrap();
    assert_eq!(
        context.requests.into_inner(),
        vec![
            (
                "target".to_string(),
                TableAccess::Merge {
                    insert: false,
                    update: false,
                    delete: true
                }
            ),
            ("source".to_string(), TableAccess::Read),
        ]
    );
}

#[test]
fn cte_alias_does_not_lookup_catalog() {
    let context =
        AccessContext::with_access(&[("target", TableAccess::Insert(InsertOp::Append))]);
    context
        .plan("INSERT INTO target WITH target AS (SELECT 1 AS id) SELECT * FROM target")
        .unwrap();
    assert_eq!(context.requests.borrow().len(), 1);
    context.requests.borrow_mut().clear();
    context
        .plan("WITH source AS (SELECT 1 AS id) SELECT * FROM source")
        .unwrap();
    assert!(context.requests.borrow().is_empty());
}

#[test]
fn metadata_lookup_does_not_require_row_reads() {
    for sql in ["DESCRIBE target", "CREATE INDEX idx ON target(id)"] {
        let context = AccessContext::with_access(&[("target", TableAccess::Metadata)]);
        context.plan(sql).unwrap();
        assert_eq!(
            *context.requests.borrow(),
            vec![("target".into(), TableAccess::Metadata)]
        );
        let context = AccessContext::with_access(&[("target", TableAccess::Read)]);
        assert!(
            context
                .plan(sql)
                .unwrap_err()
                .to_string()
                .contains("access denied: target Metadata")
        );
        assert_eq!(
            context.tables["target"].schema_calls.load(Ordering::SeqCst),
            0
        );
    }
}

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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::ControlFlow;

use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{TableAccess, dml::InsertOp};

use crate::parser::{CopyToSource, CopyToStatement, Statement as DFStatement};
use crate::planner::object_name_to_table_reference;
use datafusion_common::TableReference;
use sqlparser::ast::*;

// following constants are used in `resolve_table_references`
// and should be same as `datafusion/catalog/src/information_schema.rs`
const INFORMATION_SCHEMA: &str = "information_schema";
const TABLES: &str = "tables";
const VIEWS: &str = "views";
const COLUMNS: &str = "columns";
const DF_SETTINGS: &str = "df_settings";
const SCHEMATA: &str = "schemata";
const ROUTINES: &str = "routines";
const PARAMETERS: &str = "parameters";

/// All information schema tables
const INFORMATION_SCHEMA_TABLES: &[&str] = &[
    TABLES,
    VIEWS,
    COLUMNS,
    DF_SETTINGS,
    SCHEMATA,
    ROUTINES,
    PARAMETERS,
];

// Collect table/CTE references as `TableReference`s and normalize them during traversal.
// This avoids a second normalization/conversion pass after visiting the AST.
struct RelationVisitor {
    relations: BTreeMap<TableReference, Vec<TableAccess>>,
    access_targets: HashMap<*const ObjectName, TableAccess>,
    all_ctes: BTreeSet<TableReference>,
    ctes_in_scope: Vec<TableReference>,
    enable_ident_normalization: bool,
}

impl RelationVisitor {
    fn new(enable_ident_normalization: bool) -> Self {
        Self {
            relations: BTreeMap::new(),
            access_targets: HashMap::new(),
            all_ctes: BTreeSet::new(),
            ctes_in_scope: vec![],
            enable_ident_normalization,
        }
    }

    /// Record the reference to `relation`, if it's not a CTE reference.
    fn insert_relation(&mut self, relation: &ObjectName) -> ControlFlow<DataFusionError> {
        let access = self
            .access_targets
            .get(&std::ptr::from_ref(relation))
            .copied()
            .unwrap_or(TableAccess::Read);
        match object_name_to_table_reference(
            relation.clone(),
            self.enable_ident_normalization,
        ) {
            Ok(relation) => {
                if access != TableAccess::Read || !self.ctes_in_scope.contains(&relation)
                {
                    let accesses = self.relations.entry(relation).or_default();
                    if !accesses.contains(&access) {
                        accesses.push(access);
                    }
                }
                ControlFlow::Continue(())
            }
            Err(e) => ControlFlow::Break(e),
        }
    }

    fn mark_target(&mut self, relation: &ObjectName, access: TableAccess) {
        // Match AST occurrences, not names: INSERT INTO t SELECT * FROM t needs
        // separate insert and read lookups. These pointers are never dereferenced.
        self.access_targets
            .insert(std::ptr::from_ref(relation), access);
    }

    fn mark_table(&mut self, table: &TableFactor, access: TableAccess) {
        if let TableFactor::Table { name, .. } = table {
            self.mark_target(name, access);
        }
    }
}

impl Visitor for RelationVisitor {
    type Break = DataFusionError;

    fn pre_visit_relation(&mut self, relation: &ObjectName) -> ControlFlow<Self::Break> {
        self.insert_relation(relation)
    }

    fn pre_visit_query(&mut self, q: &Query) -> ControlFlow<Self::Break> {
        if let SetExpr::Select(select) = q.body.as_ref()
            && let Some(into) = &select.into
        {
            self.mark_target(&into.name, TableAccess::CreateTable { or_replace: false });
            self.insert_relation(&into.name)?;
        }
        if let Some(with) = &q.with {
            for cte in &with.cte_tables {
                // The non-recursive CTE name is not in scope when evaluating the CTE itself, so this is valid:
                // `WITH t AS (SELECT * FROM t) SELECT * FROM t`
                // Where the first `t` refers to a predefined table. So we are careful here
                // to visit the CTE first, before putting it in scope.
                if !with.recursive {
                    // This is a bit hackish as the CTE will be visited again as part of visiting `q`,
                    // but thankfully `insert_relation` is idempotent.
                    cte.visit(self)?;
                }
                let cte_name = ObjectName::from(vec![cte.alias.name.clone()]);
                match object_name_to_table_reference(
                    cte_name,
                    self.enable_ident_normalization,
                ) {
                    Ok(cte_ref) => self.ctes_in_scope.push(cte_ref),
                    Err(e) => return ControlFlow::Break(e),
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, q: &Query) -> ControlFlow<Self::Break> {
        if let Some(with) = &q.with {
            for _ in &with.cte_tables {
                // Unwrap: We just pushed these in `pre_visit_query`
                self.all_ctes.insert(self.ctes_in_scope.pop().unwrap());
            }
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_statement(&mut self, statement: &Statement) -> ControlFlow<Self::Break> {
        match statement {
            Statement::Insert(insert) => {
                if let TableObject::TableName(name) = &insert.table {
                    let replace_into = insert.replace_into
                        || insert.or == Some(SqliteOnConflict::Replace);
                    let operation = match (insert.overwrite, replace_into) {
                        (false, false) => InsertOp::Append,
                        (true, false) => InsertOp::Overwrite,
                        (false, true) => InsertOp::Replace,
                        (true, true) => {
                            return ControlFlow::Break(
                                datafusion_common::plan_datafusion_err!(
                                    "Conflicting insert operations: `overwrite` and `replace_into` cannot both be true"
                                ),
                            );
                        }
                    };
                    self.mark_target(name, TableAccess::Insert(operation));
                }
            }
            Statement::Update(update) => {
                self.mark_table(&update.table.relation, TableAccess::Update);
            }
            Statement::Delete(delete) => {
                let (FromTable::WithFromKeyword(from) | FromTable::WithoutKeyword(from)) =
                    &delete.from;
                for table in from {
                    self.mark_table(&table.relation, TableAccess::Delete);
                }
            }
            Statement::Merge(merge) => {
                self.mark_table(&merge.table, merge_table_access(&merge.clauses));
            }
            Statement::Truncate(truncate) => {
                for table in &truncate.table_names {
                    self.mark_target(&table.name, TableAccess::Truncate);
                }
            }
            Statement::ExplainTable { table_name, .. } => {
                self.mark_target(table_name, TableAccess::Metadata);
                self.insert_relation(table_name)?;
            }
            Statement::ShowCreate { obj_name, .. } => {
                self.mark_target(obj_name, TableAccess::Metadata);
            }
            Statement::ShowColumns { show_options, .. } => {
                if let Some(name) = show_options
                    .show_in
                    .as_ref()
                    .and_then(|show_in| show_in.parent_name.as_ref())
                {
                    self.mark_target(name, TableAccess::Metadata);
                    self.insert_relation(name)?;
                }
            }
            Statement::CreateIndex(index) => {
                self.mark_target(&index.table_name, TableAccess::CreateIndex);
                self.insert_relation(&index.table_name)?;
            }
            Statement::CreateTable(table) => {
                self.mark_target(
                    &table.name,
                    TableAccess::CreateTable {
                        or_replace: table.or_replace,
                    },
                );
                self.insert_relation(&table.name)?;
            }
            Statement::CreateView(view) => {
                self.mark_target(
                    &view.name,
                    TableAccess::CreateView {
                        or_replace: view.or_replace,
                    },
                );
                self.insert_relation(&view.name)?;
            }
            Statement::Drop {
                object_type: object_type @ (ObjectType::Table | ObjectType::View),
                names,
                ..
            } => {
                for name in names {
                    self.mark_target(
                        name,
                        if *object_type == ObjectType::Table {
                            TableAccess::DropTable
                        } else {
                            TableAccess::DropView
                        },
                    );
                    self.insert_relation(name)?;
                }
            }
            _ => {}
        }

        if let Statement::ShowCreate {
            obj_type: ShowCreateObject::Table | ShowCreateObject::View,
            obj_name,
        } = statement
        {
            self.insert_relation(obj_name)?;
        }

        // SHOW statements will later be rewritten into a SELECT from the information_schema
        let requires_information_schema = matches!(
            statement,
            Statement::ShowFunctions { .. }
                | Statement::ShowVariable { .. }
                | Statement::ShowStatus { .. }
                | Statement::ShowVariables { .. }
                | Statement::ShowCreate { .. }
                | Statement::ShowColumns { .. }
                | Statement::ShowTables { .. }
                | Statement::ShowCollation { .. }
        );
        if requires_information_schema {
            for s in INFORMATION_SCHEMA_TABLES {
                // Information schema references are synthesized here, so convert directly.
                let obj = ObjectName::from(vec![
                    Ident::new(INFORMATION_SCHEMA),
                    Ident::new(*s),
                ]);
                match object_name_to_table_reference(obj, self.enable_ident_normalization)
                {
                    Ok(tbl_ref) => {
                        self.relations
                            .entry(tbl_ref)
                            .or_insert_with(|| vec![TableAccess::Read]);
                    }
                    Err(e) => return ControlFlow::Break(e),
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// Determine all possible target mutations before resolving the target schema.
pub(crate) fn merge_table_access(clauses: &[MergeClause]) -> TableAccess {
    let (mut insert, mut update, mut delete) = (false, false, false);
    for clause in clauses {
        match clause.action {
            MergeAction::Insert(_) => insert = true,
            MergeAction::Update(_) => update = true,
            MergeAction::Delete { .. } => delete = true,
        }
    }
    TableAccess::Merge {
        insert,
        update,
        delete,
    }
}

fn control_flow_to_result(flow: ControlFlow<DataFusionError>) -> Result<()> {
    match flow {
        ControlFlow::Continue(()) => Ok(()),
        ControlFlow::Break(err) => Err(err),
    }
}

fn visit_statement(statement: &DFStatement, visitor: &mut RelationVisitor) -> Result<()> {
    match statement {
        DFStatement::Statement(s) => {
            control_flow_to_result(s.as_ref().visit(visitor))?;
        }
        DFStatement::CreateExternalTable(table) => {
            visitor.mark_target(
                &table.name,
                TableAccess::CreateTable {
                    or_replace: table.or_replace,
                },
            );
            control_flow_to_result(visitor.insert_relation(&table.name))?;
        }
        DFStatement::CopyTo(CopyToStatement { source, .. }) => match source {
            CopyToSource::Relation(table_name) => {
                control_flow_to_result(visitor.insert_relation(table_name))?;
            }
            CopyToSource::Query(query) => {
                control_flow_to_result(query.visit(visitor))?;
            }
        },
        DFStatement::Explain(explain) => {
            visit_statement(&explain.statement, visitor)?;
        }
        DFStatement::Reset(_) => {}
    }
    Ok(())
}

/// Collects all tables and views referenced in the SQL statement. CTEs are collected separately.
/// This can be used to determine which tables need to be in the catalog for a query to be planned.
///
/// # Returns
///
/// A `(table_refs, ctes)` tuple, the first element contains table and view references and the second
/// element contains any CTE aliases that were defined and possibly referenced.
///
/// ## Example
///
/// ```
/// # use datafusion_sql::parser::DFParser;
/// # use datafusion_sql::resolve::resolve_table_references;
/// let query = "SELECT a FROM foo where x IN (SELECT y FROM bar)";
/// let statement = DFParser::parse_sql(query).unwrap().pop_back().unwrap();
/// let (table_refs, ctes) = resolve_table_references(&statement, true).unwrap();
/// assert_eq!(table_refs.len(), 2);
/// assert_eq!(table_refs[0].to_string(), "bar");
/// assert_eq!(table_refs[1].to_string(), "foo");
/// assert_eq!(ctes.len(), 0);
/// ```
///
/// ## Example with CTEs  
///  
/// ```  
/// # use datafusion_sql::parser::DFParser;
/// # use datafusion_sql::resolve::resolve_table_references;
/// let query = "with my_cte as (values (1), (2)) SELECT * from my_cte;";
/// let statement = DFParser::parse_sql(query).unwrap().pop_back().unwrap();
/// let (table_refs, ctes) = resolve_table_references(&statement, true).unwrap();
/// assert_eq!(table_refs.len(), 0);
/// assert_eq!(ctes.len(), 1);
/// assert_eq!(ctes[0].to_string(), "my_cte");
/// ```
pub fn resolve_table_references(
    statement: &crate::parser::Statement,
    enable_ident_normalization: bool,
) -> Result<(Vec<TableReference>, Vec<TableReference>)> {
    let mut visitor = RelationVisitor::new(enable_ident_normalization);

    visit_statement(statement, &mut visitor)?;

    Ok((
        visitor.relations.into_keys().collect(),
        visitor.all_ctes.into_iter().collect(),
    ))
}

/// Table references paired with the operations required to resolve them.
pub type TableReferencesWithAccess = Vec<(TableReference, TableAccess)>;

/// Collects table references together with the operation performed on each occurrence.
///
/// Write targets use their specific operation, while query sources use [`TableAccess::Read`].
/// A table used as both a target and a source appears once for each distinct operation.
/// CTE aliases are excluded from table references and returned separately. Metadata and
/// DDL target lookups carry their own operation; they do not imply a separate read grant.
///
/// Prefetch caches must retain the access alongside the table reference so that a provider
/// authorized for one operation cannot satisfy a lookup for another operation.
pub fn resolve_table_references_with_access(
    statement: &crate::parser::Statement,
    enable_ident_normalization: bool,
) -> Result<(TableReferencesWithAccess, Vec<TableReference>)> {
    let mut visitor = RelationVisitor::new(enable_ident_normalization);
    visit_statement(statement, &mut visitor)?;
    Ok((
        visitor
            .relations
            .into_iter()
            .flat_map(|(table, accesses)| {
                accesses
                    .into_iter()
                    .map(move |access| (table.clone(), access))
            })
            .collect(),
        visitor.all_ctes.into_iter().collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_aware_table_references() {
        use crate::parser::DFParser;
        use TableAccess::{Delete, Insert, Merge, Read, Truncate, Update};

        let cases = [
            (
                "INSERT INTO dst VALUES (1)",
                vec![("dst", Insert(InsertOp::Append))],
            ),
            (
                "INSERT OVERWRITE dst SELECT * FROM src",
                vec![("dst", Insert(InsertOp::Overwrite)), ("src", Read)],
            ),
            (
                "REPLACE INTO dst SELECT * FROM src",
                vec![("dst", Insert(InsertOp::Replace)), ("src", Read)],
            ),
            (
                "INSERT OR REPLACE INTO dst VALUES (1)",
                vec![("dst", Insert(InsertOp::Replace))],
            ),
            (
                "INSERT INTO t SELECT * FROM t",
                vec![("t", Insert(InsertOp::Append)), ("t", Read)],
            ),
            (
                "INSERT INTO t WITH t AS (SELECT 1) SELECT * FROM t",
                vec![("t", Insert(InsertOp::Append))],
            ),
            (
                "INSERT INTO dst WITH t AS (SELECT * FROM src) SELECT * FROM t",
                vec![("dst", Insert(InsertOp::Append)), ("src", Read)],
            ),
            ("UPDATE t SET id = 1", vec![("t", Update)]),
            (
                "UPDATE t SET id = (SELECT id FROM t)",
                vec![("t", Update), ("t", Read)],
            ),
            (
                "DELETE FROM t WHERE id IN (SELECT id FROM t)",
                vec![("t", Delete), ("t", Read)],
            ),
            ("TRUNCATE TABLE t", vec![("t", Truncate)]),
            (
                "MERGE INTO t USING t AS s ON t.id = s.id WHEN MATCHED THEN DELETE",
                vec![
                    (
                        "t",
                        Merge {
                            insert: false,
                            update: false,
                            delete: true,
                        },
                    ),
                    ("t", Read),
                ],
            ),
            (
                "EXPLAIN INSERT INTO dst SELECT * FROM src",
                vec![("dst", Insert(InsertOp::Append)), ("src", Read)],
            ),
            ("COPY t TO 'out.csv'", vec![("t", Read)]),
            (
                "SELECT * FROM t a JOIN t b ON a.id = b.id",
                vec![("t", Read)],
            ),
        ];
        for (sql, expected) in cases {
            let statement = DFParser::parse_sql(sql).unwrap().pop_back().unwrap();
            let (tables, _) =
                resolve_table_references_with_access(&statement, true).unwrap();
            let tables: Vec<_> = tables
                .into_iter()
                .map(|(name, access)| (name.to_string(), access))
                .collect();
            let expected: Vec<_> = expected
                .iter()
                .map(|(name, access)| (name.to_string(), *access))
                .collect();
            assert_eq!(tables, expected, "{sql}");
        }
    }

    #[test]
    fn metadata_and_ddl_targets_are_distinct_from_query_sources() {
        use crate::parser::DFParser;
        for (sql, access) in [
            ("DESCRIBE target", TableAccess::Metadata),
            ("SHOW COLUMNS FROM target", TableAccess::Metadata),
            ("SHOW CREATE TABLE target", TableAccess::Metadata),
            ("CREATE INDEX idx ON target(id)", TableAccess::CreateIndex),
            (
                "CREATE TABLE target AS SELECT * FROM source",
                TableAccess::CreateTable { or_replace: false },
            ),
            (
                "CREATE VIEW target AS SELECT * FROM source",
                TableAccess::CreateView { or_replace: false },
            ),
            (
                "SELECT * INTO target FROM source",
                TableAccess::CreateTable { or_replace: false },
            ),
            ("DROP TABLE target", TableAccess::DropTable),
        ] {
            let statement = DFParser::parse_sql(sql).unwrap().pop_back().unwrap();
            let (tables, _) =
                resolve_table_references_with_access(&statement, true).unwrap();
            assert!(
                tables.contains(&(TableReference::bare("target"), access)),
                "{sql}: {tables:?}"
            );
            assert!(
                !tables.contains(&(TableReference::bare("target"), TableAccess::Read)),
                "{sql}"
            );
            if sql.contains("FROM source") {
                assert!(
                    tables.contains(&(TableReference::bare("source"), TableAccess::Read)),
                    "{sql}"
                );
            }
        }
    }

    #[test]
    fn resolve_table_references_shadowed_cte() {
        use crate::parser::DFParser;

        // An interesting edge case where the `t` name is used both as an ordinary table reference
        // and as a CTE reference.
        let query = "WITH t AS (SELECT * FROM t) SELECT * FROM t";
        let statement = DFParser::parse_sql(query).unwrap().pop_back().unwrap();
        let (table_refs, ctes) = resolve_table_references(&statement, true).unwrap();
        assert_eq!(table_refs.len(), 1);
        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].to_string(), "t");
        assert_eq!(table_refs[0].to_string(), "t");

        // UNION is a special case where the CTE is not in scope for the second branch.
        let query = "(with t as (select 1) select * from t) union (select * from t)";
        let statement = DFParser::parse_sql(query).unwrap().pop_back().unwrap();
        let (table_refs, ctes) = resolve_table_references(&statement, true).unwrap();
        assert_eq!(table_refs.len(), 1);
        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].to_string(), "t");
        assert_eq!(table_refs[0].to_string(), "t");

        // Nested CTEs are also handled.
        // Here the first `u` is a CTE, but the second `u` is a table reference.
        // While `t` is always a CTE.
        let query = "(with t as (with u as (select 1) select * from u) select * from u cross join t)";
        let statement = DFParser::parse_sql(query).unwrap().pop_back().unwrap();
        let (table_refs, ctes) = resolve_table_references(&statement, true).unwrap();
        assert_eq!(table_refs.len(), 1);
        assert_eq!(ctes.len(), 2);
        assert_eq!(ctes[0].to_string(), "t");
        assert_eq!(ctes[1].to_string(), "u");
        assert_eq!(table_refs[0].to_string(), "u");
    }

    #[test]
    fn resolve_table_references_recursive_cte() {
        use crate::parser::DFParser;

        let query = "
            WITH RECURSIVE nodes AS ( 
                SELECT 1 as id
                UNION ALL 
                SELECT id + 1 as id 
                FROM nodes
                WHERE id < 10
            )
            SELECT * FROM nodes
        ";
        let statement = DFParser::parse_sql(query).unwrap().pop_back().unwrap();
        let (table_refs, ctes) = resolve_table_references(&statement, true).unwrap();
        assert_eq!(table_refs.len(), 0);
        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].to_string(), "nodes");
    }

    #[test]
    fn resolve_table_references_cte_with_quoted_reference() {
        use crate::parser::DFParser;

        let query = r#"with barbaz as (select 1) select * from "barbaz""#;
        let statement = DFParser::parse_sql(query).unwrap().pop_back().unwrap();
        let (table_refs, ctes) = resolve_table_references(&statement, true).unwrap();
        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].to_string(), "barbaz");
        // Quoted reference should still resolve to the CTE when normalization is on
        assert_eq!(table_refs.len(), 0);
    }

    #[test]
    fn resolve_table_references_cte_with_quoted_reference_normalization_off() {
        use crate::parser::DFParser;

        let query = r#"with barbaz as (select 1) select * from "barbaz""#;
        let statement = DFParser::parse_sql(query).unwrap().pop_back().unwrap();
        let (table_refs, ctes) = resolve_table_references(&statement, false).unwrap();
        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].to_string(), "barbaz");
        // Even with normalization off, quoted reference matches same-case CTE name
        assert_eq!(table_refs.len(), 0);
    }

    #[test]
    fn resolve_table_references_cte_with_quoted_reference_uppercase_normalization_on() {
        use crate::parser::DFParser;

        let query = r#"with FOObar as (select 1) select * from "FOObar""#;
        let statement = DFParser::parse_sql(query).unwrap().pop_back().unwrap();
        let (table_refs, ctes) = resolve_table_references(&statement, true).unwrap();
        // CTE name is normalized to lowercase, quoted reference preserves case, so they differ
        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].to_string(), "foobar");
        assert_eq!(table_refs.len(), 1);
        assert_eq!(table_refs[0].to_string(), "FOObar");
    }

    #[test]
    fn resolve_table_references_cte_with_quoted_reference_uppercase_normalization_off() {
        use crate::parser::DFParser;

        let query = r#"with FOObar as (select 1) select * from "FOObar""#;
        let statement = DFParser::parse_sql(query).unwrap().pop_back().unwrap();
        let (table_refs, ctes) = resolve_table_references(&statement, false).unwrap();
        // Without normalization, cases match exactly, so quoted reference resolves to the CTE
        assert_eq!(ctes.len(), 1);
        assert_eq!(ctes[0].to_string(), "FOObar");
        assert_eq!(table_refs.len(), 0);
    }
}

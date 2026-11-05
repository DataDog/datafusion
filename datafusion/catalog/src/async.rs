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

use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::{HashMap, TableReference, error::Result, not_impl_err, plan_err};
use datafusion_execution::config::SessionConfig;
use datafusion_expr::TableAccess;

use crate::{CatalogProvider, CatalogProviderList, SchemaProvider, TableProvider};

/// A schema provider that looks up tables in a cache
///
/// Instances are created by the [`AsyncSchemaProvider::resolve`] method
#[derive(Debug)]
struct ResolvedSchemaProvider {
    owner_name: Option<String>,
    cached_tables: ResolvedTableCache,
}

#[derive(Debug)]
enum ResolvedTableCache {
    Legacy(HashMap<String, Arc<dyn TableProvider>>),
    WithAccess(HashMap<(String, TableAccess), Arc<dyn TableProvider>>),
}

impl ResolvedTableCache {
    fn new(
        tables: HashMap<(String, TableAccess), Option<Arc<dyn TableProvider>>>,
        access_aware: bool,
    ) -> Self {
        let tables = tables
            .into_iter()
            .filter_map(|(key, table)| table.map(|table| (key, table)));
        if access_aware {
            Self::WithAccess(tables.collect())
        } else {
            Self::Legacy(tables.map(|((name, _), table)| (name, table)).collect())
        }
    }
}
#[async_trait]
impl SchemaProvider for ResolvedSchemaProvider {
    fn owner_name(&self) -> Option<&str> {
        self.owner_name.as_deref()
    }

    fn table_names(&self) -> Vec<String> {
        match &self.cached_tables {
            ResolvedTableCache::Legacy(tables) => tables.keys().cloned().collect(),
            ResolvedTableCache::WithAccess(tables) => tables
                .keys()
                .map(|(name, _)| name.clone())
                .collect::<datafusion_common::HashSet<_>>()
                .into_iter()
                .collect(),
        }
    }

    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        self.table_with_access(name, TableAccess::Read).await
    }

    async fn table_with_access(
        &self,
        name: &str,
        access: TableAccess,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        Ok(match &self.cached_tables {
            ResolvedTableCache::Legacy(tables) => tables.get(name).cloned(),
            ResolvedTableCache::WithAccess(tables) => {
                if let Some(table) = tables.get(&(name.to_string(), access)) {
                    Some(Arc::clone(table))
                } else if tables.keys().any(|(table, _)| table == name) {
                    // An unresolved operation must not trigger missing-table fallbacks.
                    return plan_err!(
                        "Table '{name}' was not resolved for operation {access:?}"
                    );
                } else {
                    None
                }
            }
        })
    }

    fn register_table(
        &self,
        name: String,
        _table: Arc<dyn TableProvider>,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        not_impl_err!(
            "Attempt to register table '{name}' with ResolvedSchemaProvider which is not supported"
        )
    }

    fn deregister_table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        not_impl_err!(
            "Attempt to deregister table '{name}' with ResolvedSchemaProvider which is not supported"
        )
    }

    fn table_exist(&self, name: &str) -> bool {
        match &self.cached_tables {
            ResolvedTableCache::Legacy(tables) => tables.contains_key(name),
            ResolvedTableCache::WithAccess(tables) => {
                tables.keys().any(|(table, _)| table == name)
            }
        }
    }
}

/// Helper class for building a [`ResolvedSchemaProvider`]
struct ResolvedSchemaProviderBuilder {
    owner_name: String,
    async_provider: Arc<dyn AsyncSchemaProvider>,
    cached_tables: HashMap<(String, TableAccess), Option<Arc<dyn TableProvider>>>,
    access_aware: bool,
}
impl ResolvedSchemaProviderBuilder {
    fn new(
        owner_name: String,
        async_provider: Arc<dyn AsyncSchemaProvider>,
        access_aware: bool,
    ) -> Self {
        Self {
            owner_name,
            async_provider,
            cached_tables: HashMap::new(),
            access_aware,
        }
    }

    async fn resolve_table(
        &mut self,
        table_name: &str,
        access: TableAccess,
    ) -> Result<()> {
        let key = (table_name.to_string(), access);
        if !self.cached_tables.contains_key(&key) {
            let resolved_table = if self.access_aware {
                self.async_provider
                    .table_with_access(table_name, access)
                    .await?
            } else {
                self.async_provider.table(table_name).await?
            };
            self.cached_tables.insert(key, resolved_table);
        }
        Ok(())
    }

    fn finish(self) -> Arc<dyn SchemaProvider> {
        let cached_tables =
            ResolvedTableCache::new(self.cached_tables, self.access_aware);
        Arc::new(ResolvedSchemaProvider {
            owner_name: Some(self.owner_name),
            cached_tables,
        })
    }
}

/// A catalog provider that looks up schemas in a cache
///
/// Instances are created by the [`AsyncCatalogProvider::resolve`] method
#[derive(Debug)]
struct ResolvedCatalogProvider {
    cached_schemas: HashMap<String, Arc<dyn SchemaProvider>>,
}
impl CatalogProvider for ResolvedCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.cached_schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.cached_schemas.get(name).cloned()
    }
}

/// Helper class for building a [`ResolvedCatalogProvider`]
struct ResolvedCatalogProviderBuilder {
    cached_schemas: HashMap<String, Option<ResolvedSchemaProviderBuilder>>,
    async_provider: Arc<dyn AsyncCatalogProvider>,
}
impl ResolvedCatalogProviderBuilder {
    fn new(async_provider: Arc<dyn AsyncCatalogProvider>) -> Self {
        Self {
            cached_schemas: HashMap::new(),
            async_provider,
        }
    }
    fn finish(self) -> Arc<dyn CatalogProvider> {
        let cached_schemas = self
            .cached_schemas
            .into_iter()
            .filter_map(|(key, maybe_value)| {
                maybe_value.map(|value| (key, value.finish()))
            })
            .collect();
        Arc::new(ResolvedCatalogProvider { cached_schemas })
    }
}

/// A catalog provider list that looks up catalogs in a cache
///
/// Instances are created by the [`AsyncCatalogProviderList::resolve`] method
#[derive(Debug)]
struct ResolvedCatalogProviderList {
    cached_catalogs: HashMap<String, Arc<dyn CatalogProvider>>,
}
impl CatalogProviderList for ResolvedCatalogProviderList {
    fn register_catalog(
        &self,
        _name: String,
        _catalog: Arc<dyn CatalogProvider>,
    ) -> Option<Arc<dyn CatalogProvider>> {
        unimplemented!("resolved providers cannot handle registration APIs")
    }

    fn catalog_names(&self) -> Vec<String> {
        self.cached_catalogs.keys().cloned().collect()
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn CatalogProvider>> {
        self.cached_catalogs.get(name).cloned()
    }
}

/// A trait for schema providers that must resolve tables asynchronously
///
/// The [`SchemaProvider::table`] method _is_ asynchronous.  However, this is primarily for convenience and
/// it is not a good idea for this method to be slow as this will cause poor planning performance.
///
/// It is a better idea to resolve the tables once and cache them in memory for the duration of
/// planning.  This trait helps implement that pattern.
///
/// After implementing this trait you can call the [`AsyncSchemaProvider::resolve`] method to get an
/// `Arc<dyn SchemaProvider>` that contains a cached copy of the referenced tables.  The `resolve`
/// method can be slow and asynchronous as it is only called once, before planning.
///
/// See the [remote_catalog.rs] for an end to end example
///
/// [remote_catalog.rs]: https://github.com/apache/datafusion/blob/main/datafusion-examples/examples/data_io/remote_catalog.rs
#[async_trait]
pub trait AsyncSchemaProvider: Send + Sync {
    /// Lookup a table in the schema provider
    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>>;

    /// Resolve a table for an operation before its schema is used.
    ///
    /// The default delegates to [`Self::table`]. Authorization-aware providers
    /// should override this method and make legacy lookups conservative.
    async fn table_with_access(
        &self,
        name: &str,
        _access: TableAccess,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        self.table(name).await
    }

    /// Creates a cached provider that can be used to execute a query containing given references
    ///
    /// This method will walk through the references and look them up once, creating a cache of table
    /// providers.  This cache will be returned as a synchronous TableProvider that can be used to plan
    /// and execute a query containing the given references.
    ///
    /// This legacy cache does not distinguish operations. Use
    /// [`Self::resolve_with_access`] for operation-aware authorization.
    ///
    /// This cache is intended to be short-lived for the execution of a single query.  There is no mechanism
    /// for refresh or eviction of stale entries.
    ///
    /// See the [`AsyncSchemaProvider`] documentation for additional details
    async fn resolve(
        &self,
        references: &[TableReference],
        config: &SessionConfig,
        catalog_name: &str,
        schema_name: &str,
    ) -> Result<Arc<dyn SchemaProvider>> {
        resolve_schema(
            self,
            TableReferences::Legacy(references),
            config,
            catalog_name,
            schema_name,
        )
        .await
    }

    /// Resolve tables for the requested operations, retaining each operation in the cache.
    ///
    /// A lookup for a cached table with an unresolved operation returns an error.
    /// Legacy table lookup on this cache requests [`TableAccess::Read`].
    async fn resolve_with_access(
        &self,
        references: &[(TableReference, TableAccess)],
        config: &SessionConfig,
        catalog_name: &str,
        schema_name: &str,
    ) -> Result<Arc<dyn SchemaProvider>> {
        resolve_schema(
            self,
            TableReferences::WithAccess(references),
            config,
            catalog_name,
            schema_name,
        )
        .await
    }
}

/// A trait for catalog providers that must resolve schemas asynchronously
///
/// The [`CatalogProvider::schema`] method is synchronous because asynchronous operations should
/// not be used during planning.  This trait makes it easy to lookup schema references once and cache
/// them for future planning use.  See [`AsyncSchemaProvider`] for more details on motivation.

#[async_trait]
pub trait AsyncCatalogProvider: Send + Sync {
    /// Lookup a schema in the provider
    async fn schema(&self, name: &str) -> Result<Option<Arc<dyn AsyncSchemaProvider>>>;

    /// Creates a cached provider that can be used to execute a query containing given references
    ///
    /// This method will walk through the references and look them up once, creating a cache of schema
    /// providers (each with their own cache of table providers).  This cache will be returned as a
    /// synchronous CatalogProvider that can be used to plan and execute a query containing the given
    /// references.
    ///
    /// This legacy cache does not distinguish operations. Use
    /// [`Self::resolve_with_access`] for operation-aware authorization.
    ///
    /// This cache is intended to be short-lived for the execution of a single query.  There is no mechanism
    /// for refresh or eviction of stale entries.
    async fn resolve(
        &self,
        references: &[TableReference],
        config: &SessionConfig,
        catalog_name: &str,
    ) -> Result<Arc<dyn CatalogProvider>> {
        resolve_catalog(
            self,
            TableReferences::Legacy(references),
            config,
            catalog_name,
        )
        .await
    }

    /// Resolve tables for the requested operations, retaining each operation in the cache.
    ///
    /// A lookup for a cached table with an unresolved operation returns an error.
    /// Legacy table lookup on this cache requests [`TableAccess::Read`].
    async fn resolve_with_access(
        &self,
        references: &[(TableReference, TableAccess)],
        config: &SessionConfig,
        catalog_name: &str,
    ) -> Result<Arc<dyn CatalogProvider>> {
        resolve_catalog(
            self,
            TableReferences::WithAccess(references),
            config,
            catalog_name,
        )
        .await
    }
}

/// A trait for catalog provider lists that must resolve catalogs asynchronously
///
/// The [`CatalogProviderList::catalog`] method is synchronous because asynchronous operations should
/// not be used during planning.  This trait makes it easy to lookup catalog references once and cache
/// them for future planning use.  See [`AsyncSchemaProvider`] for more details on motivation.
#[async_trait]
pub trait AsyncCatalogProviderList: Send + Sync {
    /// Lookup a catalog in the provider
    async fn catalog(&self, name: &str) -> Result<Option<Arc<dyn AsyncCatalogProvider>>>;

    /// Creates a cached provider that can be used to execute a query containing given references
    ///
    /// This method will walk through the references and look them up once, creating a cache of catalog
    /// providers, schema providers, and table providers.  This cache will be returned as a
    /// synchronous CatalogProvider that can be used to plan and execute a query containing the given
    /// references.
    ///
    /// This legacy cache does not distinguish operations. Use
    /// [`Self::resolve_with_access`] for operation-aware authorization.
    ///
    /// This cache is intended to be short-lived for the execution of a single query.  There is no mechanism
    /// for refresh or eviction of stale entries.
    async fn resolve(
        &self,
        references: &[TableReference],
        config: &SessionConfig,
    ) -> Result<Arc<dyn CatalogProviderList>> {
        resolve_catalog_list(self, TableReferences::Legacy(references), config).await
    }

    /// Resolve tables for the requested operations, retaining each operation in the cache.
    ///
    /// A lookup for a cached table with an unresolved operation returns an error.
    /// Legacy table lookup on this cache requests [`TableAccess::Read`].
    async fn resolve_with_access(
        &self,
        references: &[(TableReference, TableAccess)],
        config: &SessionConfig,
    ) -> Result<Arc<dyn CatalogProviderList>> {
        resolve_catalog_list(self, TableReferences::WithAccess(references), config).await
    }
}

enum TableReferences<'a> {
    Legacy(&'a [TableReference]),
    WithAccess(&'a [(TableReference, TableAccess)]),
}

impl TableReferences<'_> {
    fn access_aware(&self) -> bool {
        matches!(self, Self::WithAccess(_))
    }

    fn iter(&self) -> impl Iterator<Item = (&TableReference, TableAccess)> + Send {
        let (legacy, with_access) = match self {
            Self::Legacy(references) => (*references, &[][..]),
            Self::WithAccess(references) => (&[][..], *references),
        };
        legacy
            .iter()
            .map(|reference| (reference, TableAccess::Read))
            .chain(
                with_access
                    .iter()
                    .map(|(reference, access)| (reference, *access)),
            )
    }
}

async fn resolve_schema<P: AsyncSchemaProvider + ?Sized>(
    provider: &P,
    references: TableReferences<'_>,
    config: &SessionConfig,
    catalog_name: &str,
    schema_name: &str,
) -> Result<Arc<dyn SchemaProvider>> {
    let mut cached_tables =
        HashMap::<(String, TableAccess), Option<Arc<dyn TableProvider>>>::new();

    for (reference, access) in references.iter() {
        let ref_catalog_name = reference
            .catalog()
            .unwrap_or(&config.options().catalog.default_catalog);

        // Maybe this is a reference to some other catalog provided in another way
        if ref_catalog_name != catalog_name {
            continue;
        }

        let ref_schema_name = reference
            .schema()
            .unwrap_or(&config.options().catalog.default_schema);

        if ref_schema_name != schema_name {
            continue;
        }

        let key = (reference.table().to_string(), access);
        if !cached_tables.contains_key(&key) {
            let resolved_table = if references.access_aware() {
                provider
                    .table_with_access(reference.table(), access)
                    .await?
            } else {
                provider.table(reference.table()).await?
            };
            cached_tables.insert(key, resolved_table);
        }
    }

    let cached_tables = ResolvedTableCache::new(cached_tables, references.access_aware());

    Ok(Arc::new(ResolvedSchemaProvider {
        cached_tables,
        owner_name: Some(catalog_name.to_string()),
    }))
}

async fn resolve_catalog<P: AsyncCatalogProvider + ?Sized>(
    provider: &P,
    references: TableReferences<'_>,
    config: &SessionConfig,
    catalog_name: &str,
) -> Result<Arc<dyn CatalogProvider>> {
    let mut cached_schemas =
        HashMap::<String, Option<ResolvedSchemaProviderBuilder>>::new();

    for (reference, access) in references.iter() {
        let ref_catalog_name = reference
            .catalog()
            .unwrap_or(&config.options().catalog.default_catalog);

        // Maybe this is a reference to some other catalog provided in another way
        if ref_catalog_name != catalog_name {
            continue;
        }

        let schema_name = reference
            .schema()
            .unwrap_or(&config.options().catalog.default_schema);

        let schema = if let Some(schema) = cached_schemas.get_mut(schema_name) {
            schema
        } else {
            let resolved_schema = provider.schema(schema_name).await?;
            let resolved_schema = resolved_schema.map(|resolved_schema| {
                ResolvedSchemaProviderBuilder::new(
                    catalog_name.to_string(),
                    resolved_schema,
                    references.access_aware(),
                )
            });
            cached_schemas.insert(schema_name.to_string(), resolved_schema);
            cached_schemas.get_mut(schema_name).unwrap()
        };

        // If we can't find the catalog don't bother checking the table
        let Some(schema) = schema else { continue };

        schema.resolve_table(reference.table(), access).await?;
    }

    let cached_schemas = cached_schemas
        .into_iter()
        .filter_map(|(key, maybe_builder)| {
            maybe_builder.map(|schema_builder| (key, schema_builder.finish()))
        })
        .collect::<HashMap<_, _>>();

    Ok(Arc::new(ResolvedCatalogProvider { cached_schemas }))
}

async fn resolve_catalog_list<P: AsyncCatalogProviderList + ?Sized>(
    provider: &P,
    references: TableReferences<'_>,
    config: &SessionConfig,
) -> Result<Arc<dyn CatalogProviderList>> {
    let mut cached_catalogs =
        HashMap::<String, Option<ResolvedCatalogProviderBuilder>>::new();

    for (reference, access) in references.iter() {
        let catalog_name = reference
            .catalog()
            .unwrap_or(&config.options().catalog.default_catalog);

        // We will do three lookups here, one for the catalog, one for the schema, and one for the table
        // We cache the result (both found results and not-found results) to speed up future lookups
        //
        // Note that a cache-miss is not an error at this point.  We allow for the possibility that
        // other providers may supply the reference.
        //
        // If this is the only provider then a not-found error will be raised during planning when it can't
        // find the reference in the cache.

        let catalog = if let Some(catalog) = cached_catalogs.get_mut(catalog_name) {
            catalog
        } else {
            let resolved_catalog = provider.catalog(catalog_name).await?;
            let resolved_catalog =
                resolved_catalog.map(ResolvedCatalogProviderBuilder::new);
            cached_catalogs.insert(catalog_name.to_string(), resolved_catalog);
            cached_catalogs.get_mut(catalog_name).unwrap()
        };

        // If we can't find the catalog don't bother checking the schema / table
        let Some(catalog) = catalog else { continue };

        let schema_name = reference
            .schema()
            .unwrap_or(&config.options().catalog.default_schema);

        let schema = if let Some(schema) = catalog.cached_schemas.get_mut(schema_name) {
            schema
        } else {
            let resolved_schema = catalog.async_provider.schema(schema_name).await?;
            let resolved_schema = resolved_schema.map(|async_schema| {
                ResolvedSchemaProviderBuilder::new(
                    catalog_name.to_string(),
                    async_schema,
                    references.access_aware(),
                )
            });
            catalog
                .cached_schemas
                .insert(schema_name.to_string(), resolved_schema);
            catalog.cached_schemas.get_mut(schema_name).unwrap()
        };

        // If we can't find the catalog don't bother checking the table
        let Some(schema) = schema else { continue };

        schema.resolve_table(reference.table(), access).await?;
    }

    // Build the cached catalog provider list
    let cached_catalogs = cached_catalogs
        .into_iter()
        .filter_map(|(key, maybe_builder)| {
            maybe_builder.map(|catalog_builder| (key, catalog_builder.finish()))
        })
        .collect::<HashMap<_, _>>();

    Ok(Arc::new(ResolvedCatalogProviderList { cached_catalogs }))
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    };

    use arrow::datatypes::SchemaRef;
    use async_trait::async_trait;
    use datafusion_common::{
        Statistics, TableReference, assert_contains, error::Result, exec_err,
    };
    use datafusion_execution::config::SessionConfig;
    use datafusion_expr::{Expr, TableAccess, TableType, dml::InsertOp};
    use datafusion_physical_plan::ExecutionPlan;

    use crate::{
        CatalogProviderList, DynamicFileSchemaProvider, SchemaProvider, Session,
        TableProvider, UrlTableFactory,
    };

    use super::{AsyncCatalogProvider, AsyncCatalogProviderList, AsyncSchemaProvider};

    #[derive(Debug)]
    struct MockTableProvider {}
    #[async_trait]
    impl TableProvider for MockTableProvider {
        /// Get a reference to the schema for this table
        fn schema(&self) -> SchemaRef {
            unimplemented!()
        }

        fn table_type(&self) -> TableType {
            unimplemented!()
        }

        async fn scan(
            &self,
            _state: &dyn Session,
            _projection: Option<&Vec<usize>>,
            _filters: &[Expr],
            _limit: Option<usize>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            unimplemented!()
        }

        fn statistics(&self) -> Option<Statistics> {
            unimplemented!()
        }
    }

    #[derive(Default)]
    struct MockAsyncSchemaProvider {
        lookup_count: AtomicU32,
    }

    const MOCK_CATALOG: &str = "mock_catalog";
    const MOCK_SCHEMA: &str = "mock_schema";
    const MOCK_TABLE: &str = "mock_table";

    #[async_trait]
    impl AsyncSchemaProvider for MockAsyncSchemaProvider {
        async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
            self.lookup_count.fetch_add(1, Ordering::Release);
            if name == MOCK_TABLE {
                Ok(Some(Arc::new(MockTableProvider {})))
            } else {
                Ok(None)
            }
        }
    }

    struct OperationAwareSchema {
        allow_read: bool,
        lookups: Mutex<Vec<TableAccess>>,
    }

    #[async_trait]
    impl AsyncSchemaProvider for OperationAwareSchema {
        async fn table(&self, _name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
            exec_err!("legacy lookup denied")
        }

        async fn table_with_access(
            &self,
            name: &str,
            access: TableAccess,
        ) -> Result<Option<Arc<dyn TableProvider>>> {
            self.lookups.lock().unwrap().push(access);
            if access != TableAccess::Insert(InsertOp::Append)
                && !(self.allow_read && access == TableAccess::Read)
            {
                return exec_err!("operation denied: {access:?}");
            }
            Ok((name == MOCK_TABLE)
                .then(|| Arc::new(MockTableProvider {}) as Arc<dyn TableProvider>))
        }
    }

    #[derive(Clone)]
    struct OperationAwareCatalog {
        schema: Arc<dyn AsyncSchemaProvider>,
    }

    #[async_trait]
    impl AsyncCatalogProvider for OperationAwareCatalog {
        async fn schema(
            &self,
            name: &str,
        ) -> Result<Option<Arc<dyn AsyncSchemaProvider>>> {
            Ok((name == MOCK_SCHEMA).then(|| Arc::clone(&self.schema)))
        }
    }

    #[async_trait]
    impl AsyncCatalogProviderList for OperationAwareCatalog {
        async fn catalog(
            &self,
            name: &str,
        ) -> Result<Option<Arc<dyn AsyncCatalogProvider>>> {
            Ok((name == MOCK_CATALOG)
                .then(|| Arc::new(self.clone()) as Arc<dyn AsyncCatalogProvider>))
        }
    }

    #[derive(Clone, Copy)]
    enum ResolveLevel {
        Schema,
        Catalog,
        CatalogList,
    }

    async fn resolve_operations(
        provider: Arc<dyn AsyncSchemaProvider>,
        references: &[(TableReference, TableAccess)],
        level: ResolveLevel,
    ) -> Result<Arc<dyn SchemaProvider>> {
        let config = test_config();
        let catalog = OperationAwareCatalog {
            schema: Arc::clone(&provider),
        };
        match level {
            ResolveLevel::Schema => {
                provider
                    .resolve_with_access(references, &config, MOCK_CATALOG, MOCK_SCHEMA)
                    .await
            }
            ResolveLevel::Catalog => Ok(AsyncCatalogProvider::resolve_with_access(
                &catalog,
                references,
                &config,
                MOCK_CATALOG,
            )
            .await?
            .schema(MOCK_SCHEMA)
            .unwrap()),
            ResolveLevel::CatalogList => {
                let catalogs: Arc<dyn CatalogProviderList> =
                    AsyncCatalogProviderList::resolve_with_access(
                        &catalog, references, &config,
                    )
                    .await?;
                Ok(catalogs
                    .catalog(MOCK_CATALOG)
                    .unwrap()
                    .schema(MOCK_SCHEMA)
                    .unwrap())
            }
        }
    }

    #[tokio::test]
    async fn operation_aware_caches_do_not_authorize_unresolved_operations() {
        let insert = TableAccess::Insert(InsertOp::Append);
        let references = vec![
            (TableReference::bare(MOCK_TABLE), insert),
            (
                TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, MOCK_TABLE),
                insert,
            ),
        ];
        for level in [
            ResolveLevel::Schema,
            ResolveLevel::Catalog,
            ResolveLevel::CatalogList,
        ] {
            let provider = Arc::new(OperationAwareSchema {
                allow_read: false,
                lookups: Mutex::new(vec![]),
            });
            let cached = resolve_operations(provider.clone(), &references, level)
                .await
                .unwrap();
            assert_eq!(*provider.lookups.lock().unwrap(), vec![insert]);
            assert!(
                cached
                    .table_with_access(MOCK_TABLE, insert)
                    .await
                    .unwrap()
                    .is_some()
            );
            for access in [TableAccess::Read, TableAccess::Insert(InsertOp::Overwrite)] {
                let error = cached
                    .table_with_access(MOCK_TABLE, access)
                    .await
                    .unwrap_err();
                assert_contains!(
                    error.to_string(),
                    format!("was not resolved for operation {access:?}")
                );
            }
            let error = cached.table(MOCK_TABLE).await.unwrap_err();
            assert_contains!(error.to_string(), "was not resolved for operation Read");
            assert!(cached.table("missing").await.unwrap().is_none());
            assert_eq!(cached.table_names(), vec![MOCK_TABLE]);
            assert!(cached.table_exist(MOCK_TABLE));
        }
    }

    #[derive(Debug, Default)]
    struct RecordingUrlFactory {
        calls: AtomicU32,
    }

    #[async_trait]
    impl UrlTableFactory for RecordingUrlFactory {
        async fn try_new(&self, _url: &str) -> Result<Option<Arc<dyn TableProvider>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Some(Arc::new(MockTableProvider {})))
        }
    }

    #[tokio::test]
    async fn unresolved_operations_do_not_trigger_dynamic_file_fallback() {
        let insert = TableAccess::Insert(InsertOp::Append);
        let references = [(TableReference::bare(MOCK_TABLE), insert)];
        for level in [
            ResolveLevel::Schema,
            ResolveLevel::Catalog,
            ResolveLevel::CatalogList,
        ] {
            let provider = Arc::new(OperationAwareSchema {
                allow_read: false,
                lookups: Mutex::new(vec![]),
            });
            let cached = resolve_operations(provider, &references, level)
                .await
                .unwrap();
            let factory = Arc::new(RecordingUrlFactory::default());
            let wrapped = DynamicFileSchemaProvider::new(cached, factory.clone());

            for access in [TableAccess::Read, TableAccess::Insert(InsertOp::Overwrite)] {
                let error = wrapped
                    .table_with_access(MOCK_TABLE, access)
                    .await
                    .unwrap_err();
                assert_contains!(
                    error.to_string(),
                    format!("was not resolved for operation {access:?}")
                );
            }
            let error = wrapped.table(MOCK_TABLE).await.unwrap_err();
            assert_contains!(error.to_string(), "was not resolved for operation Read");
            assert_eq!(factory.calls.load(Ordering::Relaxed), 0);

            assert!(wrapped.table("missing.csv").await.unwrap().is_some());
            assert_eq!(factory.calls.load(Ordering::Relaxed), 1);
        }
    }

    #[tokio::test]
    async fn operation_aware_caches_check_each_operation() {
        let insert = TableAccess::Insert(InsertOp::Append);
        let references = vec![
            (TableReference::bare(MOCK_TABLE), insert),
            (TableReference::bare(MOCK_TABLE), TableAccess::Read),
        ];
        for level in [
            ResolveLevel::Schema,
            ResolveLevel::Catalog,
            ResolveLevel::CatalogList,
        ] {
            let denied = Arc::new(OperationAwareSchema {
                allow_read: false,
                lookups: Mutex::new(vec![]),
            });
            let error = resolve_operations(denied.clone(), &references, level)
                .await
                .unwrap_err();
            assert!(error.to_string().contains("operation denied: Read"));
            assert_eq!(
                *denied.lookups.lock().unwrap(),
                vec![insert, TableAccess::Read]
            );

            let allowed = Arc::new(OperationAwareSchema {
                allow_read: true,
                lookups: Mutex::new(vec![]),
            });
            let cached = resolve_operations(allowed.clone(), &references, level)
                .await
                .unwrap();
            assert_eq!(
                *allowed.lookups.lock().unwrap(),
                vec![insert, TableAccess::Read]
            );
            let write_provider = cached
                .table_with_access(MOCK_TABLE, insert)
                .await
                .unwrap()
                .unwrap();
            let read_provider = cached.table(MOCK_TABLE).await.unwrap().unwrap();
            assert!(!Arc::ptr_eq(&write_provider, &read_provider));
            assert_eq!(cached.table_names(), vec![MOCK_TABLE]);
        }
    }

    #[tokio::test]
    async fn legacy_async_providers_remain_compatible() {
        let provider = MockAsyncSchemaProvider::default();
        let insert = TableAccess::Insert(InsertOp::Append);
        let config = test_config();
        let cached = provider
            .resolve(
                &[TableReference::bare(MOCK_TABLE)],
                &config,
                MOCK_CATALOG,
                MOCK_SCHEMA,
            )
            .await
            .unwrap();
        assert!(cached.table(MOCK_TABLE).await.unwrap().is_some());
        assert!(
            cached
                .table_with_access(MOCK_TABLE, insert)
                .await
                .unwrap()
                .is_some()
        );

        let cached = provider
            .resolve_with_access(
                &[(TableReference::bare(MOCK_TABLE), insert)],
                &config,
                MOCK_CATALOG,
                MOCK_SCHEMA,
            )
            .await
            .unwrap();
        assert!(
            cached
                .table_with_access(MOCK_TABLE, insert)
                .await
                .unwrap()
                .is_some()
        );
        let error = cached.table(MOCK_TABLE).await.unwrap_err();
        assert_contains!(error.to_string(), "was not resolved for operation Read");
        assert_eq!(provider.lookup_count.load(Ordering::Acquire), 2);
    }

    fn test_config() -> SessionConfig {
        let mut config = SessionConfig::default();
        config.options_mut().catalog.default_catalog = MOCK_CATALOG.to_string();
        config.options_mut().catalog.default_schema = MOCK_SCHEMA.to_string();
        config
    }

    #[tokio::test]
    async fn test_async_schema_provider_resolve() {
        async fn check(
            refs: Vec<TableReference>,
            expected_lookup_count: u32,
            found_tables: &[&str],
            not_found_tables: &[&str],
        ) {
            let async_provider = MockAsyncSchemaProvider::default();
            let cached_provider = async_provider
                .resolve(&refs, &test_config(), MOCK_CATALOG, MOCK_SCHEMA)
                .await
                .unwrap();

            assert_eq!(
                async_provider.lookup_count.load(Ordering::Acquire),
                expected_lookup_count
            );

            for table_ref in found_tables {
                let table = cached_provider.table(table_ref).await.unwrap();
                assert!(table.is_some());
            }

            for table_ref in not_found_tables {
                assert!(cached_provider.table(table_ref).await.unwrap().is_none());
            }
        }

        // Basic full lookups
        check(
            vec![
                TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, MOCK_TABLE),
                TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, "not_exists"),
            ],
            2,
            &[MOCK_TABLE],
            &["not_exists"],
        )
        .await;

        // Catalog / schema mismatch doesn't even search
        check(
            vec![
                TableReference::full(MOCK_CATALOG, "foo", MOCK_TABLE),
                TableReference::full("foo", MOCK_SCHEMA, MOCK_TABLE),
            ],
            0,
            &[],
            &[MOCK_TABLE],
        )
        .await;

        // Both hits and misses cached
        check(
            vec![
                TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, MOCK_TABLE),
                TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, MOCK_TABLE),
                TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, "not_exists"),
                TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, "not_exists"),
            ],
            2,
            &[MOCK_TABLE],
            &["not_exists"],
        )
        .await;
    }

    #[derive(Default)]
    struct MockAsyncCatalogProvider {
        lookup_count: AtomicU32,
    }

    #[async_trait]
    impl AsyncCatalogProvider for MockAsyncCatalogProvider {
        async fn schema(
            &self,
            name: &str,
        ) -> Result<Option<Arc<dyn AsyncSchemaProvider>>> {
            self.lookup_count.fetch_add(1, Ordering::Release);
            if name == MOCK_SCHEMA {
                Ok(Some(Arc::new(MockAsyncSchemaProvider::default())))
            } else {
                Ok(None)
            }
        }
    }

    #[tokio::test]
    async fn test_async_catalog_provider_resolve() {
        async fn check(
            refs: Vec<TableReference>,
            expected_lookup_count: u32,
            found_schemas: &[&str],
            not_found_schemas: &[&str],
        ) {
            let async_provider = MockAsyncCatalogProvider::default();
            let cached_provider = async_provider
                .resolve(&refs, &test_config(), MOCK_CATALOG)
                .await
                .unwrap();

            assert_eq!(
                async_provider.lookup_count.load(Ordering::Acquire),
                expected_lookup_count
            );

            for schema_ref in found_schemas {
                let schema = cached_provider.schema(schema_ref);
                assert!(schema.is_some());
            }

            for schema_ref in not_found_schemas {
                assert!(cached_provider.schema(schema_ref).is_none());
            }
        }

        // Basic full lookups
        check(
            vec![
                TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, "x"),
                TableReference::full(MOCK_CATALOG, "not_exists", "x"),
            ],
            2,
            &[MOCK_SCHEMA],
            &["not_exists"],
        )
        .await;

        // Catalog mismatch doesn't even search
        check(
            vec![TableReference::full("foo", MOCK_SCHEMA, "x")],
            0,
            &[],
            &[MOCK_SCHEMA],
        )
        .await;

        // Both hits and misses cached
        check(
            vec![
                TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, "x"),
                TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, "x"),
                TableReference::full(MOCK_CATALOG, "not_exists", "x"),
                TableReference::full(MOCK_CATALOG, "not_exists", "x"),
            ],
            2,
            &[MOCK_SCHEMA],
            &["not_exists"],
        )
        .await;
    }

    #[derive(Default)]
    struct MockAsyncCatalogProviderList {
        lookup_count: AtomicU32,
    }

    #[async_trait]
    impl AsyncCatalogProviderList for MockAsyncCatalogProviderList {
        async fn catalog(
            &self,
            name: &str,
        ) -> Result<Option<Arc<dyn AsyncCatalogProvider>>> {
            self.lookup_count.fetch_add(1, Ordering::Release);
            if name == MOCK_CATALOG {
                Ok(Some(Arc::new(MockAsyncCatalogProvider::default())))
            } else {
                Ok(None)
            }
        }
    }

    #[tokio::test]
    async fn test_async_catalog_provider_list_resolve() {
        async fn check(
            refs: Vec<TableReference>,
            expected_lookup_count: u32,
            found_catalogs: &[&str],
            not_found_catalogs: &[&str],
        ) {
            let async_provider = MockAsyncCatalogProviderList::default();
            let cached_provider =
                async_provider.resolve(&refs, &test_config()).await.unwrap();

            assert_eq!(
                async_provider.lookup_count.load(Ordering::Acquire),
                expected_lookup_count
            );

            for catalog_ref in found_catalogs {
                let catalog = cached_provider.catalog(catalog_ref);
                assert!(catalog.is_some());
            }

            for catalog_ref in not_found_catalogs {
                assert!(cached_provider.catalog(catalog_ref).is_none());
            }
        }

        // Basic full lookups
        check(
            vec![
                TableReference::full(MOCK_CATALOG, "x", "x"),
                TableReference::full("not_exists", "x", "x"),
            ],
            2,
            &[MOCK_CATALOG],
            &["not_exists"],
        )
        .await;

        // Both hits and misses cached
        check(
            vec![
                TableReference::full(MOCK_CATALOG, "x", "x"),
                TableReference::full(MOCK_CATALOG, "x", "x"),
                TableReference::full("not_exists", "x", "x"),
                TableReference::full("not_exists", "x", "x"),
            ],
            2,
            &[MOCK_CATALOG],
            &["not_exists"],
        )
        .await;
    }

    #[tokio::test]
    async fn test_defaults() {
        for table_ref in &[
            TableReference::full(MOCK_CATALOG, MOCK_SCHEMA, MOCK_TABLE),
            TableReference::partial(MOCK_SCHEMA, MOCK_TABLE),
            TableReference::bare(MOCK_TABLE),
        ] {
            let async_provider = MockAsyncCatalogProviderList::default();
            let cached_provider = async_provider
                .resolve(std::slice::from_ref(table_ref), &test_config())
                .await
                .unwrap();

            let catalog = cached_provider
                .catalog(table_ref.catalog().unwrap_or(MOCK_CATALOG))
                .unwrap();
            let schema = catalog
                .schema(table_ref.schema().unwrap_or(MOCK_SCHEMA))
                .unwrap();
            assert!(schema.table(table_ref.table()).await.unwrap().is_some());
        }
    }
}

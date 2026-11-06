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

//! [`DynamicFileCatalog`] that creates tables from file paths

use crate::{CatalogProvider, CatalogProviderList, SchemaProvider, TableProvider};
use async_trait::async_trait;
use datafusion_expr::TableIntent;
use std::fmt::Debug;
use std::sync::Arc;

/// Wrap another catalog provider list
#[derive(Debug)]
pub struct DynamicFileCatalog {
    /// The inner catalog provider list
    inner: Arc<dyn CatalogProviderList>,
    /// The factory that can create a table provider from the file path
    factory: Arc<dyn UrlTableFactory>,
}

impl DynamicFileCatalog {
    pub fn new(
        inner: Arc<dyn CatalogProviderList>,
        factory: Arc<dyn UrlTableFactory>,
    ) -> Self {
        Self { inner, factory }
    }
}

impl CatalogProviderList for DynamicFileCatalog {
    fn register_catalog(
        &self,
        name: String,
        catalog: Arc<dyn CatalogProvider>,
    ) -> Option<Arc<dyn CatalogProvider>> {
        self.inner.register_catalog(name, catalog)
    }

    fn catalog_names(&self) -> Vec<String> {
        self.inner.catalog_names()
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn CatalogProvider>> {
        self.inner.catalog(name).map(|catalog| {
            Arc::new(DynamicFileCatalogProvider::new(
                catalog,
                Arc::clone(&self.factory),
            )) as _
        })
    }
}

/// Wraps another catalog provider
#[derive(Debug)]
struct DynamicFileCatalogProvider {
    /// The inner catalog provider
    inner: Arc<dyn CatalogProvider>,
    /// The factory that can create a table provider from the file path
    factory: Arc<dyn UrlTableFactory>,
}

impl DynamicFileCatalogProvider {
    pub fn new(
        inner: Arc<dyn CatalogProvider>,
        factory: Arc<dyn UrlTableFactory>,
    ) -> Self {
        Self { inner, factory }
    }
}

impl CatalogProvider for DynamicFileCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.inner.schema_names()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.inner.schema(name).map(|schema| {
            Arc::new(DynamicFileSchemaProvider::new(
                schema,
                Arc::clone(&self.factory),
            )) as _
        })
    }

    fn register_schema(
        &self,
        name: &str,
        schema: Arc<dyn SchemaProvider>,
    ) -> datafusion_common::Result<Option<Arc<dyn SchemaProvider>>> {
        self.inner.register_schema(name, schema)
    }
}

/// Implements the [DynamicFileSchemaProvider] that can create tables provider from the file path.
///
/// The provider will try to create a table provider from the file path if the table provider
/// does not exist in the inner schema provider. Errors never trigger fallback.
/// DDL intents only consult the inner catalog. File locations have a separate
/// policy boundary, enforced by the configured [`UrlTableFactory`].
#[derive(Debug)]
pub struct DynamicFileSchemaProvider {
    /// The inner schema provider
    inner: Arc<dyn SchemaProvider>,
    /// The factory that can create a table provider from the file path
    factory: Arc<dyn UrlTableFactory>,
}

impl DynamicFileSchemaProvider {
    /// Create a new [DynamicFileSchemaProvider] with the given inner schema provider.
    pub fn new(
        inner: Arc<dyn SchemaProvider>,
        factory: Arc<dyn UrlTableFactory>,
    ) -> Self {
        Self { inner, factory }
    }
}

#[async_trait]
impl SchemaProvider for DynamicFileSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.inner.table_names()
    }

    async fn table(
        &self,
        name: &str,
        intent: TableIntent,
    ) -> datafusion_common::Result<Option<Arc<dyn TableProvider>>> {
        if let Some(table) = self.inner.table(name, intent).await? {
            return Ok(Some(table));
        };

        // DDL addresses catalog entries, including names that do not exist yet.
        // Only known description and row operations may resolve implicit files.
        if !matches!(
            intent,
            TableIntent::Describe
                | TableIntent::Read
                | TableIntent::Insert(_)
                | TableIntent::Update
                | TableIntent::Delete
                | TableIntent::Truncate
                | TableIntent::Merge { .. }
        ) {
            return Ok(None);
        }
        self.factory.try_new(name, intent).await
    }

    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> datafusion_common::Result<Option<Arc<dyn TableProvider>>> {
        self.inner.register_table(name, table)
    }

    fn deregister_table(
        &self,
        name: &str,
    ) -> datafusion_common::Result<Option<Arc<dyn TableProvider>>> {
        self.inner.deregister_table(name)
    }

    fn table_exist(&self, name: &str) -> bool {
        self.inner.table_exist(name)
    }
}

/// [UrlTableFactory] is a factory that can create a table provider from the given url.
#[async_trait]
pub trait UrlTableFactory: Debug + Sync + Send {
    /// Resolve an implicit file table for the requested operation.
    ///
    /// Called only after a successful catalog miss for description or row access.
    /// Implementations must preserve intent. Authorizing factories must check the
    /// location before parsing paths, configuring stores, or inferring schemas;
    /// an authorized missing catalog name does not authorize access to a file.
    /// Return `Ok(None)` for an absent or unrecognized location, and propagate
    /// authorization, configuration, and storage failures as errors.
    async fn try_new(
        &self,
        url: &str,
        intent: TableIntent,
    ) -> datafusion_common::Result<Option<Arc<dyn TableProvider>>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemorySchemaProvider;
    use datafusion_common::{Result, plan_err};
    use datafusion_expr::dml::InsertOp;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct DeniedSchema;

    #[async_trait]
    impl SchemaProvider for DeniedSchema {
        fn table_names(&self) -> Vec<String> {
            vec![]
        }

        async fn table(
            &self,
            _name: &str,
            _intent: TableIntent,
        ) -> Result<Option<Arc<dyn TableProvider>>> {
            plan_err!("catalog denied")
        }

        fn table_exist(&self, _name: &str) -> bool {
            false
        }
    }

    #[derive(Debug, Default)]
    struct RecordingFactory(Mutex<Vec<TableIntent>>);

    #[async_trait]
    impl UrlTableFactory for RecordingFactory {
        async fn try_new(
            &self,
            _url: &str,
            intent: TableIntent,
        ) -> Result<Option<Arc<dyn TableProvider>>> {
            self.0.lock().unwrap().push(intent);
            plan_err!("factory denied")
        }
    }

    #[tokio::test]
    async fn catalog_denial_never_invokes_file_factory() {
        let factory = Arc::new(RecordingFactory::default());
        let schema =
            DynamicFileSchemaProvider::new(Arc::new(DeniedSchema), factory.clone());
        let error = schema
            .table("s3://private/data.csv", TableIntent::Read)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("catalog denied"));
        assert!(factory.0.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn file_factory_receives_original_intent_and_propagates_errors() {
        let factory = Arc::new(RecordingFactory::default());
        let schema = DynamicFileSchemaProvider::new(
            Arc::new(MemorySchemaProvider::new()),
            factory.clone(),
        );
        let intents = [
            TableIntent::Read,
            TableIntent::Describe,
            TableIntent::Insert(InsertOp::Append),
            TableIntent::Insert(InsertOp::Overwrite),
            TableIntent::Insert(InsertOp::Replace),
            TableIntent::Update,
            TableIntent::Delete,
            TableIntent::Truncate,
            TableIntent::Merge {
                insert: true,
                update: false,
                delete: true,
            },
        ];
        for intent in intents {
            let error = schema.table("data.csv", intent).await.unwrap_err();
            assert!(error.to_string().contains("factory denied"));
        }
        assert_eq!(*factory.0.lock().unwrap(), intents);
    }

    #[tokio::test]
    async fn ddl_misses_never_infer_file_tables() {
        let factory = Arc::new(RecordingFactory::default());
        let schema = DynamicFileSchemaProvider::new(
            Arc::new(MemorySchemaProvider::new()),
            factory.clone(),
        );
        for intent in [
            TableIntent::CreateTable { or_replace: false },
            TableIntent::CreateTable { or_replace: true },
            TableIntent::CreateView { or_replace: true },
            TableIntent::DropTable,
            TableIntent::DropView,
            TableIntent::CreateIndex,
        ] {
            assert!(schema.table("data.csv", intent).await.unwrap().is_none());
        }
        assert!(factory.0.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn registered_tables_take_precedence_over_files() {
        let inner = Arc::new(MemorySchemaProvider::new());
        let table: Arc<dyn TableProvider> = Arc::new(crate::empty::EmptyTable::new(
            Arc::new(arrow::datatypes::Schema::empty()),
        ));
        inner
            .register_table("data.csv".to_string(), Arc::clone(&table))
            .unwrap();
        let factory = Arc::new(RecordingFactory::default());
        let schema = DynamicFileSchemaProvider::new(inner, factory.clone());
        let resolved = schema
            .table("data.csv", TableIntent::Read)
            .await
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&resolved, &table));
        assert!(factory.0.lock().unwrap().is_empty());
    }
}

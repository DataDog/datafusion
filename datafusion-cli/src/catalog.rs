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

use std::sync::{Arc, Weak};

use crate::object_storage::{AwsOptions, GcpOptions, get_object_store};

use datafusion::catalog::{
    CatalogProvider, CatalogProviderList, DynamicFileCatalog, SchemaProvider,
    UrlTableFactory,
};
use datafusion::datasource::dynamic_file::DynamicListTableFactory;
use datafusion::prelude::SessionContext;

use datafusion::common::plan_datafusion_err;
use datafusion::datasource::TableProvider;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::error::Result;
use datafusion::execution::context::SessionState;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::TableAccess;

use async_trait::async_trait;
use dirs::home_dir;
use parking_lot::RwLock;

/// Enable file queries with object-store registration after catalog lookup.
pub fn register_dynamic_catalog(ctx: &SessionContext) {
    let factory = Arc::new(DynamicListTableFactory::default());
    factory.session_store().with_state(ctx.state_weak_ref());
    register_dynamic_catalog_with_factory(ctx, factory);
}

fn register_dynamic_catalog_with_factory(
    ctx: &SessionContext,
    factory: Arc<dyn UrlTableFactory>,
) {
    let catalog = Arc::new(DynamicObjectStoreCatalog::new(
        Arc::clone(ctx.state().catalog_list()),
        ctx.state_weak_ref(),
    ));
    // File inference needs a registered store. Catalog errors must propagate
    // before either fallback, rather than triggering object-store registration.
    ctx.register_catalog_list(Arc::new(DynamicFileCatalog::new(catalog, factory)));
}

/// Wraps another catalog, automatically register require object stores for the file locations
#[derive(Debug)]
pub struct DynamicObjectStoreCatalog {
    inner: Arc<dyn CatalogProviderList>,
    state: Weak<RwLock<SessionState>>,
}

impl DynamicObjectStoreCatalog {
    pub fn new(
        inner: Arc<dyn CatalogProviderList>,
        state: Weak<RwLock<SessionState>>,
    ) -> Self {
        Self { inner, state }
    }
}

impl CatalogProviderList for DynamicObjectStoreCatalog {
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
        let state = self.state.clone();
        self.inner.catalog(name).map(|catalog| {
            Arc::new(DynamicObjectStoreCatalogProvider::new(catalog, state)) as _
        })
    }
}

/// Wraps another catalog provider
#[derive(Debug)]
struct DynamicObjectStoreCatalogProvider {
    inner: Arc<dyn CatalogProvider>,
    state: Weak<RwLock<SessionState>>,
}

impl DynamicObjectStoreCatalogProvider {
    pub fn new(
        inner: Arc<dyn CatalogProvider>,
        state: Weak<RwLock<SessionState>>,
    ) -> Self {
        Self { inner, state }
    }
}

impl CatalogProvider for DynamicObjectStoreCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.inner.schema_names()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        let state = self.state.clone();
        self.inner.schema(name).map(|schema| {
            Arc::new(DynamicObjectStoreSchemaProvider::new(schema, state)) as _
        })
    }

    fn register_schema(
        &self,
        name: &str,
        schema: Arc<dyn SchemaProvider>,
    ) -> Result<Option<Arc<dyn SchemaProvider>>> {
        self.inner.register_schema(name, schema)
    }
}

/// Wraps another schema provider. [DynamicObjectStoreSchemaProvider] is responsible for registering the required
/// object stores for the file locations.
#[derive(Debug)]
struct DynamicObjectStoreSchemaProvider {
    inner: Arc<dyn SchemaProvider>,
    state: Weak<RwLock<SessionState>>,
}

impl DynamicObjectStoreSchemaProvider {
    pub fn new(
        inner: Arc<dyn SchemaProvider>,
        state: Weak<RwLock<SessionState>>,
    ) -> Self {
        Self { inner, state }
    }
}

#[async_trait]
impl SchemaProvider for DynamicObjectStoreSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.inner.table_names()
    }

    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        self.inner.register_table(name, table)
    }

    async fn table(
        &self,
        name: &str,
        access: TableAccess,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        if let Some(inner_table) = self.inner.table(name, access).await? {
            return Ok(Some(inner_table));
        }

        // if the inner schema provider didn't have a table by
        // that name, try to treat it as a listing table
        let mut state = self
            .state
            .upgrade()
            .ok_or_else(|| plan_datafusion_err!("locking error"))?
            .read()
            .clone();
        let mut builder = SessionStateBuilder::from(state.clone());
        let optimized_name = substitute_tilde(name.to_owned());
        let table_url = ListingTableUrl::parse(optimized_name.as_str())?;
        let scheme = table_url.scheme();
        let url = table_url.as_ref();

        // If the store is already registered for this URL then `get_store`
        // will return `Ok` which means we don't need to register it again. However,
        // if `get_store` returns an `Err` then it means the corresponding store is
        // not registered yet and we need to register it
        match state.runtime_env().object_store_registry.get_store(url) {
            Ok(_) => { /*Nothing to do here, store for this URL is already registered*/ }
            Err(_) => {
                // Register the store for this URL. Here we don't have access
                // to any command options so the only choice is to use an empty collection
                match scheme {
                    "s3" | "oss" | "cos" => {
                        if let Some(table_options) = builder.table_options() {
                            table_options.extensions.insert(AwsOptions::default())
                        }
                    }
                    "gs" | "gcs" => {
                        if let Some(table_options) = builder.table_options() {
                            table_options.extensions.insert(GcpOptions::default())
                        }
                    }
                    _ => {}
                };
                state = builder.build();
                let store = get_object_store(
                    &state,
                    table_url.scheme(),
                    url,
                    &state.default_table_options(),
                    false,
                )
                .await?;
                state.runtime_env().register_object_store(url, store);
            }
        }
        self.inner.table(name, access).await
    }

    fn deregister_table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        self.inner.deregister_table(name)
    }

    fn table_exist(&self, name: &str) -> bool {
        self.inner.table_exist(name)
    }
}

pub fn substitute_tilde(cur: String) -> String {
    if let Some(usr_dir_path) = home_dir()
        && let Some(usr_dir) = usr_dir_path.to_str()
        && cur.starts_with('~')
        && !usr_dir.is_empty()
    {
        return cur.replacen('~', usr_dir, 1);
    }
    cur
}
#[cfg(test)]
mod tests {
    use std::{env, vec};

    use super::*;

    use datafusion::catalog::SchemaProvider;
    use datafusion::prelude::SessionContext;

    #[derive(Debug)]
    struct ReadOnlySchema;

    #[async_trait]
    impl SchemaProvider for ReadOnlySchema {
        fn table_names(&self) -> Vec<String> {
            vec![]
        }
        fn table_exist(&self, _: &str) -> bool {
            false
        }
        async fn table(
            &self,
            _: &str,
            access: TableAccess,
        ) -> Result<Option<Arc<dyn TableProvider>>> {
            if access != TableAccess::Read {
                return datafusion::common::plan_err!("operation denied: {access:?}");
            }
            Ok(None)
        }
    }

    #[tokio::test]
    async fn denied_insert_does_not_enter_object_store_fallback() {
        let wrapper =
            DynamicObjectStoreSchemaProvider::new(Arc::new(ReadOnlySchema), Weak::new());
        let error = wrapper
            .table(
                "target",
                TableAccess::Insert(datafusion::logical_expr::dml::InsertOp::Append),
            )
            .await
            .unwrap_err();
        // The dead session reference would produce "locking error" if fallback ran.
        assert!(
            error
                .to_string()
                .contains("operation denied: Insert(Append)")
        );
    }

    #[derive(Debug)]
    struct StoreCheckingFactory {
        runtime: Arc<datafusion::execution::runtime_env::RuntimeEnv>,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl UrlTableFactory for StoreCheckingFactory {
        async fn try_new(&self, url: &str) -> Result<Option<Arc<dyn TableProvider>>> {
            self.runtime.object_store(ListingTableUrl::parse(url)?)?;
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(None)
        }
    }

    #[tokio::test]
    async fn dynamic_catalog_registers_store_before_file_inference() {
        let ctx = SessionContext::new();
        ctx.catalog("datafusion")
            .unwrap()
            .register_schema("public", Arc::new(ReadOnlySchema))
            .unwrap();
        let factory = Arc::new(StoreCheckingFactory {
            runtime: ctx.runtime_env(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        register_dynamic_catalog_with_factory(
            &ctx,
            Arc::clone(&factory) as Arc<dyn UrlTableFactory>,
        );
        let schema = ctx.catalog("datafusion").unwrap().schema("public").unwrap();
        let location = "http://example.com/file.csv";
        let error = schema
            .table(
                location,
                TableAccess::Insert(datafusion::logical_expr::dml::InsertOp::Append),
            )
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("operation denied: Insert(Append)")
        );
        assert!(
            ctx.runtime_env()
                .object_store(ListingTableUrl::parse(location).unwrap())
                .is_err()
        );
        assert_eq!(factory.calls.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert!(
            schema
                .table(location, TableAccess::Read)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(factory.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn dynamic_catalog_reads_a_local_csv() {
        let path = format!(
            "{}/csv/aggregate_test_100.csv",
            datafusion::common::test_util::arrow_test_data()
        );
        let ctx = SessionContext::new();
        register_dynamic_catalog(&ctx);
        let batches = ctx
            .table(path.as_str())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert!(batches.iter().any(|batch| batch.num_rows() > 0));
    }

    fn setup_context() -> (SessionContext, Arc<dyn SchemaProvider>) {
        let ctx = SessionContext::new();
        ctx.register_catalog_list(Arc::new(DynamicObjectStoreCatalog::new(
            ctx.state().catalog_list().clone(),
            ctx.state_weak_ref(),
        )));

        let provider = &DynamicObjectStoreCatalog::new(
            ctx.state().catalog_list().clone(),
            ctx.state_weak_ref(),
        ) as &dyn CatalogProviderList;
        let catalog = provider
            .catalog(provider.catalog_names().first().unwrap())
            .unwrap();
        let schema = catalog
            .schema(catalog.schema_names().first().unwrap())
            .unwrap();
        (ctx, schema)
    }

    #[tokio::test]
    async fn query_http_location_test() -> Result<()> {
        // This is a unit test so not expecting a connection or a file to be
        // available
        let domain = "example.com";
        let location = format!("http://{domain}/file.parquet");

        let (ctx, schema) = setup_context();

        // That's a non registered table so expecting None here
        let table = schema.table(&location, TableAccess::Read).await?;
        assert!(table.is_none());

        // It should still create an object store for the location in the SessionState
        let store = ctx
            .runtime_env()
            .object_store(ListingTableUrl::parse(location)?)?;

        assert_eq!(format!("{store}"), "HttpStore");

        // The store must be configured for this domain
        let expected_domain = format!("Domain(\"{domain}\")");
        assert!(format!("{store:?}").contains(&expected_domain));

        Ok(())
    }

    #[tokio::test]
    async fn query_s3_location_test() -> Result<()> {
        let aws_envs = vec![
            "AWS_ENDPOINT",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ALLOW_HTTP",
        ];
        for aws_env in aws_envs {
            if env::var(aws_env).is_err() {
                eprint!("aws envs not set, skipping s3 test");
                return Ok(());
            }
        }

        let bucket = "examples3bucket";
        let location = format!("s3://{bucket}/file.parquet");

        let (ctx, schema) = setup_context();

        let table = schema.table(&location, TableAccess::Read).await?;
        assert!(table.is_none());

        let store = ctx
            .runtime_env()
            .object_store(ListingTableUrl::parse(location)?)?;
        assert_eq!(format!("{store}"), format!("AmazonS3({bucket})"));

        // The store must be configured for this domain
        let expected_bucket = format!("bucket: \"{bucket}\"");
        assert!(format!("{store:?}").contains(&expected_bucket));

        Ok(())
    }

    #[tokio::test]
    async fn query_gs_location_test() -> Result<()> {
        let bucket = "examplegsbucket";
        let location = format!("gs://{bucket}/file.parquet");

        let (ctx, schema) = setup_context();

        let table = schema.table(&location, TableAccess::Read).await?;
        assert!(table.is_none());

        let store = ctx
            .runtime_env()
            .object_store(ListingTableUrl::parse(location)?)?;
        assert_eq!(format!("{store}"), format!("GoogleCloudStorage({bucket})"));

        // The store must be configured for this domain
        let expected_bucket = format!("bucket_name_encoded: \"{bucket}\"");
        assert!(format!("{store:?}").contains(&expected_bucket));

        Ok(())
    }

    #[tokio::test]
    async fn query_invalid_location_test() {
        let location = "ts://file.parquet";
        let (_ctx, schema) = setup_context();

        assert!(schema.table(location, TableAccess::Read).await.is_err());
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_substitute_tilde() {
        use std::{env, path::PathBuf};
        let original_home = home_dir();
        let test_home_path = if cfg!(windows) {
            "C:\\Users\\user"
        } else {
            "/home/user"
        };
        unsafe {
            env::set_var(
                if cfg!(windows) { "USERPROFILE" } else { "HOME" },
                test_home_path,
            );
        }
        let input = "~/Code/datafusion/benchmarks/data/tpch_sf1/part/part-0.parquet";
        let expected = PathBuf::from(test_home_path)
            .join("Code")
            .join("datafusion")
            .join("benchmarks")
            .join("data")
            .join("tpch_sf1")
            .join("part")
            .join("part-0.parquet")
            .to_string_lossy()
            .to_string();
        let actual = substitute_tilde(input.to_string());
        assert_eq!(actual, expected);
        unsafe {
            match original_home {
                Some(home_path) => env::set_var(
                    if cfg!(windows) { "USERPROFILE" } else { "HOME" },
                    home_path.to_str().unwrap(),
                ),
                None => {
                    env::remove_var(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
                }
            }
        }
    }
}

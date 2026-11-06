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
use async_trait::async_trait;
use datafusion::catalog::{
    CatalogProvider, CatalogProviderList, DynamicFileCatalog, TableProvider,
    UrlTableFactory,
};
use datafusion::common::plan_datafusion_err;
use datafusion::datasource::dynamic_file::DynamicListTableFactory;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::SessionState;
use datafusion::execution::object_store::ObjectStoreNotFound;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::TableIntent;
use dirs::home_dir;
use parking_lot::RwLock;

/// Resolves implicit file tables and configures their object stores after a
/// successful catalog miss. Wrap an ordinary catalog, without `enable_url_table`:
/// this wrapper installs its own file factory so store setup precedes inference.
///
/// This CLI convenience enables file access using the session's credentials.
/// It does not enforce location policy; restricted applications need an
/// authorizing file factory or must leave implicit file resolution disabled.
#[derive(Debug)]
pub struct DynamicObjectStoreCatalog {
    inner: DynamicFileCatalog,
}

impl DynamicObjectStoreCatalog {
    pub fn new(
        inner: Arc<dyn CatalogProviderList>,
        state: Weak<RwLock<SessionState>>,
    ) -> Self {
        Self {
            inner: DynamicFileCatalog::new(
                inner,
                Arc::new(DynamicObjectStoreTableFactory::new(state)),
            ),
        }
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
        self.inner.catalog(name)
    }
}

#[derive(Debug)]
struct DynamicObjectStoreTableFactory {
    state: Weak<RwLock<SessionState>>,
    inner: DynamicListTableFactory,
}

impl DynamicObjectStoreTableFactory {
    fn new(state: Weak<RwLock<SessionState>>) -> Self {
        let inner = DynamicListTableFactory::default();
        inner.session_store().with_state(state.clone());
        Self { state, inner }
    }

    async fn prepare_store(&self, name: &str) -> Result<()> {
        let mut state = self
            .state
            .upgrade()
            .ok_or_else(|| plan_datafusion_err!("locking error"))?
            .read()
            .clone();
        let mut builder = SessionStateBuilder::from(state.clone());
        let table_url = ListingTableUrl::parse(name)?;
        let scheme = table_url.scheme();
        let url = table_url.as_ref();

        match state.runtime_env().object_store_registry.get_store(url) {
            Ok(_) => return Ok(()),
            Err(DataFusionError::External(error))
                if error.is::<ObjectStoreNotFound>() =>
            {
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
            Err(error) => return Err(error),
        }
        Ok(())
    }
}

#[async_trait]
impl UrlTableFactory for DynamicObjectStoreTableFactory {
    async fn try_new(
        &self,
        url: &str,
        intent: TableIntent,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        let location = substitute_tilde(url.to_owned());
        self.prepare_store(&location).await?;
        self.inner.try_new(&location, intent).await
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

    use datafusion::prelude::SessionContext;

    fn setup_context() -> (SessionContext, DynamicObjectStoreTableFactory) {
        let ctx = SessionContext::new();
        let factory = DynamicObjectStoreTableFactory::new(ctx.state_weak_ref());
        (ctx, factory)
    }

    #[derive(Debug)]
    struct DeniedRegistry;

    impl datafusion::execution::object_store::ObjectStoreRegistry for DeniedRegistry {
        fn register_store(
            &self,
            _url: &url::Url,
            _store: Arc<dyn object_store::ObjectStore>,
        ) -> Option<Arc<dyn object_store::ObjectStore>> {
            panic!("denial must not register a replacement store")
        }
        fn get_store(
            &self,
            _url: &url::Url,
        ) -> Result<Arc<dyn object_store::ObjectStore>> {
            datafusion::common::plan_err!("registry denied")
        }
    }

    #[tokio::test]
    async fn registry_failure_does_not_trigger_automatic_registration() {
        let runtime = datafusion::execution::runtime_env::RuntimeEnvBuilder::new()
            .with_object_store_registry(Arc::new(DeniedRegistry))
            .build_arc()
            .unwrap();
        let ctx = SessionContext::new_with_config_rt(Default::default(), runtime);
        let factory = DynamicObjectStoreTableFactory::new(ctx.state_weak_ref());
        let error = factory
            .prepare_store("https://example.com/private.csv")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("registry denied"));
    }

    #[derive(Debug)]
    struct DeniedSchema;

    #[async_trait]
    impl datafusion::catalog::SchemaProvider for DeniedSchema {
        fn table_names(&self) -> Vec<String> {
            vec![]
        }
        fn table_exist(&self, _name: &str) -> bool {
            false
        }
        async fn table(
            &self,
            _name: &str,
            _intent: TableIntent,
        ) -> Result<Option<Arc<dyn TableProvider>>> {
            datafusion::common::plan_err!("catalog denied")
        }
    }

    #[tokio::test]
    async fn catalog_denial_does_not_register_a_store() {
        let ctx = SessionContext::new();
        ctx.catalog("datafusion")
            .unwrap()
            .register_schema("public", Arc::new(DeniedSchema))
            .unwrap();
        let catalog = DynamicObjectStoreCatalog::new(
            ctx.state().catalog_list().clone(),
            ctx.state_weak_ref(),
        );
        let schema = catalog
            .catalog("datafusion")
            .unwrap()
            .schema("public")
            .unwrap();
        let location = "https://example.com/private.csv";
        for intent in [
            TableIntent::Read,
            TableIntent::Describe,
            TableIntent::CreateTable { or_replace: false },
        ] {
            let error = schema.table(location, intent).await.unwrap_err();
            assert!(error.to_string().contains("catalog denied"));
        }
        assert!(
            ctx.runtime_env()
                .object_store(ListingTableUrl::parse(location).unwrap())
                .is_err()
        );
    }

    #[tokio::test]
    async fn ddl_miss_does_not_register_a_store() {
        let ctx = SessionContext::new();
        let catalog = DynamicObjectStoreCatalog::new(
            ctx.state().catalog_list().clone(),
            ctx.state_weak_ref(),
        );
        let schema = catalog
            .catalog("datafusion")
            .unwrap()
            .schema("public")
            .unwrap();
        let location = "https://example.com/private.csv";
        assert!(
            schema
                .table(location, TableIntent::CreateTable { or_replace: false })
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            ctx.runtime_env()
                .object_store(ListingTableUrl::parse(location).unwrap())
                .is_err()
        );
    }

    #[tokio::test]
    async fn query_http_location_test() -> Result<()> {
        // This is a unit test so not expecting a connection or a file to be
        // available
        let domain = "example.com";
        let location = format!("http://{domain}/file.parquet");

        let (ctx, factory) = setup_context();

        factory.prepare_store(&location).await?;

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

        let (ctx, factory) = setup_context();

        factory.prepare_store(&location).await?;

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

        let (ctx, factory) = setup_context();

        factory.prepare_store(&location).await?;

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
        let (_ctx, factory) = setup_context();

        assert!(factory.prepare_store(location).await.is_err());
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

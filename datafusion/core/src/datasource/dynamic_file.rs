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

//! dynamic_file_schema contains an [`UrlTableFactory`] implementation that
//! can create a [`ListingTable`] from the given url.

use std::sync::Arc;

use crate::datasource::TableProvider;
use crate::datasource::listing::infer_listing_options;
use crate::datasource::listing::{ListingTable, ListingTableConfig, ListingTableUrl};
use crate::error::Result;
use crate::execution::context::SessionState;

use datafusion_catalog::UrlTableFactory;
use datafusion_common::{DataFusionError, plan_datafusion_err};
use datafusion_expr::TableIntent;
use datafusion_session::SessionStore;

use async_trait::async_trait;

/// [DynamicListTableFactory] creates a [ListingTable] from a file location.
///
/// This factory does not authorize locations or restrict operations. Enabling it
/// permits implicit file access using the session's configured object stores.
/// Applications requiring authorization must wrap or replace the factory and
/// check locations before invoking it.
#[derive(Default, Debug)]
pub struct DynamicListTableFactory {
    /// The session store that contains the current session.
    session_store: SessionStore,
}

impl DynamicListTableFactory {
    /// Create a new [DynamicListTableFactory] with the given state store.
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    /// Get the session store.
    pub fn session_store(&self) -> &SessionStore {
        &self.session_store
    }
}

#[async_trait]
impl UrlTableFactory for DynamicListTableFactory {
    async fn try_new(
        &self,
        url: &str,
        _intent: TableIntent,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        let Ok(table_url) = ListingTableUrl::parse(url) else {
            return Ok(None);
        };

        let state = &self
            .session_store()
            .get_session()
            .upgrade()
            .and_then(|session| {
                session
                    .read()
                    .as_any()
                    .downcast_ref::<SessionState>()
                    .cloned()
            })
            .ok_or_else(|| plan_datafusion_err!("get current SessionStore error"))?;

        let options = match infer_listing_options(&table_url, state).await {
            Ok(Some(options)) => options,
            Ok(None) => return Ok(None),
            Err(DataFusionError::ObjectStore(error))
                if matches!(*error, object_store::Error::NotFound { .. }) =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let cfg = ListingTableConfig::new(table_url)
            .with_listing_options(options)
            .infer_partitions_from_path(state)
            .await?
            .infer_schema(state)
            .await?;
        Ok(Some(Arc::new(ListingTable::try_new(cfg)?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{SessionConfig, SessionContext};
    use datafusion_execution::object_store::ObjectStoreRegistry;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use object_store::ObjectStore;
    use url::Url;

    fn factory(ctx: &SessionContext) -> DynamicListTableFactory {
        let factory = DynamicListTableFactory::default();
        factory.session_store().with_state(ctx.state_weak_ref());
        factory
    }

    #[tokio::test]
    async fn missing_and_empty_locations_are_absent_but_invalid_formats_are_errors() {
        let ctx = SessionContext::new();
        let factory = factory(&ctx);
        let dir = tempfile::tempdir().unwrap();
        for path in [dir.path().to_path_buf(), dir.path().join("missing.csv")] {
            assert!(
                factory
                    .try_new(path.to_str().unwrap(), TableIntent::Read)
                    .await
                    .unwrap()
                    .is_none()
            );
        }
        let path = dir.path().join("invalid.unknown_format");
        std::fs::write(&path, "1\n").unwrap();
        let error = factory
            .try_new(path.to_str().unwrap(), TableIntent::Read)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("No file_format found"));
    }

    #[tokio::test]
    async fn unconfigured_store_is_not_a_missing_file() {
        let ctx = SessionContext::new();
        let error = factory(&ctx)
            .try_new("s3://unconfigured/data.csv", TableIntent::Read)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("No suitable object store"));
    }

    #[derive(Debug)]
    struct DeniedStoreRegistry;

    impl ObjectStoreRegistry for DeniedStoreRegistry {
        fn register_store(
            &self,
            _url: &Url,
            _store: Arc<dyn ObjectStore>,
        ) -> Option<Arc<dyn ObjectStore>> {
            None
        }
        fn get_store(&self, url: &Url) -> Result<Arc<dyn ObjectStore>> {
            Err(object_store::Error::PermissionDenied {
                path: url.to_string(),
                source: "storage denied".into(),
            }
            .into())
        }
    }

    #[tokio::test]
    async fn storage_denial_is_not_a_missing_file() {
        let runtime = RuntimeEnvBuilder::new()
            .with_object_store_registry(Arc::new(DeniedStoreRegistry))
            .build_arc()
            .unwrap();
        let ctx = SessionContext::new_with_config_rt(SessionConfig::new(), runtime);
        let error = factory(&ctx)
            .try_new("s3://private/data.csv", TableIntent::Read)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("storage denied"));
    }

    #[tokio::test]
    async fn implicit_files_support_describe_and_insert() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("data.csv"), "value\n1\n2\n").unwrap();
        let path = format!("{}/", dir.path().display());
        let ctx = SessionContext::new().enable_url_table();
        ctx.sql(&format!("DESCRIBE '{path}'")).await.unwrap();
        ctx.sql(&format!("INSERT INTO '{path}' VALUES (3)"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batches = ctx
            .sql(&format!("SELECT * FROM '{path}'"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            3
        );
    }
}

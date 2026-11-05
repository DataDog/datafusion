<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Catalogs, Schemas, and Tables

This section describes how to create and manage catalogs, schemas, and tables in DataFusion. For those wanting to dive into the code quickly please see the [example](https://github.com/apache/datafusion/blob/main/datafusion-examples/examples/data_io/catalog.rs).

## General Concepts

Catalog providers, catalogs, schemas, and tables are organized in a hierarchy. A `CatalogProviderList` contains `CatalogProvider`s, a `CatalogProvider` contains `SchemaProviders` and a `SchemaProvider` contains `TableProvider`s.

DataFusion comes with a basic in memory catalog functionality in the [`catalog` module]. You can use these in memory implementations as is, or extend DataFusion with your own catalog implementations, for example based on local files or files on remote object storage.

DataFusion supports DDL queries (e.g. `CREATE TABLE`) using the catalog API described in this section. See the [TableProvider] section for information on DML queries (e.g. `INSERT INTO`).

[`catalog` module]: https://docs.rs/datafusion/latest/datafusion/catalog/index.html
[tableprovider]: ./custom-table-providers.md

Similarly to other concepts in DataFusion, you'll implement various traits to create your own catalogs, schemas, and tables. The following sections describe the traits you'll need to implement.

The `CatalogProviderList` trait has methods to register new catalogs, get a catalog by name and list all catalogs .The `CatalogProvider` trait has methods to set a schema to a name, get a schema by name, and list all schemas. The `SchemaProvider`, which can be registered with a `CatalogProvider`, has methods to set a table to a name, get a table by name, list all tables, deregister a table, and check for a table's existence. The `TableProvider` trait has methods to scan underlying data and use it in DataFusion. The `TableProvider` trait is covered in more detail [here](./custom-table-providers.md).

In the following example, we'll implement an in memory catalog, starting with the `SchemaProvider` trait as we need one to register with the `CatalogProvider`. Finally we will implement `CatalogProviderList` to register the `CatalogProvider`.

## Implementing `MemorySchemaProvider`

The `MemorySchemaProvider` is a simple implementation of the `SchemaProvider` trait. It stores state (i.e. tables) in a `DashMap`, which then underlies the `SchemaProvider` trait.

```rust
use std::sync::Arc;
use dashmap::DashMap;
use datafusion::catalog::{TableProvider, SchemaProvider};

#[derive(Debug)]
pub struct MemorySchemaProvider {
    tables: DashMap<String, Arc<dyn TableProvider>>,
}
```

`tables` is the key-value pair described above. The underlying state could also be another data structure or other storage mechanism such as a file or transactional database.

Then we implement the `SchemaProvider` trait for `MemorySchemaProvider`.

```rust
# use std::sync::Arc;
# use dashmap::DashMap;
# use datafusion::catalog::TableProvider;
#
# #[derive(Debug)]
# pub struct MemorySchemaProvider {
#     tables: DashMap<String, Arc<dyn TableProvider>>,
# }

use std::any::Any;
use datafusion::catalog::SchemaProvider;
use async_trait::async_trait;
use datafusion::common::{Result, exec_err};

#[async_trait]
impl SchemaProvider for MemorySchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.tables
            .iter()
            .map(|table| table.key().clone())
            .collect()
    }

    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        Ok(self.tables.get(name).map(|table| table.value().clone()))
    }

    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        if self.table_exist(name.as_str()) {
            return exec_err!(
                "The table {name} already exists"
            );
        }
        Ok(self.tables.insert(name, table))
    }

    fn deregister_table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        Ok(self.tables.remove(name).map(|(_, table)| table))
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }
}
```

Without getting into a `CatalogProvider` implementation, we can create a `MemorySchemaProvider` and register `TableProvider`s with it.

```rust
# use std::sync::Arc;
# use dashmap::DashMap;
# use datafusion::catalog::TableProvider;
#
# #[derive(Debug)]
# pub struct MemorySchemaProvider {
#     tables: DashMap<String, Arc<dyn TableProvider>>,
# }
#
# use std::any::Any;
# use datafusion::catalog::SchemaProvider;
# use async_trait::async_trait;
# use datafusion::common::{Result, exec_err};
#
# #[async_trait]
# impl SchemaProvider for MemorySchemaProvider {
#     fn table_names(&self) -> Vec<String> {
#         self.tables
#             .iter()
#             .map(|table| table.key().clone())
#             .collect()
#     }
#
#     async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
#         Ok(self.tables.get(name).map(|table| table.value().clone()))
#     }
#
#     fn register_table(
#         &self,
#         name: String,
#         table: Arc<dyn TableProvider>,
#     ) -> Result<Option<Arc<dyn TableProvider>>> {
#         if self.table_exist(name.as_str()) {
#             return exec_err!(
#                 "The table {name} already exists"
#             );
#         }
#         Ok(self.tables.insert(name, table))
#     }
#
#     fn deregister_table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
#         Ok(self.tables.remove(name).map(|(_, table)| table))
#     }
#
#     fn table_exist(&self, name: &str) -> bool {
#         self.tables.contains_key(name)
#     }
# }

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use arrow::array::{self, Array, ArrayRef, Int32Array};

impl MemorySchemaProvider {
    /// Instantiates a new MemorySchemaProvider with an empty collection of tables.
    pub fn new() -> Self {
        Self {
            tables: DashMap::new(),
        }
    }
}

let schema_provider = Arc::new(MemorySchemaProvider::new());

let table_provider = {
    let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, true)]));
    let arr = Arc::new(Int32Array::from((1..=1).collect::<Vec<_>>()));
    let partitions = vec![vec![RecordBatch::try_new(schema.clone(), vec![arr as ArrayRef]).unwrap()]];
    Arc::new(MemTable::try_new(schema, partitions).unwrap())
};

schema_provider.register_table("users".to_string(), table_provider);

let table = schema_provider.table("users");
```

### Asynchronous `SchemaProvider`

It's often useful to fetch metadata about which tables are in a schema, from a remote source. For example, a schema provider could fetch metadata from a remote database. To support this, the `SchemaProvider` trait has an asynchronous `table` method.

The trait is roughly the same except for the `table` method, and the addition of the `#[async_trait]` attribute.

```rust
# use async_trait::async_trait;
# use std::sync::Arc;
# use datafusion::catalog::{TableProvider, SchemaProvider};
# use datafusion::common::Result;
#
# type OriginSchema = arrow::datatypes::Schema;
#
# #[derive(Debug)]
# struct Schema(OriginSchema);

#[async_trait]
impl SchemaProvider for Schema {
    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
#       todo!();
    }

#    fn table_names(&self) -> Vec<std::string::String> { todo!() }
#    fn table_exist(&self, _: &str) -> bool { todo!() }
}
```

## Implementing `MemoryCatalogProvider`

As mentioned, the `CatalogProvider` can manage the schemas in a catalog, and the `MemoryCatalogProvider` is a simple implementation of the `CatalogProvider` trait. It stores schemas in a `DashMap`. With that the `CatalogProvider` trait can be implemented.

```rust
use std::any::Any;
use std::sync::Arc;
use dashmap::DashMap;
use datafusion::catalog::{CatalogProvider, SchemaProvider};
use datafusion::common::Result;

#[derive(Debug)]
pub struct MemoryCatalogProvider {
    schemas: DashMap<String, Arc<dyn SchemaProvider>>,
}

impl CatalogProvider for MemoryCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.schemas.iter().map(|s| s.key().clone()).collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas.get(name).map(|s| s.value().clone())
    }

    fn register_schema(
        &self,
        name: &str,
        schema: Arc<dyn SchemaProvider>,
    ) -> Result<Option<Arc<dyn SchemaProvider>>> {
        Ok(self.schemas.insert(name.into(), schema))
    }

    fn deregister_schema(
        &self,
        name: &str,
        cascade: bool,
    ) -> Result<Option<Arc<dyn SchemaProvider>>> {
        /// `cascade` is not used here, but can be used to control whether
        /// to delete all tables in the schema or not.
        if let Some(schema) = self.schema(name) {
            let (_, removed) = self.schemas.remove(name).unwrap();
            Ok(Some(removed))
        } else {
            Ok(None)
        }
    }
}
```

Again, this is fairly straightforward, as there's an underlying data structure to store the state, via key-value pairs. With that the `CatalogProviderList` trait can be implemented.

## Implementing `MemoryCatalogProviderList`

```rust

use std::any::Any;
use std::sync::Arc;
use dashmap::DashMap;
use datafusion::catalog::{CatalogProviderList, CatalogProvider};
use datafusion::common::Result;

#[derive(Debug)]
pub struct MemoryCatalogProviderList {
    /// Collection of catalogs containing schemas and ultimately TableProviders
    pub catalogs: DashMap<String, Arc<dyn CatalogProvider>>,
}

impl CatalogProviderList for MemoryCatalogProviderList {
    fn register_catalog(
        &self,
        name: String,
        catalog: Arc<dyn CatalogProvider>,
    ) -> Option<Arc<dyn CatalogProvider>> {
        self.catalogs.insert(name, catalog)
    }

    fn catalog_names(&self) -> Vec<String> {
        self.catalogs.iter().map(|c| c.key().clone()).collect()
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn CatalogProvider>> {
        self.catalogs.get(name).map(|c| c.value().clone())
    }
}
```

Like other traits, it also maintains the mapping of the Catalog's name to the CatalogProvider.

## Recap

To recap, you need to:

1. Implement the `TableProvider` trait to create a table provider, or use an existing one.
2. Implement the `SchemaProvider` trait to create a schema provider, or use an existing one.
3. Implement the `CatalogProvider` trait to create a catalog provider, or use an existing one.
4. Implement the `CatalogProviderList` trait to create a CatalogProviderList, or use an existing one.

## Operation-aware table resolution

A caller may be allowed to insert into a table without being allowed to select
from it. The SQL planner still needs the destination schema to validate the
insert. Implement `SchemaProvider::table_with_access` to authorize that lookup
before returning the provider and before the planner uses its schema.

`TableAccess` describes the operation being planned. Applications decide which
privileges each operation requires:

| SQL occurrence                                         | Access                        |
| ------------------------------------------------------ | ----------------------------- |
| A source relation, including a subquery within a write | `Read`                        |
| `INSERT INTO` destination                              | `Insert(InsertOp::Append)`    |
| `INSERT OVERWRITE` destination                         | `Insert(InsertOp::Overwrite)` |
| `REPLACE INTO` or `INSERT OR REPLACE` destination      | `Insert(InsertOp::Replace)`   |
| `UPDATE` destination                                   | `Update`                      |
| `DELETE` destination                                   | `Delete`                      |
| `TRUNCATE` destination                                 | `Truncate`                    |
| `MERGE INTO` destination                               | `Merge`                       |

For `INSERT INTO destination SELECT * FROM source`, the destination lookup uses
insert intent and the source lookup uses read intent. A self-insert checks both.
Session planning caches providers by resolved table name and access, so a cached
insert provider cannot satisfy a read lookup. CTE references use their planned
relations without a catalog lookup; the tables inside their definitions are
resolved normally.

The default implementation calls the existing `table` method, preserving
existing catalogs. A protected catalog must also define conservative behavior
for legacy `table` calls: require read access, or reject them. Caller identity,
query context, stable object identities, and the mapping from operations to
privileges stay in the application.

Custom SQL planning contexts can implement
`ContextProvider::get_table_source_with_access`. `SessionContext` also exposes
`table_provider_with_access`, and `DataFrame::write_table` forwards the insert
operation. Catalog wrappers must forward access to their inner provider. For
remote catalogs, use `resolve_table_references_with_access` together with the
`AsyncSchemaProvider`, `AsyncCatalogProvider`, or `AsyncCatalogProviderList`
`resolve_with_access` method. Those caches retain operation distinctions and
only satisfy legacy table lookups when read access was resolved. Looking up a
cached table for an unresolved operation returns an error, preventing wrappers
from treating it as a missing table and falling back to another provider. Legacy
`resolve` methods retain their previous behavior and do not establish
operation-aware authorization. The existing FFI schema-provider ABI does not
carry operation intent; catalogs exposed through FFI use legacy lookup behavior.

This is a table-resolution hook, not a complete authorization system. DDL and
metadata lookups without a dedicated operation conservatively request `Read`;
that does not authorize creation, deletion, or alteration of catalog objects.
Applications must separately protect metadata enumeration, table functions,
externally supplied plans and providers, execution methods, and distributed
execution. Permission changes after planning and provider reuse also require
application policy; retain provider operation checks where needed.

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

    async fn table(&self, name: &str, _intent: datafusion::logical_expr::TableIntent) -> Result<Option<Arc<dyn TableProvider>>> {
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
#     async fn table(&self, name: &str, _intent: datafusion::logical_expr::TableIntent) -> Result<Option<Arc<dyn TableProvider>>> {
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

let table = schema_provider.table("users", datafusion::logical_expr::TableIntent::Read);
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
    async fn table(&self, name: &str, _intent: datafusion::logical_expr::TableIntent) -> Result<Option<Arc<dyn TableProvider>>> {
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

Authorizing catalogs need to distinguish operations to make fine-grained
permission decisions. A caller may be allowed to insert into a table without
being allowed to select from it. The planner still needs the destination schema.
`SchemaProvider::table(name, intent)` lets the catalog check the intended operation
before returning a provider or exposing its schema.

`TableIntent` describes why a table or view name is being resolved. It describes
the operation, not granted permissions. Applications supply caller identity and
map each intent to the privileges their policy requires.

| Table occurrence                                          | Intent                             |
| --------------------------------------------------------- | ---------------------------------- |
| `DESCRIBE table`, `SHOW COLUMNS`, `SHOW CREATE`           | `Describe`                         |
| Query source, including a subquery within a mutation      | `Read`                             |
| `INSERT INTO` destination                                 | `Insert(InsertOp::Append)`         |
| `INSERT OVERWRITE` destination                            | `Insert(InsertOp::Overwrite)`      |
| `REPLACE INTO` or `INSERT OR REPLACE` destination         | `Insert(InsertOp::Replace)`        |
| `UPDATE` destination                                      | `Update`                           |
| `DELETE` destination                                      | `Delete`                           |
| `TRUNCATE` destination                                    | `Truncate`                         |
| `MERGE INTO` destination                                  | `Merge { insert, update, delete }` |
| `CREATE TABLE`, external table, `SELECT INTO` destination | `CreateTable { or_replace }`       |
| `CREATE VIEW` destination                                 | `CreateView { or_replace }`        |
| `DROP TABLE` / `DROP VIEW` target                         | `DropTable` / `DropView`           |
| `CREATE INDEX` target table                               | `CreateIndex`                      |

Each operation includes the metadata needed to plan it. An insert lookup does
not request a separate `Describe` or `Read` intent. Replacement does not cause a
separate drop lookup. DataFusion defines no permission hierarchy: an application
can require multiple privileges for one intent or map several intents to one
privilege.

Merge flags include every action present in the statement, regardless of which
clauses will execute. Update, delete, and merge operations can inspect target
rows without a separate target `Read` lookup. Applications must map these intents
to all required privileges. Column-level permissions and row filtering require
additional enforcement. Explicit sources and subqueries request `Read`.

### Required lookup APIs

`SchemaProvider::table(name, intent)`, `AsyncSchemaProvider::table(name, intent)`,
and `ContextProvider::get_table_source(name, intent)` require an explicit
`TableIntent`. There is one lookup method per trait, with no name-only overload
or default intent. Existing implementations and callers must migrate.
Forwarding wrappers must pass intent unchanged to their inner provider. Catalogs
that do not enforce authorization can explicitly ignore the parameter.
Authorizing implementations should reject unknown intent variants.

All three trait methods return `Result<Option<...>>`:

- `Ok(Some(provider))`: resolution is authorized and the object exists.
- `Ok(None)`: resolution is authorized and the name is absent.
- `Err(...)`: authorization or lookup failed.

Authorize the name before returning `None`, including for `CREATE TABLE IF NOT EXISTS`. A creation policy can use privileges on the containing schema even when
there is no existing table. An error must not be treated as absence or suppressed
by `IF EXISTS` / `IF NOT EXISTS`. Direct `SqlToRel` callers receive the same DDL
lookup checks as built-in session planning.

`SessionContext::table_provider(name, intent)` also requires an explicit intent.
`SessionContext::table` constructs a read DataFrame and supplies `Read` itself;
`DataFrame::write_table` supplies the requested insert mode.

### Planning and caches

The SQL planner identifies the operation for each table occurrence. Built-in
session planning uses this information to fetch providers asynchronously before
synchronous SQL planning. Both the fetch and the subsequent cached lookup carry
the intent through to `SchemaProvider::table`.

For `INSERT INTO destination SELECT * FROM source`, the destination requests
`Insert(InsertOp::Append)` and the source requests `Read`. A self-insert checks
both intents. Built-in session planning caches providers by resolved table name
and intent, so a cached insert provider cannot satisfy a read lookup. CTE aliases
use their planned relations; tables inside their definitions resolve normally.

Remote catalogs can use `resolve_table_references_with_intent` together with
`AsyncSchemaProvider`, `AsyncCatalogProvider`, or `AsyncCatalogProviderList`'s
`resolve`. The remote-catalog example demonstrates this path. These are optional
snapshot helpers; built-in session planning performs its own prefetch and does
not call them.

`resolve` requires `(TableReference, TableIntent)` pairs and builds a snapshot
scoped to one caller and query. Every cache entry retains its authorized intent,
including authorized absence. Requesting any name/intent pair not resolved in
advance returns an error. Name and intent keys do not encode caller identity or
policy versions. Sharing snapshots across identities or retaining them after
permission changes can reuse stale authorization.

Snapshot name enumeration includes providers resolved for any intent. It does
not automatically limit discovery to objects resolved for `Describe`.

A shared metadata cache can still fetch a schema once by stable identity; it
must check authorization for the current caller and intent before returning the
cached provider. A self-insert can cause two remote lookups, one per intent;
implementations can share the metadata fetch while preserving both checks.
Metadata queries may enumerate more tables than occur explicitly in SQL. A
snapshot must resolve those `Describe` operations separately or provide a catalog
that can authorize them live; `resolve` is not a metadata-enumeration API.

### Describe and metadata visibility

`Describe` requests an object's metadata without reading its rows. It is used
for named table description and for underlying objects visited by
`information_schema.columns` and `information_schema.views`. The default
`SchemaProvider::table_type`, used by `information_schema.tables`, also requests
`Describe`. `DESCRIBE` of a query plans that query normally and retains its read
checks.

This does not automatically filter system-table results. Catalogs still control
`table_names`, schema and catalog names, existence checks, and `table_type`
overrides. Authorizing implementations should apply their visibility policy to
these APIs too; a common choice is to enumerate only names the caller may
describe. A denied lookup during information-schema traversal returns an error,
rather than silently hiding an object.

Querying a system table and inspecting the underlying objects are separate
operations. The built-in information-schema provider does not enforce policy on
the system table itself. Applications that restrict access to system tables must
enforce that restriction separately. Approval of `Describe` does not grant row
access, and approval of another operation need not make an object discoverable.

### Implicit file resolution

`SessionContext::enable_url_table()` enables file paths and URLs as table names;
the CLI enables this convenience by default. A dynamic schema first calls its
inner catalog with the original intent. An existing provider wins. Errors stop
resolution. Only `Ok(None)` may invoke `UrlTableFactory::try_new(url, intent)`.

Description and row operations retain their intent through the factory. DDL
intents inspect catalog entries without file fallback, including absent creation
destinations. The built-in listing factory returns `None` for missing files or
empty listings and propagates inference failures. Its shared options-inference
helper avoids an extra file listing to distinguish absence from failure.

The CLI configures object stores inside its file factory before inference. It
wraps an ordinary catalog instead of retrying an unsuccessful lookup through a
second wrapper. Registered table hits and catalog denials never enter that file
factory. Automatic store creation requires the typed `ObjectStoreNotFound`
registry error; permission, configuration, and discovery failures propagate.

File locations have a separate policy boundary. The built-in factories provide
convenience, not authorization. An authorizing application must check the
location and intent before calling them, or leave implicit file access disabled.
An approved absent catalog name alone is insufficient permission to open a file.

### Integration boundaries

- Optional implicit file resolution and explicit file APIs, external tables,
  COPY, and table functions require application policy for locations. The catalog
  intent does not provide a filesystem or object-store sandbox.
- Table DDL requests its actual intent during planning; built-in create/drop
  execution checks it again. Registration/deregistration APIs, schema/catalog
  commands, and custom DDL execution still need their own checks. Keep mutation
  authorization atomic with the mutation where necessary: a successful lookup is
  not a capability, and objects or policy can change between lookup and execution.
  `CreateIndex` is planned but its execution is application-owned.
- The current FFI schema-provider ABI cannot carry intent. A foreign adapter
  supports `Read` and rejects other intents; it does not downgrade them to reads.
  A same-library round trip retains the original provider and its intent handling.
  Supporting other intents across the FFI boundary needs an ABI extension.
- A returned provider is not restricted to the requested operation. Physical DML
  uses the provider embedded in the plan without another catalog check. Prepared
  plans, stored view plans, externally supplied providers/plans, table functions,
  direct file access, and distributed execution need separate enforcement.
  Retain execution-time checks when required for revocation or provider reuse.

This hook enables authorization during resolution. It does not install a policy
engine or make an embedding application secure automatically.

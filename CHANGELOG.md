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

## Unreleased changes in this fork

**Breaking changes:** table lookup now requires `TableIntent` across catalog,
SQL planner, and remote snapshot APIs. `ContextProvider::get_table_source` returns
an optional source; unresolved snapshot lookups fail instead of returning
absence. Implicit file lookup preserves intent and propagates failures;
`UrlTableFactory::try_new` now requires intent too. Missing object-store
registrations use a typed `ObjectStoreNotFound` error. Foreign FFI schema adapters
reject non-read intents. See the
[migration guide](docs/source/library-user-guide/upgrading/55.0.0.md#den-fork-operation-aware-table-resolution)
for required implementation changes and authorization boundaries.

## Upstream releases

Change logs for each release can be found [here](dev/changelog).

For older versions, see [apache/arrow/CHANGELOG.md](https://github.com/apache/arrow/blob/master/CHANGELOG.md).

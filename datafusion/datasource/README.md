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

# Apache DataFusion DataSource

[Apache DataFusion] is an extensible query execution framework, written in Rust, that uses [Apache Arrow] as its in-memory format.

This crate is a submodule of DataFusion that defines common DataSource related components like FileScanConfig, FileCompression etc.

Most projects should use the [`datafusion`] crate directly, which re-exports
this module. If you are already using the [`datafusion`] crate, there is no
reason to use this crate directly in your project as well.

[apache arrow]: https://arrow.apache.org/
[apache datafusion]: https://datafusion.apache.org/
[`datafusion`]: https://crates.io/crates/datafusion

## Application scan instrumentation

`instrumentation::DataSourceInstrumentationExtension` installs an optional
session hook around `DataSourceExec` partition execution. Applications can use
it to count output rows and batches, attach tracing context, and observe opening
errors, stream errors, completion, and cancellation.

The callback returns the scan output after source-level limits and batch
splitting. The hook receives the concrete source for optional source selection,
the task context, and a persistent metrics set exposed by the execution plan.
Metric names and export policy belong to the application. See the runnable
example in [`instrumentation`](src/instrumentation.rs).

The extension is resolved from the executing session, so it adds no plan nodes
or serialized fields. Applications executing plans in another process install
the extension in that process's session as well. Direct calls to `DataSource::open`
and custom execution plans that do not use `DataSourceExec` are outside its scope.

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

//! Operation intent supplied when resolving a table.

use crate::dml::InsertOp;

/// The operation for which a table is being resolved.
///
/// This describes intent, not a privilege model. Applications decide which
/// permissions each operation requires and supply caller identity through their
/// catalog or planner context. For example, an application may permit an insert
/// without granting permission to read the destination.
///
/// Resolution checks complement provider operation checks. They do not secure
/// externally supplied plans, arbitrary providers, metadata enumeration, or
/// distributed execution. DDL and metadata lookups that do not have a dedicated
/// variant conservatively use [`Self::Read`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TableAccess {
    /// Read a table, including a source relation within a write statement.
    Read,
    /// Insert into a table, retaining append, overwrite, and replace intent.
    Insert(InsertOp),
    /// Update rows in a table.
    Update,
    /// Delete rows from a table.
    Delete,
    /// Truncate a table.
    Truncate,
    /// Merge rows into a table.
    ///
    /// Unlike [`crate::logical_plan::WriteOp::MergeInto`], this intent does not
    /// contain expressions that require the target schema to have been resolved.
    Merge,
}

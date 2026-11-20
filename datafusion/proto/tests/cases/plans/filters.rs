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

//! `FilterExec`.

use super::{roundtrip_test, roundtrip_test_and_return};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::logical_expr::Operator;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::expressions::{NotExpr, binary, col, in_list, lit};
use datafusion::physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;
use datafusion_common::Result;
use datafusion_proto::physical_plan::{
    AsExecutionPlan, DefaultPhysicalExtensionCodec, DefaultPhysicalProtoConverter,
};
use datafusion_proto::protobuf::PhysicalPlanNode;
use std::sync::Arc;
use std::vec;

#[test]
fn roundtrip_filter_with_not_and_in_list() -> Result<()> {
    let field_a = Field::new("a", DataType::Boolean, false);
    let field_b = Field::new("b", DataType::Int64, false);
    let field_c = Field::new("c", DataType::Int64, false);
    let schema = Arc::new(Schema::new(vec![field_a, field_b, field_c]));
    let not = Arc::new(NotExpr::new(col("a", &schema)?));
    let in_list = in_list(
        col("b", &schema)?,
        vec![
            lit(ScalarValue::Int64(Some(1))),
            lit(ScalarValue::Int64(Some(2))),
        ],
        &false,
        schema.as_ref(),
    )?;
    let and = binary(not, Operator::And, in_list, &schema)?;
    roundtrip_test(Arc::new(FilterExec::try_new(
        and,
        Arc::new(EmptyExec::new(schema.clone())),
    )?))
}

#[test]
fn roundtrip_filter_with_fetch() -> Result<()> {
    let field_a = Field::new("a", DataType::Boolean, false);
    let field_b = Field::new("b", DataType::Int64, false);
    let schema = Arc::new(Schema::new(vec![field_a, field_b]));
    let predicate = col("a", &schema)?;
    let filter = FilterExecBuilder::new(predicate, Arc::new(EmptyExec::new(schema)))
        .with_fetch(Some(10))
        .build()?;
    assert_eq!(filter.fetch(), Some(10));
    roundtrip_test(Arc::new(filter))
}

/// `FilterExec` carries its `fetch` in a `uint32` field. The largest
/// representable value must round-trip exactly, and the first value above it
/// must be a named encode error rather than wrap to a different limit.
#[cfg(target_pointer_width = "64")]
#[test]
fn filter_fetch_across_the_wire_range() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Boolean, false)]));
    let ctx = SessionContext::new();
    let codec = DefaultPhysicalExtensionCodec {};
    let proto_converter = DefaultPhysicalProtoConverter {};

    let filter = FilterExecBuilder::new(
        col("a", &schema)?,
        Arc::new(EmptyExec::new(Arc::clone(&schema))),
    )
    .with_fetch(Some(u32::MAX as usize))
    .build()?;
    let decoded =
        roundtrip_test_and_return(Arc::new(filter), &ctx, &codec, &proto_converter)?;
    let decoded = decoded.downcast_ref::<FilterExec>().unwrap();
    assert_eq!(decoded.fetch(), Some(u32::MAX as usize));

    let filter = FilterExecBuilder::new(
        col("a", &schema)?,
        Arc::new(EmptyExec::new(Arc::clone(&schema))),
    )
    .with_fetch(Some(u32::MAX as usize + 1))
    .build()?;
    let err =
        PhysicalPlanNode::try_from_physical_plan(Arc::new(filter), &codec).unwrap_err();
    assert!(
        err.to_string().contains(&format!(
            "FilterExec: fetch value {} is out of range for the plan wire format",
            u32::MAX as usize + 1
        )),
        "unexpected error: {err}"
    );
    Ok(())
}

#[test]
fn roundtrip_filter_projection_states() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Boolean, false),
        Field::new("b", DataType::Int64, false),
    ]));
    let ctx = SessionContext::new();
    let codec = DefaultPhysicalExtensionCodec {};
    let proto_converter = DefaultPhysicalProtoConverter {};

    for projection in [None, Some(vec![]), Some(vec![0])] {
        let filter = FilterExecBuilder::new(
            col("a", &schema)?,
            Arc::new(EmptyExec::new(Arc::clone(&schema))),
        )
        .apply_projection(projection.clone())?
        .with_default_selectivity(37)
        .with_batch_size(1024)
        .with_fetch(Some(5))
        .build()?;

        let result =
            roundtrip_test_and_return(Arc::new(filter), &ctx, &codec, &proto_converter)?;
        let result = result.downcast_ref::<FilterExec>().unwrap();
        assert_eq!(result.projection().as_deref(), projection.as_deref());
        assert_eq!(result.default_selectivity(), 37);
        assert_eq!(result.batch_size(), 1024);
        assert_eq!(result.fetch(), Some(5));
    }

    Ok(())
}

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

use super::literal::name_struct_literals_like_siblings;
use crate::logical_plan::consumer::{SubstraitConsumer, from_substrait_rex_vec};
use datafusion::common::DFSchema;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::expr::InList;
use substrait::proto::expression::SingularOrList;

pub async fn from_singular_or_list(
    consumer: &impl SubstraitConsumer,
    expr: &SingularOrList,
    input_schema: &DFSchema,
) -> datafusion::common::Result<Expr> {
    let substrait_expr = expr.value.as_ref().unwrap();
    let substrait_list = expr.options.as_ref();
    let mut value = consumer
        .consume_expression(substrait_expr, input_schema)
        .await?;
    let mut list = from_substrait_rex_vec(consumer, substrait_list, input_schema).await?;
    // the value is compared with each list item, field names included
    let mut compared = std::iter::once(&mut value)
        .chain(list.iter_mut())
        .collect::<Vec<_>>();
    name_struct_literals_like_siblings(&mut compared, input_schema);
    Ok(Expr::InList(InList {
        expr: Box::new(value),
        list,
        negated: false,
    }))
}

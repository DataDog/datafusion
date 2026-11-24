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

use super::*;

#[test]
fn test_type_coercion_logical_op() -> Result<()> {
    test_coercion_binary_rule!(
        DataType::Boolean,
        DataType::Boolean,
        Operator::And,
        DataType::Boolean
    );

    test_coercion_binary_rule!(
        DataType::Boolean,
        DataType::Boolean,
        Operator::Or,
        DataType::Boolean
    );
    test_coercion_binary_rule!(
        DataType::Boolean,
        DataType::Null,
        Operator::And,
        DataType::Boolean
    );
    test_coercion_binary_rule!(
        DataType::Boolean,
        DataType::Null,
        Operator::Or,
        DataType::Boolean
    );
    test_coercion_binary_rule!(
        DataType::Null,
        DataType::Null,
        Operator::Or,
        DataType::Boolean
    );
    test_coercion_binary_rule!(
        DataType::Null,
        DataType::Null,
        Operator::And,
        DataType::Boolean
    );
    test_coercion_binary_rule!(
        DataType::Null,
        DataType::Boolean,
        Operator::And,
        DataType::Boolean
    );
    test_coercion_binary_rule!(
        DataType::Null,
        DataType::Boolean,
        Operator::Or,
        DataType::Boolean
    );
    Ok(())
}

fn struct_with_child(child: DataType, metadata: &[(&str, &str)]) -> DataType {
    let metadata = metadata
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    DataType::Struct(Fields::from(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", child, true).with_metadata(metadata),
    ]))
}

fn child_metadata(data_type: &DataType) -> &HashMap<String, String> {
    let DataType::Struct(fields) = data_type else {
        panic!("expected struct, got {data_type}");
    };
    fields[1].metadata()
}

#[test]
fn test_struct_coercion_keeps_metadata_of_children_null_in_every_input() -> Result<()> {
    let single = struct_with_child(DataType::Null, &[("k", "v")]);
    let coerced = try_type_union_resolution_with_struct(std::slice::from_ref(&single))?;
    assert_eq!(coerced, vec![single]);

    let first = struct_with_child(DataType::Null, &[("k", "first"), ("only_first", "1")]);
    let second =
        struct_with_child(DataType::Null, &[("k", "second"), ("only_second", "2")]);
    let coerced = try_type_union_resolution_with_struct(&[first, second])?;
    assert_eq!(
        child_metadata(&coerced[0]),
        &HashMap::from([
            ("k".to_string(), "first".to_string()),
            ("only_first".to_string(), "1".to_string()),
            ("only_second".to_string(), "2".to_string()),
        ])
    );
    Ok(())
}

#[test]
fn test_struct_coercion_prefers_typed_child_metadata_over_null_children() -> Result<()> {
    let null_child = struct_with_child(DataType::Null, &[("k", "null")]);
    let typed_child = struct_with_child(DataType::Int64, &[("k", "typed")]);
    for inputs in [
        [null_child.clone(), typed_child.clone()],
        [typed_child.clone(), null_child.clone()],
    ] {
        let coerced = try_type_union_resolution_with_struct(&inputs)?;
        assert_eq!(coerced[0], typed_child);
    }
    Ok(())
}

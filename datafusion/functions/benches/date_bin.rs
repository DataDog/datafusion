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

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, TimestampNanosecondArray, TimestampSecondArray};
use arrow::datatypes::Field;
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion_common::ScalarValue;
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs};
use datafusion_functions::datetime::date_bin;
use rand::prelude::*;

fn timestamps(rng: &mut StdRng) -> TimestampSecondArray {
    let mut seconds = vec![];
    for _ in 0..1000 {
        seconds.push(rng.random_range(0..1_000_000));
    }

    TimestampSecondArray::from(seconds)
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("date_bin_1000", |b| {
        let mut rng = StdRng::seed_from_u64(0);
        let timestamps_array = Arc::new(timestamps(&mut rng)) as ArrayRef;
        let batch_len = timestamps_array.len();
        let interval = ColumnarValue::Scalar(ScalarValue::new_interval_dt(0, 1_000_000));
        let timestamps = ColumnarValue::Array(timestamps_array);
        let udf = date_bin();
        let return_type = udf
            .return_type(&[interval.data_type(), timestamps.data_type()])
            .unwrap();
        let return_field = Arc::new(Field::new("f", return_type, true));

        let arg_fields = vec![
            Field::new("a", interval.data_type(), true).into(),
            Field::new("b", timestamps.data_type(), true).into(),
        ];
        let config_options = Arc::new(ConfigOptions::default());

        b.iter(|| {
            black_box(
                udf.invoke_with_args(ScalarFunctionArgs {
                    args: vec![interval.clone(), timestamps.clone()],
                    arg_fields: arg_fields.clone(),
                    number_rows: batch_len,
                    return_field: Arc::clone(&return_field),
                    config_options: Arc::clone(&config_options),
                })
                .expect("date_bin should work on valid values"),
            )
        })
    });

    c.bench_function("date_bin_nanoseconds_8192", |b| {
        // Ordered five-second samples, as in the captured four-hour metrics batch.
        let timestamps =
            TimestampNanosecondArray::from_iter_values((0..8192).map(|index| {
                1_787_068_800_000_000_000_i64 + (index % 2880) * 5_000_000_000
            }));
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::new_interval_mdn(0, 0, 30_000_000_000)),
            ColumnarValue::Array(Arc::new(timestamps)),
            ColumnarValue::Scalar(ScalarValue::TimestampNanosecond(
                Some(1_262_304_000_000_000_000),
                None,
            )),
        ];
        let arg_fields = args
            .iter()
            .map(|arg| Arc::new(Field::new("a", arg.data_type(), true)))
            .collect::<Vec<_>>();
        let udf = date_bin();
        let return_type = udf
            .return_type(
                &args
                    .iter()
                    .map(ColumnarValue::data_type)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let return_field = Arc::new(Field::new("f", return_type, true));
        let config_options = Arc::new(ConfigOptions::default());
        b.iter(|| {
            black_box(
                udf.invoke_with_args(ScalarFunctionArgs {
                    args: args.clone(),
                    arg_fields: arg_fields.clone(),
                    number_rows: 8192,
                    return_field: Arc::clone(&return_field),
                    config_options: Arc::clone(&config_options),
                })
                .expect("date_bin should work on valid nanosecond values"),
            )
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

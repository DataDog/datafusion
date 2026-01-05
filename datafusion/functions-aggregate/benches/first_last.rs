//! Comprehensive benchmarks for first_value/last_value aggregate functions.
//!
//! These benchmarks measure the index-finding performance which is the
//! core bottleneck for FIRST_VALUE/LAST_VALUE with ORDER BY.
//!
//! Run with: cargo bench --bench first_last

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampNanosecondArray};
use arrow::compute::{LexicographicalComparator, SortColumn, SortOptions};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use std::hint::black_box;
use std::sync::Arc;

// ============================================================================
// Data generators
// ============================================================================

fn generate_f64_array(size: usize) -> ArrayRef {
    let mut rng = rand::rng();
    let values: Vec<f64> = (0..size).map(|_| rng.random::<f64>()).collect();
    Arc::new(Float64Array::from(values))
}

fn generate_i64_array(size: usize) -> ArrayRef {
    let mut rng = rand::rng();
    let values: Vec<i64> = (0..size).map(|_| rng.random::<i64>()).collect();
    Arc::new(Int64Array::from(values))
}

fn generate_timestamp_array(size: usize) -> ArrayRef {
    let mut rng = rand::rng();
    let values: Vec<i64> = (0..size).map(|_| rng.random::<i64>()).collect();
    Arc::new(TimestampNanosecondArray::from(values))
}

fn generate_string_array(size: usize) -> ArrayRef {
    let mut rng = rand::rng();
    let values: Vec<String> = (0..size)
        .map(|_| {
            (0..10)
                .map(|_| rng.random_range(b'a'..=b'z') as char)
                .collect()
        })
        .collect();
    Arc::new(StringArray::from(values))
}

fn generate_f64_array_with_nulls(size: usize, null_ratio: f64) -> ArrayRef {
    let mut rng = rand::rng();
    let values: Vec<Option<f64>> = (0..size)
        .map(|_| {
            if rng.random::<f64>() < null_ratio {
                None
            } else {
                Some(rng.random::<f64>())
            }
        })
        .collect();
    Arc::new(Float64Array::from(values))
}

// ============================================================================
// Implementations to compare
// ============================================================================

/// Current implementation: LexicographicalComparator
fn find_min_index_lexicographic(ordering_array: &ArrayRef) -> Option<usize> {
    let sort_columns = vec![SortColumn {
        values: Arc::clone(ordering_array),
        options: Some(SortOptions::default()),
    }];
    
    let comparator = LexicographicalComparator::try_new(&sort_columns).ok()?;
    (0..ordering_array.len()).min_by(|&a, &b| comparator.compare(a, b))
}

fn find_max_index_lexicographic(ordering_array: &ArrayRef) -> Option<usize> {
    let sort_columns = vec![SortColumn {
        values: Arc::clone(ordering_array),
        options: Some(SortOptions::default()),
    }];
    
    let comparator = LexicographicalComparator::try_new(&sort_columns).ok()?;
    (0..ordering_array.len()).max_by(|&a, &b| comparator.compare(a, b))
}

/// Optimized: Type-specialized for Float64
fn find_min_index_f64(array: &ArrayRef) -> Option<usize> {
    let typed_array = array.as_any().downcast_ref::<Float64Array>()?;
    let mut min_idx: Option<usize> = None;
    let mut min_value: Option<f64> = None;

    for i in 0..typed_array.len() {
        if typed_array.is_null(i) {
            continue;
        }
        let current = typed_array.value(i);
        match min_value {
            None => {
                min_idx = Some(i);
                min_value = Some(current);
            }
            Some(min_val) if current < min_val => {
                min_idx = Some(i);
                min_value = Some(current);
            }
            _ => {}
        }
    }
    min_idx
}

fn find_max_index_f64(array: &ArrayRef) -> Option<usize> {
    let typed_array = array.as_any().downcast_ref::<Float64Array>()?;
    let mut max_idx: Option<usize> = None;
    let mut max_value: Option<f64> = None;

    for i in 0..typed_array.len() {
        if typed_array.is_null(i) {
            continue;
        }
        let current = typed_array.value(i);
        match max_value {
            None => {
                max_idx = Some(i);
                max_value = Some(current);
            }
            Some(max_val) if current > max_val => {
                max_idx = Some(i);
                max_value = Some(current);
            }
            _ => {}
        }
    }
    max_idx
}

fn find_min_index_i64(array: &ArrayRef) -> Option<usize> {
    let typed_array = array.as_any().downcast_ref::<Int64Array>()?;
    let mut min_idx: Option<usize> = None;
    let mut min_value: Option<i64> = None;

    for i in 0..typed_array.len() {
        if typed_array.is_null(i) {
            continue;
        }
        let current = typed_array.value(i);
        match min_value {
            None => {
                min_idx = Some(i);
                min_value = Some(current);
            }
            Some(min_val) if current < min_val => {
                min_idx = Some(i);
                min_value = Some(current);
            }
            _ => {}
        }
    }
    min_idx
}

fn find_min_index_timestamp(array: &ArrayRef) -> Option<usize> {
    let typed_array = array.as_any().downcast_ref::<TimestampNanosecondArray>()?;
    let mut min_idx: Option<usize> = None;
    let mut min_value: Option<i64> = None;

    for i in 0..typed_array.len() {
        if typed_array.is_null(i) {
            continue;
        }
        let current = typed_array.value(i);
        match min_value {
            None => {
                min_idx = Some(i);
                min_value = Some(current);
            }
            Some(min_val) if current < min_val => {
                min_idx = Some(i);
                min_value = Some(current);
            }
            _ => {}
        }
    }
    min_idx
}

// ============================================================================
// Benchmarks
// ============================================================================

/// Benchmark: Scaling with array size for Float64
fn bench_scaling_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_f64");

    // Test various sizes from 100 to 10M
    for size in [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000] {
        let array = generate_f64_array(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("lexicographic", size),
            &array,
            |b, array| {
                b.iter(|| black_box(find_min_index_lexicographic(array)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("typed", size),
            &array,
            |b, array| {
                b.iter(|| black_box(find_min_index_f64(array)));
            },
        );
    }

    group.finish();
}

/// Benchmark: Scaling with array size for Int64
fn bench_scaling_i64(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_i64");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000] {
        let array = generate_i64_array(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("lexicographic", size),
            &array,
            |b, array| {
                b.iter(|| black_box(find_min_index_lexicographic(array)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("typed", size),
            &array,
            |b, array| {
                b.iter(|| black_box(find_min_index_i64(array)));
            },
        );
    }

    group.finish();
}

/// Benchmark: Timestamp ordering (common for time-series data)
fn bench_timestamp_ordering(c: &mut Criterion) {
    let mut group = c.benchmark_group("timestamp_ordering");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let array = generate_timestamp_array(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("lexicographic", size),
            &array,
            |b, array| {
                b.iter(|| black_box(find_min_index_lexicographic(array)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("typed", size),
            &array,
            |b, array| {
                b.iter(|| black_box(find_min_index_timestamp(array)));
            },
        );
    }

    group.finish();
}

/// Benchmark: String ordering (baseline, no typed optimization)
fn bench_string_ordering(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_ordering");

    for size in [1_000, 10_000, 100_000] {
        let array = generate_string_array(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("lexicographic", size),
            &array,
            |b, array| {
                b.iter(|| black_box(find_min_index_lexicographic(array)));
            },
        );
    }

    group.finish();
}

/// Benchmark: Full LAST_VALUE pattern (max_by simulation)
fn bench_last_value_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("last_value_pattern");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let value_array = generate_string_array(size);
        let ordering_array = generate_f64_array(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("lexicographic", size),
            &(&value_array, &ordering_array),
            |b, (value, ordering)| {
                b.iter(|| {
                    let idx = find_max_index_lexicographic(ordering);
                    black_box(idx.map(|i| value.as_any().downcast_ref::<StringArray>().unwrap().value(i)))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("typed", size),
            &(&value_array, &ordering_array),
            |b, (value, ordering)| {
                b.iter(|| {
                    let idx = find_max_index_f64(ordering);
                    black_box(idx.map(|i| value.as_any().downcast_ref::<StringArray>().unwrap().value(i)))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Impact of null values in ordering array
fn bench_with_nulls(c: &mut Criterion) {
    let mut group = c.benchmark_group("with_nulls");
    let size = 100_000;

    for null_ratio in [0.0, 0.1, 0.25, 0.5] {
        let array = generate_f64_array_with_nulls(size, null_ratio);
        let label = format!("{}pct_nulls", (null_ratio * 100.0) as u32);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("lexicographic", &label),
            &array,
            |b, array| {
                b.iter(|| black_box(find_min_index_lexicographic(array)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("typed", &label),
            &array,
            |b, array| {
                b.iter(|| black_box(find_min_index_f64(array)));
            },
        );
    }

    group.finish();
}

/// Benchmark: Multiple consecutive calls (simulating multiple aggregations)
fn bench_multiple_calls(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_calls");
    let size = 100_000;
    
    for num_calls in [1, 3, 6, 9, 12] {
        let arrays: Vec<ArrayRef> = (0..num_calls).map(|_| generate_f64_array(size)).collect();
        group.throughput(Throughput::Elements((size * num_calls) as u64));

        group.bench_with_input(
            BenchmarkId::new("lexicographic", num_calls),
            &arrays,
            |b, arrays| {
                b.iter(|| {
                    for array in arrays {
                        black_box(find_max_index_lexicographic(array));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("typed", num_calls),
            &arrays,
            |b, arrays| {
                b.iter(|| {
                    for array in arrays {
                        black_box(find_max_index_f64(array));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_scaling_f64,
    bench_scaling_i64,
    bench_timestamp_ordering,
    bench_string_ordering,
    bench_last_value_pattern,
    bench_with_nulls,
    bench_multiple_calls,
);
criterion_main!(benches);

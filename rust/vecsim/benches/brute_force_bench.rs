//! Benchmarks for BruteForce index operations.
//!
//! Run with: cargo bench --bench brute_force_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use vecsim::distance::Metric;
use vecsim::index::brute_force::{BruteForceMulti, BruteForceParams, BruteForceSingle};
use vecsim::index::VecSimIndex;

const DIM: usize = 128;

/// Generate random vectors for benchmarking.
fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Benchmark adding vectors to BruteForceSingle.
fn bench_bf_single_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_single_add");

    for size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_vectors(size, DIM);
            b.iter(|| {
                let params = BruteForceParams::new(DIM, Metric::L2);
                let mut index = BruteForceSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark adding vectors to BruteForceMulti.
fn bench_bf_multi_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_multi_add");

    for size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_vectors(size, DIM);
            b.iter(|| {
                let params = BruteForceParams::new(DIM, Metric::L2);
                let mut index = BruteForceMulti::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    // Use fewer labels to have multiple vectors per label
                    index.add_vector(black_box(v), (i % 100) as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark top-k queries on BruteForceSingle.
fn bench_bf_single_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_single_topk");

    for size in [100, 1000, 10000] {
        let vectors = generate_vectors(size, DIM);
        let query = generate_vectors(1, DIM).pop().unwrap();

        let params = BruteForceParams::new(DIM, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(10), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark top-k queries with varying k values.
fn bench_bf_single_topk_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_single_topk_k");

    let size = 10000;
    let vectors = generate_vectors(size, DIM);
    let query = generate_vectors(1, DIM).pop().unwrap();

    let params = BruteForceParams::new(DIM, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    for k in [1, 10, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(k), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark range queries on BruteForceSingle.
fn bench_bf_single_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_single_range");

    for size in [100, 1000, 10000] {
        let vectors = generate_vectors(size, DIM);
        let query = generate_vectors(1, DIM).pop().unwrap();

        let params = BruteForceParams::new(DIM, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        // Use a radius that returns ~10% of vectors
        let radius = 10.0;

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| index.range_query(black_box(&query), black_box(radius), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark delete operations on BruteForceSingle.
fn bench_bf_single_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_single_delete");

    for size in [100, 1000, 5000] {
        let vectors = generate_vectors(size, DIM);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let params = BruteForceParams::new(DIM, Metric::L2);
                    let mut index = BruteForceSingle::<f32>::new(params);
                    for (i, v) in vectors.iter().enumerate() {
                        index.add_vector(v, i as u64).unwrap();
                    }
                    index
                },
                |mut index| {
                    // Delete half the vectors
                    for i in (0..size).step_by(2) {
                        index.delete_vector(i as u64).unwrap();
                    }
                    index
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark different distance metrics.
fn bench_bf_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_metrics_5000");

    let size = 5000;
    let vectors = generate_vectors(size, DIM);
    let query = generate_vectors(1, DIM).pop().unwrap();

    for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
        let params = BruteForceParams::new(DIM, metric);
        let mut index = BruteForceSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        let metric_name = match metric {
            Metric::L2 => "L2",
            Metric::InnerProduct => "IP",
            Metric::Cosine => "Cosine",
        };

        group.bench_function(metric_name, |b| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(10), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark different vector dimensions.
fn bench_bf_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_dimensions_1000");

    let size = 1000;

    for dim in [32, 128, 512, 1024] {
        let vectors = generate_vectors(size, dim);
        let query = generate_vectors(1, dim).pop().unwrap();

        let params = BruteForceParams::new(dim, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(10), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark serialization round-trip.
fn bench_bf_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_serialization");

    for size in [1000, 5000, 10000] {
        let vectors = generate_vectors(size, DIM);

        let params = BruteForceParams::new(DIM, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("save", size), &size, |b, _| {
            b.iter(|| {
                let mut buffer = Vec::new();
                index.save(black_box(&mut buffer)).unwrap();
                buffer
            });
        });

        let mut buffer = Vec::new();
        index.save(&mut buffer).unwrap();

        group.bench_with_input(BenchmarkId::new("load", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = std::io::Cursor::new(&buffer);
                BruteForceSingle::<f32>::load(black_box(&mut cursor)).unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bf_single_add,
    bench_bf_multi_add,
    bench_bf_single_topk,
    bench_bf_single_topk_varying_k,
    bench_bf_single_range,
    bench_bf_single_delete,
    bench_bf_metrics,
    bench_bf_dimensions,
    bench_bf_serialization,
);

criterion_main!(benches);

//! Benchmarks for HNSW index operations.
//!
//! Run with: cargo bench --bench hnsw_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use vecsim::distance::Metric;
use vecsim::index::hnsw::{HnswMulti, HnswParams, HnswSingle};
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

/// Benchmark adding vectors to HnswSingle.
fn bench_hnsw_single_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_single_add");
    group.sample_size(10); // HNSW add is slow, reduce samples

    for size in [100, 500, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_vectors(size, DIM);
            b.iter(|| {
                let params = HnswParams::new(DIM, Metric::L2)
                    .with_m(16)
                    .with_ef_construction(100);
                let mut index = HnswSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark adding vectors with varying M parameter.
fn bench_hnsw_add_varying_m(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_add_m");
    group.sample_size(10);

    let size = 500;
    let vectors = generate_vectors(size, DIM);

    for m in [4, 8, 16, 32] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(m), &m, |b, &m| {
            b.iter(|| {
                let params = HnswParams::new(DIM, Metric::L2)
                    .with_m(m)
                    .with_ef_construction(100);
                let mut index = HnswSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark adding vectors with varying ef_construction.
fn bench_hnsw_add_varying_ef(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_add_ef_construction");
    group.sample_size(10);

    let size = 500;
    let vectors = generate_vectors(size, DIM);

    for ef in [50, 100, 200, 400] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(ef), &ef, |b, &ef| {
            b.iter(|| {
                let params = HnswParams::new(DIM, Metric::L2)
                    .with_m(16)
                    .with_ef_construction(ef);
                let mut index = HnswSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark top-k queries on HnswSingle.
fn bench_hnsw_single_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_single_topk");

    for size in [1000, 5000, 10000] {
        let vectors = generate_vectors(size, DIM);
        let query = generate_vectors(1, DIM).pop().unwrap();

        let params = HnswParams::new(DIM, Metric::L2)
            .with_m(16)
            .with_ef_construction(100)
            .with_ef_runtime(50);
        let mut index = HnswSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(10), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark top-k queries with varying ef_runtime.
fn bench_hnsw_topk_varying_ef_runtime(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_topk_ef_runtime");

    let size = 10000;
    let vectors = generate_vectors(size, DIM);
    let query = generate_vectors(1, DIM).pop().unwrap();

    let params = HnswParams::new(DIM, Metric::L2)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_runtime(10); // Will be overridden per query
    let mut index = HnswSingle::<f32>::new(params);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    for ef in [10, 50, 100, 200] {
        let query_params = vecsim::query::QueryParams::new().with_ef_runtime(ef);
        group.bench_with_input(BenchmarkId::from_parameter(ef), &ef, |b, _| {
            b.iter(|| {
                index
                    .top_k_query(black_box(&query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark top-k queries with varying k values.
fn bench_hnsw_topk_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_topk_k");

    let size = 10000;
    let vectors = generate_vectors(size, DIM);
    let query = generate_vectors(1, DIM).pop().unwrap();

    let params = HnswParams::new(DIM, Metric::L2)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_runtime(100);
    let mut index = HnswSingle::<f32>::new(params);
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

/// Benchmark range queries on HnswSingle.
fn bench_hnsw_single_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_single_range");

    for size in [1000, 5000, 10000] {
        let vectors = generate_vectors(size, DIM);
        let query = generate_vectors(1, DIM).pop().unwrap();

        let params = HnswParams::new(DIM, Metric::L2)
            .with_m(16)
            .with_ef_construction(100)
            .with_ef_runtime(100);
        let mut index = HnswSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        let radius = 10.0;

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| index.range_query(black_box(&query), black_box(radius), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark delete operations on HnswSingle.
fn bench_hnsw_single_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_single_delete");
    group.sample_size(10);

    for size in [500, 1000, 2000] {
        let vectors = generate_vectors(size, DIM);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let params = HnswParams::new(DIM, Metric::L2)
                        .with_m(16)
                        .with_ef_construction(100);
                    let mut index = HnswSingle::<f32>::new(params);
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

/// Benchmark HnswMulti with multiple vectors per label.
fn bench_hnsw_multi_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_multi_add");
    group.sample_size(10);

    for size in [100, 500, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_vectors(size, DIM);
            b.iter(|| {
                let params = HnswParams::new(DIM, Metric::L2)
                    .with_m(16)
                    .with_ef_construction(100);
                let mut index = HnswMulti::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    // Use fewer labels to have multiple vectors per label
                    index.add_vector(black_box(v), (i % 50) as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark different distance metrics for HNSW.
fn bench_hnsw_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_metrics_5000");

    let size = 5000;
    let vectors = generate_vectors(size, DIM);
    let query = generate_vectors(1, DIM).pop().unwrap();

    for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
        let params = HnswParams::new(DIM, metric)
            .with_m(16)
            .with_ef_construction(100)
            .with_ef_runtime(50);
        let mut index = HnswSingle::<f32>::new(params);
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

/// Benchmark different vector dimensions for HNSW.
fn bench_hnsw_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_dimensions_1000");

    let size = 1000;

    for dim in [32, 128, 512] {
        let vectors = generate_vectors(size, dim);
        let query = generate_vectors(1, dim).pop().unwrap();

        let params = HnswParams::new(dim, Metric::L2)
            .with_m(16)
            .with_ef_construction(100)
            .with_ef_runtime(50);
        let mut index = HnswSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(10), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark serialization round-trip for HNSW.
fn bench_hnsw_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_serialization");
    group.sample_size(10);

    for size in [1000, 5000] {
        let vectors = generate_vectors(size, DIM);

        let params = HnswParams::new(DIM, Metric::L2)
            .with_m(16)
            .with_ef_construction(100);
        let mut index = HnswSingle::<f32>::new(params);
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
                HnswSingle::<f32>::load(black_box(&mut cursor)).unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_single_add,
    bench_hnsw_add_varying_m,
    bench_hnsw_add_varying_ef,
    bench_hnsw_single_topk,
    bench_hnsw_topk_varying_ef_runtime,
    bench_hnsw_topk_varying_k,
    bench_hnsw_single_range,
    bench_hnsw_single_delete,
    bench_hnsw_multi_add,
    bench_hnsw_metrics,
    bench_hnsw_dimensions,
    bench_hnsw_serialization,
);

criterion_main!(benches);

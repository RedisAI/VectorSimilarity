//! Benchmarks for SVS (Vamana) index operations.
//!
//! Run with: cargo bench --bench svs_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use vecsim::distance::Metric;
use vecsim::index::svs::{SvsMulti, SvsParams, SvsSingle};
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

/// Benchmark adding vectors to SvsSingle.
fn bench_svs_single_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_single_add");
    group.sample_size(10); // SVS add is slow due to graph construction

    for size in [100, 500, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_vectors(size, DIM);
            b.iter(|| {
                let params = SvsParams::new(DIM, Metric::L2)
                    .with_graph_degree(32)
                    .with_construction_l(100)
                    .with_two_pass(false); // Single pass for faster construction
                let mut index = SvsSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark adding vectors with varying graph degree (R).
fn bench_svs_add_varying_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_add_degree");
    group.sample_size(10);

    let size = 500;
    let vectors = generate_vectors(size, DIM);

    for degree in [16, 32, 48, 64] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(degree), &degree, |b, &degree| {
            b.iter(|| {
                let params = SvsParams::new(DIM, Metric::L2)
                    .with_graph_degree(degree)
                    .with_construction_l(100)
                    .with_two_pass(false);
                let mut index = SvsSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark adding vectors with varying alpha parameter.
fn bench_svs_add_varying_alpha(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_add_alpha");
    group.sample_size(10);

    let size = 500;
    let vectors = generate_vectors(size, DIM);

    for alpha in [1.0f32, 1.1, 1.2, 1.4] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(alpha), &alpha, |b, &alpha| {
            b.iter(|| {
                let params = SvsParams::new(DIM, Metric::L2)
                    .with_graph_degree(32)
                    .with_alpha(alpha)
                    .with_construction_l(100)
                    .with_two_pass(false);
                let mut index = SvsSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark adding vectors with varying construction window size (L).
fn bench_svs_add_varying_construction_l(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_add_construction_l");
    group.sample_size(10);

    let size = 500;
    let vectors = generate_vectors(size, DIM);

    for l in [50, 100, 200, 400] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(l), &l, |b, &l| {
            b.iter(|| {
                let params = SvsParams::new(DIM, Metric::L2)
                    .with_graph_degree(32)
                    .with_construction_l(l)
                    .with_two_pass(false);
                let mut index = SvsSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark two-pass vs single-pass construction.
fn bench_svs_two_pass_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_two_pass");
    group.sample_size(10);

    let size = 500;
    let vectors = generate_vectors(size, DIM);

    for two_pass in [false, true] {
        let label = if two_pass { "two_pass" } else { "single_pass" };
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(label, |b| {
            b.iter(|| {
                let params = SvsParams::new(DIM, Metric::L2)
                    .with_graph_degree(32)
                    .with_construction_l(100)
                    .with_two_pass(two_pass);
                let mut index = SvsSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark top-k queries on SvsSingle.
fn bench_svs_single_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_single_topk");

    for size in [1000, 5000, 10000] {
        let vectors = generate_vectors(size, DIM);
        let query = generate_vectors(1, DIM).pop().unwrap();

        let params = SvsParams::new(DIM, Metric::L2)
            .with_graph_degree(32)
            .with_construction_l(100)
            .with_search_l(50)
            .with_two_pass(false);
        let mut index = SvsSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(10), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark top-k queries with varying default search window size.
/// Note: SVS search_l is set at index creation time, not at query time.
fn bench_svs_topk_varying_search_l(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_topk_search_l");
    group.sample_size(10); // Need to rebuild index for each search_l

    let size = 5000;
    let vectors = generate_vectors(size, DIM);
    let query = generate_vectors(1, DIM).pop().unwrap();

    for search_l in [10, 50, 100, 200] {
        // SVS search_l is an index parameter, so we need to rebuild for each value
        let params = SvsParams::new(DIM, Metric::L2)
            .with_graph_degree(32)
            .with_construction_l(100)
            .with_search_l(search_l)
            .with_two_pass(false);
        let mut index = SvsSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(search_l), &search_l, |b, _| {
            b.iter(|| {
                index
                    .top_k_query(black_box(&query), black_box(10), None)
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark top-k queries with varying k values.
fn bench_svs_topk_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_topk_k");

    let size = 10000;
    let vectors = generate_vectors(size, DIM);
    let query = generate_vectors(1, DIM).pop().unwrap();

    let params = SvsParams::new(DIM, Metric::L2)
        .with_graph_degree(32)
        .with_construction_l(100)
        .with_search_l(100)
        .with_two_pass(false);
    let mut index = SvsSingle::<f32>::new(params);
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

/// Benchmark range queries on SvsSingle.
fn bench_svs_single_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_single_range");

    for size in [1000, 5000, 10000] {
        let vectors = generate_vectors(size, DIM);
        let query = generate_vectors(1, DIM).pop().unwrap();

        let params = SvsParams::new(DIM, Metric::L2)
            .with_graph_degree(32)
            .with_construction_l(100)
            .with_search_l(100)
            .with_two_pass(false);
        let mut index = SvsSingle::<f32>::new(params);
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

/// Benchmark delete operations on SvsSingle.
fn bench_svs_single_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_single_delete");
    group.sample_size(10);

    for size in [500, 1000, 2000] {
        let vectors = generate_vectors(size, DIM);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let params = SvsParams::new(DIM, Metric::L2)
                        .with_graph_degree(32)
                        .with_construction_l(100)
                        .with_two_pass(false);
                    let mut index = SvsSingle::<f32>::new(params);
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

/// Benchmark SvsMulti with multiple vectors per label.
fn bench_svs_multi_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_multi_add");
    group.sample_size(10);

    for size in [100, 500, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_vectors(size, DIM);
            b.iter(|| {
                let params = SvsParams::new(DIM, Metric::L2)
                    .with_graph_degree(32)
                    .with_construction_l(100)
                    .with_two_pass(false);
                let mut index = SvsMulti::<f32>::new(params);
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

/// Benchmark different distance metrics for SVS.
fn bench_svs_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_metrics_5000");

    let size = 5000;
    let vectors = generate_vectors(size, DIM);
    let query = generate_vectors(1, DIM).pop().unwrap();

    for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
        let params = SvsParams::new(DIM, metric)
            .with_graph_degree(32)
            .with_construction_l(100)
            .with_search_l(50)
            .with_two_pass(false);
        let mut index = SvsSingle::<f32>::new(params);
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

/// Benchmark different vector dimensions for SVS.
fn bench_svs_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("svs_dimensions_1000");

    let size = 1000;

    for dim in [32, 128, 512] {
        let vectors = generate_vectors(size, dim);
        let query = generate_vectors(1, dim).pop().unwrap();

        let params = SvsParams::new(dim, Metric::L2)
            .with_graph_degree(32)
            .with_construction_l(100)
            .with_search_l(50)
            .with_two_pass(false);
        let mut index = SvsSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(10), None).unwrap());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_svs_single_add,
    bench_svs_add_varying_degree,
    bench_svs_add_varying_alpha,
    bench_svs_add_varying_construction_l,
    bench_svs_two_pass_construction,
    bench_svs_single_topk,
    bench_svs_topk_varying_search_l,
    bench_svs_topk_varying_k,
    bench_svs_single_range,
    bench_svs_single_delete,
    bench_svs_multi_add,
    bench_svs_metrics,
    bench_svs_dimensions,
);

criterion_main!(benches);

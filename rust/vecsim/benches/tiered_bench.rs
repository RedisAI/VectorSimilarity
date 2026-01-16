//! Benchmarks for TieredIndex operations.
//!
//! Run with: cargo bench --bench tiered_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use vecsim::distance::Metric;
use vecsim::index::brute_force::{BruteForceParams, BruteForceSingle};
use vecsim::index::hnsw::{HnswParams, HnswSingle};
use vecsim::index::tiered::{TieredParams, TieredSingle, WriteMode};
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

/// Benchmark adding vectors to TieredSingle in async mode.
fn bench_tiered_add_async(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered_add_async");

    for size in [100, 1000, 5000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_vectors(size, DIM);
            b.iter(|| {
                let params = TieredParams::new(DIM, Metric::L2)
                    .with_flat_buffer_limit(size * 2) // Keep in async mode
                    .with_write_mode(WriteMode::Async);
                let mut index = TieredSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark adding vectors to TieredSingle in in-place mode.
fn bench_tiered_add_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered_add_inplace");

    for size in [100, 1000, 5000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_vectors(size, DIM);
            b.iter(|| {
                let params = TieredParams::new(DIM, Metric::L2)
                    .with_write_mode(WriteMode::InPlace)
                    .with_m(16)
                    .with_ef_construction(100);
                let mut index = TieredSingle::<f32>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

/// Benchmark top-k queries on TieredSingle with vectors in flat buffer only.
fn bench_tiered_query_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered_query_flat");

    for size in [100, 1000, 5000] {
        let vectors = generate_vectors(size, DIM);
        let query = generate_vectors(1, DIM).pop().unwrap();

        let params = TieredParams::new(DIM, Metric::L2)
            .with_flat_buffer_limit(size * 2);
        let mut index = TieredSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(10), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark top-k queries on TieredSingle with vectors in HNSW only.
fn bench_tiered_query_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered_query_hnsw");

    for size in [100, 1000, 5000] {
        let vectors = generate_vectors(size, DIM);
        let query = generate_vectors(1, DIM).pop().unwrap();

        let params = TieredParams::new(DIM, Metric::L2)
            .with_m(16)
            .with_ef_construction(100)
            .with_ef_runtime(50);
        let mut index = TieredSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }
        index.flush().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(10), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark top-k queries on TieredSingle with vectors in both tiers.
fn bench_tiered_query_both(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered_query_both_tiers");

    for size in [100, 1000, 5000] {
        let vectors = generate_vectors(size, DIM);
        let query = generate_vectors(1, DIM).pop().unwrap();

        let params = TieredParams::new(DIM, Metric::L2)
            .with_flat_buffer_limit(size / 2)
            .with_m(16)
            .with_ef_construction(100)
            .with_ef_runtime(50);
        let mut index = TieredSingle::<f32>::new(params);

        // Add half to flat, flush to HNSW
        for (i, v) in vectors.iter().take(size / 2).enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }
        index.flush().unwrap();

        // Add other half to flat
        for (i, v) in vectors.iter().skip(size / 2).enumerate() {
            index.add_vector(v, (size / 2 + i) as u64).unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| index.top_k_query(black_box(&query), black_box(10), None).unwrap());
        });
    }

    group.finish();
}

/// Benchmark flush operation (migrating from flat to HNSW).
fn bench_tiered_flush(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered_flush");

    for size in [100, 500, 1000] {
        let vectors = generate_vectors(size, DIM);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let params = TieredParams::new(DIM, Metric::L2)
                        .with_flat_buffer_limit(size * 2)
                        .with_m(16)
                        .with_ef_construction(100);
                    let mut index = TieredSingle::<f32>::new(params);
                    for (i, v) in vectors.iter().enumerate() {
                        index.add_vector(v, i as u64).unwrap();
                    }
                    index
                },
                |mut index| {
                    index.flush().unwrap();
                    index
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Compare TieredSingle query performance against BruteForce and HNSW.
fn bench_comparison_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_query_5000");

    let size = 5000;
    let vectors = generate_vectors(size, DIM);
    let query = generate_vectors(1, DIM).pop().unwrap();

    // BruteForce
    let bf_params = BruteForceParams::new(DIM, Metric::L2);
    let mut bf_index = BruteForceSingle::<f32>::new(bf_params);
    for (i, v) in vectors.iter().enumerate() {
        bf_index.add_vector(v, i as u64).unwrap();
    }

    group.bench_function("brute_force", |b| {
        b.iter(|| bf_index.top_k_query(black_box(&query), black_box(10), None).unwrap());
    });

    // HNSW
    let hnsw_params = HnswParams::new(DIM, Metric::L2)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_runtime(50);
    let mut hnsw_index = HnswSingle::<f32>::new(hnsw_params);
    for (i, v) in vectors.iter().enumerate() {
        hnsw_index.add_vector(v, i as u64).unwrap();
    }

    group.bench_function("hnsw", |b| {
        b.iter(|| hnsw_index.top_k_query(black_box(&query), black_box(10), None).unwrap());
    });

    // Tiered (all in HNSW after flush)
    let tiered_params = TieredParams::new(DIM, Metric::L2)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_runtime(50);
    let mut tiered_index = TieredSingle::<f32>::new(tiered_params);
    for (i, v) in vectors.iter().enumerate() {
        tiered_index.add_vector(v, i as u64).unwrap();
    }
    tiered_index.flush().unwrap();

    group.bench_function("tiered_flushed", |b| {
        b.iter(|| tiered_index.top_k_query(black_box(&query), black_box(10), None).unwrap());
    });

    // Tiered (half in each tier)
    let tiered_params2 = TieredParams::new(DIM, Metric::L2)
        .with_flat_buffer_limit(size / 2)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_runtime(50);
    let mut tiered_index2 = TieredSingle::<f32>::new(tiered_params2);
    for (i, v) in vectors.iter().take(size / 2).enumerate() {
        tiered_index2.add_vector(v, i as u64).unwrap();
    }
    tiered_index2.flush().unwrap();
    for (i, v) in vectors.iter().skip(size / 2).enumerate() {
        tiered_index2.add_vector(v, (size / 2 + i) as u64).unwrap();
    }

    group.bench_function("tiered_both_tiers", |b| {
        b.iter(|| tiered_index2.top_k_query(black_box(&query), black_box(10), None).unwrap());
    });

    group.finish();
}

/// Compare add performance across index types.
fn bench_comparison_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_add_1000");

    let size = 1000;
    let vectors = generate_vectors(size, DIM);

    group.throughput(Throughput::Elements(size as u64));

    // BruteForce
    group.bench_function("brute_force", |b| {
        b.iter(|| {
            let params = BruteForceParams::new(DIM, Metric::L2);
            let mut index = BruteForceSingle::<f32>::new(params);
            for (i, v) in vectors.iter().enumerate() {
                index.add_vector(black_box(v), i as u64).unwrap();
            }
            index
        });
    });

    // HNSW
    group.bench_function("hnsw", |b| {
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

    // Tiered (async mode - writes to flat buffer)
    group.bench_function("tiered_async", |b| {
        b.iter(|| {
            let params = TieredParams::new(DIM, Metric::L2)
                .with_flat_buffer_limit(size * 2)
                .with_write_mode(WriteMode::Async);
            let mut index = TieredSingle::<f32>::new(params);
            for (i, v) in vectors.iter().enumerate() {
                index.add_vector(black_box(v), i as u64).unwrap();
            }
            index
        });
    });

    // Tiered (in-place mode - writes directly to HNSW)
    group.bench_function("tiered_inplace", |b| {
        b.iter(|| {
            let params = TieredParams::new(DIM, Metric::L2)
                .with_write_mode(WriteMode::InPlace)
                .with_m(16)
                .with_ef_construction(100);
            let mut index = TieredSingle::<f32>::new(params);
            for (i, v) in vectors.iter().enumerate() {
                index.add_vector(black_box(v), i as u64).unwrap();
            }
            index
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tiered_add_async,
    bench_tiered_add_inplace,
    bench_tiered_query_flat,
    bench_tiered_query_hnsw,
    bench_tiered_query_both,
    bench_tiered_flush,
    bench_comparison_query,
    bench_comparison_add,
);

criterion_main!(benches);

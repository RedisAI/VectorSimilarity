//! Benchmarks using DBPedia dataset (same as C++ benchmarks).
//!
//! This benchmark uses the same data files as the C++ benchmarks for
//! direct comparison. If data files are not available, it falls back
//! to random data with the same dimensions.
//!
//! Dataset: DBPedia embeddings
//! - 1M vectors, 768 dimensions, Cosine similarity
//! - 10K query vectors
//! - HNSW parameters: M=64, EF_C=512
//!
//! To download the benchmark data files, run from repository root:
//!   bash tests/benchmark/bm_files.sh benchmarks-all
//!
//! Run with: cargo bench --bench dbpedia_bench

mod data_loader;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use data_loader::{
    generate_normalized_vectors, try_load_dataset_queries, try_load_dataset_vectors,
    DBPEDIA_SINGLE_FP32,
};
use vecsim::distance::Metric;
use vecsim::index::brute_force::{BruteForceParams, BruteForceSingle};
use vecsim::index::hnsw::{HnswParams, HnswSingle};
use vecsim::index::VecSimIndex;
use vecsim::query::QueryParams;

/// Benchmark data holder - loaded once for all benchmarks.
struct BenchmarkData {
    vectors: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    dim: usize,
    is_real_data: bool,
}

impl BenchmarkData {
    fn load(max_vectors: usize, max_queries: usize) -> Self {
        let config = &DBPEDIA_SINGLE_FP32;

        // Try to load real data
        if let (Some(vectors), Some(queries)) = (
            try_load_dataset_vectors(config, max_vectors),
            try_load_dataset_queries(config, max_queries),
        ) {
            println!("Loaded real DBPedia dataset: {} vectors, {} queries, dim={}",
                vectors.len(), queries.len(), config.dim);
            return Self {
                vectors,
                queries,
                dim: config.dim,
                is_real_data: true,
            };
        }

        // Fall back to random data
        println!("Using random data (real dataset not found)");
        println!("To use real data, run: bash tests/benchmark/bm_files.sh benchmarks-all");
        let dim = config.dim;
        Self {
            vectors: generate_normalized_vectors(max_vectors, dim),
            queries: generate_normalized_vectors(max_queries, dim),
            dim,
            is_real_data: false,
        }
    }
}

/// Build HNSW index with DBPedia parameters (M=64, EF_C=512).
fn build_hnsw_index(data: &BenchmarkData, n_vectors: usize) -> HnswSingle<f32> {
    let params = HnswParams::new(data.dim, Metric::Cosine)
        .with_m(DBPEDIA_SINGLE_FP32.m)
        .with_ef_construction(DBPEDIA_SINGLE_FP32.ef_construction)
        .with_ef_runtime(10);

    let mut index = HnswSingle::<f32>::new(params);
    for (i, v) in data.vectors.iter().take(n_vectors).enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }
    index
}

/// Build BruteForce index for comparison.
fn build_bf_index(data: &BenchmarkData, n_vectors: usize) -> BruteForceSingle<f32> {
    let params = BruteForceParams::new(data.dim, Metric::Cosine);
    let mut index = BruteForceSingle::<f32>::new(params);
    for (i, v) in data.vectors.iter().take(n_vectors).enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }
    index
}

/// Benchmark top-k queries on HNSW with varying ef_runtime.
fn bench_hnsw_topk_ef_runtime(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);
    let index = build_hnsw_index(&data, 100_000);

    let mut group = c.benchmark_group("dbpedia_hnsw_topk_ef");
    let data_label = if data.is_real_data { "real" } else { "random" };

    for ef in [10, 50, 100, 200, 500] {
        let query_params = QueryParams::new().with_ef_runtime(ef);
        let label = format!("{}_{}", data_label, ef);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &data.queries[query_idx % data.queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark top-k queries with varying k.
fn bench_hnsw_topk_k(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);
    let index = build_hnsw_index(&data, 100_000);

    let mut group = c.benchmark_group("dbpedia_hnsw_topk_k");
    let data_label = if data.is_real_data { "real" } else { "random" };

    // Use ef_runtime = 200 like C++ benchmarks
    let query_params = QueryParams::new().with_ef_runtime(200);

    for k in [1, 10, 50, 100, 500] {
        let label = format!("{}_{}", data_label, k);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &data.queries[query_idx % data.queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(k), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark HNSW vs BruteForce comparison (like C++ benchmarks).
fn bench_hnsw_vs_bf(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);

    let hnsw_index = build_hnsw_index(&data, 100_000);
    let bf_index = build_bf_index(&data, 100_000);

    let mut group = c.benchmark_group("dbpedia_hnsw_vs_bf");
    let data_label = if data.is_real_data { "real" } else { "random" };

    // BruteForce baseline
    group.bench_function(format!("{}_bf", data_label), |b| {
        let mut query_idx = 0;
        b.iter(|| {
            let query = &data.queries[query_idx % data.queries.len()];
            query_idx += 1;
            bf_index
                .top_k_query(black_box(query), black_box(10), None)
                .unwrap()
        });
    });

    // HNSW with ef=10 (fastest, lowest quality)
    let query_params_10 = QueryParams::new().with_ef_runtime(10);
    group.bench_function(format!("{}_hnsw_ef10", data_label), |b| {
        let mut query_idx = 0;
        b.iter(|| {
            let query = &data.queries[query_idx % data.queries.len()];
            query_idx += 1;
            hnsw_index
                .top_k_query(black_box(query), black_box(10), Some(&query_params_10))
                .unwrap()
        });
    });

    // HNSW with ef=100 (balanced)
    let query_params_100 = QueryParams::new().with_ef_runtime(100);
    group.bench_function(format!("{}_hnsw_ef100", data_label), |b| {
        let mut query_idx = 0;
        b.iter(|| {
            let query = &data.queries[query_idx % data.queries.len()];
            query_idx += 1;
            hnsw_index
                .top_k_query(black_box(query), black_box(10), Some(&query_params_100))
                .unwrap()
        });
    });

    // HNSW with ef=500 (high quality)
    let query_params_500 = QueryParams::new().with_ef_runtime(500);
    group.bench_function(format!("{}_hnsw_ef500", data_label), |b| {
        let mut query_idx = 0;
        b.iter(|| {
            let query = &data.queries[query_idx % data.queries.len()];
            query_idx += 1;
            hnsw_index
                .top_k_query(black_box(query), black_box(10), Some(&query_params_500))
                .unwrap()
        });
    });

    group.finish();
}

/// Benchmark adding vectors to HNSW.
fn bench_hnsw_add(c: &mut Criterion) {
    let data = BenchmarkData::load(10_000, 100);

    let mut group = c.benchmark_group("dbpedia_hnsw_add");
    group.sample_size(10); // HNSW add is slow

    let data_label = if data.is_real_data { "real" } else { "random" };

    for n_vectors in [1_000, 5_000, 10_000] {
        let label = format!("{}_{}", data_label, n_vectors);
        let vectors: Vec<_> = data.vectors.iter().take(n_vectors).cloned().collect();

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let params = HnswParams::new(data.dim, Metric::Cosine)
                    .with_m(DBPEDIA_SINGLE_FP32.m)
                    .with_ef_construction(DBPEDIA_SINGLE_FP32.ef_construction);
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

/// Benchmark delete operations on HNSW.
fn bench_hnsw_delete(c: &mut Criterion) {
    let data = BenchmarkData::load(10_000, 100);

    let mut group = c.benchmark_group("dbpedia_hnsw_delete");
    group.sample_size(10);

    let data_label = if data.is_real_data { "real" } else { "random" };

    for n_vectors in [1_000, 5_000] {
        let label = format!("{}_{}", data_label, n_vectors);
        let vectors: Vec<_> = data.vectors.iter().take(n_vectors).cloned().collect();

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter_batched(
                || {
                    let params = HnswParams::new(data.dim, Metric::Cosine)
                        .with_m(DBPEDIA_SINGLE_FP32.m)
                        .with_ef_construction(DBPEDIA_SINGLE_FP32.ef_construction);
                    let mut index = HnswSingle::<f32>::new(params);
                    for (i, v) in vectors.iter().enumerate() {
                        index.add_vector(v, i as u64).unwrap();
                    }
                    index
                },
                |mut index| {
                    // Delete half the vectors
                    for i in (0..n_vectors).step_by(2) {
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

/// Benchmark range queries on HNSW.
fn bench_hnsw_range(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);
    let index = build_hnsw_index(&data, 100_000);

    let mut group = c.benchmark_group("dbpedia_hnsw_range");
    let data_label = if data.is_real_data { "real" } else { "random" };

    // For cosine distance, typical radius values
    for radius in [0.1f32, 0.2, 0.3, 0.5] {
        let label = format!("{}_{}", data_label, radius);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &data.queries[query_idx % data.queries.len()];
                query_idx += 1;
                index
                    .range_query(black_box(query), black_box(radius), None)
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark index scaling (varying index size).
fn bench_hnsw_scaling(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);

    let mut group = c.benchmark_group("dbpedia_hnsw_scaling");
    let data_label = if data.is_real_data { "real" } else { "random" };

    let query_params = QueryParams::new().with_ef_runtime(100);

    for n_vectors in [10_000, 50_000, 100_000] {
        let index = build_hnsw_index(&data, n_vectors);
        let label = format!("{}_{}", data_label, n_vectors);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &data.queries[query_idx % data.queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_topk_ef_runtime,
    bench_hnsw_topk_k,
    bench_hnsw_vs_bf,
    bench_hnsw_add,
    bench_hnsw_delete,
    bench_hnsw_range,
    bench_hnsw_scaling,
);

criterion_main!(benches);

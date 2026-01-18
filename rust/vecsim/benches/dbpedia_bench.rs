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
//! Tested data types (matching C++ benchmarks):
//! - f32 (fp32)
//! - f64 (fp64)
//! - BFloat16 (bf16)
//! - Float16 (fp16)
//! - Int8 (int8)
//! - UInt8 (uint8)
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
use vecsim::types::{BFloat16, Float16, Int8, UInt8, VectorElement};

// ============================================================================
// Data Loading
// ============================================================================

/// Benchmark data holder - loaded once for all benchmarks.
struct BenchmarkData {
    vectors_f32: Vec<Vec<f32>>,
    queries_f32: Vec<Vec<f32>>,
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
            println!(
                "Loaded real DBPedia dataset: {} vectors, {} queries, dim={}",
                vectors.len(),
                queries.len(),
                config.dim
            );
            return Self {
                vectors_f32: vectors,
                queries_f32: queries,
                dim: config.dim,
                is_real_data: true,
            };
        }

        // Fall back to random data
        println!("Using random data (real dataset not found)");
        println!("To use real data, run: bash tests/benchmark/bm_files.sh benchmarks-all");
        let dim = config.dim;
        Self {
            vectors_f32: generate_normalized_vectors(max_vectors, dim),
            queries_f32: generate_normalized_vectors(max_queries, dim),
            dim,
            is_real_data: false,
        }
    }

    /// Convert vectors to a different element type.
    fn vectors_as<T: VectorElement>(&self) -> Vec<Vec<T>> {
        self.vectors_f32
            .iter()
            .map(|v| v.iter().map(|&x| T::from_f32(x)).collect())
            .collect()
    }

    /// Convert queries to a different element type.
    fn queries_as<T: VectorElement>(&self) -> Vec<Vec<T>> {
        self.queries_f32
            .iter()
            .map(|v| v.iter().map(|&x| T::from_f32(x)).collect())
            .collect()
    }

    fn data_label(&self) -> &'static str {
        if self.is_real_data {
            "real"
        } else {
            "random"
        }
    }
}

// ============================================================================
// Generic Index Builders
// ============================================================================

fn build_hnsw_index<T: VectorElement>(
    vectors: &[Vec<T>],
    dim: usize,
    n_vectors: usize,
) -> HnswSingle<T> {
    let params = HnswParams::new(dim, Metric::Cosine)
        .with_m(DBPEDIA_SINGLE_FP32.m)
        .with_ef_construction(DBPEDIA_SINGLE_FP32.ef_construction)
        .with_ef_runtime(10);

    let mut index = HnswSingle::<T>::new(params);
    for (i, v) in vectors.iter().take(n_vectors).enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }
    index
}

fn build_bf_index<T: VectorElement>(
    vectors: &[Vec<T>],
    dim: usize,
    n_vectors: usize,
) -> BruteForceSingle<T> {
    let params = BruteForceParams::new(dim, Metric::Cosine);
    let mut index = BruteForceSingle::<T>::new(params);
    for (i, v) in vectors.iter().take(n_vectors).enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }
    index
}

// ============================================================================
// F32 Benchmarks (Primary - most detailed)
// ============================================================================

/// Benchmark top-k queries on HNSW with varying ef_runtime (f32).
fn bench_f32_topk_ef_runtime(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);
    let vectors = data.vectors_as::<f32>();
    let queries = data.queries_as::<f32>();
    let index = build_hnsw_index(&vectors, data.dim, 100_000);

    let mut group = c.benchmark_group("f32_hnsw_topk_ef");

    for ef in [10, 50, 100, 200, 500] {
        let query_params = QueryParams::new().with_ef_runtime(ef);
        let label = format!("{}_{}", data.data_label(), ef);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark top-k queries with varying k (f32).
fn bench_f32_topk_k(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);
    let vectors = data.vectors_as::<f32>();
    let queries = data.queries_as::<f32>();
    let index = build_hnsw_index(&vectors, data.dim, 100_000);

    let mut group = c.benchmark_group("f32_hnsw_topk_k");
    let query_params = QueryParams::new().with_ef_runtime(200);

    for k in [1, 10, 50, 100, 500] {
        let label = format!("{}_{}", data.data_label(), k);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(k), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark HNSW vs BruteForce comparison (f32).
fn bench_f32_hnsw_vs_bf(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);
    let vectors = data.vectors_as::<f32>();
    let queries = data.queries_as::<f32>();

    let hnsw_index = build_hnsw_index(&vectors, data.dim, 100_000);
    let bf_index = build_bf_index(&vectors, data.dim, 100_000);

    let mut group = c.benchmark_group("f32_hnsw_vs_bf");

    // BruteForce baseline
    group.bench_function(format!("{}_bf", data.data_label()), |b| {
        let mut query_idx = 0;
        b.iter(|| {
            let query = &queries[query_idx % queries.len()];
            query_idx += 1;
            bf_index
                .top_k_query(black_box(query), black_box(10), None)
                .unwrap()
        });
    });

    // HNSW with various ef values
    for ef in [10, 100, 500] {
        let query_params = QueryParams::new().with_ef_runtime(ef);
        group.bench_function(format!("{}_hnsw_ef{}", data.data_label(), ef), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                hnsw_index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================================
// F64 Benchmarks
// ============================================================================

fn bench_f64_topk(c: &mut Criterion) {
    let data = BenchmarkData::load(50_000, 500);
    let vectors = data.vectors_as::<f64>();
    let queries = data.queries_as::<f64>();
    let index = build_hnsw_index(&vectors, data.dim, 50_000);

    let mut group = c.benchmark_group("f64_hnsw_topk");

    for ef in [10, 100, 500] {
        let query_params = QueryParams::new().with_ef_runtime(ef);
        let label = format!("{}_{}", data.data_label(), ef);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================================
// BFloat16 Benchmarks
// ============================================================================

fn bench_bf16_topk(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);
    let vectors = data.vectors_as::<BFloat16>();
    let queries = data.queries_as::<BFloat16>();
    let index = build_hnsw_index(&vectors, data.dim, 100_000);

    let mut group = c.benchmark_group("bf16_hnsw_topk");

    for ef in [10, 100, 500] {
        let query_params = QueryParams::new().with_ef_runtime(ef);
        let label = format!("{}_{}", data.data_label(), ef);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Float16 Benchmarks
// ============================================================================

fn bench_fp16_topk(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);
    let vectors = data.vectors_as::<Float16>();
    let queries = data.queries_as::<Float16>();
    let index = build_hnsw_index(&vectors, data.dim, 100_000);

    let mut group = c.benchmark_group("fp16_hnsw_topk");

    for ef in [10, 100, 500] {
        let query_params = QueryParams::new().with_ef_runtime(ef);
        let label = format!("{}_{}", data.data_label(), ef);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Int8 Benchmarks
// ============================================================================

fn bench_int8_topk(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);
    let vectors = data.vectors_as::<Int8>();
    let queries = data.queries_as::<Int8>();
    let index = build_hnsw_index(&vectors, data.dim, 100_000);

    let mut group = c.benchmark_group("int8_hnsw_topk");

    for ef in [10, 100, 500] {
        let query_params = QueryParams::new().with_ef_runtime(ef);
        let label = format!("{}_{}", data.data_label(), ef);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================================
// UInt8 Benchmarks
// ============================================================================

fn bench_uint8_topk(c: &mut Criterion) {
    let data = BenchmarkData::load(100_000, 1_000);
    let vectors = data.vectors_as::<UInt8>();
    let queries = data.queries_as::<UInt8>();
    let index = build_hnsw_index(&vectors, data.dim, 100_000);

    let mut group = c.benchmark_group("uint8_hnsw_topk");

    for ef in [10, 100, 500] {
        let query_params = QueryParams::new().with_ef_runtime(ef);
        let label = format!("{}_{}", data.data_label(), ef);

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Cross-Type Comparison Benchmarks
// ============================================================================

/// Compare query performance across all data types.
fn bench_all_types_comparison(c: &mut Criterion) {
    let data = BenchmarkData::load(50_000, 500);
    let query_params = QueryParams::new().with_ef_runtime(100);

    let mut group = c.benchmark_group("all_types_topk10_ef100");

    // f32
    {
        let vectors = data.vectors_as::<f32>();
        let queries = data.queries_as::<f32>();
        let index = build_hnsw_index(&vectors, data.dim, 50_000);

        group.bench_function("f32", |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    // f64
    {
        let vectors = data.vectors_as::<f64>();
        let queries = data.queries_as::<f64>();
        let index = build_hnsw_index(&vectors, data.dim, 50_000);

        group.bench_function("f64", |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    // bf16
    {
        let vectors = data.vectors_as::<BFloat16>();
        let queries = data.queries_as::<BFloat16>();
        let index = build_hnsw_index(&vectors, data.dim, 50_000);

        group.bench_function("bf16", |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    // fp16
    {
        let vectors = data.vectors_as::<Float16>();
        let queries = data.queries_as::<Float16>();
        let index = build_hnsw_index(&vectors, data.dim, 50_000);

        group.bench_function("fp16", |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    // int8
    {
        let vectors = data.vectors_as::<Int8>();
        let queries = data.queries_as::<Int8>();
        let index = build_hnsw_index(&vectors, data.dim, 50_000);

        group.bench_function("int8", |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    // uint8
    {
        let vectors = data.vectors_as::<UInt8>();
        let queries = data.queries_as::<UInt8>();
        let index = build_hnsw_index(&vectors, data.dim, 50_000);

        group.bench_function("uint8", |b| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx % queries.len()];
                query_idx += 1;
                index
                    .top_k_query(black_box(query), black_box(10), Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Index Construction Benchmarks (all types)
// ============================================================================

fn bench_all_types_add(c: &mut Criterion) {
    let data = BenchmarkData::load(5_000, 100);

    let mut group = c.benchmark_group("all_types_add_5000");
    group.sample_size(10);

    // f32
    {
        let vectors = data.vectors_as::<f32>();
        group.bench_function("f32", |b| {
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

    // f64
    {
        let vectors = data.vectors_as::<f64>();
        group.bench_function("f64", |b| {
            b.iter(|| {
                let params = HnswParams::new(data.dim, Metric::Cosine)
                    .with_m(DBPEDIA_SINGLE_FP32.m)
                    .with_ef_construction(DBPEDIA_SINGLE_FP32.ef_construction);
                let mut index = HnswSingle::<f64>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    // bf16
    {
        let vectors = data.vectors_as::<BFloat16>();
        group.bench_function("bf16", |b| {
            b.iter(|| {
                let params = HnswParams::new(data.dim, Metric::Cosine)
                    .with_m(DBPEDIA_SINGLE_FP32.m)
                    .with_ef_construction(DBPEDIA_SINGLE_FP32.ef_construction);
                let mut index = HnswSingle::<BFloat16>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    // fp16
    {
        let vectors = data.vectors_as::<Float16>();
        group.bench_function("fp16", |b| {
            b.iter(|| {
                let params = HnswParams::new(data.dim, Metric::Cosine)
                    .with_m(DBPEDIA_SINGLE_FP32.m)
                    .with_ef_construction(DBPEDIA_SINGLE_FP32.ef_construction);
                let mut index = HnswSingle::<Float16>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    // int8
    {
        let vectors = data.vectors_as::<Int8>();
        group.bench_function("int8", |b| {
            b.iter(|| {
                let params = HnswParams::new(data.dim, Metric::Cosine)
                    .with_m(DBPEDIA_SINGLE_FP32.m)
                    .with_ef_construction(DBPEDIA_SINGLE_FP32.ef_construction);
                let mut index = HnswSingle::<Int8>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    // uint8
    {
        let vectors = data.vectors_as::<UInt8>();
        group.bench_function("uint8", |b| {
            b.iter(|| {
                let params = HnswParams::new(data.dim, Metric::Cosine)
                    .with_m(DBPEDIA_SINGLE_FP32.m)
                    .with_ef_construction(DBPEDIA_SINGLE_FP32.ef_construction);
                let mut index = HnswSingle::<UInt8>::new(params);
                for (i, v) in vectors.iter().enumerate() {
                    index.add_vector(black_box(v), i as u64).unwrap();
                }
                index
            });
        });
    }

    group.finish();
}

// ============================================================================
// Main Benchmark Groups
// ============================================================================

criterion_group!(
    benches,
    // F32 detailed benchmarks
    bench_f32_topk_ef_runtime,
    bench_f32_topk_k,
    bench_f32_hnsw_vs_bf,
    // Per-type benchmarks
    bench_f64_topk,
    bench_bf16_topk,
    bench_fp16_topk,
    bench_int8_topk,
    bench_uint8_topk,
    // Cross-type comparisons
    bench_all_types_comparison,
    bench_all_types_add,
);

criterion_main!(benches);

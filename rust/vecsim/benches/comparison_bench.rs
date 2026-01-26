//! Benchmarks designed to compare with C++ implementation.
//!
//! Uses similar parameters to C++ benchmarks for fair comparison:
//! - 10,000 vectors (smaller than C++ 1M for quick testing)
//! - 128 dimensions
//! - HNSW M=16, ef_construction=200, ef_runtime=10/100/200
//!
//! Run with: cargo bench --bench comparison_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vecsim::distance::Metric;
use vecsim::index::brute_force::{BruteForceParams, BruteForceSingle};
use vecsim::index::hnsw::{HnswParams, HnswSingle};
use vecsim::index::VecSimIndex;
use vecsim::query::QueryParams;

const DIM: usize = 128;
const N_VECTORS: usize = 10_000;
const N_QUERIES: usize = 100;

/// Generate normalized random vectors (for cosine similarity).
fn generate_normalized_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
}

/// Benchmark: BruteForce Top-K query (baseline)
fn bench_bf_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_bf_topk");

    let vectors = generate_normalized_vectors(N_VECTORS, DIM);
    let queries = generate_normalized_vectors(N_QUERIES, DIM);

    let params = BruteForceParams::new(DIM, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    for k in [10, 100] {
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.iter(|| {
                for query in &queries {
                    black_box(index.top_k_query(query, k, None).unwrap());
                }
            });
        });
    }

    group.finish();
}

/// Benchmark: HNSW Top-K query with varying ef_runtime
fn bench_hnsw_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_hnsw_topk");

    let vectors = generate_normalized_vectors(N_VECTORS, DIM);
    let queries = generate_normalized_vectors(N_QUERIES, DIM);

    let params = HnswParams::new(DIM, Metric::L2)
        .with_m(16)
        .with_ef_construction(200)
        .with_ef_runtime(100);
    let mut index = HnswSingle::<f32>::new(params);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    // Test different ef_runtime values
    for ef in [10, 100, 200] {
        let query_params = QueryParams::new().with_ef_runtime(ef);
        group.bench_with_input(BenchmarkId::new("ef", ef), &ef, |b, _| {
            b.iter(|| {
                for query in &queries {
                    black_box(index.top_k_query(query, 10, Some(&query_params)).unwrap());
                }
            });
        });
    }

    group.finish();
}

/// Benchmark: HNSW index construction
fn bench_hnsw_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_hnsw_construction");
    group.sample_size(10);

    let vectors = generate_normalized_vectors(N_VECTORS, DIM);

    group.bench_function("10k_vectors", |b| {
        b.iter(|| {
            let params = HnswParams::new(DIM, Metric::L2)
                .with_m(16)
                .with_ef_construction(200);
            let mut index = HnswSingle::<f32>::new(params);
            for (i, v) in vectors.iter().enumerate() {
                index.add_vector(black_box(v), i as u64).unwrap();
            }
            index
        });
    });

    group.finish();
}

/// Benchmark: Different metrics
fn bench_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_metrics");

    let vectors = generate_normalized_vectors(N_VECTORS, DIM);
    let queries = generate_normalized_vectors(N_QUERIES, DIM);

    for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
        let params = HnswParams::new(DIM, metric)
            .with_m(16)
            .with_ef_construction(200)
            .with_ef_runtime(100);
        let mut index = HnswSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }

        let name = match metric {
            Metric::L2 => "L2",
            Metric::InnerProduct => "IP",
            Metric::Cosine => "Cosine",
        };

        group.bench_function(name, |b| {
            b.iter(|| {
                for query in &queries {
                    black_box(index.top_k_query(query, 10, None).unwrap());
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bf_topk,
    bench_hnsw_topk,
    bench_hnsw_construction,
    bench_metrics,
);

criterion_main!(benches);

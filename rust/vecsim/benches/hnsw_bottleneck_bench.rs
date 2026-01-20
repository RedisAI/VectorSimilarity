//! Benchmarks for measuring HNSW performance bottlenecks.
//!
//! This module measures individual components to identify performance bottlenecks:
//! - Distance computation (L2, IP, Cosine) for different dimensions
//! - SIMD vs scalar implementations
//! - Visited nodes tracking operations
//! - Neighbor selection algorithms
//! - Search layer performance at different ef values
//! - Memory access patterns
//!
//! Run with: cargo bench --bench hnsw_bottleneck_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use vecsim::distance::cosine::CosineDistance;
use vecsim::distance::ip::InnerProductDistance;
use vecsim::distance::l2::L2Distance;
use vecsim::distance::{DistanceFunction, Metric};
use vecsim::index::hnsw::search::{select_neighbors_heuristic, select_neighbors_simple};
use vecsim::index::hnsw::{HnswParams, HnswSingle, VisitedNodesHandler, VisitedNodesHandlerPool};
use vecsim::index::VecSimIndex;
use vecsim::query::QueryParams;

/// Generate random vectors for benchmarking.
fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Benchmark distance computation across different dimensions and metrics.
///
/// This helps identify if distance computation is a bottleneck for specific dimensions.
fn bench_distance_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_computation");

    for dim in [32, 128, 384, 768, 1536] {
        let v1: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 / dim as f32).collect();

        group.throughput(Throughput::Elements(dim as u64));

        // L2 distance with SIMD
        let dist_fn = L2Distance::<f32>::with_simd(dim);
        group.bench_with_input(BenchmarkId::new("L2_simd", dim), &dim, |b, &d| {
            b.iter(|| dist_fn.compute(black_box(&v1), black_box(&v2), d));
        });

        // L2 distance scalar
        let dist_fn_scalar = L2Distance::<f32>::scalar(dim);
        group.bench_with_input(BenchmarkId::new("L2_scalar", dim), &dim, |b, &d| {
            b.iter(|| dist_fn_scalar.compute(black_box(&v1), black_box(&v2), d));
        });

        // Inner product with SIMD
        let ip_fn = InnerProductDistance::<f32>::with_simd(dim);
        group.bench_with_input(BenchmarkId::new("IP_simd", dim), &dim, |b, &d| {
            b.iter(|| ip_fn.compute(black_box(&v1), black_box(&v2), d));
        });

        // Inner product scalar
        let ip_fn_scalar = InnerProductDistance::<f32>::scalar(dim);
        group.bench_with_input(BenchmarkId::new("IP_scalar", dim), &dim, |b, &d| {
            b.iter(|| ip_fn_scalar.compute(black_box(&v1), black_box(&v2), d));
        });

        // Cosine distance with SIMD
        let cos_fn = CosineDistance::<f32>::with_simd(dim);
        group.bench_with_input(BenchmarkId::new("Cosine_simd", dim), &dim, |b, &d| {
            b.iter(|| cos_fn.compute(black_box(&v1), black_box(&v2), d));
        });

        // Cosine distance scalar
        let cos_fn_scalar = CosineDistance::<f32>::scalar(dim);
        group.bench_with_input(BenchmarkId::new("Cosine_scalar", dim), &dim, |b, &d| {
            b.iter(|| cos_fn_scalar.compute(black_box(&v1), black_box(&v2), d));
        });
    }

    group.finish();
}

/// Benchmark visited nodes tracking operations.
///
/// Measures visit(), is_visited(), reset(), and pool checkout/return overhead.
fn bench_visited_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group("visited_nodes");

    for capacity in [1_000, 10_000, 100_000] {
        // Visit operation (mark node as visited)
        group.bench_with_input(
            BenchmarkId::new("visit", capacity),
            &capacity,
            |b, &cap| {
                let handler = VisitedNodesHandler::new(cap);
                let mut rng = rand::thread_rng();
                b.iter(|| {
                    let id = rng.gen_range(0..cap as u32);
                    handler.visit(black_box(id))
                });
            },
        );

        // is_visited check operation
        group.bench_with_input(
            BenchmarkId::new("is_visited", capacity),
            &capacity,
            |b, &cap| {
                let handler = VisitedNodesHandler::new(cap);
                // Pre-visit some nodes
                for i in (0..cap as u32).step_by(2) {
                    handler.visit(i);
                }
                let mut rng = rand::thread_rng();
                b.iter(|| {
                    let id = rng.gen_range(0..cap as u32);
                    handler.is_visited(black_box(id))
                });
            },
        );

        // Reset operation (O(1) with tag-based approach)
        group.bench_with_input(
            BenchmarkId::new("reset", capacity),
            &capacity,
            |b, &cap| {
                let mut handler = VisitedNodesHandler::new(cap);
                b.iter(|| {
                    handler.reset();
                });
            },
        );

        // Pool checkout and return
        group.bench_with_input(
            BenchmarkId::new("pool_get_return", capacity),
            &capacity,
            |b, &cap| {
                let pool = VisitedNodesHandlerPool::new(cap);
                // Warm up the pool with one handler
                {
                    let _h = pool.get();
                }
                b.iter(|| {
                    let _handler = pool.get();
                    // Handler returned automatically on drop
                });
            },
        );
    }

    group.finish();
}

/// Benchmark neighbor selection algorithms.
///
/// Compares simple selection (sort + take) vs heuristic selection (diversity-aware).
fn bench_neighbor_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_selection");
    let dim = 128;

    for num_candidates in [10, 50, 100, 200] {
        // Generate candidates as (id, distance) pairs
        let candidates: Vec<(u32, f32)> = (0..num_candidates)
            .map(|i| (i as u32, i as f32 * 0.1))
            .collect();

        // Simple selection (just keep M closest)
        group.bench_with_input(
            BenchmarkId::new("simple", num_candidates),
            &candidates,
            |b, cands| {
                b.iter(|| select_neighbors_simple(black_box(cands), 16));
            },
        );

        // Heuristic selection (diversity-aware, requires vector data)
        let vectors = generate_vectors(num_candidates, dim);
        let data_getter = |id: u32| -> Option<&[f32]> { vectors.get(id as usize).map(|v| v.as_slice()) };
        let dist_fn = L2Distance::<f32>::with_simd(dim);

        group.bench_with_input(
            BenchmarkId::new("heuristic", num_candidates),
            &candidates,
            |b, cands| {
                b.iter(|| {
                    select_neighbors_heuristic(
                        0,     // target id
                        black_box(cands),
                        16,    // M (max neighbors)
                        &data_getter,
                        &dist_fn as &dyn DistanceFunction<f32, Output = f32>,
                        dim,
                        false, // extend_candidates
                        true,  // keep_pruned
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark HNSW search with varying ef values.
///
/// Isolates search_layer performance to measure the impact of ef_runtime.
fn bench_search_ef_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_ef_impact");
    group.sample_size(50);

    let dim = 128;
    let size = 10_000;
    let vectors = generate_vectors(size, dim);

    // Build index once
    let params = HnswParams::new(dim, Metric::L2)
        .with_m(16)
        .with_ef_construction(100);
    let mut index = HnswSingle::<f32>::new(params);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    let query = generate_vectors(1, dim).pop().unwrap();

    for ef in [10, 20, 50, 100, 200, 400] {
        let query_params = QueryParams::new().with_ef_runtime(ef);
        group.bench_with_input(BenchmarkId::from_parameter(ef), &ef, |b, _| {
            b.iter(|| {
                index
                    .top_k_query(black_box(&query), 10, Some(&query_params))
                    .unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark search with and without filters.
///
/// Measures the overhead of filter evaluation during search.
fn bench_search_with_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_filters");
    group.sample_size(50);

    let dim = 128;
    let size = 10_000;
    let vectors = generate_vectors(size, dim);

    let params = HnswParams::new(dim, Metric::L2)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_runtime(100);
    let mut index = HnswSingle::<f32>::new(params);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    let query = generate_vectors(1, dim).pop().unwrap();

    // Without filter
    group.bench_function("no_filter", |b| {
        b.iter(|| {
            index
                .top_k_query(black_box(&query), 10, None)
                .unwrap()
        });
    });

    // With filter that accepts all (minimal overhead measurement)
    group.bench_function("filter_accept_all", |b| {
        b.iter(|| {
            let params = QueryParams::new()
                .with_ef_runtime(100)
                .with_filter(|_| true);
            index
                .top_k_query(black_box(&query), 10, Some(&params))
                .unwrap()
        });
    });

    // With filter that accepts 50%
    group.bench_function("filter_accept_50pct", |b| {
        b.iter(|| {
            let params = QueryParams::new()
                .with_ef_runtime(100)
                .with_filter(|label| label % 2 == 0);
            index
                .top_k_query(black_box(&query), 10, Some(&params))
                .unwrap()
        });
    });

    // With filter that accepts 10%
    group.bench_function("filter_accept_10pct", |b| {
        b.iter(|| {
            let params = QueryParams::new()
                .with_ef_runtime(100)
                .with_filter(|label| label % 10 == 0);
            index
                .top_k_query(black_box(&query), 10, Some(&params))
                .unwrap()
        });
    });

    group.finish();
}

/// Benchmark memory access patterns (sequential vs random).
///
/// This helps understand cache effects in vector access patterns.
fn bench_memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access");

    let size = 100_000;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    // Sequential access - cache friendly
    group.bench_function("sequential_1000", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..1000 {
                sum += black_box(data[i]);
            }
            sum
        });
    });

    // Sequential access with stride (simulates vector access)
    let dim = 128;
    group.bench_function("sequential_stride_128", |b| {
        let num_vectors = size / dim;
        b.iter(|| {
            let mut sum = 0.0f32;
            for v in 0..num_vectors.min(100) {
                for d in 0..dim {
                    sum += black_box(data[v * dim + d]);
                }
            }
            sum
        });
    });

    // Random access - cache unfriendly
    let mut rng = rand::thread_rng();
    let random_indices: Vec<usize> = (0..1000).map(|_| rng.gen_range(0..size)).collect();
    group.bench_function("random_1000", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &i in &random_indices {
                sum += black_box(data[i]);
            }
            sum
        });
    });

    // Random vector access (simulates HNSW neighbor traversal)
    let random_vector_indices: Vec<usize> = (0..100)
        .map(|_| rng.gen_range(0..(size / dim)))
        .collect();
    group.bench_function("random_vectors_100", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &vi in &random_vector_indices {
                let base = vi * dim;
                for d in 0..dim {
                    sum += black_box(data[base + d]);
                }
            }
            sum
        });
    });

    group.finish();
}

/// Benchmark batch distance computations.
///
/// Measures throughput when computing distances against multiple candidates.
fn bench_batch_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distance");

    for dim in [128, 384, 768] {
        let query: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
        let candidates: Vec<Vec<f32>> = (0..100)
            .map(|_| {
                let mut rng = rand::thread_rng();
                (0..dim).map(|_| rng.gen::<f32>()).collect()
            })
            .collect();

        let dist_fn = L2Distance::<f32>::with_simd(dim);

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("100_candidates", dim),
            &dim,
            |b, &d| {
                b.iter(|| {
                    let mut min_dist = f32::MAX;
                    for cand in &candidates {
                        let dist = dist_fn.compute(black_box(&query), black_box(cand), d);
                        if dist < min_dist {
                            min_dist = dist;
                        }
                    }
                    min_dist
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_computation,
    bench_visited_nodes,
    bench_neighbor_selection,
    bench_search_ef_impact,
    bench_search_with_filters,
    bench_memory_access_patterns,
    bench_batch_distance,
);
criterion_main!(benches);

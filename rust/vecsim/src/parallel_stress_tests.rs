//! Parallelism stress tests for concurrent index operations.
//!
//! These tests verify thread safety and correctness under high contention
//! scenarios with multiple threads performing concurrent operations.

use crate::distance::Metric;
use crate::index::brute_force::{BruteForceMulti, BruteForceParams, BruteForceSingle};
use crate::index::hnsw::{HnswMulti, HnswParams, HnswSingle};
use crate::index::svs::{SvsParams, SvsSingle};
use crate::index::traits::VecSimIndex;
use crate::query::{CancellationToken, QueryParams};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ============================================================================
// Concurrent Query Tests
// ============================================================================

#[test]
fn test_brute_force_concurrent_queries() {
    let params = BruteForceParams::new(8, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    // Add 1000 vectors
    for i in 0..1000u64 {
        let mut v = vec![0.0f32; 8];
        v[0] = i as f32;
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 8;
    let queries_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    let q = vec![(t * queries_per_thread + i) as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 5, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_brute_force_parallel_queries_rayon() {
    let params = BruteForceParams::new(8, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    // Add enough vectors to trigger parallel execution (>1000)
    for i in 0..2000u64 {
        let mut v = vec![0.0f32; 8];
        v[0] = i as f32;
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 8;
    let queries_per_thread = 50;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    let q = vec![(t * 100 + i) as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    let params = QueryParams::new().with_parallel(true);
                    let results = index.top_k_query(&q, 10, Some(&params)).unwrap();
                    assert_eq!(results.len(), 10);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_hnsw_concurrent_queries() {
    let params = HnswParams::new(8, Metric::L2)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_runtime(50);
    let mut index = HnswSingle::<f32>::new(params);

    // Add 1000 vectors
    for i in 0..1000u64 {
        let mut v = vec![0.0f32; 8];
        v[0] = i as f32;
        v[1] = (i % 100) as f32;
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 8;
    let queries_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    let q = vec![(t * 50 + i) as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 5, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_svs_concurrent_queries() {
    let params = SvsParams::new(8, Metric::L2);
    let mut index = SvsSingle::<f32>::new(params);

    // Add 500 vectors
    for i in 0..500u64 {
        let mut v = vec![0.0f32; 8];
        v[0] = i as f32;
        v[1] = (i % 50) as f32;
        index.add_vector(&v, i).unwrap();
    }

    // Build for optimal search
    index.build();

    let index = Arc::new(index);
    let num_threads = 4;
    let queries_per_thread = 50;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    let q = vec![(t * 50 + i) as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 5, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// ============================================================================
// Concurrent Query + Range Query Tests
// ============================================================================

#[test]
fn test_brute_force_concurrent_mixed_queries() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    for i in 0..500u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 6;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..50 {
                    let q = vec![(t * 50 + i) as f32, 0.0, 0.0, 0.0];
                    if i % 2 == 0 {
                        // Top-k query
                        let results = index.top_k_query(&q, 5, None).unwrap();
                        assert!(!results.is_empty());
                    } else {
                        // Range query
                        let results = index.range_query(&q, 100.0, None).unwrap();
                        assert!(!results.is_empty());
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// ============================================================================
// Concurrent Filtered Query Tests
// ============================================================================

#[test]
fn test_brute_force_concurrent_filtered_queries() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    for i in 0..1000u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..100 {
                    let q = vec![(t * 100 + i) as f32, 0.0, 0.0, 0.0];
                    // Filter to only even labels in some queries
                    let params = if i % 2 == 0 {
                        QueryParams::new().with_filter(|label| label % 2 == 0)
                    } else {
                        QueryParams::new()
                    };
                    let results = index.top_k_query(&q, 10, Some(&params)).unwrap();
                    if i % 2 == 0 {
                        for r in &results.results {
                            assert!(r.label % 2 == 0);
                        }
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_hnsw_concurrent_filtered_queries() {
    let params = HnswParams::new(4, Metric::L2)
        .with_m(16)
        .with_ef_construction(50);
    let mut index = HnswSingle::<f32>::new(params);

    for i in 0..500u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..50 {
                    let q = vec![(t * 50 + i) as f32, 0.0, 0.0, 0.0];
                    let divisor = (t + 2) as u64; // Different filter per thread
                    let params = QueryParams::new().with_filter(move |label| label % divisor == 0);
                    let results = index.top_k_query(&q, 10, Some(&params)).unwrap();
                    for r in &results.results {
                        assert!(r.label % divisor == 0);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// ============================================================================
// Cancellation Token Tests
// ============================================================================

#[test]
fn test_cancellation_token_concurrent_cancel() {
    let token = CancellationToken::new();
    let num_readers = 10;

    // Clone tokens for reader threads
    let reader_tokens: Vec<_> = (0..num_readers).map(|_| token.clone()).collect();

    // Spawn reader threads that check cancellation
    let check_count = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = reader_tokens
        .into_iter()
        .map(|t| {
            let count = Arc::clone(&check_count);
            thread::spawn(move || {
                while !t.is_cancelled() {
                    count.fetch_add(1, Ordering::Relaxed);
                    thread::yield_now();
                }
            })
        })
        .collect();

    // Let readers run for a bit
    thread::sleep(Duration::from_millis(10));

    // Cancel
    token.cancel();

    // All threads should exit
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify cancellation was detected
    assert!(token.is_cancelled());
    assert!(check_count.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_cancellation_token_with_query() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    for i in 0..100u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let token = CancellationToken::new();
    let query_params = QueryParams::new().with_timeout_callback(token.as_callback());

    // Query before cancellation should work
    let q = vec![50.0, 0.0, 0.0, 0.0];
    let results = index.top_k_query(&q, 5, Some(&query_params)).unwrap();
    assert!(!results.is_empty());

    // Cancel the token
    token.cancel();

    // The timeout_callback now returns true, but the query still completes
    // (it just checks the callback during execution)
    let results2 = index.top_k_query(&q, 5, Some(&query_params)).unwrap();
    // Results may be partial or complete depending on implementation
    assert!(results2.len() <= 5);
}

// ============================================================================
// Multi-Index Concurrent Tests
// ============================================================================

#[test]
fn test_brute_force_multi_concurrent_queries() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceMulti::<f32>::new(params);

    // Add multiple vectors per label
    for label in 0..100u64 {
        for i in 0..5 {
            let v = vec![label as f32 + i as f32 * 0.01, 0.0, 0.0, 0.0];
            index.add_vector(&v, label).unwrap();
        }
    }

    let index = Arc::new(index);
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..50 {
                    let q = vec![(t * 25 + i) as f32, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 10, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_hnsw_multi_concurrent_queries() {
    let params = HnswParams::new(4, Metric::L2)
        .with_m(8)
        .with_ef_construction(50);
    let mut index = HnswMulti::<f32>::new(params);

    // Add multiple vectors per label
    for label in 0..100u64 {
        for i in 0..3 {
            let v = vec![label as f32 + i as f32 * 0.01, 0.0, 0.0, 0.0];
            index.add_vector(&v, label).unwrap();
        }
    }

    let index = Arc::new(index);
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..50 {
                    let q = vec![(t * 25 + i) as f32, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 10, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// ============================================================================
// High Contention Query Tests
// ============================================================================

#[test]
fn test_brute_force_high_contention_queries() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    for i in 0..200u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 16; // High thread count for contention
    let queries_per_thread = 200;
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let index = Arc::clone(&index);
            let count = Arc::clone(&success_count);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    let q = vec![(i % 200) as f32, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 5, None).unwrap();
                    if !results.is_empty() {
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // All queries should succeed
    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads * queries_per_thread
    );
}

#[test]
fn test_hnsw_high_contention_queries() {
    let params = HnswParams::new(8, Metric::L2)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_runtime(50);
    let mut index = HnswSingle::<f32>::new(params);

    for i in 0..500u64 {
        let mut v = vec![0.0f32; 8];
        v[0] = i as f32;
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 16;
    let queries_per_thread = 100;
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let index = Arc::clone(&index);
            let count = Arc::clone(&success_count);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    let q = vec![(i % 500) as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 10, None).unwrap();
                    if !results.is_empty() {
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads * queries_per_thread
    );
}

// ============================================================================
// Batch Iterator Concurrent Tests
// ============================================================================

#[test]
fn test_brute_force_concurrent_batch_iterators() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    for i in 0..200u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for _ in 0..10 {
                    let q = vec![(t * 50) as f32, 0.0, 0.0, 0.0];
                    let mut iter = index.batch_iterator(&q, None).unwrap();
                    let mut total = 0;
                    while let Some(batch) = iter.next_batch(20) {
                        total += batch.len();
                    }
                    assert_eq!(total, 200);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// ============================================================================
// Different Metrics Concurrent Tests
// ============================================================================

#[test]
fn test_concurrent_queries_different_metrics() {
    // Test L2
    let params_l2 = BruteForceParams::new(4, Metric::L2);
    let mut index_l2 = BruteForceSingle::<f32>::new(params_l2);

    // Test InnerProduct
    let params_ip = BruteForceParams::new(4, Metric::InnerProduct);
    let mut index_ip = BruteForceSingle::<f32>::new(params_ip);

    // Test Cosine
    let params_cos = BruteForceParams::new(4, Metric::Cosine);
    let mut index_cos = BruteForceSingle::<f32>::new(params_cos);

    for i in 0..200u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index_l2.add_vector(&v, i).unwrap();
        index_ip.add_vector(&v, i).unwrap();
        index_cos.add_vector(&v, i).unwrap();
    }

    let index_l2 = Arc::new(index_l2);
    let index_ip = Arc::new(index_ip);
    let index_cos = Arc::new(index_cos);

    let handles: Vec<_> = vec![
        {
            let index = Arc::clone(&index_l2);
            thread::spawn(move || {
                for i in 0..100 {
                    let q = vec![i as f32, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 5, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        },
        {
            let index = Arc::clone(&index_ip);
            thread::spawn(move || {
                for i in 0..100 {
                    let q = vec![i as f32, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 5, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        },
        {
            let index = Arc::clone(&index_cos);
            thread::spawn(move || {
                for i in 0..100 {
                    let q = vec![i as f32, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 5, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        },
    ];

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// ============================================================================
// Stress Test with Long Duration
// ============================================================================

#[test]
fn test_sustained_concurrent_load() {
    let params = HnswParams::new(4, Metric::L2)
        .with_m(8)
        .with_ef_construction(50);
    let mut index = HnswSingle::<f32>::new(params);

    for i in 0..300u64 {
        let v = vec![i as f32, (i % 50) as f32, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let duration = Duration::from_millis(500);
    let num_threads = 8;
    let query_counts: Vec<_> = (0..num_threads)
        .map(|_| Arc::new(AtomicUsize::new(0)))
        .collect();

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            let count = Arc::clone(&query_counts[t]);
            let start = std::time::Instant::now();
            thread::spawn(move || {
                let mut i = 0u64;
                while start.elapsed() < duration {
                    let q = vec![(i % 300) as f32, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 5, None).unwrap();
                    assert!(!results.is_empty());
                    count.fetch_add(1, Ordering::Relaxed);
                    i += 1;
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify all threads performed queries
    let total: usize = query_counts.iter().map(|c| c.load(Ordering::Relaxed)).sum();
    assert!(total > num_threads * 10, "Expected more queries to complete");
}

// ============================================================================
// Memory Ordering Tests
// ============================================================================

#[test]
fn test_atomic_count_consistency() {
    let params = BruteForceParams::new(4, Metric::L2);
    let index = Arc::new(parking_lot::RwLock::new(BruteForceSingle::<f32>::new(params)));

    let num_threads = 4;
    let adds_per_thread = 50;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..adds_per_thread {
                    let label = (t * adds_per_thread + i) as u64;
                    let v = vec![label as f32, 0.0, 0.0, 0.0];
                    index.write().add_vector(&v, label).unwrap();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Final count should be exact
    let final_count = index.read().index_size();
    assert_eq!(final_count, num_threads * adds_per_thread);
}

// ============================================================================
// Query Result Correctness Under Concurrency
// ============================================================================

#[test]
fn test_query_result_correctness_concurrent() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    // Add vectors with known distances
    // Vector at position i has value [i, 0, 0, 0]
    for i in 0..100u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 8;
    let error_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let index = Arc::clone(&index);
            let errors = Arc::clone(&error_count);
            thread::spawn(move || {
                for target in 0..100u64 {
                    let q = vec![target as f32, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 1, None).unwrap();
                    // The closest vector should be the exact match
                    if results.results[0].label != target {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // No errors should occur - results should always be correct
    assert_eq!(error_count.load(Ordering::Relaxed), 0);
}

#[test]
fn test_hnsw_query_result_stability() {
    let params = HnswParams::new(4, Metric::L2)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_runtime(100);
    let mut index = HnswSingle::<f32>::new(params);

    for i in 0..100u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 8;
    let error_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let index = Arc::clone(&index);
            let errors = Arc::clone(&error_count);
            thread::spawn(move || {
                for target in 0..100u64 {
                    let q = vec![target as f32, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 1, None).unwrap();
                    // With high ef_runtime, exact match should be found
                    if results.results[0].label != target {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Allow very few errors due to HNSW approximation
    let errors = error_count.load(Ordering::Relaxed);
    assert!(errors < 10, "Too many incorrect results: {}", errors);
}

// ============================================================================
// Visited Nodes Handler Tests (via HNSW search)
// ============================================================================

#[test]
fn test_hnsw_visited_handler_concurrent_searches() {
    // This tests the visited handler pool indirectly through concurrent HNSW searches
    let params = HnswParams::new(8, Metric::L2)
        .with_m(16)
        .with_ef_construction(100)
        .with_ef_runtime(100);
    let mut index = HnswSingle::<f32>::new(params);

    // Build a larger index to stress the visited handler
    for i in 0..1000u64 {
        let mut v = vec![0.0f32; 8];
        v[0] = (i % 100) as f32;
        v[1] = (i / 100) as f32;
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 8;
    let queries_per_thread = 200;

    // Many concurrent searches will stress the visited handler pool
    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    let q = vec![
                        ((t * queries_per_thread + i) % 100) as f32,
                        ((t * queries_per_thread + i) / 100) as f32,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    ];
                    let results = index.top_k_query(&q, 10, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_hnsw_visited_handler_with_varying_ef() {
    // Test with different ef_runtime values to stress visited handler resizing
    let params = HnswParams::new(4, Metric::L2)
        .with_m(8)
        .with_ef_construction(50);
    let mut index = HnswSingle::<f32>::new(params);

    for i in 0..500u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..100 {
                    let q = vec![(t * 100 + i) as f32, 0.0, 0.0, 0.0];
                    // Vary ef_runtime to stress handler
                    let ef = 20 + (i % 80);
                    let params = QueryParams::new().with_ef_runtime(ef);
                    let results = index.top_k_query(&q, 10, Some(&params)).unwrap();
                    assert!(!results.is_empty());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// ============================================================================
// Concurrent Modification Tests (Read-Write)
// ============================================================================

#[test]
fn test_brute_force_concurrent_read_write() {
    let params = BruteForceParams::new(4, Metric::L2);
    let index = Arc::new(parking_lot::RwLock::new(BruteForceSingle::<f32>::new(params)));

    // Pre-populate with some data
    {
        let mut idx = index.write();
        for i in 0..100u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            idx.add_vector(&v, i).unwrap();
        }
    }

    let num_readers = 4;
    let num_writers = 2;
    let read_count = Arc::new(AtomicUsize::new(0));
    let write_count = Arc::new(AtomicUsize::new(0));

    // Spawn reader threads
    let reader_handles: Vec<_> = (0..num_readers)
        .map(|_| {
            let index = Arc::clone(&index);
            let count = Arc::clone(&read_count);
            thread::spawn(move || {
                for i in 0..100 {
                    let idx = index.read();
                    let q = vec![(i % 100) as f32, 0.0, 0.0, 0.0];
                    let results = idx.top_k_query(&q, 5, None).unwrap();
                    assert!(!results.is_empty());
                    count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // Spawn writer threads
    let writer_handles: Vec<_> = (0..num_writers)
        .map(|t| {
            let index = Arc::clone(&index);
            let count = Arc::clone(&write_count);
            thread::spawn(move || {
                for i in 0..50 {
                    let label = 1000 + (t as u64) * 100 + (i as u64);
                    let v = vec![label as f32, 0.0, 0.0, 0.0];
                    index.write().add_vector(&v, label).unwrap();
                    count.fetch_add(1, Ordering::Relaxed);
                    thread::yield_now();
                }
            })
        })
        .collect();

    for handle in reader_handles {
        handle.join().expect("Reader thread panicked");
    }
    for handle in writer_handles {
        handle.join().expect("Writer thread panicked");
    }

    // Verify counts
    assert_eq!(read_count.load(Ordering::Relaxed), num_readers * 100);
    assert_eq!(write_count.load(Ordering::Relaxed), num_writers * 50);

    // Verify final index state
    let final_size = index.read().index_size();
    assert_eq!(final_size, 100 + num_writers * 50);
}

#[test]
fn test_hnsw_concurrent_read_write() {
    let params = HnswParams::new(4, Metric::L2)
        .with_m(8)
        .with_ef_construction(50);
    let index = Arc::new(parking_lot::RwLock::new(HnswSingle::<f32>::new(params)));

    // Pre-populate
    {
        let mut idx = index.write();
        for i in 0..100u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            idx.add_vector(&v, i).unwrap();
        }
    }

    let num_readers = 4;
    let num_writers = 2;

    let reader_handles: Vec<_> = (0..num_readers)
        .map(|_| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..50 {
                    let idx = index.read();
                    let q = vec![(i % 100) as f32, 0.0, 0.0, 0.0];
                    let results = idx.top_k_query(&q, 5, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        })
        .collect();

    let writer_handles: Vec<_> = (0..num_writers)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..25 {
                    let label = 1000 + (t as u64) * 100 + (i as u64);
                    let v = vec![label as f32, 0.0, 0.0, 0.0];
                    index.write().add_vector(&v, label).unwrap();
                    thread::yield_now();
                }
            })
        })
        .collect();

    for handle in reader_handles {
        handle.join().expect("Reader thread panicked");
    }
    for handle in writer_handles {
        handle.join().expect("Writer thread panicked");
    }

    let final_size = index.read().index_size();
    assert_eq!(final_size, 100 + num_writers * 25);
}

#[test]
fn test_concurrent_add_delete() {
    let params = BruteForceParams::new(4, Metric::L2);
    let index = Arc::new(parking_lot::RwLock::new(BruteForceSingle::<f32>::new(params)));

    // Pre-populate
    {
        let mut idx = index.write();
        for i in 0..200u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            idx.add_vector(&v, i).unwrap();
        }
    }

    let add_count = Arc::new(AtomicUsize::new(0));
    let delete_count = Arc::new(AtomicUsize::new(0));

    // Adder thread
    let add_idx = Arc::clone(&index);
    let add_cnt = Arc::clone(&add_count);
    let add_handle = thread::spawn(move || {
        for i in 200..300u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            add_idx.write().add_vector(&v, i).unwrap();
            add_cnt.fetch_add(1, Ordering::Relaxed);
        }
    });

    // Deleter thread
    let del_idx = Arc::clone(&index);
    let del_cnt = Arc::clone(&delete_count);
    let delete_handle = thread::spawn(move || {
        for i in 0..100u64 {
            // Deletes might fail if already deleted
            if del_idx.write().delete_vector(i).is_ok() {
                del_cnt.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    // Query thread while modifications happen
    let query_idx = Arc::clone(&index);
    let query_handle = thread::spawn(move || {
        for i in 0..100 {
            let q = vec![(i + 100) as f32, 0.0, 0.0, 0.0];
            let idx = query_idx.read();
            let results = idx.top_k_query(&q, 5, None);
            // Query should not panic
            assert!(results.is_ok());
        }
    });

    add_handle.join().expect("Add thread panicked");
    delete_handle.join().expect("Delete thread panicked");
    query_handle.join().expect("Query thread panicked");

    // Verify final state
    let adds = add_count.load(Ordering::Relaxed);
    let deletes = delete_count.load(Ordering::Relaxed);
    let final_size = index.read().index_size();

    assert_eq!(adds, 100);
    assert_eq!(deletes, 100);
    assert_eq!(final_size, 200 + adds - deletes); // 200 initial + 100 adds - 100 deletes = 200
}

// ============================================================================
// Multi-Index Concurrent Modification Tests
// ============================================================================

#[test]
fn test_brute_force_multi_concurrent_read_write() {
    let params = BruteForceParams::new(4, Metric::L2);
    let index = Arc::new(parking_lot::RwLock::new(BruteForceMulti::<f32>::new(params)));

    // Pre-populate
    {
        let mut idx = index.write();
        for label in 0..50u64 {
            for i in 0..3 {
                let v = vec![label as f32 + i as f32 * 0.01, 0.0, 0.0, 0.0];
                idx.add_vector(&v, label).unwrap();
            }
        }
    }

    let num_readers = 4;
    let num_writers = 2;

    let reader_handles: Vec<_> = (0..num_readers)
        .map(|_| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..50 {
                    let idx = index.read();
                    let q = vec![(i % 50) as f32, 0.0, 0.0, 0.0];
                    let results = idx.top_k_query(&q, 5, None).unwrap();
                    assert!(!results.is_empty());
                }
            })
        })
        .collect();

    let writer_handles: Vec<_> = (0..num_writers)
        .map(|t| {
            let index = Arc::clone(&index);
            thread::spawn(move || {
                for i in 0..25 {
                    let label = 100 + (t as u64) * 50 + (i as u64);
                    let v = vec![label as f32, 0.0, 0.0, 0.0];
                    index.write().add_vector(&v, label).unwrap();
                    thread::yield_now();
                }
            })
        })
        .collect();

    for handle in reader_handles {
        handle.join().expect("Reader thread panicked");
    }
    for handle in writer_handles {
        handle.join().expect("Writer thread panicked");
    }

    let final_size = index.read().index_size();
    assert_eq!(final_size, 50 * 3 + num_writers * 25); // 150 initial + 50 added
}

// ============================================================================
// Extreme Contention Tests
// ============================================================================

#[test]
fn test_extreme_read_contention() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    for i in 0..100u64 {
        let v = vec![i as f32, 0.0, 0.0, 0.0];
        index.add_vector(&v, i).unwrap();
    }

    let index = Arc::new(index);
    let num_threads = 32; // Many threads
    let queries_per_thread = 500;
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let index = Arc::clone(&index);
            let count = Arc::clone(&success_count);
            thread::spawn(move || {
                for i in 0..queries_per_thread {
                    let q = vec![(i % 100) as f32, 0.0, 0.0, 0.0];
                    let results = index.top_k_query(&q, 3, None).unwrap();
                    if !results.is_empty() {
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(
        success_count.load(Ordering::Relaxed),
        num_threads * queries_per_thread
    );
}

#[test]
fn test_rapid_add_query_interleave() {
    let params = BruteForceParams::new(4, Metric::L2);
    let index = Arc::new(parking_lot::RwLock::new(BruteForceSingle::<f32>::new(params)));

    let num_iterations = 100;
    let operations_complete = Arc::new(AtomicUsize::new(0));

    // Thread that adds and immediately queries
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let index = Arc::clone(&index);
            let ops = Arc::clone(&operations_complete);
            thread::spawn(move || {
                for i in 0..num_iterations {
                    let label = (t * num_iterations + i) as u64;
                    let v = vec![label as f32, 0.0, 0.0, 0.0];

                    // Add
                    index.write().add_vector(&v, label).unwrap();

                    // Immediately query
                    let q = vec![label as f32, 0.0, 0.0, 0.0];
                    let results = index.read().top_k_query(&q, 1, None).unwrap();
                    assert!(!results.is_empty());

                    ops.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    assert_eq!(operations_complete.load(Ordering::Relaxed), 4 * num_iterations);
}

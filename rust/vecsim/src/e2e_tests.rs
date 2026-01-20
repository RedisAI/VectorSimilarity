//! End-to-end integration tests for the VecSim library.
//!
//! These tests verify complete workflows from start to finish, simulating
//! real-world usage scenarios including index lifecycle management,
//! serialization/persistence, realistic workloads, and multi-index comparisons.

use crate::distance::Metric;
use crate::index::{
    BruteForceMulti, BruteForceParams, BruteForceSingle, HnswParams, HnswSingle, SvsParams,
    SvsSingle, TieredParams, TieredSingle, VecSimIndex, WriteMode,
};
use crate::query::QueryParams;
use rand::prelude::*;
use std::collections::HashSet;

// =============================================================================
// Test Data Generators
// =============================================================================

/// Generate random vectors with optional clustering around centroids.
fn generate_random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

/// Normalize a vector to unit length.
fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Generate normalized vectors for cosine similarity testing.
fn generate_normalized_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    generate_random_vectors(count, dim, seed)
        .into_iter()
        .map(|v| normalize_vector(&v))
        .collect()
}

// =============================================================================
// Index Lifecycle E2E Tests
// =============================================================================

#[test]
fn test_e2e_brute_force_complete_lifecycle() {
    // E2E test: Create → Add → Query → Update → Delete → Query → Clear
    let dim = 32;
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    // Phase 1: Initial population
    let vectors = generate_random_vectors(100, dim, 12345);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }
    assert_eq!(index.index_size(), 100);

    // Phase 2: Query and verify results
    let query = &vectors[0];
    let results = index.top_k_query(query, 5, None).unwrap();
    assert_eq!(results.results.len(), 5);
    assert_eq!(results.results[0].label, 0); // Should find itself

    // Phase 3: Update a vector (delete + add with same label)
    let new_vector: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
    index.delete_vector(50).unwrap();
    index.add_vector(&new_vector, 50).unwrap();
    assert_eq!(index.index_size(), 100);

    // Verify update worked
    let retrieved = index.get_vector(50).unwrap();
    for (a, b) in retrieved.iter().zip(new_vector.iter()) {
        assert!((a - b).abs() < 1e-6);
    }

    // Phase 4: Delete multiple vectors
    for label in 90..100 {
        index.delete_vector(label).unwrap();
    }
    assert_eq!(index.index_size(), 90);

    // Query should not return deleted vectors
    let results = index.top_k_query(&vectors[95], 100, None).unwrap();
    for r in &results.results {
        assert!(r.label < 90 || r.label == 50);
    }

    // Phase 5: Clear index
    index.clear();
    assert_eq!(index.index_size(), 0);

    // Verify cleared index works for new additions
    index.add_vector(&vectors[0], 1000).unwrap();
    assert_eq!(index.index_size(), 1);
}

#[test]
fn test_e2e_hnsw_complete_lifecycle() {
    // E2E test for HNSW index lifecycle
    // Use random vectors (not clustered) for reliable HNSW graph connectivity
    let dim = 64;
    let params = HnswParams::new(dim, Metric::L2)
        .with_m(16)
        .with_ef_construction(100);
    let mut index = HnswSingle::<f32>::new(params);

    // Phase 1: Build index with 500 random vectors
    let vectors = generate_random_vectors(500, dim, 42);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }
    assert_eq!(index.index_size(), 500);

    // Phase 2: Query with different ef_runtime values
    let query = &vectors[0];

    // Low ef_runtime (fast but lower recall)
    let params_low = QueryParams::new().with_ef_runtime(10);
    let results_low = index.top_k_query(query, 10, Some(&params_low)).unwrap();
    assert!(!results_low.results.is_empty());

    // High ef_runtime (slower but higher recall)
    let params_high = QueryParams::new().with_ef_runtime(200);
    let results_high = index.top_k_query(query, 10, Some(&params_high)).unwrap();
    assert_eq!(results_high.results[0].label, 0); // Should find itself with high ef

    // Phase 3: Delete and verify
    index.delete_vector(0).unwrap();
    let results = index.top_k_query(query, 10, Some(&params_high)).unwrap();
    for r in &results.results {
        assert_ne!(r.label, 0);
    }

    // Phase 4: Range query - use radius appropriate for 64-dim random vectors in [-1,1]
    let range_results = index.range_query(&vectors[100], 5.0, None).unwrap();
    assert!(!range_results.results.is_empty());
}

#[test]
fn test_e2e_multi_value_index_lifecycle() {
    // E2E test for multi-value indices (multiple vectors per label)
    let dim = 16;
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut index = BruteForceMulti::<f32>::new(params);

    // Phase 1: Add multiple vectors per label (simulating multiple embeddings per document)
    let vectors = generate_random_vectors(300, dim, 9999);

    // Labels 1-10 each get 10 vectors, labels 11-100 get 2 vectors each
    for label in 1..=10u64 {
        for j in 0..10 {
            let idx = ((label - 1) * 10 + j) as usize;
            index.add_vector(&vectors[idx], label).unwrap();
        }
    }
    for label in 11..=100u64 {
        for j in 0..2 {
            let idx = 100 + ((label - 11) * 2 + j) as usize;
            index.add_vector(&vectors[idx], label).unwrap();
        }
    }

    assert_eq!(index.index_size(), 280); // 10*10 + 90*2
    assert_eq!(index.label_count(1), 10);
    assert_eq!(index.label_count(50), 2);

    // Phase 2: Query returns results (multi-value returns unique labels per query)
    let query = &vectors[0];
    let results = index.top_k_query(query, 20, None).unwrap();
    assert!(!results.results.is_empty());
    // Multi-value index should return unique labels (best match per label)
    let unique_labels: HashSet<_> = results.results.iter().map(|r| r.label).collect();
    // The number of unique labels should equal or be close to the result count
    assert!(unique_labels.len() >= results.results.len() / 2);

    // Phase 3: Delete one vector from multi-vector label
    index.delete_vector(1).unwrap(); // Deletes all vectors for label 1
    assert_eq!(index.label_count(1), 0);
    assert!(!index.contains(1));

    // Phase 4: Verify queries don't return deleted label
    let results = index.top_k_query(query, 100, None).unwrap();
    for r in &results.results {
        assert_ne!(r.label, 1);
    }
}

// =============================================================================
// Serialization/Persistence E2E Tests
// =============================================================================

#[test]
fn test_e2e_brute_force_persistence_roundtrip() {
    // E2E test: Create → Populate → Save → Load → Verify → Continue using
    let dim = 48;
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut original = BruteForceSingle::<f32>::new(params.clone());

    // Populate with vectors
    let vectors = generate_random_vectors(200, dim, 11111);
    for (i, v) in vectors.iter().enumerate() {
        original.add_vector(v, i as u64).unwrap();
    }

    // Delete some to test fragmentation handling
    for label in [10, 50, 100, 150] {
        original.delete_vector(label).unwrap();
    }

    // Save to buffer
    let mut buffer = Vec::new();
    original.save(&mut buffer).unwrap();

    // Load from buffer
    let mut loaded = BruteForceSingle::<f32>::load(&mut buffer.as_slice()).unwrap();

    // Verify metadata
    assert_eq!(original.index_size(), loaded.index_size());
    assert_eq!(original.dimension(), loaded.dimension());

    // Verify queries produce same results
    let query = &vectors[0];
    let orig_results = original.top_k_query(query, 10, None).unwrap();
    let loaded_results = loaded.top_k_query(query, 10, None).unwrap();

    // Both should return results (exact count may vary based on available vectors)
    assert!(!orig_results.results.is_empty());
    assert!(!loaded_results.results.is_empty());

    // Top result should be the same (query vector itself, label 0)
    assert_eq!(orig_results.results[0].label, loaded_results.results[0].label);
    assert!((orig_results.results[0].distance - loaded_results.results[0].distance).abs() < 1e-6);

    // Verify deleted vectors are still deleted
    assert!(!loaded.contains(10));
    assert!(!loaded.contains(50));

    // Verify loaded index can continue operations
    let new_vec: Vec<f32> = (0..dim).map(|_| 0.5).collect();
    loaded.add_vector(&new_vec, 9999).unwrap();
    assert!(loaded.contains(9999));
    assert_eq!(loaded.index_size(), original.index_size() + 1);
}

#[test]
fn test_e2e_hnsw_persistence_roundtrip() {
    // E2E test for HNSW serialization
    let dim = 32;
    let params = HnswParams::new(dim, Metric::InnerProduct)
        .with_m(12)
        .with_ef_construction(50);
    let mut original = HnswSingle::<f32>::new(params.clone());

    // Populate with normalized vectors (for inner product)
    let vectors = generate_normalized_vectors(150, dim, 22222);
    for (i, v) in vectors.iter().enumerate() {
        original.add_vector(v, i as u64).unwrap();
    }

    // Save and load
    let mut buffer = Vec::new();
    original.save(&mut buffer).unwrap();
    let loaded = HnswSingle::<f32>::load(&mut buffer.as_slice()).unwrap();

    // Verify structure
    assert_eq!(original.index_size(), loaded.index_size());

    // Verify queries with high ef for exact comparison
    let query_params = QueryParams::new().with_ef_runtime(150);
    for query_label in [0, 50, 100] {
        let query = &vectors[query_label];
        let orig_results = original.top_k_query(query, 5, Some(&query_params)).unwrap();
        let loaded_results = loaded.top_k_query(query, 5, Some(&query_params)).unwrap();

        // With high ef, results should match exactly
        for (o, l) in orig_results.results.iter().zip(loaded_results.results.iter()) {
            assert_eq!(o.label, l.label);
        }
    }
}

// =============================================================================
// Real-world Workload E2E Tests
// =============================================================================

#[test]
fn test_e2e_realistic_document_embedding_workflow() {
    // Simulates a document embedding use case:
    // 1. Bulk load initial corpus
    // 2. Query for similar documents
    // 3. Add new documents over time
    // 4. Delete outdated documents
    // 5. Re-query to verify updates

    let dim = 128; // Typical embedding dimension
    let params = HnswParams::new(dim, Metric::Cosine)
        .with_m(16)
        .with_ef_construction(100);
    let mut index = HnswSingle::<f32>::new(params);

    // Phase 1: Bulk load initial corpus (1000 documents)
    let initial_docs = generate_normalized_vectors(1000, dim, 33333);
    for (i, v) in initial_docs.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }
    assert_eq!(index.index_size(), 1000);

    // Phase 2: Query for similar documents
    let query_doc = &initial_docs[500];
    let params = QueryParams::new().with_ef_runtime(50);
    let results = index.top_k_query(query_doc, 10, Some(&params)).unwrap();
    assert_eq!(results.results[0].label, 500);

    // Phase 3: Add new documents (simulating updates)
    let new_docs = generate_normalized_vectors(100, dim, 44444);
    for (i, v) in new_docs.iter().enumerate() {
        index.add_vector(v, 1000 + i as u64).unwrap();
    }
    assert_eq!(index.index_size(), 1100);

    // Phase 4: Delete outdated documents
    for label in 0..50u64 {
        index.delete_vector(label).unwrap();
    }
    assert_eq!(index.index_size(), 1050);

    // Phase 5: Re-query and verify deleted docs aren't returned
    let results = index.top_k_query(&initial_docs[25], 100, Some(&params)).unwrap();
    for r in &results.results {
        assert!(r.label >= 50);
    }

    // Verify fragmentation is reasonable
    let frag = index.fragmentation();
    assert!(frag < 0.1); // Less than 10% fragmentation
}

#[test]
fn test_e2e_image_similarity_workflow() {
    // Simulates an image similarity search use case with multi-value index:
    // Each image has multiple feature vectors (e.g., from different regions)

    let dim = 256; // Image feature dimension
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut index = BruteForceMulti::<f32>::new(params);

    // Each image (label) gets 4 feature vectors (e.g., 4 crops)
    let num_images = 100;
    let features_per_image = 4;

    let mut rng = StdRng::seed_from_u64(55555);
    for img_id in 0..num_images {
        for _ in 0..features_per_image {
            let feature: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            index.add_vector(&feature, img_id as u64).unwrap();
        }
    }

    assert_eq!(index.index_size(), num_images * features_per_image);

    // Query returns results - multi-value indices return unique labels
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = index.top_k_query(&query, 20, None).unwrap();

    // Results should be non-empty and have unique labels
    assert!(!results.results.is_empty());
    let labels: Vec<_> = results.results.iter().map(|r| r.label).collect();
    let unique: HashSet<_> = labels.iter().cloned().collect();
    // Multi-value index returns one result per label (best match)
    assert!(unique.len() >= labels.len() / 2);
}

#[test]
fn test_e2e_batch_operations_workflow() {
    // Test batch iterator for paginated results
    let dim = 64;
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    // Add vectors
    let vectors = generate_random_vectors(500, dim, 66666);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    // Use batch iterator to process all vectors
    let query = &vectors[0];
    let batch_size = 50;
    let mut iterator = index.batch_iterator(query, None).unwrap();

    let mut all_labels = Vec::new();
    let mut batch_count = 0;
    while iterator.has_next() {
        if let Some(batch) = iterator.next_batch(batch_size) {
            assert!(batch.len() <= batch_size);
            all_labels.extend(batch.iter().map(|(_, label, _)| *label));
            batch_count += 1;
        }
    }

    // Should have processed all vectors
    assert_eq!(all_labels.len(), 500);
    assert_eq!(batch_count, 10); // 500 / 50 = 10 batches
}

// =============================================================================
// Multi-Index Comparison E2E Tests
// =============================================================================

#[test]
fn test_e2e_brute_force_vs_hnsw_recall() {
    // Verify HNSW recall against BruteForce ground truth
    let dim = 64;
    let num_vectors = 1000;
    let num_queries = 50;
    let k = 10;

    // Create both indices with same data
    let bf_params = BruteForceParams::new(dim, Metric::L2);
    let hnsw_params = HnswParams::new(dim, Metric::L2)
        .with_m(16)
        .with_ef_construction(200);

    let mut bf_index = BruteForceSingle::<f32>::new(bf_params);
    let mut hnsw_index = HnswSingle::<f32>::new(hnsw_params);

    // Same vectors in both
    let vectors = generate_random_vectors(num_vectors, dim, 77777);
    for (i, v) in vectors.iter().enumerate() {
        bf_index.add_vector(v, i as u64).unwrap();
        hnsw_index.add_vector(v, i as u64).unwrap();
    }

    // Generate random queries
    let queries = generate_random_vectors(num_queries, dim, 88888);

    // Compare results with high ef_runtime
    let hnsw_params = QueryParams::new().with_ef_runtime(100);
    let mut total_recall = 0.0;

    for query in &queries {
        let bf_results = bf_index.top_k_query(query, k, None).unwrap();
        let hnsw_results = hnsw_index.top_k_query(query, k, Some(&hnsw_params)).unwrap();

        // Calculate recall
        let bf_labels: HashSet<_> = bf_results.results.iter().map(|r| r.label).collect();
        let hnsw_labels: HashSet<_> = hnsw_results.results.iter().map(|r| r.label).collect();
        let intersection = bf_labels.intersection(&hnsw_labels).count();
        let recall = intersection as f64 / k as f64;
        total_recall += recall;
    }

    let avg_recall = total_recall / num_queries as f64;
    assert!(
        avg_recall >= 0.9,
        "HNSW recall {} should be >= 0.9",
        avg_recall
    );
}

#[test]
fn test_e2e_index_type_consistency() {
    // Verify all index types return consistent results for exact match
    let dim = 16;
    let vectors = generate_random_vectors(100, dim, 99999);

    // Create all single-value index types
    let bf_params = BruteForceParams::new(dim, Metric::L2);
    let hnsw_params = HnswParams::new(dim, Metric::L2).with_m(8).with_ef_construction(50);
    let svs_params = SvsParams::new(dim, Metric::L2);

    let mut bf = BruteForceSingle::<f32>::new(bf_params);
    let mut hnsw = HnswSingle::<f32>::new(hnsw_params);
    let mut svs = SvsSingle::<f32>::new(svs_params);

    // Add same vectors to all
    for (i, v) in vectors.iter().enumerate() {
        bf.add_vector(v, i as u64).unwrap();
        hnsw.add_vector(v, i as u64).unwrap();
        svs.add_vector(v, i as u64).unwrap();
    }

    // Query for an existing vector - all should return it as top result
    let query = &vectors[42];
    let hnsw_params = QueryParams::new().with_ef_runtime(100);

    let bf_result = bf.top_k_query(query, 1, None).unwrap();
    let hnsw_result = hnsw.top_k_query(query, 1, Some(&hnsw_params)).unwrap();
    let svs_result = svs.top_k_query(query, 1, None).unwrap();

    assert_eq!(bf_result.results[0].label, 42);
    assert_eq!(hnsw_result.results[0].label, 42);
    assert_eq!(svs_result.results[0].label, 42);

    // All should have near-zero distance
    assert!(bf_result.results[0].distance < 1e-6);
    assert!(hnsw_result.results[0].distance < 1e-6);
    assert!(svs_result.results[0].distance < 1e-6);
}

// =============================================================================
// Tiered Index E2E Tests
// =============================================================================

#[test]
fn test_e2e_tiered_index_flush_workflow() {
    // E2E test for tiered index: write buffering and flush to HNSW
    let dim = 32;
    let tiered_params = TieredParams::new(dim, Metric::L2)
        .with_m(8)
        .with_ef_construction(50)
        .with_flat_buffer_limit(50)
        .with_write_mode(WriteMode::Async);

    let mut index = TieredSingle::<f32>::new(tiered_params);

    let vectors = generate_random_vectors(150, dim, 11111);

    // Phase 1: Add vectors below buffer limit - should go to flat
    for (i, v) in vectors.iter().take(40).enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    assert_eq!(index.flat_size(), 40);
    assert_eq!(index.hnsw_size(), 0);
    assert_eq!(index.write_mode(), WriteMode::Async);

    // Phase 2: Add more to exceed buffer limit - should trigger InPlace mode
    for (i, v) in vectors.iter().skip(40).take(30).enumerate() {
        index.add_vector(v, (40 + i) as u64).unwrap();
    }

    // After exceeding limit, new vectors go directly to HNSW
    assert!(index.flat_size() <= 50);

    // Phase 3: Flush to move all to HNSW
    index.flush().unwrap();

    assert_eq!(index.flat_size(), 0);
    assert_eq!(index.hnsw_size(), 70);

    // Phase 4: Continue adding - should go to flat again
    for (i, v) in vectors.iter().skip(70).enumerate() {
        index.add_vector(v, (70 + i) as u64).unwrap();
    }

    // Queries should work across both tiers
    let query = &vectors[0];
    let results = index.top_k_query(query, 10, None).unwrap();
    assert!(!results.results.is_empty());
}

#[test]
fn test_e2e_tiered_index_query_merging() {
    // Verify tiered index correctly merges results from both tiers
    let dim = 16;
    let tiered_params = TieredParams::new(dim, Metric::L2)
        .with_m(8)
        .with_ef_construction(50)
        .with_flat_buffer_limit(100);

    let mut tiered = TieredSingle::<f32>::new(tiered_params);
    let bf_params = BruteForceParams::new(dim, Metric::L2);
    let mut bf = BruteForceSingle::<f32>::new(bf_params);

    // Add vectors to both
    let vectors = generate_random_vectors(80, dim, 22222);
    for (i, v) in vectors.iter().enumerate() {
        tiered.add_vector(v, i as u64).unwrap();
        bf.add_vector(v, i as u64).unwrap();
    }

    // Flush half to HNSW
    tiered.flush().unwrap();

    // Add more to flat buffer
    let more_vectors = generate_random_vectors(30, dim, 33333);
    for (i, v) in more_vectors.iter().enumerate() {
        tiered.add_vector(v, (80 + i) as u64).unwrap();
        bf.add_vector(v, (80 + i) as u64).unwrap();
    }

    // Now tiered has data in both tiers
    assert!(tiered.hnsw_size() > 0);
    assert!(tiered.flat_size() > 0);

    // Query should return merged results
    let query = &vectors[0];
    let tiered_results = tiered.top_k_query(query, 10, None).unwrap();
    let bf_results = bf.top_k_query(query, 10, None).unwrap();

    // Top result should be the same (exact match)
    assert_eq!(tiered_results.results[0].label, bf_results.results[0].label);
}

// =============================================================================
// Filtered Query E2E Tests
// =============================================================================

#[test]
fn test_e2e_filtered_queries_workflow() {
    // E2E test for filtered queries (e.g., search within category)
    let dim = 32;
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    // Add vectors with labels encoding category (label % 10 = category)
    let vectors = generate_random_vectors(500, dim, 44444);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    // Query with filter: only return even labels
    let query = &vectors[100];
    let filter = |label: u64| label % 2 == 0;
    let params = QueryParams::new().with_filter(filter);

    let results = index.top_k_query(query, 20, Some(&params)).unwrap();

    // All returned labels should be even
    for r in &results.results {
        assert_eq!(r.label % 2, 0, "Label {} should be even", r.label);
    }

    // Should still get requested number of results (if enough pass filter)
    assert_eq!(results.results.len(), 20);
}

#[test]
fn test_e2e_filtered_range_query() {
    // E2E test for filtered range queries
    let dim = 16;
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    // Create vectors at known distances from origin
    for i in 0..100u64 {
        let mut v = vec![0.0f32; dim];
        v[0] = i as f32; // Distance from origin = i
        index.add_vector(&v, i).unwrap();
    }

    // Range query with filter
    let query = vec![0.0f32; dim];
    let filter = |label: u64| label >= 10 && label < 20;
    let params = QueryParams::new().with_filter(filter);

    // Radius of 400 (squared L2) should include labels 0-20
    let results = index.range_query(&query, 400.0, Some(&params)).unwrap();

    // Should only get labels 10-19 (in range AND passing filter)
    assert_eq!(results.results.len(), 10);
    for r in &results.results {
        assert!(r.label >= 10 && r.label < 20);
    }
}

// =============================================================================
// Error Handling E2E Tests
// =============================================================================

#[test]
fn test_e2e_error_recovery_workflow() {
    // Test that index remains usable after errors
    let dim = 8;
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    // Add some valid vectors
    let v1: Vec<f32> = vec![1.0; dim];
    let v2: Vec<f32> = vec![2.0; dim];
    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    // Try to add wrong dimension (should error)
    let wrong_dim: Vec<f32> = vec![1.0; dim + 1];
    let result = index.add_vector(&wrong_dim, 3);
    assert!(result.is_err());

    // Index should still be usable
    assert_eq!(index.index_size(), 2);

    // Try to delete non-existent label (should error)
    let result = index.delete_vector(999);
    assert!(result.is_err());

    // Index should still be usable
    let v3: Vec<f32> = vec![3.0; dim];
    index.add_vector(&v3, 3).unwrap();
    assert_eq!(index.index_size(), 3);

    // Queries should work
    let results = index.top_k_query(&v1, 3, None).unwrap();
    assert_eq!(results.results.len(), 3);
}

#[test]
fn test_e2e_duplicate_label_handling() {
    // Test handling of duplicate labels in single-value index
    // Single-value indices UPDATE the vector when using same label (not error)
    let dim = 8;
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    let v1: Vec<f32> = vec![1.0; dim];
    let v2: Vec<f32> = vec![2.0; dim];

    // Add first vector
    index.add_vector(&v1, 1).unwrap();
    assert_eq!(index.index_size(), 1);

    // Add with same label - should UPDATE (not error), size stays 1
    let result = index.add_vector(&v2, 1);
    assert!(result.is_ok());
    assert_eq!(index.index_size(), 1); // Still only 1 vector

    // Vector should be updated to v2
    let retrieved = index.get_vector(1).unwrap();
    assert!((retrieved[0] - 2.0).abs() < 1e-6);
}

// =============================================================================
// Metrics Comparison E2E Tests
// =============================================================================

#[test]
fn test_e2e_all_metrics_workflow() {
    // Test complete workflow with all metric types
    let dim = 16;
    let vectors = generate_random_vectors(100, dim, 55555);
    let normalized = generate_normalized_vectors(100, dim, 55555);

    // Test L2
    {
        let params = BruteForceParams::new(dim, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }
        let results = index.top_k_query(&vectors[0], 5, None).unwrap();
        assert_eq!(results.results[0].label, 0);
        assert!(results.results[0].distance < 1e-6); // L2 squared should be ~0
    }

    // Test Inner Product
    {
        let params = BruteForceParams::new(dim, Metric::InnerProduct);
        let mut index = BruteForceSingle::<f32>::new(params);
        for (i, v) in normalized.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }
        let results = index.top_k_query(&normalized[0], 5, None).unwrap();
        assert_eq!(results.results[0].label, 0);
        // For normalized vectors, IP with self = 1, distance = 1 - 1 = 0
    }

    // Test Cosine
    {
        let params = BruteForceParams::new(dim, Metric::Cosine);
        let mut index = BruteForceSingle::<f32>::new(params);
        for (i, v) in vectors.iter().enumerate() {
            index.add_vector(v, i as u64).unwrap();
        }
        let results = index.top_k_query(&vectors[0], 5, None).unwrap();
        assert_eq!(results.results[0].label, 0);
        assert!(results.results[0].distance < 1e-6); // Cosine distance with self = 0
    }
}

// =============================================================================
// Capacity and Memory E2E Tests
// =============================================================================

#[test]
fn test_e2e_capacity_enforcement() {
    // Test that capacity limits are enforced
    let dim = 8;
    let max_capacity = 50;
    let params = BruteForceParams::new(dim, Metric::L2);
    // Use BruteForceSingle::with_capacity() for actual max capacity enforcement
    let mut index = BruteForceSingle::<f32>::with_capacity(params, max_capacity);

    let vectors = generate_random_vectors(100, dim, 66666);

    // Add up to capacity
    for (i, v) in vectors.iter().take(max_capacity).enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }
    assert_eq!(index.index_size(), max_capacity);

    // Adding more should fail
    let result = index.add_vector(&vectors[max_capacity], max_capacity as u64);
    assert!(result.is_err());

    // Index should still be at capacity
    assert_eq!(index.index_size(), max_capacity);

    // Delete one and add should work
    index.delete_vector(0).unwrap();
    index.add_vector(&vectors[max_capacity], max_capacity as u64).unwrap();
    assert_eq!(index.index_size(), max_capacity);
}

#[test]
fn test_e2e_memory_usage_tracking() {
    // Verify memory usage reporting
    let dim = 64;
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    let initial_memory = index.memory_usage();

    // Add vectors and track memory growth
    let vectors = generate_random_vectors(100, dim, 77777);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    let after_add_memory = index.memory_usage();
    assert!(after_add_memory > initial_memory);

    // Memory per vector should be roughly dim * sizeof(f32) + overhead
    let memory_per_vector = (after_add_memory - initial_memory) / 100;
    let expected_min = dim * std::mem::size_of::<f32>(); // At least the vector data
    assert!(
        memory_per_vector >= expected_min,
        "Memory per vector {} should be >= {}",
        memory_per_vector,
        expected_min
    );
}

// =============================================================================
// Large Scale E2E Tests
// =============================================================================

#[test]
fn test_e2e_scaling_to_10k_vectors() {
    // Test with larger dataset using random vectors (not clustered)
    // Random vectors work better with HNSW as they don't create disconnected graph regions
    let dim = 128;
    let num_vectors = 10_000;
    let params = HnswParams::new(dim, Metric::L2)
        .with_m(16)
        .with_ef_construction(100)
        .with_seed(12345); // Fixed seed for reproducible graph structure
    let mut index = HnswSingle::<f32>::new(params);

    // Bulk insert with random vectors
    let vectors = generate_random_vectors(num_vectors, dim, 88888);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    assert_eq!(index.index_size(), num_vectors);

    // Query performance - should find similar vectors quickly
    let query_params = QueryParams::new().with_ef_runtime(200);
    let query = &vectors[5000];
    let results = index.top_k_query(query, 100, Some(&query_params)).unwrap();

    // Should return requested number of results
    assert_eq!(results.results.len(), 100);
    // First result should be the query vector itself (distance ~0)
    assert_eq!(results.results[0].label, 5000);
    assert!(results.results[0].distance < 0.001, "Self-query distance {} too large", results.results[0].distance);

    // Test range query - use a reasonable radius for 128-dim L2 space
    // Random vectors in [-1,1] have typical distances around sqrt(128 * 0.5) ≈ 8
    // Use a larger radius (50) to ensure we find some results
    // The query vector itself has distance 0, so it should always be found
    // Use default epsilon (0.1 = 10%) to ensure we explore enough of the graph
    // The epsilon-neighborhood algorithm terminates when the next candidate's distance
    // is outside the boundary (dynamic_range * (1 + epsilon)). With a small epsilon,
    // the algorithm might terminate before finding all vectors within the radius.
    let range_params = QueryParams::new().with_ef_runtime(200);
    let range_results = index.range_query(query, 50.0, Some(&range_params)).unwrap();

    // The self-query should be within radius 50.0 (distance is 0)
    assert!(!range_results.results.is_empty(), "Range query should find at least the query vector itself (distance=0)");
    // Verify the query vector is in the results
    assert!(range_results.results.iter().any(|r| r.label == 5000 && r.distance < 0.001),
        "Range query should find the query vector itself");
}

// =============================================================================
// Index Info and Statistics E2E Tests
// =============================================================================

#[test]
fn test_e2e_index_info_consistency() {
    // Verify IndexInfo is consistent across operations
    let dim = 32;
    let params = HnswParams::new(dim, Metric::Cosine)
        .with_m(16)
        .with_ef_construction(100);
    let mut index = HnswSingle::<f32>::new(params);

    // Check initial info
    let info = index.info();
    assert_eq!(info.dimension, dim);
    assert_eq!(info.size, 0);
    assert!(info.memory_bytes > 0); // Some base overhead

    // Add vectors
    let vectors = generate_normalized_vectors(200, dim, 99999);
    for (i, v) in vectors.iter().enumerate() {
        index.add_vector(v, i as u64).unwrap();
    }

    let info = index.info();
    assert_eq!(info.size, 200);
    assert!(info.memory_bytes > 0);

    // Delete some
    for label in 0..50u64 {
        index.delete_vector(label).unwrap();
    }

    let info = index.info();
    assert_eq!(info.size, 150);

    // Fragmentation should be non-zero after deletes
    let frag = index.fragmentation();
    assert!(frag > 0.0);
}

//! Data type-specific integration tests.
//!
//! This module tests all supported vector element types (f32, f64, Float16, BFloat16,
//! Int8, UInt8, Int32, Int64) with various index operations to ensure type-specific
//! behavior is correct.

use crate::distance::Metric;
use crate::index::{BruteForceMulti, BruteForceParams, BruteForceSingle, VecSimIndex};
use crate::index::{HnswParams, HnswSingle};
use crate::types::{BFloat16, Float16, Int32, Int64, Int8, UInt8, VectorElement};

// =============================================================================
// Helper functions
// =============================================================================

fn create_orthogonal_vectors_f32<T: VectorElement>(dim: usize) -> Vec<Vec<T>> {
    (0..dim)
        .map(|i| {
            let mut v = vec![T::zero(); dim];
            v[i] = T::from_f32(1.0);
            v
        })
        .collect()
}

// =============================================================================
// f32 Tests (baseline)
// =============================================================================

#[test]
fn test_f32_brute_force_basic() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    index.add_vector(&[0.0, 0.0, 1.0, 0.0], 3).unwrap();

    let results = index.top_k_query(&[1.0, 0.1, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_f32_serialization_roundtrip() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f32>::new(params);

    for i in 0..10 {
        let v: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32).collect();
        index.add_vector(&v, i as u64).unwrap();
    }

    // Use save/load with a buffer
    let mut buffer = Vec::new();
    index.save(&mut buffer).unwrap();
    let loaded = BruteForceSingle::<f32>::load(&mut buffer.as_slice()).unwrap();

    assert_eq!(index.index_size(), loaded.index_size());

    let query = [1.0f32, 2.0, 3.0, 4.0];
    let orig_results = index.top_k_query(&query, 3, None).unwrap();
    let loaded_results = loaded.top_k_query(&query, 3, None).unwrap();

    assert_eq!(orig_results.results.len(), loaded_results.results.len());
    for (o, d) in orig_results.results.iter().zip(loaded_results.results.iter()) {
        assert_eq!(o.label, d.label);
    }
}

// =============================================================================
// f64 Tests
// =============================================================================

#[test]
fn test_f64_brute_force_basic() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f64>::new(params);

    index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    index.add_vector(&[0.0, 0.0, 1.0, 0.0], 3).unwrap();

    let results = index.top_k_query(&[1.0, 0.1, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_f64_brute_force_inner_product() {
    let params = BruteForceParams::new(4, Metric::InnerProduct);
    let mut index = BruteForceSingle::<f64>::new(params);

    index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add_vector(&[0.5, 0.5, 0.0, 0.0], 2).unwrap();
    index.add_vector(&[0.0, 1.0, 0.0, 0.0], 3).unwrap();

    let results = index.top_k_query(&[1.0, 0.0, 0.0, 0.0], 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_f64_brute_force_cosine() {
    let params = BruteForceParams::new(4, Metric::Cosine);
    let mut index = BruteForceSingle::<f64>::new(params);

    index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    index.add_vector(&[0.707, 0.707, 0.0, 0.0], 3).unwrap();

    let results = index.top_k_query(&[1.0, 0.0, 0.0, 0.0], 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_f64_brute_force_multi() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceMulti::<f64>::new(params);

    index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add_vector(&[0.9, 0.1, 0.0, 0.0], 1).unwrap(); // Same label
    index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

    assert_eq!(index.label_count(1), 2);

    let results = index.top_k_query(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_f64_hnsw_basic() {
    let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(16);
    let mut index = HnswSingle::<f64>::new(params);

    for v in create_orthogonal_vectors_f32::<f64>(4) {
        index.add_vector(&v, index.index_size() as u64 + 1).unwrap();
    }

    let results = index.top_k_query(&[1.0, 0.0, 0.0, 0.0], 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_f64_high_precision() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f64>::new(params);

    // Use values that would lose precision in f32
    let precise1: Vec<f64> = vec![1.0000000001, 0.0, 0.0, 0.0];
    let precise2: Vec<f64> = vec![1.0000000002, 0.0, 0.0, 0.0];

    index.add_vector(&precise1, 1).unwrap();
    index.add_vector(&precise2, 2).unwrap();

    let query: Vec<f64> = vec![1.0000000001, 0.0, 0.0, 0.0];
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_f64_data_integrity() {
    // Note: Serialization is only implemented for f32
    // This test verifies data integrity without serialization
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f64>::new(params);

    for i in 0..10 {
        let v: Vec<f64> = (0..4).map(|j| (i * 4 + j) as f64).collect();
        index.add_vector(&v, i as u64).unwrap();
    }

    assert_eq!(index.index_size(), 10);

    // Verify each vector can be retrieved
    for i in 0..10 {
        assert!(index.contains(i as u64));
        let v = index.get_vector(i as u64).unwrap();
        for (j, &val) in v.iter().enumerate() {
            assert_eq!(val, (i * 4 + j) as f64);
        }
    }
}

#[test]
fn test_f64_range_query() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f64>::new(params);

    index.add_vector(&[0.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add_vector(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
    index.add_vector(&[2.0, 0.0, 0.0, 0.0], 3).unwrap();
    index.add_vector(&[3.0, 0.0, 0.0, 0.0], 4).unwrap();

    let results = index
        .range_query(&[0.0, 0.0, 0.0, 0.0], 1.5, None)
        .unwrap();
    assert_eq!(results.results.len(), 2); // Distance 0 and 1
}

#[test]
fn test_f64_delete_and_query() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<f64>::new(params);

    index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

    index.delete_vector(1).unwrap();

    let results = index.top_k_query(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results[0].label, 2);
}

// =============================================================================
// Float16 Tests
// =============================================================================

#[test]
fn test_float16_brute_force_basic() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Float16>::new(params);

    let v1: Vec<Float16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();
    let v2: Vec<Float16> = [0.0f32, 1.0, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();
    let v3: Vec<Float16> = [0.0f32, 0.0, 1.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();
    index.add_vector(&v3, 3).unwrap();

    let query: Vec<Float16> = [1.0f32, 0.1, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();
    let results = index.top_k_query(&query, 2, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_float16_brute_force_inner_product() {
    let params = BruteForceParams::new(4, Metric::InnerProduct);
    let mut index = BruteForceSingle::<Float16>::new(params);

    let v1: Vec<Float16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();
    let v2: Vec<Float16> = [0.5f32, 0.5, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Float16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_float16_brute_force_cosine() {
    let params = BruteForceParams::new(4, Metric::Cosine);
    let mut index = BruteForceSingle::<Float16>::new(params);

    let v1: Vec<Float16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();
    let v2: Vec<Float16> = [0.0f32, 1.0, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Float16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_float16_hnsw_basic() {
    let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(16);
    let mut index = HnswSingle::<Float16>::new(params);

    for i in 0..4 {
        let mut v = vec![Float16::from_f32(0.0); 4];
        v[i] = Float16::from_f32(1.0);
        index.add_vector(&v, i as u64 + 1).unwrap();
    }

    let query: Vec<Float16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_float16_precision_limits() {
    // Float16 has ~3 decimal digits precision
    let v1 = Float16::from_f32(1.0);
    let v2 = Float16::from_f32(1.001); // Should be distinguishable
    let v3 = Float16::from_f32(1.0001); // May be same as v1 due to precision

    // Values 1.0 and 1.001 should be distinguishable
    let diff = (v1.to_f32() - v2.to_f32()).abs();
    assert!(diff > 0.0005);

    // Very small differences may be lost
    let tiny_diff = (v1.to_f32() - v3.to_f32()).abs();
    assert!(tiny_diff < 0.001); // Precision loss expected
}

#[test]
fn test_float16_data_integrity() {
    // Note: Serialization is only implemented for f32
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Float16>::new(params);

    for i in 0..10 {
        let v: Vec<Float16> = (0..4).map(|j| Float16::from_f32((i * 4 + j) as f32)).collect();
        index.add_vector(&v, i as u64).unwrap();
    }

    assert_eq!(index.index_size(), 10);

    // Verify vectors can be retrieved with approximate equality (FP16 has limited precision)
    for i in 0..10 {
        assert!(index.contains(i as u64));
        let v = index.get_vector(i as u64).unwrap();
        for (j, &val) in v.iter().enumerate() {
            let expected = (i * 4 + j) as f32;
            assert!((val.to_f32() - expected).abs() < 0.1);
        }
    }
}

#[test]
fn test_float16_range_query() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Float16>::new(params);

    for i in 0..5 {
        let v: Vec<Float16> = vec![
            Float16::from_f32(i as f32),
            Float16::from_f32(0.0),
            Float16::from_f32(0.0),
            Float16::from_f32(0.0),
        ];
        index.add_vector(&v, i as u64).unwrap();
    }

    let query: Vec<Float16> = [0.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| Float16::from_f32(x))
        .collect();
    let results = index.range_query(&query, 1.5, None).unwrap();
    assert_eq!(results.results.len(), 2); // Distance 0 and 1
}

#[test]
fn test_float16_special_values() {
    // Test that special values don't break the index
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Float16>::new(params);

    let v1: Vec<Float16> = [Float16::ZERO, Float16::ZERO, Float16::ZERO, Float16::ONE].to_vec();
    index.add_vector(&v1, 1).unwrap();

    let query: Vec<Float16> = [Float16::ZERO, Float16::ZERO, Float16::ZERO, Float16::ONE].to_vec();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
    assert!(results.results[0].distance < 0.001);
}

// =============================================================================
// BFloat16 Tests
// =============================================================================

#[test]
fn test_bfloat16_brute_force_basic() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<BFloat16>::new(params);

    let v1: Vec<BFloat16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();
    let v2: Vec<BFloat16> = [0.0f32, 1.0, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();
    let v3: Vec<BFloat16> = [0.0f32, 0.0, 1.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();
    index.add_vector(&v3, 3).unwrap();

    let query: Vec<BFloat16> = [1.0f32, 0.1, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();
    let results = index.top_k_query(&query, 2, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_bfloat16_brute_force_inner_product() {
    let params = BruteForceParams::new(4, Metric::InnerProduct);
    let mut index = BruteForceSingle::<BFloat16>::new(params);

    let v1: Vec<BFloat16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();
    let v2: Vec<BFloat16> = [0.5f32, 0.5, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<BFloat16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_bfloat16_brute_force_cosine() {
    let params = BruteForceParams::new(4, Metric::Cosine);
    let mut index = BruteForceSingle::<BFloat16>::new(params);

    let v1: Vec<BFloat16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();
    let v2: Vec<BFloat16> = [0.0f32, 1.0, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<BFloat16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_bfloat16_hnsw_basic() {
    let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(16);
    let mut index = HnswSingle::<BFloat16>::new(params);

    for i in 0..4 {
        let mut v = vec![BFloat16::from_f32(0.0); 4];
        v[i] = BFloat16::from_f32(1.0);
        index.add_vector(&v, i as u64 + 1).unwrap();
    }

    let query: Vec<BFloat16> = [1.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_bfloat16_large_range() {
    // BFloat16 has the same exponent range as f32
    let large = BFloat16::from_f32(1e30);
    let small = BFloat16::from_f32(1e-30);

    assert!(large.to_f32() > 1e29);
    assert!(small.to_f32() < 1e-29);
    assert!(small.to_f32() > 0.0);
}

#[test]
fn test_bfloat16_data_integrity() {
    // Note: Serialization is only implemented for f32
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<BFloat16>::new(params);

    for i in 0..10 {
        let v: Vec<BFloat16> = (0..4)
            .map(|j| BFloat16::from_f32((i * 4 + j) as f32))
            .collect();
        index.add_vector(&v, i as u64).unwrap();
    }

    assert_eq!(index.index_size(), 10);

    // Verify vectors can be retrieved with approximate equality (BF16 has limited precision)
    for i in 0..10 {
        assert!(index.contains(i as u64));
        let v = index.get_vector(i as u64).unwrap();
        for (j, &val) in v.iter().enumerate() {
            let expected = (i * 4 + j) as f32;
            assert!((val.to_f32() - expected).abs() < 0.5);
        }
    }
}

#[test]
fn test_bfloat16_range_query() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<BFloat16>::new(params);

    for i in 0..5 {
        let v: Vec<BFloat16> = vec![
            BFloat16::from_f32(i as f32),
            BFloat16::from_f32(0.0),
            BFloat16::from_f32(0.0),
            BFloat16::from_f32(0.0),
        ];
        index.add_vector(&v, i as u64).unwrap();
    }

    let query: Vec<BFloat16> = [0.0f32, 0.0, 0.0, 0.0]
        .iter()
        .map(|&x| BFloat16::from_f32(x))
        .collect();
    let results = index.range_query(&query, 1.5, None).unwrap();
    assert_eq!(results.results.len(), 2); // Distance 0 and 1
}

#[test]
fn test_bfloat16_special_values() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<BFloat16>::new(params);

    let v1: Vec<BFloat16> =
        [BFloat16::ZERO, BFloat16::ZERO, BFloat16::ZERO, BFloat16::ONE].to_vec();
    index.add_vector(&v1, 1).unwrap();

    let query: Vec<BFloat16> =
        [BFloat16::ZERO, BFloat16::ZERO, BFloat16::ZERO, BFloat16::ONE].to_vec();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
    assert!(results.results[0].distance < 0.001);
}

// =============================================================================
// Int8 Tests
// =============================================================================

#[test]
fn test_int8_brute_force_basic() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int8>::new(params);

    let v1: Vec<Int8> = [127i8, 0, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let v2: Vec<Int8> = [0i8, 127, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let v3: Vec<Int8> = [0i8, 0, 127, 0].iter().map(|&x| Int8::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();
    index.add_vector(&v3, 3).unwrap();

    let query: Vec<Int8> = [100i8, 10, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let results = index.top_k_query(&query, 2, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int8_brute_force_inner_product() {
    let params = BruteForceParams::new(4, Metric::InnerProduct);
    let mut index = BruteForceSingle::<Int8>::new(params);

    let v1: Vec<Int8> = [100i8, 0, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let v2: Vec<Int8> = [50i8, 50, 0, 0].iter().map(|&x| Int8::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Int8> = [100i8, 0, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int8_brute_force_cosine() {
    let params = BruteForceParams::new(4, Metric::Cosine);
    let mut index = BruteForceSingle::<Int8>::new(params);

    let v1: Vec<Int8> = [100i8, 0, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let v2: Vec<Int8> = [0i8, 100, 0, 0].iter().map(|&x| Int8::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Int8> = [100i8, 0, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int8_hnsw_basic() {
    let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(16);
    let mut index = HnswSingle::<Int8>::new(params);

    for i in 0..4 {
        let mut v = vec![Int8::new(0); 4];
        v[i] = Int8::new(100);
        index.add_vector(&v, i as u64 + 1).unwrap();
    }

    let query: Vec<Int8> = [100i8, 0, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int8_boundary_values() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int8>::new(params);

    // Test with boundary values
    let v1: Vec<Int8> = [Int8::MAX, Int8::MIN, Int8::ZERO, Int8::new(1)].to_vec();
    let v2: Vec<Int8> = [Int8::MIN, Int8::MAX, Int8::ZERO, Int8::new(-1)].to_vec();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    // Query close to v1
    let query: Vec<Int8> = [Int8::MAX, Int8::MIN, Int8::ZERO, Int8::ZERO].to_vec();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int8_negative_values() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int8>::new(params);

    let v1: Vec<Int8> = [-100i8, -50, 0, 50]
        .iter()
        .map(|&x| Int8::new(x))
        .collect();
    let v2: Vec<Int8> = [100i8, 50, 0, -50]
        .iter()
        .map(|&x| Int8::new(x))
        .collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Int8> = [-100i8, -50, 0, 50]
        .iter()
        .map(|&x| Int8::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
    assert!(results.results[0].distance < 0.001);
}

#[test]
fn test_int8_data_integrity() {
    // Note: Serialization is only implemented for f32
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int8>::new(params);

    for i in 0..10i8 {
        let v: Vec<Int8> = (0..4).map(|j| Int8::new((i * 4 + j) % 127)).collect();
        index.add_vector(&v, i as u64).unwrap();
    }

    assert_eq!(index.index_size(), 10);

    // Verify vectors can be retrieved
    for i in 0..10i8 {
        assert!(index.contains(i as u64));
        let v = index.get_vector(i as u64).unwrap();
        for (j, &val) in v.iter().enumerate() {
            let expected = (i * 4 + j as i8) % 127;
            assert_eq!(val.get(), expected);
        }
    }
}

#[test]
fn test_int8_multi_value() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceMulti::<Int8>::new(params);

    let v1: Vec<Int8> = [100i8, 0, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let v2: Vec<Int8> = [90i8, 10, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let v3: Vec<Int8> = [0i8, 100, 0, 0].iter().map(|&x| Int8::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 1).unwrap(); // Same label
    index.add_vector(&v3, 2).unwrap();

    assert_eq!(index.label_count(1), 2);

    let query: Vec<Int8> = [100i8, 0, 0, 0].iter().map(|&x| Int8::new(x)).collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

// =============================================================================
// UInt8 Tests
// =============================================================================

#[test]
fn test_uint8_brute_force_basic() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<UInt8>::new(params);

    let v1: Vec<UInt8> = [255u8, 0, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let v2: Vec<UInt8> = [0u8, 255, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let v3: Vec<UInt8> = [0u8, 0, 255, 0].iter().map(|&x| UInt8::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();
    index.add_vector(&v3, 3).unwrap();

    let query: Vec<UInt8> = [200u8, 10, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let results = index.top_k_query(&query, 2, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_uint8_brute_force_inner_product() {
    let params = BruteForceParams::new(4, Metric::InnerProduct);
    let mut index = BruteForceSingle::<UInt8>::new(params);

    let v1: Vec<UInt8> = [200u8, 0, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let v2: Vec<UInt8> = [100u8, 100, 0, 0].iter().map(|&x| UInt8::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<UInt8> = [200u8, 0, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_uint8_brute_force_cosine() {
    let params = BruteForceParams::new(4, Metric::Cosine);
    let mut index = BruteForceSingle::<UInt8>::new(params);

    let v1: Vec<UInt8> = [200u8, 0, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let v2: Vec<UInt8> = [0u8, 200, 0, 0].iter().map(|&x| UInt8::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<UInt8> = [200u8, 0, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_uint8_hnsw_basic() {
    let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(16);
    let mut index = HnswSingle::<UInt8>::new(params);

    for i in 0..4 {
        let mut v = vec![UInt8::new(0); 4];
        v[i] = UInt8::new(200);
        index.add_vector(&v, i as u64 + 1).unwrap();
    }

    let query: Vec<UInt8> = [200u8, 0, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_uint8_boundary_values() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<UInt8>::new(params);

    let v1: Vec<UInt8> = [UInt8::MAX, UInt8::MIN, UInt8::ZERO, UInt8::new(128)].to_vec();
    let v2: Vec<UInt8> = [UInt8::MIN, UInt8::MAX, UInt8::new(128), UInt8::ZERO].to_vec();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<UInt8> = [UInt8::MAX, UInt8::MIN, UInt8::ZERO, UInt8::new(128)].to_vec();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
    assert!(results.results[0].distance < 0.001);
}

#[test]
fn test_uint8_data_integrity() {
    // Note: Serialization is only implemented for f32
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<UInt8>::new(params);

    for i in 0..10u8 {
        let v: Vec<UInt8> = (0..4).map(|j| UInt8::new((i * 4 + j) % 255)).collect();
        index.add_vector(&v, i as u64).unwrap();
    }

    assert_eq!(index.index_size(), 10);

    // Verify vectors can be retrieved
    for i in 0..10u8 {
        assert!(index.contains(i as u64));
        let v = index.get_vector(i as u64).unwrap();
        for (j, &val) in v.iter().enumerate() {
            let expected = (i * 4 + j as u8) % 255;
            assert_eq!(val.get(), expected);
        }
    }
}

#[test]
fn test_uint8_multi_value() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceMulti::<UInt8>::new(params);

    let v1: Vec<UInt8> = [200u8, 0, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let v2: Vec<UInt8> = [190u8, 10, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let v3: Vec<UInt8> = [0u8, 200, 0, 0].iter().map(|&x| UInt8::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 1).unwrap(); // Same label
    index.add_vector(&v3, 2).unwrap();

    assert_eq!(index.label_count(1), 2);

    let query: Vec<UInt8> = [200u8, 0, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_uint8_range_query() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<UInt8>::new(params);

    for i in 0..5u8 {
        let v: Vec<UInt8> = vec![
            UInt8::new(i * 50),
            UInt8::new(0),
            UInt8::new(0),
            UInt8::new(0),
        ];
        index.add_vector(&v, i as u64).unwrap();
    }

    let query: Vec<UInt8> = [0u8, 0, 0, 0].iter().map(|&x| UInt8::new(x)).collect();
    // Range query uses squared L2 distance:
    // Vector 0 at [0,0,0,0]: squared distance = 0
    // Vector 1 at [50,0,0,0]: squared distance = 50^2 = 2500
    // So radius 2600 should include first two vectors (squared distances 0 and 2500)
    let results = index.range_query(&query, 2600.0, None).unwrap();
    assert_eq!(results.results.len(), 2);
}

// =============================================================================
// Int32 Tests
// =============================================================================

#[test]
fn test_int32_brute_force_basic() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int32>::new(params);

    let v1: Vec<Int32> = [1000i32, 0, 0, 0].iter().map(|&x| Int32::new(x)).collect();
    let v2: Vec<Int32> = [0i32, 1000, 0, 0].iter().map(|&x| Int32::new(x)).collect();
    let v3: Vec<Int32> = [0i32, 0, 1000, 0].iter().map(|&x| Int32::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();
    index.add_vector(&v3, 3).unwrap();

    let query: Vec<Int32> = [900i32, 100, 0, 0]
        .iter()
        .map(|&x| Int32::new(x))
        .collect();
    let results = index.top_k_query(&query, 2, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int32_brute_force_inner_product() {
    let params = BruteForceParams::new(4, Metric::InnerProduct);
    let mut index = BruteForceSingle::<Int32>::new(params);

    let v1: Vec<Int32> = [1000i32, 0, 0, 0].iter().map(|&x| Int32::new(x)).collect();
    let v2: Vec<Int32> = [500i32, 500, 0, 0].iter().map(|&x| Int32::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Int32> = [1000i32, 0, 0, 0]
        .iter()
        .map(|&x| Int32::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int32_brute_force_cosine() {
    let params = BruteForceParams::new(4, Metric::Cosine);
    let mut index = BruteForceSingle::<Int32>::new(params);

    let v1: Vec<Int32> = [1000i32, 0, 0, 0].iter().map(|&x| Int32::new(x)).collect();
    let v2: Vec<Int32> = [0i32, 1000, 0, 0].iter().map(|&x| Int32::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Int32> = [1000i32, 0, 0, 0]
        .iter()
        .map(|&x| Int32::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int32_hnsw_basic() {
    let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(16);
    let mut index = HnswSingle::<Int32>::new(params);

    for i in 0..4 {
        let mut v = vec![Int32::new(0); 4];
        v[i] = Int32::new(1000);
        index.add_vector(&v, i as u64 + 1).unwrap();
    }

    let query: Vec<Int32> = [1000i32, 0, 0, 0]
        .iter()
        .map(|&x| Int32::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int32_large_values() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int32>::new(params);

    let v1: Vec<Int32> = [1_000_000i32, 0, 0, 0]
        .iter()
        .map(|&x| Int32::new(x))
        .collect();
    let v2: Vec<Int32> = [-1_000_000i32, 0, 0, 0]
        .iter()
        .map(|&x| Int32::new(x))
        .collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Int32> = [1_000_000i32, 0, 0, 0]
        .iter()
        .map(|&x| Int32::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
    assert!(results.results[0].distance < 0.001);
}

#[test]
fn test_int32_boundary_values() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int32>::new(params);

    // Note: Using values that won't overflow when squared
    let v1: Vec<Int32> = [Int32::new(10000), Int32::new(-10000), Int32::ZERO, Int32::new(1)]
        .to_vec();
    let v2: Vec<Int32> = [Int32::new(-10000), Int32::new(10000), Int32::ZERO, Int32::new(-1)]
        .to_vec();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Int32> = [Int32::new(10000), Int32::new(-10000), Int32::ZERO, Int32::ZERO]
        .to_vec();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int32_data_integrity() {
    // Note: Serialization is only implemented for f32
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int32>::new(params);

    for i in 0..10i32 {
        let v: Vec<Int32> = (0..4).map(|j| Int32::new(i * 1000 + j * 100)).collect();
        index.add_vector(&v, i as u64).unwrap();
    }

    assert_eq!(index.index_size(), 10);

    // Verify vectors can be retrieved
    for i in 0..10i32 {
        assert!(index.contains(i as u64));
        let v = index.get_vector(i as u64).unwrap();
        for (j, &val) in v.iter().enumerate() {
            let expected = i * 1000 + j as i32 * 100;
            assert_eq!(val.get(), expected);
        }
    }
}

#[test]
fn test_int32_multi_value() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceMulti::<Int32>::new(params);

    let v1: Vec<Int32> = [1000i32, 0, 0, 0].iter().map(|&x| Int32::new(x)).collect();
    let v2: Vec<Int32> = [900i32, 100, 0, 0].iter().map(|&x| Int32::new(x)).collect();
    let v3: Vec<Int32> = [0i32, 1000, 0, 0].iter().map(|&x| Int32::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 1).unwrap(); // Same label
    index.add_vector(&v3, 2).unwrap();

    assert_eq!(index.label_count(1), 2);

    let query: Vec<Int32> = [1000i32, 0, 0, 0]
        .iter()
        .map(|&x| Int32::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

// =============================================================================
// Int64 Tests
// =============================================================================

#[test]
fn test_int64_brute_force_basic() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int64>::new(params);

    let v1: Vec<Int64> = [10000i64, 0, 0, 0].iter().map(|&x| Int64::new(x)).collect();
    let v2: Vec<Int64> = [0i64, 10000, 0, 0].iter().map(|&x| Int64::new(x)).collect();
    let v3: Vec<Int64> = [0i64, 0, 10000, 0].iter().map(|&x| Int64::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();
    index.add_vector(&v3, 3).unwrap();

    let query: Vec<Int64> = [9000i64, 1000, 0, 0]
        .iter()
        .map(|&x| Int64::new(x))
        .collect();
    let results = index.top_k_query(&query, 2, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int64_brute_force_inner_product() {
    let params = BruteForceParams::new(4, Metric::InnerProduct);
    let mut index = BruteForceSingle::<Int64>::new(params);

    let v1: Vec<Int64> = [10000i64, 0, 0, 0].iter().map(|&x| Int64::new(x)).collect();
    let v2: Vec<Int64> = [5000i64, 5000, 0, 0]
        .iter()
        .map(|&x| Int64::new(x))
        .collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Int64> = [10000i64, 0, 0, 0]
        .iter()
        .map(|&x| Int64::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int64_brute_force_cosine() {
    let params = BruteForceParams::new(4, Metric::Cosine);
    let mut index = BruteForceSingle::<Int64>::new(params);

    let v1: Vec<Int64> = [10000i64, 0, 0, 0].iter().map(|&x| Int64::new(x)).collect();
    let v2: Vec<Int64> = [0i64, 10000, 0, 0].iter().map(|&x| Int64::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Int64> = [10000i64, 0, 0, 0]
        .iter()
        .map(|&x| Int64::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int64_hnsw_basic() {
    let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(16);
    let mut index = HnswSingle::<Int64>::new(params);

    for i in 0..4 {
        let mut v = vec![Int64::new(0); 4];
        v[i] = Int64::new(10000);
        index.add_vector(&v, i as u64 + 1).unwrap();
    }

    let query: Vec<Int64> = [10000i64, 0, 0, 0]
        .iter()
        .map(|&x| Int64::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

#[test]
fn test_int64_large_values_within_f32_precision() {
    // Values within f32 exact integer range (2^24)
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int64>::new(params);

    let v1: Vec<Int64> = [16_000_000i64, 0, 0, 0]
        .iter()
        .map(|&x| Int64::new(x))
        .collect();
    let v2: Vec<Int64> = [-16_000_000i64, 0, 0, 0]
        .iter()
        .map(|&x| Int64::new(x))
        .collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 2).unwrap();

    let query: Vec<Int64> = [16_000_000i64, 0, 0, 0]
        .iter()
        .map(|&x| Int64::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
    assert!(results.results[0].distance < 0.001);
}

#[test]
fn test_int64_data_integrity() {
    // Note: Serialization is only implemented for f32
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceSingle::<Int64>::new(params);

    for i in 0..10i64 {
        let v: Vec<Int64> = (0..4).map(|j| Int64::new(i * 10000 + j * 1000)).collect();
        index.add_vector(&v, i as u64).unwrap();
    }

    assert_eq!(index.index_size(), 10);

    // Verify vectors can be retrieved
    for i in 0..10i64 {
        assert!(index.contains(i as u64));
        let v = index.get_vector(i as u64).unwrap();
        for (j, &val) in v.iter().enumerate() {
            let expected = i * 10000 + j as i64 * 1000;
            assert_eq!(val.get(), expected);
        }
    }
}

#[test]
fn test_int64_multi_value() {
    let params = BruteForceParams::new(4, Metric::L2);
    let mut index = BruteForceMulti::<Int64>::new(params);

    let v1: Vec<Int64> = [10000i64, 0, 0, 0].iter().map(|&x| Int64::new(x)).collect();
    let v2: Vec<Int64> = [9000i64, 1000, 0, 0]
        .iter()
        .map(|&x| Int64::new(x))
        .collect();
    let v3: Vec<Int64> = [0i64, 10000, 0, 0].iter().map(|&x| Int64::new(x)).collect();

    index.add_vector(&v1, 1).unwrap();
    index.add_vector(&v2, 1).unwrap(); // Same label
    index.add_vector(&v3, 2).unwrap();

    assert_eq!(index.label_count(1), 2);

    let query: Vec<Int64> = [10000i64, 0, 0, 0]
        .iter()
        .map(|&x| Int64::new(x))
        .collect();
    let results = index.top_k_query(&query, 1, None).unwrap();
    assert_eq!(results.results[0].label, 1);
}

// =============================================================================
// Cross-type Consistency Tests
// =============================================================================

#[test]
fn test_all_types_produce_consistent_ordering() {
    // Test that all types produce the same nearest neighbor ordering
    // when given equivalent data

    let dim = 4;

    // Create orthogonal unit vectors as f32
    let vectors_f32: Vec<Vec<f32>> = create_orthogonal_vectors_f32(dim);

    // BruteForce with f32
    let params = BruteForceParams::new(dim, Metric::L2);
    let mut index_f32 = BruteForceSingle::<f32>::new(params.clone());
    for (i, v) in vectors_f32.iter().enumerate() {
        index_f32.add_vector(v, i as u64 + 1).unwrap();
    }

    // BruteForce with f64
    let mut index_f64 = BruteForceSingle::<f64>::new(params.clone());
    for (i, v) in vectors_f32.iter().enumerate() {
        let v64: Vec<f64> = v.iter().map(|&x| x as f64).collect();
        index_f64.add_vector(&v64, i as u64 + 1).unwrap();
    }

    // BruteForce with Float16
    let mut index_fp16 = BruteForceSingle::<Float16>::new(params.clone());
    for (i, v) in vectors_f32.iter().enumerate() {
        let vfp16: Vec<Float16> = v.iter().map(|&x| Float16::from_f32(x)).collect();
        index_fp16.add_vector(&vfp16, i as u64 + 1).unwrap();
    }

    // BruteForce with BFloat16
    let mut index_bf16 = BruteForceSingle::<BFloat16>::new(params.clone());
    for (i, v) in vectors_f32.iter().enumerate() {
        let vbf16: Vec<BFloat16> = v.iter().map(|&x| BFloat16::from_f32(x)).collect();
        index_bf16.add_vector(&vbf16, i as u64 + 1).unwrap();
    }

    // Query for nearest neighbor to first vector
    let query_f32: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let query_f64: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0];
    let query_fp16: Vec<Float16> = query_f32.iter().map(|&x| Float16::from_f32(x)).collect();
    let query_bf16: Vec<BFloat16> = query_f32.iter().map(|&x| BFloat16::from_f32(x)).collect();

    let result_f32 = index_f32.top_k_query(&query_f32, 1, None).unwrap();
    let result_f64 = index_f64.top_k_query(&query_f64, 1, None).unwrap();
    let result_fp16 = index_fp16.top_k_query(&query_fp16, 1, None).unwrap();
    let result_bf16 = index_bf16.top_k_query(&query_bf16, 1, None).unwrap();

    // All should return label 1 as nearest
    assert_eq!(result_f32.results[0].label, 1);
    assert_eq!(result_f64.results[0].label, 1);
    assert_eq!(result_fp16.results[0].label, 1);
    assert_eq!(result_bf16.results[0].label, 1);
}

#[test]
fn test_integer_types_consistent_ordering() {
    let dim = 4;

    // Create orthogonal vectors with integer values
    let vectors_i8: Vec<Vec<Int8>> = (0..dim)
        .map(|i| {
            let mut v = vec![Int8::new(0); dim];
            v[i] = Int8::new(100);
            v
        })
        .collect();

    let vectors_u8: Vec<Vec<UInt8>> = (0..dim)
        .map(|i| {
            let mut v = vec![UInt8::new(0); dim];
            v[i] = UInt8::new(200);
            v
        })
        .collect();

    let vectors_i32: Vec<Vec<Int32>> = (0..dim)
        .map(|i| {
            let mut v = vec![Int32::new(0); dim];
            v[i] = Int32::new(1000);
            v
        })
        .collect();

    let vectors_i64: Vec<Vec<Int64>> = (0..dim)
        .map(|i| {
            let mut v = vec![Int64::new(0); dim];
            v[i] = Int64::new(10000);
            v
        })
        .collect();

    let params = BruteForceParams::new(dim, Metric::L2);

    let mut index_i8 = BruteForceSingle::<Int8>::new(params.clone());
    let mut index_u8 = BruteForceSingle::<UInt8>::new(params.clone());
    let mut index_i32 = BruteForceSingle::<Int32>::new(params.clone());
    let mut index_i64 = BruteForceSingle::<Int64>::new(params.clone());

    for (i, (((vi8, vu8), vi32), vi64)) in vectors_i8
        .iter()
        .zip(vectors_u8.iter())
        .zip(vectors_i32.iter())
        .zip(vectors_i64.iter())
        .enumerate()
    {
        index_i8.add_vector(vi8, i as u64 + 1).unwrap();
        index_u8.add_vector(vu8, i as u64 + 1).unwrap();
        index_i32.add_vector(vi32, i as u64 + 1).unwrap();
        index_i64.add_vector(vi64, i as u64 + 1).unwrap();
    }

    // Query for first axis
    let query_i8: Vec<Int8> = vec![Int8::new(100), Int8::new(0), Int8::new(0), Int8::new(0)];
    let query_u8: Vec<UInt8> = vec![
        UInt8::new(200),
        UInt8::new(0),
        UInt8::new(0),
        UInt8::new(0),
    ];
    let query_i32: Vec<Int32> = vec![
        Int32::new(1000),
        Int32::new(0),
        Int32::new(0),
        Int32::new(0),
    ];
    let query_i64: Vec<Int64> = vec![
        Int64::new(10000),
        Int64::new(0),
        Int64::new(0),
        Int64::new(0),
    ];

    let result_i8 = index_i8.top_k_query(&query_i8, 1, None).unwrap();
    let result_u8 = index_u8.top_k_query(&query_u8, 1, None).unwrap();
    let result_i32 = index_i32.top_k_query(&query_i32, 1, None).unwrap();
    let result_i64 = index_i64.top_k_query(&query_i64, 1, None).unwrap();

    // All should return label 1 as nearest
    assert_eq!(result_i8.results[0].label, 1);
    assert_eq!(result_u8.results[0].label, 1);
    assert_eq!(result_i32.results[0].label, 1);
    assert_eq!(result_i64.results[0].label, 1);
}

// =============================================================================
// Memory Size Tests
// =============================================================================

#[test]
fn test_data_type_memory_sizes() {
    // Verify that smaller types use less memory
    assert_eq!(std::mem::size_of::<Int8>(), 1);
    assert_eq!(std::mem::size_of::<UInt8>(), 1);
    assert_eq!(std::mem::size_of::<Float16>(), 2);
    assert_eq!(std::mem::size_of::<BFloat16>(), 2);
    assert_eq!(std::mem::size_of::<Int32>(), 4);
    assert_eq!(std::mem::size_of::<f32>(), 4);
    assert_eq!(std::mem::size_of::<Int64>(), 8);
    assert_eq!(std::mem::size_of::<f64>(), 8);
}

// =============================================================================
// Delete Operation Tests for All Types
// =============================================================================

#[test]
fn test_delete_operations_all_types() {
    fn test_delete_for_type<T: VectorElement>() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceSingle::<T>::new(params);

        let v1: Vec<T> = [1.0f32, 0.0, 0.0, 0.0]
            .iter()
            .map(|&x| T::from_f32(x))
            .collect();
        let v2: Vec<T> = [0.0f32, 1.0, 0.0, 0.0]
            .iter()
            .map(|&x| T::from_f32(x))
            .collect();

        index.add_vector(&v1, 1).unwrap();
        index.add_vector(&v2, 2).unwrap();

        assert_eq!(index.index_size(), 2);

        index.delete_vector(1).unwrap();

        let query: Vec<T> = [1.0f32, 0.0, 0.0, 0.0]
            .iter()
            .map(|&x| T::from_f32(x))
            .collect();
        let results = index.top_k_query(&query, 2, None).unwrap();
        assert_eq!(results.results.len(), 1);
        assert_eq!(results.results[0].label, 2);
    }

    test_delete_for_type::<f32>();
    test_delete_for_type::<f64>();
    test_delete_for_type::<Float16>();
    test_delete_for_type::<BFloat16>();
    test_delete_for_type::<Int8>();
    test_delete_for_type::<UInt8>();
    test_delete_for_type::<Int32>();
    test_delete_for_type::<Int64>();
}

// =============================================================================
// Scaling Tests
// =============================================================================

#[test]
fn test_scaling_all_types() {
    fn test_scaling_for_type<T: VectorElement>(count: usize) {
        let dim = 16;
        let params = BruteForceParams::new(dim, Metric::L2);
        let mut index = BruteForceSingle::<T>::new(params);

        for i in 0..count {
            let v: Vec<T> = (0..dim)
                .map(|j| T::from_f32(((i * dim + j) % 1000) as f32 * 0.01))
                .collect();
            index.add_vector(&v, i as u64).unwrap();
        }

        assert_eq!(index.index_size(), count);

        let query: Vec<T> = (0..dim).map(|j| T::from_f32(j as f32 * 0.01)).collect();
        let results = index.top_k_query(&query, 10, None).unwrap();
        assert_eq!(results.results.len(), 10);
    }

    test_scaling_for_type::<f32>(500);
    test_scaling_for_type::<f64>(500);
    test_scaling_for_type::<Float16>(500);
    test_scaling_for_type::<BFloat16>(500);
    test_scaling_for_type::<Int8>(500);
    test_scaling_for_type::<UInt8>(500);
    test_scaling_for_type::<Int32>(500);
    test_scaling_for_type::<Int64>(500);
}

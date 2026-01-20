//! SIMD Cross-Consistency Tests
//!
//! These tests verify that all SIMD implementations produce results consistent
//! with the scalar implementation. This ensures correctness across different
//! hardware and SIMD instruction sets.
//!
//! Tests cover:
//! - All distance metrics (L2, inner product, cosine)
//! - Various vector dimensions (aligned and non-aligned)
//! - Edge cases (identical vectors, orthogonal vectors, zero vectors)
//! - Random data for statistical validation

#![cfg(test)]
#![allow(unused_imports)]

use crate::distance::l2::{l2_squared_scalar_f32, l2_squared_scalar_f64};
use crate::distance::ip::inner_product_scalar_f32;
use crate::distance::cosine::cosine_distance_scalar_f32;

/// Tolerance for f32 comparisons (SIMD operations may have different rounding)
const F32_TOLERANCE: f32 = 1e-4;
/// Tolerance for f64 comparisons
#[cfg(target_arch = "x86_64")]
const F64_TOLERANCE: f64 = 1e-10;
/// Relative tolerance for larger values
const RELATIVE_TOLERANCE: f64 = 1e-5;

/// Check if two f32 values are approximately equal
fn approx_eq_f32(a: f32, b: f32, abs_tol: f32, rel_tol: f64) -> bool {
    let diff = (a - b).abs();
    let max_val = a.abs().max(b.abs());
    diff <= abs_tol || diff <= (max_val as f64 * rel_tol) as f32
}

/// Check if two f64 values are approximately equal
#[cfg(target_arch = "x86_64")]
fn approx_eq_f64(a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> bool {
    let diff = (a - b).abs();
    let max_val = a.abs().max(b.abs());
    diff <= abs_tol || diff <= max_val * rel_tol
}

/// Generate test vectors of given dimension with a pattern
fn generate_test_vectors_f32(dim: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(dim);
    let mut b = Vec::with_capacity(dim);

    for i in 0..dim {
        // Use a simple deterministic pattern based on index and seed
        let val_a = ((i as f32 + seed as f32) * 0.1).sin() * 10.0;
        let val_b = ((i as f32 + seed as f32 + 1.0) * 0.15).cos() * 10.0;
        a.push(val_a);
        b.push(val_b);
    }

    (a, b)
}

/// Generate test vectors of given dimension with a pattern (f64)
#[cfg(target_arch = "x86_64")]
fn generate_test_vectors_f64(dim: usize, seed: u32) -> (Vec<f64>, Vec<f64>) {
    let mut a = Vec::with_capacity(dim);
    let mut b = Vec::with_capacity(dim);

    for i in 0..dim {
        let val_a = ((i as f64 + seed as f64) * 0.1).sin() * 10.0;
        let val_b = ((i as f64 + seed as f64 + 1.0) * 0.15).cos() * 10.0;
        a.push(val_a);
        b.push(val_b);
    }

    (a, b)
}

/// Dimensions to test - includes aligned and non-aligned sizes
const TEST_DIMENSIONS: &[usize] = &[
    1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33,
    63, 64, 65, 127, 128, 129, 255, 256, 257, 512, 768, 1024,
];

// =============================================================================
// x86_64 SIMD Cross-Consistency Tests
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_64_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // SSE Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sse_l2_cross_consistency() {
        if !is_x86_feature_detected!("sse") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = l2_squared_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::sse::l2_squared_f32_sse(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "SSE L2 mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_sse_inner_product_cross_consistency() {
        if !is_x86_feature_detected!("sse") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = inner_product_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::sse::inner_product_f32_sse(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "SSE inner product mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_sse_cosine_cross_consistency() {
        if !is_x86_feature_detected!("sse") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = cosine_distance_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::sse::cosine_f32_sse(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "SSE cosine mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // SSE4.1 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sse4_l2_cross_consistency() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = l2_squared_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::sse4::l2_squared_f32_sse4(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "SSE4 L2 mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_sse4_inner_product_cross_consistency() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = inner_product_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::sse4::inner_product_f32_sse4(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "SSE4 inner product mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_sse4_cosine_cross_consistency() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = cosine_distance_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::sse4::cosine_distance_f32_sse4(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "SSE4 cosine mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // AVX Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_avx_l2_cross_consistency() {
        if !is_x86_feature_detected!("avx") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = l2_squared_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx::l2_squared_f32_avx(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX L2 mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_avx_inner_product_cross_consistency() {
        if !is_x86_feature_detected!("avx") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = inner_product_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx::inner_product_f32_avx(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX inner product mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_avx_cosine_cross_consistency() {
        if !is_x86_feature_detected!("avx") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = cosine_distance_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx::cosine_distance_f32_avx(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX cosine mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // AVX2 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_avx2_l2_f32_cross_consistency() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = l2_squared_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx2::l2_squared_f32_avx2(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX2 L2 f32 mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_avx2_l2_f64_cross_consistency() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f64(dim, seed);

                let scalar_result = l2_squared_scalar_f64(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx2::l2_squared_f64_avx2(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f64(scalar_result, simd_result, F64_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX2 L2 f64 mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_avx2_inner_product_f32_cross_consistency() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = inner_product_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx2::inner_product_f32_avx2(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX2 inner product f32 mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_avx2_cosine_f32_cross_consistency() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = cosine_distance_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx2::cosine_distance_f32_avx2(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX2 cosine f32 mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // AVX-512 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_avx512_l2_cross_consistency() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = l2_squared_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx512::l2_squared_f32_avx512(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX-512 L2 mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_avx512_inner_product_cross_consistency() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = inner_product_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx512::inner_product_f32_avx512(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX-512 inner product mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_avx512_cosine_cross_consistency() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = cosine_distance_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx512::cosine_distance_f32_avx512(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX-512 cosine mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // AVX-512 BW Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_avx512bw_l2_cross_consistency() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            return;
        }

        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = l2_squared_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::avx512bw::l2_squared_f32_avx512bw(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "AVX-512 BW L2 mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simd_identical_vectors() {
        // All SIMD implementations should return 0 for identical vectors (L2)
        let dims = [4, 8, 16, 32, 64, 128, 256];

        for dim in dims {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();

            // Scalar
            let scalar_l2 = l2_squared_scalar_f32(&a, &a, dim);
            assert!(scalar_l2.abs() < 1e-10, "Scalar L2 of identical vectors should be 0");

            // SSE
            if is_x86_feature_detected!("sse") {
                let sse_l2 = unsafe {
                    crate::distance::simd::sse::l2_squared_f32_sse(a.as_ptr(), a.as_ptr(), dim)
                };
                assert!(sse_l2.abs() < 1e-6, "SSE L2 of identical vectors should be ~0, got {}", sse_l2);
            }

            // AVX2
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let avx2_l2 = unsafe {
                    crate::distance::simd::avx2::l2_squared_f32_avx2(a.as_ptr(), a.as_ptr(), dim)
                };
                assert!(avx2_l2.abs() < 1e-6, "AVX2 L2 of identical vectors should be ~0, got {}", avx2_l2);
            }

            // AVX-512
            if is_x86_feature_detected!("avx512f") {
                let avx512_l2 = unsafe {
                    crate::distance::simd::avx512::l2_squared_f32_avx512(a.as_ptr(), a.as_ptr(), dim)
                };
                assert!(avx512_l2.abs() < 1e-6, "AVX-512 L2 of identical vectors should be ~0, got {}", avx512_l2);
            }
        }
    }

    #[test]
    fn test_simd_orthogonal_vectors_cosine() {
        // Orthogonal vectors should have cosine distance of 1.0
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let dim = 8;

        let scalar_cos = cosine_distance_scalar_f32(&a, &b, dim);
        assert!((scalar_cos - 1.0).abs() < 0.01, "Scalar cosine of orthogonal vectors should be ~1.0");

        if is_x86_feature_detected!("sse") {
            let sse_cos = unsafe {
                crate::distance::simd::sse::cosine_f32_sse(a.as_ptr(), b.as_ptr(), dim)
            };
            assert!((sse_cos - 1.0).abs() < 0.01, "SSE cosine of orthogonal vectors should be ~1.0, got {}", sse_cos);
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx2_cos = unsafe {
                crate::distance::simd::avx2::cosine_distance_f32_avx2(a.as_ptr(), b.as_ptr(), dim)
            };
            assert!((avx2_cos - 1.0).abs() < 0.01, "AVX2 cosine of orthogonal vectors should be ~1.0, got {}", avx2_cos);
        }
    }

    #[test]
    fn test_simd_opposite_vectors_cosine() {
        // Opposite vectors should have cosine distance of 2.0
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let dim = 8;

        let scalar_cos = cosine_distance_scalar_f32(&a, &b, dim);
        assert!((scalar_cos - 2.0).abs() < 0.01, "Scalar cosine of opposite vectors should be ~2.0");

        if is_x86_feature_detected!("sse") {
            let sse_cos = unsafe {
                crate::distance::simd::sse::cosine_f32_sse(a.as_ptr(), b.as_ptr(), dim)
            };
            assert!((sse_cos - 2.0).abs() < 0.01, "SSE cosine of opposite vectors should be ~2.0, got {}", sse_cos);
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx2_cos = unsafe {
                crate::distance::simd::avx2::cosine_distance_f32_avx2(a.as_ptr(), b.as_ptr(), dim)
            };
            assert!((avx2_cos - 2.0).abs() < 0.01, "AVX2 cosine of opposite vectors should be ~2.0, got {}", avx2_cos);
        }
    }

    #[test]
    fn test_simd_all_implementations_agree() {
        // Test that all available SIMD implementations produce consistent results
        let dim = 128;
        let (a, b) = generate_test_vectors_f32(dim, 42);

        let scalar_l2 = l2_squared_scalar_f32(&a, &b, dim);
        let scalar_ip = inner_product_scalar_f32(&a, &b, dim);
        let scalar_cos = cosine_distance_scalar_f32(&a, &b, dim);

        let mut results_l2: Vec<(&str, f32)> = vec![("scalar", scalar_l2)];
        let mut results_ip: Vec<(&str, f32)> = vec![("scalar", scalar_ip)];
        let mut results_cos: Vec<(&str, f32)> = vec![("scalar", scalar_cos)];

        if is_x86_feature_detected!("sse") {
            unsafe {
                results_l2.push(("sse", crate::distance::simd::sse::l2_squared_f32_sse(a.as_ptr(), b.as_ptr(), dim)));
                results_ip.push(("sse", crate::distance::simd::sse::inner_product_f32_sse(a.as_ptr(), b.as_ptr(), dim)));
                results_cos.push(("sse", crate::distance::simd::sse::cosine_f32_sse(a.as_ptr(), b.as_ptr(), dim)));
            }
        }

        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                results_l2.push(("sse4", crate::distance::simd::sse4::l2_squared_f32_sse4(a.as_ptr(), b.as_ptr(), dim)));
                results_ip.push(("sse4", crate::distance::simd::sse4::inner_product_f32_sse4(a.as_ptr(), b.as_ptr(), dim)));
                results_cos.push(("sse4", crate::distance::simd::sse4::cosine_distance_f32_sse4(a.as_ptr(), b.as_ptr(), dim)));
            }
        }

        if is_x86_feature_detected!("avx") {
            unsafe {
                results_l2.push(("avx", crate::distance::simd::avx::l2_squared_f32_avx(a.as_ptr(), b.as_ptr(), dim)));
                results_ip.push(("avx", crate::distance::simd::avx::inner_product_f32_avx(a.as_ptr(), b.as_ptr(), dim)));
                results_cos.push(("avx", crate::distance::simd::avx::cosine_distance_f32_avx(a.as_ptr(), b.as_ptr(), dim)));
            }
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                results_l2.push(("avx2", crate::distance::simd::avx2::l2_squared_f32_avx2(a.as_ptr(), b.as_ptr(), dim)));
                results_ip.push(("avx2", crate::distance::simd::avx2::inner_product_f32_avx2(a.as_ptr(), b.as_ptr(), dim)));
                results_cos.push(("avx2", crate::distance::simd::avx2::cosine_distance_f32_avx2(a.as_ptr(), b.as_ptr(), dim)));
            }
        }

        if is_x86_feature_detected!("avx512f") {
            unsafe {
                results_l2.push(("avx512", crate::distance::simd::avx512::l2_squared_f32_avx512(a.as_ptr(), b.as_ptr(), dim)));
                results_ip.push(("avx512", crate::distance::simd::avx512::inner_product_f32_avx512(a.as_ptr(), b.as_ptr(), dim)));
                results_cos.push(("avx512", crate::distance::simd::avx512::cosine_distance_f32_avx512(a.as_ptr(), b.as_ptr(), dim)));
            }
        }

        // Verify all L2 results are consistent
        for (name, result) in &results_l2[1..] {
            assert!(
                approx_eq_f32(scalar_l2, *result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                "L2 mismatch between scalar ({}) and {} ({})",
                scalar_l2, name, result
            );
        }

        // Verify all inner product results are consistent
        for (name, result) in &results_ip[1..] {
            assert!(
                approx_eq_f32(scalar_ip, *result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                "Inner product mismatch between scalar ({}) and {} ({})",
                scalar_ip, name, result
            );
        }

        // Verify all cosine results are consistent
        for (name, result) in &results_cos[1..] {
            assert!(
                approx_eq_f32(scalar_cos, *result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                "Cosine mismatch between scalar ({}) and {} ({})",
                scalar_cos, name, result
            );
        }
    }

    // -------------------------------------------------------------------------
    // Large Vector Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simd_large_vectors() {
        // Test with large vectors typical of embeddings
        let large_dims = [384, 512, 768, 1024, 1536];

        for dim in large_dims {
            let (a, b) = generate_test_vectors_f32(dim, 123);

            let scalar_l2 = l2_squared_scalar_f32(&a, &b, dim);

            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let avx2_l2 = unsafe {
                    crate::distance::simd::avx2::l2_squared_f32_avx2(a.as_ptr(), b.as_ptr(), dim)
                };
                assert!(
                    approx_eq_f32(scalar_l2, avx2_l2, F32_TOLERANCE * 10.0, RELATIVE_TOLERANCE),
                    "AVX2 L2 mismatch for large dim={}: scalar={}, avx2={}",
                    dim, scalar_l2, avx2_l2
                );
            }

            if is_x86_feature_detected!("avx512f") {
                let avx512_l2 = unsafe {
                    crate::distance::simd::avx512::l2_squared_f32_avx512(a.as_ptr(), b.as_ptr(), dim)
                };
                assert!(
                    approx_eq_f32(scalar_l2, avx512_l2, F32_TOLERANCE * 10.0, RELATIVE_TOLERANCE),
                    "AVX-512 L2 mismatch for large dim={}: scalar={}, avx512={}",
                    dim, scalar_l2, avx512_l2
                );
            }
        }
    }

    #[test]
    fn test_simd_small_values() {
        // Test with very small values that might cause precision issues
        let dim = 64;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 1e-6).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) * 1e-6).collect();

        let scalar_l2 = l2_squared_scalar_f32(&a, &b, dim);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx2_l2 = unsafe {
                crate::distance::simd::avx2::l2_squared_f32_avx2(a.as_ptr(), b.as_ptr(), dim)
            };
            // Use relative tolerance for small values
            let rel_diff = if scalar_l2.abs() > 1e-20 {
                (scalar_l2 - avx2_l2).abs() / scalar_l2.abs()
            } else {
                (scalar_l2 - avx2_l2).abs()
            };
            assert!(
                rel_diff < 0.01,
                "AVX2 L2 mismatch for small values: scalar={}, avx2={}, rel_diff={}",
                scalar_l2, avx2_l2, rel_diff
            );
        }
    }

    #[test]
    fn test_simd_large_values() {
        // Test with large values
        let dim = 64;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 1e4).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) * 1e4).collect();

        let scalar_l2 = l2_squared_scalar_f32(&a, &b, dim);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx2_l2 = unsafe {
                crate::distance::simd::avx2::l2_squared_f32_avx2(a.as_ptr(), b.as_ptr(), dim)
            };
            assert!(
                approx_eq_f32(scalar_l2, avx2_l2, 1.0, RELATIVE_TOLERANCE),
                "AVX2 L2 mismatch for large values: scalar={}, avx2={}",
                scalar_l2, avx2_l2
            );
        }
    }

    #[test]
    fn test_simd_negative_values() {
        // Test with negative values
        let dim = 64;
        let a: Vec<f32> = (0..dim).map(|i| -((i as f32) * 0.1)).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.15)).collect();

        let scalar_l2 = l2_squared_scalar_f32(&a, &b, dim);
        let scalar_ip = inner_product_scalar_f32(&a, &b, dim);

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx2_l2 = unsafe {
                crate::distance::simd::avx2::l2_squared_f32_avx2(a.as_ptr(), b.as_ptr(), dim)
            };
            let avx2_ip = unsafe {
                crate::distance::simd::avx2::inner_product_f32_avx2(a.as_ptr(), b.as_ptr(), dim)
            };

            assert!(
                approx_eq_f32(scalar_l2, avx2_l2, F32_TOLERANCE, RELATIVE_TOLERANCE),
                "AVX2 L2 mismatch for negative values: scalar={}, avx2={}",
                scalar_l2, avx2_l2
            );

            assert!(
                approx_eq_f32(scalar_ip, avx2_ip, F32_TOLERANCE, RELATIVE_TOLERANCE),
                "AVX2 IP mismatch for negative values: scalar={}, avx2={}",
                scalar_ip, avx2_ip
            );
        }
    }
}

// =============================================================================
// aarch64 NEON Cross-Consistency Tests
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod aarch64_tests {
    use super::*;

    #[test]
    fn test_neon_l2_cross_consistency() {
        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = l2_squared_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::neon::l2_squared_f32_neon(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "NEON L2 mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_neon_inner_product_cross_consistency() {
        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = inner_product_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::neon::inner_product_f32_neon(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "NEON inner product mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_neon_cosine_cross_consistency() {
        for &dim in TEST_DIMENSIONS {
            for seed in 0..3 {
                let (a, b) = generate_test_vectors_f32(dim, seed);

                let scalar_result = cosine_distance_scalar_f32(&a, &b, dim);
                let simd_result = unsafe {
                    crate::distance::simd::neon::cosine_distance_f32_neon(
                        a.as_ptr(), b.as_ptr(), dim
                    )
                };

                assert!(
                    approx_eq_f32(scalar_result, simd_result, F32_TOLERANCE, RELATIVE_TOLERANCE),
                    "NEON cosine mismatch at dim={}, seed={}: scalar={}, simd={}",
                    dim, seed, scalar_result, simd_result
                );
            }
        }
    }

    #[test]
    fn test_neon_identical_vectors() {
        let dims = [4, 8, 16, 32, 64, 128];

        for dim in dims {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();

            let neon_l2 = unsafe {
                crate::distance::simd::neon::l2_squared_f32_neon(a.as_ptr(), a.as_ptr(), dim)
            };
            assert!(neon_l2.abs() < 1e-6, "NEON L2 of identical vectors should be ~0, got {}", neon_l2);
        }
    }

    #[test]
    fn test_neon_large_vectors() {
        let large_dims = [384, 512, 768, 1024];

        for dim in large_dims {
            let (a, b) = generate_test_vectors_f32(dim, 123);

            let scalar_l2 = l2_squared_scalar_f32(&a, &b, dim);
            let neon_l2 = unsafe {
                crate::distance::simd::neon::l2_squared_f32_neon(a.as_ptr(), b.as_ptr(), dim)
            };

            assert!(
                approx_eq_f32(scalar_l2, neon_l2, F32_TOLERANCE * 10.0, RELATIVE_TOLERANCE),
                "NEON L2 mismatch for large dim={}: scalar={}, neon={}",
                dim, scalar_l2, neon_l2
            );
        }
    }
}

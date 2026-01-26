//! AVX (AVX1) SIMD implementations for distance functions.
//!
//! AVX provides 256-bit vectors, processing 8 f32 values at a time.
//! Unlike AVX2, AVX1 does not include FMA instructions, so we use
//! separate multiply and add operations.
//!
//! Available on Intel Sandy Bridge+ and AMD Bulldozer+ processors.

#![cfg(target_arch = "x86_64")]

use crate::types::{DistanceType, VectorElement};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX1 L2 squared distance for f32 vectors.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - AVX must be available (checked at runtime by caller)
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn l2_squared_f32_avx(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.add(offset));
        let vb = _mm256_loadu_ps(b.add(offset));
        let diff = _mm256_sub_ps(va, vb);
        // AVX1 doesn't have FMA, so use mul + add
        let sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }

    // Horizontal sum
    let mut result = hsum256_ps_avx(sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let diff = *a.add(base + i) - *b.add(base + i);
        result += diff * diff;
    }

    result
}

/// AVX1 inner product for f32 vectors.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - AVX must be available (checked at runtime by caller)
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn inner_product_f32_avx(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.add(offset));
        let vb = _mm256_loadu_ps(b.add(offset));
        // AVX1 doesn't have FMA, so use mul + add
        let prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
    }

    // Horizontal sum
    let mut result = hsum256_ps_avx(sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        result += *a.add(base + i) * *b.add(base + i);
    }

    result
}

/// AVX1 cosine distance for f32 vectors.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - AVX must be available (checked at runtime by caller)
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn cosine_distance_f32_avx(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();

    let chunks = dim / 8;
    let remainder = dim % 8;

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.add(offset));
        let vb = _mm256_loadu_ps(b.add(offset));

        // AVX1 doesn't have FMA, so use mul + add
        let dot_prod = _mm256_mul_ps(va, vb);
        dot_sum = _mm256_add_ps(dot_sum, dot_prod);

        let norm_a_prod = _mm256_mul_ps(va, va);
        norm_a_sum = _mm256_add_ps(norm_a_sum, norm_a_prod);

        let norm_b_prod = _mm256_mul_ps(vb, vb);
        norm_b_sum = _mm256_add_ps(norm_b_sum, norm_b_prod);
    }

    // Horizontal sums
    let mut dot = hsum256_ps_avx(dot_sum);
    let mut norm_a = hsum256_ps_avx(norm_a_sum);
    let mut norm_b = hsum256_ps_avx(norm_b_sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let va = *a.add(base + i);
        let vb = *b.add(base + i);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-30 {
        return 1.0;
    }

    let cosine_sim = (dot / denom).max(-1.0).min(1.0);
    1.0 - cosine_sim
}

/// Horizontal sum of 8 f32 values in a 256-bit register (AVX1 version).
#[target_feature(enable = "avx")]
#[inline]
unsafe fn hsum256_ps_avx(v: __m256) -> f32 {
    // Extract high 128 bits and add to low 128 bits
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);

    // Horizontal add within 128 bits
    let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
    let sums = _mm_add_ps(sum128, shuf); // [0+1,1+1,2+3,3+3]
    let shuf = _mm_movehl_ps(sums, sums); // [2+3,3+3,2+3,3+3]
    let sums = _mm_add_ss(sums, shuf); // [0+1+2+3,...]

    _mm_cvtss_f32(sums)
}

/// Safe wrapper for L2 squared distance.
#[inline]
pub fn l2_squared_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("avx") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { l2_squared_f32_avx(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::l2::l2_squared_scalar(a, b, dim)
    }
}

/// Safe wrapper for inner product.
#[inline]
pub fn inner_product_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("avx") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { inner_product_f32_avx(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::ip::inner_product_scalar(a, b, dim)
    }
}

/// Safe wrapper for cosine distance.
#[inline]
pub fn cosine_distance_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("avx") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { cosine_distance_f32_avx(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::cosine::cosine_distance_scalar(a, b, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_avx_available() -> bool {
        is_x86_feature_detected!("avx")
    }

    #[test]
    fn test_l2_squared_avx() {
        if !is_avx_available() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = unsafe { l2_squared_f32_avx(a.as_ptr(), b.as_ptr(), a.len()) };
        // Each difference is 1.0, so squared sum = 8.0
        assert!((result - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_squared_avx_remainder() {
        if !is_avx_available() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];

        let result = unsafe { l2_squared_f32_avx(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_avx() {
        if !is_avx_available() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = unsafe { inner_product_f32_avx(a.as_ptr(), b.as_ptr(), a.len()) };
        // 1+2+3+4+5+6+7+8 = 36
        assert!((result - 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_avx() {
        if !is_avx_available() {
            return;
        }

        // Test with identical normalized vectors (cosine distance = 0)
        let a = vec![0.5f32, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.5f32, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0];

        let result = unsafe { cosine_distance_f32_avx(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!(result.abs() < 1e-6);

        // Test with orthogonal vectors (cosine distance = 1)
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let result = unsafe { cosine_distance_f32_avx(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_safe_wrapper() {
        if !is_avx_available() {
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();

        let avx_result = l2_squared_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::l2::l2_squared_scalar(&a, &b, 128);

        assert!((avx_result - scalar_result).abs() < 0.1);
    }

    #[test]
    fn test_inner_product_safe_wrapper() {
        if !is_avx_available() {
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32 / 100.0).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 / 100.0).collect();

        let avx_result = inner_product_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::ip::inner_product_scalar(&a, &b, 128);

        assert!((avx_result - scalar_result).abs() < 0.01);
    }
}

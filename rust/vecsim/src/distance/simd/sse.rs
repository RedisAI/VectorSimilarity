//! SSE (Streaming SIMD Extensions) optimized distance functions.
//!
//! SSE provides 128-bit vectors, processing 4 f32 values at a time.
//! This is available on virtually all x86-64 processors.

#![cfg(target_arch = "x86_64")]

use crate::types::{DistanceType, VectorElement};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute squared L2 distance using SSE.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - SSE must be available (checked at runtime by caller)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
pub unsafe fn l2_squared_f32_sse(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm_setzero_ps();
    let chunks = dim / 4;
    let remainder = dim % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a.add(offset));
        let vb = _mm_loadu_ps(b.add(offset));
        let diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    // Horizontal sum of the 4 floats in sum
    // sum = [a, b, c, d]
    // After first hadd: [a+b, c+d, a+b, c+d]
    // After second hadd: [a+b+c+d, ...]
    let shuf = _mm_shuffle_ps(sum, sum, 0b10_11_00_01); // [b, a, d, c]
    let sums = _mm_add_ps(sum, shuf); // [a+b, a+b, c+d, c+d]
    let shuf2 = _mm_movehl_ps(sums, sums); // [c+d, c+d, ?, ?]
    let result = _mm_add_ss(sums, shuf2); // [a+b+c+d, ?, ?, ?]
    let mut total = _mm_cvtss_f32(result);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let diff = *a.add(base + i) - *b.add(base + i);
        total += diff * diff;
    }

    total
}

/// Compute inner product using SSE.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - SSE must be available (checked at runtime by caller)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
pub unsafe fn inner_product_f32_sse(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm_setzero_ps();
    let chunks = dim / 4;
    let remainder = dim % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a.add(offset));
        let vb = _mm_loadu_ps(b.add(offset));
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }

    // Horizontal sum
    let shuf = _mm_shuffle_ps(sum, sum, 0b10_11_00_01);
    let sums = _mm_add_ps(sum, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        total += *a.add(base + i) * *b.add(base + i);
    }

    total
}

/// Compute cosine distance using SSE.
/// Returns 1 - cosine_similarity.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - SSE must be available (checked at runtime by caller)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
pub unsafe fn cosine_f32_sse(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut dot_sum = _mm_setzero_ps();
    let mut norm_a_sum = _mm_setzero_ps();
    let mut norm_b_sum = _mm_setzero_ps();

    let chunks = dim / 4;
    let remainder = dim % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a.add(offset));
        let vb = _mm_loadu_ps(b.add(offset));

        dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(va, vb));
        norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(va, va));
        norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(vb, vb));
    }

    // Horizontal sums
    let hsum = |v: __m128| -> f32 {
        let shuf = _mm_shuffle_ps(v, v, 0b10_11_00_01);
        let sums = _mm_add_ps(v, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    };

    let mut dot = hsum(dot_sum);
    let mut norm_a = hsum(norm_a_sum);
    let mut norm_b = hsum(norm_b_sum);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let va = *a.add(base + i);
        let vb = *b.add(base + i);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 {
        1.0 - (dot / denom)
    } else {
        0.0
    }
}

/// Safe wrapper for L2 squared distance.
#[inline]
pub fn l2_squared_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("sse") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { l2_squared_f32_sse(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::l2::l2_squared_scalar(a, b, dim)
    }
}

/// Safe wrapper for inner product.
#[inline]
pub fn inner_product_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("sse") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { inner_product_f32_sse(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::ip::inner_product_scalar(a, b, dim)
    }
}

/// Safe wrapper for cosine distance.
#[inline]
pub fn cosine_distance_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("sse") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { cosine_f32_sse(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::cosine::cosine_distance_scalar(a, b, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_sse_available() -> bool {
        is_x86_feature_detected!("sse")
    }

    #[test]
    fn test_l2_squared_sse() {
        if !is_sse_available() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = unsafe { l2_squared_f32_sse(a.as_ptr(), b.as_ptr(), a.len()) };
        // Each difference is 1.0, so squared sum = 8.0
        assert!((result - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_squared_sse_remainder() {
        if !is_sse_available() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0];

        let result = unsafe { l2_squared_f32_sse(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_sse() {
        if !is_sse_available() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0];

        let result = unsafe { inner_product_f32_sse(a.as_ptr(), b.as_ptr(), a.len()) };
        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        assert!((result - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sse() {
        if !is_sse_available() {
            return;
        }

        // Test with identical normalized vectors (cosine distance = 0)
        let a = vec![0.6f32, 0.8, 0.0, 0.0];
        let b = vec![0.6f32, 0.8, 0.0, 0.0];

        let result = unsafe { cosine_f32_sse(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!(result.abs() < 1e-6);

        // Test with orthogonal vectors (cosine distance = 1)
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];

        let result = unsafe { cosine_f32_sse(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_safe_wrapper() {
        if !is_sse_available() {
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();

        let sse_result = l2_squared_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::l2::l2_squared_scalar(&a, &b, 128);

        assert!((sse_result - scalar_result).abs() < 0.1);
    }

    #[test]
    fn test_inner_product_safe_wrapper() {
        if !is_sse_available() {
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32 / 100.0).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 / 100.0).collect();

        let sse_result = inner_product_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::ip::inner_product_scalar(&a, &b, 128);

        assert!((sse_result - scalar_result).abs() < 0.01);
    }
}

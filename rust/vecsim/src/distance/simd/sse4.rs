//! SSE4.1 optimized distance functions.
//!
//! SSE4.1 provides the `_mm_dp_ps` instruction for single-instruction dot products,
//! which is significantly faster than manual multiply-accumulate for inner products.
//!
//! Available on Intel Penryn+ (2008) and AMD Bulldozer+ (2011).

#![cfg(target_arch = "x86_64")]

use crate::types::VectorElement;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE4.1 L2 squared distance for f32 vectors.
///
/// Uses SSE4.1 `_mm_dp_ps` for efficient dot product computation.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - SSE4.1 must be available (checked at runtime by caller)
#[target_feature(enable = "sse4.1")]
#[inline]
pub unsafe fn l2_squared_f32_sse4(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut total = 0.0f32;
    let chunks = dim / 4;
    let remainder = dim % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a.add(offset));
        let vb = _mm_loadu_ps(b.add(offset));
        let diff = _mm_sub_ps(va, vb);
        // _mm_dp_ps with mask 0xF1: multiply all 4 pairs, sum to lowest element
        let sq_sum = _mm_dp_ps(diff, diff, 0xF1);
        total += _mm_cvtss_f32(sq_sum);
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let diff = *a.add(base + i) - *b.add(base + i);
        total += diff * diff;
    }

    total
}

/// SSE4.1 inner product for f32 vectors.
///
/// Uses SSE4.1 `_mm_dp_ps` for efficient dot product computation.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - SSE4.1 must be available (checked at runtime by caller)
#[target_feature(enable = "sse4.1")]
#[inline]
pub unsafe fn inner_product_f32_sse4(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut total = 0.0f32;
    let chunks = dim / 4;
    let remainder = dim % 4;

    // Process 4 elements at a time using _mm_dp_ps
    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a.add(offset));
        let vb = _mm_loadu_ps(b.add(offset));
        // _mm_dp_ps with mask 0xF1: multiply all 4 pairs, sum to lowest element
        let dp = _mm_dp_ps(va, vb, 0xF1);
        total += _mm_cvtss_f32(dp);
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        total += *a.add(base + i) * *b.add(base + i);
    }

    total
}

/// SSE4.1 cosine distance for f32 vectors.
///
/// Uses SSE4.1 `_mm_dp_ps` for efficient dot product computation.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - SSE4.1 must be available (checked at runtime by caller)
#[target_feature(enable = "sse4.1")]
#[inline]
pub unsafe fn cosine_distance_f32_sse4(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    let chunks = dim / 4;
    let remainder = dim % 4;

    // Process 4 elements at a time using _mm_dp_ps
    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a.add(offset));
        let vb = _mm_loadu_ps(b.add(offset));

        // Dot product a·b
        let dp = _mm_dp_ps(va, vb, 0xF1);
        dot += _mm_cvtss_f32(dp);

        // Norm squared ||a||²
        let na = _mm_dp_ps(va, va, 0xF1);
        norm_a += _mm_cvtss_f32(na);

        // Norm squared ||b||²
        let nb = _mm_dp_ps(vb, vb, 0xF1);
        norm_b += _mm_cvtss_f32(nb);
    }

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
    if denom < 1e-30 {
        return 1.0;
    }

    let cosine_sim = (dot / denom).max(-1.0).min(1.0);
    1.0 - cosine_sim
}

/// Safe wrapper for L2 squared distance.
#[inline]
pub fn l2_squared_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("sse4.1") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { l2_squared_f32_sse4(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::l2::l2_squared_scalar(a, b, dim)
    }
}

/// Safe wrapper for inner product.
#[inline]
pub fn inner_product_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("sse4.1") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { inner_product_f32_sse4(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::ip::inner_product_scalar(a, b, dim)
    }
}

/// Safe wrapper for cosine distance.
#[inline]
pub fn cosine_distance_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("sse4.1") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { cosine_distance_f32_sse4(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::cosine::cosine_distance_scalar(a, b, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_sse4_available() -> bool {
        is_x86_feature_detected!("sse4.1")
    }

    #[test]
    fn test_l2_squared_sse4() {
        if !is_sse4_available() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = unsafe { l2_squared_f32_sse4(a.as_ptr(), b.as_ptr(), a.len()) };
        // Each difference is 1.0, so squared sum = 8.0
        assert!((result - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_squared_sse4_remainder() {
        if !is_sse4_available() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0];

        let result = unsafe { l2_squared_f32_sse4(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((result - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_inner_product_sse4() {
        if !is_sse4_available() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = unsafe { inner_product_f32_sse4(a.as_ptr(), b.as_ptr(), a.len()) };
        // 1+2+3+4+5+6+7+8 = 36
        assert!((result - 36.0).abs() < 1e-5);
    }

    #[test]
    fn test_inner_product_sse4_remainder() {
        if !is_sse4_available() {
            return;
        }

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0f32, 2.0, 2.0, 2.0, 2.0];

        let result = unsafe { inner_product_f32_sse4(a.as_ptr(), b.as_ptr(), a.len()) };
        // 2+4+6+8+10 = 30
        assert!((result - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sse4_identical() {
        if !is_sse4_available() {
            return;
        }

        let a = vec![0.6f32, 0.8, 0.0, 0.0];
        let result = unsafe { cosine_distance_f32_sse4(a.as_ptr(), a.as_ptr(), a.len()) };
        assert!(result.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sse4_orthogonal() {
        if !is_sse4_available() {
            return;
        }

        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];

        let result = unsafe { cosine_distance_f32_sse4(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sse4_opposite() {
        if !is_sse4_available() {
            return;
        }

        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0, 0.0];

        let result = unsafe { cosine_distance_f32_sse4(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_safe_wrapper() {
        if !is_sse4_available() {
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();

        let sse4_result = l2_squared_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::l2::l2_squared_scalar(&a, &b, 128);

        assert!((sse4_result - scalar_result).abs() < 0.1);
    }

    #[test]
    fn test_inner_product_safe_wrapper() {
        if !is_sse4_available() {
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32 / 100.0).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 / 100.0).collect();

        let sse4_result = inner_product_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::ip::inner_product_scalar(&a, &b, 128);

        assert!((sse4_result - scalar_result).abs() < 0.01);
    }

    #[test]
    fn test_cosine_safe_wrapper() {
        if !is_sse4_available() {
            return;
        }

        let a: Vec<f32> = (1..129).map(|i| i as f32 / 100.0).collect();
        let b: Vec<f32> = (1..129).map(|i| (129 - i) as f32 / 100.0).collect();

        let sse4_result = cosine_distance_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::cosine::cosine_distance_scalar(&a, &b, 128);

        assert!((sse4_result - scalar_result).abs() < 0.001);
    }
}

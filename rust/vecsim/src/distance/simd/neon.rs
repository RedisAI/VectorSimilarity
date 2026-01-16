//! ARM NEON SIMD implementations for distance functions.
//!
//! These functions use 128-bit NEON instructions for ARM processors.
//! Available on all aarch64 (ARM64) platforms.

use crate::types::{DistanceType, VectorElement};

use std::arch::aarch64::*;

/// NEON L2 squared distance for f32 vectors.
///
/// # Safety
/// - Pointers `a` and `b` must be valid for reads of `dim` f32 elements.
/// - Must only be called on aarch64 platforms with NEON support.
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn l2_squared_f32_neon(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Process 8 elements at a time (two 4-element vectors)
    for i in 0..chunks {
        let offset = i * 8;

        let va0 = vld1q_f32(a.add(offset));
        let vb0 = vld1q_f32(b.add(offset));
        let diff0 = vsubq_f32(va0, vb0);
        sum0 = vfmaq_f32(sum0, diff0, diff0);

        let va1 = vld1q_f32(a.add(offset + 4));
        let vb1 = vld1q_f32(b.add(offset + 4));
        let diff1 = vsubq_f32(va1, vb1);
        sum1 = vfmaq_f32(sum1, diff1, diff1);
    }

    // Combine and reduce
    let sum = vaddq_f32(sum0, sum1);
    let mut result = vaddvq_f32(sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let diff = *a.add(base + i) - *b.add(base + i);
        result += diff * diff;
    }

    result
}

/// NEON inner product for f32 vectors.
///
/// # Safety
/// - Pointers `a` and `b` must be valid for reads of `dim` f32 elements.
/// - Must only be called on aarch64 platforms with NEON support.
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn inner_product_f32_neon(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;

        let va0 = vld1q_f32(a.add(offset));
        let vb0 = vld1q_f32(b.add(offset));
        sum0 = vfmaq_f32(sum0, va0, vb0);

        let va1 = vld1q_f32(a.add(offset + 4));
        let vb1 = vld1q_f32(b.add(offset + 4));
        sum1 = vfmaq_f32(sum1, va1, vb1);
    }

    // Combine and reduce
    let sum = vaddq_f32(sum0, sum1);
    let mut result = vaddvq_f32(sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        result += *a.add(base + i) * *b.add(base + i);
    }

    result
}

/// NEON cosine distance for f32 vectors.
///
/// # Safety
/// - Pointers `a` and `b` must be valid for reads of `dim` f32 elements.
/// - Must only be called on aarch64 platforms with NEON support.
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cosine_distance_f32_neon(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut dot_sum = vdupq_n_f32(0.0);
    let mut norm_a_sum = vdupq_n_f32(0.0);
    let mut norm_b_sum = vdupq_n_f32(0.0);

    let chunks = dim / 4;
    let remainder = dim % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));

        dot_sum = vfmaq_f32(dot_sum, va, vb);
        norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
        norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
    }

    // Reduce to scalars
    let mut dot = vaddvq_f32(dot_sum);
    let mut norm_a = vaddvq_f32(norm_a_sum);
    let mut norm_b = vaddvq_f32(norm_b_sum);

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

    let cosine_sim = (dot / denom).clamp(-1.0, 1.0);
    1.0 - cosine_sim
}

/// Safe wrapper for L2 squared distance.
#[inline]
pub fn l2_squared_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

    let result = unsafe { l2_squared_f32_neon(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
    T::DistanceType::from_f64(result as f64)
}

/// Safe wrapper for inner product.
#[inline]
pub fn inner_product_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

    let result = unsafe { inner_product_f32_neon(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
    T::DistanceType::from_f64(result as f64)
}

/// Safe wrapper for cosine distance.
#[inline]
pub fn cosine_distance_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

    let result = unsafe { cosine_distance_f32_neon(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
    T::DistanceType::from_f64(result as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_l2_squared() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();

        let neon_result = l2_squared_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::l2::l2_squared_scalar(&a, &b, 128);

        assert!((neon_result - scalar_result).abs() < 0.1);
    }

    #[test]
    fn test_neon_inner_product() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 / 100.0).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 / 100.0).collect();

        let neon_result = inner_product_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::ip::inner_product_scalar(&a, &b, 128);

        assert!((neon_result - scalar_result).abs() < 0.01);
    }
}

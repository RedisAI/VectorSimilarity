//! AVX-512 SIMD implementations for distance functions.
//!
//! These functions use 512-bit AVX-512 instructions for maximum throughput.
//! Only available on x86_64 with AVX-512F and AVX-512BW support.

#![cfg(target_arch = "x86_64")]

use crate::types::VectorElement;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX-512 L2 squared distance for f32 vectors.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn l2_squared_f32_avx512(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let chunks = dim / 16;
    let remainder = dim % 16;

    // Process 16 elements at a time
    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    // Reduce to scalar
    let mut result = _mm512_reduce_add_ps(sum);

    // Handle remainder
    let base = chunks * 16;
    for i in 0..remainder {
        let diff = *a.add(base + i) - *b.add(base + i);
        result += diff * diff;
    }

    result
}

/// AVX-512 inner product for f32 vectors.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn inner_product_f32_avx512(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let chunks = dim / 16;
    let remainder = dim % 16;

    // Process 16 elements at a time
    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    // Reduce to scalar
    let mut result = _mm512_reduce_add_ps(sum);

    // Handle remainder
    let base = chunks * 16;
    for i in 0..remainder {
        result += *a.add(base + i) * *b.add(base + i);
    }

    result
}

/// AVX-512 cosine distance for f32 vectors.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn cosine_distance_f32_avx512(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut dot_sum = _mm512_setzero_ps();
    let mut norm_a_sum = _mm512_setzero_ps();
    let mut norm_b_sum = _mm512_setzero_ps();

    let chunks = dim / 16;
    let remainder = dim % 16;

    // Process 16 elements at a time
    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));

        dot_sum = _mm512_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm512_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm512_fmadd_ps(vb, vb, norm_b_sum);
    }

    // Reduce to scalars
    let mut dot = _mm512_reduce_add_ps(dot_sum);
    let mut norm_a = _mm512_reduce_add_ps(norm_a_sum);
    let mut norm_b = _mm512_reduce_add_ps(norm_b_sum);

    // Handle remainder
    let base = chunks * 16;
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
    if is_x86_feature_detected!("avx512f") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { l2_squared_f32_avx512(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        // Fallback to AVX2 or scalar
        super::avx2::l2_squared_f32(a, b, dim)
    }
}

/// Safe wrapper for inner product.
#[inline]
pub fn inner_product_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("avx512f") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { inner_product_f32_avx512(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        super::avx2::inner_product_f32(a, b, dim)
    }
}

/// Safe wrapper for cosine distance.
#[inline]
pub fn cosine_distance_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("avx512f") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { cosine_distance_f32_avx512(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        super::avx2::cosine_distance_f32(a, b, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx512_l2_squared() {
        if !is_x86_feature_detected!("avx512f") {
            println!("AVX-512 not available, skipping test");
            return;
        }

        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..256).map(|i| (i + 1) as f32).collect();

        let avx512_result = l2_squared_f32::<f32>(&a, &b, 256);
        let scalar_result = crate::distance::l2::l2_squared_scalar(&a, &b, 256);

        assert!((avx512_result - scalar_result).abs() < 0.1);
    }
}

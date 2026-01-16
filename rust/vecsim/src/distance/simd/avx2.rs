//! AVX2 SIMD implementations for distance functions.
//!
//! These functions use 256-bit AVX2 instructions for fast distance computation.
//! Only available on x86_64 with AVX2 and FMA support.

#![cfg(target_arch = "x86_64")]

use crate::types::VectorElement;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 L2 squared distance for f32 vectors.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn l2_squared_f32_avx2(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.add(offset));
        let vb = _mm256_loadu_ps(b.add(offset));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let mut result = hsum256_ps(sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let diff = *a.add(base + i) - *b.add(base + i);
        result += diff * diff;
    }

    result
}

/// AVX2 inner product for f32 vectors.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn inner_product_f32_avx2(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.add(offset));
        let vb = _mm256_loadu_ps(b.add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    let mut result = hsum256_ps(sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        result += *a.add(base + i) * *b.add(base + i);
    }

    result
}

/// AVX2 cosine distance for f32 vectors.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosine_distance_f32_avx2(a: *const f32, b: *const f32, dim: usize) -> f32 {
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

        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
    }

    // Horizontal sums
    let mut dot = hsum256_ps(dot_sum);
    let mut norm_a = hsum256_ps(norm_a_sum);
    let mut norm_b = hsum256_ps(norm_b_sum);

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

/// Horizontal sum of 8 f32 values in a 256-bit register.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
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
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // Convert to f32 slices if needed
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { l2_squared_f32_avx2(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        // Fallback to scalar
        crate::distance::l2::l2_squared_scalar(a, b, dim)
    }
}

/// Safe wrapper for inner product.
#[inline]
pub fn inner_product_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { inner_product_f32_avx2(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::ip::inner_product_scalar(a, b, dim)
    }
}

/// Safe wrapper for cosine distance.
#[inline]
pub fn cosine_distance_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { cosine_distance_f32_avx2(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        crate::distance::cosine::cosine_distance_scalar(a, b, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_l2_squared() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();

        let avx2_result = l2_squared_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::l2::l2_squared_scalar(&a, &b, 128);

        assert!((avx2_result - scalar_result).abs() < 0.1);
    }

    #[test]
    fn test_avx2_inner_product() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32 / 100.0).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 / 100.0).collect();

        let avx2_result = inner_product_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::ip::inner_product_scalar(&a, &b, 128);

        assert!((avx2_result - scalar_result).abs() < 0.01);
    }
}

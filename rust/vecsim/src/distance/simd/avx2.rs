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

/// Horizontal sum of 4 f64 values in a 256-bit register.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_pd(v: __m256d) -> f64 {
    // Extract high 128 bits and add to low 128 bits
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(high, low);

    // Horizontal add within 128 bits
    let high64 = _mm_unpackhi_pd(sum128, sum128);
    let result = _mm_add_sd(sum128, high64);

    _mm_cvtsd_f64(result)
}

// =============================================================================
// AVX2 f64 SIMD operations
// =============================================================================

/// AVX2 L2 squared distance for f64 vectors.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn l2_squared_f64_avx2(a: *const f64, b: *const f64, dim: usize) -> f64 {
    let mut sum0 = _mm256_setzero_pd();
    let mut sum1 = _mm256_setzero_pd();
    let chunks = dim / 8;
    let remainder = dim % 8;

    // Process 8 elements at a time (two sets of 4)
    for i in 0..chunks {
        let offset = i * 8;

        let va0 = _mm256_loadu_pd(a.add(offset));
        let vb0 = _mm256_loadu_pd(b.add(offset));
        let diff0 = _mm256_sub_pd(va0, vb0);
        sum0 = _mm256_fmadd_pd(diff0, diff0, sum0);

        let va1 = _mm256_loadu_pd(a.add(offset + 4));
        let vb1 = _mm256_loadu_pd(b.add(offset + 4));
        let diff1 = _mm256_sub_pd(va1, vb1);
        sum1 = _mm256_fmadd_pd(diff1, diff1, sum1);
    }

    // Combine and reduce
    let sum = _mm256_add_pd(sum0, sum1);
    let mut result = hsum256_pd(sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let diff = *a.add(base + i) - *b.add(base + i);
        result += diff * diff;
    }

    result
}

/// AVX2 inner product for f64 vectors.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn inner_product_f64_avx2(a: *const f64, b: *const f64, dim: usize) -> f64 {
    let mut sum0 = _mm256_setzero_pd();
    let mut sum1 = _mm256_setzero_pd();
    let chunks = dim / 8;
    let remainder = dim % 8;

    for i in 0..chunks {
        let offset = i * 8;

        let va0 = _mm256_loadu_pd(a.add(offset));
        let vb0 = _mm256_loadu_pd(b.add(offset));
        sum0 = _mm256_fmadd_pd(va0, vb0, sum0);

        let va1 = _mm256_loadu_pd(a.add(offset + 4));
        let vb1 = _mm256_loadu_pd(b.add(offset + 4));
        sum1 = _mm256_fmadd_pd(va1, vb1, sum1);
    }

    let sum = _mm256_add_pd(sum0, sum1);
    let mut result = hsum256_pd(sum);

    let base = chunks * 8;
    for i in 0..remainder {
        result += *a.add(base + i) * *b.add(base + i);
    }

    result
}

/// AVX2 cosine distance for f64 vectors.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosine_distance_f64_avx2(a: *const f64, b: *const f64, dim: usize) -> f64 {
    let mut dot_sum = _mm256_setzero_pd();
    let mut norm_a_sum = _mm256_setzero_pd();
    let mut norm_b_sum = _mm256_setzero_pd();

    let chunks = dim / 4;
    let remainder = dim % 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm256_loadu_pd(a.add(offset));
        let vb = _mm256_loadu_pd(b.add(offset));

        dot_sum = _mm256_fmadd_pd(va, vb, dot_sum);
        norm_a_sum = _mm256_fmadd_pd(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_pd(vb, vb, norm_b_sum);
    }

    let mut dot = hsum256_pd(dot_sum);
    let mut norm_a = hsum256_pd(norm_a_sum);
    let mut norm_b = hsum256_pd(norm_b_sum);

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

/// Safe wrapper for f64 L2 squared distance.
#[inline]
pub fn l2_squared_f64(a: &[f64], b: &[f64], dim: usize) -> f64 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { l2_squared_f64_avx2(a.as_ptr(), b.as_ptr(), dim) }
    } else {
        crate::distance::l2::l2_squared_scalar_f64(a, b, dim)
    }
}

/// Safe wrapper for f64 inner product.
#[inline]
pub fn inner_product_f64(a: &[f64], b: &[f64], dim: usize) -> f64 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { inner_product_f64_avx2(a.as_ptr(), b.as_ptr(), dim) }
    } else {
        crate::distance::ip::inner_product_scalar_f64(a, b, dim)
    }
}

/// Safe wrapper for f64 cosine distance.
#[inline]
pub fn cosine_distance_f64(a: &[f64], b: &[f64], dim: usize) -> f64 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { cosine_distance_f64_avx2(a.as_ptr(), b.as_ptr(), dim) }
    } else {
        // Scalar fallback
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for i in 0..dim {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom < 1e-30 {
            return 1.0;
        }
        1.0 - (dot / denom).clamp(-1.0, 1.0)
    }
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

    fn is_avx2_available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    #[test]
    fn test_avx2_l2_squared() {
        if !is_avx2_available() {
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
        if !is_avx2_available() {
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32 / 100.0).collect();
        let b: Vec<f32> = (0..128).map(|i| (128 - i) as f32 / 100.0).collect();

        let avx2_result = inner_product_f32::<f32>(&a, &b, 128);
        let scalar_result = crate::distance::ip::inner_product_scalar(&a, &b, 128);

        assert!((avx2_result - scalar_result).abs() < 0.01);
    }

    // f64 tests
    #[test]
    fn test_avx2_l2_squared_f64() {
        if !is_avx2_available() {
            return;
        }

        let a: Vec<f64> = (0..128).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..128).map(|i| (i + 1) as f64).collect();

        let result = l2_squared_f64(&a, &b, 128);
        // Each diff is 1, squared is 1, sum of 128 ones = 128
        assert!((result - 128.0).abs() < 1e-10);
    }

    #[test]
    fn test_avx2_inner_product_f64() {
        if !is_avx2_available() {
            return;
        }

        let a: Vec<f64> = (0..128).map(|i| i as f64 / 100.0).collect();
        let b: Vec<f64> = (0..128).map(|i| (128 - i) as f64 / 100.0).collect();

        let result = inner_product_f64(&a, &b, 128);
        let expected: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_avx2_cosine_f64() {
        if !is_avx2_available() {
            return;
        }

        // Identical vectors should have distance 0
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let result = cosine_distance_f64(&a, &a, 4);
        assert!(result.abs() < 1e-10);

        // Orthogonal vectors should have distance 1
        let a: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0];
        let b: Vec<f64> = vec![0.0, 1.0, 0.0, 0.0];
        let result = cosine_distance_f64(&a, &b, 4);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_avx2_f64_remainder() {
        if !is_avx2_available() {
            return;
        }

        // Test with non-aligned dimension
        let a: Vec<f64> = (0..131).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..131).map(|i| (i + 1) as f64).collect();

        let result = l2_squared_f64(&a, &b, 131);
        assert!((result - 131.0).abs() < 1e-10);
    }
}

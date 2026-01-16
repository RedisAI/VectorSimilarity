//! AVX-512 BF16 optimized distance functions for BFloat16 vectors.
//!
//! This module provides SIMD-optimized distance computations for bfloat16
//! floating point vectors using AVX-512.
//!
//! BFloat16 has a simple relationship with f32: the bf16 bits are the upper
//! 16 bits of the f32 representation. This allows very efficient conversion:
//! - bf16 -> f32: shift left by 16 bits
//! - f32 -> bf16: shift right by 16 bits (with rounding)
//!
//! This module uses AVX-512 to:
//! - Load 32 bf16 values at a time (512 bits)
//! - Efficiently convert to f32 using the shift trick
//! - Compute distances in f32 using FMA

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::types::BFloat16;

/// Check if AVX-512 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
}

/// Compute L2 squared distance between two BFloat16 vectors using AVX-512.
///
/// # Safety
/// - Requires AVX-512F support
/// - Pointers must be valid for `dim` elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_squared_bf16_avx512(
    a: *const BFloat16,
    b: *const BFloat16,
    dim: usize,
) -> f32 {
    let a = a as *const u16;
    let b = b as *const u16;

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();

    let unroll = dim / 32 * 32;
    let mut offset = 0;

    // Process 32 bf16 elements per iteration (2x16)
    while offset < unroll {
        // Load 16 bf16 values at a time (256 bits)
        let a_raw0 = _mm256_loadu_si256((a.add(offset)) as *const __m256i);
        let a_raw1 = _mm256_loadu_si256((a.add(offset + 16)) as *const __m256i);
        let b_raw0 = _mm256_loadu_si256((b.add(offset)) as *const __m256i);
        let b_raw1 = _mm256_loadu_si256((b.add(offset + 16)) as *const __m256i);

        // Convert bf16 to f32 by zero-extending to 32-bit and shifting left by 16
        // bf16 bits are the upper 16 bits of f32
        let a_32_0 = _mm512_cvtepu16_epi32(a_raw0);
        let a_32_1 = _mm512_cvtepu16_epi32(a_raw1);
        let b_32_0 = _mm512_cvtepu16_epi32(b_raw0);
        let b_32_1 = _mm512_cvtepu16_epi32(b_raw1);

        // Shift left by 16 to get f32 representation
        let a_shifted0 = _mm512_slli_epi32(a_32_0, 16);
        let a_shifted1 = _mm512_slli_epi32(a_32_1, 16);
        let b_shifted0 = _mm512_slli_epi32(b_32_0, 16);
        let b_shifted1 = _mm512_slli_epi32(b_32_1, 16);

        // Reinterpret as f32
        let a_f32_0 = _mm512_castsi512_ps(a_shifted0);
        let a_f32_1 = _mm512_castsi512_ps(a_shifted1);
        let b_f32_0 = _mm512_castsi512_ps(b_shifted0);
        let b_f32_1 = _mm512_castsi512_ps(b_shifted1);

        // Compute differences
        let diff0 = _mm512_sub_ps(a_f32_0, b_f32_0);
        let diff1 = _mm512_sub_ps(a_f32_1, b_f32_1);

        // Accumulate squared differences using FMA
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);

        offset += 32;
    }

    // Process remaining 16-element chunks
    while offset + 16 <= dim {
        let a_raw = _mm256_loadu_si256((a.add(offset)) as *const __m256i);
        let b_raw = _mm256_loadu_si256((b.add(offset)) as *const __m256i);

        let a_32 = _mm512_cvtepu16_epi32(a_raw);
        let b_32 = _mm512_cvtepu16_epi32(b_raw);

        let a_shifted = _mm512_slli_epi32(a_32, 16);
        let b_shifted = _mm512_slli_epi32(b_32, 16);

        let a_f32 = _mm512_castsi512_ps(a_shifted);
        let b_f32 = _mm512_castsi512_ps(b_shifted);

        let diff = _mm512_sub_ps(a_f32, b_f32);
        sum0 = _mm512_fmadd_ps(diff, diff, sum0);

        offset += 16;
    }

    // Reduce sums
    let total = _mm512_add_ps(sum0, sum1);
    let mut result = _mm512_reduce_add_ps(total);

    // Handle remaining elements with scalar code
    while offset < dim {
        let av = BFloat16::from_bits(*a.add(offset)).to_f32();
        let bv = BFloat16::from_bits(*b.add(offset)).to_f32();
        let diff = av - bv;
        result += diff * diff;
        offset += 1;
    }

    result
}

/// Compute inner product between two BFloat16 vectors using AVX-512.
///
/// # Safety
/// - Requires AVX-512F support
/// - Pointers must be valid for `dim` elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn inner_product_bf16_avx512(
    a: *const BFloat16,
    b: *const BFloat16,
    dim: usize,
) -> f32 {
    let a = a as *const u16;
    let b = b as *const u16;

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();

    let unroll = dim / 32 * 32;
    let mut offset = 0;

    // Process 32 elements per iteration
    while offset < unroll {
        let a_raw0 = _mm256_loadu_si256((a.add(offset)) as *const __m256i);
        let a_raw1 = _mm256_loadu_si256((a.add(offset + 16)) as *const __m256i);
        let b_raw0 = _mm256_loadu_si256((b.add(offset)) as *const __m256i);
        let b_raw1 = _mm256_loadu_si256((b.add(offset + 16)) as *const __m256i);

        let a_32_0 = _mm512_cvtepu16_epi32(a_raw0);
        let a_32_1 = _mm512_cvtepu16_epi32(a_raw1);
        let b_32_0 = _mm512_cvtepu16_epi32(b_raw0);
        let b_32_1 = _mm512_cvtepu16_epi32(b_raw1);

        let a_shifted0 = _mm512_slli_epi32(a_32_0, 16);
        let a_shifted1 = _mm512_slli_epi32(a_32_1, 16);
        let b_shifted0 = _mm512_slli_epi32(b_32_0, 16);
        let b_shifted1 = _mm512_slli_epi32(b_32_1, 16);

        let a_f32_0 = _mm512_castsi512_ps(a_shifted0);
        let a_f32_1 = _mm512_castsi512_ps(a_shifted1);
        let b_f32_0 = _mm512_castsi512_ps(b_shifted0);
        let b_f32_1 = _mm512_castsi512_ps(b_shifted1);

        // Accumulate products using FMA
        sum0 = _mm512_fmadd_ps(a_f32_0, b_f32_0, sum0);
        sum1 = _mm512_fmadd_ps(a_f32_1, b_f32_1, sum1);

        offset += 32;
    }

    // Process remaining 16-element chunks
    while offset + 16 <= dim {
        let a_raw = _mm256_loadu_si256((a.add(offset)) as *const __m256i);
        let b_raw = _mm256_loadu_si256((b.add(offset)) as *const __m256i);

        let a_32 = _mm512_cvtepu16_epi32(a_raw);
        let b_32 = _mm512_cvtepu16_epi32(b_raw);

        let a_shifted = _mm512_slli_epi32(a_32, 16);
        let b_shifted = _mm512_slli_epi32(b_32, 16);

        let a_f32 = _mm512_castsi512_ps(a_shifted);
        let b_f32 = _mm512_castsi512_ps(b_shifted);

        sum0 = _mm512_fmadd_ps(a_f32, b_f32, sum0);

        offset += 16;
    }

    let total = _mm512_add_ps(sum0, sum1);
    let mut result = _mm512_reduce_add_ps(total);

    // Handle remaining elements
    while offset < dim {
        let av = BFloat16::from_bits(*a.add(offset)).to_f32();
        let bv = BFloat16::from_bits(*b.add(offset)).to_f32();
        result += av * bv;
        offset += 1;
    }

    result
}

/// Compute cosine distance between two BFloat16 vectors using AVX-512.
///
/// # Safety
/// - Requires AVX-512F support
/// - Pointers must be valid for `dim` elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn cosine_distance_bf16_avx512(
    a: *const BFloat16,
    b: *const BFloat16,
    dim: usize,
) -> f32 {
    let a = a as *const u16;
    let b = b as *const u16;

    let mut dot_sum = _mm512_setzero_ps();
    let mut norm_a_sum = _mm512_setzero_ps();
    let mut norm_b_sum = _mm512_setzero_ps();

    let unroll = dim / 16 * 16;
    let mut offset = 0;

    // Process 16 elements per iteration
    while offset < unroll {
        let a_raw = _mm256_loadu_si256((a.add(offset)) as *const __m256i);
        let b_raw = _mm256_loadu_si256((b.add(offset)) as *const __m256i);

        let a_32 = _mm512_cvtepu16_epi32(a_raw);
        let b_32 = _mm512_cvtepu16_epi32(b_raw);

        let a_shifted = _mm512_slli_epi32(a_32, 16);
        let b_shifted = _mm512_slli_epi32(b_32, 16);

        let a_f32 = _mm512_castsi512_ps(a_shifted);
        let b_f32 = _mm512_castsi512_ps(b_shifted);

        dot_sum = _mm512_fmadd_ps(a_f32, b_f32, dot_sum);
        norm_a_sum = _mm512_fmadd_ps(a_f32, a_f32, norm_a_sum);
        norm_b_sum = _mm512_fmadd_ps(b_f32, b_f32, norm_b_sum);

        offset += 16;
    }

    let mut dot = _mm512_reduce_add_ps(dot_sum);
    let mut norm_a = _mm512_reduce_add_ps(norm_a_sum);
    let mut norm_b = _mm512_reduce_add_ps(norm_b_sum);

    // Handle remaining elements
    while offset < dim {
        let av = BFloat16::from_bits(*a.add(offset)).to_f32();
        let bv = BFloat16::from_bits(*b.add(offset)).to_f32();
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
        offset += 1;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-30 {
        return 1.0;
    }

    let cosine_sim = (dot / denom).clamp(-1.0, 1.0);
    1.0 - cosine_sim
}

// Safe wrappers with runtime dispatch

/// Safe L2 squared distance for BFloat16 with automatic dispatch.
pub fn l2_squared_bf16(a: &[BFloat16], b: &[BFloat16], dim: usize) -> f32 {
    debug_assert_eq!(a.len(), dim);
    debug_assert_eq!(b.len(), dim);

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            return unsafe { l2_squared_bf16_avx512(a.as_ptr(), b.as_ptr(), dim) };
        }
    }

    // Scalar fallback
    l2_squared_bf16_scalar(a, b, dim)
}

/// Safe inner product for BFloat16 with automatic dispatch.
pub fn inner_product_bf16(a: &[BFloat16], b: &[BFloat16], dim: usize) -> f32 {
    debug_assert_eq!(a.len(), dim);
    debug_assert_eq!(b.len(), dim);

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            return unsafe { inner_product_bf16_avx512(a.as_ptr(), b.as_ptr(), dim) };
        }
    }

    // Scalar fallback
    inner_product_bf16_scalar(a, b, dim)
}

/// Safe cosine distance for BFloat16 with automatic dispatch.
pub fn cosine_distance_bf16(a: &[BFloat16], b: &[BFloat16], dim: usize) -> f32 {
    debug_assert_eq!(a.len(), dim);
    debug_assert_eq!(b.len(), dim);

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            return unsafe { cosine_distance_bf16_avx512(a.as_ptr(), b.as_ptr(), dim) };
        }
    }

    // Scalar fallback
    cosine_distance_bf16_scalar(a, b, dim)
}

// Scalar implementations

fn l2_squared_bf16_scalar(a: &[BFloat16], b: &[BFloat16], dim: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..dim {
        let diff = a[i].to_f32() - b[i].to_f32();
        sum += diff * diff;
    }
    sum
}

fn inner_product_bf16_scalar(a: &[BFloat16], b: &[BFloat16], dim: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..dim {
        sum += a[i].to_f32() * b[i].to_f32();
    }
    sum
}

fn cosine_distance_bf16_scalar(a: &[BFloat16], b: &[BFloat16], dim: usize) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..dim {
        let av = a[i].to_f32();
        let bv = b[i].to_f32();
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-30 {
        return 1.0;
    }

    let cosine_sim = (dot / denom).clamp(-1.0, 1.0);
    1.0 - cosine_sim
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors(dim: usize) -> (Vec<BFloat16>, Vec<BFloat16>) {
        let a: Vec<BFloat16> = (0..dim)
            .map(|i| BFloat16::from_f32((i as f32) / (dim as f32)))
            .collect();
        let b: Vec<BFloat16> = (0..dim)
            .map(|i| BFloat16::from_f32(((dim - i) as f32) / (dim as f32)))
            .collect();
        (a, b)
    }

    #[test]
    fn test_bf16_l2_squared() {
        for &dim in &[16, 32, 64, 100, 128, 256] {
            let (a, b) = create_test_vectors(dim);
            let scalar_result = l2_squared_bf16_scalar(&a, &b, dim);
            let simd_result = l2_squared_bf16(&a, &b, dim);

            // BFloat16 has lower precision, so allow more tolerance
            assert!(
                (scalar_result - simd_result).abs() < 0.1,
                "dim={}: scalar={}, simd={}",
                dim,
                scalar_result,
                simd_result
            );
        }
    }

    #[test]
    fn test_bf16_inner_product() {
        for &dim in &[16, 32, 64, 100, 128, 256] {
            let (a, b) = create_test_vectors(dim);
            let scalar_result = inner_product_bf16_scalar(&a, &b, dim);
            let simd_result = inner_product_bf16(&a, &b, dim);

            assert!(
                (scalar_result - simd_result).abs() < 0.1,
                "dim={}: scalar={}, simd={}",
                dim,
                scalar_result,
                simd_result
            );
        }
    }

    #[test]
    fn test_bf16_cosine_distance() {
        for &dim in &[16, 32, 64, 100, 128, 256] {
            let (a, b) = create_test_vectors(dim);
            let scalar_result = cosine_distance_bf16_scalar(&a, &b, dim);
            let simd_result = cosine_distance_bf16(&a, &b, dim);

            assert!(
                (scalar_result - simd_result).abs() < 0.1,
                "dim={}: scalar={}, simd={}",
                dim,
                scalar_result,
                simd_result
            );
        }
    }

    #[test]
    fn test_bf16_self_distance() {
        let dim = 128;
        let (a, _) = create_test_vectors(dim);

        // Self L2 distance should be 0
        let l2 = l2_squared_bf16(&a, &a, dim);
        assert!(l2.abs() < 0.01, "Self L2 distance should be ~0, got {}", l2);

        // Self cosine distance should be 0
        let cosine = cosine_distance_bf16(&a, &a, dim);
        assert!(
            cosine.abs() < 0.01,
            "Self cosine distance should be ~0, got {}",
            cosine
        );
    }
}

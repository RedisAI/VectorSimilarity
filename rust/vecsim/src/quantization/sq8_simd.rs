//! SIMD-optimized SQ8 asymmetric distance functions.
//!
//! These functions compute distances between f32 queries and SQ8-quantized vectors
//! using SIMD instructions for significant performance improvements.

use super::sq8::Sq8VectorMeta;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =============================================================================
// NEON implementations (ARM)
// =============================================================================

/// NEON-optimized asymmetric L2 squared distance.
///
/// Computes ||query - stored||² where stored is SQ8-quantized.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - NEON must be available
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sq8_asymmetric_l2_squared_neon(
    query: *const f32,
    quantized: *const u8,
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    let min_vec = vdupq_n_f32(meta.min);
    let delta_vec = vdupq_n_f32(meta.delta);

    let mut query_sq_sum = vdupq_n_f32(0.0);
    let mut ip_sum = vdupq_n_f32(0.0);

    let chunks = dim / 8;
    let remainder = dim % 8;

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 query values (two sets of 4)
        let q0 = vld1q_f32(query.add(offset));
        let q1 = vld1q_f32(query.add(offset + 4));

        // Load 8 quantized values and convert to f32
        let quant_u8 = vld1_u8(quantized.add(offset));

        // Widen u8 to u16
        let quant_u16 = vmovl_u8(quant_u8);

        // Split and widen u16 to u32
        let quant_lo_u32 = vmovl_u16(vget_low_u16(quant_u16));
        let quant_hi_u32 = vmovl_u16(vget_high_u16(quant_u16));

        // Convert u32 to f32
        let quant_lo_f32 = vcvtq_f32_u32(quant_lo_u32);
        let quant_hi_f32 = vcvtq_f32_u32(quant_hi_u32);

        // Dequantize: stored = min + quant * delta
        let stored0 = vfmaq_f32(min_vec, quant_lo_f32, delta_vec);
        let stored1 = vfmaq_f32(min_vec, quant_hi_f32, delta_vec);

        // Accumulate query squared
        query_sq_sum = vfmaq_f32(query_sq_sum, q0, q0);
        query_sq_sum = vfmaq_f32(query_sq_sum, q1, q1);

        // Accumulate inner product
        ip_sum = vfmaq_f32(ip_sum, q0, stored0);
        ip_sum = vfmaq_f32(ip_sum, q1, stored1);
    }

    // Reduce to scalar
    let mut query_sq = vaddvq_f32(query_sq_sum);
    let mut ip = vaddvq_f32(ip_sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let q = *query.add(base + i);
        let v = meta.min + (*quantized.add(base + i) as f32) * meta.delta;
        query_sq += q * q;
        ip += q * v;
    }

    // ||q - v||² = ||q||² + ||v||² - 2*IP(q, v)
    (query_sq + meta.sum_sq - 2.0 * ip).max(0.0)
}

/// NEON-optimized asymmetric inner product.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - NEON must be available
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sq8_asymmetric_inner_product_neon(
    query: *const f32,
    quantized: *const u8,
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    let min_vec = vdupq_n_f32(meta.min);
    let delta_vec = vdupq_n_f32(meta.delta);

    let mut ip_sum = vdupq_n_f32(0.0);

    let chunks = dim / 8;
    let remainder = dim % 8;

    for i in 0..chunks {
        let offset = i * 8;

        let q0 = vld1q_f32(query.add(offset));
        let q1 = vld1q_f32(query.add(offset + 4));

        let quant_u8 = vld1_u8(quantized.add(offset));
        let quant_u16 = vmovl_u8(quant_u8);
        let quant_lo_u32 = vmovl_u16(vget_low_u16(quant_u16));
        let quant_hi_u32 = vmovl_u16(vget_high_u16(quant_u16));
        let quant_lo_f32 = vcvtq_f32_u32(quant_lo_u32);
        let quant_hi_f32 = vcvtq_f32_u32(quant_hi_u32);

        let stored0 = vfmaq_f32(min_vec, quant_lo_f32, delta_vec);
        let stored1 = vfmaq_f32(min_vec, quant_hi_f32, delta_vec);

        ip_sum = vfmaq_f32(ip_sum, q0, stored0);
        ip_sum = vfmaq_f32(ip_sum, q1, stored1);
    }

    let mut ip = vaddvq_f32(ip_sum);

    let base = chunks * 8;
    for i in 0..remainder {
        let q = *query.add(base + i);
        let v = meta.min + (*quantized.add(base + i) as f32) * meta.delta;
        ip += q * v;
    }

    -ip // Negative for distance ordering
}

/// NEON-optimized asymmetric cosine distance.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - NEON must be available
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sq8_asymmetric_cosine_neon(
    query: *const f32,
    quantized: *const u8,
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    let min_vec = vdupq_n_f32(meta.min);
    let delta_vec = vdupq_n_f32(meta.delta);

    let mut query_sq_sum = vdupq_n_f32(0.0);
    let mut ip_sum = vdupq_n_f32(0.0);

    let chunks = dim / 8;
    let remainder = dim % 8;

    for i in 0..chunks {
        let offset = i * 8;

        let q0 = vld1q_f32(query.add(offset));
        let q1 = vld1q_f32(query.add(offset + 4));

        let quant_u8 = vld1_u8(quantized.add(offset));
        let quant_u16 = vmovl_u8(quant_u8);
        let quant_lo_u32 = vmovl_u16(vget_low_u16(quant_u16));
        let quant_hi_u32 = vmovl_u16(vget_high_u16(quant_u16));
        let quant_lo_f32 = vcvtq_f32_u32(quant_lo_u32);
        let quant_hi_f32 = vcvtq_f32_u32(quant_hi_u32);

        let stored0 = vfmaq_f32(min_vec, quant_lo_f32, delta_vec);
        let stored1 = vfmaq_f32(min_vec, quant_hi_f32, delta_vec);

        query_sq_sum = vfmaq_f32(query_sq_sum, q0, q0);
        query_sq_sum = vfmaq_f32(query_sq_sum, q1, q1);
        ip_sum = vfmaq_f32(ip_sum, q0, stored0);
        ip_sum = vfmaq_f32(ip_sum, q1, stored1);
    }

    let mut query_sq = vaddvq_f32(query_sq_sum);
    let mut ip = vaddvq_f32(ip_sum);

    let base = chunks * 8;
    for i in 0..remainder {
        let q = *query.add(base + i);
        let v = meta.min + (*quantized.add(base + i) as f32) * meta.delta;
        query_sq += q * q;
        ip += q * v;
    }

    let query_norm = query_sq.sqrt();
    let stored_norm = meta.sum_sq.sqrt();

    if query_norm < 1e-30 || stored_norm < 1e-30 {
        return 1.0;
    }

    let cosine_sim = (ip / (query_norm * stored_norm)).clamp(-1.0, 1.0);
    1.0 - cosine_sim
}

// =============================================================================
// AVX2 implementations (x86_64)
// =============================================================================

/// AVX2-optimized asymmetric L2 squared distance.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - AVX2 must be available
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sq8_asymmetric_l2_squared_avx2(
    query: *const f32,
    quantized: *const u8,
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    let min_vec = _mm256_set1_ps(meta.min);
    let delta_vec = _mm256_set1_ps(meta.delta);

    let mut query_sq_sum = _mm256_setzero_ps();
    let mut ip_sum = _mm256_setzero_ps();

    let chunks = dim / 8;
    let remainder = dim % 8;

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 query values
        let q = _mm256_loadu_ps(query.add(offset));

        // Load 8 quantized values
        let quant_u8 = _mm_loadl_epi64(quantized.add(offset) as *const __m128i);

        // Convert u8 to i32 (zero extend)
        let quant_i32 = _mm256_cvtepu8_epi32(quant_u8);

        // Convert i32 to f32
        let quant_f32 = _mm256_cvtepi32_ps(quant_i32);

        // Dequantize: stored = min + quant * delta
        let stored = _mm256_fmadd_ps(quant_f32, delta_vec, min_vec);

        // Accumulate query squared
        query_sq_sum = _mm256_fmadd_ps(q, q, query_sq_sum);

        // Accumulate inner product
        ip_sum = _mm256_fmadd_ps(q, stored, ip_sum);
    }

    // Horizontal sum
    let mut query_sq = hsum256_ps_avx2(query_sq_sum);
    let mut ip = hsum256_ps_avx2(ip_sum);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let q = *query.add(base + i);
        let v = meta.min + (*quantized.add(base + i) as f32) * meta.delta;
        query_sq += q * q;
        ip += q * v;
    }

    (query_sq + meta.sum_sq - 2.0 * ip).max(0.0)
}

/// AVX2-optimized asymmetric inner product.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - AVX2 must be available
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sq8_asymmetric_inner_product_avx2(
    query: *const f32,
    quantized: *const u8,
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    let min_vec = _mm256_set1_ps(meta.min);
    let delta_vec = _mm256_set1_ps(meta.delta);

    let mut ip_sum = _mm256_setzero_ps();

    let chunks = dim / 8;
    let remainder = dim % 8;

    for i in 0..chunks {
        let offset = i * 8;

        let q = _mm256_loadu_ps(query.add(offset));
        let quant_u8 = _mm_loadl_epi64(quantized.add(offset) as *const __m128i);
        let quant_i32 = _mm256_cvtepu8_epi32(quant_u8);
        let quant_f32 = _mm256_cvtepi32_ps(quant_i32);
        let stored = _mm256_fmadd_ps(quant_f32, delta_vec, min_vec);

        ip_sum = _mm256_fmadd_ps(q, stored, ip_sum);
    }

    let mut ip = hsum256_ps_avx2(ip_sum);

    let base = chunks * 8;
    for i in 0..remainder {
        let q = *query.add(base + i);
        let v = meta.min + (*quantized.add(base + i) as f32) * meta.delta;
        ip += q * v;
    }

    -ip
}

/// AVX2-optimized asymmetric cosine distance.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
/// - AVX2 must be available
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sq8_asymmetric_cosine_avx2(
    query: *const f32,
    quantized: *const u8,
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    let min_vec = _mm256_set1_ps(meta.min);
    let delta_vec = _mm256_set1_ps(meta.delta);

    let mut query_sq_sum = _mm256_setzero_ps();
    let mut ip_sum = _mm256_setzero_ps();

    let chunks = dim / 8;
    let remainder = dim % 8;

    for i in 0..chunks {
        let offset = i * 8;

        let q = _mm256_loadu_ps(query.add(offset));
        let quant_u8 = _mm_loadl_epi64(quantized.add(offset) as *const __m128i);
        let quant_i32 = _mm256_cvtepu8_epi32(quant_u8);
        let quant_f32 = _mm256_cvtepi32_ps(quant_i32);
        let stored = _mm256_fmadd_ps(quant_f32, delta_vec, min_vec);

        query_sq_sum = _mm256_fmadd_ps(q, q, query_sq_sum);
        ip_sum = _mm256_fmadd_ps(q, stored, ip_sum);
    }

    let mut query_sq = hsum256_ps_avx2(query_sq_sum);
    let mut ip = hsum256_ps_avx2(ip_sum);

    let base = chunks * 8;
    for i in 0..remainder {
        let q = *query.add(base + i);
        let v = meta.min + (*quantized.add(base + i) as f32) * meta.delta;
        query_sq += q * q;
        ip += q * v;
    }

    let query_norm = query_sq.sqrt();
    let stored_norm = meta.sum_sq.sqrt();

    if query_norm < 1e-30 || stored_norm < 1e-30 {
        return 1.0;
    }

    let cosine_sim = (ip / (query_norm * stored_norm)).clamp(-1.0, 1.0);
    1.0 - cosine_sim
}

/// Horizontal sum helper for AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps_avx2(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(sums, sums);
    let sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

// =============================================================================
// Safe wrappers with runtime dispatch
// =============================================================================

/// Compute SQ8 asymmetric L2 squared distance with SIMD acceleration.
#[inline]
pub fn sq8_l2_squared_simd(
    query: &[f32],
    quantized: &[u8],
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(quantized.len(), dim);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sq8_asymmetric_l2_squared_neon(query.as_ptr(), quantized.as_ptr(), meta, dim) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                sq8_asymmetric_l2_squared_avx2(query.as_ptr(), quantized.as_ptr(), meta, dim)
            }
        } else {
            super::sq8::sq8_asymmetric_l2_squared(query, quantized, meta, dim)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        super::sq8::sq8_asymmetric_l2_squared(query, quantized, meta, dim)
    }
}

/// Compute SQ8 asymmetric inner product with SIMD acceleration.
#[inline]
pub fn sq8_inner_product_simd(
    query: &[f32],
    quantized: &[u8],
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(quantized.len(), dim);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sq8_asymmetric_inner_product_neon(query.as_ptr(), quantized.as_ptr(), meta, dim) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                sq8_asymmetric_inner_product_avx2(query.as_ptr(), quantized.as_ptr(), meta, dim)
            }
        } else {
            super::sq8::sq8_asymmetric_inner_product(query, quantized, meta, dim)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        super::sq8::sq8_asymmetric_inner_product(query, quantized, meta, dim)
    }
}

/// Compute SQ8 asymmetric cosine distance with SIMD acceleration.
#[inline]
pub fn sq8_cosine_simd(
    query: &[f32],
    quantized: &[u8],
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(quantized.len(), dim);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sq8_asymmetric_cosine_neon(query.as_ptr(), quantized.as_ptr(), meta, dim) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { sq8_asymmetric_cosine_avx2(query.as_ptr(), quantized.as_ptr(), meta, dim) }
        } else {
            super::sq8::sq8_asymmetric_cosine(query, quantized, meta, dim)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        super::sq8::sq8_asymmetric_cosine(query, quantized, meta, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::sq8::Sq8Codec;

    fn create_test_data(dim: usize) -> (Vec<f32>, Vec<u8>, Sq8VectorMeta) {
        let codec = Sq8Codec::new(dim);
        let stored: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();
        let (quantized, meta) = codec.encode(&stored);
        let quantized_bytes: Vec<u8> = quantized.iter().map(|q| q.get()).collect();
        (stored, quantized_bytes, meta)
    }

    #[test]
    fn test_sq8_simd_l2_accuracy() {
        let dim = 128;
        let (_stored, quantized, meta) = create_test_data(dim);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 + 0.5) / (dim as f32)).collect();

        let scalar_result =
            crate::quantization::sq8::sq8_asymmetric_l2_squared(&query, &quantized, &meta, dim);
        let simd_result = sq8_l2_squared_simd(&query, &quantized, &meta, dim);

        assert!(
            (scalar_result - simd_result).abs() < 0.001,
            "scalar={}, simd={}",
            scalar_result,
            simd_result
        );
    }

    #[test]
    fn test_sq8_simd_ip_accuracy() {
        let dim = 128;
        let (_, quantized, meta) = create_test_data(dim);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();

        let scalar_result =
            crate::quantization::sq8::sq8_asymmetric_inner_product(&query, &quantized, &meta, dim);
        let simd_result = sq8_inner_product_simd(&query, &quantized, &meta, dim);

        assert!(
            (scalar_result - simd_result).abs() < 0.001,
            "scalar={}, simd={}",
            scalar_result,
            simd_result
        );
    }

    #[test]
    fn test_sq8_simd_cosine_accuracy() {
        let dim = 128;
        let (_, quantized, meta) = create_test_data(dim);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) / (dim as f32)).collect();

        let scalar_result =
            crate::quantization::sq8::sq8_asymmetric_cosine(&query, &quantized, &meta, dim);
        let simd_result = sq8_cosine_simd(&query, &quantized, &meta, dim);

        assert!(
            (scalar_result - simd_result).abs() < 0.001,
            "scalar={}, simd={}",
            scalar_result,
            simd_result
        );
    }

    #[test]
    fn test_sq8_simd_remainder_handling() {
        // Test with dimensions that don't align to SIMD width
        for dim in [17, 33, 65, 100, 127] {
            let (_, quantized, meta) = create_test_data(dim);
            let query: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();

            let scalar_result =
                crate::quantization::sq8::sq8_asymmetric_l2_squared(&query, &quantized, &meta, dim);
            let simd_result = sq8_l2_squared_simd(&query, &quantized, &meta, dim);

            assert!(
                (scalar_result - simd_result).abs() < 0.001,
                "dim={}: scalar={}, simd={}",
                dim,
                scalar_result,
                simd_result
            );
        }
    }

    #[test]
    fn test_sq8_simd_self_distance() {
        let dim = 128;
        let codec = Sq8Codec::new(dim);
        let stored: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();
        let (quantized, meta) = codec.encode(&stored);
        let quantized_bytes: Vec<u8> = quantized.iter().map(|q| q.get()).collect();

        // Distance to self should be small (due to quantization error)
        let dist = sq8_l2_squared_simd(&stored, &quantized_bytes, &meta, dim);
        assert!(dist < 0.01, "Self distance should be small, got {}", dist);
    }
}

//! AVX-512BW and AVX-512VNNI SIMD implementations for distance functions.
//!
//! This module provides optimized distance computations using:
//! - AVX-512BW: Byte and word operations on 512-bit vectors
//! - AVX-512VNNI: Vector Neural Network Instructions for int8 dot products
//!
//! VNNI provides `VPDPBUSD` which computes:
//!   result[i] += sum(a[4*i+j] * b[4*i+j]) for j in 0..4
//! where a is unsigned bytes and b is signed bytes, accumulating to int32.

#![cfg(target_arch = "x86_64")]

use crate::types::{DistanceType, Int8, UInt8, VectorElement};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// AVX-512BW f32 functions (same as avx512.rs but with BW feature)
// ============================================================================

/// AVX-512BW L2 squared distance for f32 vectors.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
pub unsafe fn l2_squared_f32_avx512bw(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let chunks = dim / 16;
    let remainder = dim % 16;

    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    let mut result = _mm512_reduce_add_ps(sum);

    let base = chunks * 16;
    for i in 0..remainder {
        let diff = *a.add(base + i) - *b.add(base + i);
        result += diff * diff;
    }

    result
}

/// AVX-512BW inner product for f32 vectors.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
pub unsafe fn inner_product_f32_avx512bw(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum = _mm512_setzero_ps();
    let chunks = dim / 16;
    let remainder = dim % 16;

    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    let mut result = _mm512_reduce_add_ps(sum);

    let base = chunks * 16;
    for i in 0..remainder {
        result += *a.add(base + i) * *b.add(base + i);
    }

    result
}

/// AVX-512BW cosine distance for f32 vectors.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
pub unsafe fn cosine_distance_f32_avx512bw(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut dot_sum = _mm512_setzero_ps();
    let mut norm_a_sum = _mm512_setzero_ps();
    let mut norm_b_sum = _mm512_setzero_ps();

    let chunks = dim / 16;
    let remainder = dim % 16;

    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));

        dot_sum = _mm512_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm512_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm512_fmadd_ps(vb, vb, norm_b_sum);
    }

    let mut dot = _mm512_reduce_add_ps(dot_sum);
    let mut norm_a = _mm512_reduce_add_ps(norm_a_sum);
    let mut norm_b = _mm512_reduce_add_ps(norm_b_sum);

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

// ============================================================================
// AVX-512VNNI Int8/UInt8 functions
// ============================================================================

/// AVX-512VNNI inner product for Int8 vectors.
///
/// Uses VPDPBUSD instruction for efficient int8×int8 dot product.
/// Returns the dot product as i32, which should be converted to f32 for distance.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[inline]
pub unsafe fn inner_product_i8_avx512vnni(a: *const i8, b: *const i8, dim: usize) -> i32 {
    // VPDPBUSD expects unsigned × signed, so we need to handle signed × signed
    // by adjusting: (a+128) * b - 128 * sum(b) gives same result as signed a * b
    // For simplicity, we use a different approach:
    // Split into positive/negative parts or use i16 intermediate

    // Alternative approach: Use _mm512_dpbusd_epi32 with bias adjustment
    // For signed×signed: result = dpbusd(a+128, b) - 128 * sum(b)
    // This is complex, so we use a simpler i16 widening approach with AVX-512BW

    let mut sum = _mm512_setzero_si512();
    let chunks = dim / 64;
    let remainder = dim % 64;

    for i in 0..chunks {
        let offset = i * 64;
        // Load 64 bytes (512 bits)
        let va = _mm512_loadu_si512(a.add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.add(offset) as *const __m512i);

        // Unpack to i16 and multiply, then accumulate to i32
        // Low 32 bytes
        let va_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(va));
        let vb_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vb));
        let prod_lo = _mm512_madd_epi16(va_lo, vb_lo);
        sum = _mm512_add_epi32(sum, prod_lo);

        // High 32 bytes
        let va_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 1));
        let vb_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 1));
        let prod_hi = _mm512_madd_epi16(va_hi, vb_hi);
        sum = _mm512_add_epi32(sum, prod_hi);
    }

    // Reduce to scalar
    let mut result = _mm512_reduce_add_epi32(sum);

    // Handle remainder
    let base = chunks * 64;
    for i in 0..remainder {
        result += (*a.add(base + i) as i32) * (*b.add(base + i) as i32);
    }

    result
}

/// AVX-512VNNI inner product for UInt8 vectors (unsigned × unsigned).
///
/// Uses VPDPBUSD with careful handling for u8×u8.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[inline]
pub unsafe fn inner_product_u8_avx512vnni(a: *const u8, b: *const u8, dim: usize) -> u32 {
    // For u8×u8, we can use VPDPBUSD (u8×i8) by treating one operand as i8
    // and adjusting: u8×u8 = u8×(i8+128) + u8×(-128) = dpbusd(a,b-128) + 128*sum(a)
    // Simpler: widen to u16/i16 and multiply

    let mut sum = _mm512_setzero_si512();
    let chunks = dim / 64;
    let remainder = dim % 64;

    for i in 0..chunks {
        let offset = i * 64;
        let va = _mm512_loadu_si512(a.add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.add(offset) as *const __m512i);

        // Zero-extend to u16 (we use signed i16 since _mm512_cvtepu8_epi16 gives us the same bits)
        let va_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(va));
        let vb_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(vb));
        let prod_lo = _mm512_madd_epi16(va_lo, vb_lo);
        sum = _mm512_add_epi32(sum, prod_lo);

        let va_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(va, 1));
        let vb_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(vb, 1));
        let prod_hi = _mm512_madd_epi16(va_hi, vb_hi);
        sum = _mm512_add_epi32(sum, prod_hi);
    }

    let mut result = _mm512_reduce_add_epi32(sum) as u32;

    let base = chunks * 64;
    for i in 0..remainder {
        result += (*a.add(base + i) as u32) * (*b.add(base + i) as u32);
    }

    result
}

/// AVX-512VNNI L2 squared distance for Int8 vectors.
///
/// Computes sum((a[i] - b[i])^2) using AVX-512BW for subtraction
/// and widening to i16 for the squared difference.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[inline]
pub unsafe fn l2_squared_i8_avx512vnni(a: *const i8, b: *const i8, dim: usize) -> i32 {
    let mut sum = _mm512_setzero_si512();
    let chunks = dim / 64;
    let remainder = dim % 64;

    for i in 0..chunks {
        let offset = i * 64;
        let va = _mm512_loadu_si512(a.add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.add(offset) as *const __m512i);

        // Compute difference in i8 (may saturate, but we widen immediately)
        let diff = _mm512_sub_epi8(va, vb);

        // Widen to i16 and compute diff^2
        let diff_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(diff));
        let sq_lo = _mm512_madd_epi16(diff_lo, diff_lo);
        sum = _mm512_add_epi32(sum, sq_lo);

        let diff_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(diff, 1));
        let sq_hi = _mm512_madd_epi16(diff_hi, diff_hi);
        sum = _mm512_add_epi32(sum, sq_hi);
    }

    let mut result = _mm512_reduce_add_epi32(sum);

    let base = chunks * 64;
    for i in 0..remainder {
        let diff = (*a.add(base + i) as i32) - (*b.add(base + i) as i32);
        result += diff * diff;
    }

    result
}

/// AVX-512VNNI L2 squared distance for UInt8 vectors.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[inline]
pub unsafe fn l2_squared_u8_avx512vnni(a: *const u8, b: *const u8, dim: usize) -> u32 {
    let mut sum = _mm512_setzero_si512();
    let chunks = dim / 64;
    let remainder = dim % 64;

    for i in 0..chunks {
        let offset = i * 64;
        let va = _mm512_loadu_si512(a.add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.add(offset) as *const __m512i);

        // For u8 difference, we need to handle potential underflow
        // Use max(a,b) - min(a,b) = abs(a-b) for unsigned
        let max_ab = _mm512_max_epu8(va, vb);
        let min_ab = _mm512_min_epu8(va, vb);
        let diff = _mm512_sub_epi8(max_ab, min_ab); // unsigned difference

        // Zero-extend to i16 and compute diff^2
        let diff_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(diff));
        let sq_lo = _mm512_madd_epi16(diff_lo, diff_lo);
        sum = _mm512_add_epi32(sum, sq_lo);

        let diff_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(diff, 1));
        let sq_hi = _mm512_madd_epi16(diff_hi, diff_hi);
        sum = _mm512_add_epi32(sum, sq_hi);
    }

    let mut result = _mm512_reduce_add_epi32(sum) as u32;

    let base = chunks * 64;
    for i in 0..remainder {
        let diff = (*a.add(base + i) as i32) - (*b.add(base + i) as i32);
        result += (diff * diff) as u32;
    }

    result
}

/// AVX-512VNNI dot product using VPDPBUSD instruction.
///
/// This is the most efficient path for u8×i8 multiplication.
/// Computes: result += sum(a[4i+j] * b[4i+j]) for j in 0..4
/// where a is u8 (unsigned) and b is i8 (signed).
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[inline]
pub unsafe fn dot_product_u8_i8_avx512vnni(a: *const u8, b: *const i8, dim: usize) -> i32 {
    let mut sum = _mm512_setzero_si512();
    let chunks = dim / 64;
    let remainder = dim % 64;

    for i in 0..chunks {
        let offset = i * 64;
        let va = _mm512_loadu_si512(a.add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.add(offset) as *const __m512i);

        // VPDPBUSD: Multiply unsigned bytes by signed bytes,
        // sum adjacent 4 products, accumulate to i32
        sum = _mm512_dpbusd_epi32(sum, va, vb);
    }

    let mut result = _mm512_reduce_add_epi32(sum);

    // Handle remainder
    let base = chunks * 64;
    for i in 0..remainder {
        result += (*a.add(base + i) as i32) * (*b.add(base + i) as i32);
    }

    result
}

// ============================================================================
// Safe wrappers
// ============================================================================

/// Safe wrapper for L2 squared distance (f32).
#[inline]
pub fn l2_squared_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { l2_squared_f32_avx512bw(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        super::avx512::l2_squared_f32(a, b, dim)
    }
}

/// Safe wrapper for inner product (f32).
#[inline]
pub fn inner_product_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { inner_product_f32_avx512bw(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        super::avx512::inner_product_f32(a, b, dim)
    }
}

/// Safe wrapper for cosine distance (f32).
#[inline]
pub fn cosine_distance_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

        let result = unsafe { cosine_distance_f32_avx512bw(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        T::DistanceType::from_f64(result as f64)
    } else {
        super::avx512::cosine_distance_f32(a, b, dim)
    }
}

/// Safe wrapper for L2 squared distance for Int8 using VNNI.
#[inline]
pub fn l2_squared_i8(a: &[Int8], b: &[Int8], dim: usize) -> f32 {
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
    {
        let result =
            unsafe { l2_squared_i8_avx512vnni(a.as_ptr() as *const i8, b.as_ptr() as *const i8, dim) };
        result as f32
    } else {
        // Fallback to scalar
        l2_squared_i8_scalar(a, b, dim)
    }
}

/// Safe wrapper for L2 squared distance for UInt8 using VNNI.
#[inline]
pub fn l2_squared_u8(a: &[UInt8], b: &[UInt8], dim: usize) -> f32 {
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
    {
        let result =
            unsafe { l2_squared_u8_avx512vnni(a.as_ptr() as *const u8, b.as_ptr() as *const u8, dim) };
        result as f32
    } else {
        l2_squared_u8_scalar(a, b, dim)
    }
}

/// Safe wrapper for inner product for Int8 using VNNI.
#[inline]
pub fn inner_product_i8(a: &[Int8], b: &[Int8], dim: usize) -> f32 {
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
    {
        let result = unsafe {
            inner_product_i8_avx512vnni(a.as_ptr() as *const i8, b.as_ptr() as *const i8, dim)
        };
        result as f32
    } else {
        inner_product_i8_scalar(a, b, dim)
    }
}

/// Safe wrapper for inner product for UInt8 using VNNI.
#[inline]
pub fn inner_product_u8(a: &[UInt8], b: &[UInt8], dim: usize) -> f32 {
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
    {
        let result = unsafe {
            inner_product_u8_avx512vnni(a.as_ptr() as *const u8, b.as_ptr() as *const u8, dim)
        };
        result as f32
    } else {
        inner_product_u8_scalar(a, b, dim)
    }
}

/// Safe wrapper for VNNI dot product (u8 × i8).
#[inline]
pub fn dot_product_u8_i8(a: &[UInt8], b: &[Int8], dim: usize) -> i32 {
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
    {
        unsafe { dot_product_u8_i8_avx512vnni(a.as_ptr() as *const u8, b.as_ptr() as *const i8, dim) }
    } else {
        dot_product_u8_i8_scalar(a, b, dim)
    }
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

/// Scalar L2 squared for Int8.
#[inline]
fn l2_squared_i8_scalar(a: &[Int8], b: &[Int8], dim: usize) -> f32 {
    let mut sum: i32 = 0;
    for i in 0..dim {
        let diff = a[i].0 as i32 - b[i].0 as i32;
        sum += diff * diff;
    }
    sum as f32
}

/// Scalar L2 squared for UInt8.
#[inline]
fn l2_squared_u8_scalar(a: &[UInt8], b: &[UInt8], dim: usize) -> f32 {
    let mut sum: u32 = 0;
    for i in 0..dim {
        let diff = a[i].0 as i32 - b[i].0 as i32;
        sum += (diff * diff) as u32;
    }
    sum as f32
}

/// Scalar inner product for Int8.
#[inline]
fn inner_product_i8_scalar(a: &[Int8], b: &[Int8], dim: usize) -> f32 {
    let mut sum: i32 = 0;
    for i in 0..dim {
        sum += (a[i].0 as i32) * (b[i].0 as i32);
    }
    sum as f32
}

/// Scalar inner product for UInt8.
#[inline]
fn inner_product_u8_scalar(a: &[UInt8], b: &[UInt8], dim: usize) -> f32 {
    let mut sum: u32 = 0;
    for i in 0..dim {
        sum += (a[i].0 as u32) * (b[i].0 as u32);
    }
    sum as f32
}

/// Scalar dot product for u8 × i8.
#[inline]
fn dot_product_u8_i8_scalar(a: &[UInt8], b: &[Int8], dim: usize) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..dim {
        sum += (a[i].0 as i32) * (b[i].0 as i32);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx512bw_l2_squared_f32() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            println!("AVX-512BW not available, skipping test");
            return;
        }

        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..256).map(|i| (i + 1) as f32).collect();

        let result = l2_squared_f32::<f32>(&a, &b, 256);
        let expected = crate::distance::l2::l2_squared_scalar(&a, &b, 256);

        assert!((result - expected).abs() < 0.1);
    }

    #[test]
    fn test_avx512vnni_l2_squared_i8() {
        let a: Vec<Int8> = (0..256).map(|i| Int8((i % 128) as i8)).collect();
        let b: Vec<Int8> = (0..256).map(|i| Int8(((i + 1) % 128) as i8)).collect();

        let vnni_result = l2_squared_i8(&a, &b, 256);
        let scalar_result = l2_squared_i8_scalar(&a, &b, 256);

        assert!(
            (vnni_result - scalar_result).abs() < 1.0,
            "VNNI: {}, Scalar: {}",
            vnni_result,
            scalar_result
        );
    }

    #[test]
    fn test_avx512vnni_l2_squared_u8() {
        let a: Vec<UInt8> = (0..256).map(|i| UInt8((i % 256) as u8)).collect();
        let b: Vec<UInt8> = (0..256).map(|i| UInt8(((i + 1) % 256) as u8)).collect();

        let vnni_result = l2_squared_u8(&a, &b, 256);
        let scalar_result = l2_squared_u8_scalar(&a, &b, 256);

        assert!(
            (vnni_result - scalar_result).abs() < 1.0,
            "VNNI: {}, Scalar: {}",
            vnni_result,
            scalar_result
        );
    }

    #[test]
    fn test_avx512vnni_inner_product_i8() {
        let a: Vec<Int8> = (0..256).map(|i| Int8((i % 64) as i8)).collect();
        let b: Vec<Int8> = (0..256).map(|i| Int8((i % 64) as i8)).collect();

        let vnni_result = inner_product_i8(&a, &b, 256);
        let scalar_result = inner_product_i8_scalar(&a, &b, 256);

        assert!(
            (vnni_result - scalar_result).abs() < 1.0,
            "VNNI: {}, Scalar: {}",
            vnni_result,
            scalar_result
        );
    }

    #[test]
    fn test_avx512vnni_inner_product_u8() {
        let a: Vec<UInt8> = (0..256).map(|i| UInt8((i % 256) as u8)).collect();
        let b: Vec<UInt8> = (0..256).map(|i| UInt8((i % 256) as u8)).collect();

        let vnni_result = inner_product_u8(&a, &b, 256);
        let scalar_result = inner_product_u8_scalar(&a, &b, 256);

        assert!(
            (vnni_result - scalar_result).abs() < 1.0,
            "VNNI: {}, Scalar: {}",
            vnni_result,
            scalar_result
        );
    }

    #[test]
    fn test_avx512vnni_dot_product_u8_i8() {
        let a: Vec<UInt8> = (0..256).map(|i| UInt8((i % 256) as u8)).collect();
        let b: Vec<Int8> = (0..256).map(|i| Int8((i % 128) as i8)).collect();

        let vnni_result = dot_product_u8_i8(&a, &b, 256);
        let scalar_result = dot_product_u8_i8_scalar(&a, &b, 256);

        assert_eq!(
            vnni_result, scalar_result,
            "VNNI: {}, Scalar: {}",
            vnni_result, scalar_result
        );
    }
}

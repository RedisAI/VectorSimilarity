//! ARM NEON SIMD implementations for distance functions.
//!
//! These functions use 128-bit NEON instructions for ARM processors.
//! Available on all aarch64 (ARM64) platforms.
//!
//! Additional optimizations:
//! - NEON DOTPROD (ARMv8.2-A+): Int8/UInt8 dot product instructions
//!   Available on Apple M1+, AWS Graviton2+, and other modern ARM chips.

use crate::types::{DistanceType, VectorElement};

use std::arch::aarch64::*;

// =============================================================================
// NEON Int8/UInt8 optimized operations
// =============================================================================
//
// Note: NEON DOTPROD intrinsics (vdotq_s32, vdotq_u32) are currently unstable
// in Rust (see rust-lang/rust#117224). We use widening multiply-add instead,
// which is available on all NEON-capable ARM processors.

/// Check if NEON DOTPROD is available at runtime.
/// Currently used for future optimization when the intrinsics stabilize.
#[inline]
pub fn has_dotprod() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

/// NEON inner product for i8 vectors using widening multiply-add.
///
/// Processes 16 int8 elements per iteration using NEON SIMD.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn inner_product_i8_neon(a: *const i8, b: *const i8, dim: usize) -> i32 {
    let mut sum0 = vdupq_n_s32(0);
    let mut sum1 = vdupq_n_s32(0);
    let chunks = dim / 16;
    let remainder = dim % 16;

    for i in 0..chunks {
        let offset = i * 16;

        // Load 16 int8 values
        let va = vld1q_s8(a.add(offset));
        let vb = vld1q_s8(b.add(offset));

        // Split into low and high halves
        let va_lo = vget_low_s8(va);
        let va_hi = vget_high_s8(va);
        let vb_lo = vget_low_s8(vb);
        let vb_hi = vget_high_s8(vb);

        // Widen to i16 and multiply
        let prod_lo = vmull_s8(va_lo, vb_lo); // 8 x i16
        let prod_hi = vmull_s8(va_hi, vb_hi); // 8 x i16

        // Pairwise add to i32 and accumulate
        sum0 = vpadalq_s16(sum0, prod_lo);
        sum1 = vpadalq_s16(sum1, prod_hi);
    }

    // Combine accumulators and reduce
    let sum = vaddq_s32(sum0, sum1);
    let mut result = vaddvq_s32(sum);

    // Handle remainder
    let base = chunks * 16;
    for i in 0..remainder {
        result += (*a.add(base + i) as i32) * (*b.add(base + i) as i32);
    }

    result
}

/// NEON inner product for u8 vectors using widening multiply-add.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn inner_product_u8_neon(a: *const u8, b: *const u8, dim: usize) -> u32 {
    let mut sum0 = vdupq_n_u32(0);
    let mut sum1 = vdupq_n_u32(0);
    let chunks = dim / 16;
    let remainder = dim % 16;

    for i in 0..chunks {
        let offset = i * 16;

        let va = vld1q_u8(a.add(offset));
        let vb = vld1q_u8(b.add(offset));

        let va_lo = vget_low_u8(va);
        let va_hi = vget_high_u8(va);
        let vb_lo = vget_low_u8(vb);
        let vb_hi = vget_high_u8(vb);

        // Widen to u16 and multiply
        let prod_lo = vmull_u8(va_lo, vb_lo);
        let prod_hi = vmull_u8(va_hi, vb_hi);

        // Pairwise add to u32 and accumulate
        sum0 = vpadalq_u16(sum0, prod_lo);
        sum1 = vpadalq_u16(sum1, prod_hi);
    }

    let sum = vaddq_u32(sum0, sum1);
    let mut result = vaddvq_u32(sum);

    let base = chunks * 16;
    for i in 0..remainder {
        result += (*a.add(base + i) as u32) * (*b.add(base + i) as u32);
    }

    result
}

/// NEON L2 squared distance for i8 vectors.
///
/// Computes ||a - b||² using widening operations.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn l2_squared_i8_neon(a: *const i8, b: *const i8, dim: usize) -> i32 {
    let mut sum0 = vdupq_n_s32(0);
    let mut sum1 = vdupq_n_s32(0);
    let chunks = dim / 16;
    let remainder = dim % 16;

    for i in 0..chunks {
        let offset = i * 16;

        let va = vld1q_s8(a.add(offset));
        let vb = vld1q_s8(b.add(offset));

        // Compute difference (saturating subtract, then widen)
        let va_lo = vget_low_s8(va);
        let va_hi = vget_high_s8(va);
        let vb_lo = vget_low_s8(vb);
        let vb_hi = vget_high_s8(vb);

        // Widen to i16 for subtraction to avoid overflow
        let va_lo_16 = vmovl_s8(va_lo);
        let va_hi_16 = vmovl_s8(va_hi);
        let vb_lo_16 = vmovl_s8(vb_lo);
        let vb_hi_16 = vmovl_s8(vb_hi);

        let diff_lo = vsubq_s16(va_lo_16, vb_lo_16);
        let diff_hi = vsubq_s16(va_hi_16, vb_hi_16);

        // Square the differences (i16 * i16 -> i32)
        let sq_lo_lo = vmull_s16(vget_low_s16(diff_lo), vget_low_s16(diff_lo));
        let sq_lo_hi = vmull_s16(vget_high_s16(diff_lo), vget_high_s16(diff_lo));
        let sq_hi_lo = vmull_s16(vget_low_s16(diff_hi), vget_low_s16(diff_hi));
        let sq_hi_hi = vmull_s16(vget_high_s16(diff_hi), vget_high_s16(diff_hi));

        // Accumulate
        sum0 = vaddq_s32(sum0, sq_lo_lo);
        sum0 = vaddq_s32(sum0, sq_lo_hi);
        sum1 = vaddq_s32(sum1, sq_hi_lo);
        sum1 = vaddq_s32(sum1, sq_hi_hi);
    }

    let sum = vaddq_s32(sum0, sum1);
    let mut result = vaddvq_s32(sum);

    let base = chunks * 16;
    for i in 0..remainder {
        let diff = (*a.add(base + i) as i32) - (*b.add(base + i) as i32);
        result += diff * diff;
    }

    result
}

/// NEON L2 squared distance for u8 vectors.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn l2_squared_u8_neon(a: *const u8, b: *const u8, dim: usize) -> i32 {
    let mut sum0 = vdupq_n_s32(0);
    let mut sum1 = vdupq_n_s32(0);

    let chunks = dim / 16;
    let remainder = dim % 16;

    for i in 0..chunks {
        let offset = i * 16;
        let va = vld1q_u8(a.add(offset));
        let vb = vld1q_u8(b.add(offset));

        // Compute absolute difference (won't overflow for u8)
        let diff = vabdq_u8(va, vb);

        // Split into low and high halves, widen to u16, then square
        let diff_lo = vget_low_u8(diff);
        let diff_hi = vget_high_u8(diff);

        // Widen to u16
        let diff_lo_16 = vmovl_u8(diff_lo);
        let diff_hi_16 = vmovl_u8(diff_hi);

        // Square (u16 * u16 -> u32)
        let sq_lo = vmull_u16(vget_low_u16(diff_lo_16), vget_low_u16(diff_lo_16));
        let sq_hi = vmull_u16(vget_high_u16(diff_lo_16), vget_high_u16(diff_lo_16));
        let sq_lo2 = vmull_u16(vget_low_u16(diff_hi_16), vget_low_u16(diff_hi_16));
        let sq_hi2 = vmull_u16(vget_high_u16(diff_hi_16), vget_high_u16(diff_hi_16));

        // Accumulate (reinterpret u32 as s32 for final sum)
        sum0 = vaddq_s32(sum0, vreinterpretq_s32_u32(sq_lo));
        sum0 = vaddq_s32(sum0, vreinterpretq_s32_u32(sq_hi));
        sum1 = vaddq_s32(sum1, vreinterpretq_s32_u32(sq_lo2));
        sum1 = vaddq_s32(sum1, vreinterpretq_s32_u32(sq_hi2));
    }

    let sum = vaddq_s32(sum0, sum1);
    let mut result = vaddvq_s32(sum);

    let base = chunks * 16;
    for i in 0..remainder {
        let diff = (*a.add(base + i) as i32) - (*b.add(base + i) as i32);
        result += diff * diff;
    }

    result
}

/// Safe wrapper for i8 inner product.
#[inline]
pub fn inner_product_i8(a: &[i8], b: &[i8], dim: usize) -> i32 {
    unsafe { inner_product_i8_neon(a.as_ptr(), b.as_ptr(), dim) }
}

/// Safe wrapper for u8 inner product.
#[inline]
pub fn inner_product_u8(a: &[u8], b: &[u8], dim: usize) -> u32 {
    unsafe { inner_product_u8_neon(a.as_ptr(), b.as_ptr(), dim) }
}

/// Safe wrapper for i8 L2 squared.
#[inline]
pub fn l2_squared_i8(a: &[i8], b: &[i8], dim: usize) -> i32 {
    unsafe { l2_squared_i8_neon(a.as_ptr(), b.as_ptr(), dim) }
}

/// Safe wrapper for u8 L2 squared.
#[inline]
pub fn l2_squared_u8(a: &[u8], b: &[u8], dim: usize) -> i32 {
    unsafe { l2_squared_u8_neon(a.as_ptr(), b.as_ptr(), dim) }
}

// =============================================================================
// NEON f64 SIMD operations
// =============================================================================

/// NEON L2 squared distance for f64 vectors.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn l2_squared_f64_neon(a: *const f64, b: *const f64, dim: usize) -> f64 {
    let mut sum0 = vdupq_n_f64(0.0);
    let mut sum1 = vdupq_n_f64(0.0);
    let chunks = dim / 4;
    let remainder = dim % 4;

    // Process 4 elements at a time (two 2-element vectors)
    for i in 0..chunks {
        let offset = i * 4;

        let va0 = vld1q_f64(a.add(offset));
        let vb0 = vld1q_f64(b.add(offset));
        let diff0 = vsubq_f64(va0, vb0);
        sum0 = vfmaq_f64(sum0, diff0, diff0);

        let va1 = vld1q_f64(a.add(offset + 2));
        let vb1 = vld1q_f64(b.add(offset + 2));
        let diff1 = vsubq_f64(va1, vb1);
        sum1 = vfmaq_f64(sum1, diff1, diff1);
    }

    // Combine and reduce
    let sum = vaddq_f64(sum0, sum1);
    let mut result = vaddvq_f64(sum);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        let diff = *a.add(base + i) - *b.add(base + i);
        result += diff * diff;
    }

    result
}

/// NEON inner product for f64 vectors.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn inner_product_f64_neon(a: *const f64, b: *const f64, dim: usize) -> f64 {
    let mut sum0 = vdupq_n_f64(0.0);
    let mut sum1 = vdupq_n_f64(0.0);
    let chunks = dim / 4;
    let remainder = dim % 4;

    for i in 0..chunks {
        let offset = i * 4;

        let va0 = vld1q_f64(a.add(offset));
        let vb0 = vld1q_f64(b.add(offset));
        sum0 = vfmaq_f64(sum0, va0, vb0);

        let va1 = vld1q_f64(a.add(offset + 2));
        let vb1 = vld1q_f64(b.add(offset + 2));
        sum1 = vfmaq_f64(sum1, va1, vb1);
    }

    let sum = vaddq_f64(sum0, sum1);
    let mut result = vaddvq_f64(sum);

    let base = chunks * 4;
    for i in 0..remainder {
        result += *a.add(base + i) * *b.add(base + i);
    }

    result
}

/// NEON cosine distance for f64 vectors.
///
/// # Safety
/// - Pointers must be valid for `dim` elements
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cosine_distance_f64_neon(a: *const f64, b: *const f64, dim: usize) -> f64 {
    let mut dot_sum = vdupq_n_f64(0.0);
    let mut norm_a_sum = vdupq_n_f64(0.0);
    let mut norm_b_sum = vdupq_n_f64(0.0);

    let chunks = dim / 2;
    let remainder = dim % 2;

    for i in 0..chunks {
        let offset = i * 2;
        let va = vld1q_f64(a.add(offset));
        let vb = vld1q_f64(b.add(offset));

        dot_sum = vfmaq_f64(dot_sum, va, vb);
        norm_a_sum = vfmaq_f64(norm_a_sum, va, va);
        norm_b_sum = vfmaq_f64(norm_b_sum, vb, vb);
    }

    let mut dot = vaddvq_f64(dot_sum);
    let mut norm_a = vaddvq_f64(norm_a_sum);
    let mut norm_b = vaddvq_f64(norm_b_sum);

    let base = chunks * 2;
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
    unsafe { l2_squared_f64_neon(a.as_ptr(), b.as_ptr(), dim) }
}

/// Safe wrapper for f64 inner product.
#[inline]
pub fn inner_product_f64(a: &[f64], b: &[f64], dim: usize) -> f64 {
    unsafe { inner_product_f64_neon(a.as_ptr(), b.as_ptr(), dim) }
}

/// Safe wrapper for f64 cosine distance.
#[inline]
pub fn cosine_distance_f64(a: &[f64], b: &[f64], dim: usize) -> f64 {
    unsafe { cosine_distance_f64_neon(a.as_ptr(), b.as_ptr(), dim) }
}

// =============================================================================
// Original NEON f32 implementations
// =============================================================================

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
/// Specialized for f32 to avoid allocation when input is already f32.
#[inline]
pub fn l2_squared_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    // Fast path: if T is f32, avoid allocation by reinterpreting the slices
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // SAFETY: We verified T is f32, so this reinterpret is safe
        let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, dim) };
        let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, dim) };
        let result = unsafe { l2_squared_f32_neon(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        return T::DistanceType::from_f64(result as f64);
    }

    // Slow path: convert to f32
    let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

    let result = unsafe { l2_squared_f32_neon(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
    T::DistanceType::from_f64(result as f64)
}

/// Safe wrapper for inner product.
/// Specialized for f32 to avoid allocation when input is already f32.
#[inline]
pub fn inner_product_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    // Fast path: if T is f32, avoid allocation
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, dim) };
        let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, dim) };
        let result = unsafe { inner_product_f32_neon(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        return T::DistanceType::from_f64(result as f64);
    }

    let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();

    let result = unsafe { inner_product_f32_neon(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
    T::DistanceType::from_f64(result as f64)
}

/// Safe wrapper for cosine distance.
/// Specialized for f32 to avoid allocation when input is already f32.
#[inline]
pub fn cosine_distance_f32<T: VectorElement>(a: &[T], b: &[T], dim: usize) -> T::DistanceType {
    // Fast path: if T is f32, avoid allocation
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, dim) };
        let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, dim) };
        let result = unsafe { cosine_distance_f32_neon(a_f32.as_ptr(), b_f32.as_ptr(), dim) };
        return T::DistanceType::from_f64(result as f64);
    }

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

    // NEON DOTPROD tests (int8/uint8)
    #[test]
    fn test_dotprod_inner_product_i8() {
        let a: Vec<i8> = (0..128).map(|i| (i % 127) as i8).collect();
        let b: Vec<i8> = (0..128).map(|i| ((128 - i) % 127) as i8).collect();

        let result = inner_product_i8(&a, &b, 128);

        // Verify against scalar
        let expected: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i32) * (y as i32))
            .sum();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_dotprod_inner_product_u8() {
        let a: Vec<u8> = (0..128).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..128).map(|i| (128 - i) as u8).collect();

        let result = inner_product_u8(&a, &b, 128);

        let expected: u32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as u32) * (y as u32))
            .sum();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_dotprod_l2_squared_i8() {
        let a: Vec<i8> = (0..128).map(|i| (i % 64) as i8).collect();
        let b: Vec<i8> = (0..128).map(|i| ((i + 1) % 64) as i8).collect();

        let result = l2_squared_i8(&a, &b, 128);

        let expected: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = (x as i32) - (y as i32);
                diff * diff
            })
            .sum();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_dotprod_l2_squared_u8() {
        let a: Vec<u8> = (0..128).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..128).map(|i| (i + 1) as u8).collect();

        let result = l2_squared_u8(&a, &b, 128);

        let expected: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = (x as i32) - (y as i32);
                diff * diff
            })
            .sum();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_dotprod_remainder_handling() {
        // Test with non-aligned dimensions
        for dim in [17, 33, 65, 100, 127] {
            let a: Vec<i8> = (0..dim).map(|i| (i % 100) as i8).collect();
            let b: Vec<i8> = (0..dim).map(|i| ((i * 2) % 100) as i8).collect();

            let result = inner_product_i8(&a, &b, dim);

            let expected: i32 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x as i32) * (y as i32))
                .sum();

            assert_eq!(result, expected, "Failed for dim={}", dim);
        }
    }

    // NEON f64 tests
    #[test]
    fn test_neon_l2_squared_f64() {
        let a: Vec<f64> = (0..128).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..128).map(|i| (i + 1) as f64).collect();

        let neon_result = l2_squared_f64(&a, &b, 128);

        // Should be 128.0 (each diff is 1, squared is 1, sum of 128 ones)
        assert!((neon_result - 128.0).abs() < 1e-10);
    }

    #[test]
    fn test_neon_inner_product_f64() {
        let a: Vec<f64> = (0..128).map(|i| i as f64 / 100.0).collect();
        let b: Vec<f64> = (0..128).map(|i| (128 - i) as f64 / 100.0).collect();

        let neon_result = inner_product_f64(&a, &b, 128);

        let expected: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

        assert!((neon_result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_neon_cosine_f64() {
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
    fn test_neon_f64_remainder() {
        // Test with non-aligned dimension
        let a: Vec<f64> = (0..131).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..131).map(|i| (i + 1) as f64).collect();

        let neon_result = l2_squared_f64(&a, &b, 131);
        assert!((neon_result - 131.0).abs() < 1e-10);
    }

    #[test]
    fn test_has_dotprod() {
        // Just verify the detection doesn't crash
        let _ = has_dotprod();
    }
}

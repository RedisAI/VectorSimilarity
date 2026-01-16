//! ARM SVE (Scalable Vector Extension) optimized distance functions.
//!
//! ARM SVE provides variable-length vector operations (128-2048 bits) that
//! automatically scale to the hardware's vector length. This makes SVE code
//! portable across different ARM implementations.
//!
//! Note: Full SVE intrinsics are currently unstable in Rust (nightly-only).
//! This module provides:
//! - Runtime detection of SVE support
//! - Optimized implementations using available NEON with larger unrolling
//! - Preparedness for native SVE when it stabilizes
//!
//! For maximum performance on SVE-capable hardware, compile with:
//! `RUSTFLAGS="-C target-feature=+sve" cargo build --release`
//!
//! SVE-capable processors include:
//! - AWS Graviton3 (256-bit vectors)
//! - Fujitsu A64FX (512-bit vectors)
//! - ARM Neoverse V1/V2

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Check if SVE is available at runtime.
///
/// Note: This requires the `std_detect_dlsym_getauxval` feature to be stable
/// for proper runtime detection. Currently returns false on stable Rust.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn has_sve() -> bool {
    // SVE detection is not yet stable in Rust
    // When stable, use: std::arch::is_aarch64_feature_detected!("sve")
    // For now, we can't reliably detect SVE at runtime on stable Rust
    #[cfg(target_feature = "sve")]
    {
        true
    }
    #[cfg(not(target_feature = "sve"))]
    {
        false
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn has_sve() -> bool {
    false
}

/// Get the SVE vector length in bytes.
/// Returns 0 if SVE is not available.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn get_sve_vector_length() -> usize {
    if has_sve() {
        // When SVE is enabled at compile time, we can assume at least 128 bits
        // The actual length depends on the hardware (128, 256, 512, 1024, 2048 bits)
        #[cfg(target_feature = "sve")]
        {
            // In real SVE code, you would use svcntb() to get the vector length
            // For now, assume minimum SVE length of 128 bits = 16 bytes
            16
        }
        #[cfg(not(target_feature = "sve"))]
        {
            0
        }
    } else {
        0
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn get_sve_vector_length() -> usize {
    0
}

// When SVE intrinsics become stable, we would have implementations like:
//
// #[cfg(target_arch = "aarch64")]
// #[target_feature(enable = "sve")]
// pub unsafe fn l2_squared_f32_sve(a: *const f32, b: *const f32, dim: usize) -> f32 {
//     use std::arch::aarch64::*;
//
//     let vl = svcntw(); // Get vector length in words (f32s)
//     let mut sum = svdup_n_f32(0.0);
//     let mut offset = 0;
//
//     // SVE has predicated operations for automatic remainder handling
//     while offset < dim {
//         let pred = svwhilelt_b32(offset as u64, dim as u64);
//         let va = svld1_f32(pred, a.add(offset));
//         let vb = svld1_f32(pred, b.add(offset));
//         let diff = svsub_f32_x(pred, va, vb);
//         sum = svmla_f32_x(pred, sum, diff, diff);
//         offset += vl;
//     }
//
//     svaddv_f32(svptrue_b32(), sum)
// }

// For now, provide optimized NEON with larger unrolling as a substitute
// These can be used on SVE-capable hardware when targeting NEON compatibility

/// L2 squared distance with aggressive unrolling for SVE-class hardware.
/// Uses NEON but with 8x unrolling to better utilize wide pipelines.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn l2_squared_f32_wide(a: *const f32, b: *const f32, dim: usize) -> f32 {
    // 8x unrolling for wide execution units (32 f32s per iteration)
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let mut sum4 = vdupq_n_f32(0.0);
    let mut sum5 = vdupq_n_f32(0.0);
    let mut sum6 = vdupq_n_f32(0.0);
    let mut sum7 = vdupq_n_f32(0.0);

    let unroll = dim / 32 * 32;
    let mut offset = 0;

    while offset < unroll {
        let a0 = vld1q_f32(a.add(offset));
        let a1 = vld1q_f32(a.add(offset + 4));
        let a2 = vld1q_f32(a.add(offset + 8));
        let a3 = vld1q_f32(a.add(offset + 12));
        let a4 = vld1q_f32(a.add(offset + 16));
        let a5 = vld1q_f32(a.add(offset + 20));
        let a6 = vld1q_f32(a.add(offset + 24));
        let a7 = vld1q_f32(a.add(offset + 28));

        let b0 = vld1q_f32(b.add(offset));
        let b1 = vld1q_f32(b.add(offset + 4));
        let b2 = vld1q_f32(b.add(offset + 8));
        let b3 = vld1q_f32(b.add(offset + 12));
        let b4 = vld1q_f32(b.add(offset + 16));
        let b5 = vld1q_f32(b.add(offset + 20));
        let b6 = vld1q_f32(b.add(offset + 24));
        let b7 = vld1q_f32(b.add(offset + 28));

        let d0 = vsubq_f32(a0, b0);
        let d1 = vsubq_f32(a1, b1);
        let d2 = vsubq_f32(a2, b2);
        let d3 = vsubq_f32(a3, b3);
        let d4 = vsubq_f32(a4, b4);
        let d5 = vsubq_f32(a5, b5);
        let d6 = vsubq_f32(a6, b6);
        let d7 = vsubq_f32(a7, b7);

        sum0 = vfmaq_f32(sum0, d0, d0);
        sum1 = vfmaq_f32(sum1, d1, d1);
        sum2 = vfmaq_f32(sum2, d2, d2);
        sum3 = vfmaq_f32(sum3, d3, d3);
        sum4 = vfmaq_f32(sum4, d4, d4);
        sum5 = vfmaq_f32(sum5, d5, d5);
        sum6 = vfmaq_f32(sum6, d6, d6);
        sum7 = vfmaq_f32(sum7, d7, d7);

        offset += 32;
    }

    // Reduce partial sums
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum4 = vaddq_f32(sum4, sum5);
    sum6 = vaddq_f32(sum6, sum7);
    sum0 = vaddq_f32(sum0, sum2);
    sum4 = vaddq_f32(sum4, sum6);
    sum0 = vaddq_f32(sum0, sum4);

    // Handle remaining 4-element chunks
    while offset + 4 <= dim {
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        let diff = vsubq_f32(va, vb);
        sum0 = vfmaq_f32(sum0, diff, diff);
        offset += 4;
    }

    // Horizontal sum
    let mut result = vaddvq_f32(sum0);

    // Handle remaining elements
    while offset < dim {
        let diff = *a.add(offset) - *b.add(offset);
        result += diff * diff;
        offset += 1;
    }

    result
}

/// Inner product with aggressive unrolling for SVE-class hardware.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn inner_product_f32_wide(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let mut sum4 = vdupq_n_f32(0.0);
    let mut sum5 = vdupq_n_f32(0.0);
    let mut sum6 = vdupq_n_f32(0.0);
    let mut sum7 = vdupq_n_f32(0.0);

    let unroll = dim / 32 * 32;
    let mut offset = 0;

    while offset < unroll {
        let a0 = vld1q_f32(a.add(offset));
        let a1 = vld1q_f32(a.add(offset + 4));
        let a2 = vld1q_f32(a.add(offset + 8));
        let a3 = vld1q_f32(a.add(offset + 12));
        let a4 = vld1q_f32(a.add(offset + 16));
        let a5 = vld1q_f32(a.add(offset + 20));
        let a6 = vld1q_f32(a.add(offset + 24));
        let a7 = vld1q_f32(a.add(offset + 28));

        let b0 = vld1q_f32(b.add(offset));
        let b1 = vld1q_f32(b.add(offset + 4));
        let b2 = vld1q_f32(b.add(offset + 8));
        let b3 = vld1q_f32(b.add(offset + 12));
        let b4 = vld1q_f32(b.add(offset + 16));
        let b5 = vld1q_f32(b.add(offset + 20));
        let b6 = vld1q_f32(b.add(offset + 24));
        let b7 = vld1q_f32(b.add(offset + 28));

        sum0 = vfmaq_f32(sum0, a0, b0);
        sum1 = vfmaq_f32(sum1, a1, b1);
        sum2 = vfmaq_f32(sum2, a2, b2);
        sum3 = vfmaq_f32(sum3, a3, b3);
        sum4 = vfmaq_f32(sum4, a4, b4);
        sum5 = vfmaq_f32(sum5, a5, b5);
        sum6 = vfmaq_f32(sum6, a6, b6);
        sum7 = vfmaq_f32(sum7, a7, b7);

        offset += 32;
    }

    // Reduce
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum4 = vaddq_f32(sum4, sum5);
    sum6 = vaddq_f32(sum6, sum7);
    sum0 = vaddq_f32(sum0, sum2);
    sum4 = vaddq_f32(sum4, sum6);
    sum0 = vaddq_f32(sum0, sum4);

    while offset + 4 <= dim {
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        sum0 = vfmaq_f32(sum0, va, vb);
        offset += 4;
    }

    let mut result = vaddvq_f32(sum0);

    while offset < dim {
        result += *a.add(offset) * *b.add(offset);
        offset += 1;
    }

    result
}

/// Cosine distance with aggressive unrolling for SVE-class hardware.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cosine_distance_f32_wide(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut norm_a0 = vdupq_n_f32(0.0);
    let mut norm_a1 = vdupq_n_f32(0.0);
    let mut norm_b0 = vdupq_n_f32(0.0);
    let mut norm_b1 = vdupq_n_f32(0.0);

    let unroll = dim / 8 * 8;
    let mut offset = 0;

    while offset < unroll {
        let a0 = vld1q_f32(a.add(offset));
        let a1 = vld1q_f32(a.add(offset + 4));
        let b0 = vld1q_f32(b.add(offset));
        let b1 = vld1q_f32(b.add(offset + 4));

        dot0 = vfmaq_f32(dot0, a0, b0);
        dot1 = vfmaq_f32(dot1, a1, b1);
        norm_a0 = vfmaq_f32(norm_a0, a0, a0);
        norm_a1 = vfmaq_f32(norm_a1, a1, a1);
        norm_b0 = vfmaq_f32(norm_b0, b0, b0);
        norm_b1 = vfmaq_f32(norm_b1, b1, b1);

        offset += 8;
    }

    let dot_sum = vaddq_f32(dot0, dot1);
    let norm_a_sum = vaddq_f32(norm_a0, norm_a1);
    let norm_b_sum = vaddq_f32(norm_b0, norm_b1);

    let mut dot = vaddvq_f32(dot_sum);
    let mut norm_a = vaddvq_f32(norm_a_sum);
    let mut norm_b = vaddvq_f32(norm_b_sum);

    while offset < dim {
        let av = *a.add(offset);
        let bv = *b.add(offset);
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

// Safe wrappers

/// Safe L2 squared distance with wide unrolling.
#[cfg(target_arch = "aarch64")]
pub fn l2_squared_f32(a: &[f32], b: &[f32], dim: usize) -> f32 {
    debug_assert_eq!(a.len(), dim);
    debug_assert_eq!(b.len(), dim);
    unsafe { l2_squared_f32_wide(a.as_ptr(), b.as_ptr(), dim) }
}

/// Safe inner product with wide unrolling.
#[cfg(target_arch = "aarch64")]
pub fn inner_product_f32(a: &[f32], b: &[f32], dim: usize) -> f32 {
    debug_assert_eq!(a.len(), dim);
    debug_assert_eq!(b.len(), dim);
    unsafe { inner_product_f32_wide(a.as_ptr(), b.as_ptr(), dim) }
}

/// Safe cosine distance with wide unrolling.
#[cfg(target_arch = "aarch64")]
pub fn cosine_distance_f32(a: &[f32], b: &[f32], dim: usize) -> f32 {
    debug_assert_eq!(a.len(), dim);
    debug_assert_eq!(b.len(), dim);
    unsafe { cosine_distance_f32_wide(a.as_ptr(), b.as_ptr(), dim) }
}

// Stubs for non-aarch64
#[cfg(not(target_arch = "aarch64"))]
pub fn l2_squared_f32(_a: &[f32], _b: &[f32], _dim: usize) -> f32 {
    unimplemented!("SVE/wide NEON only available on aarch64")
}

#[cfg(not(target_arch = "aarch64"))]
pub fn inner_product_f32(_a: &[f32], _b: &[f32], _dim: usize) -> f32 {
    unimplemented!("SVE/wide NEON only available on aarch64")
}

#[cfg(not(target_arch = "aarch64"))]
pub fn cosine_distance_f32(_a: &[f32], _b: &[f32], _dim: usize) -> f32 {
    unimplemented!("SVE/wide NEON only available on aarch64")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_sve() {
        let has = has_sve();
        println!("SVE available: {}", has);
        // Just check it doesn't crash
    }

    #[test]
    fn test_sve_vector_length() {
        let len = get_sve_vector_length();
        println!("SVE vector length: {} bytes", len);
        // Either 0 (not available) or a power of 2 >= 16
        assert!(len == 0 || (len >= 16 && len.is_power_of_two()));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_wide_l2_squared() {
        for &dim in &[32, 64, 100, 128, 256, 512] {
            let a: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
            let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 / dim as f32).collect();

            let wide_result = l2_squared_f32(&a, &b, dim);

            // Compute scalar reference
            let scalar_result: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum();

            assert!(
                (wide_result - scalar_result).abs() < 0.001,
                "dim={}: wide={}, scalar={}",
                dim,
                wide_result,
                scalar_result
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_wide_inner_product() {
        for &dim in &[32, 64, 100, 128, 256, 512] {
            let a: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
            let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 / dim as f32).collect();

            let wide_result = inner_product_f32(&a, &b, dim);
            let scalar_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

            assert!(
                (wide_result - scalar_result).abs() < 0.001,
                "dim={}: wide={}, scalar={}",
                dim,
                wide_result,
                scalar_result
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_wide_cosine_distance() {
        let dim = 128;
        let a: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();

        // Self-distance should be ~0
        let self_dist = cosine_distance_f32(&a, &a, dim);
        assert!(
            self_dist.abs() < 0.001,
            "Self cosine distance should be ~0, got {}",
            self_dist
        );
    }
}

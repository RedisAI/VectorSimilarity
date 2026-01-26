//! SIMD-optimized distance function implementations.
//!
//! This module provides hardware-accelerated distance computations:
//! - AVX-512 VNNI (x86_64) - 512-bit vectors with VNNI for int8 operations
//! - AVX-512 BW (x86_64) - 512-bit vectors with byte/word operations
//! - AVX-512 (x86_64) - 512-bit vectors, 16 f32 at a time
//! - AVX-512 FP16 (x86_64) - 512-bit vectors for half-precision (Float16)
//! - AVX-512 BF16 (x86_64) - 512-bit vectors for bfloat16 (BFloat16)
//! - AVX2 (x86_64) - 256-bit vectors, 8 f32 at a time, with FMA
//! - AVX (x86_64) - 256-bit vectors, 8 f32 at a time, no FMA
//! - SSE (x86_64) - 128-bit vectors, 4 f32 at a time
//! - NEON (aarch64) - 128-bit vectors, 4 f32 at a time
//!
//! Runtime feature detection is used to select the best implementation.

#[cfg(target_arch = "x86_64")]
pub mod avx;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "x86_64")]
pub mod avx512bf16;
#[cfg(target_arch = "x86_64")]
pub mod avx512bw;
#[cfg(target_arch = "x86_64")]
pub mod avx512fp16;
#[cfg(target_arch = "x86_64")]
pub mod sse;
#[cfg(target_arch = "x86_64")]
pub mod sse4;
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "aarch64")]
pub mod sve;

#[cfg(test)]
mod cross_consistency_tests;

/// SIMD capability levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdCapability {
    /// No SIMD support.
    None,
    /// SSE (128-bit vectors).
    #[cfg(target_arch = "x86_64")]
    Sse,
    /// SSE4.1 (128-bit vectors with dot product instruction).
    #[cfg(target_arch = "x86_64")]
    Sse4_1,
    /// AVX (256-bit vectors, no FMA).
    #[cfg(target_arch = "x86_64")]
    Avx,
    /// AVX2 (256-bit vectors, with FMA).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512 (512-bit vectors).
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// AVX-512 with BW (byte/word operations).
    #[cfg(target_arch = "x86_64")]
    Avx512Bw,
    /// AVX-512 with VNNI (int8 neural network instructions).
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
    /// ARM NEON (128-bit vectors).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Check if any SIMD capability is available.
pub fn is_simd_available() -> bool {
    detect_simd_capability() != SimdCapability::None
}

/// Detect the best available SIMD capability at runtime.
pub fn detect_simd_capability() -> SimdCapability {
    #[cfg(target_arch = "x86_64")]
    {
        // Check for AVX-512 VNNI first (best for int8 operations)
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
        {
            return SimdCapability::Avx512Vnni;
        }
        // Check for AVX-512 BW (byte/word operations)
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return SimdCapability::Avx512Bw;
        }
        // Check for basic AVX-512
        if is_x86_feature_detected!("avx512f") {
            return SimdCapability::Avx512;
        }
        // Fall back to AVX2 (with FMA)
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdCapability::Avx2;
        }
        // Fall back to AVX (without FMA)
        if is_x86_feature_detected!("avx") {
            return SimdCapability::Avx;
        }
        // Fall back to SSE4.1 (with dot product instruction)
        if is_x86_feature_detected!("sse4.1") {
            return SimdCapability::Sse4_1;
        }
        // Fall back to SSE (available on virtually all x86_64)
        if is_x86_feature_detected!("sse") {
            return SimdCapability::Sse;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return SimdCapability::Neon;
    }

    #[allow(unreachable_code)]
    SimdCapability::None
}

/// Check if AVX-512 VNNI is available at runtime.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx512_vnni() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
}

/// Check if AVX-512 VNNI is available at runtime (stub for non-x86_64).
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn has_avx512_vnni() -> bool {
    false
}

/// Get the optimal vector alignment for the detected SIMD capability.
pub fn optimal_alignment() -> usize {
    match detect_simd_capability() {
        #[cfg(target_arch = "x86_64")]
        SimdCapability::Avx512Vnni => 64,
        #[cfg(target_arch = "x86_64")]
        SimdCapability::Avx512Bw => 64,
        #[cfg(target_arch = "x86_64")]
        SimdCapability::Avx512 => 64,
        #[cfg(target_arch = "x86_64")]
        SimdCapability::Avx2 => 32,
        #[cfg(target_arch = "x86_64")]
        SimdCapability::Avx => 32,
        #[cfg(target_arch = "x86_64")]
        SimdCapability::Sse4_1 => 16,
        #[cfg(target_arch = "x86_64")]
        SimdCapability::Sse => 16,
        #[cfg(target_arch = "aarch64")]
        SimdCapability::Neon => 16,
        SimdCapability::None => 8,
        #[allow(unreachable_patterns)]
        _ => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_simd() {
        let cap = detect_simd_capability();
        println!("Detected SIMD capability: {:?}", cap);
        // Just ensure it doesn't crash
    }

    #[test]
    fn test_optimal_alignment() {
        let align = optimal_alignment();
        assert!(align >= 8);
        assert!(align.is_power_of_two());
    }
}

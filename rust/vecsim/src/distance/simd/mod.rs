//! SIMD-optimized distance function implementations.
//!
//! This module provides hardware-accelerated distance computations:
//! - AVX-512 (x86_64) - 512-bit vectors, 16 f32 at a time
//! - AVX2 (x86_64) - 256-bit vectors, 8 f32 at a time
//! - SSE (x86_64) - 128-bit vectors, 4 f32 at a time
//! - NEON (aarch64) - 128-bit vectors, 4 f32 at a time
//!
//! Runtime feature detection is used to select the best implementation.

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "x86_64")]
pub mod sse;
#[cfg(target_arch = "aarch64")]
pub mod neon;

/// SIMD capability levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdCapability {
    /// No SIMD support.
    None,
    /// SSE (128-bit vectors).
    #[cfg(target_arch = "x86_64")]
    Sse,
    /// AVX2 (256-bit vectors).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512 (512-bit vectors).
    #[cfg(target_arch = "x86_64")]
    Avx512,
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
        // Check for AVX-512 first (best performance)
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return SimdCapability::Avx512;
        }
        // Fall back to AVX2
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdCapability::Avx2;
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

/// Get the optimal vector alignment for the detected SIMD capability.
pub fn optimal_alignment() -> usize {
    match detect_simd_capability() {
        #[cfg(target_arch = "x86_64")]
        SimdCapability::Avx512 => 64,
        #[cfg(target_arch = "x86_64")]
        SimdCapability::Avx2 => 32,
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

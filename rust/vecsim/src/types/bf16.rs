//! Brain floating point (BF16/BFloat16) support.
//!
//! This module provides a wrapper around the `half` crate's `bf16` type,
//! implementing the `VectorElement` trait for use in vector similarity operations.

use super::VectorElement;
use std::fmt;

/// Brain floating point number (bfloat16).
///
/// BFloat16 is a 16-bit floating point format that uses the same exponent
/// range as f32 but with reduced mantissa precision. This format is
/// particularly popular in machine learning applications.
///
/// BF16 provides:
/// - 1 sign bit
/// - 8 exponent bits (same as f32)
/// - 7 mantissa bits
/// - Range: ~1.2e-38 to ~3.4e38 (same as f32)
/// - Precision: ~2 decimal digits
#[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct BFloat16(half::bf16);

impl BFloat16 {
    /// Create a new BFloat16 from raw bits.
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Self(half::bf16::from_bits(bits))
    }

    /// Get the raw bits of this BFloat16.
    #[inline(always)]
    pub const fn to_bits(self) -> u16 {
        self.0.to_bits()
    }

    /// Create a BFloat16 from an f32.
    #[inline(always)]
    pub fn from_f32(v: f32) -> Self {
        Self(half::bf16::from_f32(v))
    }

    /// Convert to f32.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    /// Zero value.
    pub const ZERO: Self = Self(half::bf16::ZERO);

    /// One value.
    pub const ONE: Self = Self(half::bf16::ONE);

    /// Positive infinity.
    pub const INFINITY: Self = Self(half::bf16::INFINITY);

    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(half::bf16::NEG_INFINITY);

    /// Not a number.
    pub const NAN: Self = Self(half::bf16::NAN);
}

impl fmt::Debug for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BFloat16({})", self.to_f32())
    }
}

impl fmt::Display for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for BFloat16 {
    #[inline(always)]
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl From<BFloat16> for f32 {
    #[inline(always)]
    fn from(v: BFloat16) -> Self {
        v.to_f32()
    }
}

impl VectorElement for BFloat16 {
    type DistanceType = f32;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        Self(half::bf16::from_f32(v))
    }

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn alignment() -> usize {
        32 // AVX alignment for f32 intermediate calculations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfloat16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -100.0, 1e10, -1e10];
        for v in values {
            let bf16 = BFloat16::from_f32(v);
            let back = bf16.to_f32();
            // BF16 has the same range as f32 but limited precision
            if v == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                let rel_error = ((back - v) / v).abs();
                assert!(rel_error < 0.01, "rel_error = {} for v = {}", rel_error, v);
            }
        }
    }

    #[test]
    fn test_bfloat16_vector_element() {
        let bf16 = BFloat16::from_f32(2.5);
        assert!((VectorElement::to_f32(bf16) - 2.5).abs() < 0.1);
        assert_eq!(BFloat16::zero().to_f32(), 0.0);
    }
}

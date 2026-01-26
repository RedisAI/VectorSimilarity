//! Half-precision floating point (FP16/Float16) support.
//!
//! This module provides a wrapper around the `half` crate's `f16` type,
//! implementing the `VectorElement` trait for use in vector similarity operations.

use super::VectorElement;
use crate::serialization::DataTypeId;
use std::fmt;

/// Half-precision floating point number (IEEE 754-2008 binary16).
///
/// This is a wrapper around `half::f16` that implements `VectorElement`.
/// FP16 provides:
/// - 1 sign bit
/// - 5 exponent bits
/// - 10 mantissa bits
/// - Range: ~6.0e-5 to 65504
/// - Precision: ~3 decimal digits
#[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Float16(half::f16);

impl Float16 {
    /// Create a new Float16 from raw bits.
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Self(half::f16::from_bits(bits))
    }

    /// Get the raw bits of this Float16.
    #[inline(always)]
    pub const fn to_bits(self) -> u16 {
        self.0.to_bits()
    }

    /// Create a Float16 from an f32.
    #[inline(always)]
    pub fn from_f32(v: f32) -> Self {
        Self(half::f16::from_f32(v))
    }

    /// Convert to f32.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    /// Zero value.
    pub const ZERO: Self = Self(half::f16::ZERO);

    /// One value.
    pub const ONE: Self = Self(half::f16::ONE);

    /// Positive infinity.
    pub const INFINITY: Self = Self(half::f16::INFINITY);

    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(half::f16::NEG_INFINITY);

    /// Not a number.
    pub const NAN: Self = Self(half::f16::NAN);
}

impl fmt::Debug for Float16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Float16({})", self.to_f32())
    }
}

impl fmt::Display for Float16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for Float16 {
    #[inline(always)]
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl From<Float16> for f32 {
    #[inline(always)]
    fn from(v: Float16) -> Self {
        v.to_f32()
    }
}

impl VectorElement for Float16 {
    type DistanceType = f32;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        Self(half::f16::from_f32(v))
    }

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn alignment() -> usize {
        32 // AVX alignment for f32 intermediate calculations
    }

    #[inline]
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.to_bits().to_le_bytes())
    }

    #[inline]
    fn read_from<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        Ok(Self::from_bits(u16::from_le_bytes(buf)))
    }

    fn data_type_id() -> DataTypeId {
        DataTypeId::Float16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -100.0];
        for v in values {
            let fp16 = Float16::from_f32(v);
            let back = fp16.to_f32();
            // FP16 has limited precision, so we check approximate equality
            assert!((back - v).abs() < 0.01 * v.abs().max(1.0));
        }
    }

    #[test]
    fn test_float16_vector_element() {
        let fp16 = Float16::from_f32(2.5);
        assert!((VectorElement::to_f32(fp16) - 2.5).abs() < 0.01);
        assert_eq!(Float16::zero().to_f32(), 0.0);
    }
}

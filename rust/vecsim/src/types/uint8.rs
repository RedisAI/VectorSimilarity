//! Unsigned 8-bit integer (UINT8) support.
//!
//! This module provides a `UInt8` wrapper type implementing the `VectorElement` trait
//! for use in vector similarity operations with 8-bit unsigned integer vectors.

use super::VectorElement;
use crate::serialization::DataTypeId;
use std::fmt;

/// Unsigned 8-bit integer for vector storage.
///
/// This type wraps `u8` and implements `VectorElement` for use in vector indices.
/// UINT8 provides:
/// - Range: 0 to 255
/// - Memory efficient: 4x smaller than f32
/// - Useful for quantized embeddings, image features, or SQ8 storage
///
/// Distance calculations are performed in f32 for precision.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct UInt8(pub u8);

impl UInt8 {
    /// Create a new UInt8 from a raw u8 value.
    #[inline(always)]
    pub const fn new(v: u8) -> Self {
        Self(v)
    }

    /// Get the raw u8 value.
    #[inline(always)]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Zero value.
    pub const ZERO: Self = Self(0);

    /// Maximum value (255).
    pub const MAX: Self = Self(u8::MAX);

    /// Minimum value (0).
    pub const MIN: Self = Self(0);
}

impl fmt::Debug for UInt8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UInt8({})", self.0)
    }
}

impl fmt::Display for UInt8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u8> for UInt8 {
    #[inline(always)]
    fn from(v: u8) -> Self {
        Self(v)
    }
}

impl From<UInt8> for u8 {
    #[inline(always)]
    fn from(v: UInt8) -> Self {
        v.0
    }
}

impl From<UInt8> for f32 {
    #[inline(always)]
    fn from(v: UInt8) -> Self {
        v.0 as f32
    }
}

impl VectorElement for UInt8 {
    type DistanceType = f32;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.0 as f32
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        // Clamp to u8 range and round
        Self(v.round().clamp(0.0, 255.0) as u8)
    }

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn alignment() -> usize {
        32 // AVX alignment for f32 intermediate calculations
    }

    #[inline(always)]
    fn can_normalize() -> bool {
        // UInt8 cannot be meaningfully normalized - normalized values round to 0
        false
    }

    #[inline]
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&[self.0])
    }

    #[inline]
    fn read_from<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;
        Ok(Self(buf[0]))
    }

    fn data_type_id() -> DataTypeId {
        DataTypeId::UInt8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uint8_roundtrip() {
        let values = [0u8, 1, 50, 100, 200, 255];
        for v in values {
            let uint8 = UInt8::new(v);
            assert_eq!(uint8.get(), v);
            assert_eq!(uint8.to_f32() as u8, v);
        }
    }

    #[test]
    fn test_uint8_from_f32() {
        // Exact values
        assert_eq!(UInt8::from_f32(0.0).get(), 0);
        assert_eq!(UInt8::from_f32(100.0).get(), 100);
        assert_eq!(UInt8::from_f32(255.0).get(), 255);

        // Rounding
        assert_eq!(UInt8::from_f32(50.4).get(), 50);
        assert_eq!(UInt8::from_f32(50.6).get(), 51);

        // Clamping
        assert_eq!(UInt8::from_f32(300.0).get(), 255);
        assert_eq!(UInt8::from_f32(-50.0).get(), 0);
    }

    #[test]
    fn test_uint8_vector_element() {
        let uint8 = UInt8::new(42);
        assert_eq!(VectorElement::to_f32(uint8), 42.0);
        assert_eq!(UInt8::zero().get(), 0);
    }

    #[test]
    fn test_uint8_traits() {
        // Test Copy, Clone
        let a = UInt8::new(10);
        let b = a;
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);

        // Test Ord
        assert!(UInt8::new(10) > UInt8::new(5));
        assert!(UInt8::new(100) < UInt8::new(200));

        // Test Default
        let d: UInt8 = Default::default();
        assert_eq!(d.get(), 0);
    }
}

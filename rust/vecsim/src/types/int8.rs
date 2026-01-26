//! Signed 8-bit integer (INT8) support.
//!
//! This module provides an `Int8` wrapper type implementing the `VectorElement` trait
//! for use in vector similarity operations with 8-bit signed integer vectors.

use super::VectorElement;
use crate::serialization::DataTypeId;
use std::fmt;

/// Signed 8-bit integer for vector storage.
///
/// This type wraps `i8` and implements `VectorElement` for use in vector indices.
/// INT8 provides:
/// - Range: -128 to 127
/// - Memory efficient: 4x smaller than f32
/// - Useful for quantized embeddings or integer-based features
///
/// Distance calculations are performed in f32 for precision.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Int8(pub i8);

impl Int8 {
    /// Create a new Int8 from a raw i8 value.
    #[inline(always)]
    pub const fn new(v: i8) -> Self {
        Self(v)
    }

    /// Get the raw i8 value.
    #[inline(always)]
    pub const fn get(self) -> i8 {
        self.0
    }

    /// Zero value.
    pub const ZERO: Self = Self(0);

    /// Maximum value (127).
    pub const MAX: Self = Self(i8::MAX);

    /// Minimum value (-128).
    pub const MIN: Self = Self(i8::MIN);
}

impl fmt::Debug for Int8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Int8({})", self.0)
    }
}

impl fmt::Display for Int8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i8> for Int8 {
    #[inline(always)]
    fn from(v: i8) -> Self {
        Self(v)
    }
}

impl From<Int8> for i8 {
    #[inline(always)]
    fn from(v: Int8) -> Self {
        v.0
    }
}

impl From<Int8> for f32 {
    #[inline(always)]
    fn from(v: Int8) -> Self {
        v.0 as f32
    }
}

impl VectorElement for Int8 {
    type DistanceType = f32;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.0 as f32
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        // Clamp to i8 range and round
        Self(v.round().clamp(-128.0, 127.0) as i8)
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
        // Int8 cannot be meaningfully normalized - normalized values round to 0
        false
    }

    #[inline]
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.0.to_le_bytes())
    }

    #[inline]
    fn read_from<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;
        Ok(Self(i8::from_le_bytes(buf)))
    }

    fn data_type_id() -> DataTypeId {
        DataTypeId::Int8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_roundtrip() {
        let values = [0i8, 1, -1, 50, -50, 127, -128];
        for v in values {
            let int8 = Int8::new(v);
            assert_eq!(int8.get(), v);
            assert_eq!(int8.to_f32() as i8, v);
        }
    }

    #[test]
    fn test_int8_from_f32() {
        // Exact values
        assert_eq!(Int8::from_f32(0.0).get(), 0);
        assert_eq!(Int8::from_f32(100.0).get(), 100);
        assert_eq!(Int8::from_f32(-100.0).get(), -100);

        // Rounding
        assert_eq!(Int8::from_f32(50.4).get(), 50);
        assert_eq!(Int8::from_f32(50.6).get(), 51);
        assert_eq!(Int8::from_f32(-50.4).get(), -50);
        assert_eq!(Int8::from_f32(-50.6).get(), -51);

        // Clamping
        assert_eq!(Int8::from_f32(200.0).get(), 127);
        assert_eq!(Int8::from_f32(-200.0).get(), -128);
    }

    #[test]
    fn test_int8_vector_element() {
        let int8 = Int8::new(42);
        assert_eq!(VectorElement::to_f32(int8), 42.0);
        assert_eq!(Int8::zero().get(), 0);
    }

    #[test]
    fn test_int8_traits() {
        // Test Copy, Clone
        let a = Int8::new(10);
        let b = a;
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);

        // Test Ord
        assert!(Int8::new(10) > Int8::new(5));
        assert!(Int8::new(-5) < Int8::new(5));

        // Test Default
        let d: Int8 = Default::default();
        assert_eq!(d.get(), 0);
    }
}

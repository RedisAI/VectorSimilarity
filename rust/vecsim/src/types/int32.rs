//! Signed 32-bit integer (INT32) support.
//!
//! This module provides an `Int32` wrapper type implementing the `VectorElement` trait
//! for use in vector similarity operations with 32-bit signed integer vectors.

use super::VectorElement;
use crate::serialization::DataTypeId;
use std::fmt;

/// Signed 32-bit integer for vector storage.
///
/// This type wraps `i32` and implements `VectorElement` for use in vector indices.
/// INT32 provides:
/// - Range: -2,147,483,648 to 2,147,483,647
/// - Useful for large integer-based features or sparse indices
///
/// Distance calculations are performed in f64 for precision with large values.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Int32(pub i32);

impl Int32 {
    /// Create a new Int32 from a raw i32 value.
    #[inline(always)]
    pub const fn new(v: i32) -> Self {
        Self(v)
    }

    /// Get the raw i32 value.
    #[inline(always)]
    pub const fn get(self) -> i32 {
        self.0
    }

    /// Zero value.
    pub const ZERO: Self = Self(0);

    /// Maximum value.
    pub const MAX: Self = Self(i32::MAX);

    /// Minimum value.
    pub const MIN: Self = Self(i32::MIN);
}

impl fmt::Debug for Int32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Int32({})", self.0)
    }
}

impl fmt::Display for Int32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i32> for Int32 {
    #[inline(always)]
    fn from(v: i32) -> Self {
        Self(v)
    }
}

impl From<Int32> for i32 {
    #[inline(always)]
    fn from(v: Int32) -> Self {
        v.0
    }
}

impl From<Int32> for f32 {
    #[inline(always)]
    fn from(v: Int32) -> Self {
        v.0 as f32
    }
}

impl From<Int32> for f64 {
    #[inline(always)]
    fn from(v: Int32) -> Self {
        v.0 as f64
    }
}

impl VectorElement for Int32 {
    type DistanceType = f32;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.0 as f32
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        // Clamp to i32 range and round
        Self(v.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32)
    }

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn alignment() -> usize {
        32 // AVX alignment
    }

    #[inline(always)]
    fn can_normalize() -> bool {
        // Int32 cannot be meaningfully normalized - normalized values round to 0
        false
    }

    #[inline]
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.0.to_le_bytes())
    }

    #[inline]
    fn read_from<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(Self(i32::from_le_bytes(buf)))
    }

    fn data_type_id() -> DataTypeId {
        DataTypeId::Int32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int32_roundtrip() {
        let values = [0i32, 1, -1, 1000, -1000, i32::MAX, i32::MIN];
        for v in values {
            let int32 = Int32::new(v);
            assert_eq!(int32.get(), v);
        }
    }

    #[test]
    fn test_int32_from_f32() {
        // Exact values
        assert_eq!(Int32::from_f32(0.0).get(), 0);
        assert_eq!(Int32::from_f32(1000.0).get(), 1000);
        assert_eq!(Int32::from_f32(-1000.0).get(), -1000);

        // Rounding
        assert_eq!(Int32::from_f32(500.4).get(), 500);
        assert_eq!(Int32::from_f32(500.6).get(), 501);
        assert_eq!(Int32::from_f32(-500.4).get(), -500);
        assert_eq!(Int32::from_f32(-500.6).get(), -501);
    }

    #[test]
    fn test_int32_vector_element() {
        let int32 = Int32::new(42);
        assert_eq!(VectorElement::to_f32(int32), 42.0);
        assert_eq!(Int32::zero().get(), 0);
    }

    #[test]
    fn test_int32_traits() {
        // Test Copy, Clone
        let a = Int32::new(10);
        let b = a;
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);

        // Test Ord
        assert!(Int32::new(10) > Int32::new(5));
        assert!(Int32::new(-5) < Int32::new(5));

        // Test Default
        let d: Int32 = Default::default();
        assert_eq!(d.get(), 0);
    }
}

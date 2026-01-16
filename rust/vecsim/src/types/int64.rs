//! Signed 64-bit integer (INT64) support.
//!
//! This module provides an `Int64` wrapper type implementing the `VectorElement` trait
//! for use in vector similarity operations with 64-bit signed integer vectors.

use super::VectorElement;
use std::fmt;

/// Signed 64-bit integer for vector storage.
///
/// This type wraps `i64` and implements `VectorElement` for use in vector indices.
/// INT64 provides:
/// - Range: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
/// - Useful for very large integer-based features or identifiers
///
/// Note: Conversion to f32 may lose precision for large values.
/// Distance calculations use f32 for compatibility with other types.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Int64(pub i64);

impl Int64 {
    /// Create a new Int64 from a raw i64 value.
    #[inline(always)]
    pub const fn new(v: i64) -> Self {
        Self(v)
    }

    /// Get the raw i64 value.
    #[inline(always)]
    pub const fn get(self) -> i64 {
        self.0
    }

    /// Zero value.
    pub const ZERO: Self = Self(0);

    /// Maximum value.
    pub const MAX: Self = Self(i64::MAX);

    /// Minimum value.
    pub const MIN: Self = Self(i64::MIN);
}

impl fmt::Debug for Int64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Int64({})", self.0)
    }
}

impl fmt::Display for Int64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i64> for Int64 {
    #[inline(always)]
    fn from(v: i64) -> Self {
        Self(v)
    }
}

impl From<Int64> for i64 {
    #[inline(always)]
    fn from(v: Int64) -> Self {
        v.0
    }
}

impl From<Int64> for f32 {
    #[inline(always)]
    fn from(v: Int64) -> Self {
        v.0 as f32
    }
}

impl From<Int64> for f64 {
    #[inline(always)]
    fn from(v: Int64) -> Self {
        v.0 as f64
    }
}

impl VectorElement for Int64 {
    type DistanceType = f32;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.0 as f32
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        // Clamp to i64 range (limited by f32 precision)
        // f32 can exactly represent integers up to 2^24
        Self(v.round() as i64)
    }

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn alignment() -> usize {
        64 // AVX-512 alignment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int64_roundtrip() {
        let values = [0i64, 1, -1, 1_000_000, -1_000_000];
        for v in values {
            let int64 = Int64::new(v);
            assert_eq!(int64.get(), v);
        }
    }

    #[test]
    fn test_int64_from_f32() {
        // Exact values (within f32 precision)
        assert_eq!(Int64::from_f32(0.0).get(), 0);
        assert_eq!(Int64::from_f32(1000.0).get(), 1000);
        assert_eq!(Int64::from_f32(-1000.0).get(), -1000);

        // Rounding
        assert_eq!(Int64::from_f32(500.4).get(), 500);
        assert_eq!(Int64::from_f32(500.6).get(), 501);
    }

    #[test]
    fn test_int64_vector_element() {
        let int64 = Int64::new(42);
        assert_eq!(VectorElement::to_f32(int64), 42.0);
        assert_eq!(Int64::zero().get(), 0);
    }

    #[test]
    fn test_int64_traits() {
        // Test Copy, Clone
        let a = Int64::new(10);
        let b = a;
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);

        // Test Ord
        assert!(Int64::new(10) > Int64::new(5));
        assert!(Int64::new(-5) < Int64::new(5));

        // Test Default
        let d: Int64 = Default::default();
        assert_eq!(d.get(), 0);
    }

    #[test]
    fn test_int64_large_values() {
        // Test with larger values that still fit in f32 precisely
        let large = Int64::new(16_000_000);
        assert_eq!(large.to_f32(), 16_000_000.0);
    }
}

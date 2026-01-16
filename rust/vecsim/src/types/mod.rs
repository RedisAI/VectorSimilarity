//! Core type definitions for the vector similarity library.
//!
//! This module defines the fundamental types used throughout the library:
//! - `LabelType`: External label for vectors (user-provided identifier)
//! - `IdType`: Internal vector identifier
//! - `VectorElement`: Trait for vector element types (f32, f64, Float16, BFloat16, Int8, UInt8, Int32, Int64)
//! - `DistanceType`: Trait for distance computation result types

pub mod bf16;
pub mod fp16;
pub mod int8;
pub mod int32;
pub mod int64;
pub mod uint8;

pub use bf16::BFloat16;
pub use fp16::Float16;
pub use int8::Int8;
pub use int32::Int32;
pub use int64::Int64;
pub use uint8::UInt8;

use num_traits::Float;
use std::fmt::Debug;

/// External label type for vectors (user-provided identifier).
pub type LabelType = u64;

/// Internal vector identifier.
pub type IdType = u32;

/// Invalid/sentinel value for internal IDs.
pub const INVALID_ID: IdType = IdType::MAX;

/// Trait for types that can be used as vector elements.
///
/// This trait abstracts over the different numeric types that can be stored
/// in vectors: f32, f64, Float16, and BFloat16.
pub trait VectorElement: Copy + Clone + Debug + Send + Sync + 'static {
    /// The type used for distance calculations (typically f32 or f64).
    type DistanceType: DistanceType;

    /// Convert to f32 for distance calculations.
    fn to_f32(self) -> f32;

    /// Create from f32.
    fn from_f32(v: f32) -> Self;

    /// Zero value.
    fn zero() -> Self;

    /// Alignment requirement in bytes for SIMD operations.
    fn alignment() -> usize {
        std::mem::align_of::<Self>()
    }

    /// Whether this type can be meaningfully normalized for cosine distance.
    ///
    /// Returns `true` for floating-point types that can represent values in [0, 1].
    /// Returns `false` for integer types where normalization would lose precision.
    fn can_normalize() -> bool {
        true
    }
}

/// Trait for distance computation result types.
pub trait DistanceType:
    Copy + Clone + Debug + Send + Sync + PartialOrd + 'static + Default
{
    /// Zero distance.
    fn zero() -> Self;

    /// Maximum possible distance (infinity).
    fn infinity() -> Self;

    /// Minimum possible distance (negative infinity or zero depending on metric).
    fn neg_infinity() -> Self;

    /// Convert to f64 for comparisons.
    fn to_f64(self) -> f64;

    /// Create from f64.
    fn from_f64(v: f64) -> Self;

    /// Square root (for L2 distance normalization).
    fn sqrt(self) -> Self;
}

// Implementation for f32
impl VectorElement for f32 {
    type DistanceType = f32;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }

    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn alignment() -> usize {
        32 // AVX alignment
    }
}

impl DistanceType for f32 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn infinity() -> Self {
        f32::INFINITY
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        Float::sqrt(self)
    }
}

// Implementation for f64
impl VectorElement for f64 {
    type DistanceType = f64;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v as f64
    }

    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn alignment() -> usize {
        64 // AVX-512 alignment
    }
}

impl DistanceType for f64 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }

    #[inline(always)]
    fn infinity() -> Self {
        f64::INFINITY
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        Float::sqrt(self)
    }
}

//! Quantization support for vector compression.
//!
//! This module provides quantization methods to compress vectors for efficient storage
//! and faster distance computations:
//! - `SQ8`: Scalar quantization to 8-bit unsigned integers with per-vector scaling

pub mod sq8;

pub use sq8::{Sq8Codec, Sq8VectorMeta};

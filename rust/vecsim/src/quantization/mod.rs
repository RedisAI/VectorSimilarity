//! Quantization support for vector compression.
//!
//! This module provides quantization methods to compress vectors for efficient storage
//! and faster distance computations:
//! - `SQ8`: Scalar quantization to 8-bit unsigned integers with per-vector scaling
//! - `sq8_simd`: SIMD-optimized asymmetric distance functions for SQ8

pub mod sq8;
pub mod sq8_simd;

pub use sq8::{Sq8Codec, Sq8VectorMeta};
pub use sq8_simd::{sq8_cosine_simd, sq8_inner_product_simd, sq8_l2_squared_simd};

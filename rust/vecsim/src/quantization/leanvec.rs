//! LeanVec quantization with dimension reduction.
//!
//! LeanVec combines dimension reduction with two-level quantization for efficient
//! vector compression while maintaining search accuracy:
//!
//! - **4x8 LeanVec**: 4-bit primary (on D/2 dims) + 8-bit residual (on D dims)
//! - **8x8 LeanVec**: 8-bit primary (on D/2 dims) + 8-bit residual (on D dims)
//!
//! ## Storage Layout
//!
//! For 4x8 LeanVec with D=128 and leanvec_dim=64:
//! ```text
//! Metadata (24 bytes):
//!   - min_primary, delta_primary (8 bytes)
//!   - min_residual, delta_residual (8 bytes)
//!   - leanvec_dim (8 bytes)
//!
//! Primary data: leanvec_dim * primary_bits / 8 bytes
//!   - For 4x8: 64 * 4 / 8 = 32 bytes
//!   - For 8x8: 64 * 8 / 8 = 64 bytes
//!
//! Residual data: dim * 8 / 8 bytes = 128 bytes
//! ```
//!
//! ## Memory Efficiency
//!
//! For D=128:
//! - f32 uncompressed: 512 bytes
//! - 4x8 LeanVec: 24 + 32 + 128 = 184 bytes (~2.8x compression)
//! - 8x8 LeanVec: 24 + 64 + 128 = 216 bytes (~2.4x compression)

use std::fmt;

/// LeanVec configuration specifying primary and residual quantization bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeanVecBits {
    /// 4-bit primary (reduced dims) + 8-bit residual (full dims).
    LeanVec4x8,
    /// 8-bit primary (reduced dims) + 8-bit residual (full dims).
    LeanVec8x8,
}

impl LeanVecBits {
    /// Get primary quantization bits.
    pub fn primary_bits(&self) -> usize {
        match self {
            LeanVecBits::LeanVec4x8 => 4,
            LeanVecBits::LeanVec8x8 => 8,
        }
    }

    /// Get residual quantization bits.
    pub fn residual_bits(&self) -> usize {
        8 // Always 8-bit for LeanVec
    }

    /// Get number of primary quantization levels.
    pub fn primary_levels(&self) -> usize {
        1 << self.primary_bits()
    }

    /// Calculate primary data size in bytes.
    pub fn primary_data_size(&self, leanvec_dim: usize) -> usize {
        (leanvec_dim * self.primary_bits() + 7) / 8
    }

    /// Calculate residual data size in bytes.
    pub fn residual_data_size(&self, full_dim: usize) -> usize {
        full_dim // 8 bits per dimension = 1 byte
    }

    /// Calculate total encoded size (excluding metadata).
    pub fn encoded_size(&self, full_dim: usize, leanvec_dim: usize) -> usize {
        self.primary_data_size(leanvec_dim) + self.residual_data_size(full_dim)
    }
}

/// Metadata for a LeanVec-encoded vector.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct LeanVecMeta {
    /// Minimum value for primary quantization.
    pub min_primary: f32,
    /// Scale factor for primary quantization.
    pub delta_primary: f32,
    /// Minimum value for residual quantization.
    pub min_residual: f32,
    /// Scale factor for residual quantization.
    pub delta_residual: f32,
    /// Reduced dimension used for primary quantization.
    pub leanvec_dim: u32,
    /// Padding for alignment.
    _pad: u32,
}

impl fmt::Debug for LeanVecMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LeanVecMeta")
            .field("min_primary", &self.min_primary)
            .field("delta_primary", &self.delta_primary)
            .field("min_residual", &self.min_residual)
            .field("delta_residual", &self.delta_residual)
            .field("leanvec_dim", &self.leanvec_dim)
            .finish()
    }
}

impl Default for LeanVecMeta {
    fn default() -> Self {
        Self {
            min_primary: 0.0,
            delta_primary: 1.0,
            min_residual: 0.0,
            delta_residual: 1.0,
            leanvec_dim: 0,
            _pad: 0,
        }
    }
}

impl LeanVecMeta {
    /// Size of metadata in bytes.
    pub const SIZE: usize = 24;
}

/// LeanVec codec for encoding and decoding vectors with dimension reduction.
pub struct LeanVecCodec {
    /// Full dimensionality of vectors.
    dim: usize,
    /// Reduced dimensionality for primary quantization.
    leanvec_dim: usize,
    /// Quantization configuration.
    bits: LeanVecBits,
    /// Indices of dimensions selected for primary quantization.
    /// These are the top-variance dimensions (or simply first D/2 for simplicity).
    selected_dims: Vec<usize>,
}

impl LeanVecCodec {
    /// Create a new LeanVec codec with default reduced dimension (dim/2).
    pub fn new(dim: usize, bits: LeanVecBits) -> Self {
        Self::with_leanvec_dim(dim, bits, 0)
    }

    /// Create a new LeanVec codec with custom reduced dimension.
    ///
    /// # Arguments
    /// * `dim` - Full dimensionality of vectors
    /// * `bits` - Quantization configuration
    /// * `leanvec_dim` - Reduced dimension (0 = default dim/2)
    pub fn with_leanvec_dim(dim: usize, bits: LeanVecBits, leanvec_dim: usize) -> Self {
        let leanvec_dim = if leanvec_dim == 0 {
            dim / 2
        } else {
            leanvec_dim.min(dim)
        };

        // Default: select first leanvec_dim dimensions
        // In a full implementation, this would be learned from data
        let selected_dims: Vec<usize> = (0..leanvec_dim).collect();

        Self {
            dim,
            leanvec_dim,
            bits,
            selected_dims,
        }
    }

    /// Get the full dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the reduced dimension.
    pub fn leanvec_dim(&self) -> usize {
        self.leanvec_dim
    }

    /// Get the bits configuration.
    pub fn bits(&self) -> LeanVecBits {
        self.bits
    }

    /// Get total encoded size including metadata.
    pub fn total_size(&self) -> usize {
        LeanVecMeta::SIZE + self.bits.encoded_size(self.dim, self.leanvec_dim)
    }

    /// Get the size of just the encoded data (without metadata).
    pub fn encoded_size(&self) -> usize {
        self.bits.encoded_size(self.dim, self.leanvec_dim)
    }

    /// Set custom dimension selection order (e.g., from variance analysis).
    pub fn set_selected_dims(&mut self, dims: Vec<usize>) {
        assert!(dims.len() >= self.leanvec_dim);
        self.selected_dims = dims;
    }

    /// Extract reduced-dimension vector for primary quantization.
    fn extract_reduced(&self, vector: &[f32]) -> Vec<f32> {
        self.selected_dims
            .iter()
            .take(self.leanvec_dim)
            .map(|&i| vector[i])
            .collect()
    }

    /// Encode a vector using LeanVec4x8 (4-bit primary + 8-bit residual).
    pub fn encode_4x8(&self, vector: &[f32]) -> (LeanVecMeta, Vec<u8>) {
        debug_assert_eq!(vector.len(), self.dim);

        // Extract reduced dimensions for primary quantization
        let reduced = self.extract_reduced(vector);

        // Primary quantization (4-bit) on reduced dimensions
        let (min_primary, max_primary) = reduced
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

        let range_primary = max_primary - min_primary;
        let delta_primary = if range_primary > 1e-10 {
            range_primary / 15.0 // 4-bit = 16 levels
        } else {
            1.0
        };
        let inv_delta_primary = 1.0 / delta_primary;

        // Encode primary (4-bit packed)
        let primary_bytes = (self.leanvec_dim + 1) / 2;
        let mut primary_encoded = vec![0u8; primary_bytes];

        // We also need to track the primary reconstruction for residual calculation
        let mut primary_reconstructed = vec![0.0f32; self.leanvec_dim];

        for i in 0..self.leanvec_dim {
            let normalized = (reduced[i] - min_primary) * inv_delta_primary;
            let q = (normalized.round() as u8).min(15);

            let byte_idx = i / 2;
            if i % 2 == 0 {
                primary_encoded[byte_idx] |= q;
            } else {
                primary_encoded[byte_idx] |= q << 4;
            }

            primary_reconstructed[i] = min_primary + (q as f32) * delta_primary;
        }

        // Compute residuals (full dimension - reconstruction error contribution)
        // For simplicity, we compute residuals as the full vector quantized
        // In a full implementation, this would account for primary reconstruction

        let (min_residual, max_residual) = vector
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

        let range_residual = max_residual - min_residual;
        let delta_residual = if range_residual > 1e-10 {
            range_residual / 255.0 // 8-bit = 256 levels
        } else {
            1.0
        };
        let inv_delta_residual = 1.0 / delta_residual;

        // Encode residuals (8-bit, full dimension)
        let mut residual_encoded = vec![0u8; self.dim];
        for i in 0..self.dim {
            let normalized = (vector[i] - min_residual) * inv_delta_residual;
            residual_encoded[i] = (normalized.round() as u8).min(255);
        }

        // Combine primary and residual data
        let mut encoded = primary_encoded;
        encoded.extend(residual_encoded);

        let meta = LeanVecMeta {
            min_primary,
            delta_primary,
            min_residual,
            delta_residual,
            leanvec_dim: self.leanvec_dim as u32,
            _pad: 0,
        };

        (meta, encoded)
    }

    /// Decode a LeanVec4x8 encoded vector.
    pub fn decode_4x8(&self, meta: &LeanVecMeta, encoded: &[u8]) -> Vec<f32> {
        let leanvec_dim = meta.leanvec_dim as usize;
        let primary_bytes = (leanvec_dim + 1) / 2;

        // For decoding, we primarily use the residual (full precision) part
        // The primary is for fast approximate search
        let mut decoded = Vec::with_capacity(self.dim);

        for i in 0..self.dim {
            let residual_q = encoded[primary_bytes + i];
            let value = meta.min_residual + (residual_q as f32) * meta.delta_residual;
            decoded.push(value);
        }

        decoded
    }

    /// Encode a vector using LeanVec8x8 (8-bit primary + 8-bit residual).
    pub fn encode_8x8(&self, vector: &[f32]) -> (LeanVecMeta, Vec<u8>) {
        debug_assert_eq!(vector.len(), self.dim);

        // Extract reduced dimensions for primary quantization
        let reduced = self.extract_reduced(vector);

        // Primary quantization (8-bit) on reduced dimensions
        let (min_primary, max_primary) = reduced
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

        let range_primary = max_primary - min_primary;
        let delta_primary = if range_primary > 1e-10 {
            range_primary / 255.0 // 8-bit = 256 levels
        } else {
            1.0
        };
        let inv_delta_primary = 1.0 / delta_primary;

        // Encode primary (8-bit, one byte per dim)
        let mut primary_encoded = vec![0u8; self.leanvec_dim];
        for i in 0..self.leanvec_dim {
            let normalized = (reduced[i] - min_primary) * inv_delta_primary;
            primary_encoded[i] = (normalized.round() as u8).min(255);
        }

        // Residual quantization (8-bit, full dimension)
        let (min_residual, max_residual) = vector
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

        let range_residual = max_residual - min_residual;
        let delta_residual = if range_residual > 1e-10 {
            range_residual / 255.0
        } else {
            1.0
        };
        let inv_delta_residual = 1.0 / delta_residual;

        let mut residual_encoded = vec![0u8; self.dim];
        for i in 0..self.dim {
            let normalized = (vector[i] - min_residual) * inv_delta_residual;
            residual_encoded[i] = (normalized.round() as u8).min(255);
        }

        // Combine
        let mut encoded = primary_encoded;
        encoded.extend(residual_encoded);

        let meta = LeanVecMeta {
            min_primary,
            delta_primary,
            min_residual,
            delta_residual,
            leanvec_dim: self.leanvec_dim as u32,
            _pad: 0,
        };

        (meta, encoded)
    }

    /// Decode a LeanVec8x8 encoded vector.
    pub fn decode_8x8(&self, meta: &LeanVecMeta, encoded: &[u8]) -> Vec<f32> {
        let leanvec_dim = meta.leanvec_dim as usize;

        // Use residual part for full reconstruction
        let mut decoded = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let residual_q = encoded[leanvec_dim + i];
            let value = meta.min_residual + (residual_q as f32) * meta.delta_residual;
            decoded.push(value);
        }

        decoded
    }

    /// Encode using configured bits.
    pub fn encode(&self, vector: &[f32]) -> (LeanVecMeta, Vec<u8>) {
        match self.bits {
            LeanVecBits::LeanVec4x8 => self.encode_4x8(vector),
            LeanVecBits::LeanVec8x8 => self.encode_8x8(vector),
        }
    }

    /// Decode using configured bits.
    pub fn decode(&self, meta: &LeanVecMeta, encoded: &[u8]) -> Vec<f32> {
        match self.bits {
            LeanVecBits::LeanVec4x8 => self.decode_4x8(meta, encoded),
            LeanVecBits::LeanVec8x8 => self.decode_8x8(meta, encoded),
        }
    }
}

/// Compute asymmetric L2 squared distance using primary (reduced) dimensions only.
///
/// This is a fast approximate distance for initial filtering.
#[inline]
pub fn leanvec_primary_l2_squared_4bit(
    query: &[f32],
    encoded: &[u8],
    meta: &LeanVecMeta,
    selected_dims: &[usize],
) -> f32 {
    let leanvec_dim = meta.leanvec_dim as usize;
    let mut sum = 0.0f32;

    for i in 0..leanvec_dim {
        let byte_idx = i / 2;
        let q = if i % 2 == 0 {
            encoded[byte_idx] & 0x0F
        } else {
            (encoded[byte_idx] >> 4) & 0x0F
        };

        let stored = meta.min_primary + (q as f32) * meta.delta_primary;
        let diff = query[selected_dims[i]] - stored;
        sum += diff * diff;
    }

    sum
}

/// Compute asymmetric L2 squared distance using primary (reduced) dimensions only.
///
/// This is a fast approximate distance for initial filtering. (8-bit version)
#[inline]
pub fn leanvec_primary_l2_squared_8bit(
    query: &[f32],
    encoded: &[u8],
    meta: &LeanVecMeta,
    selected_dims: &[usize],
) -> f32 {
    let leanvec_dim = meta.leanvec_dim as usize;
    let mut sum = 0.0f32;

    for i in 0..leanvec_dim {
        let stored = meta.min_primary + (encoded[i] as f32) * meta.delta_primary;
        let diff = query[selected_dims[i]] - stored;
        sum += diff * diff;
    }

    sum
}

/// Compute full asymmetric L2 squared distance using residual (full) dimensions.
#[inline]
pub fn leanvec_residual_l2_squared(
    query: &[f32],
    encoded: &[u8],
    meta: &LeanVecMeta,
    primary_bytes: usize,
) -> f32 {
    let dim = query.len();
    let mut sum = 0.0f32;

    for i in 0..dim {
        let residual_q = encoded[primary_bytes + i];
        let stored = meta.min_residual + (residual_q as f32) * meta.delta_residual;
        let diff = query[i] - stored;
        sum += diff * diff;
    }

    sum
}

/// Compute full asymmetric inner product using residual (full) dimensions.
#[inline]
pub fn leanvec_residual_inner_product(
    query: &[f32],
    encoded: &[u8],
    meta: &LeanVecMeta,
    primary_bytes: usize,
) -> f32 {
    let dim = query.len();
    let mut sum = 0.0f32;

    for i in 0..dim {
        let residual_q = encoded[primary_bytes + i];
        let stored = meta.min_residual + (residual_q as f32) * meta.delta_residual;
        sum += query[i] * stored;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leanvec_4x8_encode_decode() {
        let codec = LeanVecCodec::new(128, LeanVecBits::LeanVec4x8);
        assert_eq!(codec.leanvec_dim(), 64); // Default D/2

        // Create test vector
        let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

        let (meta, encoded) = codec.encode(&vector);
        let decoded = codec.decode(&meta, &encoded);

        // Check approximate equality
        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.01,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_leanvec_8x8_encode_decode() {
        let codec = LeanVecCodec::new(128, LeanVecBits::LeanVec8x8);
        assert_eq!(codec.leanvec_dim(), 64);

        let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

        let (meta, encoded) = codec.encode(&vector);
        let decoded = codec.decode(&meta, &encoded);

        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.01,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_leanvec_custom_dim() {
        let codec = LeanVecCodec::with_leanvec_dim(128, LeanVecBits::LeanVec4x8, 32);
        assert_eq!(codec.leanvec_dim(), 32);
        assert_eq!(codec.dim(), 128);
    }

    #[test]
    fn test_leanvec_memory_efficiency() {
        let dim = 128;

        // f32 uncompressed
        let f32_bytes = dim * 4; // 512 bytes

        // 4x8 LeanVec: primary(64*4/8=32) + residual(128) + meta(24)
        let leanvec_4x8 = LeanVecMeta::SIZE + LeanVecBits::LeanVec4x8.encoded_size(dim, dim / 2);

        // 8x8 LeanVec: primary(64*8/8=64) + residual(128) + meta(24)
        let leanvec_8x8 = LeanVecMeta::SIZE + LeanVecBits::LeanVec8x8.encoded_size(dim, dim / 2);

        println!("Memory for {} dimensions:", dim);
        println!("  f32:         {} bytes", f32_bytes);
        println!(
            "  LeanVec4x8:  {} bytes ({:.2}x compression)",
            leanvec_4x8,
            f32_bytes as f32 / leanvec_4x8 as f32
        );
        println!(
            "  LeanVec8x8:  {} bytes ({:.2}x compression)",
            leanvec_8x8,
            f32_bytes as f32 / leanvec_8x8 as f32
        );

        // Verify compression ratios
        assert!(leanvec_4x8 < f32_bytes / 2); // >2x compression
        assert!(leanvec_8x8 < f32_bytes / 2); // >2x compression
    }

    #[test]
    fn test_leanvec_primary_distance() {
        let codec = LeanVecCodec::new(8, LeanVecBits::LeanVec4x8);
        let vector: Vec<f32> = vec![1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0];
        let query = vector.clone();

        let (meta, encoded) = codec.encode(&vector);

        // Primary distance should be small for self-query
        let selected_dims: Vec<usize> = (0..4).collect();
        let primary_dist =
            leanvec_primary_l2_squared_4bit(&query, &encoded, &meta, &selected_dims);

        assert!(
            primary_dist < 0.1,
            "Primary self-distance should be small, got {}",
            primary_dist
        );
    }

    #[test]
    fn test_leanvec_residual_distance() {
        let codec = LeanVecCodec::new(8, LeanVecBits::LeanVec4x8);
        let vector: Vec<f32> = vec![1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0];
        let query = vector.clone();

        let (meta, encoded) = codec.encode(&vector);

        let primary_bytes = LeanVecBits::LeanVec4x8.primary_data_size(4);
        let residual_dist = leanvec_residual_l2_squared(&query, &encoded, &meta, primary_bytes);

        assert!(
            residual_dist < 0.01,
            "Residual self-distance should be very small, got {}",
            residual_dist
        );
    }

    #[test]
    fn test_leanvec_bits_config() {
        assert_eq!(LeanVecBits::LeanVec4x8.primary_bits(), 4);
        assert_eq!(LeanVecBits::LeanVec4x8.residual_bits(), 8);
        assert_eq!(LeanVecBits::LeanVec4x8.primary_levels(), 16);

        assert_eq!(LeanVecBits::LeanVec8x8.primary_bits(), 8);
        assert_eq!(LeanVecBits::LeanVec8x8.residual_bits(), 8);
        assert_eq!(LeanVecBits::LeanVec8x8.primary_levels(), 256);
    }

    #[test]
    fn test_leanvec_bits_data_sizes() {
        // 4x8: 4 bits primary, 8 bits residual
        assert_eq!(LeanVecBits::LeanVec4x8.primary_data_size(64), 32); // 64 * 4 / 8
        assert_eq!(LeanVecBits::LeanVec4x8.residual_data_size(128), 128); // 1 byte per dim

        // 8x8: 8 bits both
        assert_eq!(LeanVecBits::LeanVec8x8.primary_data_size(64), 64); // 64 * 8 / 8
        assert_eq!(LeanVecBits::LeanVec8x8.residual_data_size(128), 128);

        // Total encoded size
        assert_eq!(LeanVecBits::LeanVec4x8.encoded_size(128, 64), 32 + 128);
        assert_eq!(LeanVecBits::LeanVec8x8.encoded_size(128, 64), 64 + 128);
    }

    #[test]
    fn test_leanvec_meta_default() {
        let meta = LeanVecMeta::default();
        assert_eq!(meta.min_primary, 0.0);
        assert_eq!(meta.delta_primary, 1.0);
        assert_eq!(meta.min_residual, 0.0);
        assert_eq!(meta.delta_residual, 1.0);
        assert_eq!(meta.leanvec_dim, 0);
    }

    #[test]
    fn test_leanvec_meta_size() {
        assert_eq!(LeanVecMeta::SIZE, 24); // 4 f32 + 1 u32 + 1 padding
    }

    #[test]
    fn test_leanvec_codec_accessors() {
        let codec = LeanVecCodec::new(128, LeanVecBits::LeanVec4x8);
        assert_eq!(codec.dim(), 128);
        assert_eq!(codec.leanvec_dim(), 64); // Default D/2
        assert_eq!(codec.bits(), LeanVecBits::LeanVec4x8);
    }

    #[test]
    fn test_leanvec_codec_sizes() {
        let codec = LeanVecCodec::new(128, LeanVecBits::LeanVec4x8);
        // 4x8 with D=128, leanvec_dim=64:
        // primary = 64*4/8 = 32 bytes, residual = 128 bytes
        assert_eq!(codec.encoded_size(), 32 + 128);
        assert_eq!(codec.total_size(), LeanVecMeta::SIZE + 32 + 128);
    }

    #[test]
    fn test_leanvec_custom_leanvec_dim() {
        // Test various custom reduced dimensions
        let codec = LeanVecCodec::with_leanvec_dim(128, LeanVecBits::LeanVec4x8, 32);
        assert_eq!(codec.leanvec_dim(), 32);

        // 0 should default to dim/2
        let codec_default = LeanVecCodec::with_leanvec_dim(128, LeanVecBits::LeanVec4x8, 0);
        assert_eq!(codec_default.leanvec_dim(), 64);

        // Value > dim should be clamped to dim
        let codec_clamped = LeanVecCodec::with_leanvec_dim(128, LeanVecBits::LeanVec4x8, 256);
        assert_eq!(codec_clamped.leanvec_dim(), 128);
    }

    #[test]
    fn test_leanvec_4x8_negative_values() {
        let codec = LeanVecCodec::new(8, LeanVecBits::LeanVec4x8);
        let vector: Vec<f32> = vec![-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, -0.1, -0.9];

        let (meta, encoded) = codec.encode(&vector);
        let decoded = codec.decode(&meta, &encoded);

        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.02,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_leanvec_8x8_negative_values() {
        let codec = LeanVecCodec::new(8, LeanVecBits::LeanVec8x8);
        let vector: Vec<f32> = vec![-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, -0.1, -0.9];

        let (meta, encoded) = codec.encode(&vector);
        let decoded = codec.decode(&meta, &encoded);

        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.02,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_leanvec_residual_inner_product() {
        let codec = LeanVecCodec::new(8, LeanVecBits::LeanVec4x8);
        let stored: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let query: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let (meta, encoded) = codec.encode(&stored);

        let primary_bytes = LeanVecBits::LeanVec4x8.primary_data_size(4);
        let ip = leanvec_residual_inner_product(&query, &encoded, &meta, primary_bytes);

        // IP = 1+2+3+4+5+6+7+8 = 36
        assert!(
            (ip - 36.0).abs() < 2.0,
            "IP should be ~36, got {}",
            ip
        );
    }

    #[test]
    fn test_leanvec_8x8_primary_distance() {
        let codec = LeanVecCodec::new(8, LeanVecBits::LeanVec8x8);
        let vector: Vec<f32> = vec![1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0];
        let query = vector.clone();

        let (meta, encoded) = codec.encode(&vector);

        let selected_dims: Vec<usize> = (0..4).collect();
        let primary_dist =
            leanvec_primary_l2_squared_8bit(&query, &encoded, &meta, &selected_dims);

        assert!(
            primary_dist < 0.1,
            "Primary self-distance should be small, got {}",
            primary_dist
        );
    }

    #[test]
    fn test_leanvec_distance_ordering() {
        let codec = LeanVecCodec::new(8, LeanVecBits::LeanVec4x8);

        let v1 = vec![1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.25, 0.0];
        let query = vec![1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0];

        let (meta1, enc1) = codec.encode(&v1);
        let (meta2, enc2) = codec.encode(&v2);
        let (meta3, enc3) = codec.encode(&v3);

        let primary_bytes = LeanVecBits::LeanVec4x8.primary_data_size(4);

        let d1 = leanvec_residual_l2_squared(&query, &enc1, &meta1, primary_bytes);
        let d2 = leanvec_residual_l2_squared(&query, &enc2, &meta2, primary_bytes);
        let d3 = leanvec_residual_l2_squared(&query, &enc3, &meta3, primary_bytes);

        // d1 should be smallest (self), d3 should be largest
        assert!(d1 < d2, "d1={} should be < d2={}", d1, d2);
        assert!(d2 < d3, "d2={} should be < d3={}", d2, d3);
    }

    #[test]
    fn test_leanvec_large_vectors() {
        let codec = LeanVecCodec::new(512, LeanVecBits::LeanVec4x8);
        let vector: Vec<f32> = (0..512).map(|i| ((i as f32) / 512.0) - 0.5).collect();

        let (meta, encoded) = codec.encode(&vector);
        let decoded = codec.decode(&meta, &encoded);

        let max_error = vector
            .iter()
            .zip(decoded.iter())
            .map(|(orig, dec)| (orig - dec).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 0.02,
            "Large vector should decode well, max_error={}",
            max_error
        );
    }

    #[test]
    fn test_leanvec_zero_vector() {
        let codec = LeanVecCodec::new(8, LeanVecBits::LeanVec4x8);
        let vector: Vec<f32> = vec![0.0; 8];

        let (meta, encoded) = codec.encode(&vector);
        let decoded = codec.decode(&meta, &encoded);

        for dec in decoded.iter() {
            assert!(dec.abs() < 0.001, "expected ~0, got {}", dec);
        }
    }

    #[test]
    fn test_leanvec_uniform_vector() {
        let codec = LeanVecCodec::new(8, LeanVecBits::LeanVec4x8);
        let vector: Vec<f32> = vec![0.5; 8];

        let (meta, encoded) = codec.encode(&vector);
        let decoded = codec.decode(&meta, &encoded);

        for dec in decoded.iter() {
            assert!(
                (dec - 0.5).abs() < 0.01,
                "expected ~0.5, got {}",
                dec
            );
        }
    }

    #[test]
    fn test_leanvec_4x8_vs_8x8_precision() {
        let codec_4x8 = LeanVecCodec::new(64, LeanVecBits::LeanVec4x8);
        let codec_8x8 = LeanVecCodec::new(64, LeanVecBits::LeanVec8x8);
        let vector: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();

        let (_, encoded_4x8) = codec_4x8.encode(&vector);
        let (_, encoded_8x8) = codec_8x8.encode(&vector);

        // 8x8 should use more bytes for primary (8-bit vs 4-bit)
        // 4x8: primary = 32 * 4 / 8 = 16 bytes
        // 8x8: primary = 32 * 8 / 8 = 32 bytes
        assert!(
            encoded_8x8.len() > encoded_4x8.len(),
            "8x8 should use more bytes: 4x8={}, 8x8={}",
            encoded_4x8.len(),
            encoded_8x8.len()
        );
    }

    #[test]
    fn test_leanvec_custom_dim_selection() {
        let mut codec = LeanVecCodec::new(8, LeanVecBits::LeanVec4x8);

        // Set custom dimension selection (use last 4 dims instead of first 4)
        let custom_dims: Vec<usize> = vec![4, 5, 6, 7, 0, 1, 2, 3];
        codec.set_selected_dims(custom_dims);

        let vector: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.25, 0.125];

        let (meta, encoded) = codec.encode(&vector);

        // The primary quantization should now work on dims 4-7
        // Primary distance should still work with the selected dims
        let selected_dims: Vec<usize> = vec![4, 5, 6, 7];
        let dist = leanvec_primary_l2_squared_4bit(&vector, &encoded, &meta, &selected_dims);

        // Self-distance should be small
        assert!(
            dist < 0.5,
            "Self distance should be small, got {}",
            dist
        );
    }

    #[test]
    fn test_leanvec_odd_dimension() {
        // LeanVec with odd dimension
        let codec = LeanVecCodec::new(7, LeanVecBits::LeanVec4x8);
        assert_eq!(codec.leanvec_dim(), 3); // 7/2 = 3

        let vector: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];

        let (meta, encoded) = codec.encode(&vector);
        let decoded = codec.decode(&meta, &encoded);

        assert_eq!(decoded.len(), 7);

        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.02,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_leanvec_primary_4bit_packing() {
        let codec = LeanVecCodec::new(8, LeanVecBits::LeanVec4x8);
        let vector: Vec<f32> = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.9, 0.8, 0.7];

        let (_meta, encoded) = codec.encode(&vector);

        // Primary data should be (leanvec_dim + 1) / 2 = 2 bytes (4 dims * 4 bits / 8)
        let primary_bytes = LeanVecBits::LeanVec4x8.primary_data_size(4);
        assert_eq!(primary_bytes, 2);

        // Residual data should be 8 bytes
        let residual_bytes = LeanVecBits::LeanVec4x8.residual_data_size(8);
        assert_eq!(residual_bytes, 8);

        // Total encoded
        assert_eq!(encoded.len(), primary_bytes + residual_bytes);
    }

    #[test]
    fn test_leanvec_meta_debug() {
        let meta = LeanVecMeta {
            min_primary: 1.0,
            delta_primary: 0.5,
            min_residual: -1.0,
            delta_residual: 0.25,
            leanvec_dim: 32,
            _pad: 0,
        };

        let debug_str = format!("{:?}", meta);
        assert!(debug_str.contains("LeanVecMeta"));
        assert!(debug_str.contains("min_primary"));
        assert!(debug_str.contains("leanvec_dim"));
    }

    #[test]
    fn test_leanvec_generic_encode_decode_dispatch() {
        // Test that generic encode/decode correctly dispatches
        for bits in [LeanVecBits::LeanVec4x8, LeanVecBits::LeanVec8x8] {
            let codec = LeanVecCodec::new(16, bits);
            let vector: Vec<f32> = (0..16).map(|i| (i as f32) / 16.0).collect();

            let (meta, encoded) = codec.encode(&vector);
            let decoded = codec.decode(&meta, &encoded);

            assert_eq!(decoded.len(), 16);

            for (orig, dec) in vector.iter().zip(decoded.iter()) {
                assert!(
                    (orig - dec).abs() < 0.02,
                    "bits={:?}: orig={}, dec={}",
                    bits,
                    orig,
                    dec
                );
            }
        }
    }

    #[test]
    fn test_leanvec_primary_reduces_dimension() {
        let codec = LeanVecCodec::new(128, LeanVecBits::LeanVec4x8);

        // The primary quantization operates on leanvec_dim=64
        let selected_dims: Vec<usize> = (0..64).collect();

        let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let (meta, encoded) = codec.encode(&vector);

        // Primary distance only uses the first leanvec_dim dimensions
        let dist = leanvec_primary_l2_squared_4bit(&vector, &encoded, &meta, &selected_dims);

        // Should be able to compute distance without error
        assert!(dist.is_finite(), "Distance should be finite");
        assert!(dist >= 0.0, "Distance should be non-negative");
    }
}

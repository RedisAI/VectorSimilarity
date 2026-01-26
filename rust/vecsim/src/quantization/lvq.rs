//! Learned Vector Quantization (LVQ) support.
//!
//! LVQ provides memory-efficient vector storage using learned quantization:
//! - Primary quantization: 4-bit or 8-bit per dimension
//! - Optional residual quantization: additional bits for error correction
//! - Per-vector scaling factors for optimal range utilization
//!
//! ## Two-Level LVQ (LVQ4x4, LVQ8x8)
//!
//! Two-level quantization stores:
//! 1. Primary quantized values (e.g., 4 bits)
//! 2. Residual error quantized values (e.g., 4 bits)
//!
//! This provides better accuracy than single-level quantization while
//! maintaining memory efficiency.
//!
//! ## Memory Layout
//!
//! For LVQ4x4 (4-bit primary + 4-bit residual):
//! - Meta: min_primary, delta_primary, min_residual, delta_residual (16 bytes)
//! - Primary data: dim/2 bytes (4 bits packed)
//! - Residual data: dim/2 bytes (4 bits packed)
//! - Total: 16 + dim bytes per vector

/// Number of quantization levels for 4-bit (16 levels).
pub const LEVELS_4BIT: usize = 16;

/// Number of quantization levels for 8-bit (256 levels).
pub const LEVELS_8BIT: usize = 256;

/// LVQ quantization bits configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LvqBits {
    /// 4-bit primary quantization.
    Lvq4,
    /// 8-bit primary quantization.
    Lvq8,
    /// 4-bit primary + 4-bit residual (8 bits total).
    Lvq4x4,
    /// 8-bit primary + 8-bit residual (16 bits total).
    Lvq8x8,
}

impl LvqBits {
    /// Get primary bits.
    pub fn primary_bits(&self) -> usize {
        match self {
            LvqBits::Lvq4 | LvqBits::Lvq4x4 => 4,
            LvqBits::Lvq8 | LvqBits::Lvq8x8 => 8,
        }
    }

    /// Get residual bits (0 if single-level).
    pub fn residual_bits(&self) -> usize {
        match self {
            LvqBits::Lvq4 | LvqBits::Lvq8 => 0,
            LvqBits::Lvq4x4 => 4,
            LvqBits::Lvq8x8 => 8,
        }
    }

    /// Check if two-level quantization.
    pub fn is_two_level(&self) -> bool {
        matches!(self, LvqBits::Lvq4x4 | LvqBits::Lvq8x8)
    }

    /// Get total bits per dimension.
    pub fn total_bits(&self) -> usize {
        self.primary_bits() + self.residual_bits()
    }

    /// Get bytes per vector (excluding metadata).
    pub fn data_bytes(&self, dim: usize) -> usize {
        let total_bits = self.total_bits() * dim;
        (total_bits + 7) / 8
    }
}

/// Metadata for a quantized vector.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LvqVectorMeta {
    /// Minimum value for primary quantization.
    pub min_primary: f32,
    /// Scale factor for primary quantization.
    pub delta_primary: f32,
    /// Minimum value for residual quantization (0 if single-level).
    pub min_residual: f32,
    /// Scale factor for residual quantization (0 if single-level).
    pub delta_residual: f32,
}

impl Default for LvqVectorMeta {
    fn default() -> Self {
        Self {
            min_primary: 0.0,
            delta_primary: 1.0,
            min_residual: 0.0,
            delta_residual: 0.0,
        }
    }
}

impl LvqVectorMeta {
    /// Size of metadata in bytes.
    pub const SIZE: usize = 16;
}

/// LVQ codec for encoding and decoding vectors.
pub struct LvqCodec {
    dim: usize,
    bits: LvqBits,
}

impl LvqCodec {
    /// Create a new LVQ codec.
    pub fn new(dim: usize, bits: LvqBits) -> Self {
        Self { dim, bits }
    }

    /// Get the dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the bits configuration.
    pub fn bits(&self) -> LvqBits {
        self.bits
    }

    /// Get the size of encoded data in bytes (excluding metadata).
    pub fn encoded_size(&self) -> usize {
        self.bits.data_bytes(self.dim)
    }

    /// Get total size including metadata.
    pub fn total_size(&self) -> usize {
        LvqVectorMeta::SIZE + self.encoded_size()
    }

    /// Encode a vector using LVQ4 (4-bit single-level).
    pub fn encode_lvq4(&self, vector: &[f32]) -> (LvqVectorMeta, Vec<u8>) {
        debug_assert_eq!(vector.len(), self.dim);

        // Find min and max
        let (min_val, max_val) = vector
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

        let range = max_val - min_val;
        let delta = if range > 1e-10 {
            range / 15.0 // 4-bit = 16 levels (0-15)
        } else {
            1.0
        };
        let inv_delta = 1.0 / delta;

        // Encode to 4-bit (pack 2 values per byte)
        let mut encoded = vec![0u8; (self.dim + 1) / 2];

        for i in 0..self.dim {
            let normalized = (vector[i] - min_val) * inv_delta;
            let quantized = (normalized.round() as u8).min(15);

            let byte_idx = i / 2;
            if i % 2 == 0 {
                encoded[byte_idx] |= quantized;
            } else {
                encoded[byte_idx] |= quantized << 4;
            }
        }

        let meta = LvqVectorMeta {
            min_primary: min_val,
            delta_primary: delta,
            min_residual: 0.0,
            delta_residual: 0.0,
        };

        (meta, encoded)
    }

    /// Decode an LVQ4 encoded vector.
    pub fn decode_lvq4(&self, meta: &LvqVectorMeta, encoded: &[u8]) -> Vec<f32> {
        let mut decoded = Vec::with_capacity(self.dim);

        for i in 0..self.dim {
            let byte_idx = i / 2;
            let quantized = if i % 2 == 0 {
                encoded[byte_idx] & 0x0F
            } else {
                (encoded[byte_idx] >> 4) & 0x0F
            };

            let value = meta.min_primary + (quantized as f32) * meta.delta_primary;
            decoded.push(value);
        }

        decoded
    }

    /// Encode a vector using LVQ4x4 (4-bit primary + 4-bit residual).
    pub fn encode_lvq4x4(&self, vector: &[f32]) -> (LvqVectorMeta, Vec<u8>) {
        debug_assert_eq!(vector.len(), self.dim);

        // First pass: primary quantization
        let (min_primary, max_primary) = vector
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

        let range_primary = max_primary - min_primary;
        let delta_primary = if range_primary > 1e-10 {
            range_primary / 15.0
        } else {
            1.0
        };
        let inv_delta_primary = 1.0 / delta_primary;

        // Compute primary quantization and residuals
        let mut primary_quantized = vec![0u8; self.dim];
        let mut residuals = vec![0.0f32; self.dim];

        for i in 0..self.dim {
            let normalized = (vector[i] - min_primary) * inv_delta_primary;
            let q = (normalized.round() as u8).min(15);
            primary_quantized[i] = q;

            // Compute residual (original - reconstructed)
            let reconstructed = min_primary + (q as f32) * delta_primary;
            residuals[i] = vector[i] - reconstructed;
        }

        // Second pass: residual quantization
        let (min_residual, max_residual) = residuals
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

        let range_residual = max_residual - min_residual;
        let delta_residual = if range_residual > 1e-10 {
            range_residual / 15.0
        } else {
            1.0
        };
        let inv_delta_residual = 1.0 / delta_residual;

        // Encode both primary and residual (interleaved: primary in low nibble, residual in high)
        let mut encoded = vec![0u8; self.dim];

        for i in 0..self.dim {
            let residual_normalized = (residuals[i] - min_residual) * inv_delta_residual;
            let residual_q = (residual_normalized.round() as u8).min(15);

            // Pack primary (low nibble) and residual (high nibble)
            encoded[i] = primary_quantized[i] | (residual_q << 4);
        }

        let meta = LvqVectorMeta {
            min_primary,
            delta_primary,
            min_residual,
            delta_residual,
        };

        (meta, encoded)
    }

    /// Decode an LVQ4x4 encoded vector.
    pub fn decode_lvq4x4(&self, meta: &LvqVectorMeta, encoded: &[u8]) -> Vec<f32> {
        let mut decoded = Vec::with_capacity(self.dim);

        for i in 0..self.dim {
            let primary_q = encoded[i] & 0x0F;
            let residual_q = (encoded[i] >> 4) & 0x0F;

            let primary_value = meta.min_primary + (primary_q as f32) * meta.delta_primary;
            let residual_value = meta.min_residual + (residual_q as f32) * meta.delta_residual;

            decoded.push(primary_value + residual_value);
        }

        decoded
    }

    /// Encode a vector using LVQ8 (8-bit single-level).
    pub fn encode_lvq8(&self, vector: &[f32]) -> (LvqVectorMeta, Vec<u8>) {
        debug_assert_eq!(vector.len(), self.dim);

        let (min_val, max_val) = vector
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));

        let range = max_val - min_val;
        let delta = if range > 1e-10 {
            range / 255.0 // 8-bit = 256 levels (0-255)
        } else {
            1.0
        };
        let inv_delta = 1.0 / delta;

        let mut encoded = vec![0u8; self.dim];

        for i in 0..self.dim {
            let normalized = (vector[i] - min_val) * inv_delta;
            encoded[i] = (normalized.round() as u8).min(255);
        }

        let meta = LvqVectorMeta {
            min_primary: min_val,
            delta_primary: delta,
            min_residual: 0.0,
            delta_residual: 0.0,
        };

        (meta, encoded)
    }

    /// Decode an LVQ8 encoded vector.
    pub fn decode_lvq8(&self, meta: &LvqVectorMeta, encoded: &[u8]) -> Vec<f32> {
        let mut decoded = Vec::with_capacity(self.dim);

        for i in 0..self.dim {
            let value = meta.min_primary + (encoded[i] as f32) * meta.delta_primary;
            decoded.push(value);
        }

        decoded
    }

    /// Encode a vector based on configured bits.
    pub fn encode(&self, vector: &[f32]) -> (LvqVectorMeta, Vec<u8>) {
        match self.bits {
            LvqBits::Lvq4 => self.encode_lvq4(vector),
            LvqBits::Lvq4x4 => self.encode_lvq4x4(vector),
            LvqBits::Lvq8 => self.encode_lvq8(vector),
            LvqBits::Lvq8x8 => {
                // LVQ8x8 not implemented yet, fall back to LVQ8
                self.encode_lvq8(vector)
            }
        }
    }

    /// Decode a vector based on configured bits.
    pub fn decode(&self, meta: &LvqVectorMeta, encoded: &[u8]) -> Vec<f32> {
        match self.bits {
            LvqBits::Lvq4 => self.decode_lvq4(meta, encoded),
            LvqBits::Lvq4x4 => self.decode_lvq4x4(meta, encoded),
            LvqBits::Lvq8 => self.decode_lvq8(meta, encoded),
            LvqBits::Lvq8x8 => self.decode_lvq8(meta, encoded),
        }
    }
}

/// Compute asymmetric L2 squared distance between f32 query and LVQ4 quantized vector.
#[inline]
pub fn lvq4_asymmetric_l2_squared(
    query: &[f32],
    quantized: &[u8],
    meta: &LvqVectorMeta,
    dim: usize,
) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..dim {
        let byte_idx = i / 2;
        let q = if i % 2 == 0 {
            quantized[byte_idx] & 0x0F
        } else {
            (quantized[byte_idx] >> 4) & 0x0F
        };

        let stored = meta.min_primary + (q as f32) * meta.delta_primary;
        let diff = query[i] - stored;
        sum += diff * diff;
    }

    sum
}

/// Compute asymmetric L2 squared distance between f32 query and LVQ4x4 quantized vector.
#[inline]
pub fn lvq4x4_asymmetric_l2_squared(
    query: &[f32],
    quantized: &[u8],
    meta: &LvqVectorMeta,
    dim: usize,
) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..dim {
        let primary_q = quantized[i] & 0x0F;
        let residual_q = (quantized[i] >> 4) & 0x0F;

        let primary_value = meta.min_primary + (primary_q as f32) * meta.delta_primary;
        let residual_value = meta.min_residual + (residual_q as f32) * meta.delta_residual;
        let stored = primary_value + residual_value;

        let diff = query[i] - stored;
        sum += diff * diff;
    }

    sum
}

/// Compute asymmetric inner product between f32 query and LVQ4x4 quantized vector.
#[inline]
pub fn lvq4x4_asymmetric_inner_product(
    query: &[f32],
    quantized: &[u8],
    meta: &LvqVectorMeta,
    dim: usize,
) -> f32 {
    let mut sum = 0.0f32;

    for i in 0..dim {
        let primary_q = quantized[i] & 0x0F;
        let residual_q = (quantized[i] >> 4) & 0x0F;

        let primary_value = meta.min_primary + (primary_q as f32) * meta.delta_primary;
        let residual_value = meta.min_residual + (residual_q as f32) * meta.delta_residual;
        let stored = primary_value + residual_value;

        sum += query[i] * stored;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lvq4_encode_decode() {
        let codec = LvqCodec::new(8, LvqBits::Lvq4);
        let vector: Vec<f32> = vec![0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4];

        let (meta, encoded) = codec.encode_lvq4(&vector);
        let decoded = codec.decode_lvq4(&meta, &encoded);

        // Check approximate equality (4-bit has limited precision)
        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.1,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_lvq4x4_encode_decode() {
        let codec = LvqCodec::new(8, LvqBits::Lvq4x4);
        let vector: Vec<f32> = vec![0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4];

        let (meta, encoded) = codec.encode_lvq4x4(&vector);
        let decoded = codec.decode_lvq4x4(&meta, &encoded);

        // Two-level should have better precision
        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.05,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_lvq8_encode_decode() {
        let codec = LvqCodec::new(8, LvqBits::Lvq8);
        let vector: Vec<f32> = vec![0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4];

        let (meta, encoded) = codec.encode_lvq8(&vector);
        let decoded = codec.decode_lvq8(&meta, &encoded);

        // 8-bit should be very accurate
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
    fn test_lvq4_asymmetric_distance() {
        let codec = LvqCodec::new(4, LvqBits::Lvq4);
        let stored = vec![1.0, 0.0, 0.0, 0.0];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let (meta, encoded) = codec.encode_lvq4(&stored);
        let dist = lvq4_asymmetric_l2_squared(&query, &encoded, &meta, 4);

        // Self-distance should be near 0
        assert!(dist < 0.1, "Self distance should be near 0, got {}", dist);
    }

    #[test]
    fn test_lvq4x4_asymmetric_distance() {
        let codec = LvqCodec::new(4, LvqBits::Lvq4x4);
        let stored = vec![1.0, 0.0, 0.0, 0.0];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let (meta, encoded) = codec.encode_lvq4x4(&stored);
        let dist = lvq4x4_asymmetric_l2_squared(&query, &encoded, &meta, 4);

        // Self-distance should be very near 0 with two-level
        assert!(dist < 0.01, "Self distance should be near 0, got {}", dist);
    }

    #[test]
    fn test_lvq_memory_savings() {
        let dim = 128;

        // f32 storage
        let f32_bytes = dim * 4;

        // LVQ4 storage (4 bits per dim + meta)
        let lvq4_bytes = (dim + 1) / 2 + LvqVectorMeta::SIZE;

        // LVQ4x4 storage (8 bits per dim + meta)
        let lvq4x4_bytes = dim + LvqVectorMeta::SIZE;

        // LVQ8 storage (8 bits per dim + meta)
        let lvq8_bytes = dim + LvqVectorMeta::SIZE;

        println!("Memory for {} dimensions:", dim);
        println!("  f32:    {} bytes", f32_bytes);
        println!("  LVQ4:   {} bytes ({}x compression)", lvq4_bytes, f32_bytes as f32 / lvq4_bytes as f32);
        println!("  LVQ4x4: {} bytes ({}x compression)", lvq4x4_bytes, f32_bytes as f32 / lvq4x4_bytes as f32);
        println!("  LVQ8:   {} bytes ({}x compression)", lvq8_bytes, f32_bytes as f32 / lvq8_bytes as f32);

        // Verify compression ratios
        assert!(lvq4_bytes < f32_bytes / 4); // >4x compression
        assert!(lvq4x4_bytes < f32_bytes / 2); // >2x compression
    }

    #[test]
    fn test_lvq_bits_config() {
        assert_eq!(LvqBits::Lvq4.primary_bits(), 4);
        assert_eq!(LvqBits::Lvq4.residual_bits(), 0);
        assert!(!LvqBits::Lvq4.is_two_level());

        assert_eq!(LvqBits::Lvq4x4.primary_bits(), 4);
        assert_eq!(LvqBits::Lvq4x4.residual_bits(), 4);
        assert!(LvqBits::Lvq4x4.is_two_level());

        assert_eq!(LvqBits::Lvq8.primary_bits(), 8);
        assert_eq!(LvqBits::Lvq8.residual_bits(), 0);
    }

    #[test]
    fn test_lvq_bits_total_bits() {
        assert_eq!(LvqBits::Lvq4.total_bits(), 4);
        assert_eq!(LvqBits::Lvq8.total_bits(), 8);
        assert_eq!(LvqBits::Lvq4x4.total_bits(), 8);
        assert_eq!(LvqBits::Lvq8x8.total_bits(), 16);
    }

    #[test]
    fn test_lvq_bits_data_bytes() {
        // LVQ4: 4 bits per dim, so for 8 dims = 32 bits = 4 bytes
        assert_eq!(LvqBits::Lvq4.data_bytes(8), 4);
        // Odd dimension: 9 dims = 36 bits = 5 bytes (rounded up)
        assert_eq!(LvqBits::Lvq4.data_bytes(9), 5);

        // LVQ8: 8 bits per dim = 1 byte per dim
        assert_eq!(LvqBits::Lvq8.data_bytes(8), 8);

        // LVQ4x4: 8 bits total per dim
        assert_eq!(LvqBits::Lvq4x4.data_bytes(8), 8);
    }

    #[test]
    fn test_lvq4_negative_values() {
        let codec = LvqCodec::new(8, LvqBits::Lvq4);
        let vector: Vec<f32> = vec![-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, -0.1, -0.9];

        let (meta, encoded) = codec.encode_lvq4(&vector);
        let decoded = codec.decode_lvq4(&meta, &encoded);

        // 4-bit has limited precision but should be reasonably close
        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.15,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_lvq4_odd_dimension() {
        // Test odd dimension where 4-bit packing doesn't align perfectly
        let codec = LvqCodec::new(7, LvqBits::Lvq4);
        let vector: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];

        let (meta, encoded) = codec.encode_lvq4(&vector);
        // Encoded should be (7 + 1) / 2 = 4 bytes
        assert_eq!(encoded.len(), 4);

        let decoded = codec.decode_lvq4(&meta, &encoded);
        assert_eq!(decoded.len(), 7);

        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.1,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_lvq4_uniform_vector() {
        let codec = LvqCodec::new(8, LvqBits::Lvq4);
        let vector: Vec<f32> = vec![0.5; 8];

        let (meta, encoded) = codec.encode_lvq4(&vector);
        let decoded = codec.decode_lvq4(&meta, &encoded);

        // All values should decode to approximately the same
        for dec in decoded.iter() {
            assert!(
                (dec - 0.5).abs() < 0.01,
                "expected ~0.5, got {}",
                dec
            );
        }

        // All encoded values should be 0 (since all at min)
        for byte in encoded.iter() {
            assert_eq!(*byte, 0, "uniform vector should encode to 0");
        }
    }

    #[test]
    fn test_lvq8_high_precision() {
        let codec = LvqCodec::new(128, LvqBits::Lvq8);
        let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

        let (meta, encoded) = codec.encode_lvq8(&vector);
        let decoded = codec.decode_lvq8(&meta, &encoded);

        // 8-bit should have very good precision
        let max_error = vector
            .iter()
            .zip(decoded.iter())
            .map(|(orig, dec)| (orig - dec).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 0.005,
            "8-bit should have high precision, max_error={}",
            max_error
        );
    }

    #[test]
    fn test_lvq4x4_negative_values() {
        let codec = LvqCodec::new(8, LvqBits::Lvq4x4);
        let vector: Vec<f32> = vec![-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, -0.1, -0.9];

        let (meta, encoded) = codec.encode_lvq4x4(&vector);
        let decoded = codec.decode_lvq4x4(&meta, &encoded);

        // Two-level should be better than single-level
        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.1,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_lvq4x4_residual_improves_accuracy() {
        let codec4 = LvqCodec::new(8, LvqBits::Lvq4);
        let codec4x4 = LvqCodec::new(8, LvqBits::Lvq4x4);
        let vector: Vec<f32> = vec![0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4];

        let (meta4, encoded4) = codec4.encode_lvq4(&vector);
        let decoded4 = codec4.decode_lvq4(&meta4, &encoded4);

        let (meta4x4, encoded4x4) = codec4x4.encode_lvq4x4(&vector);
        let decoded4x4 = codec4x4.decode_lvq4x4(&meta4x4, &encoded4x4);

        // Calculate MSE for both
        let mse4: f32 = vector
            .iter()
            .zip(decoded4.iter())
            .map(|(orig, dec)| (orig - dec).powi(2))
            .sum::<f32>()
            / vector.len() as f32;

        let mse4x4: f32 = vector
            .iter()
            .zip(decoded4x4.iter())
            .map(|(orig, dec)| (orig - dec).powi(2))
            .sum::<f32>()
            / vector.len() as f32;

        // Two-level quantization should have lower error
        assert!(
            mse4x4 <= mse4,
            "LVQ4x4 should have lower or equal MSE: mse4={}, mse4x4={}",
            mse4,
            mse4x4
        );
    }

    #[test]
    fn test_lvq_codec_generic_encode_decode() {
        // Test the generic encode/decode methods that dispatch based on bits
        for bits in [LvqBits::Lvq4, LvqBits::Lvq4x4, LvqBits::Lvq8] {
            let codec = LvqCodec::new(8, bits);
            let vector: Vec<f32> = vec![0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4];

            let (meta, encoded) = codec.encode(&vector);
            let decoded = codec.decode(&meta, &encoded);

            // All should reasonably decode
            for (orig, dec) in vector.iter().zip(decoded.iter()) {
                assert!(
                    (orig - dec).abs() < 0.15,
                    "bits={:?}: orig={}, dec={}",
                    bits,
                    orig,
                    dec
                );
            }
        }
    }

    #[test]
    fn test_lvq4_distance_ordering() {
        let codec = LvqCodec::new(4, LvqBits::Lvq4);

        // Three stored vectors
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.5, 0.5, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 0.0, 1.0];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let (meta1, enc1) = codec.encode_lvq4(&v1);
        let (meta2, enc2) = codec.encode_lvq4(&v2);
        let (meta3, enc3) = codec.encode_lvq4(&v3);

        let d1 = lvq4_asymmetric_l2_squared(&query, &enc1, &meta1, 4);
        let d2 = lvq4_asymmetric_l2_squared(&query, &enc2, &meta2, 4);
        let d3 = lvq4_asymmetric_l2_squared(&query, &enc3, &meta3, 4);

        // d1 should be smallest (self), d3 should be largest
        assert!(d1 < d2, "d1={} should be < d2={}", d1, d2);
        assert!(d2 < d3, "d2={} should be < d3={}", d2, d3);
    }

    #[test]
    fn test_lvq4x4_distance_ordering() {
        let codec = LvqCodec::new(4, LvqBits::Lvq4x4);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.5, 0.5, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 0.0, 1.0];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let (meta1, enc1) = codec.encode_lvq4x4(&v1);
        let (meta2, enc2) = codec.encode_lvq4x4(&v2);
        let (meta3, enc3) = codec.encode_lvq4x4(&v3);

        let d1 = lvq4x4_asymmetric_l2_squared(&query, &enc1, &meta1, 4);
        let d2 = lvq4x4_asymmetric_l2_squared(&query, &enc2, &meta2, 4);
        let d3 = lvq4x4_asymmetric_l2_squared(&query, &enc3, &meta3, 4);

        assert!(d1 < d2, "d1={} should be < d2={}", d1, d2);
        assert!(d2 < d3, "d2={} should be < d3={}", d2, d3);
    }

    #[test]
    fn test_lvq4x4_inner_product() {
        let codec = LvqCodec::new(4, LvqBits::Lvq4x4);

        let stored = vec![1.0, 2.0, 3.0, 4.0];
        let query = vec![1.0, 1.0, 1.0, 1.0];

        let (meta, encoded) = codec.encode_lvq4x4(&stored);

        // IP = 1*1 + 1*2 + 1*3 + 1*4 = 10
        let ip = lvq4x4_asymmetric_inner_product(&query, &encoded, &meta, 4);

        assert!(
            (ip - 10.0).abs() < 1.0,
            "IP should be ~10, got {}",
            ip
        );
    }

    #[test]
    fn test_lvq_vector_meta_default() {
        let meta = LvqVectorMeta::default();
        assert_eq!(meta.min_primary, 0.0);
        assert_eq!(meta.delta_primary, 1.0);
        assert_eq!(meta.min_residual, 0.0);
        assert_eq!(meta.delta_residual, 0.0);
    }

    #[test]
    fn test_lvq_vector_meta_size() {
        assert_eq!(LvqVectorMeta::SIZE, 16); // 4 f32 values
    }

    #[test]
    fn test_lvq_codec_accessors() {
        let codec = LvqCodec::new(64, LvqBits::Lvq4x4);
        assert_eq!(codec.dim(), 64);
        assert_eq!(codec.bits(), LvqBits::Lvq4x4);
        assert_eq!(codec.encoded_size(), 64); // 8 bits per dim for 4x4
        assert_eq!(codec.total_size(), LvqVectorMeta::SIZE + 64);
    }

    #[test]
    fn test_lvq4_zero_vector() {
        let codec = LvqCodec::new(8, LvqBits::Lvq4);
        let vector: Vec<f32> = vec![0.0; 8];

        let (meta, encoded) = codec.encode_lvq4(&vector);
        let decoded = codec.decode_lvq4(&meta, &encoded);

        for dec in decoded.iter() {
            assert!(dec.abs() < 0.001, "expected ~0, got {}", dec);
        }
    }

    #[test]
    fn test_lvq8_negative_values() {
        let codec = LvqCodec::new(8, LvqBits::Lvq8);
        let vector: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0, -0.25, 0.25, 0.75];

        let (meta, encoded) = codec.encode_lvq8(&vector);
        let decoded = codec.decode_lvq8(&meta, &encoded);

        // 8-bit should have very good precision
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
    fn test_lvq4x4_large_vectors() {
        let codec = LvqCodec::new(512, LvqBits::Lvq4x4);
        let vector: Vec<f32> = (0..512).map(|i| ((i as f32) / 512.0) - 0.5).collect();

        let (meta, encoded) = codec.encode_lvq4x4(&vector);
        let decoded = codec.decode_lvq4x4(&meta, &encoded);

        let max_error = vector
            .iter()
            .zip(decoded.iter())
            .map(|(orig, dec)| (orig - dec).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 0.05,
            "Large vector should decode well, max_error={}",
            max_error
        );
    }

    #[test]
    fn test_lvq4_packed_nibble_roundtrip() {
        let codec = LvqCodec::new(4, LvqBits::Lvq4);
        // Create values that should map to specific nibbles
        let vector: Vec<f32> = vec![0.0, 0.33, 0.67, 1.0];

        let (meta, encoded) = codec.encode_lvq4(&vector);

        // Check encoded size
        assert_eq!(encoded.len(), 2); // 4 values * 4 bits / 8 = 2 bytes

        // Decode and verify
        let decoded = codec.decode_lvq4(&meta, &encoded);
        assert_eq!(decoded.len(), 4);

        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.15,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_lvq_levels_constants() {
        assert_eq!(LEVELS_4BIT, 16);
        assert_eq!(LEVELS_8BIT, 256);
    }
}

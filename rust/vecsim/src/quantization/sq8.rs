//! Scalar Quantization to 8-bit unsigned integers (SQ8).
//!
//! SQ8 provides a simple yet effective compression scheme that reduces f32 vectors
//! to u8 vectors with per-vector metadata for dequantization.
//!
//! ## How it works
//!
//! For each vector:
//! 1. Find min and max values
//! 2. Compute delta = (max - min) / 255
//! 3. Quantize: q[i] = round((v[i] - min) / delta)
//! 4. Store metadata for dequantization and asymmetric distance computation
//!
//! ## Asymmetric Distance
//!
//! For L2 squared distance: ||q - v||² = ||q||² + ||v||² - 2*IP(q, v)
//! - Query vector stays in f32
//! - Stored vector is quantized
//! - Precomputed ||v||² avoids full dequantization

use crate::types::UInt8;

/// Per-vector metadata for SQ8 dequantization and asymmetric distance.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sq8VectorMeta {
    /// Minimum value in the original vector.
    pub min: f32,
    /// Scale factor: (max - min) / 255.
    pub delta: f32,
    /// Precomputed sum of squares ||v||² for asymmetric L2.
    pub sum_sq: f32,
    /// Precomputed sum of elements for asymmetric IP.
    pub sum: f32,
}

impl Sq8VectorMeta {
    /// Create new metadata.
    pub fn new(min: f32, delta: f32, sum_sq: f32, sum: f32) -> Self {
        Self {
            min,
            delta,
            sum_sq,
            sum,
        }
    }

    /// Dequantize a single value.
    #[inline(always)]
    pub fn dequantize(&self, q: u8) -> f32 {
        self.min + (q as f32) * self.delta
    }

    /// Size in bytes when serialized.
    pub const SERIALIZED_SIZE: usize = 4 * 4; // 4 f32 values
}

/// SQ8 encoder/decoder.
#[derive(Debug, Clone)]
pub struct Sq8Codec {
    dim: usize,
}

impl Sq8Codec {
    /// Create a new SQ8 codec for vectors of the given dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Get the dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Encode an f32 vector to SQ8 format.
    ///
    /// Returns the quantized vector and per-vector metadata.
    pub fn encode(&self, vector: &[f32]) -> (Vec<UInt8>, Sq8VectorMeta) {
        assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");

        // Find min and max
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;

        for &v in vector.iter() {
            min = min.min(v);
            max = max.max(v);
            sum += v;
            sum_sq += v * v;
        }

        // Handle degenerate case where all values are the same
        let delta = if (max - min).abs() < f32::EPSILON {
            1.0 // Avoid division by zero; all values will quantize to 0
        } else {
            (max - min) / 255.0
        };

        // Quantize
        let inv_delta = 1.0 / delta;
        let quantized: Vec<UInt8> = vector
            .iter()
            .map(|&v| {
                let q = ((v - min) * inv_delta).round().clamp(0.0, 255.0) as u8;
                UInt8::new(q)
            })
            .collect();

        let meta = Sq8VectorMeta::new(min, delta, sum_sq, sum);
        (quantized, meta)
    }

    /// Decode an SQ8 vector back to f32 format.
    pub fn decode(&self, quantized: &[UInt8], meta: &Sq8VectorMeta) -> Vec<f32> {
        assert_eq!(quantized.len(), self.dim, "Quantized vector dimension mismatch");

        quantized
            .iter()
            .map(|&q| meta.dequantize(q.get()))
            .collect()
    }

    /// Encode directly from u8 slice (for performance when reading raw bytes).
    pub fn encode_from_f32_slice(&self, vector: &[f32]) -> (Vec<u8>, Sq8VectorMeta) {
        let (quantized, meta) = self.encode(vector);
        let bytes: Vec<u8> = quantized.iter().map(|q| q.get()).collect();
        (bytes, meta)
    }

    /// Decode from raw u8 slice.
    pub fn decode_from_u8_slice(&self, quantized: &[u8], meta: &Sq8VectorMeta) -> Vec<f32> {
        assert_eq!(quantized.len(), self.dim, "Quantized vector dimension mismatch");

        quantized
            .iter()
            .map(|&q| meta.dequantize(q))
            .collect()
    }

    /// Compute quantization error (mean squared error) for a vector.
    pub fn quantization_error(&self, original: &[f32], quantized: &[UInt8], meta: &Sq8VectorMeta) -> f32 {
        assert_eq!(original.len(), self.dim);
        assert_eq!(quantized.len(), self.dim);

        let mse: f32 = original
            .iter()
            .zip(quantized.iter())
            .map(|(&orig, &q)| {
                let reconstructed = meta.dequantize(q.get());
                let diff = orig - reconstructed;
                diff * diff
            })
            .sum();

        mse / self.dim as f32
    }

    /// Encode a batch of vectors.
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<(Vec<UInt8>, Sq8VectorMeta)> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }
}

/// Compute asymmetric L2 squared distance between f32 query and SQ8 stored vector.
///
/// Uses the formula: ||q - v||² = ||q||² + ||v||² - 2*IP(q, v)
/// where ||v||² is precomputed in metadata.
#[inline]
pub fn sq8_asymmetric_l2_squared(
    query: &[f32],
    quantized: &[u8],
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(quantized.len(), dim);

    // Compute ||q||² and IP(q, v) in one pass
    let mut query_sq = 0.0f32;
    let mut ip = 0.0f32;

    for i in 0..dim {
        let q = query[i];
        let v = meta.min + (quantized[i] as f32) * meta.delta;
        query_sq += q * q;
        ip += q * v;
    }

    // ||q - v||² = ||q||² + ||v||² - 2*IP(q, v)
    (query_sq + meta.sum_sq - 2.0 * ip).max(0.0)
}

/// Compute asymmetric inner product between f32 query and SQ8 stored vector.
///
/// Returns negative inner product (for use as distance where lower is better).
#[inline]
pub fn sq8_asymmetric_inner_product(
    query: &[f32],
    quantized: &[u8],
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(quantized.len(), dim);

    let mut ip = 0.0f32;

    for i in 0..dim {
        let q = query[i];
        let v = meta.min + (quantized[i] as f32) * meta.delta;
        ip += q * v;
    }

    -ip // Negative for distance ordering
}

/// Compute asymmetric cosine distance between f32 query and SQ8 stored vector.
///
/// Returns 1 - cosine_similarity.
#[inline]
pub fn sq8_asymmetric_cosine(
    query: &[f32],
    quantized: &[u8],
    meta: &Sq8VectorMeta,
    dim: usize,
) -> f32 {
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(quantized.len(), dim);

    let mut query_sq = 0.0f32;
    let mut ip = 0.0f32;

    for i in 0..dim {
        let q = query[i];
        let v = meta.min + (quantized[i] as f32) * meta.delta;
        query_sq += q * q;
        ip += q * v;
    }

    let query_norm = query_sq.sqrt();
    let stored_norm = meta.sum_sq.sqrt();

    if query_norm < 1e-30 || stored_norm < 1e-30 {
        return 1.0;
    }

    let cosine_sim = (ip / (query_norm * stored_norm)).clamp(-1.0, 1.0);
    1.0 - cosine_sim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_encode_decode_roundtrip() {
        let codec = Sq8Codec::new(4);
        let original = vec![0.1, 0.5, 0.3, 0.9];

        let (quantized, meta) = codec.encode(&original);
        let decoded = codec.decode(&quantized, &meta);

        // Check that decoded values are close to original
        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.01,
                "orig={}, dec={}, diff={}",
                orig,
                dec,
                (orig - dec).abs()
            );
        }
    }

    #[test]
    fn test_sq8_uniform_vector() {
        let codec = Sq8Codec::new(4);
        let original = vec![0.5, 0.5, 0.5, 0.5];

        let (quantized, meta) = codec.encode(&original);
        let decoded = codec.decode(&quantized, &meta);

        // All values should decode to approximately the same value
        for dec in decoded.iter() {
            assert!((dec - 0.5).abs() < 0.01);
        }

        // All quantized values should be 0 (since all are at min)
        for q in quantized.iter() {
            assert_eq!(q.get(), 0);
        }
    }

    #[test]
    fn test_sq8_metadata() {
        let codec = Sq8Codec::new(4);
        let original = vec![0.0, 0.5, 1.0, 0.25];

        let (_, meta) = codec.encode(&original);

        assert_eq!(meta.min, 0.0);
        assert!((meta.delta - (1.0 / 255.0)).abs() < 0.0001);
        assert!((meta.sum - 1.75).abs() < 0.0001);
        // sum_sq = 0 + 0.25 + 1 + 0.0625 = 1.3125
        assert!((meta.sum_sq - 1.3125).abs() < 0.0001);
    }

    #[test]
    fn test_sq8_asymmetric_l2() {
        let codec = Sq8Codec::new(4);
        let stored = vec![1.0, 2.0, 3.0, 4.0];
        let query = vec![1.0, 2.0, 3.0, 4.0];

        let (quantized, meta) = codec.encode(&stored);
        let quantized_bytes: Vec<u8> = quantized.iter().map(|q| q.get()).collect();

        let dist = sq8_asymmetric_l2_squared(&query, &quantized_bytes, &meta, 4);

        // Distance to self should be very small (due to quantization error)
        assert!(dist < 0.1, "dist={}", dist);
    }

    #[test]
    fn test_sq8_asymmetric_l2_different() {
        let codec = Sq8Codec::new(4);
        let stored = vec![0.0, 0.0, 0.0, 0.0];
        let query = vec![1.0, 1.0, 1.0, 1.0];

        let (quantized, meta) = codec.encode(&stored);
        let quantized_bytes: Vec<u8> = quantized.iter().map(|q| q.get()).collect();

        let dist = sq8_asymmetric_l2_squared(&query, &quantized_bytes, &meta, 4);

        // Distance should be approximately 4 (sum of (1-0)^2 = 4)
        assert!((dist - 4.0).abs() < 0.1, "dist={}", dist);
    }

    #[test]
    fn test_sq8_asymmetric_ip() {
        let codec = Sq8Codec::new(4);
        let stored = vec![1.0, 2.0, 3.0, 4.0];
        let query = vec![1.0, 1.0, 1.0, 1.0];

        let (quantized, meta) = codec.encode(&stored);
        let quantized_bytes: Vec<u8> = quantized.iter().map(|q| q.get()).collect();

        let dist = sq8_asymmetric_inner_product(&query, &quantized_bytes, &meta, 4);

        // IP = 1*1 + 1*2 + 1*3 + 1*4 = 10, so distance = -10
        assert!((dist + 10.0).abs() < 0.5, "dist={}", dist);
    }

    #[test]
    fn test_sq8_asymmetric_cosine() {
        let codec = Sq8Codec::new(3);
        let stored = vec![1.0, 0.0, 0.0];
        let query = vec![1.0, 0.0, 0.0];

        let (quantized, meta) = codec.encode(&stored);
        let quantized_bytes: Vec<u8> = quantized.iter().map(|q| q.get()).collect();

        let dist = sq8_asymmetric_cosine(&query, &quantized_bytes, &meta, 3);

        // Same direction should have cosine distance near 0
        assert!(dist < 0.1, "dist={}", dist);
    }

    #[test]
    fn test_sq8_quantization_error() {
        let codec = Sq8Codec::new(4);
        let original = vec![0.1, 0.5, 0.3, 0.9];

        let (quantized, meta) = codec.encode(&original);
        let mse = codec.quantization_error(&original, &quantized, &meta);

        // MSE should be small
        assert!(mse < 0.001, "mse={}", mse);
    }

    #[test]
    fn test_sq8_negative_values() {
        let codec = Sq8Codec::new(4);
        let original = vec![-1.0, -0.5, 0.0, 0.5];

        let (quantized, meta) = codec.encode(&original);
        let decoded = codec.decode(&quantized, &meta);

        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.01,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_sq8_large_range() {
        let codec = Sq8Codec::new(4);
        let original = vec![-100.0, 0.0, 50.0, 100.0];

        let (quantized, meta) = codec.encode(&original);
        let decoded = codec.decode(&quantized, &meta);

        // With large range, quantization error will be larger
        let max_error = 200.0 / 255.0; // delta
        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() <= max_error,
                "orig={}, dec={}, max_error={}",
                orig,
                dec,
                max_error
            );
        }
    }

    #[test]
    fn test_sq8_zero_vector() {
        let codec = Sq8Codec::new(4);
        let original = vec![0.0, 0.0, 0.0, 0.0];

        let (quantized, meta) = codec.encode(&original);
        let decoded = codec.decode(&quantized, &meta);

        // All values should decode to 0
        for dec in decoded.iter() {
            assert!(dec.abs() < 0.001, "expected ~0, got {}", dec);
        }

        // Metadata should reflect zero vector
        assert!((meta.sum - 0.0).abs() < 0.001);
        assert!((meta.sum_sq - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_sq8_very_small_values() {
        let codec = Sq8Codec::new(4);
        let original = vec![1e-6, 2e-6, 3e-6, 4e-6];

        let (quantized, meta) = codec.encode(&original);
        let decoded = codec.decode(&quantized, &meta);

        // Very small values should still be preserved approximately
        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 1e-5,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_sq8_very_large_values() {
        let codec = Sq8Codec::new(4);
        let original = vec![1e6, 2e6, 3e6, 4e6];

        let (quantized, meta) = codec.encode(&original);
        let decoded = codec.decode(&quantized, &meta);

        // With large range, relative error should be reasonable
        for (orig, dec) in original.iter().zip(decoded.iter()) {
            let relative_error = (orig - dec).abs() / orig;
            assert!(
                relative_error < 0.01,
                "orig={}, dec={}, relative_error={}",
                orig,
                dec,
                relative_error
            );
        }
    }

    #[test]
    fn test_sq8_encode_from_f32_slice() {
        let codec = Sq8Codec::new(4);
        let original = vec![0.1, 0.5, 0.3, 0.9];

        let (bytes, meta) = codec.encode_from_f32_slice(&original);

        // Bytes should have the same length as dimension
        assert_eq!(bytes.len(), 4);

        // Should be able to decode back
        let decoded = codec.decode_from_u8_slice(&bytes, &meta);

        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.01,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_sq8_batch_encode() {
        let codec = Sq8Codec::new(4);
        let vectors = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![0.9, 0.8, 0.7, 0.6],
        ];

        let batch_result = codec.encode_batch(&vectors);

        assert_eq!(batch_result.len(), 3);

        // Verify each result
        for (i, (quantized, meta)) in batch_result.iter().enumerate() {
            let decoded = codec.decode(quantized, meta);
            for (orig, dec) in vectors[i].iter().zip(decoded.iter()) {
                assert!(
                    (orig - dec).abs() < 0.01,
                    "vector {}: orig={}, dec={}",
                    i,
                    orig,
                    dec
                );
            }
        }
    }

    #[test]
    fn test_sq8_asymmetric_l2_distance_ordering() {
        let codec = Sq8Codec::new(4);

        // Three stored vectors: v1 is closest to query, v3 is farthest
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.5, 0.5, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 0.0, 1.0];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let (q1, m1) = codec.encode(&v1);
        let (q2, m2) = codec.encode(&v2);
        let (q3, m3) = codec.encode(&v3);

        let b1: Vec<u8> = q1.iter().map(|q| q.get()).collect();
        let b2: Vec<u8> = q2.iter().map(|q| q.get()).collect();
        let b3: Vec<u8> = q3.iter().map(|q| q.get()).collect();

        let d1 = sq8_asymmetric_l2_squared(&query, &b1, &m1, 4);
        let d2 = sq8_asymmetric_l2_squared(&query, &b2, &m2, 4);
        let d3 = sq8_asymmetric_l2_squared(&query, &b3, &m3, 4);

        // d1 should be smallest (near 0), d3 should be largest
        assert!(d1 < d2, "d1={} should be < d2={}", d1, d2);
        assert!(d2 < d3, "d2={} should be < d3={}", d2, d3);
    }

    #[test]
    fn test_sq8_asymmetric_ip_distance_ordering() {
        let codec = Sq8Codec::new(4);

        // For inner product, higher IP means more similar
        let v1 = vec![1.0, 0.0, 0.0, 0.0]; // IP with query = 1
        let v2 = vec![0.5, 0.0, 0.0, 0.0]; // IP with query = 0.5
        let v3 = vec![0.0, 1.0, 0.0, 0.0]; // IP with query = 0
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let (q1, m1) = codec.encode(&v1);
        let (q2, m2) = codec.encode(&v2);
        let (q3, m3) = codec.encode(&v3);

        let b1: Vec<u8> = q1.iter().map(|q| q.get()).collect();
        let b2: Vec<u8> = q2.iter().map(|q| q.get()).collect();
        let b3: Vec<u8> = q3.iter().map(|q| q.get()).collect();

        // Note: asymmetric_inner_product returns negative IP
        let d1 = sq8_asymmetric_inner_product(&query, &b1, &m1, 4);
        let d2 = sq8_asymmetric_inner_product(&query, &b2, &m2, 4);
        let d3 = sq8_asymmetric_inner_product(&query, &b3, &m3, 4);

        // d1 should be most negative (closest), d3 should be near 0 (farthest)
        assert!(d1 < d2, "d1={} should be < d2={}", d1, d2);
        assert!(d2 < d3, "d2={} should be < d3={}", d2, d3);
    }

    #[test]
    fn test_sq8_asymmetric_cosine_orthogonal() {
        let codec = Sq8Codec::new(4);

        // Two orthogonal vectors
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];

        let (q1, m1) = codec.encode(&v1);
        let (q2, m2) = codec.encode(&v2);

        let b1: Vec<u8> = q1.iter().map(|q| q.get()).collect();
        let b2: Vec<u8> = q2.iter().map(|q| q.get()).collect();

        // v1 vs v2 (orthogonal) - cosine distance should be ~1
        let dist = sq8_asymmetric_cosine(&v1, &b2, &m2, 4);
        assert!(
            (dist - 1.0).abs() < 0.1,
            "Orthogonal vectors should have cosine distance ~1, got {}",
            dist
        );

        // v1 vs v1 (same direction) - cosine distance should be ~0
        let self_dist = sq8_asymmetric_cosine(&v1, &b1, &m1, 4);
        assert!(
            self_dist < 0.1,
            "Same direction should have cosine distance ~0, got {}",
            self_dist
        );
    }

    #[test]
    fn test_sq8_meta_dequantize() {
        let meta = Sq8VectorMeta::new(0.0, 0.1, 0.0, 0.0);

        // Dequantize value 0 should give min
        assert!((meta.dequantize(0) - 0.0).abs() < 0.001);

        // Dequantize value 10 should give 0 + 10 * 0.1 = 1.0
        assert!((meta.dequantize(10) - 1.0).abs() < 0.001);

        // Dequantize value 255 should give max
        assert!((meta.dequantize(255) - 25.5).abs() < 0.001);
    }

    #[test]
    fn test_sq8_meta_serialized_size() {
        // Should be 4 f32 values = 16 bytes
        assert_eq!(Sq8VectorMeta::SERIALIZED_SIZE, 16);
    }

    #[test]
    fn test_sq8_codec_dimension() {
        let codec = Sq8Codec::new(128);
        assert_eq!(codec.dimension(), 128);
    }

    #[test]
    fn test_sq8_high_dimension() {
        // Test with a high-dimension vector (common for embeddings)
        let codec = Sq8Codec::new(768);
        let original: Vec<f32> = (0..768).map(|i| (i as f32) / 768.0).collect();

        let (quantized, meta) = codec.encode(&original);
        assert_eq!(quantized.len(), 768);

        let decoded = codec.decode(&quantized, &meta);
        assert_eq!(decoded.len(), 768);

        // Check quantization error
        let mse = codec.quantization_error(&original, &quantized, &meta);
        assert!(mse < 0.001, "MSE should be small for normalized data, got {}", mse);
    }

    #[test]
    fn test_sq8_mixed_positive_negative() {
        let codec = Sq8Codec::new(4);
        let original = vec![-0.5, -0.1, 0.2, 0.7];

        let (quantized, meta) = codec.encode(&original);
        let decoded = codec.decode(&quantized, &meta);

        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.01,
                "orig={}, dec={}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_sq8_extreme_range() {
        let codec = Sq8Codec::new(4);
        // Mix of very small and very large values
        let original = vec![-1e5, 1e-5, 0.0, 1e5];

        let (quantized, meta) = codec.encode(&original);
        let decoded = codec.decode(&quantized, &meta);

        // With such extreme range, precision is limited
        let delta = meta.delta;
        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() <= delta * 1.5,
                "orig={}, dec={}, delta={}",
                orig,
                dec,
                delta
            );
        }
    }
}

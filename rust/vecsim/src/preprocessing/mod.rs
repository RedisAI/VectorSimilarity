//! Vector preprocessing pipeline.
//!
//! This module provides preprocessing transformations applied to vectors before
//! storage or querying:
//!
//! - **Normalization**: L2-normalize vectors for cosine similarity
//! - **Quantization**: Convert to quantized representation for storage
//!
//! ## Asymmetric Preprocessing
//!
//! Storage and query vectors may be processed differently:
//! - Storage vectors may be quantized to reduce memory
//! - Query vectors remain in original format with precomputed metadata
//!
//! ## Example
//!
//! ```rust,ignore
//! use vecsim::preprocessing::{PreprocessorChain, CosinePreprocessor};
//!
//! let preprocessor = CosinePreprocessor::new(128);
//! let vector = vec![1.0f32, 2.0, 3.0, 4.0];
//!
//! let processed = preprocessor.preprocess_storage(&vector);
//! ```

use crate::distance::normalize_in_place;

/// Trait for vector preprocessors.
///
/// Preprocessors transform vectors before storage or querying. They may apply
/// different transformations to storage vs query vectors (asymmetric preprocessing).
pub trait Preprocessor: Send + Sync {
    /// Preprocess a vector for storage.
    ///
    /// Returns the processed vector data as bytes. The output may be a different
    /// size or format than the input.
    fn preprocess_storage(&self, vector: &[f32]) -> Vec<u8>;

    /// Preprocess a vector for querying.
    ///
    /// Returns the processed query data as bytes. For asymmetric preprocessing,
    /// this may include precomputed metadata for faster distance computation.
    fn preprocess_query(&self, vector: &[f32]) -> Vec<u8>;

    /// Preprocess a vector in-place for storage.
    ///
    /// This modifies the vector directly when possible, avoiding allocation.
    /// Returns true if preprocessing was successful.
    fn preprocess_storage_in_place(&self, vector: &mut [f32]) -> bool;

    /// Get the storage size in bytes for a vector of given dimension.
    fn storage_size(&self, dim: usize) -> usize;

    /// Get the query size in bytes for a vector of given dimension.
    fn query_size(&self, dim: usize) -> usize;

    /// Get the name of this preprocessor.
    fn name(&self) -> &'static str;
}

/// Identity preprocessor that performs no transformation.
#[derive(Debug, Clone, Default)]
pub struct IdentityPreprocessor {
    #[allow(dead_code)]
    dim: usize,
}

impl IdentityPreprocessor {
    /// Create a new identity preprocessor.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Preprocessor for IdentityPreprocessor {
    fn preprocess_storage(&self, vector: &[f32]) -> Vec<u8> {
        // Just copy the raw bytes
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vector.as_ptr() as *const u8, vector.len() * 4)
        };
        bytes.to_vec()
    }

    fn preprocess_query(&self, vector: &[f32]) -> Vec<u8> {
        self.preprocess_storage(vector)
    }

    fn preprocess_storage_in_place(&self, _vector: &mut [f32]) -> bool {
        true // No-op
    }

    fn storage_size(&self, dim: usize) -> usize {
        dim * std::mem::size_of::<f32>()
    }

    fn query_size(&self, dim: usize) -> usize {
        dim * std::mem::size_of::<f32>()
    }

    fn name(&self) -> &'static str {
        "Identity"
    }
}

/// Cosine preprocessor that L2-normalizes vectors.
///
/// After normalization, cosine similarity can be computed as a simple dot product.
#[derive(Debug, Clone)]
pub struct CosinePreprocessor {
    #[allow(dead_code)]
    dim: usize,
}

impl CosinePreprocessor {
    /// Create a new cosine preprocessor.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Preprocessor for CosinePreprocessor {
    fn preprocess_storage(&self, vector: &[f32]) -> Vec<u8> {
        let mut normalized = vector.to_vec();
        normalize_in_place(&mut normalized);

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(normalized.as_ptr() as *const u8, normalized.len() * 4)
        };
        bytes.to_vec()
    }

    fn preprocess_query(&self, vector: &[f32]) -> Vec<u8> {
        self.preprocess_storage(vector)
    }

    fn preprocess_storage_in_place(&self, vector: &mut [f32]) -> bool {
        normalize_in_place(vector);
        true
    }

    fn storage_size(&self, dim: usize) -> usize {
        dim * std::mem::size_of::<f32>()
    }

    fn query_size(&self, dim: usize) -> usize {
        dim * std::mem::size_of::<f32>()
    }

    fn name(&self) -> &'static str {
        "Cosine"
    }
}

/// Quantization preprocessor for asymmetric SQ8 encoding.
///
/// Storage vectors are quantized to uint8, while query vectors remain as f32
/// with precomputed metadata for efficient asymmetric distance computation.
#[derive(Debug, Clone)]
pub struct QuantPreprocessor {
    #[allow(dead_code)]
    dim: usize,
    /// Whether to include sum of squares (needed for L2).
    include_sum_squares: bool,
}

/// Metadata stored alongside quantized vectors.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QuantMeta {
    /// Minimum value for dequantization.
    pub min_val: f32,
    /// Scale factor for dequantization.
    pub delta: f32,
    /// Sum of original values (for IP computation).
    pub x_sum: f32,
    /// Sum of squared values (for L2 computation).
    pub x_sum_squares: f32,
}

impl QuantPreprocessor {
    /// Create a quantization preprocessor.
    ///
    /// # Arguments
    /// * `dim` - Vector dimension
    /// * `include_sum_squares` - Include sum of squares for L2 distance
    pub fn new(dim: usize, include_sum_squares: bool) -> Self {
        Self {
            dim,
            include_sum_squares,
        }
    }

    /// Create for L2 distance (includes sum of squares).
    pub fn for_l2(dim: usize) -> Self {
        Self::new(dim, true)
    }

    /// Create for inner product (excludes sum of squares).
    pub fn for_ip(dim: usize) -> Self {
        Self::new(dim, false)
    }

    /// Quantize a vector to uint8.
    fn quantize(&self, vector: &[f32]) -> (Vec<u8>, QuantMeta) {
        let dim = vector.len();

        // Find min and max
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in vector {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        // Compute delta
        let range = max_val - min_val;
        let delta = if range > 1e-10 { range / 255.0 } else { 1.0 };
        let inv_delta = 1.0 / delta;

        // Quantize and compute sums
        let mut quantized = vec![0u8; dim];
        let mut x_sum = 0.0f32;
        let mut x_sum_squares = 0.0f32;

        // 4-way unrolled loop for cache efficiency
        let chunks = dim / 4;
        for i in 0..chunks {
            let base = i * 4;
            for j in 0..4 {
                let idx = base + j;
                let v = vector[idx];
                let q = ((v - min_val) * inv_delta).round() as u8;
                quantized[idx] = q.min(255);
                x_sum += v;
                x_sum_squares += v * v;
            }
        }

        // Handle remainder
        for idx in (chunks * 4)..dim {
            let v = vector[idx];
            let q = ((v - min_val) * inv_delta).round() as u8;
            quantized[idx] = q.min(255);
            x_sum += v;
            x_sum_squares += v * v;
        }

        let meta = QuantMeta {
            min_val,
            delta,
            x_sum,
            x_sum_squares,
        };

        (quantized, meta)
    }
}

impl Preprocessor for QuantPreprocessor {
    fn preprocess_storage(&self, vector: &[f32]) -> Vec<u8> {
        let (quantized, meta) = self.quantize(vector);

        // Build storage blob: quantized values + metadata
        let meta_size = if self.include_sum_squares { 16 } else { 12 };
        let mut blob = Vec::with_capacity(quantized.len() + meta_size);
        blob.extend_from_slice(&quantized);
        blob.extend_from_slice(&meta.min_val.to_le_bytes());
        blob.extend_from_slice(&meta.delta.to_le_bytes());
        blob.extend_from_slice(&meta.x_sum.to_le_bytes());
        if self.include_sum_squares {
            blob.extend_from_slice(&meta.x_sum_squares.to_le_bytes());
        }

        blob
    }

    fn preprocess_query(&self, vector: &[f32]) -> Vec<u8> {
        // Query remains as f32 with precomputed sums
        let mut y_sum = 0.0f32;
        let mut y_sum_squares = 0.0f32;

        for &v in vector {
            y_sum += v;
            y_sum_squares += v * v;
        }

        // Build query blob: f32 values + sums
        let meta_size = if self.include_sum_squares { 8 } else { 4 };
        let mut blob = Vec::with_capacity(vector.len() * 4 + meta_size);

        for &v in vector {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        blob.extend_from_slice(&y_sum.to_le_bytes());
        if self.include_sum_squares {
            blob.extend_from_slice(&y_sum_squares.to_le_bytes());
        }

        blob
    }

    fn preprocess_storage_in_place(&self, _vector: &mut [f32]) -> bool {
        // Can't do in-place quantization to different type
        false
    }

    fn storage_size(&self, dim: usize) -> usize {
        let meta_size = if self.include_sum_squares { 16 } else { 12 };
        dim + meta_size
    }

    fn query_size(&self, dim: usize) -> usize {
        let meta_size = if self.include_sum_squares { 8 } else { 4 };
        dim * std::mem::size_of::<f32>() + meta_size
    }

    fn name(&self) -> &'static str {
        "Quant"
    }
}

/// Chain of preprocessors applied sequentially.
pub struct PreprocessorChain {
    preprocessors: Vec<Box<dyn Preprocessor>>,
    dim: usize,
}

impl PreprocessorChain {
    /// Create a new empty preprocessor chain.
    pub fn new(dim: usize) -> Self {
        Self {
            preprocessors: Vec::new(),
            dim,
        }
    }

    /// Add a preprocessor to the chain.
    pub fn add<P: Preprocessor + 'static>(mut self, preprocessor: P) -> Self {
        self.preprocessors.push(Box::new(preprocessor));
        self
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.preprocessors.is_empty()
    }

    /// Get the number of preprocessors in the chain.
    pub fn len(&self) -> usize {
        self.preprocessors.len()
    }

    /// Preprocess for storage through the chain.
    pub fn preprocess_storage(&self, vector: &[f32]) -> Vec<u8> {
        if self.preprocessors.is_empty() {
            // No preprocessing - just return raw bytes
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(vector.as_ptr() as *const u8, vector.len() * 4)
            };
            return bytes.to_vec();
        }

        // Apply first preprocessor
        let result = self.preprocessors[0].preprocess_storage(vector);

        // Chain remaining preprocessors (if any)
        // Note: Chaining is complex when types change; for now, just use first
        result
    }

    /// Preprocess for query through the chain.
    pub fn preprocess_query(&self, vector: &[f32]) -> Vec<u8> {
        if self.preprocessors.is_empty() {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(vector.as_ptr() as *const u8, vector.len() * 4)
            };
            return bytes.to_vec();
        }

        self.preprocessors[0].preprocess_query(vector)
    }

    /// Preprocess in-place for storage.
    pub fn preprocess_storage_in_place(&self, vector: &mut [f32]) -> bool {
        for pp in &self.preprocessors {
            if !pp.preprocess_storage_in_place(vector) {
                return false;
            }
        }
        true
    }

    /// Get the final storage size.
    pub fn storage_size(&self) -> usize {
        if let Some(last) = self.preprocessors.last() {
            last.storage_size(self.dim)
        } else {
            self.dim * std::mem::size_of::<f32>()
        }
    }

    /// Get the final query size.
    pub fn query_size(&self) -> usize {
        if let Some(last) = self.preprocessors.last() {
            last.query_size(self.dim)
        } else {
            self.dim * std::mem::size_of::<f32>()
        }
    }
}

/// Processed vector blobs for storage and query.
#[derive(Debug)]
pub struct ProcessedBlobs {
    /// Processed storage data.
    pub storage: Vec<u8>,
    /// Processed query data.
    pub query: Vec<u8>,
}

impl ProcessedBlobs {
    /// Create from separate storage and query blobs.
    pub fn new(storage: Vec<u8>, query: Vec<u8>) -> Self {
        Self { storage, query }
    }

    /// Create when storage and query are identical.
    pub fn symmetric(data: Vec<u8>) -> Self {
        Self {
            query: data.clone(),
            storage: data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_preprocessor() {
        let pp = IdentityPreprocessor::new(4);
        let vector = vec![1.0f32, 2.0, 3.0, 4.0];

        let storage = pp.preprocess_storage(&vector);
        assert_eq!(storage.len(), 16); // 4 * 4 bytes

        let query = pp.preprocess_query(&vector);
        assert_eq!(query.len(), 16);

        assert_eq!(storage, query);
    }

    #[test]
    fn test_cosine_preprocessor() {
        let pp = CosinePreprocessor::new(4);
        let vector = vec![1.0f32, 0.0, 0.0, 0.0];

        let storage = pp.preprocess_storage(&vector);
        assert_eq!(storage.len(), 16);

        // Parse back to f32
        let normalized: Vec<f32> = storage
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Should be normalized (unit vector)
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_preprocessor_in_place() {
        let pp = CosinePreprocessor::new(4);
        let mut vector = vec![3.0f32, 4.0, 0.0, 0.0];

        assert!(pp.preprocess_storage_in_place(&mut vector));

        // Should be normalized
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);

        // Original direction preserved
        assert!((vector[0] - 0.6).abs() < 0.001);
        assert!((vector[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_quant_preprocessor() {
        let pp = QuantPreprocessor::for_l2(4);
        let vector = vec![0.0f32, 0.5, 0.75, 1.0];

        let storage = pp.preprocess_storage(&vector);
        // 4 bytes quantized + 16 bytes metadata
        assert_eq!(storage.len(), 20);

        // Check quantized values
        assert_eq!(storage[0], 0); // 0.0 -> 0
        assert_eq!(storage[3], 255); // 1.0 -> 255
    }

    #[test]
    fn test_quant_preprocessor_query() {
        let pp = QuantPreprocessor::for_l2(4);
        let vector = vec![1.0f32, 2.0, 3.0, 4.0];

        let query = pp.preprocess_query(&vector);
        // 16 bytes f32 values + 8 bytes (y_sum + y_sum_squares)
        assert_eq!(query.len(), 24);
    }

    #[test]
    fn test_preprocessor_chain() {
        let chain = PreprocessorChain::new(4).add(CosinePreprocessor::new(4));

        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());

        let vector = vec![3.0f32, 4.0, 0.0, 0.0];
        let storage = chain.preprocess_storage(&vector);

        // Parse and verify normalization
        let normalized: Vec<f32> = storage
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_empty_chain() {
        let chain = PreprocessorChain::new(4);
        assert!(chain.is_empty());

        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = chain.preprocess_storage(&vector);

        // Should be unchanged
        let parsed: Vec<f32> = storage
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(parsed, vector);
    }

    #[test]
    fn test_quant_meta_layout() {
        // Verify QuantMeta is 16 bytes as expected
        assert_eq!(std::mem::size_of::<QuantMeta>(), 16);
    }
}

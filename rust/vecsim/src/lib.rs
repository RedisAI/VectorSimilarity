//! VecSim - High-performance vector similarity search library.
//!
//! This library provides efficient implementations of vector similarity indices
//! for nearest neighbor search. It supports multiple index types and distance metrics,
//! with SIMD-optimized distance computations for high performance.
//!
//! # Index Types
//!
//! - **BruteForce**: Linear scan over all vectors. Provides exact results but O(n) query time.
//!   Best for small datasets or when exact results are required.
//!
//! - **HNSW**: Hierarchical Navigable Small World graphs. Provides approximate results with
//!   logarithmic query time. Best for large datasets where some recall loss is acceptable.
//!
//! # Distance Metrics
//!
//! - **L2** (Euclidean): Squared Euclidean distance. Lower values = more similar.
//! - **Inner Product**: Dot product (negated for distance). For normalized vectors.
//! - **Cosine**: 1 - cosine similarity. Measures angle between vectors.
//!
//! # Data Types
//!
//! - `f32`: Single precision float (default)
//! - `f64`: Double precision float
//! - `Float16`: Half precision float (IEEE 754-2008)
//! - `BFloat16`: Brain float (same exponent range as f32)
//!
//! # Examples
//!
//! ## BruteForce Index
//!
//! ```rust
//! use vecsim::prelude::*;
//!
//! // Create a BruteForce index
//! let params = BruteForceParams::new(4, Metric::L2);
//! let mut index = BruteForceSingle::<f32>::new(params);
//!
//! // Add vectors
//! index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
//! index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
//! index.add_vector(&[0.0, 0.0, 1.0, 0.0], 3).unwrap();
//!
//! // Query for nearest neighbors
//! let query = [1.0, 0.1, 0.0, 0.0];
//! let results = index.top_k_query(&query, 2, None).unwrap();
//!
//! assert_eq!(results.results[0].label, 1); // Closest to query
//! ```
//!
//! ## HNSW Index
//!
//! ```rust
//! use vecsim::prelude::*;
//!
//! // Create an HNSW index
//! let params = HnswParams::new(128, Metric::Cosine)
//!     .with_m(16)
//!     .with_ef_construction(200);
//! let mut index = HnswSingle::<f32>::new(params);
//!
//! // Add vectors (would normally add many more)
//! for i in 0..1000 {
//!     let mut v = vec![0.0f32; 128];
//!     v[i % 128] = 1.0;
//!     index.add_vector(&v, i as u64).unwrap();
//! }
//!
//! // Query with custom ef_runtime
//! let query = vec![1.0f32; 128];
//! let params = QueryParams::new().with_ef_runtime(50);
//! let results = index.top_k_query(&query, 10, Some(&params)).unwrap();
//! ```
//!
//! ## Multi-value Index
//!
//! ```rust
//! use vecsim::prelude::*;
//!
//! // Create a multi-value index (multiple vectors per label)
//! let params = BruteForceParams::new(4, Metric::L2);
//! let mut index = BruteForceMulti::<f32>::new(params);
//!
//! // Add multiple vectors with the same label
//! index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
//! index.add_vector(&[0.9, 0.1, 0.0, 0.0], 1).unwrap(); // Same label
//! index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
//!
//! assert_eq!(index.label_count(1), 2);
//! ```

pub mod containers;
pub mod distance;
pub mod index;
pub mod memory;
pub mod quantization;
pub mod query;
pub mod serialization;
pub mod types;
pub mod utils;

/// Prelude module for convenient imports.
///
/// Use `use vecsim::prelude::*;` to import commonly used types.
pub mod prelude {
    // Types
    pub use crate::types::{
        BFloat16, DistanceType, Float16, IdType, Int32, Int64, Int8, LabelType, UInt8,
        VectorElement, INVALID_ID,
    };

    // Quantization
    pub use crate::quantization::{Sq8Codec, Sq8VectorMeta};

    // Distance
    pub use crate::distance::{
        batch_normalize, cosine_similarity, dot_product, euclidean_distance, l2_norm, l2_squared,
        normalize, normalize_in_place, Metric,
    };

    // Query
    pub use crate::query::{QueryParams, QueryReply, QueryResult};

    // Index traits and errors
    pub use crate::index::{
        BatchIterator, IndexError, IndexInfo, IndexType, MultiValue, QueryError, VecSimIndex,
    };

    // BruteForce
    pub use crate::index::{BruteForceMulti, BruteForceParams, BruteForceSingle, BruteForceStats};

    // Index estimation
    pub use crate::index::{
        estimate_brute_force_element_size, estimate_brute_force_initial_size,
        estimate_hnsw_element_size, estimate_hnsw_initial_size,
    };

    // HNSW
    pub use crate::index::{HnswMulti, HnswParams, HnswSingle, HnswStats};

    // Serialization
    pub use crate::serialization::{Deserializable, Serializable, SerializationError};
}

/// Create a BruteForce index with the given parameters.
///
/// This is a convenience function for creating single-value BruteForce indices.
pub fn create_brute_force<T: types::VectorElement>(
    dim: usize,
    metric: distance::Metric,
) -> index::BruteForceSingle<T> {
    let params = index::BruteForceParams::new(dim, metric);
    index::BruteForceSingle::new(params)
}

/// Create an HNSW index with the given parameters.
///
/// This is a convenience function for creating single-value HNSW indices.
pub fn create_hnsw<T: types::VectorElement>(
    dim: usize,
    metric: distance::Metric,
) -> index::HnswSingle<T> {
    let params = index::HnswParams::new(dim, metric);
    index::HnswSingle::new(params)
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_prelude_imports() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

        let results = index.top_k_query(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.results[0].label, 1);
    }

    #[test]
    fn test_convenience_functions() {
        let mut bf_index = super::create_brute_force::<f32>(4, Metric::L2);
        bf_index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(bf_index.index_size(), 1);

        let mut hnsw_index = super::create_hnsw::<f32>(4, Metric::L2);
        hnsw_index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hnsw_index.index_size(), 1);
    }
}

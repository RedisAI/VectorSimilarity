//! Vector similarity index implementations.
//!
//! This module provides different index types for vector similarity search:
//! - `brute_force`: Linear scan over all vectors (exact results)
//! - `hnsw`: Hierarchical Navigable Small World graphs (approximate, fast)

pub mod brute_force;
pub mod hnsw;
pub mod traits;

// Re-export traits
pub use traits::{
    BatchIterator, IndexError, IndexInfo, IndexType, MultiValue, QueryError, VecSimIndex,
};

// Re-export BruteForce types
pub use brute_force::{
    BruteForceParams, BruteForceSingle, BruteForceMulti, BruteForceBatchIterator, BruteForceStats,
};

// Re-export HNSW types
pub use hnsw::{
    HnswParams, HnswSingle, HnswMulti, HnswBatchIterator, HnswStats,
};

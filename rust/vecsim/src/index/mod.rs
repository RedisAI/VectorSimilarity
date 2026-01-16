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

/// Estimate the initial memory size for a BruteForce index.
///
/// This estimates the memory needed before any vectors are added.
pub fn estimate_brute_force_initial_size(dim: usize, initial_capacity: usize) -> usize {
    // Base struct overhead
    let base = std::mem::size_of::<BruteForceSingle<f32>>();
    // Data storage
    let data = dim * std::mem::size_of::<f32>() * initial_capacity;
    // Label maps
    let maps = initial_capacity * std::mem::size_of::<(u64, u32)>() * 2;
    base + data + maps
}

/// Estimate the memory size per element for a BruteForce index.
pub fn estimate_brute_force_element_size(dim: usize) -> usize {
    // Vector data
    let vector = dim * std::mem::size_of::<f32>();
    // Label entry overhead
    let label_overhead = std::mem::size_of::<(u64, u32)>() + std::mem::size_of::<(u64, bool)>();
    vector + label_overhead
}

/// Estimate the initial memory size for an HNSW index.
///
/// This estimates the memory needed before any vectors are added.
pub fn estimate_hnsw_initial_size(dim: usize, initial_capacity: usize, m: usize) -> usize {
    // Base struct overhead
    let base = std::mem::size_of::<HnswSingle<f32>>();
    // Data storage
    let data = dim * std::mem::size_of::<f32>() * initial_capacity;
    // Graph overhead per node (rough estimate: neighbors at level 0 + higher levels)
    let graph = initial_capacity * (m * 2 + m) * std::mem::size_of::<u32>();
    // Label maps
    let maps = initial_capacity * std::mem::size_of::<(u64, u32)>() * 2;
    // Visited pool
    let visited = initial_capacity * std::mem::size_of::<u32>();
    base + data + graph + maps + visited
}

/// Estimate the memory size per element for an HNSW index.
pub fn estimate_hnsw_element_size(dim: usize, m: usize) -> usize {
    // Vector data
    let vector = dim * std::mem::size_of::<f32>();
    // Graph connections (average across levels)
    // Level 0 has 2*M neighbors, higher levels have M each
    // Average number of levels is ~1.3 for typical M values
    let avg_levels = 1.3f64;
    let graph = ((m * 2) as f64 + (avg_levels * m as f64)) as usize * std::mem::size_of::<u32>();
    // Label entry overhead
    let label_overhead = std::mem::size_of::<(u64, u32)>() * 2;
    vector + graph + label_overhead
}

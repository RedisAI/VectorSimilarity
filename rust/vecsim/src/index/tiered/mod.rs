//! Tiered index combining BruteForce frontend with HNSW backend.
//!
//! The tiered architecture optimizes for:
//! - **Fast writes**: New vectors go to flat BruteForce buffer
//! - **Efficient queries**: HNSW provides logarithmic search complexity
//! - **Flexible migration**: Vectors can be flushed from flat to HNSW
//!
//! # Architecture
//!
//! ```text
//! TieredIndex
//! ├── Frontend: BruteForce index (fast write buffer)
//! ├── Backend: HNSW index (efficient approximate search)
//! └── Query: Searches both tiers, merges results
//! ```
//!
//! # Write Modes
//!
//! - **Async**: Vectors added to flat buffer, later migrated to HNSW via `flush()`
//! - **InPlace**: Vectors added directly to HNSW (used when buffer is full)

pub mod batch_iterator;
pub mod multi;
pub mod single;

pub use batch_iterator::TieredBatchIterator;
pub use multi::TieredMulti;
pub use single::TieredSingle;

use crate::distance::Metric;
use crate::index::hnsw::HnswParams;

/// Write mode for the tiered index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WriteMode {
    /// Async mode: vectors go to flat buffer, migrated to HNSW via flush().
    #[default]
    Async,
    /// InPlace mode: vectors go directly to HNSW.
    InPlace,
}

/// Configuration parameters for TieredIndex.
#[derive(Debug, Clone)]
pub struct TieredParams {
    /// Vector dimension.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Parameters for the HNSW backend index.
    pub hnsw_params: HnswParams,
    /// Maximum size of the flat buffer before forcing in-place writes.
    /// When flat buffer reaches this limit, new writes go directly to HNSW.
    pub flat_buffer_limit: usize,
    /// Write mode for the index.
    pub write_mode: WriteMode,
    /// Initial capacity hint.
    pub initial_capacity: usize,
}

impl TieredParams {
    /// Create new tiered index parameters.
    ///
    /// # Arguments
    /// * `dim` - Vector dimension
    /// * `metric` - Distance metric (L2, InnerProduct, Cosine)
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            hnsw_params: HnswParams::new(dim, metric),
            flat_buffer_limit: 10_000,
            write_mode: WriteMode::Async,
            initial_capacity: 1000,
        }
    }

    /// Set the HNSW M parameter (max connections per node).
    pub fn with_m(mut self, m: usize) -> Self {
        self.hnsw_params = self.hnsw_params.with_m(m);
        self
    }

    /// Set the ef_construction parameter for HNSW.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.hnsw_params = self.hnsw_params.with_ef_construction(ef);
        self
    }

    /// Set the ef_runtime parameter for HNSW queries.
    pub fn with_ef_runtime(mut self, ef: usize) -> Self {
        self.hnsw_params = self.hnsw_params.with_ef_runtime(ef);
        self
    }

    /// Set the flat buffer limit.
    ///
    /// When the flat buffer reaches this size, the index switches to
    /// in-place writes (directly to HNSW) until flush() is called.
    pub fn with_flat_buffer_limit(mut self, limit: usize) -> Self {
        self.flat_buffer_limit = limit;
        self
    }

    /// Set the write mode.
    pub fn with_write_mode(mut self, mode: WriteMode) -> Self {
        self.write_mode = mode;
        self
    }

    /// Set the initial capacity hint.
    pub fn with_initial_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = capacity;
        self.hnsw_params = self.hnsw_params.with_capacity(capacity);
        self
    }
}

/// Merge two sorted query replies, keeping the top k results.
///
/// Both replies are assumed to be sorted by distance (ascending).
pub fn merge_top_k<D: crate::types::DistanceType>(
    flat_results: crate::query::QueryReply<D>,
    hnsw_results: crate::query::QueryReply<D>,
    k: usize,
) -> crate::query::QueryReply<D> {
    use crate::query::QueryReply;

    // Fast paths
    if flat_results.is_empty() {
        let mut results = hnsw_results;
        results.results.truncate(k);
        return results;
    }

    if hnsw_results.is_empty() {
        let mut results = flat_results;
        results.results.truncate(k);
        return results;
    }

    // Merge both result sets
    let mut merged = QueryReply::with_capacity(flat_results.len() + hnsw_results.len());

    // Add all results
    for result in flat_results.results {
        merged.push(result);
    }
    for result in hnsw_results.results {
        merged.push(result);
    }

    // Sort by distance and truncate
    merged.sort_by_distance();
    merged.results.truncate(k);

    merged
}

/// Merge two query replies for range queries.
///
/// Combines all results from both tiers.
pub fn merge_range<D: crate::types::DistanceType>(
    flat_results: crate::query::QueryReply<D>,
    hnsw_results: crate::query::QueryReply<D>,
) -> crate::query::QueryReply<D> {
    use crate::query::QueryReply;

    if flat_results.is_empty() {
        return hnsw_results;
    }

    if hnsw_results.is_empty() {
        return flat_results;
    }

    let mut merged = QueryReply::with_capacity(flat_results.len() + hnsw_results.len());

    for result in flat_results.results {
        merged.push(result);
    }
    for result in hnsw_results.results {
        merged.push(result);
    }

    merged.sort_by_distance();
    merged
}

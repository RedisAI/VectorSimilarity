//! Tiered index combining BruteForce frontend with SVS (Vamana) backend.
//!
//! Similar to the HNSW-backed tiered index, but uses the single-layer Vamana
//! graph as the backend for approximate nearest neighbor search.
//!
//! # Architecture
//!
//! ```text
//! TieredSvsIndex
//! ├── Frontend: BruteForce index (fast write buffer)
//! ├── Backend: SVS (Vamana) index (efficient approximate search)
//! └── Query: Searches both tiers, merges results
//! ```
//!
//! # Write Modes
//!
//! - **Async**: Vectors added to flat buffer, later migrated to SVS via `flush()`
//! - **InPlace**: Vectors added directly to SVS (used when buffer is full)

pub mod batch_iterator;
pub mod multi;
pub mod single;

pub use batch_iterator::TieredSvsBatchIterator;
pub use multi::TieredSvsMulti;
pub use single::TieredSvsSingle;

use crate::distance::Metric;
use crate::index::svs::SvsParams;

/// Write mode for the tiered SVS index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvsWriteMode {
    /// Async mode: vectors go to flat buffer, migrated to SVS via flush().
    #[default]
    Async,
    /// InPlace mode: vectors go directly to SVS.
    InPlace,
}

/// Configuration parameters for TieredSvsIndex.
#[derive(Debug, Clone)]
pub struct TieredSvsParams {
    /// Vector dimension.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Parameters for the SVS backend index.
    pub svs_params: SvsParams,
    /// Maximum size of the flat buffer before forcing in-place writes.
    /// When flat buffer reaches this limit, new writes go directly to SVS.
    pub flat_buffer_limit: usize,
    /// Write mode for the index.
    pub write_mode: SvsWriteMode,
    /// Initial capacity hint.
    pub initial_capacity: usize,
}

impl TieredSvsParams {
    /// Create new tiered SVS index parameters.
    ///
    /// # Arguments
    /// * `dim` - Vector dimension
    /// * `metric` - Distance metric (L2, InnerProduct, Cosine)
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            svs_params: SvsParams::new(dim, metric),
            flat_buffer_limit: 10_000,
            write_mode: SvsWriteMode::Async,
            initial_capacity: 1000,
        }
    }

    /// Set the graph max degree (R) parameter for SVS.
    pub fn with_graph_degree(mut self, r: usize) -> Self {
        self.svs_params = self.svs_params.with_graph_degree(r);
        self
    }

    /// Set the alpha parameter for robust pruning.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.svs_params = self.svs_params.with_alpha(alpha);
        self
    }

    /// Set the construction window size (L) for SVS.
    pub fn with_construction_l(mut self, l: usize) -> Self {
        self.svs_params = self.svs_params.with_construction_l(l);
        self
    }

    /// Set the search window size for SVS queries.
    pub fn with_search_l(mut self, l: usize) -> Self {
        self.svs_params = self.svs_params.with_search_l(l);
        self
    }

    /// Set the flat buffer limit.
    ///
    /// When the flat buffer reaches this size, the index switches to
    /// in-place writes (directly to SVS) until flush() is called.
    pub fn with_flat_buffer_limit(mut self, limit: usize) -> Self {
        self.flat_buffer_limit = limit;
        self
    }

    /// Set the write mode.
    pub fn with_write_mode(mut self, mode: SvsWriteMode) -> Self {
        self.write_mode = mode;
        self
    }

    /// Set the initial capacity hint.
    pub fn with_initial_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = capacity;
        self.svs_params = self.svs_params.with_capacity(capacity);
        self
    }

    /// Enable/disable two-pass construction for SVS.
    pub fn with_two_pass(mut self, enable: bool) -> Self {
        self.svs_params = self.svs_params.with_two_pass(enable);
        self
    }
}

/// Merge two sorted query replies, keeping the top k results.
///
/// Both replies are assumed to be sorted by distance (ascending).
pub fn merge_top_k<D: crate::types::DistanceType>(
    flat_results: crate::query::QueryReply<D>,
    svs_results: crate::query::QueryReply<D>,
    k: usize,
) -> crate::query::QueryReply<D> {
    use crate::query::QueryReply;

    // Fast paths
    if flat_results.is_empty() {
        let mut results = svs_results;
        results.results.truncate(k);
        return results;
    }

    if svs_results.is_empty() {
        let mut results = flat_results;
        results.results.truncate(k);
        return results;
    }

    // Merge both result sets
    let mut merged = QueryReply::with_capacity(flat_results.len() + svs_results.len());

    // Add all results
    for result in flat_results.results {
        merged.push(result);
    }
    for result in svs_results.results {
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
    svs_results: crate::query::QueryReply<D>,
) -> crate::query::QueryReply<D> {
    use crate::query::QueryReply;

    if flat_results.is_empty() {
        return svs_results;
    }

    if svs_results.is_empty() {
        return flat_results;
    }

    let mut merged = QueryReply::with_capacity(flat_results.len() + svs_results.len());

    for result in flat_results.results {
        merged.push(result);
    }
    for result in svs_results.results {
        merged.push(result);
    }

    merged.sort_by_distance();
    merged
}

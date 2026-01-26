//! SVS (Vamana) index implementation.
//!
//! SVS (Search via Satellite) is based on the Vamana algorithm, a graph-based
//! approximate nearest neighbor index that provides high recall with efficient
//! construction and query performance.
//!
//! ## Key Differences from HNSW
//!
//! | Aspect | HNSW | Vamana |
//! |--------|------|--------|
//! | Structure | Multi-layer hierarchical | Single flat layer |
//! | Entry point | Random high-level node | Medoid (centroid) |
//! | Construction | Random level + single pass | Two passes with alpha |
//! | Search | Layer traversal | Greedy beam search |
//!
//! ## Key Parameters
//!
//! - `graph_max_degree` (R): Maximum number of neighbors per node (default: 32)
//! - `alpha`: Pruning parameter for robust neighbor selection (default: 1.2)
//! - `construction_window_size` (L): Beam width during construction (default: 200)
//! - `search_window_size`: Beam width during search (runtime)
//!
//! ## Algorithm Overview
//!
//! ### Construction (Two-Pass)
//!
//! 1. Find medoid as entry point (approximate centroid)
//! 2. Pass 1: Build initial graph with alpha = 1.0
//! 3. Pass 2: Refine graph with configured alpha (e.g., 1.2)
//!
//! ### Robust Pruning (Alpha)
//!
//! A neighbor is selected only if:
//! `dist(neighbor, any_selected) * alpha >= dist(neighbor, target)`
//!
//! This ensures neighbors are diverse (not all clustered together).
//!
//! ### Search
//!
//! Greedy beam search from medoid entry point.

pub mod graph;
pub mod search;
pub mod single;
pub mod multi;

pub use graph::{VamanaGraph, VamanaGraphData};
pub use single::{SvsSingle, SvsStats, SvsSingleBatchIterator};
pub use multi::{SvsMulti, SvsMultiBatchIterator};

use crate::containers::DataBlocks;
use crate::distance::{create_distance_function, DistanceFunction, Metric};
use crate::types::{DistanceType, IdType, LabelType, VectorElement, INVALID_ID};
use crate::index::hnsw::VisitedNodesHandlerPool;
use std::sync::atomic::{AtomicU32, Ordering};

/// Default maximum graph degree (R).
pub const DEFAULT_GRAPH_DEGREE: usize = 32;

/// Default alpha for robust pruning.
pub const DEFAULT_ALPHA: f32 = 1.2;

/// Default construction window size (L).
pub const DEFAULT_CONSTRUCTION_L: usize = 200;

/// Default search window size.
pub const DEFAULT_SEARCH_L: usize = 100;

/// Parameters for creating an SVS (Vamana) index.
#[derive(Debug, Clone)]
pub struct SvsParams {
    /// Vector dimension.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Maximum number of neighbors per node (R).
    pub graph_max_degree: usize,
    /// Alpha parameter for robust pruning (1.0 = no diversity, 1.2 = moderate).
    pub alpha: f32,
    /// Beam width during construction (L).
    pub construction_window_size: usize,
    /// Default beam width during search.
    pub search_window_size: usize,
    /// Initial capacity (number of vectors).
    pub initial_capacity: usize,
    /// Use two-pass construction (recommended for better recall).
    pub two_pass_construction: bool,
}

impl SvsParams {
    /// Create new parameters with required fields.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            graph_max_degree: DEFAULT_GRAPH_DEGREE,
            alpha: DEFAULT_ALPHA,
            construction_window_size: DEFAULT_CONSTRUCTION_L,
            search_window_size: DEFAULT_SEARCH_L,
            initial_capacity: 1024,
            two_pass_construction: true,
        }
    }

    /// Set maximum graph degree (R).
    pub fn with_graph_degree(mut self, r: usize) -> Self {
        self.graph_max_degree = r;
        self
    }

    /// Set alpha parameter for robust pruning.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set construction window size (L).
    pub fn with_construction_l(mut self, l: usize) -> Self {
        self.construction_window_size = l;
        self
    }

    /// Set search window size.
    pub fn with_search_l(mut self, l: usize) -> Self {
        self.search_window_size = l;
        self
    }

    /// Set initial capacity.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = capacity;
        self
    }

    /// Enable/disable two-pass construction.
    pub fn with_two_pass(mut self, enable: bool) -> Self {
        self.two_pass_construction = enable;
        self
    }
}

/// Core SVS implementation shared between single and multi variants.
pub(crate) struct SvsCore<T: VectorElement> {
    /// Vector storage.
    pub data: DataBlocks<T>,
    /// Graph structure.
    pub graph: VamanaGraph,
    /// Distance function.
    pub dist_fn: Box<dyn DistanceFunction<T, Output = T::DistanceType>>,
    /// Medoid (entry point).
    pub medoid: AtomicU32,
    /// Pool of visited handlers for concurrent searches.
    pub visited_pool: VisitedNodesHandlerPool,
    /// Parameters.
    pub params: SvsParams,
}

impl<T: VectorElement> SvsCore<T> {
    /// Create a new SVS core.
    pub fn new(params: SvsParams) -> Self {
        let data = DataBlocks::new(params.dim, params.initial_capacity);
        let dist_fn = create_distance_function(params.metric, params.dim);
        let graph = VamanaGraph::new(params.initial_capacity, params.graph_max_degree);
        let visited_pool = VisitedNodesHandlerPool::new(params.initial_capacity);

        Self {
            data,
            graph,
            dist_fn,
            medoid: AtomicU32::new(INVALID_ID),
            visited_pool,
            params,
        }
    }

    /// Add a vector and return its internal ID.
    pub fn add_vector(&mut self, vector: &[T]) -> Option<IdType> {
        let processed = self.dist_fn.preprocess(vector, self.params.dim);
        self.data.add(&processed)
    }

    /// Get vector data by ID.
    #[inline]
    pub fn get_vector(&self, id: IdType) -> Option<&[T]> {
        self.data.get(id)
    }

    /// Find the medoid (approximate centroid) of all vectors.
    pub fn find_medoid(&self) -> Option<IdType> {
        let ids: Vec<IdType> = self.data.iter_ids().collect();
        if ids.is_empty() {
            return None;
        }
        if ids.len() == 1 {
            return Some(ids[0]);
        }

        // Sample at most 1000 vectors for medoid computation
        let sample_size = ids.len().min(1000);
        let sample: Vec<IdType> = if ids.len() <= sample_size {
            ids.clone()
        } else {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut shuffled = ids.clone();
            shuffled.shuffle(&mut rng);
            shuffled.into_iter().take(sample_size).collect()
        };

        // Find vector with minimum total distance
        let mut best_id = sample[0];
        let mut best_total_dist = f64::MAX;

        for &candidate in &sample {
            if let Some(candidate_data) = self.get_vector(candidate) {
                let total_dist: f64 = sample
                    .iter()
                    .filter(|&&id| id != candidate)
                    .filter_map(|&id| self.get_vector(id))
                    .map(|other_data| {
                        self.dist_fn
                            .compute(candidate_data, other_data, self.params.dim)
                            .to_f64()
                    })
                    .sum();

                if total_dist < best_total_dist {
                    best_total_dist = total_dist;
                    best_id = candidate;
                }
            }
        }

        Some(best_id)
    }

    /// Insert a new element into the graph.
    pub fn insert(&mut self, id: IdType, label: LabelType) {
        // Ensure graph has space
        self.graph.ensure_capacity(id as usize + 1);
        self.graph.set_label(id, label);

        // Update visited pool if needed
        if (id as usize) >= self.visited_pool.current_capacity() {
            self.visited_pool.resize(id as usize + 1024);
        }

        let medoid = self.medoid.load(Ordering::Acquire);

        if medoid == INVALID_ID {
            // First element becomes the medoid
            self.medoid.store(id, Ordering::Release);
            return;
        }

        // Get query vector
        let query = match self.get_vector(id) {
            Some(v) => v,
            None => return,
        };

        // Search for neighbors and select using robust pruning
        // Use a block to limit the lifetime of `visited` before mutable operations
        let selected = {
            let mut visited = self.visited_pool.get();
            visited.reset();

            let neighbors = search::greedy_beam_search(
                medoid,
                query,
                self.params.construction_window_size,
                &self.graph,
                |nid| self.data.get(nid),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
            );

            search::robust_prune(
                id,
                &neighbors,
                self.params.graph_max_degree,
                self.params.alpha,
                |nid| self.data.get(nid),
                self.dist_fn.as_ref(),
                self.params.dim,
            )
        };

        // Set outgoing edges
        self.graph.set_neighbors(id, &selected);

        // Add bidirectional edges (after visited is dropped)
        for &neighbor_id in &selected {
            self.add_bidirectional_link(neighbor_id, id);
        }
    }

    /// Add a bidirectional link from one node to another.
    fn add_bidirectional_link(&mut self, from: IdType, to: IdType) {
        let mut current_neighbors = self.graph.get_neighbors(from);
        if current_neighbors.contains(&to) {
            return;
        }

        current_neighbors.push(to);

        // Check if we need to prune
        if current_neighbors.len() > self.params.graph_max_degree {
            if let Some(from_data) = self.data.get(from) {
                let candidates: Vec<_> = current_neighbors
                    .iter()
                    .filter_map(|&n| {
                        self.data.get(n).map(|data| {
                            let dist = self.dist_fn.compute(data, from_data, self.params.dim);
                            (n, dist)
                        })
                    })
                    .collect();

                let selected = search::robust_prune(
                    from,
                    &candidates,
                    self.params.graph_max_degree,
                    self.params.alpha,
                    |id| self.data.get(id),
                    self.dist_fn.as_ref(),
                    self.params.dim,
                );

                self.graph.set_neighbors(from, &selected);
            }
        } else {
            self.graph.set_neighbors(from, &current_neighbors);
        }
    }

    /// Rebuild the graph (second pass of two-pass construction).
    pub fn rebuild_graph(&mut self) {
        let ids: Vec<IdType> = self.data.iter_ids().collect();

        // Update medoid
        if let Some(new_medoid) = self.find_medoid() {
            self.medoid.store(new_medoid, Ordering::Release);
        }

        // Clear existing neighbors
        for &id in &ids {
            self.graph.clear_neighbors(id);
        }

        // Reinsert all nodes with current alpha
        for &id in &ids {
            let label = self.graph.get_label(id);
            self.insert_without_label_update(id, label);
        }
    }

    /// Insert without updating the label (used during rebuild).
    fn insert_without_label_update(&mut self, id: IdType, _label: LabelType) {
        let medoid = self.medoid.load(Ordering::Acquire);
        if medoid == INVALID_ID || medoid == id {
            return;
        }

        let query = match self.get_vector(id) {
            Some(v) => v,
            None => return,
        };

        // Use a block to limit the lifetime of `visited` before mutable operations
        let selected = {
            let mut visited = self.visited_pool.get();
            visited.reset();

            let neighbors = search::greedy_beam_search(
                medoid,
                query,
                self.params.construction_window_size,
                &self.graph,
                |nid| self.data.get(nid),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
            );

            search::robust_prune(
                id,
                &neighbors,
                self.params.graph_max_degree,
                self.params.alpha,
                |nid| self.data.get(nid),
                self.dist_fn.as_ref(),
                self.params.dim,
            )
        };

        self.graph.set_neighbors(id, &selected);

        for &neighbor_id in &selected {
            self.add_bidirectional_link(neighbor_id, id);
        }
    }

    /// Mark an element as deleted.
    pub fn mark_deleted(&mut self, id: IdType) {
        self.graph.mark_deleted(id);
        self.data.mark_deleted(id);

        // Update medoid if needed
        if self.medoid.load(Ordering::Acquire) == id {
            if let Some(new_medoid) = self.find_medoid() {
                self.medoid.store(new_medoid, Ordering::Release);
            } else {
                self.medoid.store(INVALID_ID, Ordering::Release);
            }
        }
    }

    /// Search for nearest neighbors.
    pub fn search(
        &self,
        query: &[T],
        k: usize,
        search_l: usize,
        filter: Option<&dyn Fn(IdType) -> bool>,
    ) -> Vec<(IdType, T::DistanceType)> {
        let medoid = self.medoid.load(Ordering::Acquire);
        if medoid == INVALID_ID {
            return Vec::new();
        }

        let mut visited = self.visited_pool.get();
        visited.reset();

        let results = search::greedy_beam_search_filtered(
            medoid,
            query,
            search_l.max(k),
            &self.graph,
            |id| self.data.get(id),
            self.dist_fn.as_ref(),
            self.params.dim,
            &visited,
            filter,
        );

        results.into_iter().take(k).collect()
    }
}

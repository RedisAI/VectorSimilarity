//! SQ8-quantized SVS (Vamana) index implementation.
//!
//! This module provides SVS indices that use scalar 8-bit quantization for
//! memory-efficient vector storage. Vectors are stored as u8 with per-vector
//! metadata for dequantization and asymmetric distance computation.
//!
//! ## Memory Savings
//!
//! SQ8 quantization reduces memory usage by ~4x for f32 vectors:
//! - Original: 4 bytes per dimension
//! - Quantized: 1 byte per dimension + 16 bytes metadata per vector
//!
//! ## Asymmetric Distance
//!
//! During search, the query vector stays in f32 while stored vectors are in SQ8.
//! This provides better accuracy than symmetric quantization.

use super::{SvsParams, VamanaGraph};
use crate::containers::Sq8DataBlocks;
use crate::distance::Metric;
use crate::index::hnsw::VisitedNodesHandlerPool;
use crate::quantization::sq8::{sq8_asymmetric_cosine, sq8_asymmetric_inner_product, sq8_asymmetric_l2_squared};
use crate::quantization::Sq8VectorMeta;
use crate::types::{IdType, INVALID_ID};
use std::sync::atomic::AtomicU32;

/// Core SQ8-quantized SVS implementation.
pub struct SvsSq8Core {
    /// Quantized vector storage.
    pub data: Sq8DataBlocks,
    /// Graph structure.
    pub graph: VamanaGraph,
    /// Distance metric.
    pub metric: Metric,
    /// Medoid (entry point).
    pub medoid: AtomicU32,
    /// Pool of visited handlers for concurrent searches.
    pub visited_pool: VisitedNodesHandlerPool,
    /// Parameters.
    pub params: SvsParams,
}

impl SvsSq8Core {
    /// Create a new SQ8-quantized SVS core.
    pub fn new(params: SvsParams) -> Self {
        let data = Sq8DataBlocks::new(params.dim, params.initial_capacity);
        let graph = VamanaGraph::new(params.initial_capacity, params.graph_max_degree);
        let visited_pool = VisitedNodesHandlerPool::new(params.initial_capacity);

        Self {
            data,
            graph,
            metric: params.metric,
            medoid: AtomicU32::new(INVALID_ID),
            visited_pool,
            params,
        }
    }

    /// Add a vector (will be quantized) and return its internal ID.
    pub fn add_vector(&mut self, vector: &[f32]) -> Option<IdType> {
        self.data.add(vector)
    }

    /// Get quantized data and metadata by ID.
    #[inline]
    pub fn get_quantized(&self, id: IdType) -> Option<(&[u8], &Sq8VectorMeta)> {
        self.data.get(id)
    }

    /// Decode a vector back to f32.
    pub fn decode_vector(&self, id: IdType) -> Option<Vec<f32>> {
        self.data.decode(id)
    }

    /// Compute asymmetric distance between f32 query and quantized stored vector.
    #[inline]
    pub fn asymmetric_distance(&self, query: &[f32], quantized: &[u8], meta: &Sq8VectorMeta) -> f32 {
        match self.metric {
            Metric::L2 => sq8_asymmetric_l2_squared(query, quantized, meta, self.params.dim),
            Metric::InnerProduct => sq8_asymmetric_inner_product(query, quantized, meta, self.params.dim),
            Metric::Cosine => sq8_asymmetric_cosine(query, quantized, meta, self.params.dim),
        }
    }

    /// Compute distance between query and stored vector by ID.
    #[inline]
    pub fn distance_to(&self, query: &[f32], id: IdType) -> Option<f32> {
        let (quantized, meta) = self.get_quantized(id)?;
        Some(self.asymmetric_distance(query, quantized, meta))
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
        // For SQ8, we decode to f32 for accurate medoid computation
        let mut best_id = sample[0];
        let mut best_total_dist = f64::MAX;

        for &candidate in &sample {
            if let Some(candidate_data) = self.decode_vector(candidate) {
                let total_dist: f64 = sample
                    .iter()
                    .filter(|&&id| id != candidate)
                    .filter_map(|&id| self.distance_to(&candidate_data, id))
                    .map(|d| d as f64)
                    .sum();

                if total_dist < best_total_dist {
                    best_total_dist = total_dist;
                    best_id = candidate;
                }
            }
        }

        Some(best_id)
    }

    /// Get the dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.params.dim
    }

    /// Get the number of vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Mark a slot as deleted.
    pub fn mark_deleted(&mut self, id: IdType) -> bool {
        self.data.mark_deleted(id)
    }

    /// Update a vector at the given ID.
    pub fn update_vector(&mut self, id: IdType, vector: &[f32]) -> bool {
        self.data.update(id, vector)
    }

    /// Get the search window size.
    pub fn search_window_size(&self) -> usize {
        self.params.search_window_size
    }

    /// Get the medoid ID.
    pub fn get_medoid(&self) -> IdType {
        self.medoid.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Set the medoid ID.
    pub fn set_medoid(&self, id: IdType) {
        self.medoid.store(id, std::sync::atomic::Ordering::Release)
    }
}

use super::search::greedy_beam_search_sq8;
use crate::query::QueryResult;
use crate::types::LabelType;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;

/// Single-value SQ8-quantized SVS index.
///
/// Each label has exactly one associated vector, stored in SQ8 format.
pub struct SvsSq8Single {
    /// Core SQ8-quantized SVS implementation.
    core: RwLock<SvsSq8Core>,
    /// Label to internal ID mapping.
    label_to_id: RwLock<HashMap<LabelType, IdType>>,
    /// Internal ID to label mapping.
    id_to_label: RwLock<HashMap<IdType, LabelType>>,
    /// Number of vectors.
    count: AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
    /// Construction completed flag.
    construction_done: RwLock<bool>,
}

impl SvsSq8Single {
    /// Create a new SQ8-quantized SVS index.
    pub fn new(params: SvsParams) -> Self {
        let initial_capacity = params.initial_capacity;
        Self {
            core: RwLock::new(SvsSq8Core::new(params)),
            label_to_id: RwLock::new(HashMap::with_capacity(initial_capacity)),
            id_to_label: RwLock::new(HashMap::with_capacity(initial_capacity)),
            count: AtomicUsize::new(0),
            capacity: None,
            construction_done: RwLock::new(false),
        }
    }

    /// Get the distance metric.
    pub fn metric(&self) -> Metric {
        self.core.read().metric
    }

    /// Get the dimension.
    pub fn dim(&self) -> usize {
        self.core.read().params.dim
    }

    /// Get the search window size.
    pub fn search_l(&self) -> usize {
        self.core.read().search_window_size()
    }

    /// Search for the k nearest neighbors using SQ8 asymmetric distance.
    fn search_knn(&self, query: &[f32], k: usize, ef: usize) -> Vec<QueryResult<f32>> {
        let core = self.core.read();

        let medoid = core.get_medoid();
        if medoid == INVALID_ID || core.is_empty() {
            return vec![];
        }

        let visited = core.visited_pool.get();

        // Perform greedy beam search with SQ8 asymmetric distance
        let search_result = greedy_beam_search_sq8(
            medoid,
            query,
            ef.max(k),
            &core.graph,
            &core.data,
            core.metric,
            core.params.dim,
            &visited,
            None::<&fn(IdType) -> bool>,
        );

        // Map internal IDs to labels
        let id_to_label = self.id_to_label.read();
        let mut results: Vec<QueryResult<f32>> = search_result
            .results
            .into_iter()
            .filter_map(|r| {
                id_to_label.get(&r.id).map(|&label| QueryResult {
                    label,
                    distance: r.distance,
                })
            })
            .collect();

        results.truncate(k);
        results
    }
}


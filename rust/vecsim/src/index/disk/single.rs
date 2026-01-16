//! Single-value disk-based index implementation.
//!
//! This index stores one vector per label using memory-mapped storage.

use super::{DiskBackend, DiskIndexParams};
use crate::containers::MmapDataBlocks;
use crate::distance::{create_distance_function, DistanceFunction};
use crate::index::hnsw::VisitedNodesHandlerPool;
use crate::index::svs::graph::VamanaGraph;
use crate::index::svs::search::{greedy_beam_search_filtered, robust_prune};
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply, QueryResult};
use crate::types::{DistanceType, IdType, LabelType, VectorElement, INVALID_ID};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Statistics for disk index.
#[derive(Debug, Clone)]
pub struct DiskIndexStats {
    /// Number of vectors.
    pub vector_count: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Backend type.
    pub backend: DiskBackend,
    /// Data file path.
    pub data_path: String,
    /// Fragmentation ratio.
    pub fragmentation: f64,
    /// Approximate memory usage (for in-memory structures).
    pub memory_bytes: usize,
}

/// State for Vamana graph-based search.
struct VamanaState {
    /// Graph structure for neighbor relationships.
    graph: VamanaGraph,
    /// Medoid (entry point for search).
    medoid: AtomicU32,
    /// Pool of visited handlers for concurrent searches.
    visited_pool: VisitedNodesHandlerPool,
}

impl VamanaState {
    /// Create a new Vamana state.
    fn new(initial_capacity: usize, max_degree: usize) -> Self {
        Self {
            graph: VamanaGraph::new(initial_capacity, max_degree),
            medoid: AtomicU32::new(INVALID_ID),
            visited_pool: VisitedNodesHandlerPool::new(initial_capacity),
        }
    }
}

/// Single-value disk-based index.
///
/// Stores vectors in a memory-mapped file for persistence.
/// Supports BruteForce (exact) or Vamana (approximate) search.
pub struct DiskIndexSingle<T: VectorElement> {
    /// Memory-mapped vector storage.
    data: RwLock<MmapDataBlocks<T>>,
    /// Distance function.
    dist_fn: Box<dyn DistanceFunction<T, Output = T::DistanceType>>,
    /// Label to internal ID mapping.
    label_to_id: RwLock<HashMap<LabelType, IdType>>,
    /// Internal ID to label mapping.
    id_to_label: RwLock<HashMap<IdType, LabelType>>,
    /// Parameters.
    params: DiskIndexParams,
    /// Vector count.
    count: AtomicUsize,
    /// Capacity (if bounded).
    capacity: Option<usize>,
    /// Vamana graph state (for Vamana backend).
    vamana_state: Option<RwLock<VamanaState>>,
}

impl<T: VectorElement> DiskIndexSingle<T> {
    /// Create a new disk-based index.
    pub fn new(params: DiskIndexParams) -> io::Result<Self> {
        let data = MmapDataBlocks::new(&params.data_path, params.dim, params.initial_capacity)?;
        let dist_fn = create_distance_function(params.metric, params.dim);

        // Build label mappings from existing data
        let label_to_id = HashMap::new();
        let id_to_label = HashMap::new();
        let count = data.len();

        // Initialize Vamana state if using Vamana backend
        let vamana_state = match params.backend {
            DiskBackend::Vamana => Some(RwLock::new(VamanaState::new(
                params.initial_capacity,
                params.graph_max_degree,
            ))),
            DiskBackend::BruteForce => None,
        };

        // For now, we don't persist label mappings - this would need separate storage
        // New indices start fresh, existing data without labels won't work well
        // A full implementation would store labels in the mmap file

        Ok(Self {
            data: RwLock::new(data),
            dist_fn,
            label_to_id: RwLock::new(label_to_id),
            id_to_label: RwLock::new(id_to_label),
            params,
            count: AtomicUsize::new(count),
            capacity: None,
            vamana_state,
        })
    }

    /// Create with a maximum capacity.
    pub fn with_capacity(params: DiskIndexParams, max_capacity: usize) -> io::Result<Self> {
        let mut index = Self::new(params)?;
        index.capacity = Some(max_capacity);
        Ok(index)
    }

    /// Get the data file path.
    pub fn data_path(&self) -> &Path {
        &self.params.data_path
    }

    /// Get the backend type.
    pub fn backend(&self) -> DiskBackend {
        self.params.backend
    }

    /// Get index statistics.
    pub fn stats(&self) -> DiskIndexStats {
        let data = self.data.read();

        DiskIndexStats {
            vector_count: self.count.load(Ordering::Relaxed),
            dimension: self.params.dim,
            backend: self.params.backend,
            data_path: self.params.data_path.to_string_lossy().to_string(),
            fragmentation: data.fragmentation(),
            memory_bytes: self.memory_usage(),
        }
    }

    /// Get the fragmentation ratio.
    pub fn fragmentation(&self) -> f64 {
        self.data.read().fragmentation()
    }

    /// Estimate memory usage (in-memory structures only).
    pub fn memory_usage(&self) -> usize {
        let label_size = self.label_to_id.read().capacity()
            * (std::mem::size_of::<LabelType>() + std::mem::size_of::<IdType>())
            + self.id_to_label.read().capacity()
            * (std::mem::size_of::<IdType>() + std::mem::size_of::<LabelType>());

        label_size + std::mem::size_of::<Self>()
    }

    /// Flush changes to disk.
    pub fn flush(&self) -> io::Result<()> {
        self.data.read().flush()
    }

    /// Get a copy of a vector by label.
    pub fn get_vector(&self, label: LabelType) -> Option<Vec<T>> {
        let id = *self.label_to_id.read().get(&label)?;
        self.data.read().get(id).map(|v| v.to_vec())
    }

    /// Clear all vectors.
    pub fn clear(&mut self) {
        self.data.write().clear();
        self.label_to_id.write().clear();
        self.id_to_label.write().clear();
        self.count.store(0, Ordering::Relaxed);

        // Clear Vamana state if present
        if let Some(ref vamana) = self.vamana_state {
            let mut state = vamana.write();
            state.graph = VamanaGraph::new(
                self.params.initial_capacity,
                self.params.graph_max_degree,
            );
            state.medoid.store(INVALID_ID, Ordering::Release);
        }
    }

    // ========== Vamana-specific methods ==========

    /// Insert a vector into the Vamana graph.
    ///
    /// This method is called after the vector has been added to the mmap storage.
    fn insert_into_graph(&self, id: IdType, label: LabelType) {
        let vamana = match &self.vamana_state {
            Some(v) => v,
            None => return,
        };

        let mut state = vamana.write();
        let data = self.data.read();

        // Ensure graph has space
        state.graph.ensure_capacity(id as usize + 1);
        state.graph.set_label(id, label);

        // Update visited pool if needed
        if (id as usize) >= state.visited_pool.current_capacity() {
            state.visited_pool.resize(id as usize + 1024);
        }

        let medoid = state.medoid.load(Ordering::Acquire);

        if medoid == INVALID_ID {
            // First element becomes the medoid
            state.medoid.store(id, Ordering::Release);
            return;
        }

        // Get query vector
        let query = match data.get(id) {
            Some(v) => v,
            None => return,
        };

        // Search for neighbors using greedy beam search
        let selected = {
            let mut visited = state.visited_pool.get();
            visited.reset();

            let neighbors = greedy_beam_search_filtered::<T, T::DistanceType, _, fn(IdType) -> bool>(
                medoid,
                query,
                self.params.construction_l,
                &state.graph,
                |nid| data.get(nid),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
                None,
            );

            // Select neighbors using robust pruning
            robust_prune(
                id,
                &neighbors,
                self.params.graph_max_degree,
                self.params.alpha,
                |nid| data.get(nid),
                self.dist_fn.as_ref(),
                self.params.dim,
            )
        };

        // Set outgoing edges
        state.graph.set_neighbors(id, &selected);

        // Add bidirectional edges
        for &neighbor_id in &selected {
            Self::add_bidirectional_link_inner(
                &mut state.graph,
                &data,
                self.dist_fn.as_ref(),
                self.params.dim,
                self.params.graph_max_degree,
                self.params.alpha,
                neighbor_id,
                id,
            );
        }
    }

    /// Add a bidirectional link from one node to another in the graph.
    fn add_bidirectional_link_inner(
        graph: &mut VamanaGraph,
        data: &MmapDataBlocks<T>,
        dist_fn: &dyn DistanceFunction<T, Output = T::DistanceType>,
        dim: usize,
        max_degree: usize,
        alpha: f32,
        from: IdType,
        to: IdType,
    ) {
        let mut current_neighbors = graph.get_neighbors(from);
        if current_neighbors.contains(&to) {
            return;
        }

        current_neighbors.push(to);

        // Check if we need to prune
        if current_neighbors.len() > max_degree {
            if let Some(from_data) = data.get(from) {
                let candidates: Vec<_> = current_neighbors
                    .iter()
                    .filter_map(|&n| {
                        data.get(n).map(|d| {
                            let dist = dist_fn.compute(d, from_data, dim);
                            (n, dist)
                        })
                    })
                    .collect();

                let selected = robust_prune(
                    from,
                    &candidates,
                    max_degree,
                    alpha,
                    |id| data.get(id),
                    dist_fn,
                    dim,
                );

                graph.set_neighbors(from, &selected);
            }
        } else {
            graph.set_neighbors(from, &current_neighbors);
        }
    }

    /// Mark a vector as deleted in the Vamana graph.
    fn delete_from_graph(&self, id: IdType) {
        let vamana = match &self.vamana_state {
            Some(v) => v,
            None => return,
        };

        let mut state = vamana.write();
        state.graph.mark_deleted(id);

        // Update medoid if needed
        if state.medoid.load(Ordering::Acquire) == id {
            if let Some(new_medoid) = self.find_new_medoid(&state) {
                state.medoid.store(new_medoid, Ordering::Release);
            } else {
                state.medoid.store(INVALID_ID, Ordering::Release);
            }
        }
    }

    /// Find a new medoid for the graph (excluding deleted nodes).
    fn find_new_medoid(&self, state: &VamanaState) -> Option<IdType> {
        let data = self.data.read();
        let id_to_label = self.id_to_label.read();

        // Get all active IDs
        let ids: Vec<IdType> = data
            .iter_ids()
            .filter(|&id| !state.graph.is_deleted(id) && id_to_label.contains_key(&id))
            .collect();

        if ids.is_empty() {
            return None;
        }
        if ids.len() == 1 {
            return Some(ids[0]);
        }

        // Sample for medoid computation
        let sample_size = ids.len().min(100);
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
            if let Some(candidate_data) = data.get(candidate) {
                let total_dist: f64 = sample
                    .iter()
                    .filter(|&&id| id != candidate)
                    .filter_map(|&id| data.get(id))
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

    /// Perform Vamana graph-based search.
    fn vamana_search(
        &self,
        query: &[T],
        k: usize,
        filter: Option<&(dyn Fn(LabelType) -> bool + Send + Sync)>,
    ) -> Vec<(IdType, LabelType, T::DistanceType)> {
        let vamana = match &self.vamana_state {
            Some(v) => v,
            None => return Vec::new(),
        };

        let state = vamana.read();
        let medoid = state.medoid.load(Ordering::Acquire);

        if medoid == INVALID_ID {
            return Vec::new();
        }

        let data = self.data.read();
        let id_to_label = self.id_to_label.read();

        let mut visited = state.visited_pool.get();
        visited.reset();

        let search_l = self.params.search_l.max(k);

        // Perform search with internal ID filter if needed
        let results = if let Some(f) = filter {
            // Create filter that works with internal IDs
            let id_filter = |id: IdType| -> bool {
                if let Some(&label) = id_to_label.get(&id) {
                    f(label)
                } else {
                    false
                }
            };

            greedy_beam_search_filtered(
                medoid,
                query,
                search_l,
                &state.graph,
                |id| data.get(id),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
                Some(&id_filter),
            )
        } else {
            greedy_beam_search_filtered::<T, T::DistanceType, _, fn(IdType) -> bool>(
                medoid,
                query,
                search_l,
                &state.graph,
                |id| data.get(id),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
                None,
            )
        };

        // Convert to final format with labels
        results
            .into_iter()
            .take(k)
            .filter_map(|(id, dist)| {
                let label = *id_to_label.get(&id)?;
                Some((id, label, dist))
            })
            .collect()
    }

    /// Find the medoid (approximate centroid) of all vectors.
    ///
    /// This is used as the entry point for Vamana search.
    pub fn find_medoid(&self) -> Option<IdType> {
        let data = self.data.read();
        let id_to_label = self.id_to_label.read();

        let ids: Vec<IdType> = data
            .iter_ids()
            .filter(|id| id_to_label.contains_key(id))
            .collect();

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
            if let Some(candidate_data) = data.get(candidate) {
                let total_dist: f64 = sample
                    .iter()
                    .filter(|&&id| id != candidate)
                    .filter_map(|&id| data.get(id))
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

    /// Build or rebuild the Vamana graph.
    ///
    /// This performs a two-pass construction for better recall:
    /// 1. First pass with alpha = 1.0
    /// 2. Second pass with configured alpha
    pub fn build(&mut self) {
        if self.params.backend != DiskBackend::Vamana {
            return;
        }

        let vamana = match &self.vamana_state {
            Some(v) => v,
            None => return,
        };

        // Get all IDs and labels
        let ids_and_labels: Vec<(IdType, LabelType)> = {
            let id_to_label = self.id_to_label.read();
            self.data
                .read()
                .iter_ids()
                .filter_map(|id| id_to_label.get(&id).map(|&label| (id, label)))
                .collect()
        };

        if ids_and_labels.is_empty() {
            return;
        }

        // Clear and reinitialize graph
        {
            let mut state = vamana.write();
            state.graph = VamanaGraph::new(
                self.params.initial_capacity,
                self.params.graph_max_degree,
            );
            state.medoid.store(INVALID_ID, Ordering::Release);
        }

        // Find and set medoid
        if let Some(medoid) = self.find_medoid() {
            vamana.write().medoid.store(medoid, Ordering::Release);
        }

        // First pass: insert all nodes with alpha = 1.0
        let original_alpha = self.params.alpha;

        // Pass 1 with alpha = 1.0 (no diversity pruning)
        for &(id, label) in &ids_and_labels {
            self.insert_into_graph_with_alpha(id, label, 1.0);
        }

        // Pass 2: rebuild with configured alpha for better diversity
        if original_alpha > 1.0 {
            self.rebuild_graph_with_alpha(original_alpha, &ids_and_labels);
        }
    }

    /// Insert a vector into the graph with a specific alpha value.
    fn insert_into_graph_with_alpha(&self, id: IdType, label: LabelType, alpha: f32) {
        let vamana = match &self.vamana_state {
            Some(v) => v,
            None => return,
        };

        let mut state = vamana.write();
        let data = self.data.read();

        // Ensure graph has space
        state.graph.ensure_capacity(id as usize + 1);
        state.graph.set_label(id, label);

        // Update visited pool if needed
        if (id as usize) >= state.visited_pool.current_capacity() {
            state.visited_pool.resize(id as usize + 1024);
        }

        let medoid = state.medoid.load(Ordering::Acquire);

        if medoid == INVALID_ID {
            state.medoid.store(id, Ordering::Release);
            return;
        }

        let query = match data.get(id) {
            Some(v) => v,
            None => return,
        };

        let selected = {
            let mut visited = state.visited_pool.get();
            visited.reset();

            let neighbors = greedy_beam_search_filtered::<T, T::DistanceType, _, fn(IdType) -> bool>(
                medoid,
                query,
                self.params.construction_l,
                &state.graph,
                |nid| data.get(nid),
                self.dist_fn.as_ref(),
                self.params.dim,
                &visited,
                None,
            );

            robust_prune(
                id,
                &neighbors,
                self.params.graph_max_degree,
                alpha,
                |nid| data.get(nid),
                self.dist_fn.as_ref(),
                self.params.dim,
            )
        };

        state.graph.set_neighbors(id, &selected);

        for &neighbor_id in &selected {
            Self::add_bidirectional_link_inner(
                &mut state.graph,
                &data,
                self.dist_fn.as_ref(),
                self.params.dim,
                self.params.graph_max_degree,
                alpha,
                neighbor_id,
                id,
            );
        }
    }

    /// Rebuild the graph with a specific alpha value.
    fn rebuild_graph_with_alpha(&self, alpha: f32, ids_and_labels: &[(IdType, LabelType)]) {
        let vamana = match &self.vamana_state {
            Some(v) => v,
            None => return,
        };

        // Clear existing neighbors
        {
            let mut state = vamana.write();
            for &(id, _) in ids_and_labels {
                state.graph.clear_neighbors(id);
            }
        }

        // Update medoid
        if let Some(new_medoid) = self.find_medoid() {
            vamana.write().medoid.store(new_medoid, Ordering::Release);
        }

        // Reinsert all nodes with configured alpha
        for &(id, label) in ids_and_labels {
            self.insert_into_graph_with_alpha(id, label, alpha);
        }
    }

    /// Perform brute-force search.
    fn brute_force_search(
        &self,
        query: &[T],
        k: usize,
        filter: Option<&(dyn Fn(LabelType) -> bool + Send + Sync)>,
    ) -> Vec<(IdType, LabelType, T::DistanceType)> {
        let data = self.data.read();
        let id_to_label = self.id_to_label.read();

        let mut results: Vec<(IdType, LabelType, T::DistanceType)> = data
            .iter_ids()
            .filter_map(|id| {
                let label = *id_to_label.get(&id)?;

                // Apply filter if present
                if let Some(f) = filter {
                    if !f(label) {
                        return None;
                    }
                }

                let vec_data = data.get(id)?;
                let dist = self.dist_fn.compute(query, vec_data, self.params.dim);
                Some((id, label, dist))
            })
            .collect();

        // Sort by distance
        results.sort_by(|a, b| {
            a.2.to_f64()
                .partial_cmp(&b.2.to_f64())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(k);
        results
    }

    /// Perform range search.
    fn brute_force_range(
        &self,
        query: &[T],
        radius: T::DistanceType,
        filter: Option<&(dyn Fn(LabelType) -> bool + Send + Sync)>,
    ) -> Vec<(IdType, LabelType, T::DistanceType)> {
        let data = self.data.read();
        let id_to_label = self.id_to_label.read();
        let radius_f64 = radius.to_f64();

        let mut results: Vec<(IdType, LabelType, T::DistanceType)> = data
            .iter_ids()
            .filter_map(|id| {
                let label = *id_to_label.get(&id)?;

                if let Some(f) = filter {
                    if !f(label) {
                        return None;
                    }
                }

                let vec_data = data.get(id)?;
                let dist = self.dist_fn.compute(query, vec_data, self.params.dim);

                if dist.to_f64() <= radius_f64 {
                    Some((id, label, dist))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| {
            a.2.to_f64()
                .partial_cmp(&b.2.to_f64())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

impl<T: VectorElement> VecSimIndex for DiskIndexSingle<T> {
    type DataType = T;
    type DistType = T::DistanceType;

    fn add_vector(&mut self, vector: &[T], label: LabelType) -> Result<usize, IndexError> {
        if vector.len() != self.params.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.params.dim,
                got: vector.len(),
            });
        }

        // Check capacity
        if let Some(cap) = self.capacity {
            if self.count.load(Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();
        let mut data = self.data.write();

        // Check if label exists (replace)
        if let Some(&existing_id) = label_to_id.get(&label) {
            // Mark old entry as deleted in both storage and graph
            data.mark_deleted(existing_id);
            id_to_label.remove(&existing_id);

            // Mark deleted in graph for Vamana backend and handle medoid update
            if self.params.backend == DiskBackend::Vamana {
                if let Some(ref vamana) = self.vamana_state {
                    let mut state = vamana.write();
                    state.graph.mark_deleted(existing_id);
                    // If the medoid was deleted, set to INVALID_ID (will be set to new node)
                    if state.medoid.load(Ordering::Acquire) == existing_id {
                        state.medoid.store(INVALID_ID, Ordering::Release);
                    }
                }
            }

            let new_id = data.add(vector).ok_or_else(|| {
                IndexError::Internal("Failed to add vector to storage".to_string())
            })?;

            label_to_id.insert(label, new_id);
            id_to_label.insert(new_id, label);

            // Drop locks before calling insert_into_graph
            drop(data);
            drop(label_to_id);
            drop(id_to_label);

            // Insert into Vamana graph
            if self.params.backend == DiskBackend::Vamana {
                self.insert_into_graph(new_id, label);
            }

            return Ok(0); // Replacement
        }

        // Add new vector
        let id = data.add(vector).ok_or_else(|| {
            IndexError::Internal("Failed to add vector to storage".to_string())
        })?;

        label_to_id.insert(label, id);
        id_to_label.insert(id, label);
        self.count.fetch_add(1, Ordering::Relaxed);

        // Drop locks before calling insert_into_graph
        drop(data);
        drop(label_to_id);
        drop(id_to_label);

        // Insert into Vamana graph
        if self.params.backend == DiskBackend::Vamana {
            self.insert_into_graph(id, label);
        }

        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();
        let mut data = self.data.write();

        if let Some(id) = label_to_id.remove(&label) {
            data.mark_deleted(id);
            id_to_label.remove(&id);
            self.count.fetch_sub(1, Ordering::Relaxed);

            // Drop locks before calling delete_from_graph
            drop(data);
            drop(label_to_id);
            drop(id_to_label);

            // Delete from Vamana graph
            if self.params.backend == DiskBackend::Vamana {
                self.delete_from_graph(id);
            }

            Ok(1)
        } else {
            Err(IndexError::LabelNotFound(label))
        }
    }

    fn top_k_query(
        &self,
        query: &[Self::DataType],
        k: usize,
        params: Option<&QueryParams>,
    ) -> Result<QueryReply<Self::DistType>, QueryError> {
        if query.len() != self.params.dim {
            return Err(QueryError::DimensionMismatch {
                expected: self.params.dim,
                got: query.len(),
            });
        }

        if k == 0 {
            return Ok(QueryReply::new());
        }

        // Build filter if provided
        let filter = params.and_then(|p| p.filter.as_ref().map(|f| f.as_ref()));

        let results = match self.params.backend {
            DiskBackend::BruteForce => self.brute_force_search(query, k, filter),
            DiskBackend::Vamana => self.vamana_search(query, k, filter),
        };

        let mut reply = QueryReply::with_capacity(results.len());
        for (_, label, dist) in results {
            reply.push(QueryResult::new(label, dist));
        }

        Ok(reply)
    }

    fn range_query(
        &self,
        query: &[Self::DataType],
        radius: Self::DistType,
        params: Option<&QueryParams>,
    ) -> Result<QueryReply<Self::DistType>, QueryError> {
        if query.len() != self.params.dim {
            return Err(QueryError::DimensionMismatch {
                expected: self.params.dim,
                got: query.len(),
            });
        }

        let filter = params.and_then(|p| p.filter.as_ref().map(|f| f.as_ref()));

        let results = match self.params.backend {
            DiskBackend::BruteForce => self.brute_force_range(query, radius, filter),
            DiskBackend::Vamana => {
                // For Vamana, search with a large k and filter by radius
                // Use search_l as the search window, results are approximate
                let search_results = self.vamana_search(query, self.params.search_l, filter);
                let radius_f64 = radius.to_f64();

                search_results
                    .into_iter()
                    .filter(|(_, _, dist)| dist.to_f64() <= radius_f64)
                    .collect()
            }
        };

        let mut reply = QueryReply::with_capacity(results.len());
        for (_, label, dist) in results {
            reply.push(QueryResult::new(label, dist));
        }

        Ok(reply)
    }

    fn index_size(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    fn index_capacity(&self) -> Option<usize> {
        self.capacity
    }

    fn dimension(&self) -> usize {
        self.params.dim
    }

    fn batch_iterator<'a>(
        &'a self,
        query: &[Self::DataType],
        params: Option<&QueryParams>,
    ) -> Result<Box<dyn BatchIterator<DistType = Self::DistType> + 'a>, QueryError> {
        if query.len() != self.params.dim {
            return Err(QueryError::DimensionMismatch {
                expected: self.params.dim,
                got: query.len(),
            });
        }

        Ok(Box::new(DiskIndexBatchIterator::new(
            self,
            query.to_vec(),
            params.cloned(),
        )))
    }

    fn info(&self) -> IndexInfo {
        IndexInfo {
            size: self.index_size(),
            capacity: self.capacity,
            dimension: self.params.dim,
            index_type: "DiskIndexSingle",
            memory_bytes: self.memory_usage(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.label_to_id.read().contains_key(&label)
    }

    fn label_count(&self, label: LabelType) -> usize {
        if self.contains(label) { 1 } else { 0 }
    }
}

// Safety: DiskIndexSingle is thread-safe with RwLock protection
unsafe impl<T: VectorElement> Send for DiskIndexSingle<T> {}
unsafe impl<T: VectorElement> Sync for DiskIndexSingle<T> {}

/// Batch iterator for DiskIndexSingle.
pub struct DiskIndexBatchIterator<'a, T: VectorElement> {
    index: &'a DiskIndexSingle<T>,
    query: Vec<T>,
    params: Option<QueryParams>,
    results: Option<Vec<(IdType, LabelType, T::DistanceType)>>,
    position: usize,
}

impl<'a, T: VectorElement> DiskIndexBatchIterator<'a, T> {
    /// Create a new batch iterator.
    pub fn new(
        index: &'a DiskIndexSingle<T>,
        query: Vec<T>,
        params: Option<QueryParams>,
    ) -> Self {
        Self {
            index,
            query,
            params,
            results: None,
            position: 0,
        }
    }

    fn ensure_results(&mut self) {
        if self.results.is_some() {
            return;
        }

        let filter = self
            .params
            .as_ref()
            .and_then(|p| p.filter.as_ref().map(|f| f.as_ref()));

        let count = self.index.count.load(Ordering::Relaxed);
        let results = self.index.brute_force_search(&self.query, count, filter);

        self.results = Some(results);
    }
}

impl<'a, T: VectorElement> BatchIterator for DiskIndexBatchIterator<'a, T> {
    type DistType = T::DistanceType;

    fn has_next(&self) -> bool {
        match &self.results {
            Some(results) => self.position < results.len(),
            None => true,
        }
    }

    fn next_batch(
        &mut self,
        batch_size: usize,
    ) -> Option<Vec<(IdType, LabelType, Self::DistType)>> {
        self.ensure_results();

        let results = self.results.as_ref()?;
        if self.position >= results.len() {
            return None;
        }

        let end = (self.position + batch_size).min(results.len());
        let batch = results[self.position..end].to_vec();
        self.position = end;

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    fn reset(&mut self) {
        self.position = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use std::fs;

    fn temp_path() -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("disk_index_test_{}.dat", rand::random::<u64>()));
        path
    }

    #[test]
    fn test_disk_index_basic() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            let v1 = vec![1.0, 0.0, 0.0, 0.0];
            let v2 = vec![0.0, 1.0, 0.0, 0.0];
            let v3 = vec![0.0, 0.0, 1.0, 0.0];

            index.add_vector(&v1, 1).unwrap();
            index.add_vector(&v2, 2).unwrap();
            index.add_vector(&v3, 3).unwrap();

            assert_eq!(index.index_size(), 3);

            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 2, None).unwrap();

            assert_eq!(results.len(), 2);
            assert_eq!(results.results[0].label, 1);
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_delete() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

            assert_eq!(index.index_size(), 2);

            index.delete_vector(1).unwrap();
            assert_eq!(index.index_size(), 1);
            assert!(!index.contains(1));
            assert!(index.contains(2));
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_replace() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(index.index_size(), 1);

            // Replace
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(index.index_size(), 1);

            // Query should return new vector
            let query = vec![0.0, 1.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 1, None).unwrap();
            assert!(results.results[0].distance < 0.001);
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_range_query() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[0.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
            index.add_vector(&[2.0, 0.0, 0.0, 0.0], 3).unwrap();
            index.add_vector(&[10.0, 0.0, 0.0, 0.0], 4).unwrap();

            let query = vec![0.0, 0.0, 0.0, 0.0];
            let results = index.range_query(&query, 5.0, None).unwrap();

            // Should find first 3 vectors (distances: 0, 1, 4)
            assert_eq!(results.len(), 3);
        }

        fs::remove_file(&path).ok();
    }

    // ========== Vamana Backend Tests ==========

    #[test]
    fn test_disk_index_vamana_basic() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana)
                .with_graph_degree(8)
                .with_construction_l(50)
                .with_search_l(20);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            let v1 = vec![1.0, 0.0, 0.0, 0.0];
            let v2 = vec![0.0, 1.0, 0.0, 0.0];
            let v3 = vec![0.0, 0.0, 1.0, 0.0];

            index.add_vector(&v1, 1).unwrap();
            index.add_vector(&v2, 2).unwrap();
            index.add_vector(&v3, 3).unwrap();

            assert_eq!(index.index_size(), 3);

            // Query for exact match
            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 2, None).unwrap();

            assert_eq!(results.len(), 2);
            // Exact match should be first
            assert_eq!(results.results[0].label, 1);
            assert!(results.results[0].distance < 0.001);
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_scaling() {
        let path = temp_path();
        {
            let dim = 16;
            let n = 1000;

            let params = DiskIndexParams::new(dim, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana)
                .with_capacity(n + 100)
                .with_graph_degree(16)
                .with_construction_l(100)
                .with_search_l(50);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Add vectors in a grid pattern
            for i in 0..n {
                let mut vec = vec![0.0f32; dim];
                vec[i % dim] = (i / dim + 1) as f32;
                index.add_vector(&vec, i as u64).unwrap();
            }

            assert_eq!(index.index_size(), n);

            // Query for a vector that should exist
            let mut query = vec![0.0f32; dim];
            query[0] = 1.0;
            let results = index.top_k_query(&query, 10, None).unwrap();

            assert_eq!(results.len(), 10);
            // First result should be label 0 (exact match)
            assert_eq!(results.results[0].label, 0);
            assert!(results.results[0].distance < 0.001);
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_delete() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana)
                .with_graph_degree(8)
                .with_search_l(20);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
            index.add_vector(&[0.0, 0.0, 1.0, 0.0], 3).unwrap();

            assert_eq!(index.index_size(), 3);

            // Delete label 1
            index.delete_vector(1).unwrap();
            assert_eq!(index.index_size(), 2);
            assert!(!index.contains(1));

            // Query should not return deleted vector
            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 3, None).unwrap();

            // Should only get 2 results (labels 2 and 3)
            assert_eq!(results.len(), 2);
            for result in &results.results {
                assert_ne!(result.label, 1);
            }
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_update() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana)
                .with_graph_degree(8)
                .with_search_l(20);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Add initial vector
            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(index.index_size(), 1);

            // Query for original
            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 1, None).unwrap();
            assert!(results.results[0].distance < 0.001);

            // Replace with new vector
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(index.index_size(), 1); // Still 1 (replacement)

            // Query for new vector
            let query2 = vec![0.0, 1.0, 0.0, 0.0];
            let results2 = index.top_k_query(&query2, 1, None).unwrap();
            assert_eq!(results2.results[0].label, 1);
            assert!(results2.results[0].distance < 0.001);
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_range_query() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana)
                .with_graph_degree(8)
                .with_search_l(50); // Higher search_l for range query
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[0.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
            index.add_vector(&[2.0, 0.0, 0.0, 0.0], 3).unwrap();
            index.add_vector(&[10.0, 0.0, 0.0, 0.0], 4).unwrap();

            let query = vec![0.0, 0.0, 0.0, 0.0];
            let results = index.range_query(&query, 5.0, None).unwrap();

            // Should find first 3 vectors (distances: 0, 1, 4)
            assert!(results.len() >= 2); // At least 2 with Vamana (approximate)
            assert!(results.len() <= 4); // At most all 4
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_filtered() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana)
                .with_graph_degree(8)
                .with_search_l(30);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Add vectors with labels 1-10
            for i in 1..=10 {
                let vec = vec![i as f32, 0.0, 0.0, 0.0];
                index.add_vector(&vec, i).unwrap();
            }

            // Query with filter to only accept even labels
            let query = vec![5.0, 0.0, 0.0, 0.0];
            let filter_fn = |label: LabelType| label % 2 == 0;
            let params = QueryParams::default().with_filter(filter_fn);

            let results = index.top_k_query(&query, 5, Some(&params)).unwrap();

            // All results should have even labels
            for result in &results.results {
                assert_eq!(result.label % 2, 0);
            }
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_build() {
        let path = temp_path();
        {
            let dim = 8;
            let n = 100;

            let params = DiskIndexParams::new(dim, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana)
                .with_graph_degree(8)
                .with_alpha(1.2)
                .with_construction_l(50)
                .with_search_l(30);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Add vectors
            for i in 0..n {
                let mut vec = vec![0.0f32; dim];
                vec[i % dim] = (i / dim + 1) as f32;
                index.add_vector(&vec, i as u64).unwrap();
            }

            // Rebuild the graph (two-pass construction)
            index.build();

            // Query should still work correctly
            let mut query = vec![0.0f32; dim];
            query[0] = 1.0;
            let results = index.top_k_query(&query, 5, None).unwrap();

            assert!(!results.is_empty());
            // First result should be label 0 (exact match)
            assert_eq!(results.results[0].label, 0);
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_recall() {
        let path = temp_path();
        {
            let dim = 8;
            let n = 200;

            let params = DiskIndexParams::new(dim, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana)
                .with_graph_degree(16)
                .with_construction_l(100)
                .with_search_l(50);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Create brute force index for ground truth
            let bf_path = temp_path();
            let bf_params = DiskIndexParams::new(dim, Metric::L2, &bf_path);
            let mut bf_index = DiskIndexSingle::<f32>::new(bf_params).unwrap();

            // Add same vectors to both
            for i in 0..n {
                let vec: Vec<f32> = (0..dim).map(|j| ((i * dim + j) % 100) as f32 / 10.0).collect();
                index.add_vector(&vec, i as u64).unwrap();
                bf_index.add_vector(&vec, i as u64).unwrap();
            }

            // Test recall with several queries
            let k = 10;
            let mut total_recall = 0.0;
            let num_queries = 10;

            for q in 0..num_queries {
                let query: Vec<f32> = (0..dim).map(|j| ((q * 7 + j * 3) % 100) as f32 / 10.0).collect();

                let vamana_results = index.top_k_query(&query, k, None).unwrap();
                let bf_results = bf_index.top_k_query(&query, k, None).unwrap();

                // Count how many Vamana results are in ground truth
                let vamana_labels: std::collections::HashSet<_> =
                    vamana_results.results.iter().map(|r| r.label).collect();
                let bf_labels: std::collections::HashSet<_> =
                    bf_results.results.iter().map(|r| r.label).collect();

                let intersection = vamana_labels.intersection(&bf_labels).count();
                total_recall += intersection as f64 / k as f64;
            }

            let avg_recall = total_recall / num_queries as f64;
            // With reasonable parameters, we should get at least 60% recall
            assert!(
                avg_recall >= 0.6,
                "Average recall {} is too low",
                avg_recall
            );

            fs::remove_file(&bf_path).ok();
        }

        fs::remove_file(&path).ok();
    }

    // ========== DiskIndexParams Tests ==========

    #[test]
    fn test_disk_index_params_new() {
        let path = temp_path();
        let params = DiskIndexParams::new(128, Metric::L2, &path);

        assert_eq!(params.dim, 128);
        assert_eq!(params.metric, Metric::L2);
        assert_eq!(params.backend, DiskBackend::BruteForce);
        assert_eq!(params.initial_capacity, 10_000);
        assert_eq!(params.graph_max_degree, 32);
        assert!((params.alpha - 1.2).abs() < 0.01);
        assert_eq!(params.construction_l, 200);
        assert_eq!(params.search_l, 100);
    }

    #[test]
    fn test_disk_index_params_builder() {
        let path = temp_path();
        let params = DiskIndexParams::new(64, Metric::InnerProduct, &path)
            .with_backend(DiskBackend::Vamana)
            .with_capacity(5000)
            .with_graph_degree(64)
            .with_alpha(1.5)
            .with_construction_l(150)
            .with_search_l(75);

        assert_eq!(params.dim, 64);
        assert_eq!(params.metric, Metric::InnerProduct);
        assert_eq!(params.backend, DiskBackend::Vamana);
        assert_eq!(params.initial_capacity, 5000);
        assert_eq!(params.graph_max_degree, 64);
        assert!((params.alpha - 1.5).abs() < 0.01);
        assert_eq!(params.construction_l, 150);
        assert_eq!(params.search_l, 75);
    }

    #[test]
    fn test_disk_backend_default() {
        let backend = DiskBackend::default();
        assert_eq!(backend, DiskBackend::BruteForce);
    }

    #[test]
    fn test_disk_backend_debug() {
        assert_eq!(format!("{:?}", DiskBackend::BruteForce), "BruteForce");
        assert_eq!(format!("{:?}", DiskBackend::Vamana), "Vamana");
    }

    // ========== DiskIndexSingle Utility Method Tests ==========

    #[test]
    fn test_disk_index_get_vector() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            let v1 = vec![1.0, 2.0, 3.0, 4.0];
            index.add_vector(&v1, 1).unwrap();

            // Get existing vector
            let retrieved = index.get_vector(1).unwrap();
            assert_eq!(retrieved.len(), 4);
            for (a, b) in v1.iter().zip(retrieved.iter()) {
                assert!((a - b).abs() < 0.001);
            }

            // Get non-existing vector
            assert!(index.get_vector(999).is_none());
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_stats() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

            let stats = index.stats();
            assert_eq!(stats.vector_count, 2);
            assert_eq!(stats.dimension, 4);
            assert_eq!(stats.backend, DiskBackend::BruteForce);
            assert!(stats.data_path.contains("disk_index_test_"));
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_fragmentation() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Initially no fragmentation
            let frag = index.fragmentation();
            assert!(frag >= 0.0);

            // Add and delete to create fragmentation
            for i in 0..10 {
                index.add_vector(&[i as f32, 0.0, 0.0, 0.0], i).unwrap();
            }
            for i in 0..5 {
                index.delete_vector(i).unwrap();
            }

            // Now there should be some fragmentation
            let frag_after = index.fragmentation();
            assert!(frag_after >= 0.0);
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_clear() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
            assert_eq!(index.index_size(), 2);

            index.clear();

            assert_eq!(index.index_size(), 0);
            assert!(!index.contains(1));
            assert!(!index.contains(2));
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_clear() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

            index.clear();

            assert_eq!(index.index_size(), 0);

            // Should be able to add new vectors after clear
            index.add_vector(&[0.0, 0.0, 1.0, 0.0], 3).unwrap();
            assert_eq!(index.index_size(), 1);
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_data_path() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let index = DiskIndexSingle::<f32>::new(params).unwrap();

            assert_eq!(index.data_path(), &path);
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_backend() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let index = DiskIndexSingle::<f32>::new(params).unwrap();
            assert_eq!(index.backend(), DiskBackend::BruteForce);
        }
        fs::remove_file(&path).ok();

        let path2 = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path2)
                .with_backend(DiskBackend::Vamana);
            let index = DiskIndexSingle::<f32>::new(params).unwrap();
            assert_eq!(index.backend(), DiskBackend::Vamana);
        }
        fs::remove_file(&path2).ok();
    }

    #[test]
    fn test_disk_index_memory_usage() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            let initial_mem = index.memory_usage();
            assert!(initial_mem > 0);

            // Add vectors - memory usage should increase
            for i in 0..100 {
                index.add_vector(&[i as f32, 0.0, 0.0, 0.0], i).unwrap();
            }

            let final_mem = index.memory_usage();
            assert!(final_mem >= initial_mem);
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_info() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

            let info = index.info();
            assert_eq!(info.size, 1);
            assert_eq!(info.dimension, 4);
            assert_eq!(info.index_type, "DiskIndexSingle");
            assert!(info.memory_bytes > 0);
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_contains_and_label_count() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

            assert!(index.contains(1));
            assert!(index.contains(2));
            assert!(!index.contains(3));

            assert_eq!(index.label_count(1), 1);
            assert_eq!(index.label_count(2), 1);
            assert_eq!(index.label_count(3), 0);
        }
        fs::remove_file(&path).ok();
    }

    // ========== Batch Iterator Tests ==========

    #[test]
    fn test_disk_index_batch_iterator() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            for i in 0..10 {
                index.add_vector(&[i as f32, 0.0, 0.0, 0.0], i).unwrap();
            }

            let query = vec![0.0, 0.0, 0.0, 0.0];
            let mut iterator = index.batch_iterator(&query, None).unwrap();

            assert!(iterator.has_next());

            let batch1 = iterator.next_batch(5).unwrap();
            assert_eq!(batch1.len(), 5);

            let batch2 = iterator.next_batch(5).unwrap();
            assert_eq!(batch2.len(), 5);

            // No more results
            assert!(!iterator.has_next());
            assert!(iterator.next_batch(5).is_none());
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_batch_iterator_reset() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            for i in 0..5 {
                index.add_vector(&[i as f32, 0.0, 0.0, 0.0], i).unwrap();
            }

            let query = vec![0.0, 0.0, 0.0, 0.0];
            let mut iterator = index.batch_iterator(&query, None).unwrap();

            // Consume all
            let _ = iterator.next_batch(10);
            assert!(!iterator.has_next());

            // Reset
            iterator.reset();
            assert!(iterator.has_next());

            // Can iterate again
            let batch = iterator.next_batch(5).unwrap();
            assert_eq!(batch.len(), 5);
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_batch_iterator_ordering() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Add vectors at increasing distances from origin
            index.add_vector(&[0.0, 0.0, 0.0, 0.0], 0).unwrap();
            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[2.0, 0.0, 0.0, 0.0], 2).unwrap();
            index.add_vector(&[3.0, 0.0, 0.0, 0.0], 3).unwrap();

            let query = vec![0.0, 0.0, 0.0, 0.0];
            let mut iterator = index.batch_iterator(&query, None).unwrap();

            let batch = iterator.next_batch(4).unwrap();

            // Should be ordered by distance (label 0 first)
            assert_eq!(batch[0].1, 0);
            assert_eq!(batch[1].1, 1);
            assert_eq!(batch[2].1, 2);
            assert_eq!(batch[3].1, 3);
        }
        fs::remove_file(&path).ok();
    }

    // ========== Error Handling Tests ==========

    #[test]
    fn test_disk_index_dimension_mismatch_add() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Wrong dimension
            let result = index.add_vector(&[1.0, 2.0, 3.0], 1);
            assert!(result.is_err());

            match result {
                Err(IndexError::DimensionMismatch { expected, got }) => {
                    assert_eq!(expected, 4);
                    assert_eq!(got, 3);
                }
                _ => panic!("Expected DimensionMismatch error"),
            }
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_dimension_mismatch_query() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

            // Wrong dimension query
            let result = index.top_k_query(&[1.0, 0.0, 0.0], 1, None);
            assert!(result.is_err());
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_delete_not_found() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

            // Delete non-existing label
            let result = index.delete_vector(999);
            assert!(result.is_err());

            match result {
                Err(IndexError::LabelNotFound(label)) => {
                    assert_eq!(label, 999);
                }
                _ => panic!("Expected LabelNotFound error"),
            }
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_capacity_exceeded() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::with_capacity(params, 2).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

            // Third should fail
            let result = index.add_vector(&[0.0, 0.0, 1.0, 0.0], 3);
            assert!(result.is_err());

            match result {
                Err(IndexError::CapacityExceeded { capacity }) => {
                    assert_eq!(capacity, 2);
                }
                _ => panic!("Expected CapacityExceeded error"),
            }
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_empty_query() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Query on empty index
            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 10, None).unwrap();

            assert!(results.is_empty());
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_zero_k_query() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

            // Query with k=0
            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 0, None).unwrap();

            assert!(results.is_empty());
        }
        fs::remove_file(&path).ok();
    }

    // ========== Different Metrics Tests ==========

    #[test]
    fn test_disk_index_inner_product() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::InnerProduct, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // For IP, higher dot product = more similar (lower distance when negated)
            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[0.5, 0.0, 0.0, 0.0], 2).unwrap();
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 3).unwrap();

            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 3, None).unwrap();

            assert_eq!(results.len(), 3);
            // Label 1 should be first (highest IP with query)
            assert_eq!(results.results[0].label, 1);
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_cosine() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::Cosine, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Same direction should have distance ~0
            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[2.0, 0.0, 0.0, 0.0], 2).unwrap(); // Same direction, different magnitude
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 3).unwrap(); // Orthogonal

            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 3, None).unwrap();

            assert_eq!(results.len(), 3);
            // Labels 1 and 2 should have similar (small) distances
            assert!(results.results[0].distance < 0.1);
            assert!(results.results[1].distance < 0.1);
            // Label 3 (orthogonal) should have distance ~1
            assert!(results.results[2].distance > 0.9);
        }
        fs::remove_file(&path).ok();
    }

    // ========== Vamana-specific Additional Tests ==========

    #[test]
    fn test_disk_index_vamana_find_medoid() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Add vectors
            index.add_vector(&[0.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
            index.add_vector(&[0.5, 0.0, 0.0, 0.0], 3).unwrap();

            let medoid = index.find_medoid();
            assert!(medoid.is_some());
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_empty_find_medoid() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana);
            let index = DiskIndexSingle::<f32>::new(params).unwrap();

            // Empty index should return None
            let medoid = index.find_medoid();
            assert!(medoid.is_none());
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_single_vector() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 1, None).unwrap();

            assert_eq!(results.len(), 1);
            assert_eq!(results.results[0].label, 1);
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_vamana_delete_all() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path)
                .with_backend(DiskBackend::Vamana);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

            index.delete_vector(1).unwrap();
            index.delete_vector(2).unwrap();

            assert_eq!(index.index_size(), 0);

            // Query on empty index
            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = index.top_k_query(&query, 10, None).unwrap();
            assert!(results.is_empty());
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_flush() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let mut index = DiskIndexSingle::<f32>::new(params).unwrap();

            index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

            // Flush should succeed
            let result = index.flush();
            assert!(result.is_ok());
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_with_capacity() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let index = DiskIndexSingle::<f32>::with_capacity(params, 100).unwrap();

            assert_eq!(index.index_capacity(), Some(100));
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_no_capacity() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(4, Metric::L2, &path);
            let index = DiskIndexSingle::<f32>::new(params).unwrap();

            assert_eq!(index.index_capacity(), None);
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_disk_index_dimension() {
        let path = temp_path();
        {
            let params = DiskIndexParams::new(128, Metric::L2, &path);
            let index = DiskIndexSingle::<f32>::new(params).unwrap();

            assert_eq!(index.dimension(), 128);
        }
        fs::remove_file(&path).ok();
    }
}

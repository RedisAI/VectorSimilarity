//! Single-value HNSW index implementation.
//!
//! This index stores one vector per label. When adding a vector with
//! an existing label, the old vector is replaced.

use super::{ElementGraphData, HnswCore, HnswParams};
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply, QueryResult};
use crate::types::{DistanceType, IdType, LabelType, VectorElement};
use dashmap::DashMap;
use std::collections::HashMap;

/// Statistics about an HNSW index.
#[derive(Debug, Clone)]
pub struct HnswStats {
    /// Number of vectors in the index.
    pub size: usize,
    /// Number of deleted (but not yet removed) elements.
    pub deleted_count: usize,
    /// Maximum level in the graph.
    pub max_level: usize,
    /// Number of elements at each level.
    pub level_counts: Vec<usize>,
    /// Total number of connections in the graph.
    pub total_connections: usize,
    /// Average connections per element.
    pub avg_connections_per_element: f64,
    /// Approximate memory usage in bytes.
    pub memory_bytes: usize,
}

/// Single-value HNSW index.
///
/// Each label has exactly one associated vector.
///
/// This index now supports concurrent access for parallel insertion
/// via the `add_vector_concurrent` method.
pub struct HnswSingle<T: VectorElement> {
    /// Core HNSW implementation (internally thread-safe).
    pub(crate) core: HnswCore<T>,
    /// Label to internal ID mapping (concurrent hash map).
    label_to_id: DashMap<LabelType, IdType>,
    /// Internal ID to label mapping (concurrent hash map).
    pub(crate) id_to_label: DashMap<IdType, LabelType>,
    /// Number of vectors.
    count: std::sync::atomic::AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
}

impl<T: VectorElement> HnswSingle<T> {
    /// Create a new single-value HNSW index.
    pub fn new(params: HnswParams) -> Self {
        let initial_capacity = params.initial_capacity;
        let core = HnswCore::new(params);

        Self {
            core,
            label_to_id: DashMap::with_capacity(initial_capacity),
            id_to_label: DashMap::with_capacity(initial_capacity),
            count: std::sync::atomic::AtomicUsize::new(0),
            capacity: None,
        }
    }

    /// Create with a maximum capacity.
    pub fn with_capacity(params: HnswParams, max_capacity: usize) -> Self {
        let mut index = Self::new(params);
        index.capacity = Some(max_capacity);
        index
    }

    /// Get the distance metric.
    pub fn metric(&self) -> crate::distance::Metric {
        self.core.params.metric
    }

    /// Get the ef_runtime parameter.
    pub fn ef_runtime(&self) -> usize {
        self.core.params.ef_runtime
    }

    /// Set the ef_runtime parameter.
    pub fn set_ef_runtime(&self, ef: usize) {
        // This is a race condition but acceptable for runtime parameter updates
        // A more robust solution would use an AtomicUsize for ef_runtime
        unsafe {
            let params = &self.core.params as *const _ as *mut super::HnswParams;
            (*params).ef_runtime = ef;
        }
    }

    /// Get the M parameter (max connections per element per layer).
    pub fn m(&self) -> usize {
        self.core.params.m
    }

    /// Get the M_max_0 parameter (max connections at layer 0).
    pub fn m_max_0(&self) -> usize {
        self.core.params.m_max_0
    }

    /// Get the ef_construction parameter.
    pub fn ef_construction(&self) -> usize {
        self.core.params.ef_construction
    }

    /// Check if heuristic neighbor selection is enabled.
    pub fn is_heuristic_enabled(&self) -> bool {
        self.core.params.enable_heuristic
    }

    /// Get the current entry point ID (top-level node).
    pub fn entry_point(&self) -> Option<IdType> {
        let ep = self.core.entry_point.load(std::sync::atomic::Ordering::Relaxed);
        if ep == crate::types::INVALID_ID {
            None
        } else {
            Some(ep)
        }
    }

    /// Get the current maximum level in the graph.
    pub fn max_level(&self) -> usize {
        self.core.max_level.load(std::sync::atomic::Ordering::Relaxed) as usize
    }

    /// Get the number of deleted (but not yet removed) elements.
    pub fn deleted_count(&self) -> usize {
        self.core.graph.iter()
            .filter(|(_, g)| g.meta.deleted)
            .count()
    }

    /// Get detailed statistics about the index.
    pub fn stats(&self) -> HnswStats {
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        let mut level_counts = vec![0usize; self.core.max_level.load(std::sync::atomic::Ordering::Relaxed) as usize + 1];
        let mut total_connections = 0usize;
        let mut deleted_count = 0usize;

        for (_, element) in self.core.graph.iter() {
            if element.meta.deleted {
                deleted_count += 1;
                continue;
            }
            let level = element.meta.level as usize;
            for l in 0..=level {
                if l < level_counts.len() {
                    level_counts[l] += 1;
                }
            }
            for level_link in &element.levels {
                total_connections += level_link.get_neighbors().len();
            }
        }

        let active_count = count.saturating_sub(deleted_count);
        let avg_connections = if active_count > 0 {
            total_connections as f64 / active_count as f64
        } else {
            0.0
        };

        HnswStats {
            size: count,
            deleted_count,
            max_level: self.core.max_level.load(std::sync::atomic::Ordering::Relaxed) as usize,
            level_counts,
            total_connections,
            avg_connections_per_element: avg_connections,
            memory_bytes: self.memory_usage(),
        }
    }

    /// Get the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        // Vector data storage
        let vector_storage = count * self.core.params.dim * std::mem::size_of::<T>();

        // Graph structure (rough estimate)
        let graph_overhead = self.core.graph.len()
            * std::mem::size_of::<Option<super::graph::ElementGraphData>>();

        // Label mappings
        let label_maps = self.label_to_id.len()
            * std::mem::size_of::<(LabelType, IdType)>()
            * 2;

        vector_storage + graph_overhead + label_maps
    }

    /// Get a copy of the vector stored for a given label.
    ///
    /// Returns `None` if the label doesn't exist in the index.
    pub fn get_vector(&self, label: LabelType) -> Option<Vec<T>> {
        let id = *self.label_to_id.get(&label)?;
        self.core.data.get(id).map(|v| v.to_vec())
    }

    /// Get all labels currently in the index.
    pub fn get_labels(&self) -> Vec<LabelType> {
        self.label_to_id.iter().map(|r| *r.key()).collect()
    }

    /// Compute the distance between a stored vector and a query vector.
    ///
    /// Returns `None` if the label doesn't exist.
    pub fn compute_distance(&self, label: LabelType, query: &[T]) -> Option<T::DistanceType> {
        let id = *self.label_to_id.get(&label)?;
        if let Some(stored) = self.core.data.get(id) {
            Some(self.core.dist_fn.compute(stored, query, self.core.params.dim))
        } else {
            None
        }
    }

    /// Get the neighbors of an element at all levels.
    ///
    /// Returns None if the label doesn't exist in the index.
    /// Returns Some(Vec<Vec<LabelType>>) where each inner Vec contains the neighbor labels at that level.
    pub fn get_element_neighbors(&self, label: LabelType) -> Option<Vec<Vec<LabelType>>> {
        let id = *self.label_to_id.get(&label)?;
        self.core.get_element_neighbors_by_id(id)
    }

    /// Clear all vectors from the index, resetting it to empty state.
    pub fn clear(&mut self) {
        use std::sync::atomic::Ordering;

        self.core.data.clear();
        self.core.graph.clear();
        self.core.entry_point.store(crate::types::INVALID_ID, Ordering::Relaxed);
        self.core.max_level.store(0, Ordering::Relaxed);
        self.label_to_id.clear();
        self.id_to_label.clear();
        self.count.store(0, Ordering::Relaxed);
    }

    /// Compact the index by removing gaps from deleted vectors.
    ///
    /// This reorganizes the internal storage and graph structure to reclaim space
    /// from deleted vectors. After compaction, all vectors are stored contiguously.
    ///
    /// **Note**: For HNSW indices, compaction also rebuilds the graph neighbor links
    /// with updated IDs, which can be computationally expensive for large indices.
    ///
    /// # Arguments
    /// * `shrink` - If true, also release unused memory blocks
    ///
    /// # Returns
    /// The number of bytes reclaimed (approximate).
    pub fn compact(&mut self, shrink: bool) -> usize {
        use std::sync::atomic::Ordering;

        let old_capacity = self.core.data.capacity();
        let id_mapping = self.core.data.compact(shrink);

        // Rebuild graph with new IDs
        let mut new_graph: Vec<Option<ElementGraphData>> = (0..id_mapping.len()).map(|_| None).collect();

        for (&old_id, &new_id) in &id_mapping {
            if let Some(old_graph_data) = self.core.graph.get(old_id) {
                // Clone the graph data and update neighbor IDs
                let mut new_graph_data = ElementGraphData::new(
                    old_graph_data.meta.label,
                    old_graph_data.meta.level,
                    self.core.params.m_max_0,
                    self.core.params.m,
                );
                new_graph_data.meta.deleted = old_graph_data.meta.deleted;

                // Update neighbor IDs in each level
                for (level_idx, level_link) in old_graph_data.levels.iter().enumerate() {
                    let old_neighbors = level_link.get_neighbors();
                    let new_neighbors: Vec<IdType> = old_neighbors
                        .iter()
                        .filter_map(|&neighbor_id| id_mapping.get(&neighbor_id).copied())
                        .collect();

                    if level_idx < new_graph_data.levels.len() {
                        new_graph_data.levels[level_idx].set_neighbors(&new_neighbors);
                    }
                }

                new_graph[new_id as usize] = Some(new_graph_data);
            }
        }

        self.core.graph.replace(new_graph);

        // Update entry point
        let old_entry = self.core.entry_point.load(Ordering::Relaxed);
        if old_entry != crate::types::INVALID_ID {
            if let Some(&new_entry) = id_mapping.get(&old_entry) {
                self.core.entry_point.store(new_entry, Ordering::Relaxed);
            } else {
                // Entry point was deleted, find a new one
                let new_entry = self.core.graph.iter()
                    .filter(|(_, g)| !g.meta.deleted)
                    .map(|(id, _)| id)
                    .next()
                    .unwrap_or(crate::types::INVALID_ID);
                self.core.entry_point.store(new_entry, Ordering::Relaxed);
            }
        }

        // Update label_to_id mapping
        for mut entry in self.label_to_id.iter_mut() {
            if let Some(&new_id) = id_mapping.get(entry.value()) {
                *entry.value_mut() = new_id;
            }
        }

        // Rebuild id_to_label mapping
        self.id_to_label.clear();
        for entry in self.label_to_id.iter() {
            self.id_to_label.insert(*entry.value(), *entry.key());
        }

        // Resize visited pool
        if !id_mapping.is_empty() {
            let max_id = id_mapping.values().max().copied().unwrap_or(0) as usize;
            self.core.visited_pool.resize(max_id + 1);
        }

        let new_capacity = self.core.data.capacity();
        let dim = self.core.params.dim;
        let bytes_per_vector = dim * std::mem::size_of::<T>();

        (old_capacity.saturating_sub(new_capacity)) * bytes_per_vector
    }

    /// Get the fragmentation ratio of the index.
    ///
    /// Returns a value between 0.0 (no fragmentation) and 1.0 (all slots are deleted).
    pub fn fragmentation(&self) -> f64 {
        self.core.data.fragmentation()
    }

    /// Add multiple vectors at once.
    ///
    /// Returns the number of vectors successfully added.
    /// Stops on first error and returns what was added so far.
    pub fn add_vectors(
        &mut self,
        vectors: &[(&[T], LabelType)],
    ) -> Result<usize, IndexError> {
        let mut added = 0;
        for &(vector, label) in vectors {
            self.add_vector(vector, label)?;
            added += 1;
        }
        Ok(added)
    }

    /// Add a vector with parallel-safe semantics.
    ///
    /// This method can be called from multiple threads simultaneously.
    /// For single-value indices, if the label already exists, this returns
    /// an error (unlike `add_vector` which replaces the vector).
    ///
    /// # Arguments
    /// * `vector` - The vector to add
    /// * `label` - The label for this vector
    ///
    /// # Returns
    /// * `Ok(1)` if the vector was added successfully
    /// * `Err(IndexError::DuplicateLabel)` if the label already exists
    /// * `Err(IndexError::DimensionMismatch)` if vector dimension is wrong
    /// * `Err(IndexError::CapacityExceeded)` if index is full
    pub fn add_vector_concurrent(&self, vector: &[T], label: LabelType) -> Result<usize, IndexError> {
        if vector.len() != self.core.params.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.core.params.dim,
                got: vector.len(),
            });
        }

        // Check if label already exists (atomic check)
        if self.label_to_id.contains_key(&label) {
            return Err(IndexError::DuplicateLabel(label));
        }

        // Check capacity
        if let Some(cap) = self.capacity {
            if self.count.load(std::sync::atomic::Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Add the vector to storage
        let id = self.core
            .add_vector_concurrent(vector)
            .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;

        // Try to insert the label mapping atomically
        // DashMap's entry API provides atomic insert-if-not-exists
        use dashmap::mapref::entry::Entry;
        match self.label_to_id.entry(label) {
            Entry::Occupied(_) => {
                // Another thread inserted this label, rollback
                self.core.data.mark_deleted_concurrent(id);
                return Err(IndexError::DuplicateLabel(label));
            }
            Entry::Vacant(entry) => {
                entry.insert(id);
            }
        }

        // Insert into id_to_label
        self.id_to_label.insert(id, label);

        // Insert into graph
        self.core.insert_concurrent(id, label);

        self.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(1)
    }

    /// Batch insert multiple vectors using layer-at-a-time construction.
    ///
    /// This method provides significant speedup for bulk insertions by:
    /// 1. Adding all vectors to storage in parallel
    /// 2. Pre-assigning random levels for all vectors
    /// 3. Building the graph layer by layer with parallel neighbor search
    ///
    /// # Arguments
    /// * `vectors` - Slice of vector data (flattened, each vector has `dim` elements)
    /// * `labels` - Slice of labels corresponding to each vector
    /// * `dim` - Dimension of each vector
    ///
    /// # Returns
    /// * `Ok(count)` - Number of vectors successfully inserted
    /// * `Err` - If dimension mismatch or other errors occur
    pub fn add_vectors_batch(
        &self,
        vectors: &[T],
        labels: &[LabelType],
        dim: usize,
    ) -> Result<usize, IndexError> {
        use rayon::prelude::*;

        if dim != self.core.params.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.core.params.dim,
                got: dim,
            });
        }

        let num_vectors = labels.len();
        if vectors.len() != num_vectors * dim {
            return Err(IndexError::Internal(format!(
                "Vector data length {} does not match num_vectors {} * dim {}",
                vectors.len(), num_vectors, dim
            )));
        }

        if num_vectors == 0 {
            return Ok(0);
        }

        // Check capacity
        if let Some(cap) = self.capacity {
            let current = self.count.load(std::sync::atomic::Ordering::Relaxed);
            if current + num_vectors > cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Phase 1: Pre-generate random levels (single lock acquisition)
        let levels = self.core.generate_random_levels_batch(num_vectors);

        // Phase 2: Add all vectors to storage and create assignments (parallel)
        let assignments: Vec<Result<(IdType, u8, LabelType), IndexError>> = (0..num_vectors)
            .into_par_iter()
            .map(|i| {
                let label = labels[i];

                // Check for duplicate label
                if self.label_to_id.contains_key(&label) {
                    return Err(IndexError::DuplicateLabel(label));
                }

                // Add vector to storage
                let vec_start = i * dim;
                let vec_end = vec_start + dim;
                let vector = &vectors[vec_start..vec_end];

                let id = self.core
                    .add_vector_concurrent(vector)
                    .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;

                // Insert label mappings
                use dashmap::mapref::entry::Entry;
                match self.label_to_id.entry(label) {
                    Entry::Occupied(_) => {
                        self.core.data.mark_deleted_concurrent(id);
                        return Err(IndexError::DuplicateLabel(label));
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(id);
                    }
                }
                self.id_to_label.insert(id, label);

                Ok((id, levels[i], label))
            })
            .collect();

        // Filter successful assignments and count errors
        let mut successful: Vec<(IdType, u8, LabelType)> = Vec::with_capacity(num_vectors);
        let mut errors = 0;
        for result in assignments {
            match result {
                Ok(assignment) => successful.push(assignment),
                Err(_) => errors += 1,
            }
        }

        if successful.is_empty() {
            return Ok(0);
        }

        // Phase 3: Build graph using batch construction
        self.core.insert_batch(&successful);

        // Update count
        let inserted = successful.len();
        self.count.fetch_add(inserted, std::sync::atomic::Ordering::Relaxed);

        if errors > 0 {
            // Some vectors failed but others succeeded
            Ok(inserted)
        } else {
            Ok(inserted)
        }
    }
}

impl<T: VectorElement> VecSimIndex for HnswSingle<T> {
    type DataType = T;
    type DistType = T::DistanceType;

    fn add_vector(&mut self, vector: &[T], label: LabelType) -> Result<usize, IndexError> {
        if vector.len() != self.core.params.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.core.params.dim,
                got: vector.len(),
            });
        }

        // Check if label already exists
        if let Some(existing_id) = self.label_to_id.get(&label).map(|r| *r) {
            // Mark old vector as deleted
            self.core.mark_deleted_concurrent(existing_id);
            self.id_to_label.remove(&existing_id);

            // Add new vector
            let new_id = self.core
                .add_vector_concurrent(vector)
                .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;
            self.core.insert_concurrent(new_id, label);

            // Update mappings
            self.label_to_id.insert(label, new_id);
            self.id_to_label.insert(new_id, label);

            return Ok(0); // Replacement, not a new vector
        }

        // Check capacity
        if let Some(cap) = self.capacity {
            if self.count.load(std::sync::atomic::Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Add new vector
        let id = self.core
            .add_vector_concurrent(vector)
            .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;
        self.core.insert_concurrent(id, label);

        // Update mappings
        self.label_to_id.insert(label, id);
        self.id_to_label.insert(id, label);

        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        if let Some((_, id)) = self.label_to_id.remove(&label) {
            self.core.mark_deleted_concurrent(id);
            self.id_to_label.remove(&id);
            self.count
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            Ok(1)
        } else {
            Err(IndexError::LabelNotFound(label))
        }
    }

    fn top_k_query(
        &self,
        query: &[T],
        k: usize,
        params: Option<&QueryParams>,
    ) -> Result<QueryReply<T::DistanceType>, QueryError> {
        if query.len() != self.core.params.dim {
            return Err(QueryError::DimensionMismatch {
                expected: self.core.params.dim,
                got: query.len(),
            });
        }

        let ef = params
            .and_then(|p| p.ef_runtime)
            .unwrap_or(self.core.params.ef_runtime);

        // Build filter if needed
        let has_filter = params.is_some_and(|p| p.filter.is_some());
        let id_label_map: HashMap<IdType, LabelType> = if has_filter {
            self.id_to_label.iter().map(|r| (*r.key(), *r.value())).collect()
        } else {
            HashMap::new()
        };

        let filter_fn: Option<Box<dyn Fn(IdType) -> bool>> = if let Some(p) = params {
            if let Some(ref f) = p.filter {
                let f = f.as_ref();
                Some(Box::new(move |id: IdType| {
                    id_label_map.get(&id).is_some_and(|&label| f(label))
                }))
            } else {
                None
            }
        } else {
            None
        };

        let results = self.core.search(query, k, ef, filter_fn.as_ref().map(|f| f.as_ref()));

        // Look up labels for results
        let mut reply = QueryReply::with_capacity(results.len());
        for (id, dist) in results {
            if let Some(label_ref) = self.id_to_label.get(&id) {
                reply.push(QueryResult::new(*label_ref, dist));
            }
        }

        Ok(reply)
    }

    fn range_query(
        &self,
        query: &[T],
        radius: T::DistanceType,
        params: Option<&QueryParams>,
    ) -> Result<QueryReply<T::DistanceType>, QueryError> {
        if query.len() != self.core.params.dim {
            return Err(QueryError::DimensionMismatch {
                expected: self.core.params.dim,
                got: query.len(),
            });
        }

        // Get epsilon from params or use default (1.0 = 100% expansion)
        // Note: C++ uses 0.01 (1%) but the Rust HNSW graph structure may require
        // a larger epsilon to ensure reliable range search results. With 100%
        // expansion, the algorithm explores candidates up to 2x the dynamic range.
        let epsilon = params
            .and_then(|p| p.epsilon)
            .unwrap_or(1.0);

        // Build filter if needed
        let has_filter = params.is_some_and(|p| p.filter.is_some());
        let id_label_map: HashMap<IdType, LabelType> = if has_filter {
            self.id_to_label.iter().map(|r| (*r.key(), *r.value())).collect()
        } else {
            HashMap::new()
        };

        let filter_fn: Option<Box<dyn Fn(IdType) -> bool>> = if let Some(p) = params {
            if let Some(ref f) = p.filter {
                let f = f.as_ref();
                Some(Box::new(move |id: IdType| {
                    id_label_map.get(&id).is_some_and(|&label| f(label))
                }))
            } else {
                None
            }
        } else {
            None
        };

        // Use epsilon-neighborhood search for efficient range query
        let results = self.core.search_range(
            query,
            radius,
            epsilon,
            filter_fn.as_ref().map(|f| f.as_ref()),
        );

        // Look up labels for results
        let mut reply = QueryReply::with_capacity(results.len());
        for (id, dist) in results {
            if let Some(label_ref) = self.id_to_label.get(&id) {
                reply.push(QueryResult::new(*label_ref, dist));
            }
        }

        // Results are already sorted by search_range, but ensure order
        reply.sort_by_distance();
        Ok(reply)
    }

    fn index_size(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn index_capacity(&self) -> Option<usize> {
        self.capacity
    }

    fn dimension(&self) -> usize {
        self.core.params.dim
    }

    fn batch_iterator<'a>(
        &'a self,
        query: &[T],
        params: Option<&QueryParams>,
    ) -> Result<Box<dyn BatchIterator<DistType = T::DistanceType> + 'a>, QueryError> {
        if query.len() != self.core.params.dim {
            return Err(QueryError::DimensionMismatch {
                expected: self.core.params.dim,
                got: query.len(),
            });
        }

        Ok(Box::new(
            super::batch_iterator::HnswSingleBatchIterator::new(self, query.to_vec(), params.cloned()),
        ))
    }

    fn info(&self) -> IndexInfo {
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        // Base overhead for the struct itself and internal data structures
        let base_overhead = std::mem::size_of::<Self>() + std::mem::size_of::<HnswCore<T>>();

        IndexInfo {
            size: count,
            capacity: self.capacity,
            dimension: self.core.params.dim,
            index_type: "HnswSingle",
            memory_bytes: base_overhead
                + count * self.core.params.dim * std::mem::size_of::<T>()
                + self.core.graph.len() * std::mem::size_of::<Option<super::graph::ElementGraphData>>()
                + self.label_to_id.len() * std::mem::size_of::<(LabelType, IdType)>(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.label_to_id.contains_key(&label)
    }

    fn label_count(&self, label: LabelType) -> usize {
        if self.contains(label) {
            1
        } else {
            0
        }
    }
}

unsafe impl<T: VectorElement> Send for HnswSingle<T> {}
unsafe impl<T: VectorElement> Sync for HnswSingle<T> {}

// Serialization support
impl<T: VectorElement> HnswSingle<T> {
    /// Save the index to a writer.
    pub fn save<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> crate::serialization::SerializationResult<()> {
        use crate::serialization::*;
        use std::sync::atomic::Ordering;

        let count = self.count.load(Ordering::Relaxed);

        // Write header
        let header = IndexHeader::new(
            IndexTypeId::HnswSingle,
            T::data_type_id(),
            self.core.params.metric,
            self.core.params.dim,
            count,
        );
        header.write(writer)?;

        // Write HNSW-specific params
        write_usize(writer, self.core.params.m)?;
        write_usize(writer, self.core.params.m_max_0)?;
        write_usize(writer, self.core.params.ef_construction)?;
        write_usize(writer, self.core.params.ef_runtime)?;
        write_u8(writer, if self.core.params.enable_heuristic { 1 } else { 0 })?;

        // Write graph metadata
        let entry_point = self.core.entry_point.load(Ordering::Relaxed);
        let max_level = self.core.max_level.load(Ordering::Relaxed);
        write_u32(writer, entry_point)?;
        write_u32(writer, max_level)?;

        // Write capacity
        write_u8(writer, if self.capacity.is_some() { 1 } else { 0 })?;
        if let Some(cap) = self.capacity {
            write_usize(writer, cap)?;
        }

        // Write label_to_id mapping
        write_usize(writer, self.label_to_id.len())?;
        for entry in self.label_to_id.iter() {
            write_u64(writer, *entry.key())?;
            write_u32(writer, *entry.value())?;
        }

        // Write graph structure
        let graph_len = self.core.graph.len();
        write_usize(writer, graph_len)?;
        for id in 0..graph_len as u32 {
            if let Some(graph_data) = self.core.graph.get(id) {
                write_u8(writer, 1)?; // Present flag

                // Write metadata
                write_u64(writer, graph_data.meta.label)?;
                write_u8(writer, graph_data.meta.level)?;
                write_u8(writer, if graph_data.meta.deleted { 1 } else { 0 })?;

                // Write levels
                write_usize(writer, graph_data.levels.len())?;
                for level_links in &graph_data.levels {
                    let neighbors = level_links.get_neighbors();
                    write_usize(writer, neighbors.len())?;
                    for neighbor in neighbors {
                        write_u32(writer, neighbor)?;
                    }
                }

                // Write vector data
                if let Some(vector) = self.core.data.get(id) {
                    for v in vector {
                        v.write_to(writer)?;
                    }
                }
            } else {
                write_u8(writer, 0)?; // Not present
            }
        }

        Ok(())
    }

    /// Load the index from a reader.
    pub fn load<R: std::io::Read>(reader: &mut R) -> crate::serialization::SerializationResult<Self> {
        use crate::serialization::*;
        use super::graph::ElementGraphData;
        use std::sync::atomic::Ordering;

        // Read and validate header
        let header = IndexHeader::read(reader)?;

        if header.index_type != IndexTypeId::HnswSingle {
            return Err(SerializationError::IndexTypeMismatch {
                expected: "HnswSingle".to_string(),
                got: header.index_type.as_str().to_string(),
            });
        }

        if header.data_type != T::data_type_id() {
            return Err(SerializationError::InvalidData(
                format!("Expected {:?} data type, got {:?}", T::data_type_id(), header.data_type),
            ));
        }

        // Read HNSW-specific params
        let m = read_usize(reader)?;
        let m_max_0 = read_usize(reader)?;
        let ef_construction = read_usize(reader)?;
        let ef_runtime = read_usize(reader)?;
        let enable_heuristic = read_u8(reader)? != 0;

        // Read graph metadata
        let entry_point = read_u32(reader)?;
        let max_level = read_u32(reader)?;

        // Create params and index
        let params = HnswParams {
            dim: header.dimension,
            metric: header.metric,
            m,
            m_max_0,
            ef_construction,
            ef_runtime,
            initial_capacity: header.count.max(1024),
            enable_heuristic,
            seed: None, // Seed not preserved in serialization
        };
        let mut index = Self::new(params);

        // Read capacity
        let has_capacity = read_u8(reader)? != 0;
        if has_capacity {
            index.capacity = Some(read_usize(reader)?);
        }

        // Read label_to_id mapping
        let label_to_id_len = read_usize(reader)?;
        for _ in 0..label_to_id_len {
            let label = read_u64(reader)?;
            let id = read_u32(reader)?;
            index.label_to_id.insert(label, id);
            index.id_to_label.insert(id, label);
        }

        // Read graph structure
        let graph_len = read_usize(reader)?;
        let dim = header.dimension;

        // Set entry point and max level
        index.core.entry_point.store(entry_point, Ordering::Relaxed);
        index.core.max_level.store(max_level, Ordering::Relaxed);

        for id in 0..graph_len {
            let present = read_u8(reader)? != 0;
            if !present {
                continue;
            }

            // Read metadata
            let label = read_u64(reader)?;
            let level = read_u8(reader)?;
            let deleted = read_u8(reader)? != 0;

            // Read levels
            let num_levels = read_usize(reader)?;
            let mut graph_data = ElementGraphData::new(label, level, m_max_0, m);
            graph_data.meta.deleted = deleted;

            for level_idx in 0..num_levels {
                let num_neighbors = read_usize(reader)?;
                let mut neighbors = Vec::with_capacity(num_neighbors);
                for _ in 0..num_neighbors {
                    neighbors.push(read_u32(reader)?);
                }
                if level_idx < graph_data.levels.len() {
                    graph_data.levels[level_idx].set_neighbors(&neighbors);
                }
            }

            // Read vector data
            let mut vector = vec![T::zero(); dim];
            for v in &mut vector {
                *v = T::read_from(reader)?;
            }

            // Add vector to data storage
            index.core.data.add_concurrent(&vector).ok_or_else(|| {
                SerializationError::DataCorruption("Failed to add vector during deserialization".to_string())
            })?;

            // Store graph data
            index.core.graph.set(id as IdType, graph_data);
        }

        // Resize visited pool
        if graph_len > 0 {
            index.core.visited_pool.resize(graph_len);
        }

        index.count.store(header.count, Ordering::Relaxed);

        Ok(index)
    }

    /// Save the index to a file.
    pub fn save_to_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> crate::serialization::SerializationResult<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        self.save(&mut writer)
    }

    /// Load the index from a file.
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> crate::serialization::SerializationResult<Self> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::load(&mut reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_hnsw_single_basic() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswSingle::<f32>::new(params);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];

        index.add_vector(&v1, 1).unwrap();
        index.add_vector(&v2, 2).unwrap();
        index.add_vector(&v3, 3).unwrap();

        assert_eq!(index.index_size(), 3);

        // Query for nearest neighbor
        let query = vec![1.0, 0.1, 0.0, 0.0];
        let results = index.top_k_query(&query, 2, None).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results.results[0].label, 1); // Closest to v1
    }

    #[test]
    fn test_hnsw_single_large() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(8)
            .with_ef_construction(50)
            .with_ef_runtime(20);
        let mut index = HnswSingle::<f32>::new(params);

        // Add many vectors
        for i in 0..100 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, i as u64).unwrap();
        }

        assert_eq!(index.index_size(), 100);

        // Query should find closest
        let query = vec![50.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 5, None).unwrap();

        assert_eq!(results.len(), 5);
        // Exact match should be first
        assert_eq!(results.results[0].label, 50);
    }

    #[test]
    fn test_hnsw_single_delete() {
        let params = HnswParams::new(4, Metric::L2);
        let mut index = HnswSingle::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 2);

        index.delete_vector(1).unwrap();
        assert_eq!(index.index_size(), 1);

        // Deleted vector should not appear in results
        let results = index
            .top_k_query(&vec![1.0, 0.0, 0.0, 0.0], 10, None)
            .unwrap();

        for result in &results.results {
            assert_ne!(result.label, 1);
        }
    }

    #[test]
    fn test_hnsw_single_serialization() {
        use std::io::Cursor;

        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswSingle::<f32>::new(params);

        // Add some vectors
        for i in 0..20 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, i as u64).unwrap();
        }

        assert_eq!(index.index_size(), 20);

        // Serialize
        let mut buffer = Vec::new();
        index.save(&mut buffer).unwrap();

        // Deserialize
        let mut cursor = Cursor::new(buffer);
        let loaded = HnswSingle::<f32>::load(&mut cursor).unwrap();

        // Verify
        assert_eq!(loaded.index_size(), 20);
        assert_eq!(loaded.dimension(), 4);

        // Query should work the same
        let query = vec![5.0, 0.0, 0.0, 0.0];
        let results = loaded.top_k_query(&query, 3, None).unwrap();
        assert!(!results.is_empty());
        // Label 5 should be closest
        assert_eq!(results.results[0].label, 5);
    }

    #[test]
    fn test_hnsw_single_compact() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswSingle::<f32>::new(params);

        // Add vectors
        for i in 0..10 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, i as u64).unwrap();
        }

        assert_eq!(index.index_size(), 10);

        // Delete some vectors (labels 2, 4, 6, 8)
        index.delete_vector(2).unwrap();
        index.delete_vector(4).unwrap();
        index.delete_vector(6).unwrap();
        index.delete_vector(8).unwrap();

        assert_eq!(index.index_size(), 6);
        assert!(index.fragmentation() > 0.0);

        // Compact the index
        index.compact(true);

        // Verify fragmentation is gone
        assert!((index.fragmentation() - 0.0).abs() < 0.01);
        assert_eq!(index.index_size(), 6);

        // Verify queries still work correctly
        let query = vec![5.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 3, None).unwrap();

        assert!(!results.is_empty());
        // Label 5 should be closest
        assert_eq!(results.results[0].label, 5);

        // Verify we can find other remaining vectors
        let query2 = vec![0.0, 0.0, 0.0, 0.0];
        let results2 = index.top_k_query(&query2, 1, None).unwrap();
        assert_eq!(results2.results[0].label, 0);

        // Deleted labels should not appear in any query
        let all_results = index.top_k_query(&query, 10, None).unwrap();
        for result in &all_results.results {
            assert!(result.label != 2 && result.label != 4 && result.label != 6 && result.label != 8);
        }
    }
}

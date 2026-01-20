//! Multi-value HNSW index implementation.
//!
//! This index allows multiple vectors per label.

use super::{ElementGraphData, HnswCore, HnswParams};
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply, QueryResult};
use crate::types::{DistanceType, IdType, LabelType, VectorElement};
use dashmap::{DashMap, DashSet};
use std::collections::HashMap;

/// Multi-value HNSW index.
///
/// Each label can have multiple associated vectors.
pub struct HnswMulti<T: VectorElement> {
    /// Core HNSW implementation.
    pub(crate) core: HnswCore<T>,
    /// Label to set of internal IDs mapping.
    label_to_ids: DashMap<LabelType, DashSet<IdType>>,
    /// Internal ID to label mapping.
    pub(crate) id_to_label: DashMap<IdType, LabelType>,
    /// Number of vectors.
    count: std::sync::atomic::AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
}

impl<T: VectorElement> HnswMulti<T> {
    /// Create a new multi-value HNSW index.
    pub fn new(params: HnswParams) -> Self {
        let initial_capacity = params.initial_capacity;
        let core = HnswCore::new(params);

        Self {
            core,
            label_to_ids: DashMap::with_capacity(initial_capacity / 2),
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
    pub fn set_ef_runtime(&mut self, ef: usize) {
        self.core.params.ef_runtime = ef;
    }

    /// Get copies of all vectors stored for a given label.
    ///
    /// Returns `None` if the label doesn't exist in the index.
    pub fn get_vectors(&self, label: LabelType) -> Option<Vec<Vec<T>>> {
        let ids_ref = self.label_to_ids.get(&label)?;
        let vectors: Vec<Vec<T>> = ids_ref
            .iter()
            .filter_map(|id| self.core.data.get(*id).map(|v| v.to_vec()))
            .collect();
        if vectors.is_empty() {
            None
        } else {
            Some(vectors)
        }
    }

    /// Get all labels currently in the index.
    pub fn get_labels(&self) -> Vec<LabelType> {
        self.label_to_ids.iter().map(|r| *r.key()).collect()
    }

    /// Compute the minimum distance between any stored vector for a label and a query vector.
    ///
    /// Returns `None` if the label doesn't exist.
    pub fn compute_distance(&self, label: LabelType, query: &[T]) -> Option<T::DistanceType> {
        let ids_ref = self.label_to_ids.get(&label)?;
        ids_ref
            .iter()
            .filter_map(|id| {
                self.core.data.get(*id).map(|stored| {
                    self.core.dist_fn.compute(stored, query, self.core.params.dim)
                })
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        // Vector data storage
        let vector_storage = count * self.core.params.dim * std::mem::size_of::<T>();

        // Graph structure (rough estimate)
        let graph_overhead = self.core.graph.len()
            * std::mem::size_of::<Option<super::graph::ElementGraphData>>();

        // Label mappings (rough estimate with DashMap)
        let label_maps = self.label_to_ids.len()
            * std::mem::size_of::<(LabelType, DashSet<IdType>)>()
            + self.id_to_label.len() * std::mem::size_of::<(IdType, LabelType)>();

        vector_storage + graph_overhead + label_maps
    }

    /// Clear all vectors from the index, resetting it to empty state.
    pub fn clear(&mut self) {
        use std::sync::atomic::Ordering;

        self.core.data.clear();
        self.core.graph.clear();
        self.core.entry_point.store(crate::types::INVALID_ID, Ordering::Relaxed);
        self.core.max_level.store(0, Ordering::Relaxed);
        self.label_to_ids.clear();
        self.id_to_label.clear();
        self.count.store(0, Ordering::Relaxed);
    }

    /// Get the neighbors of an element in the HNSW graph by label.
    ///
    /// For multi-value indices, this returns the neighbors of the first internal ID
    /// associated with the label. Returns a vector of vectors, where each inner vector
    /// contains the neighbor labels for that level (level 0 first).
    /// Returns None if the label doesn't exist.
    pub fn get_element_neighbors(&self, label: LabelType) -> Option<Vec<Vec<LabelType>>> {
        // Get the first internal ID for this label
        let ids = self.label_to_ids.get(&label)?;
        let first_id = *ids.iter().next()?;
        self.core.get_element_neighbors_by_id(first_id)
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

        // Update label_to_ids mapping - collect keys first to avoid holding the iter
        let labels: Vec<LabelType> = self.label_to_ids.iter().map(|r| *r.key()).collect();
        for label in labels {
            if let Some(mut ids_ref) = self.label_to_ids.get_mut(&label) {
                let new_ids: DashSet<IdType> = ids_ref
                    .iter()
                    .filter_map(|id| id_mapping.get(&*id).copied())
                    .collect();
                *ids_ref = new_ids;
            }
        }

        // Remove labels with no remaining vectors
        self.label_to_ids.retain(|_, ids| !ids.is_empty());

        // Rebuild id_to_label mapping
        let old_id_to_label: HashMap<IdType, LabelType> = self.id_to_label.iter()
            .map(|r| (*r.key(), *r.value()))
            .collect();
        self.id_to_label.clear();
        for (&old_id, &new_id) in &id_mapping {
            if let Some(&label) = old_id_to_label.get(&old_id) {
                self.id_to_label.insert(new_id, label);
            }
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
    /// Can be called from multiple threads simultaneously.
    pub fn add_vector_concurrent(&self, vector: &[T], label: LabelType) -> Result<usize, IndexError> {
        if vector.len() != self.core.params.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.core.params.dim,
                got: vector.len(),
            });
        }

        // Check capacity
        if let Some(cap) = self.capacity {
            if self.count.load(std::sync::atomic::Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Add the vector concurrently
        let id = self.core
            .add_vector_concurrent(vector)
            .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;
        self.core.insert_concurrent(id, label);

        // Update mappings using DashMap
        self.label_to_ids
            .entry(label)
            .or_insert_with(DashSet::new)
            .insert(id);
        self.id_to_label.insert(id, label);

        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(1)
    }
}

impl<T: VectorElement> VecSimIndex for HnswMulti<T> {
    type DataType = T;
    type DistType = T::DistanceType;

    fn add_vector(&mut self, vector: &[T], label: LabelType) -> Result<usize, IndexError> {
        if vector.len() != self.core.params.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.core.params.dim,
                got: vector.len(),
            });
        }

        // Check capacity
        if let Some(cap) = self.capacity {
            if self.count.load(std::sync::atomic::Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Add the vector
        let id = self.core
            .add_vector(vector)
            .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;
        self.core.insert(id, label);

        // Update mappings using DashMap
        self.label_to_ids
            .entry(label)
            .or_insert_with(DashSet::new)
            .insert(id);
        self.id_to_label.insert(id, label);

        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        if let Some((_, ids)) = self.label_to_ids.remove(&label) {
            let count = ids.len();

            for id in ids.iter() {
                self.core.mark_deleted(*id);
                self.id_to_label.remove(&*id);
            }

            self.count
                .fetch_sub(count, std::sync::atomic::Ordering::Relaxed);
            Ok(count)
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

        let base_ef = params
            .and_then(|p| p.ef_runtime)
            .unwrap_or(self.core.params.ef_runtime);

        // Get the id_to_label mapping for label-aware search
        let id_to_label: HashMap<IdType, LabelType> = self.id_to_label
            .iter()
            .map(|r| (*r.key(), *r.value()))
            .collect();

        // For multi-value indices, we need a higher ef to explore enough unique labels.
        // The label heap will track unique labels, so ef controls how many labels we track.
        // Use ef * avg_per_label to compensate for label clustering.
        let total_vectors = self.count.load(std::sync::atomic::Ordering::Relaxed);
        let num_labels = self.label_to_ids.len();
        let avg_per_label = if num_labels > 0 {
            (total_vectors / num_labels).max(1)
        } else {
            1
        };
        // Scale ef by avg_per_label * 2 to ensure we find enough unique labels
        // The extra 2x factor helps compensate for graph clustering effects
        let ef = (base_ef * avg_per_label * 2).min(num_labels).max(base_ef);

        // Build filter function by wrapping the reference
        let filter_ref: Option<&dyn Fn(LabelType) -> bool> = if let Some(p) = params {
            p.filter.as_ref().map(|f| f.as_ref() as &dyn Fn(LabelType) -> bool)
        } else {
            None
        };

        // Use label-aware search that tracks unique labels during graph traversal
        let results = self.core.search_multi(
            query,
            k,
            ef,
            &id_to_label,
            filter_ref,
        );

        // Convert to QueryReply
        let mut reply = QueryReply::with_capacity(results.len());
        for (label, dist) in results {
            reply.push(QueryResult::new(label, dist));
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

        // For multi-value index, deduplicate by label and keep best distance per label
        let mut label_best: HashMap<LabelType, T::DistanceType> = HashMap::new();

        for (id, dist) in results {
            if let Some(label_ref) = self.id_to_label.get(&id) {
                let label = *label_ref;
                label_best
                    .entry(label)
                    .and_modify(|best| {
                        if dist.to_f64() < best.to_f64() {
                            *best = dist;
                        }
                    })
                    .or_insert(dist);
            }
        }

        // Convert to QueryReply
        let mut reply = QueryReply::with_capacity(label_best.len());
        for (label, dist) in label_best {
            reply.push(QueryResult::new(label, dist));
        }

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
            super::batch_iterator::HnswMultiBatchIterator::new(self, query.to_vec(), params.cloned()),
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
            index_type: "HnswMulti",
            memory_bytes: base_overhead
                + count * self.core.params.dim * std::mem::size_of::<T>()
                + self.core.graph.len() * std::mem::size_of::<Option<super::graph::ElementGraphData>>()
                + self.label_to_ids.len()
                    * std::mem::size_of::<(LabelType, DashSet<IdType>)>(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.label_to_ids.contains_key(&label)
    }

    fn label_count(&self, label: LabelType) -> usize {
        self.label_to_ids
            .get(&label)
            .map_or(0, |ids| ids.len())
    }
}

unsafe impl<T: VectorElement> Send for HnswMulti<T> {}
unsafe impl<T: VectorElement> Sync for HnswMulti<T> {}

// Serialization support
impl<T: VectorElement> HnswMulti<T> {
    /// Save the index to a writer.
    pub fn save<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> crate::serialization::SerializationResult<()> {
        use crate::serialization::*;
        use std::sync::atomic::Ordering;

        let graph_len = self.core.graph.len();
        let count = self.count.load(Ordering::Relaxed);

        // Write header
        let header = IndexHeader::new(
            IndexTypeId::HnswMulti,
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

        // Write label_to_ids mapping (label -> set of IDs)
        write_usize(writer, self.label_to_ids.len())?;
        for entry in self.label_to_ids.iter() {
            let label = *entry.key();
            let ids = entry.value();
            write_u64(writer, label)?;
            write_usize(writer, ids.len())?;
            for id in ids.iter() {
                write_u32(writer, *id)?;
            }
        }

        // Write graph structure
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

        if header.index_type != IndexTypeId::HnswMulti {
            return Err(SerializationError::IndexTypeMismatch {
                expected: "HnswMulti".to_string(),
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

        // Read label_to_ids mapping into DashMap
        let label_to_ids_len = read_usize(reader)?;
        for _ in 0..label_to_ids_len {
            let label = read_u64(reader)?;
            let num_ids = read_usize(reader)?;
            let ids: DashSet<IdType> = DashSet::with_capacity(num_ids);
            for _ in 0..num_ids {
                ids.insert(read_u32(reader)?);
            }
            index.label_to_ids.insert(label, ids);
        }

        // Build id_to_label from label_to_ids
        for entry in index.label_to_ids.iter() {
            let label = *entry.key();
            for id in entry.value().iter() {
                index.id_to_label.insert(*id, label);
            }
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
            index.core.data.add(&vector).ok_or_else(|| {
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
    fn test_hnsw_multi_basic() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Add multiple vectors with same label
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 3);
        assert_eq!(index.label_count(1), 2);
        assert_eq!(index.label_count(2), 1);
    }

    #[test]
    fn test_hnsw_multi_delete() {
        let params = HnswParams::new(4, Metric::L2);
        let mut index = HnswMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 3);

        let deleted = index.delete_vector(1).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(index.index_size(), 1);
    }

    #[test]
    fn test_hnsw_multi_serialization() {
        use std::io::Cursor;

        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Add multiple vectors with same label
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();
        index.add_vector(&vec![0.0, 0.0, 1.0, 0.0], 3).unwrap();

        assert_eq!(index.index_size(), 4);
        assert_eq!(index.label_count(1), 2);

        // Serialize
        let mut buffer = Vec::new();
        index.save(&mut buffer).unwrap();

        // Deserialize
        let mut cursor = Cursor::new(buffer);
        let loaded = HnswMulti::<f32>::load(&mut cursor).unwrap();

        // Verify
        assert_eq!(loaded.index_size(), 4);
        assert_eq!(loaded.dimension(), 4);
        assert_eq!(loaded.label_count(1), 2);
        assert_eq!(loaded.label_count(2), 1);
        assert_eq!(loaded.label_count(3), 1);

        // Query should work the same
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = loaded.top_k_query(&query, 3, None).unwrap();
        assert!(!results.is_empty());
        // First result should be one of the label 1 vectors
        assert_eq!(results.results[0].label, 1);
    }

    #[test]
    fn test_hnsw_multi_compact() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Add multiple vectors, some with same label
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();
        index.add_vector(&vec![0.0, 0.0, 1.0, 0.0], 3).unwrap();
        index.add_vector(&vec![0.0, 0.0, 0.0, 1.0], 4).unwrap();

        assert_eq!(index.index_size(), 5);
        assert_eq!(index.label_count(1), 2);

        // Delete labels 2 and 3
        index.delete_vector(2).unwrap();
        index.delete_vector(3).unwrap();

        assert_eq!(index.index_size(), 3);
        assert!(index.fragmentation() > 0.0);

        // Compact the index
        index.compact(true);

        // Verify fragmentation is gone
        assert!((index.fragmentation() - 0.0).abs() < 0.01);
        assert_eq!(index.index_size(), 3);
        assert_eq!(index.label_count(1), 2); // Both vectors for label 1 remain

        // Verify queries still work correctly
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 3, None).unwrap();

        // After deduplication by label, we can only have 2 unique labels (1 and 4)
        assert_eq!(results.len(), 2);
        // Top results should include label 1 vectors
        assert_eq!(results.results[0].label, 1);

        // Verify we can still find v4
        let query2 = vec![0.0, 0.0, 0.0, 1.0];
        let results2 = index.top_k_query(&query2, 1, None).unwrap();
        assert_eq!(results2.results[0].label, 4);

        // Deleted labels should not appear
        let all_results = index.top_k_query(&query, 10, None).unwrap();
        for result in &all_results.results {
            assert!(result.label == 1 || result.label == 4);
        }
    }

    #[test]
    fn test_hnsw_multi_multiple_labels() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Add multiple vectors per label for multiple labels
        for label in 1..=5u64 {
            for i in 0..3 {
                let v = vec![label as f32, i as f32, 0.0, 0.0];
                index.add_vector(&v, label).unwrap();
            }
        }

        assert_eq!(index.index_size(), 15);
        for label in 1..=5u64 {
            assert_eq!(index.label_count(label), 3);
        }
    }

    #[test]
    fn test_hnsw_multi_get_vectors() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![2.0, 0.0, 0.0, 0.0];
        index.add_vector(&v1, 1).unwrap();
        index.add_vector(&v2, 1).unwrap();

        let vectors = index.get_vectors(1).unwrap();
        assert_eq!(vectors.len(), 2);
        // Check that both vectors are returned
        let mut found_v1 = false;
        let mut found_v2 = false;
        for v in &vectors {
            if v[0] == 1.0 {
                found_v1 = true;
            }
            if v[0] == 2.0 {
                found_v2 = true;
            }
        }
        assert!(found_v1 && found_v2);

        // Non-existent label
        assert!(index.get_vectors(999).is_none());
    }

    #[test]
    fn test_hnsw_multi_get_labels() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 10).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 20).unwrap();
        index.add_vector(&vec![3.0, 0.0, 0.0, 0.0], 30).unwrap();

        let labels = index.get_labels();
        assert_eq!(labels.len(), 3);
        assert!(labels.contains(&10));
        assert!(labels.contains(&20));
        assert!(labels.contains(&30));
    }

    #[test]
    fn test_hnsw_multi_compute_distance() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Add two vectors for label 1, at different distances from query
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![3.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Query should return minimum distance among all vectors for the label
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let dist = index.compute_distance(1, &query).unwrap();
        // Distance to [1.0, 0.0, 0.0, 0.0] is 1.0 (L2 squared)
        assert!((dist - 1.0).abs() < 0.001);

        // Non-existent label
        assert!(index.compute_distance(999, &query).is_none());
    }

    #[test]
    fn test_hnsw_multi_contains() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();

        assert!(index.contains(1));
        assert!(!index.contains(2));
    }

    #[test]
    fn test_hnsw_multi_range_query() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Add vectors at different distances
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 2).unwrap();
        index.add_vector(&vec![3.0, 0.0, 0.0, 0.0], 3).unwrap();
        index.add_vector(&vec![10.0, 0.0, 0.0, 0.0], 4).unwrap();

        let query = vec![0.0, 0.0, 0.0, 0.0];
        // L2 squared: dist to [1,0,0,0]=1, [2,0,0,0]=4, [3,0,0,0]=9, [10,0,0,0]=100
        let results = index.range_query(&query, 10.0, None).unwrap();

        assert_eq!(results.len(), 3); // labels 1, 2, 3 are within radius 10
        for r in &results.results {
            assert!(r.label != 4); // label 4 should not be included
        }
    }

    #[test]
    fn test_hnsw_multi_filtered_query() {
        use crate::query::QueryParams;

        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        for label in 1..=5u64 {
            index.add_vector(&vec![label as f32, 0.0, 0.0, 0.0], label).unwrap();
        }

        // Filter to only allow even labels
        let query_params = QueryParams::new().with_filter(|label| label % 2 == 0);
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 10, Some(&query_params)).unwrap();

        assert_eq!(results.len(), 2); // Only labels 2 and 4
        for r in &results.results {
            assert!(r.label % 2 == 0);
        }
    }

    #[test]
    fn test_hnsw_multi_batch_iterator() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        for i in 0..10u64 {
            index.add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        let query = vec![5.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(3) {
            all_results.extend(batch);
        }

        assert!(!all_results.is_empty());
    }

    #[test]
    fn test_hnsw_multi_dimension_mismatch() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Wrong dimension on add
        let result = index.add_vector(&vec![1.0, 2.0], 1);
        assert!(matches!(result, Err(IndexError::DimensionMismatch { expected: 4, got: 2 })));

        // Add a valid vector first
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Wrong dimension on query
        let result = index.top_k_query(&vec![1.0, 2.0], 1, None);
        assert!(matches!(result, Err(QueryError::DimensionMismatch { expected: 4, got: 2 })));
    }

    #[test]
    fn test_hnsw_multi_capacity_exceeded() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::with_capacity(params, 2);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 2).unwrap();

        // Third should fail
        let result = index.add_vector(&vec![3.0, 0.0, 0.0, 0.0], 3);
        assert!(matches!(result, Err(IndexError::CapacityExceeded { capacity: 2 })));
    }

    #[test]
    fn test_hnsw_multi_delete_not_found() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();

        let result = index.delete_vector(999);
        assert!(matches!(result, Err(IndexError::LabelNotFound(999))));
    }

    #[test]
    fn test_hnsw_multi_memory_usage() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        let initial_memory = index.memory_usage();

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 2).unwrap();

        let after_memory = index.memory_usage();
        assert!(after_memory > initial_memory);
    }

    #[test]
    fn test_hnsw_multi_info() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 1).unwrap();

        let info = index.info();
        assert_eq!(info.size, 2);
        assert_eq!(info.dimension, 4);
        assert_eq!(info.index_type, "HnswMulti");
        assert!(info.memory_bytes > 0);
    }

    #[test]
    fn test_hnsw_multi_clear() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 2);

        index.clear();

        assert_eq!(index.index_size(), 0);
        assert!(index.get_labels().is_empty());
        assert!(!index.contains(1));
        assert!(!index.contains(2));
    }

    #[test]
    fn test_hnsw_multi_add_vectors_batch() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        let vectors: Vec<(&[f32], LabelType)> = vec![
            (&[1.0, 0.0, 0.0, 0.0], 1),
            (&[2.0, 0.0, 0.0, 0.0], 1),
            (&[3.0, 0.0, 0.0, 0.0], 2),
        ];

        let added = index.add_vectors(&vectors).unwrap();
        assert_eq!(added, 3);
        assert_eq!(index.index_size(), 3);
        assert_eq!(index.label_count(1), 2);
        assert_eq!(index.label_count(2), 1);
    }

    #[test]
    fn test_hnsw_multi_with_capacity() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let index = HnswMulti::<f32>::with_capacity(params, 100);

        assert_eq!(index.index_capacity(), Some(100));
    }

    #[test]
    fn test_hnsw_multi_inner_product() {
        let params = HnswParams::new(4, Metric::InnerProduct).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // For InnerProduct, higher dot product = lower "distance"
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 2, None).unwrap();

        // Label 1 has perfect alignment with query
        assert_eq!(results.results[0].label, 1);
    }

    #[test]
    fn test_hnsw_multi_cosine() {
        let params = HnswParams::new(4, Metric::Cosine).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Cosine similarity is direction-based
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 3).unwrap(); // Same direction as label 1

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 3, None).unwrap();

        // Labels 1 and 3 have the same direction (cosine=1), should be top results
        assert!(results.results[0].label == 1 || results.results[0].label == 3);
        assert!(results.results[1].label == 1 || results.results[1].label == 3);
    }

    #[test]
    fn test_hnsw_multi_metric_getter() {
        let params = HnswParams::new(4, Metric::Cosine).with_m(4).with_ef_construction(20);
        let index = HnswMulti::<f32>::new(params);

        assert_eq!(index.metric(), Metric::Cosine);
    }

    #[test]
    fn test_hnsw_multi_ef_runtime() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20)
            .with_ef_runtime(50);
        let mut index = HnswMulti::<f32>::new(params);

        assert_eq!(index.ef_runtime(), 50);

        // Modify ef_runtime
        index.set_ef_runtime(100);
        assert_eq!(index.ef_runtime(), 100);
    }

    #[test]
    fn test_hnsw_multi_query_with_ef_runtime() {
        use crate::query::QueryParams;

        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20)
            .with_ef_runtime(10);
        let mut index = HnswMulti::<f32>::new(params);

        for i in 0..50u64 {
            index.add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        let query = vec![25.0, 0.0, 0.0, 0.0];

        // Query with default ef_runtime
        let results1 = index.top_k_query(&query, 5, None).unwrap();
        assert!(!results1.is_empty());

        // Query with higher ef_runtime
        let query_params = QueryParams::new().with_ef_runtime(100);
        let results2 = index.top_k_query(&query, 5, Some(&query_params)).unwrap();
        assert!(!results2.is_empty());
    }

    #[test]
    fn test_hnsw_multi_filtered_range_query() {
        use crate::query::QueryParams;

        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        for label in 1..=10u64 {
            index.add_vector(&vec![label as f32, 0.0, 0.0, 0.0], label).unwrap();
        }

        // Filter to only allow labels 1-5, with range 50 (covers labels 1-7 by distance)
        let query_params = QueryParams::new().with_filter(|label| label <= 5);
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = index.range_query(&query, 50.0, Some(&query_params)).unwrap();

        // Should have labels 1-5 (filtered) that are within range 50
        assert_eq!(results.len(), 5);
        for r in &results.results {
            assert!(r.label <= 5);
        }
    }

    #[test]
    fn test_hnsw_multi_empty_query() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let index = HnswMulti::<f32>::new(params);

        // Query on empty index
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 10, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_multi_fragmentation() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        for i in 1..=10u64 {
            index.add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        // Initially no fragmentation
        assert!((index.fragmentation() - 0.0).abs() < 0.01);

        // Delete half the vectors
        for i in 1..=5u64 {
            index.delete_vector(i).unwrap();
        }

        // Now there should be fragmentation
        assert!(index.fragmentation() > 0.3);
    }

    #[test]
    fn test_hnsw_multi_serialization_with_capacity() {
        use std::io::Cursor;

        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::with_capacity(params, 1000);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Serialize
        let mut buffer = Vec::new();
        index.save(&mut buffer).unwrap();

        // Deserialize
        let mut cursor = Cursor::new(buffer);
        let loaded = HnswMulti::<f32>::load(&mut cursor).unwrap();

        // Capacity should be preserved
        assert_eq!(loaded.index_capacity(), Some(1000));
        assert_eq!(loaded.index_size(), 2);
        assert_eq!(loaded.label_count(1), 2);
    }

    #[test]
    fn test_hnsw_multi_query_after_compact() {
        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Add many vectors
        for i in 1..=20u64 {
            index.add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        // Delete odd labels
        for i in (1..=20u64).step_by(2) {
            index.delete_vector(i).unwrap();
        }

        // Compact
        index.compact(true);

        // Query should still work
        let query = vec![4.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 5, None).unwrap();

        assert_eq!(results.len(), 5);
        // First result should be label 4 (exact match, and it's even so not deleted)
        assert_eq!(results.results[0].label, 4);

        // All results should be even labels
        for r in &results.results {
            assert!(r.label % 2 == 0);
        }
    }

    #[test]
    fn test_hnsw_multi_larger_scale() {
        let params = HnswParams::new(8, Metric::L2)
            .with_m(16)
            .with_ef_construction(100)
            .with_ef_runtime(50);
        let mut index = HnswMulti::<f32>::new(params);

        // Add 1000 vectors, 10 per label
        for label in 0..100u64 {
            for i in 0..10 {
                let v = vec![
                    label as f32 + i as f32 * 0.01,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ];
                index.add_vector(&v, label).unwrap();
            }
        }

        assert_eq!(index.index_size(), 1000);
        assert_eq!(index.label_count(50), 10);

        // Query should work
        let query = vec![50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 10, None).unwrap();

        assert!(!results.is_empty());
        // First result should be label 50 (closest)
        assert_eq!(results.results[0].label, 50);
    }

    #[test]
    fn test_hnsw_multi_heuristic_mode() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20)
            .with_heuristic(true);
        let mut index = HnswMulti::<f32>::new(params);

        for i in 0..20u64 {
            index.add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        let query = vec![10.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 5, None).unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(results.results[0].label, 10);
    }

    #[test]
    fn test_hnsw_multi_batch_iterator_with_params() {
        use crate::query::QueryParams;

        let params = HnswParams::new(4, Metric::L2).with_m(4).with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        for i in 1..=10u64 {
            index.add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        let query = vec![5.0, 0.0, 0.0, 0.0];
        // Note: filters cannot be cloned, so QueryParams with filters
        // won't preserve the filter when cloned for batch_iterator
        let query_params = QueryParams::new().with_ef_runtime(50);

        // Batch iterator should work with params
        let mut iter = index.batch_iterator(&query, Some(&query_params)).unwrap();
        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(100) {
            all_results.extend(batch);
        }

        // Should have all 10 results
        assert_eq!(all_results.len(), 10);
    }
}

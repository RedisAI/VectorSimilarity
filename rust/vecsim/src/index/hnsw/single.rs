//! Single-value HNSW index implementation.
//!
//! This index stores one vector per label. When adding a vector with
//! an existing label, the old vector is replaced.

use super::{HnswCore, HnswParams};
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply, QueryResult};
use crate::types::{DistanceType, IdType, LabelType, VectorElement};
use parking_lot::RwLock;
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
pub struct HnswSingle<T: VectorElement> {
    /// Core HNSW implementation.
    pub(crate) core: RwLock<HnswCore<T>>,
    /// Label to internal ID mapping.
    label_to_id: RwLock<HashMap<LabelType, IdType>>,
    /// Internal ID to label mapping.
    pub(crate) id_to_label: RwLock<HashMap<IdType, LabelType>>,
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
            core: RwLock::new(core),
            label_to_id: RwLock::new(HashMap::with_capacity(initial_capacity)),
            id_to_label: RwLock::new(HashMap::with_capacity(initial_capacity)),
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
        self.core.read().params.metric
    }

    /// Get the ef_runtime parameter.
    pub fn ef_runtime(&self) -> usize {
        self.core.read().params.ef_runtime
    }

    /// Set the ef_runtime parameter.
    pub fn set_ef_runtime(&self, ef: usize) {
        self.core.write().params.ef_runtime = ef;
    }

    /// Get the M parameter (max connections per element per layer).
    pub fn m(&self) -> usize {
        self.core.read().params.m
    }

    /// Get the M_max_0 parameter (max connections at layer 0).
    pub fn m_max_0(&self) -> usize {
        self.core.read().params.m_max_0
    }

    /// Get the ef_construction parameter.
    pub fn ef_construction(&self) -> usize {
        self.core.read().params.ef_construction
    }

    /// Check if heuristic neighbor selection is enabled.
    pub fn is_heuristic_enabled(&self) -> bool {
        self.core.read().params.enable_heuristic
    }

    /// Get the current entry point ID (top-level node).
    pub fn entry_point(&self) -> Option<IdType> {
        let ep = self.core.read().entry_point.load(std::sync::atomic::Ordering::Relaxed);
        if ep == crate::types::INVALID_ID {
            None
        } else {
            Some(ep)
        }
    }

    /// Get the current maximum level in the graph.
    pub fn max_level(&self) -> usize {
        self.core.read().max_level.load(std::sync::atomic::Ordering::Relaxed) as usize
    }

    /// Get the number of deleted (but not yet removed) elements.
    pub fn deleted_count(&self) -> usize {
        let core = self.core.read();
        core.graph
            .iter()
            .filter(|e| e.as_ref().map_or(false, |g| g.meta.deleted))
            .count()
    }

    /// Get detailed statistics about the index.
    pub fn stats(&self) -> HnswStats {
        let core = self.core.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        let mut level_counts = vec![0usize; core.max_level.load(std::sync::atomic::Ordering::Relaxed) as usize + 1];
        let mut total_connections = 0usize;
        let mut deleted_count = 0usize;

        for element in core.graph.iter().flatten() {
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
            max_level: core.max_level.load(std::sync::atomic::Ordering::Relaxed) as usize,
            level_counts,
            total_connections,
            avg_connections_per_element: avg_connections,
            memory_bytes: self.memory_usage(),
        }
    }

    /// Get the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let core = self.core.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        // Vector data storage
        let vector_storage = count * core.params.dim * std::mem::size_of::<T>();

        // Graph structure (rough estimate)
        let graph_overhead = core.graph.len()
            * std::mem::size_of::<Option<super::graph::ElementGraphData>>();

        // Label mappings
        let label_maps = self.label_to_id.read().capacity()
            * std::mem::size_of::<(LabelType, IdType)>()
            * 2;

        vector_storage + graph_overhead + label_maps
    }

    /// Get a copy of the vector stored for a given label.
    ///
    /// Returns `None` if the label doesn't exist in the index.
    pub fn get_vector(&self, label: LabelType) -> Option<Vec<T>> {
        let label_to_id = self.label_to_id.read();
        let id = *label_to_id.get(&label)?;
        let core = self.core.read();
        core.data.get(id).map(|v| v.to_vec())
    }

    /// Get all labels currently in the index.
    pub fn get_labels(&self) -> Vec<LabelType> {
        self.label_to_id.read().keys().copied().collect()
    }

    /// Compute the distance between a stored vector and a query vector.
    ///
    /// Returns `None` if the label doesn't exist.
    pub fn compute_distance(&self, label: LabelType, query: &[T]) -> Option<T::DistanceType> {
        let label_to_id = self.label_to_id.read();
        let id = *label_to_id.get(&label)?;
        let core = self.core.read();
        if let Some(stored) = core.data.get(id) {
            Some(core.dist_fn.compute(stored, query, core.params.dim))
        } else {
            None
        }
    }

    /// Clear all vectors from the index, resetting it to empty state.
    pub fn clear(&mut self) {
        use std::sync::atomic::Ordering;

        let mut core = self.core.write();
        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();

        core.data.clear();
        core.graph.clear();
        core.entry_point.store(crate::types::INVALID_ID, Ordering::Relaxed);
        core.max_level.store(0, Ordering::Relaxed);
        label_to_id.clear();
        id_to_label.clear();
        self.count.store(0, Ordering::Relaxed);
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
}

impl<T: VectorElement> VecSimIndex for HnswSingle<T> {
    type DataType = T;
    type DistType = T::DistanceType;

    fn add_vector(&mut self, vector: &[T], label: LabelType) -> Result<usize, IndexError> {
        let mut core = self.core.write();

        if vector.len() != core.params.dim {
            return Err(IndexError::DimensionMismatch {
                expected: core.params.dim,
                got: vector.len(),
            });
        }

        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();

        // Check if label already exists
        if let Some(&existing_id) = label_to_id.get(&label) {
            // Mark old vector as deleted
            core.mark_deleted(existing_id);
            id_to_label.remove(&existing_id);

            // Add new vector
            let new_id = core.add_vector(vector);
            core.insert(new_id, label);

            // Update mappings
            label_to_id.insert(label, new_id);
            id_to_label.insert(new_id, label);

            return Ok(0); // Replacement, not a new vector
        }

        // Check capacity
        if let Some(cap) = self.capacity {
            if self.count.load(std::sync::atomic::Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Add new vector
        let id = core.add_vector(vector);
        core.insert(id, label);

        // Update mappings
        label_to_id.insert(label, id);
        id_to_label.insert(id, label);

        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let mut core = self.core.write();
        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();

        if let Some(id) = label_to_id.remove(&label) {
            core.mark_deleted(id);
            id_to_label.remove(&id);
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
        let core = self.core.read();

        if query.len() != core.params.dim {
            return Err(QueryError::DimensionMismatch {
                expected: core.params.dim,
                got: query.len(),
            });
        }

        let ef = params
            .and_then(|p| p.ef_runtime)
            .unwrap_or(core.params.ef_runtime);

        // Build filter if needed
        let has_filter = params.map_or(false, |p| p.filter.is_some());
        let id_label_map: HashMap<IdType, LabelType> = if has_filter {
            self.id_to_label.read().clone()
        } else {
            HashMap::new()
        };

        let filter_fn: Option<Box<dyn Fn(IdType) -> bool>> = if let Some(p) = params {
            if let Some(ref f) = p.filter {
                let f = f.as_ref();
                Some(Box::new(move |id: IdType| {
                    id_label_map.get(&id).map_or(false, |&label| f(label))
                }))
            } else {
                None
            }
        } else {
            None
        };

        let results = core.search(query, k, ef, filter_fn.as_ref().map(|f| f.as_ref()));

        // Look up labels for results
        let id_to_label = self.id_to_label.read();
        let mut reply = QueryReply::with_capacity(results.len());
        for (id, dist) in results {
            if let Some(&label) = id_to_label.get(&id) {
                reply.push(QueryResult::new(label, dist));
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
        let core = self.core.read();

        if query.len() != core.params.dim {
            return Err(QueryError::DimensionMismatch {
                expected: core.params.dim,
                got: query.len(),
            });
        }

        let ef = params
            .and_then(|p| p.ef_runtime)
            .unwrap_or(core.params.ef_runtime)
            .max(1000);

        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        // Build filter if needed
        let has_filter = params.map_or(false, |p| p.filter.is_some());
        let id_label_map: HashMap<IdType, LabelType> = if has_filter {
            self.id_to_label.read().clone()
        } else {
            HashMap::new()
        };

        let filter_fn: Option<Box<dyn Fn(IdType) -> bool>> = if let Some(p) = params {
            if let Some(ref f) = p.filter {
                let f = f.as_ref();
                Some(Box::new(move |id: IdType| {
                    id_label_map.get(&id).map_or(false, |&label| f(label))
                }))
            } else {
                None
            }
        } else {
            None
        };

        let results = core.search(query, count, ef, filter_fn.as_ref().map(|f| f.as_ref()));

        // Look up labels and filter by radius
        let id_to_label = self.id_to_label.read();
        let mut reply = QueryReply::new();
        for (id, dist) in results {
            if dist.to_f64() <= radius.to_f64() {
                if let Some(&label) = id_to_label.get(&id) {
                    reply.push(QueryResult::new(label, dist));
                }
            }
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
        self.core.read().params.dim
    }

    fn batch_iterator<'a>(
        &'a self,
        query: &[T],
        params: Option<&QueryParams>,
    ) -> Result<Box<dyn BatchIterator<DistType = T::DistanceType> + 'a>, QueryError> {
        let core = self.core.read();
        if query.len() != core.params.dim {
            return Err(QueryError::DimensionMismatch {
                expected: core.params.dim,
                got: query.len(),
            });
        }
        drop(core);

        Ok(Box::new(
            super::batch_iterator::HnswSingleBatchIterator::new(self, query.to_vec(), params.cloned()),
        ))
    }

    fn info(&self) -> IndexInfo {
        let core = self.core.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        IndexInfo {
            size: count,
            capacity: self.capacity,
            dimension: core.params.dim,
            index_type: "HnswSingle",
            memory_bytes: count * core.params.dim * std::mem::size_of::<T>()
                + core.graph.len() * std::mem::size_of::<Option<super::graph::ElementGraphData>>()
                + self.label_to_id.read().capacity() * std::mem::size_of::<(LabelType, IdType)>(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.label_to_id.read().contains_key(&label)
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
impl HnswSingle<f32> {
    /// Save the index to a writer.
    pub fn save<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> crate::serialization::SerializationResult<()> {
        use crate::serialization::*;
        use std::sync::atomic::Ordering;

        let core = self.core.read();
        let label_to_id = self.label_to_id.read();
        let count = self.count.load(Ordering::Relaxed);

        // Write header
        let header = IndexHeader::new(
            IndexTypeId::HnswSingle,
            DataTypeId::F32,
            core.params.metric,
            core.params.dim,
            count,
        );
        header.write(writer)?;

        // Write HNSW-specific params
        write_usize(writer, core.params.m)?;
        write_usize(writer, core.params.m_max_0)?;
        write_usize(writer, core.params.ef_construction)?;
        write_usize(writer, core.params.ef_runtime)?;
        write_u8(writer, if core.params.enable_heuristic { 1 } else { 0 })?;

        // Write graph metadata
        let entry_point = core.entry_point.load(Ordering::Relaxed);
        let max_level = core.max_level.load(Ordering::Relaxed);
        write_u32(writer, entry_point)?;
        write_u32(writer, max_level)?;

        // Write capacity
        write_u8(writer, if self.capacity.is_some() { 1 } else { 0 })?;
        if let Some(cap) = self.capacity {
            write_usize(writer, cap)?;
        }

        // Write label_to_id mapping
        write_usize(writer, label_to_id.len())?;
        for (&label, &id) in label_to_id.iter() {
            write_u64(writer, label)?;
            write_u32(writer, id)?;
        }

        // Write graph structure
        write_usize(writer, core.graph.len())?;
        for (id, element) in core.graph.iter().enumerate() {
            let id = id as u32;
            if let Some(ref graph_data) = element {
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
                if let Some(vector) = core.data.get(id) {
                    for &v in vector {
                        write_f32(writer, v)?;
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

        if header.data_type != DataTypeId::F32 {
            return Err(SerializationError::InvalidData(
                "Expected f32 data type".to_string(),
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
        };
        let mut index = Self::new(params);

        // Read capacity
        let has_capacity = read_u8(reader)? != 0;
        if has_capacity {
            index.capacity = Some(read_usize(reader)?);
        }

        // Read label_to_id mapping
        let label_to_id_len = read_usize(reader)?;
        let mut label_to_id = HashMap::with_capacity(label_to_id_len);
        for _ in 0..label_to_id_len {
            let label = read_u64(reader)?;
            let id = read_u32(reader)?;
            label_to_id.insert(label, id);
        }

        // Build id_to_label from label_to_id
        let mut id_to_label: HashMap<IdType, LabelType> = HashMap::with_capacity(label_to_id_len);
        for (&label, &id) in &label_to_id {
            id_to_label.insert(id, label);
        }

        // Read graph structure
        let graph_len = read_usize(reader)?;
        let dim = header.dimension;

        {
            let mut core = index.core.write();

            // Set entry point and max level
            core.entry_point.store(entry_point, Ordering::Relaxed);
            core.max_level.store(max_level, Ordering::Relaxed);

            // Pre-allocate graph
            core.graph.resize_with(graph_len, || None);

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
                let mut vector = vec![0.0f32; dim];
                for v in &mut vector {
                    *v = read_f32(reader)?;
                }

                // Add vector to data storage
                core.data.add(&vector);

                // Store graph data
                core.graph[id] = Some(graph_data);
            }

            // Resize visited pool
            if graph_len > 0 {
                core.visited_pool.resize(graph_len);
            }
        }

        // Set the internal state
        *index.label_to_id.write() = label_to_id;
        *index.id_to_label.write() = id_to_label;
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
}

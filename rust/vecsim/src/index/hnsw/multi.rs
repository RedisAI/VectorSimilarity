//! Multi-value HNSW index implementation.
//!
//! This index allows multiple vectors per label.

use super::{HnswCore, HnswParams};
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply, QueryResult};
use crate::types::{DistanceType, IdType, LabelType, VectorElement};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};

/// Multi-value HNSW index.
///
/// Each label can have multiple associated vectors.
pub struct HnswMulti<T: VectorElement> {
    /// Core HNSW implementation.
    pub(crate) core: RwLock<HnswCore<T>>,
    /// Label to set of internal IDs mapping.
    label_to_ids: RwLock<HashMap<LabelType, HashSet<IdType>>>,
    /// Internal ID to label mapping.
    pub(crate) id_to_label: RwLock<HashMap<IdType, LabelType>>,
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
            core: RwLock::new(core),
            label_to_ids: RwLock::new(HashMap::with_capacity(initial_capacity / 2)),
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

    /// Get copies of all vectors stored for a given label.
    ///
    /// Returns `None` if the label doesn't exist in the index.
    pub fn get_vectors(&self, label: LabelType) -> Option<Vec<Vec<T>>> {
        let label_to_ids = self.label_to_ids.read();
        let ids = label_to_ids.get(&label)?;
        let core = self.core.read();
        let vectors: Vec<Vec<T>> = ids
            .iter()
            .filter_map(|&id| core.data.get(id).map(|v| v.to_vec()))
            .collect();
        if vectors.is_empty() {
            None
        } else {
            Some(vectors)
        }
    }

    /// Get all labels currently in the index.
    pub fn get_labels(&self) -> Vec<LabelType> {
        self.label_to_ids.read().keys().copied().collect()
    }

    /// Compute the minimum distance between any stored vector for a label and a query vector.
    ///
    /// Returns `None` if the label doesn't exist.
    pub fn compute_distance(&self, label: LabelType, query: &[T]) -> Option<T::DistanceType> {
        let label_to_ids = self.label_to_ids.read();
        let ids = label_to_ids.get(&label)?;
        let core = self.core.read();
        ids.iter()
            .filter_map(|&id| {
                core.data.get(id).map(|stored| {
                    core.dist_fn.compute(stored, query, core.params.dim)
                })
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
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
        let label_maps = self.label_to_ids.read().capacity()
            * std::mem::size_of::<(LabelType, HashSet<IdType>)>()
            + self.id_to_label.read().capacity() * std::mem::size_of::<(IdType, LabelType)>();

        vector_storage + graph_overhead + label_maps
    }

    /// Clear all vectors from the index, resetting it to empty state.
    pub fn clear(&mut self) {
        use std::sync::atomic::Ordering;

        let mut core = self.core.write();
        let mut label_to_ids = self.label_to_ids.write();
        let mut id_to_label = self.id_to_label.write();

        core.data.clear();
        core.graph.clear();
        core.entry_point.store(crate::types::INVALID_ID, Ordering::Relaxed);
        core.max_level.store(0, Ordering::Relaxed);
        label_to_ids.clear();
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

impl<T: VectorElement> VecSimIndex for HnswMulti<T> {
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

        // Check capacity
        if let Some(cap) = self.capacity {
            if self.count.load(std::sync::atomic::Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Add the vector
        let id = core
            .add_vector(vector)
            .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;
        core.insert(id, label);

        let mut label_to_ids = self.label_to_ids.write();
        let mut id_to_label = self.id_to_label.write();

        // Update mappings
        label_to_ids.entry(label).or_default().insert(id);
        id_to_label.insert(id, label);

        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let mut core = self.core.write();
        let mut label_to_ids = self.label_to_ids.write();
        let mut id_to_label = self.id_to_label.write();

        if let Some(ids) = label_to_ids.remove(&label) {
            let count = ids.len();

            for id in ids {
                core.mark_deleted(id);
                id_to_label.remove(&id);
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
        let has_filter = params.is_some_and(|p| p.filter.is_some());
        let id_label_map: HashMap<IdType, LabelType> = if has_filter {
            self.id_to_label.read().clone()
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
        let has_filter = params.is_some_and(|p| p.filter.is_some());
        let id_label_map: HashMap<IdType, LabelType> = if has_filter {
            self.id_to_label.read().clone()
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
            super::batch_iterator::HnswMultiBatchIterator::new(self, query.to_vec(), params.cloned()),
        ))
    }

    fn info(&self) -> IndexInfo {
        let core = self.core.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        IndexInfo {
            size: count,
            capacity: self.capacity,
            dimension: core.params.dim,
            index_type: "HnswMulti",
            memory_bytes: count * core.params.dim * std::mem::size_of::<T>()
                + core.graph.len() * std::mem::size_of::<Option<super::graph::ElementGraphData>>()
                + self.label_to_ids.read().capacity()
                    * std::mem::size_of::<(LabelType, HashSet<IdType>)>(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.label_to_ids.read().contains_key(&label)
    }

    fn label_count(&self, label: LabelType) -> usize {
        self.label_to_ids
            .read()
            .get(&label)
            .map_or(0, |ids| ids.len())
    }
}

unsafe impl<T: VectorElement> Send for HnswMulti<T> {}
unsafe impl<T: VectorElement> Sync for HnswMulti<T> {}

// Serialization support
impl HnswMulti<f32> {
    /// Save the index to a writer.
    pub fn save<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> crate::serialization::SerializationResult<()> {
        use crate::serialization::*;
        use std::sync::atomic::Ordering;

        let core = self.core.read();
        let label_to_ids = self.label_to_ids.read();
        let count = self.count.load(Ordering::Relaxed);

        // Write header
        let header = IndexHeader::new(
            IndexTypeId::HnswMulti,
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

        // Write label_to_ids mapping (label -> set of IDs)
        write_usize(writer, label_to_ids.len())?;
        for (&label, ids) in label_to_ids.iter() {
            write_u64(writer, label)?;
            write_usize(writer, ids.len())?;
            for &id in ids {
                write_u32(writer, id)?;
            }
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

        if header.index_type != IndexTypeId::HnswMulti {
            return Err(SerializationError::IndexTypeMismatch {
                expected: "HnswMulti".to_string(),
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
            seed: None, // Seed not preserved in serialization
        };
        let mut index = Self::new(params);

        // Read capacity
        let has_capacity = read_u8(reader)? != 0;
        if has_capacity {
            index.capacity = Some(read_usize(reader)?);
        }

        // Read label_to_ids mapping
        let label_to_ids_len = read_usize(reader)?;
        let mut label_to_ids: HashMap<LabelType, HashSet<IdType>> =
            HashMap::with_capacity(label_to_ids_len);
        for _ in 0..label_to_ids_len {
            let label = read_u64(reader)?;
            let num_ids = read_usize(reader)?;
            let mut ids = HashSet::with_capacity(num_ids);
            for _ in 0..num_ids {
                ids.insert(read_u32(reader)?);
            }
            label_to_ids.insert(label, ids);
        }

        // Build id_to_label from label_to_ids
        let mut id_to_label: HashMap<IdType, LabelType> = HashMap::new();
        for (&label, ids) in &label_to_ids {
            for &id in ids {
                id_to_label.insert(id, label);
            }
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
                core.data.add(&vector).ok_or_else(|| {
                    SerializationError::DataCorruption("Failed to add vector during deserialization".to_string())
                })?;

                // Store graph data
                core.graph[id] = Some(graph_data);
            }

            // Resize visited pool
            if graph_len > 0 {
                core.visited_pool.resize(graph_len);
            }
        }

        // Set the internal state
        *index.label_to_ids.write() = label_to_ids;
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
}

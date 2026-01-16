//! Single-value SVS (Vamana) index implementation.
//!
//! This index stores one vector per label. When adding a vector with
//! an existing label, the old vector is replaced.

use super::{SvsCore, SvsParams};
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply, QueryResult};
use crate::types::{DistanceType, IdType, LabelType, VectorElement, INVALID_ID};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Statistics for SVS index.
#[derive(Debug, Clone)]
pub struct SvsStats {
    /// Number of vectors in the index.
    pub vector_count: usize,
    /// Number of labels in the index.
    pub label_count: usize,
    /// Dimension of vectors.
    pub dimension: usize,
    /// Distance metric.
    pub metric: crate::distance::Metric,
    /// Maximum graph degree.
    pub graph_max_degree: usize,
    /// Alpha parameter.
    pub alpha: f32,
    /// Average out-degree.
    pub avg_degree: f64,
    /// Maximum out-degree.
    pub max_degree: usize,
    /// Minimum out-degree.
    pub min_degree: usize,
    /// Memory usage (approximate).
    pub memory_bytes: usize,
}

/// Single-value SVS (Vamana) index.
///
/// Each label has exactly one associated vector.
pub struct SvsSingle<T: VectorElement> {
    /// Core SVS implementation.
    core: RwLock<SvsCore<T>>,
    /// Label to internal ID mapping.
    label_to_id: RwLock<HashMap<LabelType, IdType>>,
    /// Internal ID to label mapping.
    id_to_label: RwLock<HashMap<IdType, LabelType>>,
    /// Number of vectors.
    count: AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
    /// Construction completed flag (for two-pass).
    construction_done: RwLock<bool>,
}

impl<T: VectorElement> SvsSingle<T> {
    /// Create a new SVS index with the given parameters.
    pub fn new(params: SvsParams) -> Self {
        let initial_capacity = params.initial_capacity;
        Self {
            core: RwLock::new(SvsCore::new(params)),
            label_to_id: RwLock::new(HashMap::with_capacity(initial_capacity)),
            id_to_label: RwLock::new(HashMap::with_capacity(initial_capacity)),
            count: AtomicUsize::new(0),
            capacity: None,
            construction_done: RwLock::new(false),
        }
    }

    /// Create with a maximum capacity.
    pub fn with_capacity(params: SvsParams, max_capacity: usize) -> Self {
        let mut index = Self::new(params);
        index.capacity = Some(max_capacity);
        index
    }

    /// Get the distance metric.
    pub fn metric(&self) -> crate::distance::Metric {
        self.core.read().params.metric
    }

    /// Get index statistics.
    pub fn stats(&self) -> SvsStats {
        let core = self.core.read();
        let count = self.count.load(Ordering::Relaxed);

        SvsStats {
            vector_count: count,
            label_count: self.label_to_id.read().len(),
            dimension: core.params.dim,
            metric: core.params.metric,
            graph_max_degree: core.params.graph_max_degree,
            alpha: core.params.alpha,
            avg_degree: core.graph.average_degree(),
            max_degree: core.graph.max_degree(),
            min_degree: core.graph.min_degree(),
            memory_bytes: self.memory_usage(),
        }
    }

    /// Estimate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let core = self.core.read();
        let count = self.count.load(Ordering::Relaxed);

        // Vector data
        let vector_size = count * core.params.dim * std::mem::size_of::<T>();

        // Graph (rough estimate)
        let graph_size = count * core.params.graph_max_degree * std::mem::size_of::<IdType>();

        // Label mapping
        let label_size = self.label_to_id.read().capacity()
            * (std::mem::size_of::<LabelType>() + std::mem::size_of::<IdType>())
            + self.id_to_label.read().capacity()
            * (std::mem::size_of::<IdType>() + std::mem::size_of::<LabelType>());

        vector_size + graph_size + label_size
    }

    /// Build the index after all vectors have been added.
    ///
    /// For two-pass construction, this performs the second pass.
    /// Call this after adding all vectors for best recall.
    pub fn build(&self) {
        let mut done = self.construction_done.write();
        if *done {
            return;
        }

        let mut core = self.core.write();
        if core.params.two_pass_construction && core.data.len() > 1 {
            core.rebuild_graph();
        }

        *done = true;
    }

    /// Get the medoid (entry point) ID.
    pub fn medoid(&self) -> Option<IdType> {
        let ep = self.core.read().medoid.load(Ordering::Relaxed);
        if ep == INVALID_ID {
            None
        } else {
            Some(ep)
        }
    }

    /// Get the search window size parameter.
    pub fn search_l(&self) -> usize {
        self.core.read().params.search_window_size
    }

    /// Set the search window size parameter.
    pub fn set_search_l(&self, l: usize) {
        self.core.write().params.search_window_size = l;
    }

    /// Get the graph max degree parameter.
    pub fn graph_degree(&self) -> usize {
        self.core.read().params.graph_max_degree
    }

    /// Get the alpha parameter.
    pub fn alpha(&self) -> f32 {
        self.core.read().params.alpha
    }

    /// Get the fragmentation ratio (0.0 = none, 1.0 = all deleted).
    pub fn fragmentation(&self) -> f64 {
        self.core.read().data.fragmentation()
    }

    /// Compact the index to reclaim space from deleted vectors.
    ///
    /// Note: SVS doesn't support true compaction without rebuilding.
    /// This method returns 0 as a placeholder.
    pub fn compact(&mut self, _shrink: bool) -> usize {
        // SVS requires rebuild for true compaction
        // For now, we don't support in-place compaction
        0
    }

    /// Get a copy of the vector stored for a given label.
    pub fn get_vector(&self, label: LabelType) -> Option<Vec<T>> {
        let id = *self.label_to_id.read().get(&label)?;
        let core = self.core.read();
        core.data.get(id).map(|v| v.to_vec())
    }

    /// Clear all vectors from the index.
    pub fn clear(&mut self) {
        let mut core = self.core.write();
        let params = core.params.clone();
        *core = SvsCore::new(params);

        self.label_to_id.write().clear();
        self.id_to_label.write().clear();
        self.count.store(0, Ordering::Relaxed);
        *self.construction_done.write() = false;
    }
}

impl<T: VectorElement> VecSimIndex for SvsSingle<T> {
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
            let new_id = core
                .add_vector(vector)
                .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;
            core.insert(new_id, label);

            // Update mappings
            label_to_id.insert(label, new_id);
            id_to_label.insert(new_id, label);

            // Reset construction flag
            *self.construction_done.write() = false;

            return Ok(0); // Replacement, not a new vector
        }

        // Check capacity
        if let Some(cap) = self.capacity {
            if self.count.load(Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Add new vector
        let id = core
            .add_vector(vector)
            .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;
        core.insert(id, label);

        // Update mappings
        label_to_id.insert(label, id);
        id_to_label.insert(id, label);

        self.count.fetch_add(1, Ordering::Relaxed);
        *self.construction_done.write() = false;

        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let mut core = self.core.write();
        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();

        if let Some(id) = label_to_id.remove(&label) {
            core.mark_deleted(id);
            id_to_label.remove(&id);
            self.count.fetch_sub(1, Ordering::Relaxed);
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

        if k == 0 {
            return Ok(QueryReply::new());
        }

        let search_l = params
            .and_then(|p| p.ef_runtime)
            .unwrap_or(core.params.search_window_size);

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

        let results = core.search(query, k, search_l, filter_fn.as_ref().map(|f| f.as_ref()));

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

        let count = self.count.load(Ordering::Relaxed);
        let search_l = params
            .and_then(|p| p.ef_runtime)
            .unwrap_or(core.params.search_window_size)
            .max(count.min(1000));

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

        let results = core.search(query, count, search_l, filter_fn.as_ref().map(|f| f.as_ref()));

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
        self.count.load(Ordering::Relaxed)
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

        let search_l = params
            .and_then(|p| p.ef_runtime)
            .unwrap_or(core.params.search_window_size);

        let count = self.count.load(Ordering::Relaxed);

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

        let raw_results = core.search(
            query,
            count,
            search_l.max(count),
            filter_fn.as_ref().map(|f| f.as_ref()),
        );

        let id_to_label = self.id_to_label.read();
        let results: Vec<_> = raw_results
            .into_iter()
            .filter_map(|(id, dist)| {
                id_to_label.get(&id).map(|&label| (id, label, dist))
            })
            .collect();

        Ok(Box::new(SvsSingleBatchIterator::<T>::new(results)))
    }

    fn info(&self) -> IndexInfo {
        let core = self.core.read();
        let count = self.count.load(Ordering::Relaxed);

        IndexInfo {
            size: count,
            capacity: self.capacity,
            dimension: core.params.dim,
            index_type: "SvsSingle",
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

// Allow read-only concurrent access for queries
unsafe impl<T: VectorElement> Send for SvsSingle<T> {}
unsafe impl<T: VectorElement> Sync for SvsSingle<T> {}

/// Batch iterator for SvsSingle.
pub struct SvsSingleBatchIterator<T: VectorElement> {
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    current_idx: usize,
}

impl<T: VectorElement> SvsSingleBatchIterator<T> {
    /// Create a new batch iterator with pre-computed results.
    pub fn new(results: Vec<(IdType, LabelType, T::DistanceType)>) -> Self {
        Self {
            results,
            current_idx: 0,
        }
    }
}

impl<T: VectorElement> BatchIterator for SvsSingleBatchIterator<T> {
    type DistType = T::DistanceType;

    fn has_next(&self) -> bool {
        self.current_idx < self.results.len()
    }

    fn next_batch(&mut self, batch_size: usize) -> Option<Vec<(IdType, LabelType, T::DistanceType)>> {
        if self.current_idx >= self.results.len() {
            return None;
        }

        let end = (self.current_idx + batch_size).min(self.results.len());
        let batch = self.results[self.current_idx..end].to_vec();
        self.current_idx = end;

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    fn reset(&mut self) {
        self.current_idx = 0;
    }
}

// Serialization support for SvsSingle<f32>
impl SvsSingle<f32> {
    /// Save the index to a writer.
    pub fn save<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> crate::serialization::SerializationResult<()> {
        use crate::serialization::*;
        use std::sync::atomic::Ordering;

        let core = self.core.read();
        let label_to_id = self.label_to_id.read();
        let id_to_label = self.id_to_label.read();
        let count = self.count.load(Ordering::Relaxed);

        // Write header
        let header = IndexHeader::new(
            IndexTypeId::SvsSingle,
            DataTypeId::F32,
            core.params.metric,
            core.params.dim,
            count,
        );
        header.write(writer)?;

        // Write SVS-specific parameters
        write_usize(writer, core.params.graph_max_degree)?;
        write_f32(writer, core.params.alpha)?;
        write_usize(writer, core.params.construction_window_size)?;
        write_usize(writer, core.params.search_window_size)?;
        write_u8(writer, if core.params.two_pass_construction { 1 } else { 0 })?;

        // Write capacity
        write_u8(writer, if self.capacity.is_some() { 1 } else { 0 })?;
        if let Some(cap) = self.capacity {
            write_usize(writer, cap)?;
        }

        // Write construction_done flag
        write_u8(writer, if *self.construction_done.read() { 1 } else { 0 })?;

        // Write medoid
        write_u32(writer, core.medoid.load(Ordering::Relaxed))?;

        // Write label_to_id mapping
        write_usize(writer, label_to_id.len())?;
        for (&label, &id) in label_to_id.iter() {
            write_u64(writer, label)?;
            write_u32(writer, id)?;
        }

        // Write id_to_label mapping
        write_usize(writer, id_to_label.len())?;
        for (&id, &label) in id_to_label.iter() {
            write_u32(writer, id)?;
            write_u64(writer, label)?;
        }

        // Write vectors - collect valid IDs first
        let valid_ids: Vec<IdType> = core.data.iter_ids().collect();
        write_usize(writer, valid_ids.len())?;
        for id in &valid_ids {
            write_u32(writer, *id)?;
            if let Some(vector) = core.data.get(*id) {
                for &v in vector {
                    write_f32(writer, v)?;
                }
            }
        }

        // Write graph structure
        // For each valid ID, write its neighbors
        for id in &valid_ids {
            let neighbors = core.graph.get_neighbors(*id);
            let label = core.graph.get_label(*id);
            let deleted = core.graph.is_deleted(*id);

            write_u64(writer, label)?;
            write_u8(writer, if deleted { 1 } else { 0 })?;
            write_usize(writer, neighbors.len())?;
            for &n in &neighbors {
                write_u32(writer, n)?;
            }
        }

        Ok(())
    }

    /// Load the index from a reader.
    pub fn load<R: std::io::Read>(reader: &mut R) -> crate::serialization::SerializationResult<Self> {
        use crate::serialization::*;
        use std::sync::atomic::Ordering;

        // Read and validate header
        let header = IndexHeader::read(reader)?;

        if header.index_type != IndexTypeId::SvsSingle {
            return Err(SerializationError::IndexTypeMismatch {
                expected: "SvsSingle".to_string(),
                got: header.index_type.as_str().to_string(),
            });
        }

        if header.data_type != DataTypeId::F32 {
            return Err(SerializationError::InvalidData(
                "Expected f32 data type".to_string(),
            ));
        }

        // Read SVS-specific parameters
        let graph_max_degree = read_usize(reader)?;
        let alpha = read_f32(reader)?;
        let construction_window_size = read_usize(reader)?;
        let search_window_size = read_usize(reader)?;
        let two_pass_construction = read_u8(reader)? != 0;

        // Create params
        let params = SvsParams {
            dim: header.dimension,
            metric: header.metric,
            graph_max_degree,
            alpha,
            construction_window_size,
            search_window_size,
            initial_capacity: header.count.max(1024),
            two_pass_construction,
        };

        // Create the index
        let mut index = Self::new(params);

        // Read capacity
        let has_capacity = read_u8(reader)? != 0;
        if has_capacity {
            index.capacity = Some(read_usize(reader)?);
        }

        // Read construction_done flag
        let construction_done = read_u8(reader)? != 0;
        *index.construction_done.write() = construction_done;

        // Read medoid
        let medoid = read_u32(reader)?;

        // Read label_to_id mapping
        let label_to_id_len = read_usize(reader)?;
        let mut label_to_id = HashMap::with_capacity(label_to_id_len);
        for _ in 0..label_to_id_len {
            let label = read_u64(reader)?;
            let id = read_u32(reader)?;
            label_to_id.insert(label, id);
        }

        // Read id_to_label mapping
        let id_to_label_len = read_usize(reader)?;
        let mut id_to_label = HashMap::with_capacity(id_to_label_len);
        for _ in 0..id_to_label_len {
            let id = read_u32(reader)?;
            let label = read_u64(reader)?;
            id_to_label.insert(id, label);
        }

        // Read vectors
        let num_vectors = read_usize(reader)?;
        let dim = header.dimension;
        let mut vector_ids = Vec::with_capacity(num_vectors);

        {
            let mut core = index.core.write();
            core.data.reserve(num_vectors);

            for _ in 0..num_vectors {
                let id = read_u32(reader)?;
                let mut vector = vec![0.0f32; dim];
                for v in &mut vector {
                    *v = read_f32(reader)?;
                }

                // Add vector at specific position
                let added_id = core.data.add(&vector).ok_or_else(|| {
                    SerializationError::DataCorruption(
                        "Failed to add vector during deserialization".to_string(),
                    )
                })?;

                // Track the ID for graph restoration
                vector_ids.push((id, added_id));
            }

            // Read and restore graph structure
            core.graph.ensure_capacity(num_vectors);
            for (original_id, _) in &vector_ids {
                let label = read_u64(reader)?;
                let deleted = read_u8(reader)? != 0;
                let num_neighbors = read_usize(reader)?;

                let mut neighbors = Vec::with_capacity(num_neighbors);
                for _ in 0..num_neighbors {
                    neighbors.push(read_u32(reader)?);
                }

                core.graph.set_label(*original_id, label);
                if deleted {
                    core.graph.mark_deleted(*original_id);
                }
                core.graph.set_neighbors(*original_id, &neighbors);
            }

            // Restore medoid
            core.medoid.store(medoid, Ordering::Release);

            // Resize visited pool
            if num_vectors > 0 {
                core.visited_pool.resize(num_vectors + 1024);
            }
        }

        // Restore label mappings
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
    fn test_svs_single_serialization() {
        use std::io::Cursor;

        let params = SvsParams::new(4, Metric::L2)
            .with_graph_degree(8)
            .with_alpha(1.2);
        let mut index = SvsSingle::<f32>::new(params);

        // Add vectors
        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
        index.add_vector(&[0.0, 0.0, 1.0, 0.0], 3).unwrap();
        index.add_vector(&[0.0, 0.0, 0.0, 1.0], 4).unwrap();

        // Build to ensure graph is populated
        index.build();

        // Serialize
        let mut buffer = Vec::new();
        index.save(&mut buffer).unwrap();

        // Deserialize
        let mut cursor = Cursor::new(buffer);
        let loaded = SvsSingle::<f32>::load(&mut cursor).unwrap();

        // Verify
        assert_eq!(loaded.index_size(), 4);
        assert_eq!(loaded.dimension(), 4);

        // Query should work the same
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = loaded.top_k_query(&query, 4, None).unwrap();
        assert_eq!(results.results[0].label, 1);
    }

    #[test]
    fn test_svs_single_basic() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsSingle::<f32>::new(params);

        // Add vectors
        assert_eq!(index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap(), 1);
        assert_eq!(index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap(), 1);
        assert_eq!(index.add_vector(&[0.0, 0.0, 1.0, 0.0], 3).unwrap(), 1);
        assert_eq!(index.add_vector(&[0.0, 0.0, 0.0, 1.0], 4).unwrap(), 1);

        assert_eq!(index.index_size(), 4);

        // Query
        let query = [1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 2, None).unwrap();

        assert_eq!(results.results.len(), 2);
        assert_eq!(results.results[0].label, 1);
    }

    #[test]
    fn test_svs_single_update() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsSingle::<f32>::new(params);

        // Add initial vector
        assert_eq!(index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap(), 1);
        assert_eq!(index.index_size(), 1);

        // Update same label (should return 0 for replacement)
        assert_eq!(index.add_vector(&[0.0, 1.0, 0.0, 0.0], 1).unwrap(), 0);
        assert_eq!(index.index_size(), 1);
    }

    #[test]
    fn test_svs_single_delete() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsSingle::<f32>::new(params);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 2);

        assert_eq!(index.delete_vector(1).unwrap(), 1);
        assert_eq!(index.index_size(), 1);
        assert!(!index.contains(1));
        assert!(index.contains(2));
    }

    #[test]
    fn test_svs_single_build() {
        let params = SvsParams::new(4, Metric::L2).with_two_pass(true);
        let mut index = SvsSingle::<f32>::new(params);

        // Add vectors
        for i in 0..100 {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            index.add_vector(&v, i as u64).unwrap();
        }

        // Build (second pass)
        index.build();

        // Query should still work
        let query = [1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 10, None).unwrap();

        assert!(!results.results.is_empty());
    }

    #[test]
    fn test_svs_single_stats() {
        let params = SvsParams::new(4, Metric::L2)
            .with_graph_degree(16)
            .with_alpha(1.2);
        let mut index = SvsSingle::<f32>::new(params);

        for i in 0..50 {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            index.add_vector(&v, i as u64).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.vector_count, 50);
        assert_eq!(stats.label_count, 50);
        assert_eq!(stats.dimension, 4);
        assert_eq!(stats.graph_max_degree, 16);
        assert!((stats.alpha - 1.2).abs() < 0.01);
    }

    #[test]
    fn test_svs_single_capacity() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsSingle::<f32>::with_capacity(params, 2);

        assert_eq!(index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap(), 1);
        assert_eq!(index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap(), 1);

        // Should fail due to capacity
        let result = index.add_vector(&[0.0, 0.0, 1.0, 0.0], 3);
        assert!(matches!(result, Err(IndexError::CapacityExceeded { capacity: 2 })));
    }

    #[test]
    fn test_svs_single_batch_iterator_with_filter() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsSingle::<f32>::new(params);

        // Add vectors with labels 1-10
        for i in 1..=10u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, i).unwrap();
        }

        assert_eq!(index.index_size(), 10);

        // Create a filter that only allows even labels
        let query_params = QueryParams::new().with_filter(|label| label % 2 == 0);

        let query = vec![5.0, 0.0, 0.0, 0.0];

        // Test batch iterator with filter
        let mut iter = index.batch_iterator(&query, Some(&query_params)).unwrap();
        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(100) {
            all_results.extend(batch);
        }

        // Should only have even labels (2, 4, 6, 8, 10)
        assert_eq!(all_results.len(), 5);
        for (_, label, _) in &all_results {
            assert_eq!(label % 2, 0, "Expected only even labels, got {}", label);
        }

        // Verify specific labels are present
        let labels: Vec<_> = all_results.iter().map(|(_, l, _)| *l).collect();
        assert!(labels.contains(&2));
        assert!(labels.contains(&4));
        assert!(labels.contains(&6));
        assert!(labels.contains(&8));
        assert!(labels.contains(&10));
    }

    #[test]
    fn test_svs_single_batch_iterator_no_filter() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsSingle::<f32>::new(params);

        // Add vectors
        for i in 1..=5u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, i).unwrap();
        }

        let query = vec![3.0, 0.0, 0.0, 0.0];

        // Test batch iterator without filter
        let mut iter = index.batch_iterator(&query, None).unwrap();
        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(100) {
            all_results.extend(batch);
        }

        // Should have all 5 vectors
        assert_eq!(all_results.len(), 5);
    }
}

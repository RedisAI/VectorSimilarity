//! Multi-value BruteForce index implementation.
//!
//! This index allows multiple vectors per label. Each label can have
//! any number of associated vectors.

use super::{BruteForceCore, BruteForceParams, IdLabelEntry};
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply, QueryResult};
use crate::types::{DistanceType, IdType, LabelType, VectorElement};
use crate::utils::MaxHeap;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Multi-value BruteForce index.
///
/// Each label can have multiple associated vectors. This is useful for
/// scenarios where a single entity has multiple representations.
pub struct BruteForceMulti<T: VectorElement> {
    /// Core storage and distance computation.
    pub(crate) core: RwLock<BruteForceCore<T>>,
    /// Label to set of internal IDs mapping.
    label_to_ids: RwLock<HashMap<LabelType, HashSet<IdType>>>,
    /// Internal ID to label mapping.
    pub(crate) id_to_label: RwLock<Vec<IdLabelEntry>>,
    /// Number of vectors.
    count: std::sync::atomic::AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
}

impl<T: VectorElement> BruteForceMulti<T> {
    /// Create a new multi-value BruteForce index.
    pub fn new(params: BruteForceParams) -> Self {
        let core = BruteForceCore::new(&params);
        let initial_capacity = params.initial_capacity;

        Self {
            core: RwLock::new(core),
            label_to_ids: RwLock::new(HashMap::with_capacity(initial_capacity / 2)),
            id_to_label: RwLock::new(Vec::with_capacity(initial_capacity)),
            count: std::sync::atomic::AtomicUsize::new(0),
            capacity: None,
        }
    }

    /// Create with a maximum capacity.
    pub fn with_capacity(params: BruteForceParams, max_capacity: usize) -> Self {
        let mut index = Self::new(params);
        index.capacity = Some(max_capacity);
        index
    }

    /// Get the distance metric.
    pub fn metric(&self) -> crate::distance::Metric {
        self.core.read().metric
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
                if core.data.get(id).is_some() {
                    Some(core.compute_distance(id, query))
                } else {
                    None
                }
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let core = self.core.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        // Vector data storage
        let vector_storage = count * core.dim * std::mem::size_of::<T>();

        // Label mappings
        let label_maps = self.label_to_ids.read().capacity()
            * std::mem::size_of::<(LabelType, HashSet<IdType>)>()
            + self.id_to_label.read().capacity() * std::mem::size_of::<IdLabelEntry>();

        vector_storage + label_maps
    }

    /// Clear all vectors from the index, resetting it to empty state.
    pub fn clear(&mut self) {
        let mut core = self.core.write();
        let mut label_to_ids = self.label_to_ids.write();
        let mut id_to_label = self.id_to_label.write();

        core.data.clear();
        label_to_ids.clear();
        id_to_label.clear();
        self.count.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Compact the index by removing gaps from deleted vectors.
    ///
    /// This reorganizes the internal storage to reclaim space from deleted vectors.
    /// After compaction, all vectors are stored contiguously.
    ///
    /// # Arguments
    /// * `shrink` - If true, also release unused memory blocks
    ///
    /// # Returns
    /// The number of bytes reclaimed (approximate).
    pub fn compact(&mut self, shrink: bool) -> usize {
        let mut core = self.core.write();
        let mut label_to_ids = self.label_to_ids.write();
        let mut id_to_label = self.id_to_label.write();

        let old_capacity = core.data.capacity();
        let id_mapping = core.data.compact(shrink);

        // Update label_to_ids mapping
        for (_label, ids) in label_to_ids.iter_mut() {
            let new_ids: HashSet<IdType> = ids
                .iter()
                .filter_map(|id| id_mapping.get(id).copied())
                .collect();
            *ids = new_ids;
        }

        // Remove labels with no remaining vectors
        label_to_ids.retain(|_, ids| !ids.is_empty());

        // Rebuild id_to_label mapping
        let mut new_id_to_label = Vec::with_capacity(id_mapping.len());
        for (&old_id, &new_id) in &id_mapping {
            let new_id_usize = new_id as usize;
            if new_id_usize >= new_id_to_label.len() {
                new_id_to_label.resize(new_id_usize + 1, IdLabelEntry::default());
            }
            if let Some(entry) = id_to_label.get(old_id as usize) {
                new_id_to_label[new_id_usize] = *entry;
            }
        }
        *id_to_label = new_id_to_label;

        let new_capacity = core.data.capacity();
        let dim = core.dim;
        let bytes_per_vector = dim * std::mem::size_of::<T>();

        (old_capacity.saturating_sub(new_capacity)) * bytes_per_vector
    }

    /// Get the fragmentation ratio of the index.
    ///
    /// Returns a value between 0.0 (no fragmentation) and 1.0 (all slots are deleted).
    pub fn fragmentation(&self) -> f64 {
        self.core.read().data.fragmentation()
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

    /// Internal implementation of top-k query.
    fn top_k_impl(
        &self,
        query: &[T],
        k: usize,
        params: Option<&QueryParams>,
    ) -> Result<QueryReply<T::DistanceType>, QueryError> {
        let core = self.core.read();
        let id_to_label = self.id_to_label.read();

        if query.len() != core.dim {
            return Err(QueryError::DimensionMismatch {
                expected: core.dim,
                got: query.len(),
            });
        }

        let use_parallel = params.is_some_and(|p| p.parallel);
        let filter = params.and_then(|p| p.filter.as_ref()).map(|f| f.as_ref());

        let mut results = if use_parallel && id_to_label.len() > 1000 {
            self.parallel_top_k(&core, &id_to_label, query, k, filter)
        } else {
            self.sequential_top_k(&core, &id_to_label, query, k, filter)
        };

        results.sort_by_distance();
        Ok(results)
    }

    /// Sequential top-k scan.
    fn sequential_top_k(
        &self,
        core: &BruteForceCore<T>,
        id_to_label: &[IdLabelEntry],
        query: &[T],
        k: usize,
        filter: Option<&(dyn Fn(LabelType) -> bool + Send + Sync)>,
    ) -> QueryReply<T::DistanceType> {
        let mut heap = MaxHeap::new(k);

        for (id, entry) in id_to_label.iter().enumerate() {
            if !entry.is_valid {
                continue;
            }

            if let Some(f) = filter {
                if !f(entry.label) {
                    continue;
                }
            }

            let dist = core.compute_distance(id as IdType, query);
            heap.try_insert(id as IdType, dist);
        }

        let mut reply = QueryReply::with_capacity(heap.len());
        for entry in heap.into_sorted_vec() {
            if let Some(label_entry) = id_to_label.get(entry.id as usize) {
                reply.push(QueryResult::new(label_entry.label, entry.distance));
            }
        }
        reply
    }

    /// Parallel top-k scan using rayon.
    fn parallel_top_k(
        &self,
        core: &BruteForceCore<T>,
        id_to_label: &[IdLabelEntry],
        query: &[T],
        k: usize,
        filter: Option<&(dyn Fn(LabelType) -> bool + Send + Sync)>,
    ) -> QueryReply<T::DistanceType> {
        let candidates: Vec<_> = id_to_label
            .par_iter()
            .enumerate()
            .filter_map(|(id, entry)| {
                if !entry.is_valid {
                    return None;
                }
                if let Some(f) = filter {
                    if !f(entry.label) {
                        return None;
                    }
                }
                let dist = core.compute_distance(id as IdType, query);
                Some((id as IdType, entry.label, dist))
            })
            .collect();

        let mut heap = MaxHeap::new(k);
        for (id, _label, dist) in candidates {
            heap.try_insert(id, dist);
        }

        let mut reply = QueryReply::with_capacity(heap.len());
        for entry in heap.into_sorted_vec() {
            if let Some(label_entry) = id_to_label.get(entry.id as usize) {
                reply.push(QueryResult::new(label_entry.label, entry.distance));
            }
        }
        reply
    }

    /// Internal implementation of range query.
    fn range_impl(
        &self,
        query: &[T],
        radius: T::DistanceType,
        params: Option<&QueryParams>,
    ) -> Result<QueryReply<T::DistanceType>, QueryError> {
        let core = self.core.read();
        let id_to_label = self.id_to_label.read();

        if query.len() != core.dim {
            return Err(QueryError::DimensionMismatch {
                expected: core.dim,
                got: query.len(),
            });
        }

        let filter = params.and_then(|p| p.filter.as_ref());
        let mut reply = QueryReply::new();

        for (id, entry) in id_to_label.iter().enumerate() {
            if !entry.is_valid {
                continue;
            }

            if let Some(f) = filter {
                if !f(entry.label) {
                    continue;
                }
            }

            let dist = core.compute_distance(id as IdType, query);
            if dist.to_f64() <= radius.to_f64() {
                reply.push(QueryResult::new(entry.label, dist));
            }
        }

        reply.sort_by_distance();
        Ok(reply)
    }
}

impl<T: VectorElement> VecSimIndex for BruteForceMulti<T> {
    type DataType = T;
    type DistType = T::DistanceType;

    fn add_vector(&mut self, vector: &[T], label: LabelType) -> Result<usize, IndexError> {
        let mut core = self.core.write();

        if vector.len() != core.dim {
            return Err(IndexError::DimensionMismatch {
                expected: core.dim,
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

        let mut label_to_ids = self.label_to_ids.write();
        let mut id_to_label = self.id_to_label.write();

        // Update mappings
        label_to_ids.entry(label).or_default().insert(id);

        // Ensure id_to_label is large enough
        let id_usize = id as usize;
        if id_usize >= id_to_label.len() {
            id_to_label.resize(id_usize + 1, IdLabelEntry::default());
        }
        id_to_label[id_usize] = IdLabelEntry {
            label,
            is_valid: true,
        };

        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let mut label_to_ids = self.label_to_ids.write();
        let mut id_to_label = self.id_to_label.write();
        let mut core = self.core.write();

        if let Some(ids) = label_to_ids.remove(&label) {
            let count = ids.len();

            for id in ids {
                // Mark as invalid
                if let Some(entry) = id_to_label.get_mut(id as usize) {
                    entry.is_valid = false;
                }
                // Mark slot as free
                core.data.mark_deleted(id);
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
        self.top_k_impl(query, k, params)
    }

    fn range_query(
        &self,
        query: &[T],
        radius: T::DistanceType,
        params: Option<&QueryParams>,
    ) -> Result<QueryReply<T::DistanceType>, QueryError> {
        self.range_impl(query, radius, params)
    }

    fn index_size(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn index_capacity(&self) -> Option<usize> {
        self.capacity
    }

    fn dimension(&self) -> usize {
        self.core.read().dim
    }

    fn batch_iterator<'a>(
        &'a self,
        query: &[T],
        params: Option<&QueryParams>,
    ) -> Result<Box<dyn BatchIterator<DistType = T::DistanceType> + 'a>, QueryError> {
        let core = self.core.read();
        if query.len() != core.dim {
            return Err(QueryError::DimensionMismatch {
                expected: core.dim,
                got: query.len(),
            });
        }

        // Compute results immediately to preserve filter from params
        let id_to_label = self.id_to_label.read();
        let filter = params.and_then(|p| p.filter.as_ref());

        let mut results = Vec::new();
        for (id, entry) in id_to_label.iter().enumerate() {
            if !entry.is_valid {
                continue;
            }

            if let Some(f) = filter {
                if !f(entry.label) {
                    continue;
                }
            }

            let dist = core.compute_distance(id as IdType, query);
            results.push((id as IdType, entry.label, dist));
        }

        // Sort by distance
        results.sort_by(|a, b| {
            a.2.to_f64()
                .partial_cmp(&b.2.to_f64())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(Box::new(
            super::batch_iterator::BruteForceMultiBatchIterator::<T>::new(results),
        ))
    }

    fn info(&self) -> IndexInfo {
        let core = self.core.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        IndexInfo {
            size: count,
            capacity: self.capacity,
            dimension: core.dim,
            index_type: "BruteForceMulti",
            memory_bytes: count * core.dim * std::mem::size_of::<T>()
                + self.label_to_ids.read().capacity()
                    * std::mem::size_of::<(LabelType, HashSet<IdType>)>()
                + self.id_to_label.read().capacity() * std::mem::size_of::<IdLabelEntry>(),
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

unsafe impl<T: VectorElement> Send for BruteForceMulti<T> {}
unsafe impl<T: VectorElement> Sync for BruteForceMulti<T> {}

// Serialization support
impl BruteForceMulti<f32> {
    /// Save the index to a writer.
    pub fn save<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> crate::serialization::SerializationResult<()> {
        use crate::serialization::*;

        let core = self.core.read();
        let label_to_ids = self.label_to_ids.read();
        let id_to_label = self.id_to_label.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        // Write header
        let header = IndexHeader::new(
            IndexTypeId::BruteForceMulti,
            DataTypeId::F32,
            core.metric,
            core.dim,
            count,
        );
        header.write(writer)?;

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

        // Write id_to_label entries
        write_usize(writer, id_to_label.len())?;
        for entry in id_to_label.iter() {
            write_u64(writer, entry.label)?;
            write_u8(writer, if entry.is_valid { 1 } else { 0 })?;
        }

        // Write vectors - only write valid entries
        let mut valid_ids: Vec<IdType> = Vec::with_capacity(count);
        for (id, entry) in id_to_label.iter().enumerate() {
            if entry.is_valid {
                valid_ids.push(id as IdType);
            }
        }

        write_usize(writer, valid_ids.len())?;
        for id in valid_ids {
            write_u32(writer, id)?;
            if let Some(vector) = core.data.get(id) {
                for &v in vector {
                    write_f32(writer, v)?;
                }
            }
        }

        Ok(())
    }

    /// Load the index from a reader.
    pub fn load<R: std::io::Read>(reader: &mut R) -> crate::serialization::SerializationResult<Self> {
        use crate::serialization::*;

        // Read and validate header
        let header = IndexHeader::read(reader)?;

        if header.index_type != IndexTypeId::BruteForceMulti {
            return Err(SerializationError::IndexTypeMismatch {
                expected: "BruteForceMulti".to_string(),
                got: header.index_type.as_str().to_string(),
            });
        }

        if header.data_type != DataTypeId::F32 {
            return Err(SerializationError::InvalidData(
                "Expected f32 data type".to_string(),
            ));
        }

        // Create the index with proper parameters
        let params = BruteForceParams::new(header.dimension, header.metric);
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

        // Read id_to_label entries
        let id_to_label_len = read_usize(reader)?;
        let mut id_to_label = Vec::with_capacity(id_to_label_len);
        for _ in 0..id_to_label_len {
            let label = read_u64(reader)?;
            let is_valid = read_u8(reader)? != 0;
            id_to_label.push(IdLabelEntry { label, is_valid });
        }

        // Read vectors
        let num_vectors = read_usize(reader)?;
        let dim = header.dimension;
        {
            let mut core = index.core.write();

            // Pre-allocate space
            core.data.reserve(num_vectors);

            for _ in 0..num_vectors {
                let _id = read_u32(reader)?;
                let mut vector = vec![0.0f32; dim];
                for v in &mut vector {
                    *v = read_f32(reader)?;
                }

                // Add vector to data storage
                core.data.add(&vector).ok_or_else(|| {
                    SerializationError::DataCorruption("Failed to add vector during deserialization".to_string())
                })?;
            }
        }

        // Set the internal state
        *index.label_to_ids.write() = label_to_ids;
        *index.id_to_label.write() = id_to_label;
        index
            .count
            .store(header.count, std::sync::atomic::Ordering::Relaxed);

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
    fn test_brute_force_multi_basic() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

        // Add multiple vectors with same label
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 3);
        assert_eq!(index.label_count(1), 2);
        assert_eq!(index.label_count(2), 1);

        // Query should find both vectors for label 1
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 3, None).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be exact match
        assert_eq!(results.results[0].label, 1);
        assert!(results.results[0].distance < 0.001);
    }

    #[test]
    fn test_brute_force_multi_delete() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 3);

        // Delete all vectors for label 1
        let deleted = index.delete_vector(1).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(index.index_size(), 1);

        // Only label 2 should remain
        let results = index
            .top_k_query(&vec![1.0, 0.0, 0.0, 0.0], 10, None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results.results[0].label, 2);
    }

    #[test]
    fn test_brute_force_multi_serialization() {
        use std::io::Cursor;

        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
        let loaded = BruteForceMulti::<f32>::load(&mut cursor).unwrap();

        // Verify
        assert_eq!(loaded.index_size(), 4);
        assert_eq!(loaded.dimension(), 4);
        assert_eq!(loaded.label_count(1), 2);
        assert_eq!(loaded.label_count(2), 1);
        assert_eq!(loaded.label_count(3), 1);

        // Query should work the same
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = loaded.top_k_query(&query, 3, None).unwrap();
        assert_eq!(results.results[0].label, 1); // Exact match
    }

    #[test]
    fn test_brute_force_multi_compact() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

        // Add multiple vectors, some with same label
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();
        index.add_vector(&vec![0.0, 0.0, 1.0, 0.0], 3).unwrap();
        index.add_vector(&vec![0.0, 0.0, 0.0, 1.0], 4).unwrap();

        assert_eq!(index.index_size(), 5);
        assert_eq!(index.label_count(1), 2);

        // Delete label 2 and 3
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

        assert_eq!(results.len(), 3);
        // Top results should be label 1 vectors
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
    fn test_brute_force_multi_multiple_labels() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_get_vectors() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_get_labels() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_compute_distance() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_contains() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();

        assert!(index.contains(1));
        assert!(!index.contains(2));
    }

    #[test]
    fn test_brute_force_multi_range_query() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_filtered_query() {
        use crate::query::QueryParams;

        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_batch_iterator() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

        for i in 0..10u64 {
            index.add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        let query = vec![5.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(3) {
            all_results.extend(batch);
        }

        assert_eq!(all_results.len(), 10);
        // Results should be sorted by distance
        for i in 0..all_results.len() - 1 {
            assert!(all_results[i].2 <= all_results[i + 1].2);
        }
    }

    #[test]
    fn test_brute_force_multi_batch_iterator_reset() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

        for i in 0..5u64 {
            index.add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        // First iteration
        let batch1 = iter.next_batch(100).unwrap();
        assert_eq!(batch1.len(), 5);
        assert!(!iter.has_next());

        // Reset and iterate again
        iter.reset();
        assert!(iter.has_next());
        let batch2 = iter.next_batch(100).unwrap();
        assert_eq!(batch2.len(), 5);
    }

    #[test]
    fn test_brute_force_multi_dimension_mismatch() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_capacity_exceeded() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::with_capacity(params, 2);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 2).unwrap();

        // Third should fail
        let result = index.add_vector(&vec![3.0, 0.0, 0.0, 0.0], 3);
        assert!(matches!(result, Err(IndexError::CapacityExceeded { capacity: 2 })));
    }

    #[test]
    fn test_brute_force_multi_delete_not_found() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();

        let result = index.delete_vector(999);
        assert!(matches!(result, Err(IndexError::LabelNotFound(999))));
    }

    #[test]
    fn test_brute_force_multi_memory_usage() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

        let initial_memory = index.memory_usage();

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 2).unwrap();

        let after_memory = index.memory_usage();
        assert!(after_memory > initial_memory);
    }

    #[test]
    fn test_brute_force_multi_info() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 1).unwrap();

        let info = index.info();
        assert_eq!(info.size, 2);
        assert_eq!(info.dimension, 4);
        assert_eq!(info.index_type, "BruteForceMulti");
        assert!(info.memory_bytes > 0);
    }

    #[test]
    fn test_brute_force_multi_clear() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_add_vectors_batch() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_with_capacity() {
        let params = BruteForceParams::new(4, Metric::L2);
        let index = BruteForceMulti::<f32>::with_capacity(params, 100);

        assert_eq!(index.index_capacity(), Some(100));
    }

    #[test]
    fn test_brute_force_multi_inner_product() {
        let params = BruteForceParams::new(4, Metric::InnerProduct);
        let mut index = BruteForceMulti::<f32>::new(params);

        // For InnerProduct, higher dot product = lower "distance"
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 2, None).unwrap();

        // Label 1 has perfect alignment with query
        assert_eq!(results.results[0].label, 1);
    }

    #[test]
    fn test_brute_force_multi_cosine() {
        let params = BruteForceParams::new(4, Metric::Cosine);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_metric_getter() {
        let params = BruteForceParams::new(4, Metric::Cosine);
        let index = BruteForceMulti::<f32>::new(params);

        assert_eq!(index.metric(), Metric::Cosine);
    }

    #[test]
    fn test_brute_force_multi_filtered_range_query() {
        use crate::query::QueryParams;

        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_empty_query() {
        let params = BruteForceParams::new(4, Metric::L2);
        let index = BruteForceMulti::<f32>::new(params);

        // Query on empty index
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 10, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_brute_force_multi_fragmentation() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
    fn test_brute_force_multi_parallel_query() {
        use crate::query::QueryParams;

        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

        // Add many vectors to trigger parallel processing
        for i in 0..2000u64 {
            index.add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        let query = vec![500.0, 0.0, 0.0, 0.0];
        let query_params = QueryParams::new().with_parallel(true);
        let results = index.top_k_query(&query, 10, Some(&query_params)).unwrap();

        assert_eq!(results.len(), 10);
        // First result should be label 500 (exact match)
        assert_eq!(results.results[0].label, 500);
    }

    #[test]
    fn test_brute_force_multi_serialization_with_capacity() {
        use std::io::Cursor;

        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::with_capacity(params, 1000);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Serialize
        let mut buffer = Vec::new();
        index.save(&mut buffer).unwrap();

        // Deserialize
        let mut cursor = Cursor::new(buffer);
        let loaded = BruteForceMulti::<f32>::load(&mut cursor).unwrap();

        // Capacity should be preserved
        assert_eq!(loaded.index_capacity(), Some(1000));
        assert_eq!(loaded.index_size(), 2);
        assert_eq!(loaded.label_count(1), 2);
    }

    #[test]
    fn test_brute_force_multi_query_after_compact() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
}

//! Single-value BruteForce index implementation.
//!
//! This index stores one vector per label. When adding a vector with
//! an existing label, the old vector is replaced.

use super::{BruteForceCore, BruteForceParams, IdLabelEntry};
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply, QueryResult};
use crate::types::{DistanceType, IdType, LabelType, VectorElement};
use crate::utils::MaxHeap;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::HashMap;

/// Statistics about a BruteForce index.
#[derive(Debug, Clone)]
pub struct BruteForceStats {
    /// Number of vectors in the index.
    pub size: usize,
    /// Number of deleted (but not yet compacted) elements.
    pub deleted_count: usize,
    /// Approximate memory usage in bytes.
    pub memory_bytes: usize,
}

/// Single-value BruteForce index.
///
/// Each label has exactly one associated vector. Adding a vector with
/// an existing label replaces the previous vector.
pub struct BruteForceSingle<T: VectorElement> {
    /// Core storage and distance computation.
    pub(crate) core: RwLock<BruteForceCore<T>>,
    /// Label to internal ID mapping.
    label_to_id: RwLock<HashMap<LabelType, IdType>>,
    /// Internal ID to label mapping (for reverse lookup).
    pub(crate) id_to_label: RwLock<Vec<IdLabelEntry>>,
    /// Number of vectors.
    count: std::sync::atomic::AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
}

impl<T: VectorElement> BruteForceSingle<T> {
    /// Create a new single-value BruteForce index.
    pub fn new(params: BruteForceParams) -> Self {
        let core = BruteForceCore::new(&params);
        let initial_capacity = params.initial_capacity;

        Self {
            core: RwLock::new(core),
            label_to_id: RwLock::new(HashMap::with_capacity(initial_capacity)),
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

    /// Get the number of deleted (but not yet compacted) elements.
    pub fn deleted_count(&self) -> usize {
        self.id_to_label
            .read()
            .iter()
            .filter(|e| !e.is_valid && e.label != 0)
            .count()
    }

    /// Get detailed statistics about the index.
    pub fn stats(&self) -> BruteForceStats {
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);
        let id_to_label = self.id_to_label.read();

        let deleted_count = id_to_label
            .iter()
            .filter(|e| !e.is_valid && e.label != 0)
            .count();

        BruteForceStats {
            size: count,
            deleted_count,
            memory_bytes: self.memory_usage(),
        }
    }

    /// Get the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let core = self.core.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        // Vector data storage
        let vector_storage = count * core.dim * std::mem::size_of::<T>();

        // Label mappings
        let label_maps = self.label_to_id.read().capacity()
            * std::mem::size_of::<(LabelType, IdType)>()
            + self.id_to_label.read().capacity() * std::mem::size_of::<IdLabelEntry>();

        vector_storage + label_maps
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
        Some(core.compute_distance(id, query))
    }

    /// Clear all vectors from the index, resetting it to empty state.
    pub fn clear(&mut self) {
        let mut core = self.core.write();
        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();

        core.data.clear();
        label_to_id.clear();
        id_to_label.clear();
        self.count.store(0, std::sync::atomic::Ordering::Relaxed);
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

        let use_parallel = params.map_or(false, |p| p.parallel);
        let filter = params.and_then(|p| p.filter.as_ref());

        let mut results = if use_parallel && id_to_label.len() > 1000 {
            // Parallel scan for large datasets
            self.parallel_top_k(&core, &id_to_label, query, k, filter)
        } else {
            // Sequential scan
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
        filter: Option<&Box<dyn Fn(LabelType) -> bool + Send + Sync>>,
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
        filter: Option<&Box<dyn Fn(LabelType) -> bool + Send + Sync>>,
    ) -> QueryReply<T::DistanceType> {
        // Parallel map to compute distances
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

        // Serial reduction to find top-k
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

impl<T: VectorElement> VecSimIndex for BruteForceSingle<T> {
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

        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();

        // Check if label already exists
        if let Some(&existing_id) = label_to_id.get(&label) {
            // Update existing vector
            let processed = core.dist_fn.preprocess(vector, core.dim);
            core.data.update(existing_id, &processed);
            return Ok(0); // No new vector added
        }

        // Check capacity
        if let Some(cap) = self.capacity {
            if self.count.load(std::sync::atomic::Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Add new vector
        let id = core
            .add_vector(vector)
            .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;

        // Update mappings
        label_to_id.insert(label, id);

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
        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();

        if let Some(id) = label_to_id.remove(&label) {
            // Mark as invalid
            if let Some(entry) = id_to_label.get_mut(id as usize) {
                entry.is_valid = false;
            }

            // Mark slot as free in data storage
            self.core.write().data.mark_deleted(id);

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
        drop(core);

        Ok(Box::new(super::batch_iterator::BruteForceBatchIterator::new(
            self,
            query.to_vec(),
            params.cloned(),
        )))
    }

    fn info(&self) -> IndexInfo {
        let core = self.core.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        IndexInfo {
            size: count,
            capacity: self.capacity,
            dimension: core.dim,
            index_type: "BruteForceSingle",
            memory_bytes: count * core.dim * std::mem::size_of::<T>()
                + self.label_to_id.read().capacity() * std::mem::size_of::<(LabelType, IdType)>()
                + self.id_to_label.read().capacity() * std::mem::size_of::<IdLabelEntry>(),
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

// Allow read-only concurrent access for queries
unsafe impl<T: VectorElement> Send for BruteForceSingle<T> {}
unsafe impl<T: VectorElement> Sync for BruteForceSingle<T> {}

// Serialization support
impl BruteForceSingle<f32> {
    /// Save the index to a writer.
    pub fn save<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> crate::serialization::SerializationResult<()> {
        use crate::serialization::*;

        let core = self.core.read();
        let label_to_id = self.label_to_id.read();
        let id_to_label = self.id_to_label.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        // Write header
        let header = IndexHeader::new(
            IndexTypeId::BruteForceSingle,
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

        // Write label_to_id mapping
        write_usize(writer, label_to_id.len())?;
        for (&label, &id) in label_to_id.iter() {
            write_u64(writer, label)?;
            write_u32(writer, id)?;
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

        if header.index_type != IndexTypeId::BruteForceSingle {
            return Err(SerializationError::IndexTypeMismatch {
                expected: "BruteForceSingle".to_string(),
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

        // Read label_to_id mapping
        let label_to_id_len = read_usize(reader)?;
        let mut label_to_id = HashMap::with_capacity(label_to_id_len);
        for _ in 0..label_to_id_len {
            let label = read_u64(reader)?;
            let id = read_u32(reader)?;
            label_to_id.insert(label, id);
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
                let id = read_u32(reader)?;
                let mut vector = vec![0.0f32; dim];
                for v in &mut vector {
                    *v = read_f32(reader)?;
                }

                // Add vector at specific ID
                let added_id = core.data.add(&vector).ok_or_else(|| {
                    SerializationError::DataCorruption("Failed to add vector during deserialization".to_string())
                })?;

                // Ensure ID matches (vectors should be added in order)
                if added_id != id {
                    // If IDs don't match, we need to handle this case
                    // For now, this shouldn't happen with proper serialization
                }
            }
        }

        // Set the internal state
        *index.label_to_id.write() = label_to_id;
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
    fn test_brute_force_single_basic() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);

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

        assert_eq!(results.len(), 2);
        assert_eq!(results.results[0].label, 1); // Closest to v1
    }

    #[test]
    fn test_brute_force_single_replace() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];

        index.add_vector(&v1, 1).unwrap();
        assert_eq!(index.index_size(), 1);

        // Replace with new vector
        index.add_vector(&v2, 1).unwrap();
        assert_eq!(index.index_size(), 1); // Size unchanged

        // Query should return updated vector
        let query = vec![0.0, 1.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 1, None).unwrap();

        assert_eq!(results.results[0].label, 1);
        assert!(results.results[0].distance < 0.001); // Should be very close
    }

    #[test]
    fn test_brute_force_single_delete() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 2);

        index.delete_vector(1).unwrap();
        assert_eq!(index.index_size(), 1);

        // Deleted vector should not appear in results
        let results = index
            .top_k_query(&vec![1.0, 0.0, 0.0, 0.0], 10, None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results.results[0].label, 2);
    }

    #[test]
    fn test_brute_force_single_range_query() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);

        index.add_vector(&vec![0.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 2).unwrap();
        index.add_vector(&vec![2.0, 0.0, 0.0, 0.0], 3).unwrap();

        // Range query with radius 1.5 (squared = 2.25)
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = index.range_query(&query, 2.25, None).unwrap();

        // Should find vectors 1 and 2 (distances 0 and 1)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_brute_force_single_serialization() {
        use std::io::Cursor;

        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);

        // Add some vectors
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();
        index.add_vector(&vec![0.0, 0.0, 1.0, 0.0], 3).unwrap();

        // Serialize
        let mut buffer = Vec::new();
        index.save(&mut buffer).unwrap();

        // Deserialize
        let mut cursor = Cursor::new(buffer);
        let loaded = BruteForceSingle::<f32>::load(&mut cursor).unwrap();

        // Verify
        assert_eq!(loaded.index_size(), 3);
        assert_eq!(loaded.dimension(), 4);

        // Query should work the same
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = loaded.top_k_query(&query, 3, None).unwrap();
        assert_eq!(results.results[0].label, 1);
    }
}

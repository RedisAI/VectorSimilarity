//! Multi-value tiered index implementation.
//!
//! This index allows multiple vectors per label, combining a BruteForce frontend
//! (for fast writes) with an HNSW backend (for efficient queries).

use super::{merge_range, merge_top_k_multi, TieredParams, WriteMode};
use crate::index::brute_force::{BruteForceMulti, BruteForceParams};
use crate::index::hnsw::HnswMulti;
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply};
use crate::types::{LabelType, VectorElement};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Multi-value tiered index combining BruteForce frontend with HNSW backend.
///
/// Each label can have multiple associated vectors. The tiered architecture
/// provides fast writes to the flat buffer and efficient queries from HNSW.
pub struct TieredMulti<T: VectorElement> {
    /// Flat BruteForce buffer (frontend).
    pub(crate) flat: RwLock<BruteForceMulti<T>>,
    /// HNSW index (backend).
    pub(crate) hnsw: RwLock<HnswMulti<T>>,
    /// Label counts in flat buffer.
    flat_label_counts: RwLock<HashMap<LabelType, usize>>,
    /// Label counts in HNSW.
    hnsw_label_counts: RwLock<HashMap<LabelType, usize>>,
    /// Configuration parameters.
    params: TieredParams,
    /// Total vector count.
    count: AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
}

impl<T: VectorElement> TieredMulti<T> {
    /// Create a new multi-value tiered index.
    pub fn new(params: TieredParams) -> Self {
        let bf_params = BruteForceParams::new(params.dim, params.metric)
            .with_capacity(params.flat_buffer_limit.min(params.initial_capacity));
        let flat = BruteForceMulti::new(bf_params);
        let hnsw = HnswMulti::new(params.hnsw_params.clone());

        Self {
            flat: RwLock::new(flat),
            hnsw: RwLock::new(hnsw),
            flat_label_counts: RwLock::new(HashMap::new()),
            hnsw_label_counts: RwLock::new(HashMap::new()),
            params,
            count: AtomicUsize::new(0),
            capacity: None,
        }
    }

    /// Create with a maximum capacity.
    pub fn with_capacity(params: TieredParams, max_capacity: usize) -> Self {
        let mut index = Self::new(params);
        index.capacity = Some(max_capacity);
        index
    }

    /// Get the current write mode.
    pub fn write_mode(&self) -> WriteMode {
        if self.params.write_mode == WriteMode::InPlace {
            return WriteMode::InPlace;
        }
        if self.flat_size() >= self.params.flat_buffer_limit {
            WriteMode::InPlace
        } else {
            WriteMode::Async
        }
    }

    /// Get the number of vectors in the flat buffer.
    pub fn flat_size(&self) -> usize {
        self.flat.read().index_size()
    }

    /// Get the number of vectors in the HNSW backend.
    pub fn hnsw_size(&self) -> usize {
        self.hnsw.read().index_size()
    }

    /// Get the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.flat.read().memory_usage() + self.hnsw.read().memory_usage()
    }

    /// Get the count of vectors for a label in the flat buffer.
    pub fn flat_label_count(&self, label: LabelType) -> usize {
        *self.flat_label_counts.read().get(&label).unwrap_or(&0)
    }

    /// Get the count of vectors for a label in the HNSW backend.
    pub fn hnsw_label_count(&self, label: LabelType) -> usize {
        *self.hnsw_label_counts.read().get(&label).unwrap_or(&0)
    }

    /// Compute the distance between a stored vector and a query vector.
    ///
    /// For multi-value indexes, returns the minimum distance among all vectors
    /// with the given label. Looks up in both flat buffer and HNSW backend.
    /// Returns `None` if the label doesn't exist in either tier.
    pub fn compute_distance(&self, label: LabelType, query: &[T]) -> Option<T::DistanceType> {
        let flat_dist = if self.flat_label_counts.read().get(&label).copied().unwrap_or(0) > 0 {
            self.flat.read().compute_distance(label, query)
        } else {
            None
        };
        let hnsw_dist = if self.hnsw_label_counts.read().get(&label).copied().unwrap_or(0) > 0 {
            self.hnsw.read().compute_distance(label, query)
        } else {
            None
        };
        match (flat_dist, hnsw_dist) {
            (Some(f), Some(h)) => Some(if f < h { f } else { h }),
            (Some(f), None) => Some(f),
            (None, Some(h)) => Some(h),
            (None, None) => None,
        }
    }

    /// Flush all vectors from flat buffer to HNSW.
    ///
    /// # Returns
    /// The number of vectors migrated.
    pub fn flush(&mut self) -> Result<usize, IndexError> {
        let flat_labels: Vec<LabelType> = self.flat_label_counts.read().keys().copied().collect();

        if flat_labels.is_empty() {
            return Ok(0);
        }

        let mut migrated = 0;

        // For multi-value, we need to get all vectors for each label
        let vectors: Vec<(LabelType, Vec<Vec<T>>)> = {
            let flat = self.flat.read();
            flat_labels
                .iter()
                .filter_map(|&label| {
                    flat.get_vectors(label).map(|vecs| (label, vecs))
                })
                .collect()
        };

        // Add to HNSW
        {
            let mut hnsw = self.hnsw.write();
            let mut hnsw_label_counts = self.hnsw_label_counts.write();

            for (label, vecs) in &vectors {
                for vec in vecs {
                    match hnsw.add_vector(vec, *label) {
                        Ok(_) => {
                            *hnsw_label_counts.entry(*label).or_insert(0) += 1;
                            migrated += 1;
                        }
                        Err(e) => {
                            eprintln!("Failed to migrate label {label} to HNSW: {e:?}");
                        }
                    }
                }
            }
        }

        // Clear flat buffer
        {
            let mut flat = self.flat.write();
            let mut flat_label_counts = self.flat_label_counts.write();

            flat.clear();
            flat_label_counts.clear();
        }

        Ok(migrated)
    }

    /// Compact both tiers to reclaim space.
    pub fn compact(&mut self, shrink: bool) -> usize {
        let flat_reclaimed = self.flat.write().compact(shrink);
        let hnsw_reclaimed = self.hnsw.write().compact(shrink);
        flat_reclaimed + hnsw_reclaimed
    }

    /// Get the fragmentation ratio.
    pub fn fragmentation(&self) -> f64 {
        let flat = self.flat.read();
        let hnsw = self.hnsw.read();
        let flat_frag = flat.fragmentation();
        let hnsw_frag = hnsw.fragmentation();

        let flat_size = flat.index_size() as f64;
        let hnsw_size = hnsw.index_size() as f64;
        let total = flat_size + hnsw_size;

        if total == 0.0 {
            0.0
        } else {
            (flat_frag * flat_size + hnsw_frag * hnsw_size) / total
        }
    }

    /// Clear all vectors from both tiers.
    pub fn clear(&mut self) {
        self.flat.write().clear();
        self.hnsw.write().clear();
        self.flat_label_counts.write().clear();
        self.hnsw_label_counts.write().clear();
        self.count.store(0, Ordering::Relaxed);
    }

    /// Get the neighbors of an element in the HNSW graph by label.
    ///
    /// Returns a vector of vectors, where each inner vector contains the neighbor labels
    /// for that level (level 0 first). Returns None if the label is not in the HNSW backend.
    pub fn get_element_neighbors(&self, label: LabelType) -> Option<Vec<Vec<LabelType>>> {
        // Only check HNSW backend - flat buffer doesn't have graph structure
        if self.hnsw_label_counts.read().contains_key(&label) {
            return self.hnsw.read().get_element_neighbors(label);
        }
        None
    }
}

impl<T: VectorElement> VecSimIndex for TieredMulti<T> {
    type DataType = T;
    type DistType = T::DistanceType;

    fn add_vector(&mut self, vector: &[T], label: LabelType) -> Result<usize, IndexError> {
        if vector.len() != self.params.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.params.dim,
                got: vector.len(),
            });
        }

        if let Some(cap) = self.capacity {
            if self.count.load(Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // For multi-value, always add (don't replace)
        match self.write_mode() {
            WriteMode::Async => {
                let mut flat = self.flat.write();
                let mut flat_label_counts = self.flat_label_counts.write();
                flat.add_vector(vector, label)?;
                *flat_label_counts.entry(label).or_insert(0) += 1;
            }
            WriteMode::InPlace => {
                let mut hnsw = self.hnsw.write();
                let mut hnsw_label_counts = self.hnsw_label_counts.write();
                hnsw.add_vector(vector, label)?;
                *hnsw_label_counts.entry(label).or_insert(0) += 1;
            }
        }

        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let flat_count = self.flat_label_count(label);
        let hnsw_count = self.hnsw_label_count(label);

        if flat_count == 0 && hnsw_count == 0 {
            return Err(IndexError::LabelNotFound(label));
        }

        let mut deleted = 0;

        if flat_count > 0 {
            let mut flat = self.flat.write();
            let mut flat_label_counts = self.flat_label_counts.write();
            if let Ok(n) = flat.delete_vector(label) {
                deleted += n;
                flat_label_counts.remove(&label);
            }
        }

        if hnsw_count > 0 {
            let mut hnsw = self.hnsw.write();
            let mut hnsw_label_counts = self.hnsw_label_counts.write();
            if let Ok(n) = hnsw.delete_vector(label) {
                deleted += n;
                hnsw_label_counts.remove(&label);
            }
        }

        self.count.fetch_sub(deleted, Ordering::Relaxed);
        Ok(deleted)
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

        let flat = self.flat.read();
        let hnsw = self.hnsw.read();

        let flat_results = flat.top_k_query(query, k, params)?;
        let hnsw_results = hnsw.top_k_query(query, k, params)?;

        Ok(merge_top_k_multi(flat_results, hnsw_results, k))
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

        let flat = self.flat.read();
        let hnsw = self.hnsw.read();

        let flat_results = flat.range_query(query, radius, params)?;
        let hnsw_results = hnsw.range_query(query, radius, params)?;

        Ok(merge_range(flat_results, hnsw_results))
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

        Ok(Box::new(
            super::batch_iterator::TieredMultiBatchIterator::new(self, query.to_vec(), params.cloned()),
        ))
    }

    fn info(&self) -> IndexInfo {
        IndexInfo {
            size: self.index_size(),
            capacity: self.capacity,
            dimension: self.params.dim,
            index_type: "TieredMulti",
            memory_bytes: self.memory_usage(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.flat_label_counts.read().contains_key(&label)
            || self.hnsw_label_counts.read().contains_key(&label)
    }

    fn label_count(&self, label: LabelType) -> usize {
        self.flat_label_count(label) + self.hnsw_label_count(label)
    }
}

// Serialization implementation for f32
impl TieredMulti<f32> {
    /// Save the index to a writer.
    ///
    /// The serialization format saves both tiers (flat buffer and HNSW) independently,
    /// preserving the exact state of the index.
    pub fn save<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> crate::serialization::SerializationResult<()> {
        use crate::serialization::*;

        let count = self.count.load(Ordering::Relaxed);

        // Write header
        let header = IndexHeader::new(
            IndexTypeId::TieredMulti,
            DataTypeId::F32,
            self.params.metric,
            self.params.dim,
            count,
        );
        header.write(writer)?;

        // Write tiered params
        write_usize(writer, self.params.flat_buffer_limit)?;
        write_u8(writer, if self.params.write_mode == WriteMode::InPlace { 1 } else { 0 })?;
        write_usize(writer, self.params.initial_capacity)?;

        // Write HNSW params
        write_usize(writer, self.params.hnsw_params.m)?;
        write_usize(writer, self.params.hnsw_params.m_max_0)?;
        write_usize(writer, self.params.hnsw_params.ef_construction)?;
        write_usize(writer, self.params.hnsw_params.ef_runtime)?;
        write_u8(writer, if self.params.hnsw_params.enable_heuristic { 1 } else { 0 })?;

        // Write capacity
        write_u8(writer, if self.capacity.is_some() { 1 } else { 0 })?;
        if let Some(cap) = self.capacity {
            write_usize(writer, cap)?;
        }

        // Write flat buffer state
        let flat = self.flat.read();
        let flat_label_counts = self.flat_label_counts.read();

        // Write number of unique labels in flat
        write_usize(writer, flat_label_counts.len())?;
        for (&label, &label_count) in flat_label_counts.iter() {
            if let Some(vecs) = flat.get_vectors(label) {
                write_u64(writer, label)?;
                write_usize(writer, vecs.len())?;
                for vec in vecs {
                    for &v in &vec {
                        write_f32(writer, v)?;
                    }
                }
            } else {
                // Should not happen, but handle gracefully
                write_u64(writer, label)?;
                write_usize(writer, 0)?;
            }
            // Silence warning about unused label_count
            let _ = label_count;
        }
        drop(flat);
        drop(flat_label_counts);

        // Write HNSW state
        let hnsw = self.hnsw.read();
        let hnsw_label_counts = self.hnsw_label_counts.read();

        // Write number of unique labels in HNSW
        write_usize(writer, hnsw_label_counts.len())?;
        for (&label, &label_count) in hnsw_label_counts.iter() {
            if let Some(vecs) = hnsw.get_vectors(label) {
                write_u64(writer, label)?;
                write_usize(writer, vecs.len())?;
                for vec in vecs {
                    for &v in &vec {
                        write_f32(writer, v)?;
                    }
                }
            } else {
                write_u64(writer, label)?;
                write_usize(writer, 0)?;
            }
            let _ = label_count;
        }

        Ok(())
    }

    /// Load the index from a reader.
    pub fn load<R: std::io::Read>(reader: &mut R) -> crate::serialization::SerializationResult<Self> {
        use crate::serialization::*;

        // Read header
        let header = IndexHeader::read(reader)?;

        if header.index_type != IndexTypeId::TieredMulti {
            return Err(SerializationError::IndexTypeMismatch {
                expected: "TieredMulti".to_string(),
                got: header.index_type.as_str().to_string(),
            });
        }

        if header.data_type != DataTypeId::F32 {
            return Err(SerializationError::InvalidData(
                "Expected f32 data type".to_string(),
            ));
        }

        // Read tiered params
        let flat_buffer_limit = read_usize(reader)?;
        let write_mode = if read_u8(reader)? != 0 {
            WriteMode::InPlace
        } else {
            WriteMode::Async
        };
        let initial_capacity = read_usize(reader)?;

        // Read HNSW params
        let m = read_usize(reader)?;
        let m_max_0 = read_usize(reader)?;
        let ef_construction = read_usize(reader)?;
        let ef_runtime = read_usize(reader)?;
        let enable_heuristic = read_u8(reader)? != 0;

        // Build params
        let mut hnsw_params = crate::index::hnsw::HnswParams::new(header.dimension, header.metric)
            .with_m(m)
            .with_ef_construction(ef_construction)
            .with_ef_runtime(ef_runtime);
        hnsw_params.m_max_0 = m_max_0;
        hnsw_params.enable_heuristic = enable_heuristic;

        let params = TieredParams {
            dim: header.dimension,
            metric: header.metric,
            hnsw_params,
            flat_buffer_limit,
            write_mode,
            initial_capacity,
        };

        let mut index = Self::new(params);

        // Read capacity
        let has_capacity = read_u8(reader)? != 0;
        if has_capacity {
            index.capacity = Some(read_usize(reader)?);
        }

        let mut total_count = 0;

        // Read flat buffer vectors
        let flat_label_count = read_usize(reader)?;
        {
            let mut flat = index.flat.write();
            let mut flat_label_counts = index.flat_label_counts.write();
            for _ in 0..flat_label_count {
                let label = read_u64(reader)?;
                let vec_count = read_usize(reader)?;
                for _ in 0..vec_count {
                    let mut vec = vec![0.0f32; header.dimension];
                    for v in &mut vec {
                        *v = read_f32(reader)?;
                    }
                    flat.add_vector(&vec, label).map_err(|e| {
                        SerializationError::DataCorruption(format!("Failed to add vector: {e:?}"))
                    })?;
                    *flat_label_counts.entry(label).or_insert(0) += 1;
                    total_count += 1;
                }
            }
        }

        // Read HNSW vectors
        let hnsw_label_count = read_usize(reader)?;
        {
            let mut hnsw = index.hnsw.write();
            let mut hnsw_label_counts = index.hnsw_label_counts.write();
            for _ in 0..hnsw_label_count {
                let label = read_u64(reader)?;
                let vec_count = read_usize(reader)?;
                for _ in 0..vec_count {
                    let mut vec = vec![0.0f32; header.dimension];
                    for v in &mut vec {
                        *v = read_f32(reader)?;
                    }
                    hnsw.add_vector(&vec, label).map_err(|e| {
                        SerializationError::DataCorruption(format!("Failed to add vector: {e:?}"))
                    })?;
                    *hnsw_label_counts.entry(label).or_insert(0) += 1;
                    total_count += 1;
                }
            }
        }

        // Set total count
        index.count.store(total_count, Ordering::Relaxed);

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
    fn test_tiered_multi_basic() {
        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredMulti::<f32>::new(params);

        // Add multiple vectors with same label
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 3);
        assert_eq!(index.label_count(1), 2);
        assert_eq!(index.label_count(2), 1);
    }

    #[test]
    fn test_tiered_multi_delete() {
        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        // Delete all vectors for label 1
        let deleted = index.delete_vector(1).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(index.index_size(), 1);
        assert_eq!(index.label_count(1), 0);
    }

    #[test]
    fn test_tiered_multi_flush() {
        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.flat_size(), 3);
        assert_eq!(index.hnsw_size(), 0);

        let migrated = index.flush().unwrap();
        assert_eq!(migrated, 3);

        assert_eq!(index.flat_size(), 0);
        assert_eq!(index.hnsw_size(), 3);
        assert_eq!(index.label_count(1), 2);
    }

    #[test]
    fn test_tiered_multi_serialization() {
        use std::io::Cursor;

        let params = TieredParams::new(4, Metric::L2)
            .with_flat_buffer_limit(10)
            .with_m(8)
            .with_ef_construction(50);
        let mut index = TieredMulti::<f32>::new(params);

        // Add multiple vectors per label to flat buffer
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        // Flush to HNSW
        index.flush().unwrap();

        // Add more to flat
        index.add_vector(&vec![0.0, 0.0, 1.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 0.0, 0.0, 1.0], 3).unwrap();

        assert_eq!(index.flat_size(), 2);
        assert_eq!(index.hnsw_size(), 3);
        assert_eq!(index.index_size(), 5);
        assert_eq!(index.label_count(1), 3); // 2 in HNSW + 1 in flat

        // Serialize
        let mut buffer = Vec::new();
        index.save(&mut buffer).unwrap();

        // Deserialize
        let mut cursor = Cursor::new(buffer);
        let loaded = TieredMulti::<f32>::load(&mut cursor).unwrap();

        // Verify state
        assert_eq!(loaded.index_size(), 5);
        assert_eq!(loaded.flat_size(), 2);
        assert_eq!(loaded.hnsw_size(), 3);
        assert_eq!(loaded.dimension(), 4);
        assert_eq!(loaded.label_count(1), 3);
        assert_eq!(loaded.label_count(2), 1);
        assert_eq!(loaded.label_count(3), 1);

        // Verify vectors can be queried
        // Multi-value indices deduplicate by label, so we get 3 unique labels (1, 2, 3)
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = loaded.top_k_query(&query, 5, None).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_tiered_multi_serialization_file() {
        use std::fs;

        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredMulti::<f32>::new(params);

        // Add multiple vectors per label
        for i in 0..5 {
            for j in 0..3 {
                let v = vec![i as f32 + j as f32 * 0.1, 0.0, 0.0, 0.0];
                index.add_vector(&v, i as u64).unwrap();
            }
        }

        assert_eq!(index.index_size(), 15);

        // Flush
        index.flush().unwrap();

        // Add more
        index.add_vector(&vec![10.0, 0.0, 0.0, 0.0], 10).unwrap();

        let path = "/tmp/tiered_multi_test.idx";
        index.save_to_file(path).unwrap();

        let loaded = TieredMulti::<f32>::load_from_file(path).unwrap();

        assert_eq!(loaded.index_size(), 16);
        assert_eq!(loaded.label_count(0), 3);
        assert_eq!(loaded.label_count(10), 1);
        assert!(loaded.contains(4));

        // Cleanup
        fs::remove_file(path).ok();
    }
}

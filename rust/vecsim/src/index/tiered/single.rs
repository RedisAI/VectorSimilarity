//! Single-value tiered index implementation.
//!
//! This index stores one vector per label, combining a BruteForce frontend
//! (for fast writes) with an HNSW backend (for efficient queries).

use super::{merge_range, merge_top_k, TieredParams, WriteMode};
use crate::index::brute_force::{BruteForceParams, BruteForceSingle};
use crate::index::hnsw::HnswSingle;
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply};
use crate::types::{LabelType, VectorElement};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Statistics about a tiered index.
#[derive(Debug, Clone)]
pub struct TieredStats {
    /// Number of vectors in flat buffer.
    pub flat_size: usize,
    /// Number of vectors in HNSW backend.
    pub hnsw_size: usize,
    /// Total vector count.
    pub total_size: usize,
    /// Current write mode.
    pub write_mode: WriteMode,
    /// Flat buffer fragmentation.
    pub flat_fragmentation: f64,
    /// HNSW fragmentation.
    pub hnsw_fragmentation: f64,
    /// Approximate memory usage in bytes.
    pub memory_bytes: usize,
}

/// Single-value tiered index combining BruteForce frontend with HNSW backend.
///
/// The tiered architecture provides:
/// - **Fast writes**: New vectors go to the flat BruteForce buffer
/// - **Efficient queries**: Results are merged from both tiers
/// - **Flexible migration**: Use `flush()` to migrate vectors to HNSW
///
/// # Write Behavior
///
/// In **Async** mode (default):
/// - New vectors are added to the flat buffer
/// - When buffer reaches `flat_buffer_limit`, mode switches to InPlace
/// - Call `flush()` to migrate all vectors to HNSW and reset buffer
///
/// In **InPlace** mode:
/// - Vectors are added directly to HNSW
/// - Use this when you don't need the write buffer
pub struct TieredSingle<T: VectorElement> {
    /// Flat BruteForce buffer (frontend).
    pub(crate) flat: RwLock<BruteForceSingle<T>>,
    /// HNSW index (backend).
    pub(crate) hnsw: RwLock<HnswSingle<T>>,
    /// Labels currently in flat buffer.
    flat_labels: RwLock<HashSet<LabelType>>,
    /// Labels currently in HNSW.
    hnsw_labels: RwLock<HashSet<LabelType>>,
    /// Configuration parameters.
    params: TieredParams,
    /// Total vector count.
    count: AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
}

impl<T: VectorElement> TieredSingle<T> {
    /// Create a new single-value tiered index.
    pub fn new(params: TieredParams) -> Self {
        let bf_params = BruteForceParams::new(params.dim, params.metric)
            .with_capacity(params.flat_buffer_limit.min(params.initial_capacity));
        let flat = BruteForceSingle::new(bf_params);
        let hnsw = HnswSingle::new(params.hnsw_params.clone());

        Self {
            flat: RwLock::new(flat),
            hnsw: RwLock::new(hnsw),
            flat_labels: RwLock::new(HashSet::new()),
            hnsw_labels: RwLock::new(HashSet::new()),
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
        // In Async mode, switch to InPlace if flat buffer is full
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

    /// Get detailed statistics about the index.
    pub fn stats(&self) -> TieredStats {
        let flat = self.flat.read();
        let hnsw = self.hnsw.read();

        TieredStats {
            flat_size: flat.index_size(),
            hnsw_size: hnsw.index_size(),
            total_size: self.count.load(Ordering::Relaxed),
            write_mode: self.write_mode(),
            flat_fragmentation: flat.fragmentation(),
            hnsw_fragmentation: hnsw.fragmentation(),
            memory_bytes: flat.memory_usage() + hnsw.memory_usage(),
        }
    }

    /// Get the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.flat.read().memory_usage() + self.hnsw.read().memory_usage()
    }

    /// Check if a label exists in the flat buffer.
    pub fn is_in_flat(&self, label: LabelType) -> bool {
        self.flat_labels.read().contains(&label)
    }

    /// Check if a label exists in the HNSW backend.
    pub fn is_in_hnsw(&self, label: LabelType) -> bool {
        self.hnsw_labels.read().contains(&label)
    }

    /// Flush all vectors from flat buffer to HNSW.
    ///
    /// This migrates all vectors from the flat buffer to the HNSW backend,
    /// clearing the flat buffer afterward.
    ///
    /// # Returns
    /// The number of vectors migrated.
    pub fn flush(&mut self) -> Result<usize, IndexError> {
        let flat_labels: Vec<LabelType> = self.flat_labels.read().iter().copied().collect();

        if flat_labels.is_empty() {
            return Ok(0);
        }

        let mut migrated = 0;

        // Collect vectors from flat buffer
        let vectors: Vec<(LabelType, Vec<T>)> = {
            let flat = self.flat.read();
            flat_labels
                .iter()
                .filter_map(|&label| flat.get_vector(label).map(|v| (label, v)))
                .collect()
        };

        // Add to HNSW
        {
            let mut hnsw = self.hnsw.write();
            let mut hnsw_labels = self.hnsw_labels.write();

            for (label, vector) in &vectors {
                match hnsw.add_vector(vector, *label) {
                    Ok(_) => {
                        hnsw_labels.insert(*label);
                        migrated += 1;
                    }
                    Err(e) => {
                        // Log error but continue
                        eprintln!("Failed to migrate label {label} to HNSW: {e:?}");
                    }
                }
            }
        }

        // Clear flat buffer
        {
            let mut flat = self.flat.write();
            let mut flat_labels = self.flat_labels.write();

            flat.clear();
            flat_labels.clear();
        }

        Ok(migrated)
    }

    /// Compact both tiers to reclaim space from deleted vectors.
    ///
    /// # Arguments
    /// * `shrink` - If true, also release unused memory
    ///
    /// # Returns
    /// Approximate bytes reclaimed.
    pub fn compact(&mut self, shrink: bool) -> usize {
        let flat_reclaimed = self.flat.write().compact(shrink);
        let hnsw_reclaimed = self.hnsw.write().compact(shrink);
        flat_reclaimed + hnsw_reclaimed
    }

    /// Get the fragmentation ratio (0.0 = none, 1.0 = all deleted).
    pub fn fragmentation(&self) -> f64 {
        let flat = self.flat.read();
        let hnsw = self.hnsw.read();
        let flat_frag = flat.fragmentation();
        let hnsw_frag = hnsw.fragmentation();

        // Weighted average by size
        let flat_size = flat.index_size() as f64;
        let hnsw_size = hnsw.index_size() as f64;
        let total = flat_size + hnsw_size;

        if total == 0.0 {
            0.0
        } else {
            (flat_frag * flat_size + hnsw_frag * hnsw_size) / total
        }
    }

    /// Get a copy of the vector stored for a given label.
    pub fn get_vector(&self, label: LabelType) -> Option<Vec<T>> {
        // Check flat first
        if self.flat_labels.read().contains(&label) {
            return self.flat.read().get_vector(label);
        }
        // Then check HNSW
        if self.hnsw_labels.read().contains(&label) {
            return self.hnsw.read().get_vector(label);
        }
        None
    }

    /// Clear all vectors from both tiers.
    pub fn clear(&mut self) {
        self.flat.write().clear();
        self.hnsw.write().clear();
        self.flat_labels.write().clear();
        self.hnsw_labels.write().clear();
        self.count.store(0, Ordering::Relaxed);
    }
}

impl<T: VectorElement> VecSimIndex for TieredSingle<T> {
    type DataType = T;
    type DistType = T::DistanceType;

    fn add_vector(&mut self, vector: &[T], label: LabelType) -> Result<usize, IndexError> {
        // Validate dimension
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

        let in_flat = self.flat_labels.read().contains(&label);
        let in_hnsw = self.hnsw_labels.read().contains(&label);

        // Update existing vector
        if in_flat {
            // Update in flat buffer (replace)
            let mut flat = self.flat.write();
            flat.add_vector(vector, label)?;
            return Ok(0); // No new vector
        }

        if in_hnsw {
            // For single-value: update means replace
            // Delete from HNSW and add to flat (or direct to HNSW based on mode)
            {
                let mut hnsw = self.hnsw.write();
                let mut hnsw_labels = self.hnsw_labels.write();
                hnsw.delete_vector(label)?;
                hnsw_labels.remove(&label);
            }
            self.count.fetch_sub(1, Ordering::Relaxed);

            // Now add new vector (fall through to new vector logic)
        }

        // Add new vector based on write mode
        match self.write_mode() {
            WriteMode::Async => {
                let mut flat = self.flat.write();
                let mut flat_labels = self.flat_labels.write();
                flat.add_vector(vector, label)?;
                flat_labels.insert(label);
            }
            WriteMode::InPlace => {
                let mut hnsw = self.hnsw.write();
                let mut hnsw_labels = self.hnsw_labels.write();
                hnsw.add_vector(vector, label)?;
                hnsw_labels.insert(label);
            }
        }

        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let in_flat = self.flat_labels.read().contains(&label);
        let in_hnsw = self.hnsw_labels.read().contains(&label);

        if !in_flat && !in_hnsw {
            return Err(IndexError::LabelNotFound(label));
        }

        let mut deleted = 0;

        if in_flat {
            let mut flat = self.flat.write();
            let mut flat_labels = self.flat_labels.write();
            if flat.delete_vector(label).is_ok() {
                flat_labels.remove(&label);
                deleted += 1;
            }
        }

        if in_hnsw {
            let mut hnsw = self.hnsw.write();
            let mut hnsw_labels = self.hnsw_labels.write();
            if hnsw.delete_vector(label).is_ok() {
                hnsw_labels.remove(&label);
                deleted += 1;
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

        // Lock ordering: flat first, then HNSW
        let flat = self.flat.read();
        let hnsw = self.hnsw.read();

        // Query both tiers
        let flat_results = flat.top_k_query(query, k, params)?;
        let hnsw_results = hnsw.top_k_query(query, k, params)?;

        // Merge results
        Ok(merge_top_k(flat_results, hnsw_results, k))
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

        // Lock ordering: flat first, then HNSW
        let flat = self.flat.read();
        let hnsw = self.hnsw.read();

        // Query both tiers
        let flat_results = flat.range_query(query, radius, params)?;
        let hnsw_results = hnsw.range_query(query, radius, params)?;

        // Merge results
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

        Ok(Box::new(super::batch_iterator::TieredBatchIterator::new(
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
            index_type: "TieredSingle",
            memory_bytes: self.memory_usage(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.flat_labels.read().contains(&label) || self.hnsw_labels.read().contains(&label)
    }

    fn label_count(&self, label: LabelType) -> usize {
        if self.contains(label) {
            1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_tiered_single_basic() {
        let params = TieredParams::new(4, Metric::L2).with_flat_buffer_limit(10);
        let mut index = TieredSingle::<f32>::new(params);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];

        index.add_vector(&v1, 1).unwrap();
        index.add_vector(&v2, 2).unwrap();
        index.add_vector(&v3, 3).unwrap();

        assert_eq!(index.index_size(), 3);
        assert_eq!(index.flat_size(), 3);
        assert_eq!(index.hnsw_size(), 0);

        // Query should find all vectors
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 3, None).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results.results[0].label, 1); // Closest
    }

    #[test]
    fn test_tiered_single_query_both_tiers() {
        let params = TieredParams::new(4, Metric::L2).with_flat_buffer_limit(10);
        let mut index = TieredSingle::<f32>::new(params);

        // Add to flat
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Flush to HNSW
        let migrated = index.flush().unwrap();
        assert_eq!(migrated, 1);

        // Add more to flat
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.flat_size(), 1);
        assert_eq!(index.hnsw_size(), 1);
        assert_eq!(index.index_size(), 2);

        // Query should find both
        let query = vec![0.5, 0.5, 0.0, 0.0];
        let results = index.top_k_query(&query, 2, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_tiered_single_delete() {
        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredSingle::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 2);

        // Delete from flat
        index.delete_vector(1).unwrap();
        assert_eq!(index.index_size(), 1);
        assert!(!index.contains(1));
        assert!(index.contains(2));
    }

    #[test]
    fn test_tiered_single_flush() {
        let params = TieredParams::new(4, Metric::L2).with_flat_buffer_limit(100);
        let mut index = TieredSingle::<f32>::new(params);

        // Add several vectors
        for i in 0..10 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, i as u64).unwrap();
        }

        assert_eq!(index.flat_size(), 10);
        assert_eq!(index.hnsw_size(), 0);

        // Flush
        let migrated = index.flush().unwrap();
        assert_eq!(migrated, 10);

        assert_eq!(index.flat_size(), 0);
        assert_eq!(index.hnsw_size(), 10);
        assert_eq!(index.index_size(), 10);

        // Query should still work
        let results = index.top_k_query(&vec![5.0, 0.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results.results[0].label, 5);
    }

    #[test]
    fn test_tiered_single_in_place_mode() {
        let params = TieredParams::new(4, Metric::L2)
            .with_flat_buffer_limit(2)
            .with_write_mode(WriteMode::Async);
        let mut index = TieredSingle::<f32>::new(params);

        // Fill flat buffer
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.write_mode(), WriteMode::InPlace);

        // Next add should go directly to HNSW
        index.add_vector(&vec![0.0, 0.0, 1.0, 0.0], 3).unwrap();

        assert_eq!(index.flat_size(), 2);
        assert_eq!(index.hnsw_size(), 1);
        assert_eq!(index.index_size(), 3);
    }

    #[test]
    fn test_tiered_single_compact() {
        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredSingle::<f32>::new(params);

        // Add and delete
        for i in 0..10 {
            index.add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64).unwrap();
        }

        for i in (0..10).step_by(2) {
            index.delete_vector(i as u64).unwrap();
        }

        assert!(index.fragmentation() > 0.0);

        // Compact
        index.compact(true);

        assert!((index.fragmentation() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_tiered_single_replace() {
        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredSingle::<f32>::new(params);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];

        index.add_vector(&v1, 1).unwrap();
        assert_eq!(index.index_size(), 1);

        // Replace with new vector
        index.add_vector(&v2, 1).unwrap();
        assert_eq!(index.index_size(), 1);

        // Should return the new vector
        let query = vec![0.0, 1.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 1, None).unwrap();
        assert!((results.results[0].distance as f64) < 0.001);
    }
}

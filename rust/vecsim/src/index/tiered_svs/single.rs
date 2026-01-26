//! Single-value tiered SVS index implementation.
//!
//! This index stores one vector per label, combining a BruteForce frontend
//! (for fast writes) with an SVS (Vamana) backend (for efficient queries).

use super::{merge_range, merge_top_k, TieredSvsParams, SvsWriteMode};
use crate::index::brute_force::{BruteForceParams, BruteForceSingle};
use crate::index::svs::SvsSingle;
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply};
use crate::types::{DistanceType, LabelType, VectorElement};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Statistics about a tiered SVS index.
#[derive(Debug, Clone)]
pub struct TieredSvsStats {
    /// Number of vectors in flat buffer.
    pub flat_size: usize,
    /// Number of vectors in SVS backend.
    pub svs_size: usize,
    /// Total vector count.
    pub total_size: usize,
    /// Current write mode.
    pub write_mode: SvsWriteMode,
    /// Flat buffer fragmentation.
    pub flat_fragmentation: f64,
    /// SVS fragmentation.
    pub svs_fragmentation: f64,
    /// Approximate memory usage in bytes.
    pub memory_bytes: usize,
}

/// Single-value tiered SVS index combining BruteForce frontend with SVS backend.
///
/// The tiered architecture provides:
/// - **Fast writes**: New vectors go to the flat BruteForce buffer
/// - **Efficient queries**: Results are merged from both tiers
/// - **Flexible migration**: Use `flush()` to migrate vectors to SVS
///
/// # Write Behavior
///
/// In **Async** mode (default):
/// - New vectors are added to the flat buffer
/// - When buffer reaches `flat_buffer_limit`, mode switches to InPlace
/// - Call `flush()` to migrate all vectors to SVS and reset buffer
///
/// In **InPlace** mode:
/// - Vectors are added directly to SVS
/// - Use this when you don't need the write buffer
pub struct TieredSvsSingle<T: VectorElement> {
    /// Flat BruteForce buffer (frontend).
    pub(crate) flat: RwLock<BruteForceSingle<T>>,
    /// SVS index (backend).
    pub(crate) svs: RwLock<SvsSingle<T>>,
    /// Labels currently in flat buffer.
    flat_labels: RwLock<HashSet<LabelType>>,
    /// Labels currently in SVS.
    svs_labels: RwLock<HashSet<LabelType>>,
    /// Configuration parameters.
    params: TieredSvsParams,
    /// Total vector count.
    count: AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
}

impl<T: VectorElement> TieredSvsSingle<T> {
    /// Create a new single-value tiered SVS index.
    pub fn new(params: TieredSvsParams) -> Self {
        let bf_params = BruteForceParams::new(params.dim, params.metric)
            .with_capacity(params.flat_buffer_limit.min(params.initial_capacity));
        let flat = BruteForceSingle::new(bf_params);
        let svs = SvsSingle::new(params.svs_params.clone());

        Self {
            flat: RwLock::new(flat),
            svs: RwLock::new(svs),
            flat_labels: RwLock::new(HashSet::new()),
            svs_labels: RwLock::new(HashSet::new()),
            params,
            count: AtomicUsize::new(0),
            capacity: None,
        }
    }

    /// Create with a maximum capacity.
    pub fn with_capacity(params: TieredSvsParams, max_capacity: usize) -> Self {
        let mut index = Self::new(params);
        index.capacity = Some(max_capacity);
        index
    }

    /// Get the current write mode.
    pub fn write_mode(&self) -> SvsWriteMode {
        if self.params.write_mode == SvsWriteMode::InPlace {
            return SvsWriteMode::InPlace;
        }
        // In Async mode, switch to InPlace if flat buffer is full
        if self.flat_size() >= self.params.flat_buffer_limit {
            SvsWriteMode::InPlace
        } else {
            SvsWriteMode::Async
        }
    }

    /// Get the number of vectors in the flat buffer.
    pub fn flat_size(&self) -> usize {
        self.flat.read().index_size()
    }

    /// Get the number of vectors in the SVS backend.
    pub fn svs_size(&self) -> usize {
        self.svs.read().index_size()
    }

    /// Get detailed statistics about the index.
    pub fn stats(&self) -> TieredSvsStats {
        let flat = self.flat.read();
        let svs = self.svs.read();

        TieredSvsStats {
            flat_size: flat.index_size(),
            svs_size: svs.index_size(),
            total_size: self.count.load(Ordering::Relaxed),
            write_mode: self.write_mode(),
            flat_fragmentation: flat.fragmentation(),
            svs_fragmentation: svs.fragmentation(),
            memory_bytes: flat.memory_usage() + svs.memory_usage(),
        }
    }

    /// Get the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.flat.read().memory_usage() + self.svs.read().memory_usage()
    }

    /// Check if a label exists in the flat buffer.
    pub fn is_in_flat(&self, label: LabelType) -> bool {
        self.flat_labels.read().contains(&label)
    }

    /// Check if a label exists in the SVS backend.
    pub fn is_in_svs(&self, label: LabelType) -> bool {
        self.svs_labels.read().contains(&label)
    }

    /// Flush all vectors from flat buffer to SVS.
    ///
    /// This migrates all vectors from the flat buffer to the SVS backend,
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

        // Add to SVS
        {
            let mut svs = self.svs.write();
            let mut svs_labels = self.svs_labels.write();

            for (label, vector) in &vectors {
                match svs.add_vector(vector, *label) {
                    Ok(_) => {
                        svs_labels.insert(*label);
                        migrated += 1;
                    }
                    Err(e) => {
                        // Log error but continue
                        eprintln!("Failed to migrate label {label} to SVS: {e:?}");
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
        let svs_reclaimed = self.svs.write().compact(shrink);
        flat_reclaimed + svs_reclaimed
    }

    /// Get the fragmentation ratio (0.0 = none, 1.0 = all deleted).
    pub fn fragmentation(&self) -> f64 {
        let flat = self.flat.read();
        let svs = self.svs.read();
        let flat_frag = flat.fragmentation();
        let svs_frag = svs.fragmentation();

        // Weighted average by size
        let flat_size = flat.index_size() as f64;
        let svs_size = svs.index_size() as f64;
        let total = flat_size + svs_size;

        if total == 0.0 {
            0.0
        } else {
            (flat_frag * flat_size + svs_frag * svs_size) / total
        }
    }

    /// Get a copy of the vector stored for a given label.
    pub fn get_vector(&self, label: LabelType) -> Option<Vec<T>> {
        // Check flat first
        if self.flat_labels.read().contains(&label) {
            return self.flat.read().get_vector(label);
        }
        // Then check SVS
        if self.svs_labels.read().contains(&label) {
            return self.svs.read().get_vector(label);
        }
        None
    }

    /// Clear all vectors from both tiers.
    pub fn clear(&mut self) {
        self.flat.write().clear();
        self.svs.write().clear();
        self.flat_labels.write().clear();
        self.svs_labels.write().clear();
        self.count.store(0, Ordering::Relaxed);
    }
}

impl<T: VectorElement> VecSimIndex for TieredSvsSingle<T> {
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
        let in_svs = self.svs_labels.read().contains(&label);

        // Update existing vector
        if in_flat {
            // Update in flat buffer (replace)
            let mut flat = self.flat.write();
            flat.add_vector(vector, label)?;
            return Ok(0); // No new vector
        }

        // Track if this is a replacement (to return 0 instead of 1)
        let is_replacement = in_svs;

        if in_svs {
            // For single-value: update means replace
            // Delete from SVS and add to flat (or direct to SVS based on mode)
            {
                let mut svs = self.svs.write();
                let mut svs_labels = self.svs_labels.write();
                svs.delete_vector(label)?;
                svs_labels.remove(&label);
            }
            self.count.fetch_sub(1, Ordering::Relaxed);

            // Now add new vector (fall through to new vector logic)
        }

        // Add new vector based on write mode
        match self.write_mode() {
            SvsWriteMode::Async => {
                let mut flat = self.flat.write();
                let mut flat_labels = self.flat_labels.write();
                flat.add_vector(vector, label)?;
                flat_labels.insert(label);
            }
            SvsWriteMode::InPlace => {
                let mut svs = self.svs.write();
                let mut svs_labels = self.svs_labels.write();
                svs.add_vector(vector, label)?;
                svs_labels.insert(label);
            }
        }

        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(if is_replacement { 0 } else { 1 })
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let in_flat = self.flat_labels.read().contains(&label);
        let in_svs = self.svs_labels.read().contains(&label);

        if !in_flat && !in_svs {
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

        if in_svs {
            let mut svs = self.svs.write();
            let mut svs_labels = self.svs_labels.write();
            if svs.delete_vector(label).is_ok() {
                svs_labels.remove(&label);
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

        // Lock ordering: flat first, then SVS
        let flat = self.flat.read();
        let svs = self.svs.read();

        // Query both tiers
        let flat_results = flat.top_k_query(query, k, params)?;
        let svs_results = svs.top_k_query(query, k, params)?;

        // Merge results
        Ok(merge_top_k(flat_results, svs_results, k))
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

        // Lock ordering: flat first, then SVS
        let flat = self.flat.read();
        let svs = self.svs.read();

        // Query both tiers
        let flat_results = flat.range_query(query, radius, params)?;
        let svs_results = svs.range_query(query, radius, params)?;

        // Merge results
        Ok(merge_range(flat_results, svs_results))
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

        // Compute results immediately to preserve filter from params
        let mut results = Vec::new();

        // Get results from flat buffer
        let flat = self.flat.read();
        if let Ok(mut iter) = flat.batch_iterator(query, params) {
            while let Some(batch) = iter.next_batch(1000) {
                results.extend(batch);
            }
        }
        drop(flat);

        // Get results from SVS
        let svs = self.svs.read();
        if let Ok(mut iter) = svs.batch_iterator(query, params) {
            while let Some(batch) = iter.next_batch(1000) {
                results.extend(batch);
            }
        }
        drop(svs);

        // Sort by distance
        results.sort_by(|a, b| {
            a.2.to_f64()
                .partial_cmp(&b.2.to_f64())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(Box::new(super::batch_iterator::TieredSvsBatchIterator::<T>::new(results)))
    }

    fn info(&self) -> IndexInfo {
        IndexInfo {
            size: self.index_size(),
            capacity: self.capacity,
            dimension: self.params.dim,
            index_type: "TieredSvsSingle",
            memory_bytes: self.memory_usage(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.flat_labels.read().contains(&label) || self.svs_labels.read().contains(&label)
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
    fn test_tiered_svs_single_basic() {
        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(10);
        let mut index = TieredSvsSingle::<f32>::new(params);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];

        index.add_vector(&v1, 1).unwrap();
        index.add_vector(&v2, 2).unwrap();
        index.add_vector(&v3, 3).unwrap();

        assert_eq!(index.index_size(), 3);
        assert_eq!(index.flat_size(), 3);
        assert_eq!(index.svs_size(), 0);

        // Query should find all vectors
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 3, None).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results.results[0].label, 1); // Closest
    }

    #[test]
    fn test_tiered_svs_single_query_both_tiers() {
        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(10);
        let mut index = TieredSvsSingle::<f32>::new(params);

        // Add to flat
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Flush to SVS
        let migrated = index.flush().unwrap();
        assert_eq!(migrated, 1);

        // Add more to flat
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.flat_size(), 1);
        assert_eq!(index.svs_size(), 1);
        assert_eq!(index.index_size(), 2);

        // Query should find both
        let query = vec![0.5, 0.5, 0.0, 0.0];
        let results = index.top_k_query(&query, 2, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_tiered_svs_single_delete() {
        let params = TieredSvsParams::new(4, Metric::L2);
        let mut index = TieredSvsSingle::<f32>::new(params);

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
    fn test_tiered_svs_single_flush() {
        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(100);
        let mut index = TieredSvsSingle::<f32>::new(params);

        // Add several vectors
        for i in 0..10 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, i as u64).unwrap();
        }

        assert_eq!(index.flat_size(), 10);
        assert_eq!(index.svs_size(), 0);

        // Flush
        let migrated = index.flush().unwrap();
        assert_eq!(migrated, 10);

        assert_eq!(index.flat_size(), 0);
        assert_eq!(index.svs_size(), 10);
        assert_eq!(index.index_size(), 10);

        // Query should still work
        let results = index.top_k_query(&vec![5.0, 0.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results.results[0].label, 5);
    }

    #[test]
    fn test_tiered_svs_single_in_place_mode() {
        let params = TieredSvsParams::new(4, Metric::L2)
            .with_flat_buffer_limit(2)
            .with_write_mode(SvsWriteMode::Async);
        let mut index = TieredSvsSingle::<f32>::new(params);

        // Fill flat buffer
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.write_mode(), SvsWriteMode::InPlace);

        // Next add should go directly to SVS
        index.add_vector(&vec![0.0, 0.0, 1.0, 0.0], 3).unwrap();

        assert_eq!(index.flat_size(), 2);
        assert_eq!(index.svs_size(), 1);
        assert_eq!(index.index_size(), 3);
    }

    #[test]
    fn test_tiered_svs_single_compact() {
        let params = TieredSvsParams::new(4, Metric::L2);
        let mut index = TieredSvsSingle::<f32>::new(params);

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
    fn test_tiered_svs_single_replace() {
        let params = TieredSvsParams::new(4, Metric::L2);
        let mut index = TieredSvsSingle::<f32>::new(params);

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

    #[test]
    fn test_tiered_svs_single_batch_iterator_with_filter() {
        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(5);
        let mut index = TieredSvsSingle::<f32>::new(params);

        // Add vectors to flat buffer (labels 1-5)
        for i in 1..=5u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, i).unwrap();
        }

        // Flush to SVS
        index.flush().unwrap();

        // Add more to flat buffer (labels 6-10)
        for i in 6..=10u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, i).unwrap();
        }

        assert_eq!(index.flat_size(), 5); // Labels 6-10 in flat
        assert_eq!(index.svs_size(), 5);  // Labels 1-5 in SVS

        // Create a filter that only allows labels divisible by 3
        let query_params = QueryParams::new().with_filter(|label| label % 3 == 0);

        let query = vec![5.0, 0.0, 0.0, 0.0];

        // Test batch iterator with filter
        let mut iter = index.batch_iterator(&query, Some(&query_params)).unwrap();
        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(100) {
            all_results.extend(batch);
        }

        // Should have labels 3, 6, 9 (3 results)
        assert_eq!(all_results.len(), 3);
        for (_, label, _) in &all_results {
            assert_eq!(label % 3, 0, "Expected only labels divisible by 3, got {}", label);
        }

        let labels: Vec<_> = all_results.iter().map(|(_, l, _)| *l).collect();
        assert!(labels.contains(&3));
        assert!(labels.contains(&6));
        assert!(labels.contains(&9));
    }

    #[test]
    fn test_tiered_svs_single_batch_iterator_no_filter() {
        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(3);
        let mut index = TieredSvsSingle::<f32>::new(params);

        // Add to flat buffer
        for i in 1..=3u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, i).unwrap();
        }

        // Flush to SVS
        index.flush().unwrap();

        // Add more to flat buffer
        for i in 4..=6u64 {
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

        // Should have all 6 vectors from both tiers
        assert_eq!(all_results.len(), 6);
    }
}

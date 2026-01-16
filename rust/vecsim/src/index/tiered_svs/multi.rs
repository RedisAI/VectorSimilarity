//! Multi-value tiered SVS index implementation.
//!
//! This index allows multiple vectors per label, combining a BruteForce frontend
//! (for fast writes) with an SVS (Vamana) backend (for efficient queries).

use super::{merge_range, merge_top_k, TieredSvsParams, SvsWriteMode};
use crate::index::brute_force::{BruteForceMulti, BruteForceParams};
use crate::index::svs::SvsMulti;
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply};
use crate::types::{DistanceType, LabelType, VectorElement};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Statistics about a tiered SVS multi index.
#[derive(Debug, Clone)]
pub struct TieredSvsMultiStats {
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

/// Multi-value tiered SVS index combining BruteForce frontend with SVS backend.
///
/// Unlike TieredSvsSingle, this allows multiple vectors per label.
/// When deleting, all vectors with that label are removed from both tiers.
pub struct TieredSvsMulti<T: VectorElement> {
    /// Flat BruteForce buffer (frontend).
    pub(crate) flat: RwLock<BruteForceMulti<T>>,
    /// SVS index (backend).
    pub(crate) svs: RwLock<SvsMulti<T>>,
    /// Count of vectors per label in flat buffer.
    flat_label_counts: RwLock<HashMap<LabelType, usize>>,
    /// Count of vectors per label in SVS.
    svs_label_counts: RwLock<HashMap<LabelType, usize>>,
    /// Configuration parameters.
    params: TieredSvsParams,
    /// Total vector count.
    count: AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
}

impl<T: VectorElement> TieredSvsMulti<T> {
    /// Create a new multi-value tiered SVS index.
    pub fn new(params: TieredSvsParams) -> Self {
        let bf_params = BruteForceParams::new(params.dim, params.metric)
            .with_capacity(params.flat_buffer_limit.min(params.initial_capacity));
        let flat = BruteForceMulti::new(bf_params);
        let svs = SvsMulti::new(params.svs_params.clone());

        Self {
            flat: RwLock::new(flat),
            svs: RwLock::new(svs),
            flat_label_counts: RwLock::new(HashMap::new()),
            svs_label_counts: RwLock::new(HashMap::new()),
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
    pub fn stats(&self) -> TieredSvsMultiStats {
        let flat = self.flat.read();
        let svs = self.svs.read();

        TieredSvsMultiStats {
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
        self.flat_label_counts
            .read()
            .get(&label)
            .is_some_and(|&c| c > 0)
    }

    /// Check if a label exists in the SVS backend.
    pub fn is_in_svs(&self, label: LabelType) -> bool {
        self.svs_label_counts
            .read()
            .get(&label)
            .is_some_and(|&c| c > 0)
    }

    /// Flush all vectors from flat buffer to SVS.
    ///
    /// This migrates all vectors from the flat buffer to the SVS backend,
    /// clearing the flat buffer afterward.
    ///
    /// # Returns
    /// The number of vectors migrated.
    pub fn flush(&mut self) -> Result<usize, IndexError> {
        let flat_labels: Vec<(LabelType, usize)> = self
            .flat_label_counts
            .read()
            .iter()
            .filter(|(_, &count)| count > 0)
            .map(|(&l, &c)| (l, c))
            .collect();

        if flat_labels.is_empty() {
            return Ok(0);
        }

        let mut migrated = 0;

        // Collect all vectors from flat buffer
        let vectors: Vec<(LabelType, Vec<Vec<T>>)> = {
            let flat = self.flat.read();
            flat_labels
                .iter()
                .filter_map(|(label, _)| {
                    flat.get_vectors(*label).map(|vecs| (*label, vecs))
                })
                .collect()
        };

        // Add to SVS
        {
            let mut svs = self.svs.write();
            let mut svs_label_counts = self.svs_label_counts.write();

            for (label, vecs) in &vectors {
                for vec in vecs {
                    match svs.add_vector(vec, *label) {
                        Ok(added) => {
                            if added > 0 {
                                *svs_label_counts.entry(*label).or_insert(0) += added;
                                migrated += added;
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to migrate vector for label {label} to SVS: {e:?}");
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

    /// Get all vectors stored for a given label.
    pub fn get_all_vectors(&self, label: LabelType) -> Vec<Vec<T>> {
        let mut results = Vec::new();

        // Get from flat buffer
        if self.is_in_flat(label) {
            if let Some(vecs) = self.flat.read().get_vectors(label) {
                results.extend(vecs);
            }
        }

        // Get from SVS
        if self.is_in_svs(label) {
            if let Some(vecs) = self.svs.read().get_all_vectors(label) {
                results.extend(vecs);
            }
        }

        results
    }

    /// Clear all vectors from both tiers.
    pub fn clear(&mut self) {
        self.flat.write().clear();
        self.svs.write().clear();
        self.flat_label_counts.write().clear();
        self.svs_label_counts.write().clear();
        self.count.store(0, Ordering::Relaxed);
    }
}

impl<T: VectorElement> VecSimIndex for TieredSvsMulti<T> {
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

        // Add new vector based on write mode
        let added = match self.write_mode() {
            SvsWriteMode::Async => {
                let mut flat = self.flat.write();
                let mut flat_label_counts = self.flat_label_counts.write();
                let added = flat.add_vector(vector, label)?;
                *flat_label_counts.entry(label).or_insert(0) += added;
                added
            }
            SvsWriteMode::InPlace => {
                let mut svs = self.svs.write();
                let mut svs_label_counts = self.svs_label_counts.write();
                let added = svs.add_vector(vector, label)?;
                *svs_label_counts.entry(label).or_insert(0) += added;
                added
            }
        };

        self.count.fetch_add(added, Ordering::Relaxed);
        Ok(added)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let in_flat = self.is_in_flat(label);
        let in_svs = self.is_in_svs(label);

        if !in_flat && !in_svs {
            return Err(IndexError::LabelNotFound(label));
        }

        let mut deleted = 0;

        if in_flat {
            let mut flat = self.flat.write();
            let mut flat_label_counts = self.flat_label_counts.write();
            if let Ok(count) = flat.delete_vector(label) {
                flat_label_counts.remove(&label);
                deleted += count;
            }
        }

        if in_svs {
            let mut svs = self.svs.write();
            let mut svs_label_counts = self.svs_label_counts.write();
            if let Ok(count) = svs.delete_vector(label) {
                svs_label_counts.remove(&label);
                deleted += count;
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

        Ok(Box::new(
            super::batch_iterator::TieredSvsMultiBatchIterator::<T>::new(results),
        ))
    }

    fn info(&self) -> IndexInfo {
        IndexInfo {
            size: self.index_size(),
            capacity: self.capacity,
            dimension: self.params.dim,
            index_type: "TieredSvsMulti",
            memory_bytes: self.memory_usage(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.is_in_flat(label) || self.is_in_svs(label)
    }

    fn label_count(&self, label: LabelType) -> usize {
        let flat_count = self
            .flat_label_counts
            .read()
            .get(&label)
            .copied()
            .unwrap_or(0);
        let svs_count = self
            .svs_label_counts
            .read()
            .get(&label)
            .copied()
            .unwrap_or(0);
        flat_count + svs_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_tiered_svs_multi_basic() {
        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(10);
        let mut index = TieredSvsMulti::<f32>::new(params);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];

        index.add_vector(&v1, 1).unwrap();
        index.add_vector(&v2, 1).unwrap(); // Same label
        index.add_vector(&v3, 2).unwrap();

        assert_eq!(index.index_size(), 3);
        assert_eq!(index.label_count(1), 2);
        assert_eq!(index.label_count(2), 1);
    }

    #[test]
    fn test_tiered_svs_multi_flush() {
        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(100);
        let mut index = TieredSvsMulti::<f32>::new(params);

        // Add vectors with same label
        for i in 0..5 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            index.add_vector(&v, 1).unwrap();
        }

        assert_eq!(index.flat_size(), 5);
        assert_eq!(index.svs_size(), 0);

        // Flush
        let migrated = index.flush().unwrap();
        assert_eq!(migrated, 5);

        assert_eq!(index.flat_size(), 0);
        assert_eq!(index.svs_size(), 5);
        assert_eq!(index.label_count(1), 5);
    }

    #[test]
    fn test_tiered_svs_multi_delete() {
        let params = TieredSvsParams::new(4, Metric::L2);
        let mut index = TieredSvsMulti::<f32>::new(params);

        // Add multiple vectors with same label
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 0.0, 1.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 3);

        // Delete label 1 (should remove 2 vectors)
        let deleted = index.delete_vector(1).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(index.index_size(), 1);
        assert!(!index.contains(1));
        assert!(index.contains(2));
    }

    #[test]
    fn test_tiered_svs_multi_get_all_vectors() {
        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(10);
        let mut index = TieredSvsMulti::<f32>::new(params);

        // Add to flat
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Flush to SVS
        index.flush().unwrap();

        // Add more to flat
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 1).unwrap();

        // Get all vectors for label 1 (from both tiers)
        let vectors = index.get_all_vectors(1);
        assert_eq!(vectors.len(), 2);
    }

    #[test]
    fn test_tiered_svs_multi_query_both_tiers() {
        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(10);
        let mut index = TieredSvsMulti::<f32>::new(params);

        // Add to flat
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Flush to SVS
        index.flush().unwrap();

        // Add more to flat
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.flat_size(), 1);
        assert_eq!(index.svs_size(), 1);

        // Query should find both
        let query = vec![0.5, 0.5, 0.0, 0.0];
        let results = index.top_k_query(&query, 2, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_tiered_svs_multi_batch_iterator_with_filter() {
        use crate::query::QueryParams;

        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(10);
        let mut index = TieredSvsMulti::<f32>::new(params);

        // Add 2 vectors each for labels 1-3 to flat buffer
        for i in 1..=3u64 {
            let v1 = vec![i as f32, 0.0, 0.0, 0.0];
            let v2 = vec![i as f32, 0.1, 0.0, 0.0];
            index.add_vector(&v1, i).unwrap();
            index.add_vector(&v2, i).unwrap();
        }

        // Flush to SVS
        index.flush().unwrap();

        // Add 2 vectors each for labels 4-6 to flat buffer
        for i in 4..=6u64 {
            let v1 = vec![i as f32, 0.0, 0.0, 0.0];
            let v2 = vec![i as f32, 0.1, 0.0, 0.0];
            index.add_vector(&v1, i).unwrap();
            index.add_vector(&v2, i).unwrap();
        }

        assert_eq!(index.flat_size(), 6); // Labels 4-6, 2 each
        assert_eq!(index.svs_size(), 6);  // Labels 1-3, 2 each

        // Create a filter that only allows even labels (2, 4, 6)
        let query_params = QueryParams::new().with_filter(|label| label % 2 == 0);

        let query = vec![4.0, 0.0, 0.0, 0.0];

        // Test batch iterator with filter
        let mut iter = index.batch_iterator(&query, Some(&query_params)).unwrap();
        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(100) {
            all_results.extend(batch);
        }

        // Should have 6 vectors (2 each for labels 2, 4, 6)
        assert_eq!(all_results.len(), 6);
        for (_, label, _) in &all_results {
            assert_eq!(label % 2, 0, "Expected only even labels, got {}", label);
        }
    }

    #[test]
    fn test_tiered_svs_multi_batch_iterator_no_filter() {
        let params = TieredSvsParams::new(4, Metric::L2).with_flat_buffer_limit(10);
        let mut index = TieredSvsMulti::<f32>::new(params);

        // Add 2 vectors each for labels 1-2 to flat buffer
        for i in 1..=2u64 {
            let v1 = vec![i as f32, 0.0, 0.0, 0.0];
            let v2 = vec![i as f32, 0.1, 0.0, 0.0];
            index.add_vector(&v1, i).unwrap();
            index.add_vector(&v2, i).unwrap();
        }

        // Flush to SVS
        index.flush().unwrap();

        // Add 2 vectors each for labels 3-4 to flat buffer
        for i in 3..=4u64 {
            let v1 = vec![i as f32, 0.0, 0.0, 0.0];
            let v2 = vec![i as f32, 0.1, 0.0, 0.0];
            index.add_vector(&v1, i).unwrap();
            index.add_vector(&v2, i).unwrap();
        }

        let query = vec![2.0, 0.0, 0.0, 0.0];

        // Test batch iterator without filter
        let mut iter = index.batch_iterator(&query, None).unwrap();
        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(100) {
            all_results.extend(batch);
        }

        // Should have all 8 vectors from both tiers (4 labels * 2 vectors each)
        assert_eq!(all_results.len(), 8);
    }
}

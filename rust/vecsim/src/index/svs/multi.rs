//! Multi-value SVS (Vamana) index implementation.
//!
//! This index allows multiple vectors per label.

use super::{SvsCore, SvsParams};
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply, QueryResult};
use crate::types::{DistanceType, IdType, LabelType, VectorElement, INVALID_ID};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Multi-value SVS (Vamana) index.
///
/// Each label can have multiple associated vectors.
pub struct SvsMulti<T: VectorElement> {
    /// Core SVS implementation.
    core: RwLock<SvsCore<T>>,
    /// Label to internal IDs mapping.
    label_to_ids: RwLock<HashMap<LabelType, HashSet<IdType>>>,
    /// Internal ID to label mapping.
    id_to_label: RwLock<HashMap<IdType, LabelType>>,
    /// Number of vectors.
    count: AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
    /// Construction completed flag.
    construction_done: RwLock<bool>,
}

impl<T: VectorElement> SvsMulti<T> {
    /// Create a new multi-value SVS index.
    pub fn new(params: SvsParams) -> Self {
        let initial_capacity = params.initial_capacity;
        Self {
            core: RwLock::new(SvsCore::new(params)),
            label_to_ids: RwLock::new(HashMap::with_capacity(initial_capacity)),
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

    /// Get all internal IDs for a label.
    pub fn get_ids(&self, label: LabelType) -> Option<HashSet<IdType>> {
        self.label_to_ids.read().get(&label).cloned()
    }

    /// Build the index after all vectors have been added.
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

    /// Get the number of unique labels.
    pub fn unique_labels(&self) -> usize {
        self.label_to_ids.read().len()
    }

    /// Get all vectors for a label.
    pub fn get_all_vectors(&self, label: LabelType) -> Option<Vec<Vec<T>>> {
        let ids = self.label_to_ids.read().get(&label)?.clone();
        let core = self.core.read();

        let vectors: Vec<_> = ids
            .iter()
            .filter_map(|&id| core.data.get(id).map(|v| v.to_vec()))
            .collect();

        if vectors.is_empty() {
            None
        } else {
            Some(vectors)
        }
    }

    /// Estimate memory usage.
    pub fn memory_usage(&self) -> usize {
        let core = self.core.read();
        let count = self.count.load(Ordering::Relaxed);

        let vector_size = count * core.params.dim * std::mem::size_of::<T>();
        let graph_size = count * core.params.graph_max_degree * std::mem::size_of::<IdType>();

        let label_size: usize = self
            .label_to_ids
            .read()
            .values()
            .map(|ids| std::mem::size_of::<LabelType>() + ids.len() * std::mem::size_of::<IdType>())
            .sum();

        let id_label_size = self.id_to_label.read().capacity()
            * (std::mem::size_of::<IdType>() + std::mem::size_of::<LabelType>());

        vector_size + graph_size + label_size + id_label_size
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
        0
    }

    /// Clear all vectors from the index.
    pub fn clear(&mut self) {
        let mut core = self.core.write();
        let params = core.params.clone();
        *core = SvsCore::new(params);

        self.label_to_ids.write().clear();
        self.id_to_label.write().clear();
        self.count.store(0, Ordering::Relaxed);
        *self.construction_done.write() = false;
    }
}

impl<T: VectorElement> VecSimIndex for SvsMulti<T> {
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
            if self.count.load(Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Add vector
        let id = core
            .add_vector(vector)
            .ok_or_else(|| IndexError::Internal("Failed to add vector to storage".to_string()))?;
        core.insert(id, label);

        // Update mappings
        self.label_to_ids.write().entry(label).or_default().insert(id);
        self.id_to_label.write().insert(id, label);

        self.count.fetch_add(1, Ordering::Relaxed);
        *self.construction_done.write() = false;

        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let mut core = self.core.write();
        let mut label_to_ids = self.label_to_ids.write();
        let mut id_to_label = self.id_to_label.write();

        let ids = label_to_ids
            .remove(&label)
            .ok_or(IndexError::LabelNotFound(label))?;

        let count = ids.len();
        for id in ids {
            core.mark_deleted(id);
            id_to_label.remove(&id);
        }

        self.count.fetch_sub(count, Ordering::Relaxed);
        Ok(count)
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

        // Look up labels
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

        Ok(Box::new(SvsMultiBatchIterator::<T>::new(results)))
    }

    fn info(&self) -> IndexInfo {
        let core = self.core.read();
        let count = self.count.load(Ordering::Relaxed);

        IndexInfo {
            size: count,
            capacity: self.capacity,
            dimension: core.params.dim,
            index_type: "SvsMulti",
            memory_bytes: self.memory_usage(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.label_to_ids.read().contains_key(&label)
    }

    fn label_count(&self, label: LabelType) -> usize {
        self.label_to_ids
            .read()
            .get(&label)
            .map(|ids| ids.len())
            .unwrap_or(0)
    }
}

// Allow read-only concurrent access for queries
unsafe impl<T: VectorElement> Send for SvsMulti<T> {}
unsafe impl<T: VectorElement> Sync for SvsMulti<T> {}

/// Batch iterator for SvsMulti.
pub struct SvsMultiBatchIterator<T: VectorElement> {
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    current_idx: usize,
}

impl<T: VectorElement> SvsMultiBatchIterator<T> {
    /// Create a new batch iterator with pre-computed results.
    pub fn new(results: Vec<(IdType, LabelType, T::DistanceType)>) -> Self {
        Self {
            results,
            current_idx: 0,
        }
    }
}

impl<T: VectorElement> BatchIterator for SvsMultiBatchIterator<T> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_svs_multi_basic() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        // Add multiple vectors per label
        assert_eq!(index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap(), 1);
        assert_eq!(index.add_vector(&[0.9, 0.1, 0.0, 0.0], 1).unwrap(), 1);
        assert_eq!(index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap(), 1);

        assert_eq!(index.index_size(), 3);
        assert_eq!(index.unique_labels(), 2);
        assert_eq!(index.label_count(1), 2);
        assert_eq!(index.label_count(2), 1);
    }

    #[test]
    fn test_svs_multi_delete() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

        // Delete label 1 (removes both vectors)
        assert_eq!(index.delete_vector(1).unwrap(), 2);

        assert_eq!(index.unique_labels(), 1);
        assert_eq!(index.label_count(1), 0);
        assert_eq!(index.label_count(2), 1);
    }

    #[test]
    fn test_svs_multi_get_all_vectors() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[0.9, 0.1, 0.0, 0.0], 1).unwrap();

        let vectors = index.get_all_vectors(1).unwrap();
        assert_eq!(vectors.len(), 2);
    }

    #[test]
    fn test_svs_multi_query() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        for i in 0..50 {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            index.add_vector(&v, (i / 10) as u64).unwrap();
        }

        let query = [1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 5, None).unwrap();

        assert!(!results.results.is_empty());
    }

    #[test]
    fn test_svs_multi_batch_iterator_with_filter() {
        use crate::query::QueryParams;

        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        // Add vectors: 2 vectors each for labels 1-5
        for i in 1..=5u64 {
            let v1 = vec![i as f32, 0.0, 0.0, 0.0];
            let v2 = vec![i as f32, 0.1, 0.0, 0.0];
            index.add_vector(&v1, i).unwrap();
            index.add_vector(&v2, i).unwrap();
        }

        assert_eq!(index.index_size(), 10); // 5 labels * 2 vectors each

        // Create a filter that only allows labels 2 and 4
        let query_params = QueryParams::new().with_filter(|label| label == 2 || label == 4);

        let query = vec![3.0, 0.0, 0.0, 0.0];

        // Test batch iterator with filter
        let mut iter = index.batch_iterator(&query, Some(&query_params)).unwrap();
        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(100) {
            all_results.extend(batch);
        }

        // Should have 4 vectors (2 for label 2, 2 for label 4)
        assert_eq!(all_results.len(), 4);
        for (_, label, _) in &all_results {
            assert!(
                *label == 2 || *label == 4,
                "Expected only labels 2 or 4, got {}",
                label
            );
        }
    }

    #[test]
    fn test_svs_multi_batch_iterator_no_filter() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        // Add vectors: 2 vectors each for labels 1-3
        for i in 1..=3u64 {
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

        // Should have all 6 vectors
        assert_eq!(all_results.len(), 6);
    }

    #[test]
    fn test_svs_multi_multiple_labels() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

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
        assert_eq!(index.unique_labels(), 5);
    }

    #[test]
    fn test_svs_multi_contains() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

        assert!(index.contains(1));
        assert!(!index.contains(2));
    }

    #[test]
    fn test_svs_multi_dimension_mismatch() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        // Wrong dimension on add
        let result = index.add_vector(&[1.0, 2.0], 1);
        assert!(matches!(result, Err(IndexError::DimensionMismatch { expected: 4, got: 2 })));

        // Add a valid vector first
        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Wrong dimension on query
        let result = index.top_k_query(&[1.0, 2.0], 1, None);
        assert!(matches!(result, Err(QueryError::DimensionMismatch { expected: 4, got: 2 })));
    }

    #[test]
    fn test_svs_multi_capacity_exceeded() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::with_capacity(params, 2);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[2.0, 0.0, 0.0, 0.0], 2).unwrap();

        // Third should fail
        let result = index.add_vector(&[3.0, 0.0, 0.0, 0.0], 3);
        assert!(matches!(result, Err(IndexError::CapacityExceeded { capacity: 2 })));
    }

    #[test]
    fn test_svs_multi_delete_not_found() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

        let result = index.delete_vector(999);
        assert!(matches!(result, Err(IndexError::LabelNotFound(999))));
    }

    #[test]
    fn test_svs_multi_memory_usage() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        let initial_memory = index.memory_usage();

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[2.0, 0.0, 0.0, 0.0], 2).unwrap();

        let after_memory = index.memory_usage();
        assert!(after_memory > initial_memory);
    }

    #[test]
    fn test_svs_multi_info() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[2.0, 0.0, 0.0, 0.0], 1).unwrap();

        let info = index.info();
        assert_eq!(info.size, 2);
        assert_eq!(info.dimension, 4);
        assert_eq!(info.index_type, "SvsMulti");
        assert!(info.memory_bytes > 0);
    }

    #[test]
    fn test_svs_multi_clear() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[2.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 2);

        index.clear();

        assert_eq!(index.index_size(), 0);
        assert_eq!(index.unique_labels(), 0);
        assert!(!index.contains(1));
        assert!(!index.contains(2));
    }

    #[test]
    fn test_svs_multi_with_capacity() {
        let params = SvsParams::new(4, Metric::L2);
        let index = SvsMulti::<f32>::with_capacity(params, 100);

        assert_eq!(index.index_capacity(), Some(100));
    }

    #[test]
    fn test_svs_multi_inner_product() {
        let params = SvsParams::new(4, Metric::InnerProduct);
        let mut index = SvsMulti::<f32>::new(params);

        // For InnerProduct, higher dot product = lower "distance"
        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();

        let query = [1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 2, None).unwrap();

        // Label 1 has perfect alignment with query
        assert_eq!(results.results[0].label, 1);
    }

    #[test]
    fn test_svs_multi_cosine() {
        let params = SvsParams::new(4, Metric::Cosine);
        let mut index = SvsMulti::<f32>::new(params);

        // Cosine similarity is direction-based
        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
        index.add_vector(&[2.0, 0.0, 0.0, 0.0], 3).unwrap(); // Same direction as label 1

        let query = [1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 3, None).unwrap();

        // Labels 1 and 3 have the same direction (cosine=1), should be top results
        assert!(results.results[0].label == 1 || results.results[0].label == 3);
        assert!(results.results[1].label == 1 || results.results[1].label == 3);
    }

    #[test]
    fn test_svs_multi_metric_getter() {
        let params = SvsParams::new(4, Metric::Cosine);
        let index = SvsMulti::<f32>::new(params);

        assert_eq!(index.metric(), Metric::Cosine);
    }

    #[test]
    fn test_svs_multi_range_query() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        // Add vectors at different distances
        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[2.0, 0.0, 0.0, 0.0], 2).unwrap();
        index.add_vector(&[3.0, 0.0, 0.0, 0.0], 3).unwrap();
        index.add_vector(&[10.0, 0.0, 0.0, 0.0], 4).unwrap();

        let query = [0.0, 0.0, 0.0, 0.0];
        // L2 squared: dist to [1,0,0,0]=1, [2,0,0,0]=4, [3,0,0,0]=9, [10,0,0,0]=100
        let results = index.range_query(&query, 10.0, None).unwrap();

        assert_eq!(results.len(), 3); // labels 1, 2, 3 are within radius 10
        for r in &results.results {
            assert!(r.label != 4); // label 4 should not be included
        }
    }

    #[test]
    fn test_svs_multi_filtered_query() {
        use crate::query::QueryParams;

        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        for label in 1..=10u64 {
            index.add_vector(&[label as f32, 0.0, 0.0, 0.0], label).unwrap();
        }

        // Filter to only allow even labels
        let query_params = QueryParams::new().with_filter(|label| label % 2 == 0);
        let query = [0.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 10, Some(&query_params)).unwrap();

        assert_eq!(results.len(), 5); // Only labels 2, 4, 6, 8, 10
        for r in &results.results {
            assert!(r.label % 2 == 0);
        }
    }

    #[test]
    fn test_svs_multi_empty_query() {
        let params = SvsParams::new(4, Metric::L2);
        let index = SvsMulti::<f32>::new(params);

        // Query on empty index
        let query = [1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 10, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_svs_multi_fragmentation() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        for i in 1..=10u64 {
            index.add_vector(&[i as f32, 0.0, 0.0, 0.0], i).unwrap();
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
    fn test_svs_multi_medoid() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        // Empty index has no medoid
        assert!(index.medoid().is_none());

        // Add a vector
        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Now should have a medoid
        assert!(index.medoid().is_some());
    }

    #[test]
    fn test_svs_multi_build() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        // Add vectors
        for i in 0..20u64 {
            index.add_vector(&[i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        // Build the index (this should complete without error)
        index.build();

        // Query should work
        let query = [10.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 5, None).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_svs_multi_get_ids() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&[2.0, 0.0, 0.0, 0.0], 1).unwrap();

        let ids = index.get_ids(1).unwrap();
        assert_eq!(ids.len(), 2);

        // Non-existent label
        assert!(index.get_ids(999).is_none());
    }

    #[test]
    fn test_svs_multi_filtered_range_query() {
        use crate::query::QueryParams;

        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        for label in 1..=10u64 {
            index.add_vector(&[label as f32, 0.0, 0.0, 0.0], label).unwrap();
        }

        // Filter to only allow labels 1-5, with range 50 (covers labels 1-7 by distance)
        let query_params = QueryParams::new().with_filter(|label| label <= 5);
        let query = [0.0, 0.0, 0.0, 0.0];
        let results = index.range_query(&query, 50.0, Some(&query_params)).unwrap();

        // Should have labels 1-5 (filtered) that are within range 50
        assert_eq!(results.len(), 5);
        for r in &results.results {
            assert!(r.label <= 5);
        }
    }

    #[test]
    fn test_svs_multi_compact() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        for i in 1..=10u64 {
            index.add_vector(&[i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        // Delete some vectors
        for i in 1..=5u64 {
            index.delete_vector(i).unwrap();
        }

        // Compact returns 0 for SVS (placeholder)
        let reclaimed = index.compact(true);
        assert_eq!(reclaimed, 0);

        // Queries should still work
        let query = [8.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 3, None).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_svs_multi_query_with_ef_runtime() {
        use crate::query::QueryParams;

        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        for i in 0..50u64 {
            index.add_vector(&[i as f32, 0.0, 0.0, 0.0], i).unwrap();
        }

        let query = [25.0, 0.0, 0.0, 0.0];

        // Query with default search window
        let results1 = index.top_k_query(&query, 5, None).unwrap();
        assert!(!results1.is_empty());

        // Query with larger search window (via ef_runtime)
        let query_params = QueryParams::new().with_ef_runtime(100);
        let results2 = index.top_k_query(&query, 5, Some(&query_params)).unwrap();
        assert!(!results2.is_empty());
    }

    #[test]
    fn test_svs_multi_larger_scale() {
        let params = SvsParams::new(8, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        // Add 500 vectors, 5 per label
        for label in 0..100u64 {
            for i in 0..5 {
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

        assert_eq!(index.index_size(), 500);
        assert_eq!(index.label_count(50), 5);

        // Build the index
        index.build();

        // Query should work
        let query = vec![50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 10, None).unwrap();

        assert!(!results.is_empty());
        // First result should be label 50 (closest)
        assert_eq!(results.results[0].label, 50);
    }

    #[test]
    fn test_svs_multi_zero_k_query() {
        let params = SvsParams::new(4, Metric::L2);
        let mut index = SvsMulti::<f32>::new(params);

        index.add_vector(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

        // Query with k=0 should return empty
        let query = [1.0, 0.0, 0.0, 0.0];
        let results = index.top_k_query(&query, 0, None).unwrap();
        assert!(results.is_empty());
    }
}

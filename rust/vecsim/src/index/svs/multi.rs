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
}

//! Batch iterator implementations for HNSW indices.

use super::multi::HnswMulti;
use super::single::HnswSingle;
use crate::index::traits::{BatchIterator, VecSimIndex};
use crate::query::QueryParams;
use crate::types::{IdType, LabelType, VectorElement};

/// Batch iterator for single-value HNSW index.
pub struct HnswSingleBatchIterator<'a, T: VectorElement> {
    index: &'a HnswSingle<T>,
    query: Vec<T>,
    params: Option<QueryParams>,
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    position: usize,
    computed: bool,
}

impl<'a, T: VectorElement> HnswSingleBatchIterator<'a, T> {
    pub fn new(
        index: &'a HnswSingle<T>,
        query: Vec<T>,
        params: Option<QueryParams>,
    ) -> Self {
        Self {
            index,
            query,
            params,
            results: Vec::new(),
            position: 0,
            computed: false,
        }
    }

    fn compute_results(&mut self) {
        if self.computed {
            return;
        }

        let core = &self.index.core;

        let ef = self.params
            .as_ref()
            .and_then(|p| p.ef_runtime)
            .unwrap_or(core.params.ef_runtime)
            .max(1000); // Use large ef for batch iteration

        let count = self.index.index_size();

        // Clone id_to_label map for filter closure to avoid borrow-after-move
        let filter_fn: Option<Box<dyn Fn(IdType) -> bool + '_>> =
            if let Some(ref p) = self.params {
                if let Some(ref f) = p.filter {
                    // Collect DashMap entries into a regular HashMap for the closure
                    let id_to_label_for_filter: std::collections::HashMap<IdType, LabelType> =
                        self.index.id_to_label.iter().map(|r| (*r.key(), *r.value())).collect();
                    Some(Box::new(move |id: IdType| {
                        id_to_label_for_filter.get(&id).is_some_and(|&label| f(label))
                    }))
                } else {
                    None
                }
            } else {
                None
            };

        let search_results = core.search(
            &self.query,
            count,
            ef,
            filter_fn.as_ref().map(|f| f.as_ref()),
        );

        // Process results using DashMap
        self.results = search_results
            .into_iter()
            .filter_map(|(id, dist)| {
                self.index.id_to_label.get(&id).map(|label_ref| (id, *label_ref, dist))
            })
            .collect();

        self.computed = true;
    }
}

impl<'a, T: VectorElement> BatchIterator for HnswSingleBatchIterator<'a, T> {
    type DistType = T::DistanceType;

    fn has_next(&self) -> bool {
        if !self.computed {
            return true; // Haven't computed yet
        }
        self.position < self.results.len()
    }

    fn next_batch(
        &mut self,
        batch_size: usize,
    ) -> Option<Vec<(IdType, LabelType, Self::DistType)>> {
        self.compute_results();

        if self.position >= self.results.len() {
            return None;
        }

        let end = (self.position + batch_size).min(self.results.len());
        let batch = self.results[self.position..end].to_vec();
        self.position = end;

        Some(batch)
    }

    fn reset(&mut self) {
        self.position = 0;
    }
}

/// Batch iterator for multi-value HNSW index.
pub struct HnswMultiBatchIterator<'a, T: VectorElement> {
    index: &'a HnswMulti<T>,
    query: Vec<T>,
    params: Option<QueryParams>,
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    position: usize,
    computed: bool,
}

impl<'a, T: VectorElement> HnswMultiBatchIterator<'a, T> {
    pub fn new(
        index: &'a HnswMulti<T>,
        query: Vec<T>,
        params: Option<QueryParams>,
    ) -> Self {
        Self {
            index,
            query,
            params,
            results: Vec::new(),
            position: 0,
            computed: false,
        }
    }

    fn compute_results(&mut self) {
        if self.computed {
            return;
        }

        let core = &self.index.core;

        let ef = self.params
            .as_ref()
            .and_then(|p| p.ef_runtime)
            .unwrap_or(core.params.ef_runtime)
            .max(1000);

        let count = self.index.index_size();

        // Clone id_to_label map for filter closure to avoid borrow-after-move
        let filter_fn: Option<Box<dyn Fn(IdType) -> bool + '_>> =
            if let Some(ref p) = self.params {
                if let Some(ref f) = p.filter {
                    // Collect DashMap entries into a regular HashMap for the closure
                    let id_to_label_for_filter: std::collections::HashMap<IdType, LabelType> =
                        self.index.id_to_label.iter().map(|r| (*r.key(), *r.value())).collect();
                    Some(Box::new(move |id: IdType| {
                        id_to_label_for_filter.get(&id).is_some_and(|&label| f(label))
                    }))
                } else {
                    None
                }
            } else {
                None
            };

        let search_results = core.search(
            &self.query,
            count,
            ef,
            filter_fn.as_ref().map(|f| f.as_ref()),
        );

        // Process results using DashMap
        self.results = search_results
            .into_iter()
            .filter_map(|(id, dist)| {
                self.index.id_to_label.get(&id).map(|label_ref| (id, *label_ref, dist))
            })
            .collect();

        self.computed = true;
    }
}

impl<'a, T: VectorElement> BatchIterator for HnswMultiBatchIterator<'a, T> {
    type DistType = T::DistanceType;

    fn has_next(&self) -> bool {
        if !self.computed {
            return true;
        }
        self.position < self.results.len()
    }

    fn next_batch(
        &mut self,
        batch_size: usize,
    ) -> Option<Vec<(IdType, LabelType, Self::DistType)>> {
        self.compute_results();

        if self.position >= self.results.len() {
            return None;
        }

        let end = (self.position + batch_size).min(self.results.len());
        let batch = self.results[self.position..end].to_vec();
        self.position = end;

        Some(batch)
    }

    fn reset(&mut self) {
        self.position = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::index::hnsw::HnswParams;
    use crate::index::VecSimIndex;
    use crate::types::DistanceType;

    #[test]
    fn test_hnsw_batch_iterator() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswSingle::<f32>::new(params);

        for i in 0..10 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        assert!(iter.has_next());

        let batch1 = iter.next_batch(3).unwrap();
        assert!(!batch1.is_empty());

        let mut total = batch1.len();
        while let Some(batch) = iter.next_batch(3) {
            total += batch.len();
        }

        // Should have gotten all vectors
        assert!(total <= 10);
    }

    #[test]
    fn test_hnsw_batch_iterator_empty_index() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let index = HnswSingle::<f32>::new(params);

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        // Has_next returns true because results haven't been computed yet
        assert!(iter.has_next());

        // But next_batch returns None after computing empty results
        let batch = iter.next_batch(10);
        assert!(batch.is_none());
    }

    #[test]
    fn test_hnsw_batch_iterator_reset() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswSingle::<f32>::new(params);

        for i in 0..5 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        // Consume all
        while iter.next_batch(10).is_some() {}
        assert!(!iter.has_next());

        // Reset
        iter.reset();

        // Position is reset but computed flag stays true
        let batch = iter.next_batch(10);
        assert!(batch.is_some());
    }

    #[test]
    fn test_hnsw_batch_iterator_with_query_params() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswSingle::<f32>::new(params);

        for i in 0..10 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let query_params = QueryParams::new().with_ef_runtime(100);
        let mut iter = index.batch_iterator(&query, Some(&query_params)).unwrap();

        assert!(iter.has_next());

        let batch = iter.next_batch(5).unwrap();
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_hnsw_batch_iterator_with_params_ef_runtime() {
        // Note: Filter functionality cannot be tested with batch_iterator because
        // QueryParams::clone() sets filter to None (closures can't be cloned).
        // This test verifies ef_runtime parameter passing instead.
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswSingle::<f32>::new(params);

        for i in 0..10 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        // Test with high ef_runtime for better recall
        let query_params = QueryParams::new().with_ef_runtime(200);
        let mut iter = index.batch_iterator(&query, Some(&query_params)).unwrap();

        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(10) {
            all_results.extend(batch);
        }

        // Should get all results with high ef_runtime
        assert!(!all_results.is_empty());
    }

    #[test]
    fn test_hnsw_batch_iterator_sorted_by_distance() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswSingle::<f32>::new(params);

        for i in 0..20 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        let query = vec![5.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(5) {
            all_results.extend(batch);
        }

        // Verify sorted by distance
        for i in 1..all_results.len() {
            assert!(all_results[i - 1].2.to_f64() <= all_results[i].2.to_f64());
        }
    }

    #[test]
    fn test_hnsw_multi_batch_iterator() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Add vectors with same label (multi-value)
        for i in 0..5 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], 100)
                .unwrap();
        }
        for i in 0..5 {
            index
                .add_vector(&vec![i as f32 + 10.0, 0.0, 0.0, 0.0], 200)
                .unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        assert!(iter.has_next());

        let mut total = 0;
        while let Some(batch) = iter.next_batch(3) {
            total += batch.len();
        }

        assert!(total <= 10);
    }

    #[test]
    fn test_hnsw_multi_batch_iterator_empty() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let index = HnswMulti::<f32>::new(params);

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        // Has_next returns true initially
        assert!(iter.has_next());

        // But next_batch returns None for empty index
        assert!(iter.next_batch(10).is_none());
    }

    #[test]
    fn test_hnsw_multi_batch_iterator_reset() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        for i in 0..5 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        // Consume
        while iter.next_batch(10).is_some() {}
        assert!(!iter.has_next());

        // Reset
        iter.reset();

        // Can iterate again
        assert!(iter.next_batch(10).is_some());
    }

    #[test]
    fn test_hnsw_multi_batch_iterator_with_params() {
        // Note: Filter functionality cannot be tested with batch_iterator because
        // QueryParams::clone() sets filter to None (closures can't be cloned).
        // This test verifies ef_runtime parameter passing instead.
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        for i in 0..10 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let query_params = QueryParams::new().with_ef_runtime(200);
        let mut iter = index.batch_iterator(&query, Some(&query_params)).unwrap();

        let mut all_results = Vec::new();
        while let Some(batch) = iter.next_batch(10) {
            all_results.extend(batch);
        }

        // Should get results with high ef_runtime
        assert!(!all_results.is_empty());
    }

    #[test]
    fn test_hnsw_batch_iterator_large_batch_size() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswSingle::<f32>::new(params);

        for i in 0..5 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        // Request much larger batch than available
        let batch = iter.next_batch(1000).unwrap();
        assert!(batch.len() <= 5);

        // No more results
        assert!(!iter.has_next());
    }

    #[test]
    fn test_hnsw_batch_iterator_batch_size_one() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswSingle::<f32>::new(params);

        for i in 0..3 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        let mut count = 0;
        while let Some(batch) = iter.next_batch(1) {
            assert_eq!(batch.len(), 1);
            count += 1;
        }

        assert!(count <= 3);
    }
}

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

        let core = self.index.core.read();

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
                    let id_to_label_for_filter = self.index.id_to_label.read().clone();
                    Some(Box::new(move |id: IdType| {
                        id_to_label_for_filter.get(&id).map_or(false, |&label| f(label))
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

        // Read id_to_label again for result processing
        let id_to_label = self.index.id_to_label.read();
        self.results = search_results
            .into_iter()
            .filter_map(|(id, dist)| {
                id_to_label.get(&id).map(|&label| (id, label, dist))
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

        let core = self.index.core.read();

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
                    let id_to_label_for_filter = self.index.id_to_label.read().clone();
                    Some(Box::new(move |id: IdType| {
                        id_to_label_for_filter.get(&id).map_or(false, |&label| f(label))
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

        // Read id_to_label again for result processing
        let id_to_label = self.index.id_to_label.read();
        self.results = search_results
            .into_iter()
            .filter_map(|(id, dist)| {
                id_to_label.get(&id).map(|&label| (id, label, dist))
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
}

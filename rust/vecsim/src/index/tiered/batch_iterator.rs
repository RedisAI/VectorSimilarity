//! Batch iterator implementations for tiered indices.
//!
//! These iterators merge results from both flat buffer and HNSW backend,
//! returning them in sorted order by distance.

use super::multi::TieredMulti;
use super::single::TieredSingle;
use crate::index::traits::{BatchIterator, VecSimIndex};
use crate::query::QueryParams;
use crate::types::{DistanceType, IdType, LabelType, VectorElement};
use std::cmp::Ordering;

/// Batch iterator for single-value tiered index.
///
/// Merges and sorts results from both flat buffer and HNSW backend.
pub struct TieredBatchIterator<'a, T: VectorElement> {
    /// Reference to the tiered index.
    index: &'a TieredSingle<T>,
    /// The query vector.
    query: Vec<T>,
    /// Query parameters.
    params: Option<QueryParams>,
    /// All results sorted by distance.
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    /// Current position in results.
    position: usize,
    /// Whether results have been computed.
    computed: bool,
}

impl<'a, T: VectorElement> TieredBatchIterator<'a, T> {
    /// Create a new batch iterator.
    pub fn new(
        index: &'a TieredSingle<T>,
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

    /// Compute all results from both tiers.
    fn compute_results(&mut self) {
        if self.computed {
            return;
        }

        // Get results from flat buffer
        let flat = self.index.flat.read();
        if let Ok(mut iter) = flat.batch_iterator(&self.query, self.params.as_ref()) {
            while let Some(batch) = iter.next_batch(1000) {
                self.results.extend(batch);
            }
        }
        drop(flat);

        // Get results from HNSW
        let hnsw = self.index.hnsw.read();
        if let Ok(mut iter) = hnsw.batch_iterator(&self.query, self.params.as_ref()) {
            while let Some(batch) = iter.next_batch(1000) {
                self.results.extend(batch);
            }
        }
        drop(hnsw);

        // Sort by distance
        self.results.sort_by(|a, b| {
            a.2.to_f64()
                .partial_cmp(&b.2.to_f64())
                .unwrap_or(Ordering::Equal)
        });

        self.computed = true;
    }
}

impl<'a, T: VectorElement> BatchIterator for TieredBatchIterator<'a, T> {
    type DistType = T::DistanceType;

    fn has_next(&self) -> bool {
        !self.computed || self.position < self.results.len()
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

/// Batch iterator for multi-value tiered index.
pub struct TieredMultiBatchIterator<'a, T: VectorElement> {
    /// Reference to the tiered index.
    index: &'a TieredMulti<T>,
    /// The query vector.
    query: Vec<T>,
    /// Query parameters.
    params: Option<QueryParams>,
    /// All results sorted by distance.
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    /// Current position in results.
    position: usize,
    /// Whether results have been computed.
    computed: bool,
}

impl<'a, T: VectorElement> TieredMultiBatchIterator<'a, T> {
    /// Create a new batch iterator.
    pub fn new(
        index: &'a TieredMulti<T>,
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

    /// Compute all results from both tiers.
    fn compute_results(&mut self) {
        if self.computed {
            return;
        }

        // Get results from flat buffer
        let flat = self.index.flat.read();
        if let Ok(mut iter) = flat.batch_iterator(&self.query, self.params.as_ref()) {
            while let Some(batch) = iter.next_batch(1000) {
                self.results.extend(batch);
            }
        }
        drop(flat);

        // Get results from HNSW
        let hnsw = self.index.hnsw.read();
        if let Ok(mut iter) = hnsw.batch_iterator(&self.query, self.params.as_ref()) {
            while let Some(batch) = iter.next_batch(1000) {
                self.results.extend(batch);
            }
        }
        drop(hnsw);

        // Sort by distance
        self.results.sort_by(|a, b| {
            a.2.to_f64()
                .partial_cmp(&b.2.to_f64())
                .unwrap_or(Ordering::Equal)
        });

        self.computed = true;
    }
}

impl<'a, T: VectorElement> BatchIterator for TieredMultiBatchIterator<'a, T> {
    type DistType = T::DistanceType;

    fn has_next(&self) -> bool {
        !self.computed || self.position < self.results.len()
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
    use crate::index::tiered::TieredParams;
    use crate::index::VecSimIndex;

    #[test]
    fn test_tiered_batch_iterator() {
        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredSingle::<f32>::new(params);

        // Add vectors to flat
        for i in 0..5 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        // Flush to HNSW
        index.flush().unwrap();

        // Add more to flat
        for i in 5..10 {
            index
                .add_vector(&vec![i as f32, 0.0, 0.0, 0.0], i as u64)
                .unwrap();
        }

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let mut iter = index.batch_iterator(&query, None).unwrap();

        assert!(iter.has_next());

        // Get first batch
        let batch1 = iter.next_batch(3).unwrap();
        assert_eq!(batch1.len(), 3);

        // Verify ordering
        assert!(batch1[0].2.to_f64() <= batch1[1].2.to_f64());

        // Get all remaining
        let mut total = batch1.len();
        while let Some(batch) = iter.next_batch(3) {
            total += batch.len();
        }
        assert_eq!(total, 10);

        // Reset
        iter.reset();
        assert!(iter.has_next());
    }
}

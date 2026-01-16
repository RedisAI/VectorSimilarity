//! Batch iterator implementations for BruteForce indices.
//!
//! These iterators allow streaming results in batches, which is useful
//! for processing large result sets incrementally.

use super::single::BruteForceSingle;
use super::multi::BruteForceMulti;
use crate::index::traits::BatchIterator;
use crate::query::QueryParams;
use crate::types::{DistanceType, IdType, LabelType, VectorElement};
use std::cmp::Ordering;

/// Batch iterator for single-value BruteForce index.
pub struct BruteForceBatchIterator<'a, T: VectorElement> {
    /// Reference to the index.
    index: &'a BruteForceSingle<T>,
    /// The query vector.
    query: Vec<T>,
    /// Query parameters.
    params: Option<QueryParams>,
    /// All results sorted by distance.
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    /// Current position in results.
    position: usize,
}

impl<'a, T: VectorElement> BruteForceBatchIterator<'a, T> {
    /// Create a new batch iterator.
    pub fn new(
        index: &'a BruteForceSingle<T>,
        query: Vec<T>,
        params: Option<QueryParams>,
    ) -> Self {
        let mut iter = Self {
            index,
            query,
            params,
            results: Vec::new(),
            position: 0,
        };
        iter.compute_all_results();
        iter
    }

    /// Compute all distances and sort results.
    fn compute_all_results(&mut self) {
        let core = self.index.core.read();
        let id_to_label = self.index.id_to_label.read();
        let filter = self.params.as_ref().and_then(|p| p.filter.as_ref());

        for (id, entry) in id_to_label.iter().enumerate() {
            if !entry.is_valid {
                continue;
            }

            if let Some(f) = filter {
                if !f(entry.label) {
                    continue;
                }
            }

            let dist = core.compute_distance(id as IdType, &self.query);
            self.results.push((id as IdType, entry.label, dist));
        }

        // Sort by distance
        self.results.sort_by(|a, b| {
            a.2.to_f64()
                .partial_cmp(&b.2.to_f64())
                .unwrap_or(Ordering::Equal)
        });
    }
}

impl<'a, T: VectorElement> BatchIterator for BruteForceBatchIterator<'a, T> {
    type DistType = T::DistanceType;

    fn has_next(&self) -> bool {
        self.position < self.results.len()
    }

    fn next_batch(
        &mut self,
        batch_size: usize,
    ) -> Option<Vec<(IdType, LabelType, Self::DistType)>> {
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

/// Batch iterator for multi-value BruteForce index.
pub struct BruteForceMultiBatchIterator<'a, T: VectorElement> {
    /// Reference to the index.
    index: &'a BruteForceMulti<T>,
    /// The query vector.
    query: Vec<T>,
    /// Query parameters.
    params: Option<QueryParams>,
    /// All results sorted by distance.
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    /// Current position in results.
    position: usize,
}

impl<'a, T: VectorElement> BruteForceMultiBatchIterator<'a, T> {
    /// Create a new batch iterator.
    pub fn new(
        index: &'a BruteForceMulti<T>,
        query: Vec<T>,
        params: Option<QueryParams>,
    ) -> Self {
        let mut iter = Self {
            index,
            query,
            params,
            results: Vec::new(),
            position: 0,
        };
        iter.compute_all_results();
        iter
    }

    /// Compute all distances and sort results.
    fn compute_all_results(&mut self) {
        let core = self.index.core.read();
        let id_to_label = self.index.id_to_label.read();
        let filter = self.params.as_ref().and_then(|p| p.filter.as_ref());

        for (id, entry) in id_to_label.iter().enumerate() {
            if !entry.is_valid {
                continue;
            }

            if let Some(f) = filter {
                if !f(entry.label) {
                    continue;
                }
            }

            let dist = core.compute_distance(id as IdType, &self.query);
            self.results.push((id as IdType, entry.label, dist));
        }

        // Sort by distance
        self.results.sort_by(|a, b| {
            a.2.to_f64()
                .partial_cmp(&b.2.to_f64())
                .unwrap_or(Ordering::Equal)
        });
    }
}

impl<'a, T: VectorElement> BatchIterator for BruteForceMultiBatchIterator<'a, T> {
    type DistType = T::DistanceType;

    fn has_next(&self) -> bool {
        self.position < self.results.len()
    }

    fn next_batch(
        &mut self,
        batch_size: usize,
    ) -> Option<Vec<(IdType, LabelType, Self::DistType)>> {
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
    use crate::index::brute_force::BruteForceParams;
    use crate::index::VecSimIndex;

    #[test]
    fn test_batch_iterator_single() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceSingle::<f32>::new(params);

        for i in 0..10 {
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
        assert!(batch1[1].2.to_f64() <= batch1[2].2.to_f64());

        // Get remaining batches
        let mut total = batch1.len();
        while let Some(batch) = iter.next_batch(3) {
            total += batch.len();
        }
        assert_eq!(total, 10);

        // Reset and verify
        iter.reset();
        assert!(iter.has_next());
    }
}

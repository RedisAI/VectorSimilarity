//! Batch iterator implementations for BruteForce indices.
//!
//! These iterators hold pre-computed results, allowing streaming
//! in batches for processing large result sets incrementally.

use crate::index::traits::BatchIterator;
use crate::types::{IdType, LabelType, VectorElement};

/// Batch iterator for single-value BruteForce index.
///
/// Holds pre-computed, sorted results from the index.
pub struct BruteForceBatchIterator<T: VectorElement> {
    /// All results sorted by distance.
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    /// Current position in results.
    position: usize,
}

impl<T: VectorElement> BruteForceBatchIterator<T> {
    /// Create a new batch iterator with pre-computed results.
    pub fn new(results: Vec<(IdType, LabelType, T::DistanceType)>) -> Self {
        Self {
            results,
            position: 0,
        }
    }
}

impl<T: VectorElement> BatchIterator for BruteForceBatchIterator<T> {
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
///
/// Holds pre-computed, sorted results from the index.
pub struct BruteForceMultiBatchIterator<T: VectorElement> {
    /// All results sorted by distance.
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    /// Current position in results.
    position: usize,
}

impl<T: VectorElement> BruteForceMultiBatchIterator<T> {
    /// Create a new batch iterator with pre-computed results.
    pub fn new(results: Vec<(IdType, LabelType, T::DistanceType)>) -> Self {
        Self {
            results,
            position: 0,
        }
    }
}

impl<T: VectorElement> BatchIterator for BruteForceMultiBatchIterator<T> {
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
    use crate::distance::Metric;
    use crate::index::brute_force::{BruteForceParams, BruteForceSingle};
    use crate::index::VecSimIndex;
    use crate::types::DistanceType;

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

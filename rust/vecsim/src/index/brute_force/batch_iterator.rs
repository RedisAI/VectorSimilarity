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
    use super::*;
    use crate::distance::Metric;
    use crate::index::brute_force::{BruteForceMulti, BruteForceParams, BruteForceSingle};
    use crate::index::traits::BatchIterator;
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

    #[test]
    fn test_brute_force_batch_iterator_empty() {
        let results: Vec<(IdType, LabelType, f32)> = vec![];
        let iter = BruteForceBatchIterator::<f32>::new(results);

        assert!(!iter.has_next());
    }

    #[test]
    fn test_brute_force_batch_iterator_next_batch_empty() {
        let results: Vec<(IdType, LabelType, f32)> = vec![];
        let mut iter = BruteForceBatchIterator::<f32>::new(results);

        assert!(iter.next_batch(10).is_none());
    }

    #[test]
    fn test_brute_force_batch_iterator_single_result() {
        let results = vec![(0, 100, 0.5f32)];
        let mut iter = BruteForceBatchIterator::<f32>::new(results);

        assert!(iter.has_next());

        let batch = iter.next_batch(10).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], (0, 100, 0.5f32));

        assert!(!iter.has_next());
        assert!(iter.next_batch(10).is_none());
    }

    #[test]
    fn test_brute_force_batch_iterator_batch_size_larger_than_results() {
        let results = vec![(0, 100, 0.5f32), (1, 101, 1.0f32), (2, 102, 1.5f32)];
        let mut iter = BruteForceBatchIterator::<f32>::new(results);

        let batch = iter.next_batch(100).unwrap();
        assert_eq!(batch.len(), 3);

        assert!(!iter.has_next());
    }

    #[test]
    fn test_brute_force_batch_iterator_exact_batch_size() {
        let results = vec![
            (0, 100, 0.5f32),
            (1, 101, 1.0f32),
            (2, 102, 1.5f32),
            (3, 103, 2.0f32),
        ];
        let mut iter = BruteForceBatchIterator::<f32>::new(results);

        let batch1 = iter.next_batch(2).unwrap();
        assert_eq!(batch1.len(), 2);

        let batch2 = iter.next_batch(2).unwrap();
        assert_eq!(batch2.len(), 2);

        assert!(!iter.has_next());
    }

    #[test]
    fn test_brute_force_batch_iterator_reset() {
        let results = vec![(0, 100, 0.5f32), (1, 101, 1.0f32)];
        let mut iter = BruteForceBatchIterator::<f32>::new(results);

        // Consume all
        iter.next_batch(10);
        assert!(!iter.has_next());

        // Reset
        iter.reset();
        assert!(iter.has_next());

        // Can iterate again
        let batch = iter.next_batch(10).unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_brute_force_batch_iterator_multiple_batches() {
        let results: Vec<(IdType, LabelType, f32)> = (0..10)
            .map(|i| (i as IdType, i as LabelType + 100, i as f32 * 0.1))
            .collect();
        let mut iter = BruteForceBatchIterator::<f32>::new(results);

        let mut batches = Vec::new();
        while let Some(batch) = iter.next_batch(3) {
            batches.push(batch);
        }

        assert_eq!(batches.len(), 4); // 3 + 3 + 3 + 1
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 3);
        assert_eq!(batches[3].len(), 1);
    }

    #[test]
    fn test_brute_force_multi_batch_iterator_empty() {
        let results: Vec<(IdType, LabelType, f32)> = vec![];
        let iter = BruteForceMultiBatchIterator::<f32>::new(results);

        assert!(!iter.has_next());
    }

    #[test]
    fn test_brute_force_multi_batch_iterator_basic() {
        let results = vec![(0, 100, 0.5f32), (1, 101, 1.0f32), (2, 102, 1.5f32)];
        let mut iter = BruteForceMultiBatchIterator::<f32>::new(results);

        assert!(iter.has_next());

        let batch = iter.next_batch(2).unwrap();
        assert_eq!(batch.len(), 2);

        let batch = iter.next_batch(2).unwrap();
        assert_eq!(batch.len(), 1);

        assert!(!iter.has_next());
    }

    #[test]
    fn test_brute_force_multi_batch_iterator_reset() {
        let results = vec![(0, 100, 0.5f32), (1, 101, 1.0f32)];
        let mut iter = BruteForceMultiBatchIterator::<f32>::new(results);

        // Consume
        iter.next_batch(10);
        assert!(!iter.has_next());

        // Reset
        iter.reset();
        assert!(iter.has_next());
    }

    #[test]
    fn test_batch_iterator_multi_index() {
        let params = BruteForceParams::new(4, Metric::L2);
        let mut index = BruteForceMulti::<f32>::new(params);

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
        assert_eq!(total, 10);
    }

    #[test]
    fn test_batch_iterator_preserves_order() {
        let results = vec![
            (0, 100, 0.1f32),
            (1, 101, 0.2f32),
            (2, 102, 0.3f32),
            (3, 103, 0.4f32),
            (4, 104, 0.5f32),
        ];
        let mut iter = BruteForceBatchIterator::<f32>::new(results);

        let batch1 = iter.next_batch(2).unwrap();
        assert_eq!(batch1[0].0, 0);
        assert_eq!(batch1[1].0, 1);

        let batch2 = iter.next_batch(2).unwrap();
        assert_eq!(batch2[0].0, 2);
        assert_eq!(batch2[1].0, 3);

        let batch3 = iter.next_batch(2).unwrap();
        assert_eq!(batch3[0].0, 4);
    }

    #[test]
    fn test_batch_iterator_batch_size_one() {
        let results = vec![(0, 100, 0.5f32), (1, 101, 1.0f32), (2, 102, 1.5f32)];
        let mut iter = BruteForceBatchIterator::<f32>::new(results);

        let batch1 = iter.next_batch(1).unwrap();
        assert_eq!(batch1.len(), 1);
        assert_eq!(batch1[0].0, 0);

        let batch2 = iter.next_batch(1).unwrap();
        assert_eq!(batch2.len(), 1);
        assert_eq!(batch2[0].0, 1);

        let batch3 = iter.next_batch(1).unwrap();
        assert_eq!(batch3.len(), 1);
        assert_eq!(batch3[0].0, 2);

        assert!(!iter.has_next());
    }
}

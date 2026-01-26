//! Batch iterator implementations for tiered SVS indices.
//!
//! These iterators hold pre-computed results from both flat buffer and SVS backend,
//! returning them in sorted order by distance.

use crate::index::traits::BatchIterator;
use crate::types::{IdType, LabelType, VectorElement};

/// Batch iterator for single-value tiered SVS index.
///
/// Holds pre-computed, sorted results from both flat buffer and SVS backend.
pub struct TieredSvsBatchIterator<T: VectorElement> {
    /// All results sorted by distance.
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    /// Current position in results.
    position: usize,
}

impl<T: VectorElement> TieredSvsBatchIterator<T> {
    /// Create a new batch iterator with pre-computed results.
    pub fn new(results: Vec<(IdType, LabelType, T::DistanceType)>) -> Self {
        Self {
            results,
            position: 0,
        }
    }
}

impl<T: VectorElement> BatchIterator for TieredSvsBatchIterator<T> {
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

/// Batch iterator for multi-value tiered SVS index.
///
/// Holds pre-computed, sorted results from both flat buffer and SVS backend.
pub struct TieredSvsMultiBatchIterator<T: VectorElement> {
    /// All results sorted by distance.
    results: Vec<(IdType, LabelType, T::DistanceType)>,
    /// Current position in results.
    position: usize,
}

impl<T: VectorElement> TieredSvsMultiBatchIterator<T> {
    /// Create a new batch iterator with pre-computed results.
    pub fn new(results: Vec<(IdType, LabelType, T::DistanceType)>) -> Self {
        Self {
            results,
            position: 0,
        }
    }
}

impl<T: VectorElement> BatchIterator for TieredSvsMultiBatchIterator<T> {
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

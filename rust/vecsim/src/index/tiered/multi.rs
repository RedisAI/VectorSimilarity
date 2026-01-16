//! Multi-value tiered index implementation.
//!
//! This index allows multiple vectors per label, combining a BruteForce frontend
//! (for fast writes) with an HNSW backend (for efficient queries).

use super::{merge_range, merge_top_k, TieredParams, WriteMode};
use crate::index::brute_force::{BruteForceMulti, BruteForceParams};
use crate::index::hnsw::HnswMulti;
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply};
use crate::types::{LabelType, VectorElement};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Multi-value tiered index combining BruteForce frontend with HNSW backend.
///
/// Each label can have multiple associated vectors. The tiered architecture
/// provides fast writes to the flat buffer and efficient queries from HNSW.
pub struct TieredMulti<T: VectorElement> {
    /// Flat BruteForce buffer (frontend).
    pub(crate) flat: RwLock<BruteForceMulti<T>>,
    /// HNSW index (backend).
    pub(crate) hnsw: RwLock<HnswMulti<T>>,
    /// Label counts in flat buffer.
    flat_label_counts: RwLock<HashMap<LabelType, usize>>,
    /// Label counts in HNSW.
    hnsw_label_counts: RwLock<HashMap<LabelType, usize>>,
    /// Configuration parameters.
    params: TieredParams,
    /// Total vector count.
    count: AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
}

impl<T: VectorElement> TieredMulti<T> {
    /// Create a new multi-value tiered index.
    pub fn new(params: TieredParams) -> Self {
        let bf_params = BruteForceParams::new(params.dim, params.metric)
            .with_capacity(params.flat_buffer_limit.min(params.initial_capacity));
        let flat = BruteForceMulti::new(bf_params);
        let hnsw = HnswMulti::new(params.hnsw_params.clone());

        Self {
            flat: RwLock::new(flat),
            hnsw: RwLock::new(hnsw),
            flat_label_counts: RwLock::new(HashMap::new()),
            hnsw_label_counts: RwLock::new(HashMap::new()),
            params,
            count: AtomicUsize::new(0),
            capacity: None,
        }
    }

    /// Create with a maximum capacity.
    pub fn with_capacity(params: TieredParams, max_capacity: usize) -> Self {
        let mut index = Self::new(params);
        index.capacity = Some(max_capacity);
        index
    }

    /// Get the current write mode.
    pub fn write_mode(&self) -> WriteMode {
        if self.params.write_mode == WriteMode::InPlace {
            return WriteMode::InPlace;
        }
        if self.flat_size() >= self.params.flat_buffer_limit {
            WriteMode::InPlace
        } else {
            WriteMode::Async
        }
    }

    /// Get the number of vectors in the flat buffer.
    pub fn flat_size(&self) -> usize {
        self.flat.read().index_size()
    }

    /// Get the number of vectors in the HNSW backend.
    pub fn hnsw_size(&self) -> usize {
        self.hnsw.read().index_size()
    }

    /// Get the memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.flat.read().memory_usage() + self.hnsw.read().memory_usage()
    }

    /// Get the count of vectors for a label in the flat buffer.
    pub fn flat_label_count(&self, label: LabelType) -> usize {
        *self.flat_label_counts.read().get(&label).unwrap_or(&0)
    }

    /// Get the count of vectors for a label in the HNSW backend.
    pub fn hnsw_label_count(&self, label: LabelType) -> usize {
        *self.hnsw_label_counts.read().get(&label).unwrap_or(&0)
    }

    /// Flush all vectors from flat buffer to HNSW.
    ///
    /// # Returns
    /// The number of vectors migrated.
    pub fn flush(&mut self) -> Result<usize, IndexError> {
        let flat_labels: Vec<LabelType> = self.flat_label_counts.read().keys().copied().collect();

        if flat_labels.is_empty() {
            return Ok(0);
        }

        let mut migrated = 0;

        // For multi-value, we need to get all vectors for each label
        let vectors: Vec<(LabelType, Vec<Vec<T>>)> = {
            let flat = self.flat.read();
            flat_labels
                .iter()
                .filter_map(|&label| {
                    flat.get_vectors(label).map(|vecs| (label, vecs))
                })
                .collect()
        };

        // Add to HNSW
        {
            let mut hnsw = self.hnsw.write();
            let mut hnsw_label_counts = self.hnsw_label_counts.write();

            for (label, vecs) in &vectors {
                for vec in vecs {
                    match hnsw.add_vector(vec, *label) {
                        Ok(_) => {
                            *hnsw_label_counts.entry(*label).or_insert(0) += 1;
                            migrated += 1;
                        }
                        Err(e) => {
                            eprintln!("Failed to migrate label {label} to HNSW: {e:?}");
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

    /// Compact both tiers to reclaim space.
    pub fn compact(&mut self, shrink: bool) -> usize {
        let flat_reclaimed = self.flat.write().compact(shrink);
        let hnsw_reclaimed = self.hnsw.write().compact(shrink);
        flat_reclaimed + hnsw_reclaimed
    }

    /// Get the fragmentation ratio.
    pub fn fragmentation(&self) -> f64 {
        let flat = self.flat.read();
        let hnsw = self.hnsw.read();
        let flat_frag = flat.fragmentation();
        let hnsw_frag = hnsw.fragmentation();

        let flat_size = flat.index_size() as f64;
        let hnsw_size = hnsw.index_size() as f64;
        let total = flat_size + hnsw_size;

        if total == 0.0 {
            0.0
        } else {
            (flat_frag * flat_size + hnsw_frag * hnsw_size) / total
        }
    }

    /// Clear all vectors from both tiers.
    pub fn clear(&mut self) {
        self.flat.write().clear();
        self.hnsw.write().clear();
        self.flat_label_counts.write().clear();
        self.hnsw_label_counts.write().clear();
        self.count.store(0, Ordering::Relaxed);
    }
}

impl<T: VectorElement> VecSimIndex for TieredMulti<T> {
    type DataType = T;
    type DistType = T::DistanceType;

    fn add_vector(&mut self, vector: &[T], label: LabelType) -> Result<usize, IndexError> {
        if vector.len() != self.params.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.params.dim,
                got: vector.len(),
            });
        }

        if let Some(cap) = self.capacity {
            if self.count.load(Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // For multi-value, always add (don't replace)
        match self.write_mode() {
            WriteMode::Async => {
                let mut flat = self.flat.write();
                let mut flat_label_counts = self.flat_label_counts.write();
                flat.add_vector(vector, label)?;
                *flat_label_counts.entry(label).or_insert(0) += 1;
            }
            WriteMode::InPlace => {
                let mut hnsw = self.hnsw.write();
                let mut hnsw_label_counts = self.hnsw_label_counts.write();
                hnsw.add_vector(vector, label)?;
                *hnsw_label_counts.entry(label).or_insert(0) += 1;
            }
        }

        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let flat_count = self.flat_label_count(label);
        let hnsw_count = self.hnsw_label_count(label);

        if flat_count == 0 && hnsw_count == 0 {
            return Err(IndexError::LabelNotFound(label));
        }

        let mut deleted = 0;

        if flat_count > 0 {
            let mut flat = self.flat.write();
            let mut flat_label_counts = self.flat_label_counts.write();
            if let Ok(n) = flat.delete_vector(label) {
                deleted += n;
                flat_label_counts.remove(&label);
            }
        }

        if hnsw_count > 0 {
            let mut hnsw = self.hnsw.write();
            let mut hnsw_label_counts = self.hnsw_label_counts.write();
            if let Ok(n) = hnsw.delete_vector(label) {
                deleted += n;
                hnsw_label_counts.remove(&label);
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

        let flat = self.flat.read();
        let hnsw = self.hnsw.read();

        let flat_results = flat.top_k_query(query, k, params)?;
        let hnsw_results = hnsw.top_k_query(query, k, params)?;

        Ok(merge_top_k(flat_results, hnsw_results, k))
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

        let flat = self.flat.read();
        let hnsw = self.hnsw.read();

        let flat_results = flat.range_query(query, radius, params)?;
        let hnsw_results = hnsw.range_query(query, radius, params)?;

        Ok(merge_range(flat_results, hnsw_results))
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

        Ok(Box::new(
            super::batch_iterator::TieredMultiBatchIterator::new(self, query.to_vec(), params.cloned()),
        ))
    }

    fn info(&self) -> IndexInfo {
        IndexInfo {
            size: self.index_size(),
            capacity: self.capacity,
            dimension: self.params.dim,
            index_type: "TieredMulti",
            memory_bytes: self.memory_usage(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.flat_label_counts.read().contains_key(&label)
            || self.hnsw_label_counts.read().contains_key(&label)
    }

    fn label_count(&self, label: LabelType) -> usize {
        self.flat_label_count(label) + self.hnsw_label_count(label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_tiered_multi_basic() {
        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredMulti::<f32>::new(params);

        // Add multiple vectors with same label
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 3);
        assert_eq!(index.label_count(1), 2);
        assert_eq!(index.label_count(2), 1);
    }

    #[test]
    fn test_tiered_multi_delete() {
        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        // Delete all vectors for label 1
        let deleted = index.delete_vector(1).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(index.index_size(), 1);
        assert_eq!(index.label_count(1), 0);
    }

    #[test]
    fn test_tiered_multi_flush() {
        let params = TieredParams::new(4, Metric::L2);
        let mut index = TieredMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.flat_size(), 3);
        assert_eq!(index.hnsw_size(), 0);

        let migrated = index.flush().unwrap();
        assert_eq!(migrated, 3);

        assert_eq!(index.flat_size(), 0);
        assert_eq!(index.hnsw_size(), 3);
        assert_eq!(index.label_count(1), 2);
    }
}

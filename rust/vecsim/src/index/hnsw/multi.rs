//! Multi-value HNSW index implementation.
//!
//! This index allows multiple vectors per label.

use super::{HnswCore, HnswParams};
use crate::index::traits::{BatchIterator, IndexError, IndexInfo, QueryError, VecSimIndex};
use crate::query::{QueryParams, QueryReply, QueryResult};
use crate::types::{DistanceType, IdType, LabelType, VectorElement};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};

/// Multi-value HNSW index.
///
/// Each label can have multiple associated vectors.
pub struct HnswMulti<T: VectorElement> {
    /// Core HNSW implementation.
    pub(crate) core: RwLock<HnswCore<T>>,
    /// Label to set of internal IDs mapping.
    label_to_ids: RwLock<HashMap<LabelType, HashSet<IdType>>>,
    /// Internal ID to label mapping.
    pub(crate) id_to_label: RwLock<HashMap<IdType, LabelType>>,
    /// Number of vectors.
    count: std::sync::atomic::AtomicUsize,
    /// Maximum capacity (if set).
    capacity: Option<usize>,
}

impl<T: VectorElement> HnswMulti<T> {
    /// Create a new multi-value HNSW index.
    pub fn new(params: HnswParams) -> Self {
        let initial_capacity = params.initial_capacity;
        let core = HnswCore::new(params);

        Self {
            core: RwLock::new(core),
            label_to_ids: RwLock::new(HashMap::with_capacity(initial_capacity / 2)),
            id_to_label: RwLock::new(HashMap::with_capacity(initial_capacity)),
            count: std::sync::atomic::AtomicUsize::new(0),
            capacity: None,
        }
    }

    /// Create with a maximum capacity.
    pub fn with_capacity(params: HnswParams, max_capacity: usize) -> Self {
        let mut index = Self::new(params);
        index.capacity = Some(max_capacity);
        index
    }

    /// Get the distance metric.
    pub fn metric(&self) -> crate::distance::Metric {
        self.core.read().params.metric
    }

    /// Get the ef_runtime parameter.
    pub fn ef_runtime(&self) -> usize {
        self.core.read().params.ef_runtime
    }

    /// Set the ef_runtime parameter.
    pub fn set_ef_runtime(&self, ef: usize) {
        self.core.write().params.ef_runtime = ef;
    }
}

impl<T: VectorElement> VecSimIndex for HnswMulti<T> {
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
            if self.count.load(std::sync::atomic::Ordering::Relaxed) >= cap {
                return Err(IndexError::CapacityExceeded { capacity: cap });
            }
        }

        // Add the vector
        let id = core.add_vector(vector);
        core.insert(id, label);

        let mut label_to_ids = self.label_to_ids.write();
        let mut id_to_label = self.id_to_label.write();

        // Update mappings
        label_to_ids.entry(label).or_default().insert(id);
        id_to_label.insert(id, label);

        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(1)
    }

    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError> {
        let mut core = self.core.write();
        let mut label_to_ids = self.label_to_ids.write();
        let mut id_to_label = self.id_to_label.write();

        if let Some(ids) = label_to_ids.remove(&label) {
            let count = ids.len();

            for id in ids {
                core.mark_deleted(id);
                id_to_label.remove(&id);
            }

            self.count
                .fetch_sub(count, std::sync::atomic::Ordering::Relaxed);
            Ok(count)
        } else {
            Err(IndexError::LabelNotFound(label))
        }
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

        let ef = params
            .and_then(|p| p.ef_runtime)
            .unwrap_or(core.params.ef_runtime);

        // Build filter if needed
        let has_filter = params.map_or(false, |p| p.filter.is_some());
        let id_label_map: HashMap<IdType, LabelType> = if has_filter {
            self.id_to_label.read().clone()
        } else {
            HashMap::new()
        };

        let filter_fn: Option<Box<dyn Fn(IdType) -> bool>> = if let Some(p) = params {
            if let Some(ref f) = p.filter {
                let f = f.as_ref();
                Some(Box::new(move |id: IdType| {
                    id_label_map.get(&id).map_or(false, |&label| f(label))
                }))
            } else {
                None
            }
        } else {
            None
        };

        let results = core.search(query, k, ef, filter_fn.as_ref().map(|f| f.as_ref()));

        // Look up labels for results
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

        let ef = params
            .and_then(|p| p.ef_runtime)
            .unwrap_or(core.params.ef_runtime)
            .max(1000);

        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        // Build filter if needed
        let has_filter = params.map_or(false, |p| p.filter.is_some());
        let id_label_map: HashMap<IdType, LabelType> = if has_filter {
            self.id_to_label.read().clone()
        } else {
            HashMap::new()
        };

        let filter_fn: Option<Box<dyn Fn(IdType) -> bool>> = if let Some(p) = params {
            if let Some(ref f) = p.filter {
                let f = f.as_ref();
                Some(Box::new(move |id: IdType| {
                    id_label_map.get(&id).map_or(false, |&label| f(label))
                }))
            } else {
                None
            }
        } else {
            None
        };

        let results = core.search(query, count, ef, filter_fn.as_ref().map(|f| f.as_ref()));

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
        self.count.load(std::sync::atomic::Ordering::Relaxed)
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
        drop(core);

        Ok(Box::new(
            super::batch_iterator::HnswMultiBatchIterator::new(self, query.to_vec(), params.cloned()),
        ))
    }

    fn info(&self) -> IndexInfo {
        let core = self.core.read();
        let count = self.count.load(std::sync::atomic::Ordering::Relaxed);

        IndexInfo {
            size: count,
            capacity: self.capacity,
            dimension: core.params.dim,
            index_type: "HnswMulti",
            memory_bytes: count * core.params.dim * std::mem::size_of::<T>()
                + core.graph.len() * std::mem::size_of::<Option<super::graph::ElementGraphData>>()
                + self.label_to_ids.read().capacity()
                    * std::mem::size_of::<(LabelType, HashSet<IdType>)>(),
        }
    }

    fn contains(&self, label: LabelType) -> bool {
        self.label_to_ids.read().contains_key(&label)
    }

    fn label_count(&self, label: LabelType) -> usize {
        self.label_to_ids
            .read()
            .get(&label)
            .map_or(0, |ids| ids.len())
    }
}

unsafe impl<T: VectorElement> Send for HnswMulti<T> {}
unsafe impl<T: VectorElement> Sync for HnswMulti<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_hnsw_multi_basic() {
        let params = HnswParams::new(4, Metric::L2)
            .with_m(4)
            .with_ef_construction(20);
        let mut index = HnswMulti::<f32>::new(params);

        // Add multiple vectors with same label
        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 3);
        assert_eq!(index.label_count(1), 2);
        assert_eq!(index.label_count(2), 1);
    }

    #[test]
    fn test_hnsw_multi_delete() {
        let params = HnswParams::new(4, Metric::L2);
        let mut index = HnswMulti::<f32>::new(params);

        index.add_vector(&vec![1.0, 0.0, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.9, 0.1, 0.0, 0.0], 1).unwrap();
        index.add_vector(&vec![0.0, 1.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(index.index_size(), 3);

        let deleted = index.delete_vector(1).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(index.index_size(), 1);
    }
}

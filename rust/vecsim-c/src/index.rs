//! Index wrapper and lifecycle functions for C FFI.

use crate::params::{BFParams, DiskParams, HNSWParams, SVSParams, TieredParams, VecSimQueryParams};
use crate::types::{
    labelType, QueryReplyInternal, QueryResultInternal, VecSimAlgo, VecSimMetric,
    VecSimQueryReply_Code, VecSimQueryReply_Order, VecSimType,
};
use std::ffi::c_void;
use std::slice;
use vecsim::index::traits::BatchIterator as VecSimBatchIteratorTrait;
use vecsim::index::{
    disk::DiskIndexSingle, BruteForceMulti, BruteForceSingle, HnswMulti, HnswSingle, SvsMulti,
    SvsSingle, TieredMulti, TieredSingle, VecSimIndex as VecSimIndexTrait,
};
use vecsim::query::QueryReply;
use vecsim::types::{BFloat16, DistanceType, Float16, Int8, UInt8, VectorElement};

/// Trait for type-erased index operations.
pub trait IndexWrapper: Send + Sync {
    /// Add a vector to the index.
    fn add_vector(&mut self, vector: *const c_void, label: labelType) -> i32;

    /// Delete a vector by label.
    fn delete_vector(&mut self, label: labelType) -> i32;

    /// Perform top-k query.
    fn top_k_query(
        &self,
        query: *const c_void,
        k: usize,
        params: Option<&VecSimQueryParams>,
    ) -> QueryReplyInternal;

    /// Perform range query.
    fn range_query(
        &self,
        query: *const c_void,
        radius: f64,
        params: Option<&VecSimQueryParams>,
    ) -> QueryReplyInternal;

    /// Get distance from a stored vector to a query vector.
    fn get_distance_from(&self, label: labelType, query: *const c_void) -> f64;

    /// Get index size.
    fn index_size(&self) -> usize;

    /// Get vector dimension.
    fn dimension(&self) -> usize;

    /// Check if label exists.
    fn contains(&self, label: labelType) -> bool;

    /// Get label count for a label.
    fn label_count(&self, label: labelType) -> usize;

    /// Get data type.
    fn data_type(&self) -> VecSimType;

    /// Get algorithm type.
    fn algo(&self) -> VecSimAlgo;

    /// Get metric.
    fn metric(&self) -> VecSimMetric;

    /// Is multi-value index.
    fn is_multi(&self) -> bool;

    /// Create a batch iterator.
    fn create_batch_iterator(
        &self,
        query: *const c_void,
        params: Option<&VecSimQueryParams>,
    ) -> Option<Box<dyn BatchIteratorWrapper>>;

    /// Get memory usage.
    fn memory_usage(&self) -> usize;

    /// Save the index to a file.
    /// Returns true on success, false on failure.
    fn save_to_file(&self, path: &std::path::Path) -> bool;

    /// Check if this is a tiered index.
    fn is_tiered(&self) -> bool {
        false
    }

    /// Flush the flat buffer to the backend (tiered only).
    /// Returns the number of vectors flushed.
    fn tiered_flush(&mut self) -> usize {
        0
    }

    /// Get the size of the flat buffer (tiered only).
    fn tiered_flat_size(&self) -> usize {
        0
    }

    /// Get the size of the backend index (tiered only).
    fn tiered_backend_size(&self) -> usize {
        0
    }

    /// Run garbage collection on a tiered index.
    fn tiered_gc(&mut self) {}

    /// Acquire shared locks on a tiered index.
    fn tiered_acquire_shared_locks(&mut self) {}

    /// Release shared locks on a tiered index.
    fn tiered_release_shared_locks(&mut self) {}

    /// Check if this is a disk-based index.
    fn is_disk(&self) -> bool {
        false
    }

    /// Flush changes to disk (disk indices only).
    /// Returns true on success, false on failure.
    fn disk_flush(&self) -> bool {
        false
    }

    /// Get the neighbors of an element at all levels (HNSW/Tiered only).
    ///
    /// Returns None if the label doesn't exist or if the index type doesn't support this.
    /// Returns Some(Vec<Vec<u64>>) where each inner Vec contains the neighbor labels at that level.
    fn get_element_neighbors(&self, _label: u64) -> Option<Vec<Vec<u64>>> {
        None
    }
}

/// Trait for type-erased batch iterator operations.
pub trait BatchIteratorWrapper: Send {
    /// Check if more results are available.
    fn has_next(&self) -> bool;

    /// Get next batch of results.
    fn next_batch(&mut self, n: usize, order: VecSimQueryReply_Order) -> QueryReplyInternal;

    /// Reset the iterator.
    fn reset(&mut self);
}

/// Owned batch iterator that pre-computes all results.
/// This is used to work around the lifetime issues with type-erased batch iterators.
pub struct OwnedBatchIterator {
    /// Pre-computed results as (label, distance) pairs
    results: Vec<QueryResultInternal>,
    /// Current position in results
    position: usize,
}

impl OwnedBatchIterator {
    /// Create a new owned batch iterator from pre-computed results.
    pub fn new(results: Vec<QueryResultInternal>) -> Self {
        Self {
            results,
            position: 0,
        }
    }
}

impl BatchIteratorWrapper for OwnedBatchIterator {
    fn has_next(&self) -> bool {
        self.position < self.results.len()
    }

    fn next_batch(&mut self, n: usize, order: VecSimQueryReply_Order) -> QueryReplyInternal {
        if self.position >= self.results.len() {
            return QueryReplyInternal::new();
        }

        let end = (self.position + n).min(self.results.len());
        let mut batch: Vec<QueryResultInternal> = self.results[self.position..end].to_vec();
        self.position = end;

        // Sort by requested order
        match order {
            VecSimQueryReply_Order::BY_SCORE => {
                batch.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
            }
            VecSimQueryReply_Order::BY_ID => {
                batch.sort_by_key(|r| r.id);
            }
        }

        QueryReplyInternal {
            results: batch,
            code: VecSimQueryReply_Code::VecSim_QueryReply_OK,
        }
    }

    fn reset(&mut self) {
        self.position = 0;
    }
}

/// Helper function to create an owned batch iterator from an index.
/// This pre-computes all results by calling next_batch until exhausted.
fn create_owned_batch_iterator_from_results<D: DistanceType>(
    mut inner: Box<dyn VecSimBatchIteratorTrait<DistType = D> + '_>,
) -> OwnedBatchIterator {
    let mut all_results = Vec::new();

    // Drain all results from the iterator
    while inner.has_next() {
        if let Some(batch) = inner.next_batch(10000) {
            for (_, label, dist) in batch {
                all_results.push(QueryResultInternal {
                    id: label,
                    score: dist.to_f64(),
                });
            }
        } else {
            break;
        }
    }

    // Sort by score (distance) to ensure results are in order of increasing distance
    all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

    OwnedBatchIterator::new(all_results)
}

/// Macro to implement IndexWrapper for a specific index type without serialization.
macro_rules! impl_index_wrapper {
    ($wrapper:ident, $index:ty, $data:ty, $algo:expr, $is_multi:expr) => {
        pub struct $wrapper {
            index: $index,
            data_type: VecSimType,
        }

        impl $wrapper {
            pub fn new(index: $index, data_type: VecSimType) -> Self {
                Self { index, data_type }
            }

            #[allow(dead_code)]
            pub fn inner(&self) -> &$index {
                &self.index
            }
        }

        impl IndexWrapper for $wrapper {
            fn add_vector(&mut self, vector: *const c_void, label: labelType) -> i32 {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(vector as *const $data, dim) };
                match self.index.add_vector(slice, label) {
                    Ok(count) => count as i32,
                    Err(_) => -1,
                }
            }

            fn delete_vector(&mut self, label: labelType) -> i32 {
                match self.index.delete_vector(label) {
                    Ok(count) => count as i32,
                    Err(_) => 0,
                }
            }

            fn top_k_query(
                &self,
                query: *const c_void,
                k: usize,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());

                match self.index.top_k_query(slice, k, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn range_query(
                &self,
                query: *const c_void,
                radius: f64,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());
                let radius_typed =
                    <<$data as VectorElement>::DistanceType as DistanceType>::from_f64(radius);

                match self.index.range_query(slice, radius_typed, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn get_distance_from(&self, label: labelType, query: *const c_void) -> f64 {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                match self.index.compute_distance(label, slice) {
                    Some(dist) => dist.to_f64(),
                    None => f64::NAN, // Label not found
                }
            }

            fn index_size(&self) -> usize {
                self.index.index_size()
            }

            fn dimension(&self) -> usize {
                self.index.dimension()
            }

            fn contains(&self, label: labelType) -> bool {
                self.index.contains(label)
            }

            fn label_count(&self, label: labelType) -> usize {
                self.index.label_count(label)
            }

            fn data_type(&self) -> VecSimType {
                self.data_type
            }

            fn algo(&self) -> VecSimAlgo {
                $algo
            }

            fn metric(&self) -> VecSimMetric {
                self.index.info().index_type; // Not directly available, use placeholder
                VecSimMetric::VecSimMetric_L2
            }

            fn is_multi(&self) -> bool {
                $is_multi
            }

            fn create_batch_iterator(
                &self,
                query: *const c_void,
                params: Option<&VecSimQueryParams>,
            ) -> Option<Box<dyn BatchIteratorWrapper>> {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());

                match self.index.batch_iterator(slice, rust_params.as_ref()) {
                    Ok(inner) => Some(Box::new(create_owned_batch_iterator_from_results(inner))),
                    Err(_) => None,
                }
            }

            fn memory_usage(&self) -> usize {
                self.index.info().memory_bytes
            }

            fn save_to_file(&self, _path: &std::path::Path) -> bool {
                false  // Serialization not supported for this type
            }
        }
    };
}

/// Macro to implement IndexWrapper for a specific index type WITH serialization support.
macro_rules! impl_index_wrapper_with_serialization {
    ($wrapper:ident, $index:ty, $data:ty, $algo:expr, $is_multi:expr) => {
        pub struct $wrapper {
            index: $index,
            data_type: VecSimType,
        }

        impl $wrapper {
            pub fn new(index: $index, data_type: VecSimType) -> Self {
                Self { index, data_type }
            }

            #[allow(dead_code)]
            pub fn inner(&self) -> &$index {
                &self.index
            }
        }

        impl IndexWrapper for $wrapper {
            fn add_vector(&mut self, vector: *const c_void, label: labelType) -> i32 {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(vector as *const $data, dim) };
                match self.index.add_vector(slice, label) {
                    Ok(count) => count as i32,
                    Err(_) => -1,
                }
            }

            fn delete_vector(&mut self, label: labelType) -> i32 {
                match self.index.delete_vector(label) {
                    Ok(count) => count as i32,
                    Err(_) => 0,
                }
            }

            fn top_k_query(
                &self,
                query: *const c_void,
                k: usize,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());

                match self.index.top_k_query(slice, k, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn range_query(
                &self,
                query: *const c_void,
                radius: f64,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());
                let radius_typed =
                    <<$data as VectorElement>::DistanceType as DistanceType>::from_f64(radius);

                match self.index.range_query(slice, radius_typed, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn get_distance_from(&self, _label: labelType, _query: *const c_void) -> f64 {
                // SVS indices don't support compute_distance
                f64::NAN
            }

            fn index_size(&self) -> usize {
                self.index.index_size()
            }

            fn dimension(&self) -> usize {
                self.index.dimension()
            }

            fn contains(&self, label: labelType) -> bool {
                self.index.contains(label)
            }

            fn label_count(&self, label: labelType) -> usize {
                self.index.label_count(label)
            }

            fn data_type(&self) -> VecSimType {
                self.data_type
            }

            fn algo(&self) -> VecSimAlgo {
                $algo
            }

            fn metric(&self) -> VecSimMetric {
                self.index.info().index_type;
                VecSimMetric::VecSimMetric_L2
            }

            fn is_multi(&self) -> bool {
                $is_multi
            }

            fn create_batch_iterator(
                &self,
                query: *const c_void,
                params: Option<&VecSimQueryParams>,
            ) -> Option<Box<dyn BatchIteratorWrapper>> {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());

                match self.index.batch_iterator(slice, rust_params.as_ref()) {
                    Ok(inner) => Some(Box::new(create_owned_batch_iterator_from_results(inner))),
                    Err(_) => None,
                }
            }

            fn memory_usage(&self) -> usize {
                self.index.info().memory_bytes
            }

            fn save_to_file(&self, path: &std::path::Path) -> bool {
                self.index.save_to_file(path).is_ok()
            }
        }
    };
}

// Implement wrappers for BruteForce indices
// Note: Serialization is only supported for f32 types
impl_index_wrapper_with_serialization!(
    BruteForceSingleF32Wrapper,
    BruteForceSingle<f32>,
    f32,
    VecSimAlgo::VecSimAlgo_BF,
    false
);
impl_index_wrapper!(
    BruteForceSingleF64Wrapper,
    BruteForceSingle<f64>,
    f64,
    VecSimAlgo::VecSimAlgo_BF,
    false
);
impl_index_wrapper!(
    BruteForceSingleBF16Wrapper,
    BruteForceSingle<BFloat16>,
    BFloat16,
    VecSimAlgo::VecSimAlgo_BF,
    false
);
impl_index_wrapper!(
    BruteForceSingleFP16Wrapper,
    BruteForceSingle<Float16>,
    Float16,
    VecSimAlgo::VecSimAlgo_BF,
    false
);
impl_index_wrapper!(
    BruteForceSingleI8Wrapper,
    BruteForceSingle<Int8>,
    Int8,
    VecSimAlgo::VecSimAlgo_BF,
    false
);
impl_index_wrapper!(
    BruteForceSingleU8Wrapper,
    BruteForceSingle<UInt8>,
    UInt8,
    VecSimAlgo::VecSimAlgo_BF,
    false
);

impl_index_wrapper_with_serialization!(
    BruteForceMultiF32Wrapper,
    BruteForceMulti<f32>,
    f32,
    VecSimAlgo::VecSimAlgo_BF,
    true
);
impl_index_wrapper!(
    BruteForceMultiF64Wrapper,
    BruteForceMulti<f64>,
    f64,
    VecSimAlgo::VecSimAlgo_BF,
    true
);
impl_index_wrapper!(
    BruteForceMultiBF16Wrapper,
    BruteForceMulti<BFloat16>,
    BFloat16,
    VecSimAlgo::VecSimAlgo_BF,
    true
);
impl_index_wrapper!(
    BruteForceMultiFP16Wrapper,
    BruteForceMulti<Float16>,
    Float16,
    VecSimAlgo::VecSimAlgo_BF,
    true
);
impl_index_wrapper!(
    BruteForceMultiI8Wrapper,
    BruteForceMulti<Int8>,
    Int8,
    VecSimAlgo::VecSimAlgo_BF,
    true
);
impl_index_wrapper!(
    BruteForceMultiU8Wrapper,
    BruteForceMulti<UInt8>,
    UInt8,
    VecSimAlgo::VecSimAlgo_BF,
    true
);

// Macro for HNSW Single wrappers with get_element_neighbors support
macro_rules! impl_hnsw_single_wrapper {
    ($wrapper:ident, $index:ty, $data:ty) => {
        pub struct $wrapper {
            index: $index,
            data_type: VecSimType,
        }

        impl $wrapper {
            pub fn new(index: $index, data_type: VecSimType) -> Self {
                Self { index, data_type }
            }

            #[allow(dead_code)]
            pub fn inner(&self) -> &$index {
                &self.index
            }
        }

        impl IndexWrapper for $wrapper {
            fn add_vector(&mut self, vector: *const c_void, label: labelType) -> i32 {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(vector as *const $data, dim) };
                match self.index.add_vector(slice, label) {
                    Ok(count) => count as i32,
                    Err(_) => -1,
                }
            }

            fn delete_vector(&mut self, label: labelType) -> i32 {
                match self.index.delete_vector(label) {
                    Ok(count) => count as i32,
                    Err(_) => 0,
                }
            }

            fn top_k_query(
                &self,
                query: *const c_void,
                k: usize,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());

                match self.index.top_k_query(slice, k, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn range_query(
                &self,
                query: *const c_void,
                radius: f64,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());
                let radius_typed =
                    <<$data as VectorElement>::DistanceType as DistanceType>::from_f64(radius);

                match self.index.range_query(slice, radius_typed, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn get_distance_from(&self, label: labelType, query: *const c_void) -> f64 {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                match self.index.compute_distance(label, slice) {
                    Some(dist) => dist.to_f64(),
                    None => f64::NAN, // Label not found
                }
            }

            fn index_size(&self) -> usize {
                self.index.index_size()
            }

            fn dimension(&self) -> usize {
                self.index.dimension()
            }

            fn contains(&self, label: labelType) -> bool {
                self.index.contains(label)
            }

            fn label_count(&self, label: labelType) -> usize {
                self.index.label_count(label)
            }

            fn data_type(&self) -> VecSimType {
                self.data_type
            }

            fn algo(&self) -> VecSimAlgo {
                VecSimAlgo::VecSimAlgo_HNSWLIB
            }

            fn metric(&self) -> VecSimMetric {
                // Metric is not directly available from IndexInfo, use placeholder
                VecSimMetric::VecSimMetric_L2
            }

            fn is_multi(&self) -> bool {
                false
            }

            fn create_batch_iterator(
                &self,
                query: *const c_void,
                params: Option<&VecSimQueryParams>,
            ) -> Option<Box<dyn BatchIteratorWrapper>> {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());

                match self.index.batch_iterator(slice, rust_params.as_ref()) {
                    Ok(inner) => Some(Box::new(create_owned_batch_iterator_from_results(inner))),
                    Err(_) => None,
                }
            }

            fn memory_usage(&self) -> usize {
                self.index.info().memory_bytes
            }

            fn save_to_file(&self, path: &std::path::Path) -> bool {
                self.index.save_to_file(path).is_ok()
            }

            fn get_element_neighbors(&self, label: u64) -> Option<Vec<Vec<u64>>> {
                self.index.get_element_neighbors(label)
            }
        }
    };
}

// Implement wrappers for HNSW Single indices
impl_hnsw_single_wrapper!(HnswSingleF32Wrapper, HnswSingle<f32>, f32);
impl_hnsw_single_wrapper!(HnswSingleF64Wrapper, HnswSingle<f64>, f64);
impl_hnsw_single_wrapper!(HnswSingleBF16Wrapper, HnswSingle<BFloat16>, BFloat16);
impl_hnsw_single_wrapper!(HnswSingleFP16Wrapper, HnswSingle<Float16>, Float16);
impl_hnsw_single_wrapper!(HnswSingleI8Wrapper, HnswSingle<Int8>, Int8);
impl_hnsw_single_wrapper!(HnswSingleU8Wrapper, HnswSingle<UInt8>, UInt8);

impl_index_wrapper_with_serialization!(
    HnswMultiF32Wrapper,
    HnswMulti<f32>,
    f32,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);
impl_index_wrapper_with_serialization!(
    HnswMultiF64Wrapper,
    HnswMulti<f64>,
    f64,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);
impl_index_wrapper_with_serialization!(
    HnswMultiBF16Wrapper,
    HnswMulti<BFloat16>,
    BFloat16,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);
impl_index_wrapper_with_serialization!(
    HnswMultiFP16Wrapper,
    HnswMulti<Float16>,
    Float16,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);
impl_index_wrapper_with_serialization!(
    HnswMultiI8Wrapper,
    HnswMulti<Int8>,
    Int8,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);
impl_index_wrapper_with_serialization!(
    HnswMultiU8Wrapper,
    HnswMulti<UInt8>,
    UInt8,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);

// Macro for SVS indices (no compute_distance support)
macro_rules! impl_svs_wrapper {
    ($wrapper:ident, $index:ty, $data:ty, $is_multi:expr) => {
        pub struct $wrapper {
            index: $index,
            data_type: VecSimType,
        }

        impl $wrapper {
            pub fn new(index: $index, data_type: VecSimType) -> Self {
                Self { index, data_type }
            }

            #[allow(dead_code)]
            pub fn inner(&self) -> &$index {
                &self.index
            }
        }

        impl IndexWrapper for $wrapper {
            fn add_vector(&mut self, vector: *const c_void, label: labelType) -> i32 {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(vector as *const $data, dim) };
                match self.index.add_vector(slice, label) {
                    Ok(count) => count as i32,
                    Err(_) => -1,
                }
            }

            fn delete_vector(&mut self, label: labelType) -> i32 {
                match self.index.delete_vector(label) {
                    Ok(count) => count as i32,
                    Err(_) => 0,
                }
            }

            fn top_k_query(
                &self,
                query: *const c_void,
                k: usize,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());

                match self.index.top_k_query(slice, k, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn range_query(
                &self,
                query: *const c_void,
                radius: f64,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());
                let radius_typed =
                    <<$data as VectorElement>::DistanceType as DistanceType>::from_f64(radius);

                match self.index.range_query(slice, radius_typed, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn get_distance_from(&self, _label: labelType, _query: *const c_void) -> f64 {
                // SVS indices don't support compute_distance
                f64::NAN
            }

            fn index_size(&self) -> usize {
                self.index.index_size()
            }

            fn dimension(&self) -> usize {
                self.index.dimension()
            }

            fn contains(&self, label: labelType) -> bool {
                self.index.contains(label)
            }

            fn label_count(&self, label: labelType) -> usize {
                self.index.label_count(label)
            }

            fn data_type(&self) -> VecSimType {
                self.data_type
            }

            fn algo(&self) -> VecSimAlgo {
                VecSimAlgo::VecSimAlgo_SVS
            }

            fn metric(&self) -> VecSimMetric {
                VecSimMetric::VecSimMetric_L2
            }

            fn is_multi(&self) -> bool {
                $is_multi
            }

            fn create_batch_iterator(
                &self,
                query: *const c_void,
                params: Option<&VecSimQueryParams>,
            ) -> Option<Box<dyn BatchIteratorWrapper>> {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());

                match self.index.batch_iterator(slice, rust_params.as_ref()) {
                    Ok(inner) => Some(Box::new(create_owned_batch_iterator_from_results(inner))),
                    Err(_) => None,
                }
            }

            fn memory_usage(&self) -> usize {
                self.index.info().memory_bytes
            }

            fn save_to_file(&self, _path: &std::path::Path) -> bool {
                false  // Serialization not supported for most SVS types
            }
        }
    };
}

// Implement wrappers for SVS indices
// Note: SVS serialization is only supported for f32 single
impl_svs_wrapper!(SvsSingleF32Wrapper, SvsSingle<f32>, f32, false);
impl_svs_wrapper!(SvsSingleF64Wrapper, SvsSingle<f64>, f64, false);
impl_svs_wrapper!(SvsSingleBF16Wrapper, SvsSingle<BFloat16>, BFloat16, false);
impl_svs_wrapper!(SvsSingleFP16Wrapper, SvsSingle<Float16>, Float16, false);
impl_svs_wrapper!(SvsSingleI8Wrapper, SvsSingle<Int8>, Int8, false);
impl_svs_wrapper!(SvsSingleU8Wrapper, SvsSingle<UInt8>, UInt8, false);

impl_svs_wrapper!(SvsMultiF32Wrapper, SvsMulti<f32>, f32, true);
impl_svs_wrapper!(SvsMultiF64Wrapper, SvsMulti<f64>, f64, true);
impl_svs_wrapper!(SvsMultiBF16Wrapper, SvsMulti<BFloat16>, BFloat16, true);
impl_svs_wrapper!(SvsMultiFP16Wrapper, SvsMulti<Float16>, Float16, true);
impl_svs_wrapper!(SvsMultiI8Wrapper, SvsMulti<Int8>, Int8, true);
impl_svs_wrapper!(SvsMultiU8Wrapper, SvsMulti<UInt8>, UInt8, true);

// ============================================================================
// Tiered Index Wrappers
// ============================================================================

/// Macro to implement IndexWrapper for Tiered index types.
macro_rules! impl_tiered_wrapper {
    ($wrapper:ident, $index:ty, $data:ty, $is_multi:expr) => {
        pub struct $wrapper {
            index: $index,
            data_type: VecSimType,
        }

        impl $wrapper {
            pub fn new(index: $index, data_type: VecSimType) -> Self {
                Self { index, data_type }
            }

            #[allow(dead_code)]
            pub fn inner(&self) -> &$index {
                &self.index
            }

            #[allow(dead_code)]
            pub fn inner_mut(&mut self) -> &mut $index {
                &mut self.index
            }
        }

        impl IndexWrapper for $wrapper {
            fn add_vector(&mut self, vector: *const c_void, label: labelType) -> i32 {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(vector as *const $data, dim) };
                match self.index.add_vector(slice, label) {
                    Ok(count) => count as i32,
                    Err(_) => -1,
                }
            }

            fn delete_vector(&mut self, label: labelType) -> i32 {
                match self.index.delete_vector(label) {
                    Ok(count) => count as i32,
                    Err(_) => 0,
                }
            }

            fn top_k_query(
                &self,
                query: *const c_void,
                k: usize,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());

                match self.index.top_k_query(slice, k, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn range_query(
                &self,
                query: *const c_void,
                radius: f64,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());
                let radius_typed =
                    <<$data as VectorElement>::DistanceType as DistanceType>::from_f64(radius);

                match self.index.range_query(slice, radius_typed, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn get_distance_from(&self, _label: labelType, _query: *const c_void) -> f64 {
                // Tiered indices don't support compute_distance directly
                f64::NAN
            }

            fn index_size(&self) -> usize {
                self.index.index_size()
            }

            fn dimension(&self) -> usize {
                self.index.dimension()
            }

            fn contains(&self, label: labelType) -> bool {
                self.index.contains(label)
            }

            fn label_count(&self, label: labelType) -> usize {
                self.index.label_count(label)
            }

            fn data_type(&self) -> VecSimType {
                self.data_type
            }

            fn algo(&self) -> VecSimAlgo {
                VecSimAlgo::VecSimAlgo_TIERED
            }

            fn metric(&self) -> VecSimMetric {
                VecSimMetric::VecSimMetric_L2
            }

            fn is_multi(&self) -> bool {
                $is_multi
            }

            fn create_batch_iterator(
                &self,
                query: *const c_void,
                params: Option<&VecSimQueryParams>,
            ) -> Option<Box<dyn BatchIteratorWrapper>> {
                let dim = self.index.dimension();
                let slice = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());

                match self.index.batch_iterator(slice, rust_params.as_ref()) {
                    Ok(inner) => Some(Box::new(create_owned_batch_iterator_from_results(inner))),
                    Err(_) => None,
                }
            }

            fn memory_usage(&self) -> usize {
                self.index.info().memory_bytes
            }

            fn save_to_file(&self, _path: &std::path::Path) -> bool {
                false // Tiered serialization not yet exposed via C API
            }

            fn is_tiered(&self) -> bool {
                true
            }

            fn tiered_flush(&mut self) -> usize {
                self.index.flush().unwrap_or(0)
            }

            fn tiered_flat_size(&self) -> usize {
                self.index.flat_size()
            }

            fn tiered_backend_size(&self) -> usize {
                self.index.hnsw_size()
            }

            fn tiered_gc(&mut self) {
                // Flush vectors from flat buffer to HNSW backend.
                // This simulates the C++ behavior where GC processes pending
                // swap jobs that move vectors from flat to HNSW.
                let _ = self.index.flush();
            }

            fn tiered_acquire_shared_locks(&mut self) {
                // No-op for now
            }

            fn tiered_release_shared_locks(&mut self) {
                // No-op for now
            }

            fn get_element_neighbors(&self, label: u64) -> Option<Vec<Vec<u64>>> {
                self.index.get_element_neighbors(label)
            }
        }
    };
}

// Implement wrappers for Tiered indices
impl_tiered_wrapper!(TieredSingleF32Wrapper, TieredSingle<f32>, f32, false);
impl_tiered_wrapper!(TieredMultiF32Wrapper, TieredMulti<f32>, f32, true);
impl_tiered_wrapper!(TieredSingleF64Wrapper, TieredSingle<f64>, f64, false);
impl_tiered_wrapper!(TieredMultiF64Wrapper, TieredMulti<f64>, f64, true);
impl_tiered_wrapper!(TieredSingleBF16Wrapper, TieredSingle<BFloat16>, BFloat16, false);
impl_tiered_wrapper!(TieredMultiBF16Wrapper, TieredMulti<BFloat16>, BFloat16, true);
impl_tiered_wrapper!(TieredSingleFP16Wrapper, TieredSingle<Float16>, Float16, false);
impl_tiered_wrapper!(TieredMultiFP16Wrapper, TieredMulti<Float16>, Float16, true);
impl_tiered_wrapper!(TieredSingleI8Wrapper, TieredSingle<Int8>, Int8, false);
impl_tiered_wrapper!(TieredMultiI8Wrapper, TieredMulti<Int8>, Int8, true);
impl_tiered_wrapper!(TieredSingleU8Wrapper, TieredSingle<UInt8>, UInt8, false);
impl_tiered_wrapper!(TieredMultiU8Wrapper, TieredMulti<UInt8>, UInt8, true);

// ============================================================================
// Disk Index Wrappers
// ============================================================================

/// Macro to implement IndexWrapper for Disk index types.
macro_rules! impl_disk_wrapper {
    ($wrapper:ident, $data:ty) => {
        pub struct $wrapper {
            index: DiskIndexSingle<$data>,
            data_type: VecSimType,
            metric: VecSimMetric,
        }

        impl $wrapper {
            pub fn new(
                index: DiskIndexSingle<$data>,
                data_type: VecSimType,
                metric: VecSimMetric,
            ) -> Self {
                Self {
                    index,
                    data_type,
                    metric,
                }
            }
        }

        impl IndexWrapper for $wrapper {
            fn add_vector(&mut self, vector: *const c_void, label: labelType) -> i32 {
                let dim = self.index.dimension();
                let data = unsafe { slice::from_raw_parts(vector as *const $data, dim) };
                match self.index.add_vector(data, label) {
                    Ok(count) => count as i32,
                    Err(_) => -1,
                }
            }

            fn delete_vector(&mut self, label: labelType) -> i32 {
                match self.index.delete_vector(label) {
                    Ok(count) => count as i32,
                    Err(_) => 0,
                }
            }

            fn top_k_query(
                &self,
                query: *const c_void,
                k: usize,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let query_data = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());
                match self.index.top_k_query(query_data, k, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn range_query(
                &self,
                query: *const c_void,
                radius: f64,
                params: Option<&VecSimQueryParams>,
            ) -> QueryReplyInternal {
                let dim = self.index.dimension();
                let query_data = unsafe { slice::from_raw_parts(query as *const $data, dim) };
                let rust_params = params.map(|p| p.to_rust_params());
                let radius_typed =
                    <<$data as VectorElement>::DistanceType as DistanceType>::from_f64(radius);
                match self.index.range_query(query_data, radius_typed, rust_params.as_ref()) {
                    Ok(reply) => convert_query_reply(reply),
                    Err(_) => QueryReplyInternal::new(),
                }
            }

            fn get_distance_from(&self, _label: labelType, _query: *const c_void) -> f64 {
                // Disk indices don't support get_distance_from directly
                f64::NAN
            }

            fn index_size(&self) -> usize {
                self.index.index_size()
            }

            fn dimension(&self) -> usize {
                self.index.dimension()
            }

            fn contains(&self, label: labelType) -> bool {
                self.index.contains(label)
            }

            fn label_count(&self, label: labelType) -> usize {
                self.index.label_count(label)
            }

            fn data_type(&self) -> VecSimType {
                self.data_type
            }

            fn algo(&self) -> VecSimAlgo {
                // Disk indices use BF or SVS backend, report as BF for now
                VecSimAlgo::VecSimAlgo_BF
            }

            fn metric(&self) -> VecSimMetric {
                self.metric
            }

            fn is_multi(&self) -> bool {
                false // DiskIndexSingle is always single-value
            }

            fn create_batch_iterator(
                &self,
                _query: *const c_void,
                _params: Option<&VecSimQueryParams>,
            ) -> Option<Box<dyn BatchIteratorWrapper>> {
                // Batch iteration not yet implemented for disk indices
                None
            }

            fn memory_usage(&self) -> usize {
                // Memory usage is minimal since data is on disk
                std::mem::size_of::<Self>()
            }

            fn save_to_file(&self, _path: &std::path::Path) -> bool {
                // Disk indices are already persisted
                self.disk_flush()
            }

            fn is_disk(&self) -> bool {
                true
            }

            fn disk_flush(&self) -> bool {
                self.index.flush().is_ok()
            }
        }
    };
}

// Implement wrappers for Disk indices (f32 only for now)
impl_disk_wrapper!(DiskSingleF32Wrapper, f32);

/// Create a new disk-based index.
pub fn create_disk_index(params: &DiskParams) -> Option<Box<IndexHandle>> {
    let rust_params = unsafe { params.to_rust_params()? };
    let data_type = params.base.type_;
    let metric = params.base.metric;
    let dim = params.base.dim;

    // Only f32 is supported for now
    if data_type != VecSimType::VecSimType_FLOAT32 {
        return None;
    }

    let index = DiskIndexSingle::<f32>::new(rust_params).ok()?;
    let wrapper: Box<dyn IndexWrapper> =
        Box::new(DiskSingleF32Wrapper::new(index, data_type, metric));

    Some(Box::new(IndexHandle::new(
        wrapper,
        data_type,
        VecSimAlgo::VecSimAlgo_BF, // Report as BF for now
        metric,
        dim,
        false, // Disk indices are single-value only
    )))
}

/// Convert a Rust QueryReply to QueryReplyInternal.
fn convert_query_reply<D: DistanceType>(reply: QueryReply<D>) -> QueryReplyInternal {
    let results: Vec<QueryResultInternal> = reply
        .results
        .into_iter()
        .map(|r| QueryResultInternal {
            id: r.label,
            score: r.distance.to_f64(),
        })
        .collect();
    QueryReplyInternal::from_results(results)
}

/// Internal index handle that stores the type-erased wrapper.
pub struct IndexHandle {
    pub wrapper: Box<dyn IndexWrapper>,
    pub data_type: VecSimType,
    pub algo: VecSimAlgo,
    pub metric: VecSimMetric,
    pub dim: usize,
    pub is_multi: bool,
}

impl IndexHandle {
    pub fn new(
        wrapper: Box<dyn IndexWrapper>,
        data_type: VecSimType,
        algo: VecSimAlgo,
        metric: VecSimMetric,
        dim: usize,
        is_multi: bool,
    ) -> Self {
        Self {
            wrapper,
            data_type,
            algo,
            metric,
            dim,
            is_multi,
        }
    }
}

/// Create a new BruteForce index.
pub fn create_brute_force_index(params: &BFParams) -> Option<Box<IndexHandle>> {
    let rust_params = params.to_rust_params();
    let data_type = params.base.type_;
    let metric = params.base.metric;
    let dim = params.base.dim;
    let is_multi = params.base.multi;

    let wrapper: Box<dyn IndexWrapper> = match (data_type, is_multi) {
        (VecSimType::VecSimType_FLOAT32, false) => Box::new(BruteForceSingleF32Wrapper::new(
            BruteForceSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT64, false) => Box::new(BruteForceSingleF64Wrapper::new(
            BruteForceSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_BFLOAT16, false) => Box::new(BruteForceSingleBF16Wrapper::new(
            BruteForceSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT16, false) => Box::new(BruteForceSingleFP16Wrapper::new(
            BruteForceSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_INT8, false) => Box::new(BruteForceSingleI8Wrapper::new(
            BruteForceSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_UINT8, false) => Box::new(BruteForceSingleU8Wrapper::new(
            BruteForceSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT32, true) => Box::new(BruteForceMultiF32Wrapper::new(
            BruteForceMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT64, true) => Box::new(BruteForceMultiF64Wrapper::new(
            BruteForceMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_BFLOAT16, true) => Box::new(BruteForceMultiBF16Wrapper::new(
            BruteForceMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT16, true) => Box::new(BruteForceMultiFP16Wrapper::new(
            BruteForceMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_INT8, true) => Box::new(BruteForceMultiI8Wrapper::new(
            BruteForceMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_UINT8, true) => Box::new(BruteForceMultiU8Wrapper::new(
            BruteForceMulti::new(rust_params),
            data_type,
        )),
        // INT32 and INT64 types not yet supported for vector indices
        (VecSimType::VecSimType_INT32, _) | (VecSimType::VecSimType_INT64, _) => return None,
    };

    Some(Box::new(IndexHandle::new(
        wrapper,
        data_type,
        VecSimAlgo::VecSimAlgo_BF,
        metric,
        dim,
        is_multi,
    )))
}

/// Create a new HNSW index.
pub fn create_hnsw_index(params: &HNSWParams) -> Option<Box<IndexHandle>> {
    let rust_params = params.to_rust_params();
    let data_type = params.base.type_;
    let metric = params.base.metric;
    let dim = params.base.dim;
    let is_multi = params.base.multi;

    let wrapper: Box<dyn IndexWrapper> = match (data_type, is_multi) {
        (VecSimType::VecSimType_FLOAT32, false) => Box::new(HnswSingleF32Wrapper::new(
            HnswSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT64, false) => Box::new(HnswSingleF64Wrapper::new(
            HnswSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_BFLOAT16, false) => Box::new(HnswSingleBF16Wrapper::new(
            HnswSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT16, false) => Box::new(HnswSingleFP16Wrapper::new(
            HnswSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_INT8, false) => Box::new(HnswSingleI8Wrapper::new(
            HnswSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_UINT8, false) => Box::new(HnswSingleU8Wrapper::new(
            HnswSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT32, true) => Box::new(HnswMultiF32Wrapper::new(
            HnswMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT64, true) => Box::new(HnswMultiF64Wrapper::new(
            HnswMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_BFLOAT16, true) => Box::new(HnswMultiBF16Wrapper::new(
            HnswMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT16, true) => Box::new(HnswMultiFP16Wrapper::new(
            HnswMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_INT8, true) => Box::new(HnswMultiI8Wrapper::new(
            HnswMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_UINT8, true) => Box::new(HnswMultiU8Wrapper::new(
            HnswMulti::new(rust_params),
            data_type,
        )),
        // INT32 and INT64 types not yet supported for vector indices
        (VecSimType::VecSimType_INT32, _) | (VecSimType::VecSimType_INT64, _) => return None,
    };

    Some(Box::new(IndexHandle::new(
        wrapper,
        data_type,
        VecSimAlgo::VecSimAlgo_HNSWLIB,
        metric,
        dim,
        is_multi,
    )))
}

/// Create a new SVS index.
pub fn create_svs_index(params: &SVSParams) -> Option<Box<IndexHandle>> {
    let rust_params = params.to_rust_params();
    let data_type = params.base.type_;
    let metric = params.base.metric;
    let dim = params.base.dim;
    let is_multi = params.base.multi;

    let wrapper: Box<dyn IndexWrapper> = match (data_type, is_multi) {
        (VecSimType::VecSimType_FLOAT32, false) => Box::new(SvsSingleF32Wrapper::new(
            SvsSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT64, false) => Box::new(SvsSingleF64Wrapper::new(
            SvsSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_BFLOAT16, false) => Box::new(SvsSingleBF16Wrapper::new(
            SvsSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT16, false) => Box::new(SvsSingleFP16Wrapper::new(
            SvsSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_INT8, false) => Box::new(SvsSingleI8Wrapper::new(
            SvsSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_UINT8, false) => Box::new(SvsSingleU8Wrapper::new(
            SvsSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT32, true) => Box::new(SvsMultiF32Wrapper::new(
            SvsMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT64, true) => Box::new(SvsMultiF64Wrapper::new(
            SvsMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_BFLOAT16, true) => Box::new(SvsMultiBF16Wrapper::new(
            SvsMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT16, true) => Box::new(SvsMultiFP16Wrapper::new(
            SvsMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_INT8, true) => Box::new(SvsMultiI8Wrapper::new(
            SvsMulti::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_UINT8, true) => Box::new(SvsMultiU8Wrapper::new(
            SvsMulti::new(rust_params),
            data_type,
        )),
        // INT32 and INT64 types not yet supported for vector indices
        (VecSimType::VecSimType_INT32, _) | (VecSimType::VecSimType_INT64, _) => return None,
    };

    Some(Box::new(IndexHandle::new(
        wrapper,
        data_type,
        VecSimAlgo::VecSimAlgo_SVS,
        metric,
        dim,
        is_multi,
    )))
}

/// Create a new Tiered index.
pub fn create_tiered_index(params: &TieredParams) -> Option<Box<IndexHandle>> {
    let rust_params = params.to_rust_params();
    let data_type = params.base.type_;
    let metric = params.base.metric;
    let dim = params.base.dim;
    let is_multi = params.base.multi;

    // Tiered index currently only supports f32
    let wrapper: Box<dyn IndexWrapper> = match (data_type, is_multi) {
        (VecSimType::VecSimType_FLOAT32, false) => Box::new(TieredSingleF32Wrapper::new(
            TieredSingle::new(rust_params),
            data_type,
        )),
        (VecSimType::VecSimType_FLOAT32, true) => Box::new(TieredMultiF32Wrapper::new(
            TieredMulti::new(rust_params),
            data_type,
        )),
        _ => return None, // Tiered only supports f32 currently
    };

    Some(Box::new(IndexHandle::new(
        wrapper,
        data_type,
        VecSimAlgo::VecSimAlgo_TIERED,
        metric,
        dim,
        is_multi,
    )))
}

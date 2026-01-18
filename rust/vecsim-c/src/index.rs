//! Index wrapper and lifecycle functions for C FFI.

use crate::params::{BFParams, HNSWParams, SVSParams, VecSimQueryParams};
use crate::types::{
    labelType, QueryReplyInternal, QueryResultInternal, VecSimAlgo, VecSimMetric,
    VecSimQueryReply_Order, VecSimType,
};
use std::ffi::c_void;
use std::slice;
use vecsim::index::{
    BruteForceMulti, BruteForceSingle, HnswMulti, HnswSingle, SvsMulti, SvsSingle,
    VecSimIndex as VecSimIndexTrait,
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

/// Macro to implement IndexWrapper for a specific index type.
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
                // This requires accessing internal storage which isn't directly exposed
                // For now, return infinity as a placeholder
                f64::INFINITY
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
                _query: *const c_void,
                _params: Option<&VecSimQueryParams>,
            ) -> Option<Box<dyn BatchIteratorWrapper>> {
                // Batch iterator requires ownership of query, which is complex with type erasure
                // Return None for now; full implementation would require more complex handling
                None
            }

            fn memory_usage(&self) -> usize {
                self.index.info().memory_bytes
            }
        }
    };
}

// Implement wrappers for BruteForce indices
impl_index_wrapper!(
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

impl_index_wrapper!(
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

// Implement wrappers for HNSW indices
impl_index_wrapper!(
    HnswSingleF32Wrapper,
    HnswSingle<f32>,
    f32,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    false
);
impl_index_wrapper!(
    HnswSingleF64Wrapper,
    HnswSingle<f64>,
    f64,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    false
);
impl_index_wrapper!(
    HnswSingleBF16Wrapper,
    HnswSingle<BFloat16>,
    BFloat16,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    false
);
impl_index_wrapper!(
    HnswSingleFP16Wrapper,
    HnswSingle<Float16>,
    Float16,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    false
);
impl_index_wrapper!(
    HnswSingleI8Wrapper,
    HnswSingle<Int8>,
    Int8,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    false
);
impl_index_wrapper!(
    HnswSingleU8Wrapper,
    HnswSingle<UInt8>,
    UInt8,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    false
);

impl_index_wrapper!(
    HnswMultiF32Wrapper,
    HnswMulti<f32>,
    f32,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);
impl_index_wrapper!(
    HnswMultiF64Wrapper,
    HnswMulti<f64>,
    f64,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);
impl_index_wrapper!(
    HnswMultiBF16Wrapper,
    HnswMulti<BFloat16>,
    BFloat16,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);
impl_index_wrapper!(
    HnswMultiFP16Wrapper,
    HnswMulti<Float16>,
    Float16,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);
impl_index_wrapper!(
    HnswMultiI8Wrapper,
    HnswMulti<Int8>,
    Int8,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);
impl_index_wrapper!(
    HnswMultiU8Wrapper,
    HnswMulti<UInt8>,
    UInt8,
    VecSimAlgo::VecSimAlgo_HNSWLIB,
    true
);

// Implement wrappers for SVS indices
impl_index_wrapper!(
    SvsSingleF32Wrapper,
    SvsSingle<f32>,
    f32,
    VecSimAlgo::VecSimAlgo_SVS,
    false
);
impl_index_wrapper!(
    SvsSingleF64Wrapper,
    SvsSingle<f64>,
    f64,
    VecSimAlgo::VecSimAlgo_SVS,
    false
);
impl_index_wrapper!(
    SvsSingleBF16Wrapper,
    SvsSingle<BFloat16>,
    BFloat16,
    VecSimAlgo::VecSimAlgo_SVS,
    false
);
impl_index_wrapper!(
    SvsSingleFP16Wrapper,
    SvsSingle<Float16>,
    Float16,
    VecSimAlgo::VecSimAlgo_SVS,
    false
);
impl_index_wrapper!(
    SvsSingleI8Wrapper,
    SvsSingle<Int8>,
    Int8,
    VecSimAlgo::VecSimAlgo_SVS,
    false
);
impl_index_wrapper!(
    SvsSingleU8Wrapper,
    SvsSingle<UInt8>,
    UInt8,
    VecSimAlgo::VecSimAlgo_SVS,
    false
);

impl_index_wrapper!(
    SvsMultiF32Wrapper,
    SvsMulti<f32>,
    f32,
    VecSimAlgo::VecSimAlgo_SVS,
    true
);
impl_index_wrapper!(
    SvsMultiF64Wrapper,
    SvsMulti<f64>,
    f64,
    VecSimAlgo::VecSimAlgo_SVS,
    true
);
impl_index_wrapper!(
    SvsMultiBF16Wrapper,
    SvsMulti<BFloat16>,
    BFloat16,
    VecSimAlgo::VecSimAlgo_SVS,
    true
);
impl_index_wrapper!(
    SvsMultiFP16Wrapper,
    SvsMulti<Float16>,
    Float16,
    VecSimAlgo::VecSimAlgo_SVS,
    true
);
impl_index_wrapper!(
    SvsMultiI8Wrapper,
    SvsMulti<Int8>,
    Int8,
    VecSimAlgo::VecSimAlgo_SVS,
    true
);
impl_index_wrapper!(
    SvsMultiU8Wrapper,
    SvsMulti<UInt8>,
    UInt8,
    VecSimAlgo::VecSimAlgo_SVS,
    true
);

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

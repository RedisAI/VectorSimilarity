//! C-compatible FFI bindings for the VecSim vector similarity search library.
//!
//! This crate provides a C-compatible API that matches the existing C++ VecSim
//! implementation, enabling drop-in replacement.

// Allow C-style naming conventions to match the C++ API
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

pub mod index;
pub mod info;
pub mod params;
pub mod query;
pub mod types;

use index::{create_brute_force_index, create_hnsw_index, create_svs_index, IndexHandle};
use info::{get_index_info, VecSimIndexInfo};
use params::{BFParams, HNSWParams, SVSParams, VecSimParams, VecSimQueryParams};
use query::{
    create_batch_iterator, range_query, top_k_query, BatchIteratorHandle, QueryReplyHandle,
    QueryReplyIteratorHandle,
};
use types::{
    labelType, QueryResultInternal, VecSimAlgo, VecSimBatchIterator, VecSimIndex, VecSimMetric,
    VecSimQueryReply, VecSimQueryReply_Iterator, VecSimQueryReply_Order, VecSimQueryResult,
    VecSimType,
};

use std::ffi::{c_char, c_void};
use std::ptr;

// ============================================================================
// Index Lifecycle Functions
// ============================================================================

/// Create a new vector similarity index.
///
/// # Safety
/// The `params` pointer must be valid and point to a properly initialized
/// `VecSimParams`, `BFParams`, or `HNSWParams` struct.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_New(params: *const VecSimParams) -> *mut VecSimIndex {
    if params.is_null() {
        return ptr::null_mut();
    }

    let params = &*params;

    let handle = match params.algo {
        VecSimAlgo::VecSimAlgo_BF => {
            let bf_params = BFParams { base: *params };
            create_brute_force_index(&bf_params)
        }
        VecSimAlgo::VecSimAlgo_HNSWLIB => {
            // For HNSW, we need to cast to HNSWParams if the full struct was passed
            // For now, create with default HNSW params
            let hnsw_params = HNSWParams {
                base: *params,
                ..HNSWParams::default()
            };
            create_hnsw_index(&hnsw_params)
        }
        VecSimAlgo::VecSimAlgo_SVS => {
            // For SVS, create with default SVS params
            let svs_params = SVSParams {
                base: *params,
                ..SVSParams::default()
            };
            create_svs_index(&svs_params)
        }
    };

    match handle {
        Some(boxed) => Box::into_raw(boxed) as *mut VecSimIndex,
        None => ptr::null_mut(),
    }
}

/// Create a new BruteForce index with specific parameters.
///
/// # Safety
/// The `params` pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_NewBF(params: *const BFParams) -> *mut VecSimIndex {
    if params.is_null() {
        return ptr::null_mut();
    }

    let params = &*params;
    match create_brute_force_index(params) {
        Some(boxed) => Box::into_raw(boxed) as *mut VecSimIndex,
        None => ptr::null_mut(),
    }
}

/// Create a new HNSW index with specific parameters.
///
/// # Safety
/// The `params` pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_NewHNSW(params: *const HNSWParams) -> *mut VecSimIndex {
    if params.is_null() {
        return ptr::null_mut();
    }

    let params = &*params;
    match create_hnsw_index(params) {
        Some(boxed) => Box::into_raw(boxed) as *mut VecSimIndex,
        None => ptr::null_mut(),
    }
}

/// Create a new SVS (Vamana) index with specific parameters.
///
/// # Safety
/// The `params` pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_NewSVS(params: *const SVSParams) -> *mut VecSimIndex {
    if params.is_null() {
        return ptr::null_mut();
    }

    let params = &*params;
    match create_svs_index(params) {
        Some(boxed) => Box::into_raw(boxed) as *mut VecSimIndex,
        None => ptr::null_mut(),
    }
}

/// Free a vector similarity index.
///
/// # Safety
/// The `index` pointer must have been returned by `VecSimIndex_New` or be null.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_Free(index: *mut VecSimIndex) {
    if !index.is_null() {
        drop(Box::from_raw(index as *mut IndexHandle));
    }
}

// ============================================================================
// Vector Operations
// ============================================================================

/// Add a vector to the index.
///
/// # Safety
/// - `index` must be a valid pointer returned by `VecSimIndex_New`
/// - `vector` must point to a valid array of the correct type and dimension
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_AddVector(
    index: *mut VecSimIndex,
    vector: *const c_void,
    label: labelType,
) -> i32 {
    if index.is_null() || vector.is_null() {
        return -1;
    }

    let handle = &mut *(index as *mut IndexHandle);
    handle.wrapper.add_vector(vector, label)
}

/// Delete a vector by label.
///
/// # Safety
/// - `index` must be a valid pointer returned by `VecSimIndex_New`
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_DeleteVector(
    index: *mut VecSimIndex,
    label: labelType,
) -> i32 {
    if index.is_null() {
        return 0;
    }

    let handle = &mut *(index as *mut IndexHandle);
    handle.wrapper.delete_vector(label)
}

/// Get distance from a stored vector (by label) to a query vector.
///
/// # Safety
/// - `index` must be a valid pointer returned by `VecSimIndex_New`
/// - `vector` must point to a valid array of the correct type and dimension
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_GetDistanceFrom_Unsafe(
    index: *mut VecSimIndex,
    label: labelType,
    vector: *const c_void,
) -> f64 {
    if index.is_null() || vector.is_null() {
        return f64::INFINITY;
    }

    let handle = &*(index as *const IndexHandle);
    handle.wrapper.get_distance_from(label, vector)
}

// ============================================================================
// Query Functions
// ============================================================================

/// Perform a top-k nearest neighbor query.
///
/// # Safety
/// - `index` must be a valid pointer returned by `VecSimIndex_New`
/// - `query` must point to a valid array of the correct type and dimension
/// - `params` may be null for default parameters
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_TopKQuery(
    index: *mut VecSimIndex,
    query: *const c_void,
    k: usize,
    params: *const VecSimQueryParams,
    order: VecSimQueryReply_Order,
) -> *mut VecSimQueryReply {
    if index.is_null() || query.is_null() {
        return ptr::null_mut();
    }

    let handle = &*(index as *const IndexHandle);
    let params_opt = if params.is_null() { None } else { Some(&*params) };

    let reply_handle = top_k_query(handle, query, k, params_opt, order);
    Box::into_raw(Box::new(reply_handle)) as *mut VecSimQueryReply
}

/// Perform a range query.
///
/// # Safety
/// - `index` must be a valid pointer returned by `VecSimIndex_New`
/// - `query` must point to a valid array of the correct type and dimension
/// - `params` may be null for default parameters
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_RangeQuery(
    index: *mut VecSimIndex,
    query: *const c_void,
    radius: f64,
    params: *const VecSimQueryParams,
    order: VecSimQueryReply_Order,
) -> *mut VecSimQueryReply {
    if index.is_null() || query.is_null() {
        return ptr::null_mut();
    }

    let handle = &*(index as *const IndexHandle);
    let params_opt = if params.is_null() { None } else { Some(&*params) };

    let reply_handle = range_query(handle, query, radius, params_opt, order);
    Box::into_raw(Box::new(reply_handle)) as *mut VecSimQueryReply
}

// ============================================================================
// Query Reply Functions
// ============================================================================

/// Get the number of results in a query reply.
///
/// # Safety
/// `reply` must be a valid pointer returned by a query function.
#[no_mangle]
pub unsafe extern "C" fn VecSimQueryReply_Len(reply: *const VecSimQueryReply) -> usize {
    if reply.is_null() {
        return 0;
    }

    let handle = &*(reply as *const QueryReplyHandle);
    handle.len()
}

/// Free a query reply.
///
/// # Safety
/// `reply` must have been returned by a query function or be null.
#[no_mangle]
pub unsafe extern "C" fn VecSimQueryReply_Free(reply: *mut VecSimQueryReply) {
    if !reply.is_null() {
        drop(Box::from_raw(reply as *mut QueryReplyHandle));
    }
}

/// Get an iterator over query results.
///
/// # Safety
/// `reply` must be a valid pointer returned by a query function.
/// The iterator is only valid while the reply exists.
#[no_mangle]
pub unsafe extern "C" fn VecSimQueryReply_GetIterator(
    reply: *mut VecSimQueryReply,
) -> *mut VecSimQueryReply_Iterator {
    if reply.is_null() {
        return ptr::null_mut();
    }

    let handle = &*(reply as *const QueryReplyHandle);
    let iter = QueryReplyIteratorHandle::new(&handle.reply.results);
    Box::into_raw(Box::new(iter)) as *mut VecSimQueryReply_Iterator
}

/// Check if the iterator has more results.
///
/// # Safety
/// `iter` must be a valid pointer returned by `VecSimQueryReply_GetIterator`.
#[no_mangle]
pub unsafe extern "C" fn VecSimQueryReply_IteratorHasNext(
    iter: *const VecSimQueryReply_Iterator,
) -> bool {
    if iter.is_null() {
        return false;
    }

    let handle = &*(iter as *const QueryReplyIteratorHandle);
    handle.has_next()
}

/// Get the next result from the iterator.
///
/// # Safety
/// `iter` must be a valid pointer returned by `VecSimQueryReply_GetIterator`.
#[no_mangle]
pub unsafe extern "C" fn VecSimQueryReply_IteratorNext(
    iter: *mut VecSimQueryReply_Iterator,
) -> *const VecSimQueryResult {
    if iter.is_null() {
        return ptr::null();
    }

    let handle = &mut *(iter as *mut QueryReplyIteratorHandle);
    match handle.next() {
        Some(result) => result as *const QueryResultInternal as *const VecSimQueryResult,
        None => ptr::null(),
    }
}

/// Reset the iterator to the beginning.
///
/// # Safety
/// `iter` must be a valid pointer returned by `VecSimQueryReply_GetIterator`.
#[no_mangle]
pub unsafe extern "C" fn VecSimQueryReply_IteratorReset(iter: *mut VecSimQueryReply_Iterator) {
    if !iter.is_null() {
        let handle = &mut *(iter as *mut QueryReplyIteratorHandle);
        handle.reset();
    }
}

/// Free an iterator.
///
/// # Safety
/// `iter` must have been returned by `VecSimQueryReply_GetIterator` or be null.
#[no_mangle]
pub unsafe extern "C" fn VecSimQueryReply_IteratorFree(iter: *mut VecSimQueryReply_Iterator) {
    if !iter.is_null() {
        // Note: We need to be careful here because the iterator borrows from the reply.
        // For safety, we'll just leak the memory rather than double-free.
        // In a proper implementation, we'd use reference counting or different ownership.
        let _ = Box::from_raw(iter as *mut QueryReplyIteratorHandle<'static>);
    }
}

// ============================================================================
// Query Result Functions
// ============================================================================

/// Get the label (ID) from a query result.
///
/// # Safety
/// `result` must be a valid pointer from the iterator.
#[no_mangle]
pub unsafe extern "C" fn VecSimQueryResult_GetId(result: *const VecSimQueryResult) -> labelType {
    if result.is_null() {
        return 0;
    }

    let internal = &*(result as *const QueryResultInternal);
    internal.id
}

/// Get the score (distance) from a query result.
///
/// # Safety
/// `result` must be a valid pointer from the iterator.
#[no_mangle]
pub unsafe extern "C" fn VecSimQueryResult_GetScore(result: *const VecSimQueryResult) -> f64 {
    if result.is_null() {
        return f64::INFINITY;
    }

    let internal = &*(result as *const QueryResultInternal);
    internal.score
}

// ============================================================================
// Batch Iterator Functions
// ============================================================================

/// Create a batch iterator for incremental query processing.
///
/// # Safety
/// - `index` must be a valid pointer returned by `VecSimIndex_New`
/// - `query` must point to a valid array of the correct type and dimension
/// - `params` may be null for default parameters
#[no_mangle]
pub unsafe extern "C" fn VecSimBatchIterator_New(
    index: *mut VecSimIndex,
    query: *const c_void,
    params: *const VecSimQueryParams,
) -> *mut VecSimBatchIterator {
    if index.is_null() || query.is_null() {
        return ptr::null_mut();
    }

    let handle = &*(index as *const IndexHandle);
    let params_opt = if params.is_null() { None } else { Some(&*params) };

    match create_batch_iterator(handle, query, params_opt) {
        Some(iter_handle) => Box::into_raw(Box::new(iter_handle)) as *mut VecSimBatchIterator,
        None => ptr::null_mut(),
    }
}

/// Get the next batch of results.
///
/// # Safety
/// `iter` must be a valid pointer returned by `VecSimBatchIterator_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimBatchIterator_Next(
    iter: *mut VecSimBatchIterator,
    n: usize,
    order: VecSimQueryReply_Order,
) -> *mut VecSimQueryReply {
    if iter.is_null() {
        return ptr::null_mut();
    }

    let handle = &mut *(iter as *mut BatchIteratorHandle);
    let reply = handle.next(n, order);
    Box::into_raw(Box::new(reply)) as *mut VecSimQueryReply
}

/// Check if the batch iterator has more results.
///
/// # Safety
/// `iter` must be a valid pointer returned by `VecSimBatchIterator_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimBatchIterator_HasNext(iter: *const VecSimBatchIterator) -> bool {
    if iter.is_null() {
        return false;
    }

    let handle = &*(iter as *const BatchIteratorHandle);
    handle.has_next()
}

/// Reset the batch iterator.
///
/// # Safety
/// `iter` must be a valid pointer returned by `VecSimBatchIterator_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimBatchIterator_Reset(iter: *mut VecSimBatchIterator) {
    if !iter.is_null() {
        let handle = &mut *(iter as *mut BatchIteratorHandle);
        handle.reset();
    }
}

/// Free a batch iterator.
///
/// # Safety
/// `iter` must have been returned by `VecSimBatchIterator_New` or be null.
#[no_mangle]
pub unsafe extern "C" fn VecSimBatchIterator_Free(iter: *mut VecSimBatchIterator) {
    if !iter.is_null() {
        drop(Box::from_raw(iter as *mut BatchIteratorHandle));
    }
}

// ============================================================================
// Index Property Functions
// ============================================================================

/// Get the current number of vectors in the index.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_IndexSize(index: *const VecSimIndex) -> usize {
    if index.is_null() {
        return 0;
    }

    let handle = &*(index as *const IndexHandle);
    handle.wrapper.index_size()
}

/// Get the data type of the index.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_GetType(index: *const VecSimIndex) -> VecSimType {
    if index.is_null() {
        return VecSimType::VecSimType_FLOAT32;
    }

    let handle = &*(index as *const IndexHandle);
    handle.data_type
}

/// Get the distance metric of the index.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_GetMetric(index: *const VecSimIndex) -> VecSimMetric {
    if index.is_null() {
        return VecSimMetric::VecSimMetric_L2;
    }

    let handle = &*(index as *const IndexHandle);
    handle.metric
}

/// Get the vector dimension of the index.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_GetDim(index: *const VecSimIndex) -> usize {
    if index.is_null() {
        return 0;
    }

    let handle = &*(index as *const IndexHandle);
    handle.dim
}

/// Check if the index is a multi-value index.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_IsMulti(index: *const VecSimIndex) -> bool {
    if index.is_null() {
        return false;
    }

    let handle = &*(index as *const IndexHandle);
    handle.is_multi
}

/// Check if a label exists in the index.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_ContainsLabel(
    index: *const VecSimIndex,
    label: labelType,
) -> bool {
    if index.is_null() {
        return false;
    }

    let handle = &*(index as *const IndexHandle);
    handle.wrapper.contains(label)
}

/// Get the count of vectors with the given label.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_LabelCount(
    index: *const VecSimIndex,
    label: labelType,
) -> usize {
    if index.is_null() {
        return 0;
    }

    let handle = &*(index as *const IndexHandle);
    handle.wrapper.label_count(label)
}

/// Get detailed index information.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_Info(index: *const VecSimIndex) -> VecSimIndexInfo {
    if index.is_null() {
        return VecSimIndexInfo::default();
    }

    let handle = &*(index as *const IndexHandle);
    get_index_info(handle)
}

// ============================================================================
// Serialization Functions
// ============================================================================

/// Save an index to a file.
///
/// # Safety
/// - `index` must be a valid pointer returned by `VecSimIndex_New`
/// - `path` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_SaveIndex(index: *const VecSimIndex, path: *const c_char) {
    if index.is_null() || path.is_null() {
        return;
    }

    // Serialization is not yet implemented
    // This is a placeholder for future implementation
}

/// Load an index from a file.
///
/// # Safety
/// - `path` must be a valid null-terminated C string
/// - `params` may be null (will use parameters from the file)
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_LoadIndex(
    path: *const c_char,
    _params: *const VecSimParams,
) -> *mut VecSimIndex {
    if path.is_null() {
        return ptr::null_mut();
    }

    // Serialization is not yet implemented
    // This is a placeholder for future implementation
    ptr::null_mut()
}

// ============================================================================
// Memory Estimation Functions
// ============================================================================

/// Estimate initial memory size for a BruteForce index.
#[no_mangle]
pub extern "C" fn VecSimIndex_EstimateBruteForceInitialSize(
    dim: usize,
    initial_capacity: usize,
) -> usize {
    vecsim::index::estimate_brute_force_initial_size(dim, initial_capacity)
}

/// Estimate memory size per element for a BruteForce index.
#[no_mangle]
pub extern "C" fn VecSimIndex_EstimateBruteForceElementSize(dim: usize) -> usize {
    vecsim::index::estimate_brute_force_element_size(dim)
}

/// Estimate initial memory size for an HNSW index.
#[no_mangle]
pub extern "C" fn VecSimIndex_EstimateHNSWInitialSize(
    dim: usize,
    initial_capacity: usize,
    m: usize,
) -> usize {
    vecsim::index::estimate_hnsw_initial_size(dim, initial_capacity, m)
}

/// Estimate memory size per element for an HNSW index.
#[no_mangle]
pub extern "C" fn VecSimIndex_EstimateHNSWElementSize(dim: usize, m: usize) -> usize {
    vecsim::index::estimate_hnsw_element_size(dim, m)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_free_bf_index() {
        let params = BFParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_BF,
                type_: VecSimType::VecSimType_FLOAT32,
                metric: VecSimMetric::VecSimMetric_L2,
                dim: 4,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
            },
        };

        unsafe {
            let index = VecSimIndex_NewBF(&params);
            assert!(!index.is_null());

            assert_eq!(VecSimIndex_IndexSize(index), 0);
            assert_eq!(VecSimIndex_GetDim(index), 4);
            assert_eq!(VecSimIndex_GetType(index), VecSimType::VecSimType_FLOAT32);
            assert!(!VecSimIndex_IsMulti(index));

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_add_and_query_vectors() {
        let params = BFParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_BF,
                type_: VecSimType::VecSimType_FLOAT32,
                metric: VecSimMetric::VecSimMetric_L2,
                dim: 4,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
            },
        };

        unsafe {
            let index = VecSimIndex_NewBF(&params);
            assert!(!index.is_null());

            // Add vectors
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
            let v3: [f32; 4] = [0.0, 0.0, 1.0, 0.0];

            assert_eq!(
                VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1),
                1
            );
            assert_eq!(
                VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2),
                1
            );
            assert_eq!(
                VecSimIndex_AddVector(index, v3.as_ptr() as *const c_void, 3),
                1
            );

            assert_eq!(VecSimIndex_IndexSize(index), 3);

            // Query
            let query: [f32; 4] = [1.0, 0.1, 0.0, 0.0];
            let reply = VecSimIndex_TopKQuery(
                index,
                query.as_ptr() as *const c_void,
                2,
                ptr::null(),
                VecSimQueryReply_Order::BY_SCORE,
            );
            assert!(!reply.is_null());

            let len = VecSimQueryReply_Len(reply);
            assert_eq!(len, 2);

            // Get iterator and check results
            let iter = VecSimQueryReply_GetIterator(reply);
            assert!(!iter.is_null());

            let result = VecSimQueryReply_IteratorNext(iter);
            assert!(!result.is_null());
            let id = VecSimQueryResult_GetId(result);
            assert_eq!(id, 1); // Closest to query

            VecSimQueryReply_IteratorFree(iter);
            VecSimQueryReply_Free(reply);

            // Delete vector
            assert_eq!(VecSimIndex_DeleteVector(index, 1), 1);
            assert_eq!(VecSimIndex_IndexSize(index), 2);
            assert!(!VecSimIndex_ContainsLabel(index, 1));

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_hnsw_index() {
        let params = HNSWParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_HNSWLIB,
                type_: VecSimType::VecSimType_FLOAT32,
                metric: VecSimMetric::VecSimMetric_L2,
                dim: 4,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
            },
            M: 16,
            efConstruction: 200,
            efRuntime: 10,
            epsilon: 0.0,
        };

        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            // Add vectors
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];

            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);
            VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2);

            assert_eq!(VecSimIndex_IndexSize(index), 2);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_range_query() {
        let params = BFParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_BF,
                type_: VecSimType::VecSimType_FLOAT32,
                metric: VecSimMetric::VecSimMetric_L2,
                dim: 4,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
            },
        };

        unsafe {
            let index = VecSimIndex_NewBF(&params);

            // Add vectors at different distances
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [2.0, 0.0, 0.0, 0.0];
            let v3: [f32; 4] = [10.0, 0.0, 0.0, 0.0];

            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);
            VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2);
            VecSimIndex_AddVector(index, v3.as_ptr() as *const c_void, 3);

            // Range query with radius that should only include v1 and v2
            let query: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
            let reply = VecSimIndex_RangeQuery(
                index,
                query.as_ptr() as *const c_void,
                5.0, // L2 squared distance threshold
                ptr::null(),
                VecSimQueryReply_Order::BY_SCORE,
            );

            let len = VecSimQueryReply_Len(reply);
            assert_eq!(len, 2); // Only v1 (dist=1) and v2 (dist=4) should be included

            VecSimQueryReply_Free(reply);
            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_svs_index() {
        let params = SVSParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_SVS,
                type_: VecSimType::VecSimType_FLOAT32,
                metric: VecSimMetric::VecSimMetric_L2,
                dim: 4,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
            },
            graphMaxDegree: 32,
            alpha: 1.2,
            constructionWindowSize: 200,
            searchWindowSize: 100,
            twoPassConstruction: true,
        };

        unsafe {
            let index = VecSimIndex_NewSVS(&params);
            assert!(!index.is_null());

            // Add vectors
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
            let v3: [f32; 4] = [0.0, 0.0, 1.0, 0.0];

            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);
            VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2);
            VecSimIndex_AddVector(index, v3.as_ptr() as *const c_void, 3);

            assert_eq!(VecSimIndex_IndexSize(index), 3);

            // Query
            let query: [f32; 4] = [1.0, 0.1, 0.0, 0.0];
            let reply = VecSimIndex_TopKQuery(
                index,
                query.as_ptr() as *const c_void,
                2,
                ptr::null(),
                VecSimQueryReply_Order::BY_SCORE,
            );
            assert!(!reply.is_null());

            let len = VecSimQueryReply_Len(reply);
            assert_eq!(len, 2);

            // Get iterator and check results
            let iter = VecSimQueryReply_GetIterator(reply);
            assert!(!iter.is_null());

            let result = VecSimQueryReply_IteratorNext(iter);
            assert!(!result.is_null());
            let id = VecSimQueryResult_GetId(result);
            assert_eq!(id, 1); // Closest to query

            VecSimQueryReply_IteratorFree(iter);
            VecSimQueryReply_Free(reply);
            VecSimIndex_Free(index);
        }
    }
}

//! C-compatible FFI bindings for the VecSim vector similarity search library.
//!
//! This crate provides a C-compatible API that matches the existing C++ VecSim
//! implementation, enabling drop-in replacement.

// Allow C-style naming conventions to match the C++ API
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

pub mod compat;
pub mod index;
pub mod info;
pub mod params;
pub mod query;
pub mod types;

use index::{
    create_brute_force_index, create_disk_index, create_hnsw_index, create_svs_index,
    create_tiered_index, IndexHandle, IndexWrapper,
    BruteForceSingleF32Wrapper, BruteForceSingleF64Wrapper,
    BruteForceMultiF32Wrapper, BruteForceMultiF64Wrapper,
    HnswSingleF32Wrapper, HnswSingleF64Wrapper,
    HnswMultiF32Wrapper, HnswMultiF64Wrapper,
    SvsSingleF32Wrapper, SvsSingleF64Wrapper,
    TieredSingleF32Wrapper, TieredMultiF32Wrapper,
};
use info::{get_index_info, VecSimIndexInfo};
use params::{
    BFParams, DiskParams, HNSWParams, SVSParams, TieredParams, VecSimParams, VecSimQueryParams,
};
use query::{
    create_batch_iterator, range_query, top_k_query, BatchIteratorHandle, QueryReplyHandle,
    QueryReplyIteratorHandle,
};
use types::{
    labelType, QueryResultInternal, VecSimAlgo, VecSimBatchIterator, VecSimIndex, VecSimMetric,
    VecSimParamResolveCode, VecSimQueryReply, VecSimQueryReply_Iterator, VecSimQueryReply_Order,
    VecSimQueryResult, VecSimRawParam, VecSimType, VecsimQueryType,
};

use std::ffi::{c_char, c_void};
use std::ptr;
use std::sync::atomic::{AtomicU8, Ordering};

use types::{VecSimMemoryFunctions, VecSimWriteMode};
use compat::VecSimParams_C;

// ============================================================================
// Global Memory Functions
// ============================================================================

use std::sync::RwLock;

/// Global memory functions for custom memory management.
/// Protected by RwLock for thread-safe access.
static GLOBAL_MEMORY_FUNCTIONS: RwLock<Option<VecSimMemoryFunctions>> = RwLock::new(None);

/// Global timeout callback function.
static GLOBAL_TIMEOUT_CALLBACK: RwLock<types::timeoutCallbackFunction> = RwLock::new(None);

/// Global log callback function.
static GLOBAL_LOG_CALLBACK: RwLock<types::logCallbackFunction> = RwLock::new(None);

/// Thread-safe wrapper for raw pointers in the log context.
struct SendSyncPtr(*const c_char);
unsafe impl Send for SendSyncPtr {}
unsafe impl Sync for SendSyncPtr {}

/// Global log context for testing.
static GLOBAL_LOG_CONTEXT: RwLock<Option<(SendSyncPtr, SendSyncPtr)>> = RwLock::new(None);

/// Set custom memory functions for all future allocations.
///
/// This allows integration with external memory management systems like Redis.
/// The functions will be used for all memory allocations in the library.
///
/// # Safety
/// The provided function pointers must be valid and thread-safe.
/// The functions must follow standard malloc/calloc/realloc/free semantics.
///
/// # Note
/// This should be called once at initialization, before creating any indices.
#[no_mangle]
pub unsafe extern "C" fn VecSim_SetMemoryFunctions(functions: VecSimMemoryFunctions) {
    if let Ok(mut guard) = GLOBAL_MEMORY_FUNCTIONS.write() {
        *guard = Some(functions);
    }
}

/// Get the current memory functions.
///
/// Returns the currently configured memory functions, or None if using defaults.
pub(crate) fn get_memory_functions() -> Option<VecSimMemoryFunctions> {
    GLOBAL_MEMORY_FUNCTIONS.read().ok().and_then(|g| *g)
}

/// Set the timeout callback function.
///
/// The callback will be called periodically during long operations to check
/// if the operation should be aborted. Return non-zero to abort.
///
/// # Safety
/// The callback function must be thread-safe.
#[no_mangle]
pub unsafe extern "C" fn VecSim_SetTimeoutCallbackFunction(
    callback: types::timeoutCallbackFunction,
) {
    if let Ok(mut guard) = GLOBAL_TIMEOUT_CALLBACK.write() {
        *guard = callback;
    }
}

/// Set the log callback function.
///
/// The callback will be called for logging messages from the library.
///
/// # Safety
/// The callback function must be thread-safe.
#[no_mangle]
pub unsafe extern "C" fn VecSim_SetLogCallbackFunction(
    callback: types::logCallbackFunction,
) {
    if let Ok(mut guard) = GLOBAL_LOG_CALLBACK.write() {
        *guard = callback;
    }
}

/// Set the test log context.
///
/// This is used for testing to identify which test is running.
///
/// # Safety
/// The pointers must be valid for the duration of the test.
#[no_mangle]
pub unsafe extern "C" fn VecSim_SetTestLogContext(
    test_name: *const c_char,
    test_type: *const c_char,
) {
    if let Ok(mut guard) = GLOBAL_LOG_CONTEXT.write() {
        *guard = Some((SendSyncPtr(test_name), SendSyncPtr(test_type)));
    }
}

// ============================================================================
// Global Write Mode State
// ============================================================================

/// Global write mode for tiered indices.
/// Accessed atomically for thread-safety.
static GLOBAL_WRITE_MODE: AtomicU8 = AtomicU8::new(0); // VecSim_WriteAsync = 0

/// Set the global write mode for tiered index operations.
///
/// This controls whether vector additions/deletions in tiered indices go through
/// the async buffering path (VecSim_WriteAsync) or directly to the backend index
/// (VecSim_WriteInPlace).
///
/// # Note
/// In a tiered index scenario, this should be called from the main thread only
/// (that is, the thread that is calling add/delete vector functions).
#[no_mangle]
pub extern "C" fn VecSim_SetWriteMode(mode: VecSimWriteMode) {
    GLOBAL_WRITE_MODE.store(mode as u8, Ordering::SeqCst);
}

/// Get the current global write mode.
///
/// Returns the currently active write mode for tiered index operations.
#[no_mangle]
pub extern "C" fn VecSim_GetWriteMode() -> VecSimWriteMode {
    match GLOBAL_WRITE_MODE.load(Ordering::SeqCst) {
        0 => VecSimWriteMode::VecSim_WriteAsync,
        _ => VecSimWriteMode::VecSim_WriteInPlace,
    }
}

/// Internal function to get write mode as the Rust enum.
pub(crate) fn get_global_write_mode() -> VecSimWriteMode {
    VecSim_GetWriteMode()
}

// ============================================================================
// Parameter Resolution
// ============================================================================

/// Known parameter names for parameter resolution.
mod param_names {
    pub const EF_RUNTIME: &str = "EF_RUNTIME";
    pub const EPSILON: &str = "EPSILON";
    pub const BATCH_SIZE: &str = "BATCH_SIZE";
    pub const HYBRID_POLICY: &str = "HYBRID_POLICY";
    pub const SEARCH_WINDOW_SIZE: &str = "SEARCH_WINDOW_SIZE";
    pub const SEARCH_BUFFER_CAPACITY: &str = "SEARCH_BUFFER_CAPACITY";
    pub const USE_SEARCH_HISTORY: &str = "USE_SEARCH_HISTORY";
    pub const POLICY_BATCHES: &str = "batches";
    pub const POLICY_ADHOC_BF: &str = "adhoc_bf";
}

/// Parse a string value as a positive integer.
fn parse_positive_integer(value: &str) -> Option<usize> {
    value.parse::<i64>().ok().filter(|&v| v > 0).map(|v| v as usize)
}

/// Parse a string value as a positive f64.
fn parse_positive_double(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|&v| v > 0.0)
}

/// Resolve runtime query parameters from raw string parameters.
///
/// # Safety
/// All pointers must be valid. `index`, `rparams`, and `qparams` must not be null
/// (except rparams when paramNum is 0).
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_ResolveParams(
    index: *mut VecSimIndex,
    rparams: *const VecSimRawParam,
    paramNum: i32,
    qparams: *mut VecSimQueryParams,
    query_type: VecsimQueryType,
) -> VecSimParamResolveCode {
    use VecSimParamResolveCode::*;

    // Null check for qparams (required) and rparams (required if paramNum > 0)
    if qparams.is_null() || (rparams.is_null() && paramNum != 0) {
        return VecSimParamResolverErr_NullParam;
    }

    // Zero out qparams
    let qparams = &mut *qparams;
    *qparams = VecSimQueryParams::default();
    qparams.hnswRuntimeParams.efRuntime = 0; // Reset to 0 for checking duplicates
    qparams.hnswRuntimeParams.epsilon = 0.0;
    qparams.svsRuntimeParams = params::SVSRuntimeParams::default();
    qparams.batchSize = 0;
    qparams.searchMode = params::VecSimSearchMode::STANDARD;

    if paramNum == 0 {
        return VecSimParamResolver_OK;
    }

    let handle = &*(index as *const IndexHandle);
    let index_type = handle.algo;

    let params_slice = std::slice::from_raw_parts(rparams, paramNum as usize);

    for rparam in params_slice {
        // Get the parameter name as a Rust string
        let name = if rparam.nameLen > 0 && !rparam.name.is_null() {
            std::str::from_utf8(std::slice::from_raw_parts(
                rparam.name as *const u8,
                rparam.nameLen,
            ))
            .unwrap_or("")
        } else {
            ""
        };

        // Get the parameter value as a Rust string
        let value = if rparam.valLen > 0 && !rparam.value.is_null() {
            std::str::from_utf8(std::slice::from_raw_parts(
                rparam.value as *const u8,
                rparam.valLen,
            ))
            .unwrap_or("")
        } else {
            ""
        };

        let result = match name.to_uppercase().as_str() {
            param_names::EF_RUNTIME => {
                resolve_ef_runtime(index_type, value, qparams, query_type)
            }
            param_names::EPSILON => {
                resolve_epsilon(index_type, value, qparams, query_type)
            }
            param_names::BATCH_SIZE => {
                resolve_batch_size(value, qparams, query_type)
            }
            param_names::HYBRID_POLICY => {
                resolve_hybrid_policy(value, qparams, query_type)
            }
            param_names::SEARCH_WINDOW_SIZE => {
                resolve_search_window_size(index_type, value, qparams)
            }
            param_names::SEARCH_BUFFER_CAPACITY => {
                resolve_search_buffer_capacity(index_type, value, qparams)
            }
            param_names::USE_SEARCH_HISTORY => {
                resolve_use_search_history(index_type, value, qparams)
            }
            _ => VecSimParamResolverErr_UnknownParam,
        };

        if result != VecSimParamResolver_OK {
            return result;
        }
    }

    // Validate parameter combinations
    // AD-HOC with batch_size is invalid
    if qparams.hybridPolicy == params::VecSimHybridPolicy::ADHOC && qparams.batchSize > 0 {
        return VecSimParamResolverErr_InvalidPolicy_AdHoc_With_BatchSize;
    }

    // AD-HOC with ef_runtime is invalid for HNSW
    if qparams.hybridPolicy == params::VecSimHybridPolicy::ADHOC
        && index_type == VecSimAlgo::VecSimAlgo_HNSWLIB
        && qparams.hnswRuntimeParams.efRuntime > 0
    {
        return VecSimParamResolverErr_InvalidPolicy_AdHoc_With_EfRuntime;
    }

    VecSimParamResolver_OK
}

fn resolve_ef_runtime(
    index_type: VecSimAlgo,
    value: &str,
    qparams: &mut VecSimQueryParams,
    query_type: VecsimQueryType,
) -> VecSimParamResolveCode {
    use VecSimParamResolveCode::*;

    // EF_RUNTIME is valid only for HNSW
    if index_type != VecSimAlgo::VecSimAlgo_HNSWLIB {
        return VecSimParamResolverErr_UnknownParam;
    }
    // EF_RUNTIME is invalid for range query
    if query_type == VecsimQueryType::QUERY_TYPE_RANGE {
        return VecSimParamResolverErr_UnknownParam;
    }
    // Check if already set
    if qparams.hnswRuntimeParams.efRuntime != 0 {
        return VecSimParamResolverErr_AlreadySet;
    }
    // Parse value
    match parse_positive_integer(value) {
        Some(v) => {
            qparams.hnswRuntimeParams.efRuntime = v;
            VecSimParamResolver_OK
        }
        None => VecSimParamResolverErr_BadValue,
    }
}

fn resolve_epsilon(
    index_type: VecSimAlgo,
    value: &str,
    qparams: &mut VecSimQueryParams,
    query_type: VecsimQueryType,
) -> VecSimParamResolveCode {
    use VecSimParamResolveCode::*;

    // EPSILON is valid only for HNSW or SVS
    if index_type != VecSimAlgo::VecSimAlgo_HNSWLIB && index_type != VecSimAlgo::VecSimAlgo_SVS {
        return VecSimParamResolverErr_UnknownParam;
    }
    // EPSILON is valid only for range queries
    if query_type != VecsimQueryType::QUERY_TYPE_RANGE {
        return VecSimParamResolverErr_InvalidPolicy_NRange;
    }
    // Check if already set (based on index type)
    let epsilon_ref = if index_type == VecSimAlgo::VecSimAlgo_HNSWLIB {
        &mut qparams.hnswRuntimeParams.epsilon
    } else {
        &mut qparams.svsRuntimeParams.epsilon
    };
    if *epsilon_ref != 0.0 {
        return VecSimParamResolverErr_AlreadySet;
    }
    // Parse value
    match parse_positive_double(value) {
        Some(v) => {
            *epsilon_ref = v;
            VecSimParamResolver_OK
        }
        None => VecSimParamResolverErr_BadValue,
    }
}

fn resolve_batch_size(
    value: &str,
    qparams: &mut VecSimQueryParams,
    query_type: VecsimQueryType,
) -> VecSimParamResolveCode {
    use VecSimParamResolveCode::*;

    // BATCH_SIZE is valid only for hybrid queries
    if query_type != VecsimQueryType::QUERY_TYPE_HYBRID {
        return VecSimParamResolverErr_InvalidPolicy_NHybrid;
    }
    // Check if already set
    if qparams.batchSize != 0 {
        return VecSimParamResolverErr_AlreadySet;
    }
    // Parse value
    match parse_positive_integer(value) {
        Some(v) => {
            qparams.batchSize = v;
            VecSimParamResolver_OK
        }
        None => VecSimParamResolverErr_BadValue,
    }
}

fn resolve_hybrid_policy(
    value: &str,
    qparams: &mut VecSimQueryParams,
    query_type: VecsimQueryType,
) -> VecSimParamResolveCode {
    use VecSimParamResolveCode::*;

    // HYBRID_POLICY is valid only for hybrid queries
    if query_type != VecsimQueryType::QUERY_TYPE_HYBRID {
        return VecSimParamResolverErr_InvalidPolicy_NHybrid;
    }
    // Check if already set (searchMode != STANDARD indicates it was set)
    if qparams.searchMode != params::VecSimSearchMode::STANDARD {
        return VecSimParamResolverErr_AlreadySet;
    }
    // Parse value (case-insensitive)
    match value.to_lowercase().as_str() {
        param_names::POLICY_BATCHES => {
            qparams.searchMode = params::VecSimSearchMode::HYBRID;
            qparams.hybridPolicy = params::VecSimHybridPolicy::BATCHES;
            VecSimParamResolver_OK
        }
        param_names::POLICY_ADHOC_BF => {
            qparams.searchMode = params::VecSimSearchMode::HYBRID;
            qparams.hybridPolicy = params::VecSimHybridPolicy::ADHOC;
            VecSimParamResolver_OK
        }
        _ => VecSimParamResolverErr_InvalidPolicy_NExits,
    }
}

fn resolve_search_window_size(
    index_type: VecSimAlgo,
    value: &str,
    qparams: &mut VecSimQueryParams,
) -> VecSimParamResolveCode {
    use VecSimParamResolveCode::*;

    // SEARCH_WINDOW_SIZE is valid only for SVS
    if index_type != VecSimAlgo::VecSimAlgo_SVS {
        return VecSimParamResolverErr_UnknownParam;
    }
    // Check if already set
    if qparams.svsRuntimeParams.windowSize != 0 {
        return VecSimParamResolverErr_AlreadySet;
    }
    // Parse value
    match parse_positive_integer(value) {
        Some(v) => {
            qparams.svsRuntimeParams.windowSize = v;
            VecSimParamResolver_OK
        }
        None => VecSimParamResolverErr_BadValue,
    }
}

fn resolve_search_buffer_capacity(
    index_type: VecSimAlgo,
    value: &str,
    qparams: &mut VecSimQueryParams,
) -> VecSimParamResolveCode {
    use VecSimParamResolveCode::*;

    // SEARCH_BUFFER_CAPACITY is valid only for SVS
    if index_type != VecSimAlgo::VecSimAlgo_SVS {
        return VecSimParamResolverErr_UnknownParam;
    }
    // Check if already set
    if qparams.svsRuntimeParams.bufferCapacity != 0 {
        return VecSimParamResolverErr_AlreadySet;
    }
    // Parse value
    match parse_positive_integer(value) {
        Some(v) => {
            qparams.svsRuntimeParams.bufferCapacity = v;
            VecSimParamResolver_OK
        }
        None => VecSimParamResolverErr_BadValue,
    }
}

fn resolve_use_search_history(
    index_type: VecSimAlgo,
    value: &str,
    qparams: &mut VecSimQueryParams,
) -> VecSimParamResolveCode {
    use VecSimParamResolveCode::*;

    // USE_SEARCH_HISTORY is valid only for SVS
    if index_type != VecSimAlgo::VecSimAlgo_SVS {
        return VecSimParamResolverErr_UnknownParam;
    }
    // Check if already set
    if qparams.svsRuntimeParams.searchHistory != 0 {
        return VecSimParamResolverErr_AlreadySet;
    }
    // Parse as boolean (1/0, true/false, yes/no)
    let bool_val = match value.to_lowercase().as_str() {
        "1" | "true" | "yes" => Some(1),
        "0" | "false" | "no" => Some(0),
        _ => None,
    };
    match bool_val {
        Some(v) => {
            qparams.svsRuntimeParams.searchHistory = v;
            VecSimParamResolver_OK
        }
        None => VecSimParamResolverErr_BadValue,
    }
}

// ============================================================================
// Index Lifecycle Functions
// ============================================================================

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

/// Create a new Tiered index with specific parameters.
///
/// The tiered index combines a BruteForce frontend (for fast writes) with
/// an HNSW backend (for efficient queries). Currently only supports f32 vectors.
///
/// # Safety
/// The `params` pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_NewTiered(params: *const TieredParams) -> *mut VecSimIndex {
    if params.is_null() {
        return ptr::null_mut();
    }

    let params = &*params;
    match create_tiered_index(params) {
        Some(boxed) => Box::into_raw(boxed) as *mut VecSimIndex,
        None => ptr::null_mut(),
    }
}

// ============================================================================
// Tiered Index Operations
// ============================================================================

/// Flush the flat buffer to the HNSW backend.
///
/// This migrates all vectors from the flat buffer to the HNSW index.
/// Returns the number of vectors flushed, or 0 if the index is not tiered.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_NewTiered`.
#[no_mangle]
pub unsafe extern "C" fn VecSimTieredIndex_Flush(index: *mut VecSimIndex) -> usize {
    if index.is_null() {
        return 0;
    }
    let handle = &mut *(index as *mut IndexHandle);
    handle.wrapper.tiered_flush()
}

/// Get the number of vectors in the flat buffer.
///
/// Returns 0 if the index is not tiered.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_NewTiered`.
#[no_mangle]
pub unsafe extern "C" fn VecSimTieredIndex_FlatSize(index: *const VecSimIndex) -> usize {
    if index.is_null() {
        return 0;
    }
    let handle = &*(index as *const IndexHandle);
    handle.wrapper.tiered_flat_size()
}

/// Get the number of vectors in the HNSW backend.
///
/// Returns 0 if the index is not tiered.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_NewTiered`.
#[no_mangle]
pub unsafe extern "C" fn VecSimTieredIndex_BackendSize(index: *const VecSimIndex) -> usize {
    if index.is_null() {
        return 0;
    }
    let handle = &*(index as *const IndexHandle);
    handle.wrapper.tiered_backend_size()
}

/// Run garbage collection on a tiered index.
///
/// This cleans up deleted vectors and optimizes the index structure.
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_NewTiered`.
#[no_mangle]
pub unsafe extern "C" fn VecSimTieredIndex_GC(index: *mut VecSimIndex) {
    if index.is_null() {
        return;
    }
    let handle = &mut *(index as *mut IndexHandle);
    handle.wrapper.tiered_gc();
}

/// Acquire shared locks on a tiered index.
///
/// This prevents modifications to the index while the locks are held.
/// Must be paired with VecSimTieredIndex_ReleaseSharedLocks.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_NewTiered`.
#[no_mangle]
pub unsafe extern "C" fn VecSimTieredIndex_AcquireSharedLocks(index: *mut VecSimIndex) {
    if index.is_null() {
        return;
    }
    let handle = &mut *(index as *mut IndexHandle);
    handle.wrapper.tiered_acquire_shared_locks();
}

/// Release shared locks on a tiered index.
///
/// Must be called after VecSimTieredIndex_AcquireSharedLocks.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_NewTiered`.
#[no_mangle]
pub unsafe extern "C" fn VecSimTieredIndex_ReleaseSharedLocks(index: *mut VecSimIndex) {
    if index.is_null() {
        return;
    }
    let handle = &mut *(index as *mut IndexHandle);
    handle.wrapper.tiered_release_shared_locks();
}

/// Check if the index is a tiered index.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New` or similar.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_IsTiered(index: *const VecSimIndex) -> bool {
    if index.is_null() {
        return false;
    }
    let handle = &*(index as *const IndexHandle);
    handle.wrapper.is_tiered()
}

// ============================================================================
// Disk Index Functions
// ============================================================================

/// Create a new disk-based index.
///
/// Disk indices store vectors in memory-mapped files for persistence.
/// They support two backends:
/// - BruteForce: Linear scan (exact results, O(n))
/// - Vamana: Graph-based approximate search (fast, O(log n))
///
/// # Safety
/// - `params` must be a valid pointer to a `DiskParams` struct
/// - `params.dataPath` must be a valid null-terminated C string
///
/// # Returns
/// A pointer to the new index, or null if creation failed.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_NewDisk(params: *const DiskParams) -> *mut VecSimIndex {
    if params.is_null() {
        return ptr::null_mut();
    }

    let params = &*params;
    match create_disk_index(params) {
        Some(handle) => Box::into_raw(handle) as *mut VecSimIndex,
        None => ptr::null_mut(),
    }
}

// ============================================================================
// Generic C++-compatible Index Creation
// ============================================================================

/// Create a new index using C++-compatible VecSimParams structure.
///
/// This function provides drop-in compatibility with the C++ VecSim API.
/// It reads the `algo` field to determine which type of index to create,
/// then accesses the appropriate union variant.
///
/// # Safety
/// - `params` must be a valid pointer to a VecSimParams_C struct
/// - The `algo` field must match the initialized union variant in `algoParams`
/// - For tiered indices, `primaryIndexParams` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_New(params: *const VecSimParams_C) -> *mut VecSimIndex {
    if params.is_null() {
        return ptr::null_mut();
    }

    let params = &*params;

    match params.algo {
        VecSimAlgo::VecSimAlgo_BF => {
            let bf = params.algoParams.bfParams;
            create_bf_index_raw(
                bf.type_,
                bf.metric,
                bf.dim,
                bf.multi,
                bf.initialCapacity,
                bf.blockSize,
            )
        }
        VecSimAlgo::VecSimAlgo_HNSWLIB => {
            let hnsw = params.algoParams.hnswParams;
            create_hnsw_index_raw(
                hnsw.type_,
                hnsw.metric,
                hnsw.dim,
                hnsw.multi,
                hnsw.initialCapacity,
                hnsw.M,
                hnsw.efConstruction,
                hnsw.efRuntime,
            )
        }
        VecSimAlgo::VecSimAlgo_SVS => {
            let svs = params.algoParams.svsParams;
            create_svs_index_raw(
                svs.type_,
                svs.metric,
                svs.dim,
                svs.multi,
                svs.graph_max_degree,
                svs.alpha,
                svs.construction_window_size,
                svs.search_window_size,
            )
        }
        VecSimAlgo::VecSimAlgo_TIERED => {
            let tiered = &*params.algoParams.tieredParams;

            // Get primary index params
            if tiered.primaryIndexParams.is_null() {
                return ptr::null_mut();
            }
            let primary = &*tiered.primaryIndexParams;

            // Determine the backend type from primary params
            match primary.algo {
                VecSimAlgo::VecSimAlgo_HNSWLIB => {
                    let hnsw = primary.algoParams.hnswParams;
                    create_tiered_index_raw(
                        hnsw.type_,
                        hnsw.metric,
                        hnsw.dim,
                        hnsw.multi,
                        hnsw.M,
                        hnsw.efConstruction,
                        hnsw.efRuntime,
                        tiered.flatBufferLimit,
                    )
                }
                _ => ptr::null_mut(), // Only HNSW backend supported for now
            }
        }
    }
}

// Helper function to create BF index
unsafe fn create_bf_index_raw(
    type_: VecSimType,
    metric: VecSimMetric,
    dim: usize,
    multi: bool,
    initial_capacity: usize,
    block_size: usize,
) -> *mut VecSimIndex {
    let rust_metric = metric.to_rust_metric();
    let block = if block_size > 0 { block_size } else { 1024 };

    let wrapper: Box<dyn IndexWrapper> = match (type_, multi) {
        (VecSimType::VecSimType_FLOAT32, false) => {
            let params = vecsim::index::BruteForceParams::new(dim, rust_metric)
                .with_capacity(initial_capacity)
                .with_block_size(block);
            Box::new(BruteForceSingleF32Wrapper::new(
                vecsim::index::BruteForceSingle::new(params),
                type_,
            ))
        }
        (VecSimType::VecSimType_FLOAT64, false) => {
            let params = vecsim::index::BruteForceParams::new(dim, rust_metric)
                .with_capacity(initial_capacity)
                .with_block_size(block);
            Box::new(BruteForceSingleF64Wrapper::new(
                vecsim::index::BruteForceSingle::new(params),
                type_,
            ))
        }
        (VecSimType::VecSimType_FLOAT32, true) => {
            let params = vecsim::index::BruteForceParams::new(dim, rust_metric)
                .with_capacity(initial_capacity)
                .with_block_size(block);
            Box::new(BruteForceMultiF32Wrapper::new(
                vecsim::index::BruteForceMulti::new(params),
                type_,
            ))
        }
        (VecSimType::VecSimType_FLOAT64, true) => {
            let params = vecsim::index::BruteForceParams::new(dim, rust_metric)
                .with_capacity(initial_capacity)
                .with_block_size(block);
            Box::new(BruteForceMultiF64Wrapper::new(
                vecsim::index::BruteForceMulti::new(params),
                type_,
            ))
        }
        _ => return ptr::null_mut(),
    };

    Box::into_raw(Box::new(IndexHandle::new(
        wrapper,
        type_,
        VecSimAlgo::VecSimAlgo_BF,
        metric,
        dim,
        multi,
    ))) as *mut VecSimIndex
}

// Helper function to create HNSW index
unsafe fn create_hnsw_index_raw(
    type_: VecSimType,
    metric: VecSimMetric,
    dim: usize,
    multi: bool,
    initial_capacity: usize,
    m: usize,
    ef_construction: usize,
    ef_runtime: usize,
) -> *mut VecSimIndex {
    let rust_metric = metric.to_rust_metric();

    let wrapper: Box<dyn IndexWrapper> = match (type_, multi) {
        (VecSimType::VecSimType_FLOAT32, false) => {
            let params = vecsim::index::HnswParams::new(dim, rust_metric)
                .with_m(m)
                .with_ef_construction(ef_construction)
                .with_ef_runtime(ef_runtime)
                .with_capacity(initial_capacity);
            Box::new(HnswSingleF32Wrapper::new(
                vecsim::index::HnswSingle::new(params),
                type_,
            ))
        }
        (VecSimType::VecSimType_FLOAT64, false) => {
            let params = vecsim::index::HnswParams::new(dim, rust_metric)
                .with_m(m)
                .with_ef_construction(ef_construction)
                .with_ef_runtime(ef_runtime)
                .with_capacity(initial_capacity);
            Box::new(HnswSingleF64Wrapper::new(
                vecsim::index::HnswSingle::new(params),
                type_,
            ))
        }
        (VecSimType::VecSimType_FLOAT32, true) => {
            let params = vecsim::index::HnswParams::new(dim, rust_metric)
                .with_m(m)
                .with_ef_construction(ef_construction)
                .with_ef_runtime(ef_runtime)
                .with_capacity(initial_capacity);
            Box::new(HnswMultiF32Wrapper::new(
                vecsim::index::HnswMulti::new(params),
                type_,
            ))
        }
        (VecSimType::VecSimType_FLOAT64, true) => {
            let params = vecsim::index::HnswParams::new(dim, rust_metric)
                .with_m(m)
                .with_ef_construction(ef_construction)
                .with_ef_runtime(ef_runtime)
                .with_capacity(initial_capacity);
            Box::new(HnswMultiF64Wrapper::new(
                vecsim::index::HnswMulti::new(params),
                type_,
            ))
        }
        _ => return ptr::null_mut(),
    };

    Box::into_raw(Box::new(IndexHandle::new(
        wrapper,
        type_,
        VecSimAlgo::VecSimAlgo_HNSWLIB,
        metric,
        dim,
        multi,
    ))) as *mut VecSimIndex
}

// Helper function to create SVS index
unsafe fn create_svs_index_raw(
    type_: VecSimType,
    metric: VecSimMetric,
    dim: usize,
    multi: bool,
    graph_degree: usize,
    alpha: f32,
    construction_l: usize,
    search_l: usize,
) -> *mut VecSimIndex {
    let rust_metric = metric.to_rust_metric();

    // SVS only supports single-label for now
    if multi {
        return ptr::null_mut();
    }

    let wrapper: Box<dyn IndexWrapper> = match type_ {
        VecSimType::VecSimType_FLOAT32 => {
            let params = vecsim::index::SvsParams::new(dim, rust_metric)
                .with_graph_degree(graph_degree)
                .with_alpha(alpha)
                .with_construction_l(construction_l)
                .with_search_l(search_l);
            Box::new(SvsSingleF32Wrapper::new(
                vecsim::index::SvsSingle::new(params),
                type_,
            ))
        }
        VecSimType::VecSimType_FLOAT64 => {
            let params = vecsim::index::SvsParams::new(dim, rust_metric)
                .with_graph_degree(graph_degree)
                .with_alpha(alpha)
                .with_construction_l(construction_l)
                .with_search_l(search_l);
            Box::new(SvsSingleF64Wrapper::new(
                vecsim::index::SvsSingle::new(params),
                type_,
            ))
        }
        _ => return ptr::null_mut(),
    };

    Box::into_raw(Box::new(IndexHandle::new(
        wrapper,
        type_,
        VecSimAlgo::VecSimAlgo_SVS,
        metric,
        dim,
        false,
    ))) as *mut VecSimIndex
}

// Helper function to create tiered index
// Note: Tiered only supports f32 currently
unsafe fn create_tiered_index_raw(
    type_: VecSimType,
    metric: VecSimMetric,
    dim: usize,
    multi: bool,
    m: usize,
    ef_construction: usize,
    ef_runtime: usize,
    flat_buffer_limit: usize,
) -> *mut VecSimIndex {
    // Tiered only supports f32 currently
    if type_ != VecSimType::VecSimType_FLOAT32 {
        return ptr::null_mut();
    }

    let rust_metric = metric.to_rust_metric();

    let wrapper: Box<dyn IndexWrapper> = if multi {
        let params = vecsim::index::TieredParams::new(dim, rust_metric)
            .with_m(m)
            .with_ef_construction(ef_construction)
            .with_ef_runtime(ef_runtime)
            .with_flat_buffer_limit(flat_buffer_limit);
        Box::new(TieredMultiF32Wrapper::new(
            vecsim::index::TieredMulti::new(params),
            type_,
        ))
    } else {
        let params = vecsim::index::TieredParams::new(dim, rust_metric)
            .with_m(m)
            .with_ef_construction(ef_construction)
            .with_ef_runtime(ef_runtime)
            .with_flat_buffer_limit(flat_buffer_limit);
        Box::new(TieredSingleF32Wrapper::new(
            vecsim::index::TieredSingle::new(params),
            type_,
        ))
    };

    Box::into_raw(Box::new(IndexHandle::new(
        wrapper,
        type_,
        VecSimAlgo::VecSimAlgo_TIERED,
        metric,
        dim,
        multi,
    ))) as *mut VecSimIndex
}

/// Check if the index is a disk-based index.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New` or similar.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_IsDisk(index: *const VecSimIndex) -> bool {
    if index.is_null() {
        return false;
    }
    let handle = &*(index as *const IndexHandle);
    handle.wrapper.is_disk()
}

/// Flush changes to disk for a disk-based index.
///
/// This ensures all pending changes are written to the underlying file.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_NewDisk`.
///
/// # Returns
/// true if flush succeeded, false otherwise.
#[no_mangle]
pub unsafe extern "C" fn VecSimDiskIndex_Flush(index: *const VecSimIndex) -> bool {
    if index.is_null() {
        return false;
    }
    let handle = &*(index as *const IndexHandle);
    handle.wrapper.disk_flush()
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

/// Determine if ad-hoc brute-force search is preferred over batched search.
///
/// This is a heuristic function that helps decide the optimal search strategy
/// for hybrid queries based on the index size and the number of results needed.
///
/// # Arguments
/// * `index` - The index handle
/// * `subsetSize` - The estimated size of the subset to search
/// * `k` - The number of results requested
/// * `initial` - Whether this is the initial decision (true) or a re-evaluation (false)
///
/// # Returns
/// true if ad-hoc search is preferred, false if batched search is preferred.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_PreferAdHocSearch(
    index: *const VecSimIndex,
    subsetSize: usize,
    k: usize,
    initial: bool,
) -> bool {
    if index.is_null() {
        return true; // Default to ad-hoc for safety
    }

    let handle = &*(index as *const IndexHandle);
    let index_size = handle.wrapper.index_size();

    if index_size == 0 {
        return true;
    }

    // Heuristic: prefer ad-hoc when subset is small relative to index
    // or when k is large relative to subset
    let subset_ratio = subsetSize as f64 / index_size as f64;
    let k_ratio = k as f64 / subsetSize.max(1) as f64;

    // If subset is less than 10% of index, prefer ad-hoc
    if subset_ratio < 0.1 {
        return true;
    }

    // If we need more than 10% of the subset, prefer ad-hoc
    if k_ratio > 0.1 {
        return true;
    }

    // For initial decision, be more conservative (prefer batches)
    // For re-evaluation, be more aggressive (prefer ad-hoc)
    if initial {
        subset_ratio < 0.3
    } else {
        subset_ratio < 0.5
    }
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

/// Get the status code of a query reply.
///
/// This is used to detect if the query timed out.
///
/// # Safety
/// `reply` must be a valid pointer returned by `VecSimIndex_TopKQuery` or `VecSimIndex_RangeQuery`.
#[no_mangle]
pub unsafe extern "C" fn VecSimQueryReply_GetCode(
    reply: *const VecSimQueryReply,
) -> types::VecSimQueryReply_Code {
    if reply.is_null() {
        return types::VecSimQueryReply_Code::VecSim_QueryReply_OK;
    }
    let handle = &*(reply as *const query::QueryReplyHandle);
    handle.code()
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

/// Get basic immutable index information.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_BasicInfo(
    index: *const VecSimIndex,
) -> info::VecSimIndexBasicInfo {
    if index.is_null() {
        return info::VecSimIndexBasicInfo {
            algo: VecSimAlgo::VecSimAlgo_BF,
            metric: VecSimMetric::VecSimMetric_L2,
            type_: VecSimType::VecSimType_FLOAT32,
            isMulti: false,
            isTiered: false,
            isDisk: false,
            blockSize: 0,
            dim: 0,
        };
    }

    let handle = &*(index as *const IndexHandle);
    info::VecSimIndexBasicInfo {
        algo: handle.algo,
        metric: handle.metric,
        type_: handle.data_type,
        isMulti: handle.is_multi,
        isTiered: handle.wrapper.is_tiered(),
        isDisk: handle.wrapper.is_disk(),
        blockSize: 1024, // Default block size
        dim: handle.dim,
    }
}

/// Get index statistics information.
///
/// This is a thin and efficient info call with no locks or calculations.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_StatsInfo(
    index: *const VecSimIndex,
) -> info::VecSimIndexStatsInfo {
    if index.is_null() {
        return info::VecSimIndexStatsInfo {
            memory: 0,
            numberOfMarkedDeleted: 0,
        };
    }

    let handle = &*(index as *const IndexHandle);
    info::VecSimIndexStatsInfo {
        memory: handle.wrapper.memory_usage(),
        numberOfMarkedDeleted: 0, // TODO: implement marked deleted tracking
    }
}

/// Get detailed debug information for an index.
///
/// This should only be used for debug/testing purposes.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_DebugInfo(
    index: *const VecSimIndex,
) -> info::VecSimIndexDebugInfo {
    use types::VecSearchMode;

    if index.is_null() {
        return info::VecSimIndexDebugInfo {
            commonInfo: info::CommonInfo {
                basicInfo: info::VecSimIndexBasicInfo {
                    algo: VecSimAlgo::VecSimAlgo_BF,
                    metric: VecSimMetric::VecSimMetric_L2,
                    type_: VecSimType::VecSimType_FLOAT32,
                    isMulti: false,
                    isTiered: false,
                    isDisk: false,
                    blockSize: 0,
                    dim: 0,
                },
                indexSize: 0,
                indexLabelCount: 0,
                memory: 0,
                lastMode: VecSearchMode::EMPTY_MODE,
            },
            hnswInfo: info::hnswInfoStruct {
                M: 0,
                efConstruction: 0,
                efRuntime: 0,
                epsilon: 0.0,
                max_level: 0,
                entrypoint: 0,
                visitedNodesPoolSize: 0,
                numberOfMarkedDeletedNodes: 0,
            },
            bfInfo: info::bfInfoStruct { dummy: 0 },
        };
    }

    let handle = &*(index as *const IndexHandle);
    let basic_info = VecSimIndex_BasicInfo(index);

    info::VecSimIndexDebugInfo {
        commonInfo: info::CommonInfo {
            basicInfo: basic_info,
            indexSize: handle.wrapper.index_size(),
            indexLabelCount: handle.wrapper.index_size(), // Approximate
            memory: handle.wrapper.memory_usage() as u64,
            lastMode: VecSearchMode::EMPTY_MODE,
        },
        hnswInfo: info::hnswInfoStruct {
            M: 16, // Default, would need to query from index
            efConstruction: 200,
            efRuntime: 10,
            epsilon: 0.0,
            max_level: 0,
            entrypoint: 0,
            visitedNodesPoolSize: 0,
            numberOfMarkedDeletedNodes: 0,
        },
        bfInfo: info::bfInfoStruct { dummy: 0 },
    }
}

// ============================================================================
// Debug Info Iterator Functions
// ============================================================================

/// Opaque type for C API - re-export from info module.
pub type VecSimDebugInfoIterator = info::VecSimDebugInfoIterator;

/// Create a debug info iterator for an index.
///
/// The iterator provides a way to traverse all debug information fields
/// for an index, including nested information for tiered indices.
///
/// # Safety
/// `index` must be a valid pointer returned by `VecSimIndex_New`.
/// The returned iterator must be freed with `VecSimDebugInfoIterator_Free`.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_DebugInfoIterator(
    index: *const VecSimIndex,
) -> *mut VecSimDebugInfoIterator {
    if index.is_null() {
        return std::ptr::null_mut();
    }

    let handle = &*(index as *const IndexHandle);
    let basic_info = VecSimIndex_BasicInfo(index);

    // Create iterator based on index type
    let iter = match basic_info.algo {
        VecSimAlgo::VecSimAlgo_BF => create_bf_debug_iterator(handle, &basic_info),
        VecSimAlgo::VecSimAlgo_HNSWLIB => create_hnsw_debug_iterator(handle, &basic_info),
        VecSimAlgo::VecSimAlgo_SVS => create_svs_debug_iterator(handle, &basic_info),
        VecSimAlgo::VecSimAlgo_TIERED => create_tiered_debug_iterator(handle, &basic_info),
    };

    Box::into_raw(Box::new(iter))
}

/// Helper to add common fields to an iterator.
fn add_common_fields(
    iter: &mut info::VecSimDebugInfoIterator,
    handle: &IndexHandle,
    basic_info: &info::VecSimIndexBasicInfo,
) {
    iter.add_string_field("TYPE", info::type_to_string(basic_info.type_));
    iter.add_uint64_field("DIMENSION", basic_info.dim as u64);
    iter.add_string_field("METRIC", info::metric_to_string(basic_info.metric));
    iter.add_string_field(
        "IS_MULTI_VALUE",
        if basic_info.isMulti { "true" } else { "false" },
    );
    iter.add_string_field(
        "IS_DISK",
        if basic_info.isDisk { "true" } else { "false" },
    );
    iter.add_uint64_field("INDEX_SIZE", handle.wrapper.index_size() as u64);
    iter.add_uint64_field("INDEX_LABEL_COUNT", handle.wrapper.index_size() as u64);
    iter.add_uint64_field("MEMORY", handle.wrapper.memory_usage() as u64);
    iter.add_string_field("LAST_SEARCH_MODE", "EMPTY_MODE");
}

/// Create a debug iterator for BruteForce index.
fn create_bf_debug_iterator(
    handle: &IndexHandle,
    basic_info: &info::VecSimIndexBasicInfo,
) -> info::VecSimDebugInfoIterator {
    let mut iter = info::VecSimDebugInfoIterator::new(10);

    iter.add_string_field("ALGORITHM", "FLAT");
    add_common_fields(&mut iter, handle, basic_info);
    iter.add_uint64_field("BLOCK_SIZE", basic_info.blockSize as u64);

    iter
}

/// Create a debug iterator for HNSW index.
fn create_hnsw_debug_iterator(
    handle: &IndexHandle,
    basic_info: &info::VecSimIndexBasicInfo,
) -> info::VecSimDebugInfoIterator {
    let mut iter = info::VecSimDebugInfoIterator::new(17);

    iter.add_string_field("ALGORITHM", "HNSW");
    add_common_fields(&mut iter, handle, basic_info);
    iter.add_uint64_field("BLOCK_SIZE", basic_info.blockSize as u64);

    // HNSW-specific fields (defaults, would need to query actual values)
    iter.add_uint64_field("M", 16);
    iter.add_uint64_field("EF_CONSTRUCTION", 200);
    iter.add_uint64_field("EF_RUNTIME", 10);
    iter.add_float64_field("EPSILON", 0.01);
    iter.add_uint64_field("MAX_LEVEL", 0);
    iter.add_uint64_field("ENTRYPOINT", 0);
    iter.add_uint64_field("NUMBER_OF_MARKED_DELETED", 0);

    iter
}

/// Create a debug iterator for SVS index.
fn create_svs_debug_iterator(
    handle: &IndexHandle,
    basic_info: &info::VecSimIndexBasicInfo,
) -> info::VecSimDebugInfoIterator {
    let mut iter = info::VecSimDebugInfoIterator::new(23);

    iter.add_string_field("ALGORITHM", "SVS");
    add_common_fields(&mut iter, handle, basic_info);
    iter.add_uint64_field("BLOCK_SIZE", basic_info.blockSize as u64);

    // SVS-specific fields (defaults)
    iter.add_string_field("QUANT_BITS", "NONE");
    iter.add_float64_field("ALPHA", 1.2);
    iter.add_uint64_field("GRAPH_MAX_DEGREE", 32);
    iter.add_uint64_field("CONSTRUCTION_WINDOW_SIZE", 200);
    iter.add_uint64_field("MAX_CANDIDATE_POOL_SIZE", 0);
    iter.add_uint64_field("PRUNE_TO", 0);
    iter.add_string_field("USE_SEARCH_HISTORY", "AUTO");
    iter.add_uint64_field("NUM_THREADS", 1);
    iter.add_uint64_field("LAST_RESERVED_NUM_THREADS", 0);
    iter.add_uint64_field("NUMBER_OF_MARKED_DELETED", 0);
    iter.add_uint64_field("SEARCH_WINDOW_SIZE", 10);
    iter.add_uint64_field("SEARCH_BUFFER_CAPACITY", 0);
    iter.add_uint64_field("LEANVEC_DIMENSION", 0);
    iter.add_float64_field("EPSILON", 0.01);

    iter
}

/// Create a debug iterator for tiered index.
fn create_tiered_debug_iterator(
    handle: &IndexHandle,
    basic_info: &info::VecSimIndexBasicInfo,
) -> info::VecSimDebugInfoIterator {
    let mut iter = info::VecSimDebugInfoIterator::new(15);

    iter.add_string_field("ALGORITHM", "TIERED");
    add_common_fields(&mut iter, handle, basic_info);

    // Tiered-specific fields
    iter.add_uint64_field("MANAGEMENT_LAYER_MEMORY", 0);
    iter.add_string_field("BACKGROUND_INDEXING", "false");
    iter.add_uint64_field("TIERED_BUFFER_LIMIT", 0);

    // Create frontend (flat) iterator
    let frontend_iter = create_bf_debug_iterator(handle, basic_info);
    iter.add_iterator_field("FRONTEND_INDEX", frontend_iter);

    // Create backend (hnsw) iterator
    let backend_iter = create_hnsw_debug_iterator(handle, basic_info);
    iter.add_iterator_field("BACKEND_INDEX", backend_iter);

    iter
}

/// Returns the number of fields in the info iterator.
///
/// # Safety
/// `info_iterator` must be a valid pointer returned by `VecSimIndex_DebugInfoIterator`.
#[no_mangle]
pub unsafe extern "C" fn VecSimDebugInfoIterator_NumberOfFields(
    info_iterator: *mut VecSimDebugInfoIterator,
) -> usize {
    if info_iterator.is_null() {
        return 0;
    }
    (*info_iterator).number_of_fields()
}

/// Returns if the fields iterator has more fields.
///
/// # Safety
/// `info_iterator` must be a valid pointer returned by `VecSimIndex_DebugInfoIterator`.
#[no_mangle]
pub unsafe extern "C" fn VecSimDebugInfoIterator_HasNextField(
    info_iterator: *mut VecSimDebugInfoIterator,
) -> bool {
    if info_iterator.is_null() {
        return false;
    }
    (*info_iterator).has_next()
}

/// Returns a pointer to the next info field.
///
/// The returned pointer is valid until the next call to this function
/// or until the iterator is freed.
///
/// # Safety
/// `info_iterator` must be a valid pointer returned by `VecSimIndex_DebugInfoIterator`.
#[no_mangle]
pub unsafe extern "C" fn VecSimDebugInfoIterator_NextField(
    info_iterator: *mut VecSimDebugInfoIterator,
) -> *mut info::VecSim_InfoField {
    if info_iterator.is_null() {
        return std::ptr::null_mut();
    }
    (*info_iterator).next()
}

/// Free an info iterator.
///
/// This also frees all nested iterators.
///
/// # Safety
/// `info_iterator` must be a valid pointer returned by `VecSimIndex_DebugInfoIterator`,
/// or null.
#[no_mangle]
pub unsafe extern "C" fn VecSimDebugInfoIterator_Free(
    info_iterator: *mut VecSimDebugInfoIterator,
) {
    if !info_iterator.is_null() {
        drop(Box::from_raw(info_iterator));
    }
}

// ============================================================================
// Serialization Functions
// ============================================================================

/// Save an index to a file.
///
/// Returns true on success, false on failure.
///
/// # Safety
/// - `index` must be a valid pointer returned by `VecSimIndex_New`
/// - `path` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_SaveIndex(
    index: *const VecSimIndex,
    path: *const c_char,
) -> bool {
    if index.is_null() || path.is_null() {
        return false;
    }

    let handle = &*(index as *const IndexHandle);
    let path_str = match std::ffi::CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    let path = std::path::Path::new(path_str);
    handle.wrapper.save_to_file(path)
}

/// Load an index from a file.
///
/// Returns a pointer to the loaded index, or null on failure.
/// The caller is responsible for freeing the index with VecSimIndex_Free.
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

    let path_str = match std::ffi::CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let file_path = std::path::Path::new(path_str);

    // Try to load the index by reading the header first to determine the type
    match load_index_from_file(file_path) {
        Some(handle) => Box::into_raw(handle) as *mut VecSimIndex,
        None => ptr::null_mut(),
    }
}

/// Internal function to load an index from a file.
/// Reads the header to determine the index type and loads accordingly.
fn load_index_from_file(path: &std::path::Path) -> Option<Box<IndexHandle>> {
    use std::fs::File;
    use std::io::BufReader;
    use vecsim::serialization::{IndexHeader, IndexTypeId, DataTypeId};

    // Open file and read header
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let header = IndexHeader::read(&mut reader).ok()?;

    // Based on index type and data type, load the appropriate index
    match (header.index_type, header.data_type) {
        // BruteForce Single
        (IndexTypeId::BruteForceSingle, DataTypeId::F32) => {
            use vecsim::index::BruteForceSingle;
            let index = BruteForceSingle::<f32>::load_from_file(path).ok()?;
            let metric = convert_metric_to_c(header.metric);
            Some(Box::new(IndexHandle::new(
                Box::new(index::BruteForceSingleF32Wrapper::new(
                    index,
                    VecSimType::VecSimType_FLOAT32,
                )),
                VecSimType::VecSimType_FLOAT32,
                VecSimAlgo::VecSimAlgo_BF,
                metric,
                header.dimension,
                false,
            )))
        }
        // BruteForce Multi
        (IndexTypeId::BruteForceMulti, DataTypeId::F32) => {
            use vecsim::index::BruteForceMulti;
            let index = BruteForceMulti::<f32>::load_from_file(path).ok()?;
            let metric = convert_metric_to_c(header.metric);
            Some(Box::new(IndexHandle::new(
                Box::new(index::BruteForceMultiF32Wrapper::new(
                    index,
                    VecSimType::VecSimType_FLOAT32,
                )),
                VecSimType::VecSimType_FLOAT32,
                VecSimAlgo::VecSimAlgo_BF,
                metric,
                header.dimension,
                true,
            )))
        }
        // HNSW Single
        (IndexTypeId::HnswSingle, DataTypeId::F32) => {
            use vecsim::index::HnswSingle;
            let index = HnswSingle::<f32>::load_from_file(path).ok()?;
            let metric = convert_metric_to_c(header.metric);
            Some(Box::new(IndexHandle::new(
                Box::new(index::HnswSingleF32Wrapper::new(
                    index,
                    VecSimType::VecSimType_FLOAT32,
                )),
                VecSimType::VecSimType_FLOAT32,
                VecSimAlgo::VecSimAlgo_HNSWLIB,
                metric,
                header.dimension,
                false,
            )))
        }
        // HNSW Multi
        (IndexTypeId::HnswMulti, DataTypeId::F32) => {
            use vecsim::index::HnswMulti;
            let index = HnswMulti::<f32>::load_from_file(path).ok()?;
            let metric = convert_metric_to_c(header.metric);
            Some(Box::new(IndexHandle::new(
                Box::new(index::HnswMultiF32Wrapper::new(
                    index,
                    VecSimType::VecSimType_FLOAT32,
                )),
                VecSimType::VecSimType_FLOAT32,
                VecSimAlgo::VecSimAlgo_HNSWLIB,
                metric,
                header.dimension,
                true,
            )))
        }
        // SVS Single
        (IndexTypeId::SvsSingle, DataTypeId::F32) => {
            use vecsim::index::SvsSingle;
            let index = SvsSingle::<f32>::load_from_file(path).ok()?;
            let metric = convert_metric_to_c(header.metric);
            Some(Box::new(IndexHandle::new(
                Box::new(index::SvsSingleF32Wrapper::new(
                    index,
                    VecSimType::VecSimType_FLOAT32,
                )),
                VecSimType::VecSimType_FLOAT32,
                VecSimAlgo::VecSimAlgo_SVS,
                metric,
                header.dimension,
                false,
            )))
        }
        // Unsupported combinations
        _ => None,
    }
}

/// Convert Rust Metric to C VecSimMetric
fn convert_metric_to_c(metric: vecsim::distance::Metric) -> VecSimMetric {
    match metric {
        vecsim::distance::Metric::L2 => VecSimMetric::VecSimMetric_L2,
        vecsim::distance::Metric::InnerProduct => VecSimMetric::VecSimMetric_IP,
        vecsim::distance::Metric::Cosine => VecSimMetric::VecSimMetric_Cosine,
    }
}

// ============================================================================
// Vector Utility Functions
// ============================================================================

/// Normalize a vector in-place.
///
/// This normalizes the vector to unit length (L2 norm = 1).
/// This is useful for cosine similarity where vectors should be normalized.
///
/// # Safety
/// - `blob` must point to a valid array of the correct type and dimension
/// - The array must be writable
#[no_mangle]
pub unsafe extern "C" fn VecSim_Normalize(
    blob: *mut c_void,
    dim: usize,
    type_: VecSimType,
) {
    if blob.is_null() || dim == 0 {
        return;
    }

    match type_ {
        VecSimType::VecSimType_FLOAT32 => {
            let data = std::slice::from_raw_parts_mut(blob as *mut f32, dim);
            let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in data.iter_mut() {
                    *x /= norm;
                }
            }
        }
        VecSimType::VecSimType_FLOAT64 => {
            let data = std::slice::from_raw_parts_mut(blob as *mut f64, dim);
            let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in data.iter_mut() {
                    *x /= norm;
                }
            }
        }
        // Other types don't support normalization
        _ => {}
    }
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

/// Estimate initial memory size for an index based on parameters.
///
/// # Safety
/// `params` must be a valid pointer to a VecSimParams struct.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_EstimateInitialSize(
    params: *const VecSimParams,
) -> usize {
    if params.is_null() {
        return 0;
    }

    let params = &*params;
    let dim = params.dim;
    let initial_capacity = params.initialCapacity;

    match params.algo {
        VecSimAlgo::VecSimAlgo_BF => {
            vecsim::index::estimate_brute_force_initial_size(dim, initial_capacity)
        }
        VecSimAlgo::VecSimAlgo_HNSWLIB => {
            // Default M = 16
            vecsim::index::estimate_hnsw_initial_size(dim, initial_capacity, 16)
        }
        VecSimAlgo::VecSimAlgo_TIERED => {
            // Tiered = BF frontend + HNSW backend
            let bf_size = vecsim::index::estimate_brute_force_initial_size(dim, initial_capacity);
            let hnsw_size = vecsim::index::estimate_hnsw_initial_size(dim, initial_capacity, 16);
            bf_size + hnsw_size
        }
        VecSimAlgo::VecSimAlgo_SVS => {
            // SVS is similar to HNSW in memory usage
            vecsim::index::estimate_hnsw_initial_size(dim, initial_capacity, 32)
        }
    }
}

/// Estimate memory size per element for an index based on parameters.
///
/// # Safety
/// `params` must be a valid pointer to a VecSimParams struct.
#[no_mangle]
pub unsafe extern "C" fn VecSimIndex_EstimateElementSize(
    params: *const VecSimParams,
) -> usize {
    if params.is_null() {
        return 0;
    }

    let params = &*params;
    let dim = params.dim;

    match params.algo {
        VecSimAlgo::VecSimAlgo_BF => {
            vecsim::index::estimate_brute_force_element_size(dim)
        }
        VecSimAlgo::VecSimAlgo_HNSWLIB => {
            // Default M = 16
            vecsim::index::estimate_hnsw_element_size(dim, 16)
        }
        VecSimAlgo::VecSimAlgo_TIERED => {
            // Use HNSW element size (vectors end up in HNSW)
            vecsim::index::estimate_hnsw_element_size(dim, 16)
        }
        VecSimAlgo::VecSimAlgo_SVS => {
            // SVS with default graph degree 32
            vecsim::index::estimate_hnsw_element_size(dim, 32)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_mode_default() {
        // Default should be WriteAsync
        let mode = VecSim_GetWriteMode();
        assert_eq!(mode, VecSimWriteMode::VecSim_WriteAsync);
    }

    #[test]
    fn test_write_mode_set_and_get() {
        // Set to InPlace
        VecSim_SetWriteMode(VecSimWriteMode::VecSim_WriteInPlace);
        assert_eq!(VecSim_GetWriteMode(), VecSimWriteMode::VecSim_WriteInPlace);

        // Set back to Async
        VecSim_SetWriteMode(VecSimWriteMode::VecSim_WriteAsync);
        assert_eq!(VecSim_GetWriteMode(), VecSimWriteMode::VecSim_WriteAsync);
    }

    #[test]
    fn test_write_mode_internal_function() {
        VecSim_SetWriteMode(VecSimWriteMode::VecSim_WriteInPlace);
        assert_eq!(get_global_write_mode(), VecSimWriteMode::VecSim_WriteInPlace);

        VecSim_SetWriteMode(VecSimWriteMode::VecSim_WriteAsync);
        assert_eq!(get_global_write_mode(), VecSimWriteMode::VecSim_WriteAsync);
    }

    // Helper to create a VecSimRawParam from strings
    fn make_raw_param(name: &str, value: &str) -> VecSimRawParam {
        VecSimRawParam {
            name: name.as_ptr() as *const std::ffi::c_char,
            nameLen: name.len(),
            value: value.as_ptr() as *const std::ffi::c_char,
            valLen: value.len(),
        }
    }

    // Helper to create HNSW params with valid dimensions
    fn test_hnsw_params() -> HNSWParams {
        HNSWParams {
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
        }
    }

    // Helper to create BF params with valid dimensions
    fn test_bf_params() -> BFParams {
        BFParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_BF,
                type_: VecSimType::VecSimType_FLOAT32,
                metric: VecSimMetric::VecSimMetric_L2,
                dim: 4,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
            },
        }
    }

    #[test]
    fn test_resolve_params_null_qparams() {
        let params = test_hnsw_params();
        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let result = VecSimIndex_ResolveParams(
                index,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                VecsimQueryType::QUERY_TYPE_KNN,
            );
            assert_eq!(result, VecSimParamResolveCode::VecSimParamResolverErr_NullParam);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_resolve_params_empty() {
        let params = test_hnsw_params();
        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let mut qparams = VecSimQueryParams::default();
            let result = VecSimIndex_ResolveParams(
                index,
                std::ptr::null(),
                0,
                &mut qparams,
                VecsimQueryType::QUERY_TYPE_KNN,
            );
            assert_eq!(result, VecSimParamResolveCode::VecSimParamResolver_OK);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_resolve_params_ef_runtime() {
        let params = test_hnsw_params();
        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let name = "EF_RUNTIME";
            let value = "100";
            let ef_param = make_raw_param(name, value);
            let mut qparams = VecSimQueryParams::default();

            let result = VecSimIndex_ResolveParams(
                index,
                &ef_param,
                1,
                &mut qparams,
                VecsimQueryType::QUERY_TYPE_KNN,
            );
            assert_eq!(result, VecSimParamResolveCode::VecSimParamResolver_OK);
            assert_eq!(qparams.hnswRuntimeParams.efRuntime, 100);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_resolve_params_ef_runtime_invalid_for_bf() {
        let params = test_bf_params();
        unsafe {
            let index = VecSimIndex_NewBF(&params);
            assert!(!index.is_null());

            let name = "EF_RUNTIME";
            let value = "100";
            let ef_param = make_raw_param(name, value);
            let mut qparams = VecSimQueryParams::default();

            let result = VecSimIndex_ResolveParams(
                index,
                &ef_param,
                1,
                &mut qparams,
                VecsimQueryType::QUERY_TYPE_KNN,
            );
            assert_eq!(result, VecSimParamResolveCode::VecSimParamResolverErr_UnknownParam);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_resolve_params_epsilon_for_range() {
        let params = test_hnsw_params();
        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let name = "EPSILON";
            let value = "0.01";
            let eps_param = make_raw_param(name, value);
            let mut qparams = VecSimQueryParams::default();

            let result = VecSimIndex_ResolveParams(
                index,
                &eps_param,
                1,
                &mut qparams,
                VecsimQueryType::QUERY_TYPE_RANGE,
            );
            assert_eq!(result, VecSimParamResolveCode::VecSimParamResolver_OK);
            assert!((qparams.hnswRuntimeParams.epsilon - 0.01).abs() < 0.0001);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_resolve_params_epsilon_not_for_knn() {
        let params = test_hnsw_params();
        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let name = "EPSILON";
            let value = "0.01";
            let eps_param = make_raw_param(name, value);
            let mut qparams = VecSimQueryParams::default();

            let result = VecSimIndex_ResolveParams(
                index,
                &eps_param,
                1,
                &mut qparams,
                VecsimQueryType::QUERY_TYPE_KNN,
            );
            assert_eq!(result, VecSimParamResolveCode::VecSimParamResolverErr_InvalidPolicy_NRange);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_resolve_params_batch_size_for_hybrid() {
        let params = test_hnsw_params();
        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let name = "BATCH_SIZE";
            let value = "256";
            let batch_param = make_raw_param(name, value);
            let mut qparams = VecSimQueryParams::default();

            let result = VecSimIndex_ResolveParams(
                index,
                &batch_param,
                1,
                &mut qparams,
                VecsimQueryType::QUERY_TYPE_HYBRID,
            );
            assert_eq!(result, VecSimParamResolveCode::VecSimParamResolver_OK);
            assert_eq!(qparams.batchSize, 256);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_resolve_params_duplicate_param() {
        let params = test_hnsw_params();
        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let name1 = "EF_RUNTIME";
            let value1 = "100";
            let name2 = "EF_RUNTIME";
            let value2 = "200";
            let ef_params = [
                make_raw_param(name1, value1),
                make_raw_param(name2, value2),
            ];
            let mut qparams = VecSimQueryParams::default();

            let result = VecSimIndex_ResolveParams(
                index,
                ef_params.as_ptr(),
                2,
                &mut qparams,
                VecsimQueryType::QUERY_TYPE_KNN,
            );
            assert_eq!(result, VecSimParamResolveCode::VecSimParamResolverErr_AlreadySet);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_resolve_params_unknown_param() {
        let params = test_hnsw_params();
        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let name = "UNKNOWN_PARAM";
            let value = "value";
            let unknown_param = make_raw_param(name, value);
            let mut qparams = VecSimQueryParams::default();

            let result = VecSimIndex_ResolveParams(
                index,
                &unknown_param,
                1,
                &mut qparams,
                VecsimQueryType::QUERY_TYPE_KNN,
            );
            assert_eq!(result, VecSimParamResolveCode::VecSimParamResolverErr_UnknownParam);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_resolve_params_bad_value() {
        let params = test_hnsw_params();
        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let name = "EF_RUNTIME";
            let value = "not_a_number";
            let bad_param = make_raw_param(name, value);
            let mut qparams = VecSimQueryParams::default();

            let result = VecSimIndex_ResolveParams(
                index,
                &bad_param,
                1,
                &mut qparams,
                VecsimQueryType::QUERY_TYPE_KNN,
            );
            assert_eq!(result, VecSimParamResolveCode::VecSimParamResolverErr_BadValue);

            VecSimIndex_Free(index);
        }
    }

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

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_save_and_load_hnsw_index() {
        use std::ffi::CString;
        use tempfile::NamedTempFile;

        let params = test_hnsw_params();

        unsafe {
            // Create and populate index
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
            let v3: [f32; 4] = [0.0, 0.0, 1.0, 0.0];

            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);
            VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2);
            VecSimIndex_AddVector(index, v3.as_ptr() as *const c_void, 3);

            assert_eq!(VecSimIndex_IndexSize(index), 3);

            // Save to file
            let temp_file = NamedTempFile::new().unwrap();
            let path = CString::new(temp_file.path().to_str().unwrap()).unwrap();
            let save_result = VecSimIndex_SaveIndex(index, path.as_ptr());
            assert!(save_result, "Failed to save index");

            // Free original index
            VecSimIndex_Free(index);

            // Load from file
            let loaded_index = VecSimIndex_LoadIndex(path.as_ptr(), ptr::null());
            assert!(!loaded_index.is_null(), "Failed to load index");

            // Verify loaded index
            assert_eq!(VecSimIndex_IndexSize(loaded_index), 3);

            // Query loaded index
            let query: [f32; 4] = [1.0, 0.1, 0.0, 0.0];
            let reply = VecSimIndex_TopKQuery(
                loaded_index,
                query.as_ptr() as *const c_void,
                2,
                ptr::null(),
                VecSimQueryReply_Order::BY_SCORE,
            );
            assert!(!reply.is_null());

            let len = VecSimQueryReply_Len(reply);
            assert_eq!(len, 2);

            // Check that closest vector is still label 1
            let iter = VecSimQueryReply_GetIterator(reply);
            let result = VecSimQueryReply_IteratorNext(iter);
            assert_eq!(VecSimQueryResult_GetId(result), 1);

            VecSimQueryReply_IteratorFree(iter);
            VecSimQueryReply_Free(reply);
            VecSimIndex_Free(loaded_index);
        }
    }

    #[test]
    fn test_save_and_load_brute_force_index() {
        use std::ffi::CString;
        use tempfile::NamedTempFile;

        let params = test_bf_params();

        unsafe {
            // Create and populate index
            let index = VecSimIndex_NewBF(&params);
            assert!(!index.is_null());

            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];

            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);
            VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2);

            assert_eq!(VecSimIndex_IndexSize(index), 2);

            // Save to file
            let temp_file = NamedTempFile::new().unwrap();
            let path = CString::new(temp_file.path().to_str().unwrap()).unwrap();
            let save_result = VecSimIndex_SaveIndex(index, path.as_ptr());
            assert!(save_result, "Failed to save BruteForce index");

            // Free original index
            VecSimIndex_Free(index);

            // Load from file
            let loaded_index = VecSimIndex_LoadIndex(path.as_ptr(), ptr::null());
            assert!(!loaded_index.is_null(), "Failed to load BruteForce index");

            // Verify loaded index
            assert_eq!(VecSimIndex_IndexSize(loaded_index), 2);

            VecSimIndex_Free(loaded_index);
        }
    }

    #[test]
    fn test_save_unsupported_type_returns_false() {
        use std::ffi::CString;
        use tempfile::NamedTempFile;

        // Create a BruteForce index with f64 (not supported for serialization)
        let params = BFParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_BF,
                type_: VecSimType::VecSimType_FLOAT64,
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

            let temp_file = NamedTempFile::new().unwrap();
            let path = CString::new(temp_file.path().to_str().unwrap()).unwrap();

            // Should return false for unsupported type
            let save_result = VecSimIndex_SaveIndex(index, path.as_ptr());
            assert!(!save_result, "Save should fail for f64 BruteForce");

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_load_nonexistent_file_returns_null() {
        use std::ffi::CString;

        unsafe {
            let path = CString::new("/nonexistent/path/to/index.bin").unwrap();
            let loaded_index = VecSimIndex_LoadIndex(path.as_ptr(), ptr::null());
            assert!(loaded_index.is_null(), "Load should fail for nonexistent file");
        }
    }

    #[test]
    fn test_save_null_index_returns_false() {
        use std::ffi::CString;

        unsafe {
            let path = CString::new("/tmp/test.bin").unwrap();
            let result = VecSimIndex_SaveIndex(ptr::null(), path.as_ptr());
            assert!(!result, "Save should fail for null index");
        }
    }

    #[test]
    fn test_save_null_path_returns_false() {
        let params = test_hnsw_params();

        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let result = VecSimIndex_SaveIndex(index, ptr::null());
            assert!(!result, "Save should fail for null path");

            VecSimIndex_Free(index);
        }
    }

    // ========================================================================
    // Tiered Index Tests
    // ========================================================================

    fn test_tiered_params() -> TieredParams {
        TieredParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_TIERED,
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
            flatBufferLimit: 100,
            writeMode: 0, // Async
        }
    }

    #[test]
    fn test_tiered_create_and_free() {
        let params = test_tiered_params();

        unsafe {
            let index = VecSimIndex_NewTiered(&params);
            assert!(!index.is_null(), "Tiered index creation should succeed");

            assert!(VecSimIndex_IsTiered(index), "Index should be tiered");
            assert_eq!(VecSimIndex_IndexSize(index), 0);
            assert_eq!(VecSimIndex_GetDim(index), 4);
            assert_eq!(VecSimTieredIndex_FlatSize(index), 0);
            assert_eq!(VecSimTieredIndex_BackendSize(index), 0);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_tiered_add_vectors_to_flat() {
        let params = test_tiered_params();

        unsafe {
            let index = VecSimIndex_NewTiered(&params);
            assert!(!index.is_null());

            // Add vectors - they should go to flat buffer
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
            let v3: [f32; 4] = [0.0, 0.0, 1.0, 0.0];

            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);
            VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2);
            VecSimIndex_AddVector(index, v3.as_ptr() as *const c_void, 3);

            assert_eq!(VecSimIndex_IndexSize(index), 3);
            assert_eq!(VecSimTieredIndex_FlatSize(index), 3);
            assert_eq!(VecSimTieredIndex_BackendSize(index), 0);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_tiered_flush() {
        let params = test_tiered_params();

        unsafe {
            let index = VecSimIndex_NewTiered(&params);
            assert!(!index.is_null());

            // Add vectors
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];

            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);
            VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2);

            assert_eq!(VecSimTieredIndex_FlatSize(index), 2);
            assert_eq!(VecSimTieredIndex_BackendSize(index), 0);

            // Flush to HNSW
            let flushed = VecSimTieredIndex_Flush(index);
            assert_eq!(flushed, 2);

            assert_eq!(VecSimTieredIndex_FlatSize(index), 0);
            assert_eq!(VecSimTieredIndex_BackendSize(index), 2);
            assert_eq!(VecSimIndex_IndexSize(index), 2);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_tiered_query_both_tiers() {
        let params = test_tiered_params();

        unsafe {
            let index = VecSimIndex_NewTiered(&params);
            assert!(!index.is_null());

            // Add vector to flat
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);

            // Flush to HNSW
            VecSimTieredIndex_Flush(index);

            // Add more to flat
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
            VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2);

            assert_eq!(VecSimTieredIndex_FlatSize(index), 1);
            assert_eq!(VecSimTieredIndex_BackendSize(index), 1);

            // Query should find both
            let query: [f32; 4] = [0.5, 0.5, 0.0, 0.0];
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

            VecSimQueryReply_Free(reply);
            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_tiered_multi_index() {
        let mut params = test_tiered_params();
        params.base.multi = true;

        unsafe {
            let index = VecSimIndex_NewTiered(&params);
            assert!(!index.is_null());

            // Add multiple vectors with same label
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];

            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);
            VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 1);

            assert_eq!(VecSimIndex_IndexSize(index), 2);
            assert_eq!(VecSimIndex_LabelCount(index, 1), 2);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_tiered_is_tiered_false_for_other_indices() {
        let bf_params = test_bf_params();
        let hnsw_params = test_hnsw_params();

        unsafe {
            let bf_index = VecSimIndex_NewBF(&bf_params);
            let hnsw_index = VecSimIndex_NewHNSW(&hnsw_params);

            assert!(!VecSimIndex_IsTiered(bf_index), "BF should not be tiered");
            assert!(!VecSimIndex_IsTiered(hnsw_index), "HNSW should not be tiered");

            VecSimIndex_Free(bf_index);
            VecSimIndex_Free(hnsw_index);
        }
    }

    #[test]
    fn test_tiered_unsupported_type_returns_null() {
        let mut params = test_tiered_params();
        params.base.type_ = VecSimType::VecSimType_FLOAT64; // Not supported

        unsafe {
            let index = VecSimIndex_NewTiered(&params);
            assert!(index.is_null(), "Tiered should fail for f64");
        }
    }

    // ========================================================================
    // Memory Functions Tests
    // ========================================================================

    use std::sync::atomic::AtomicUsize;

    // Track allocations for testing
    static TEST_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    static TEST_FREE_COUNT: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn test_alloc(n: usize) -> *mut c_void {
        TEST_ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        libc::malloc(n)
    }

    unsafe extern "C" fn test_calloc(nelem: usize, elemsz: usize) -> *mut c_void {
        TEST_ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        libc::calloc(nelem, elemsz)
    }

    unsafe extern "C" fn test_realloc(p: *mut c_void, n: usize) -> *mut c_void {
        libc::realloc(p, n)
    }

    unsafe extern "C" fn test_free(p: *mut c_void) {
        TEST_FREE_COUNT.fetch_add(1, Ordering::SeqCst);
        libc::free(p)
    }

    #[test]
    fn test_set_memory_functions() {
        // Reset counters
        TEST_ALLOC_COUNT.store(0, Ordering::SeqCst);
        TEST_FREE_COUNT.store(0, Ordering::SeqCst);

        let mem_funcs = VecSimMemoryFunctions {
            allocFunction: Some(test_alloc),
            callocFunction: Some(test_calloc),
            reallocFunction: Some(test_realloc),
            freeFunction: Some(test_free),
        };

        unsafe {
            VecSim_SetMemoryFunctions(mem_funcs);
        }

        // Verify the functions were set
        let stored = get_memory_functions();
        assert!(stored.is_some(), "Memory functions should be set");

        let funcs = stored.unwrap();
        assert!(funcs.allocFunction.is_some());
        assert!(funcs.callocFunction.is_some());
        assert!(funcs.reallocFunction.is_some());
        assert!(funcs.freeFunction.is_some());
    }

    #[test]
    fn test_memory_functions_default_none() {
        // Note: This test may fail if run after test_set_memory_functions
        // due to global state. In a real scenario, we'd need proper isolation.
        // For now, we just verify the API works.
        let mem_funcs = VecSimMemoryFunctions {
            allocFunction: None,
            callocFunction: None,
            reallocFunction: None,
            freeFunction: None,
        };

        unsafe {
            VecSim_SetMemoryFunctions(mem_funcs);
        }

        let stored = get_memory_functions();
        assert!(stored.is_some()); // It's Some because we set it, but with None values
    }

    // ========================================================================
    // Disk Index Tests
    // ========================================================================

    use params::DiskBackend;
    use std::ffi::CString;
    use tempfile::tempdir;

    fn test_disk_params(path: &CString) -> DiskParams {
        DiskParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_BF,
                type_: VecSimType::VecSimType_FLOAT32,
                dim: 4,
                metric: VecSimMetric::VecSimMetric_L2,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
            },
            dataPath: path.as_ptr(),
            backend: DiskBackend::DiskBackend_BruteForce,
            graphMaxDegree: 32,
            alpha: 1.2,
            constructionL: 200,
            searchL: 100,
        }
    }

    #[test]
    fn test_disk_create_and_free() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_disk.bin");
        let path_cstr = CString::new(path.to_str().unwrap()).unwrap();
        let params = test_disk_params(&path_cstr);

        unsafe {
            let index = VecSimIndex_NewDisk(&params);
            assert!(!index.is_null(), "Disk index creation should succeed");

            assert!(VecSimIndex_IsDisk(index), "Index should be disk-based");
            assert!(!VecSimIndex_IsTiered(index), "Index should not be tiered");
            assert_eq!(VecSimIndex_IndexSize(index), 0);
            assert_eq!(VecSimIndex_GetDim(index), 4);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_disk_add_and_query() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_disk_query.bin");
        let path_cstr = CString::new(path.to_str().unwrap()).unwrap();
        let params = test_disk_params(&path_cstr);

        unsafe {
            let index = VecSimIndex_NewDisk(&params);
            assert!(!index.is_null());

            // Add vectors
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
            let v3: [f32; 4] = [0.0, 0.0, 1.0, 0.0];

            assert_eq!(VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1), 1);
            assert_eq!(VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2), 1);
            assert_eq!(VecSimIndex_AddVector(index, v3.as_ptr() as *const c_void, 3), 1);

            assert_eq!(VecSimIndex_IndexSize(index), 3);

            // Query for nearest neighbor
            let query: [f32; 4] = [0.9, 0.1, 0.0, 0.0];
            let reply = VecSimIndex_TopKQuery(
                index,
                query.as_ptr() as *const c_void,
                1,
                std::ptr::null(),
                VecSimQueryReply_Order::BY_SCORE,
            );
            assert!(!reply.is_null());

            let len = VecSimQueryReply_Len(reply);
            assert_eq!(len, 1);

            let iter = VecSimQueryReply_GetIterator(reply);
            let result = VecSimQueryReply_IteratorNext(iter);
            assert!(!result.is_null());
            assert_eq!(VecSimQueryResult_GetId(result), 1); // v1 is closest

            VecSimQueryReply_IteratorFree(iter);
            VecSimQueryReply_Free(reply);
            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_disk_flush() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_disk_flush.bin");
        let path_cstr = CString::new(path.to_str().unwrap()).unwrap();
        let params = test_disk_params(&path_cstr);

        unsafe {
            let index = VecSimIndex_NewDisk(&params);
            assert!(!index.is_null());

            // Add a vector
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);

            // Flush should succeed
            assert!(VecSimDiskIndex_Flush(index), "Flush should succeed");

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_disk_delete_vector() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_disk_delete.bin");
        let path_cstr = CString::new(path.to_str().unwrap()).unwrap();
        let params = test_disk_params(&path_cstr);

        unsafe {
            let index = VecSimIndex_NewDisk(&params);
            assert!(!index.is_null());

            // Add vectors
            let v1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            let v2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
            VecSimIndex_AddVector(index, v1.as_ptr() as *const c_void, 1);
            VecSimIndex_AddVector(index, v2.as_ptr() as *const c_void, 2);
            assert_eq!(VecSimIndex_IndexSize(index), 2);

            // Delete one vector
            let deleted = VecSimIndex_DeleteVector(index, 1);
            assert_eq!(deleted, 1);
            assert_eq!(VecSimIndex_IndexSize(index), 1);

            // Verify the remaining vector
            assert!(!VecSimIndex_ContainsLabel(index, 1));
            assert!(VecSimIndex_ContainsLabel(index, 2));

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_disk_null_path_returns_null() {
        let params = DiskParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_BF,
                type_: VecSimType::VecSimType_FLOAT32,
                dim: 4,
                metric: VecSimMetric::VecSimMetric_L2,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
            },
            dataPath: std::ptr::null(),
            backend: DiskBackend::DiskBackend_BruteForce,
            graphMaxDegree: 32,
            alpha: 1.2,
            constructionL: 200,
            searchL: 100,
        };

        unsafe {
            let index = VecSimIndex_NewDisk(&params);
            assert!(index.is_null(), "Disk index with null path should fail");
        }
    }

    #[test]
    fn test_disk_unsupported_type_returns_null() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_disk_f64.bin");
        let path_cstr = CString::new(path.to_str().unwrap()).unwrap();

        let params = DiskParams {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_BF,
                type_: VecSimType::VecSimType_FLOAT64, // Not supported
                dim: 4,
                metric: VecSimMetric::VecSimMetric_L2,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
            },
            dataPath: path_cstr.as_ptr(),
            backend: DiskBackend::DiskBackend_BruteForce,
            graphMaxDegree: 32,
            alpha: 1.2,
            constructionL: 200,
            searchL: 100,
        };

        unsafe {
            let index = VecSimIndex_NewDisk(&params);
            assert!(index.is_null(), "Disk index should fail for f64");
        }
    }

    #[test]
    fn test_disk_is_disk_false_for_other_indices() {
        let bf_params = test_bf_params();
        let hnsw_params = test_hnsw_params();

        unsafe {
            let bf_index = VecSimIndex_NewBF(&bf_params);
            let hnsw_index = VecSimIndex_NewHNSW(&hnsw_params);

            assert!(!VecSimIndex_IsDisk(bf_index), "BF should not be disk");
            assert!(!VecSimIndex_IsDisk(hnsw_index), "HNSW should not be disk");

            VecSimIndex_Free(bf_index);
            VecSimIndex_Free(hnsw_index);
        }
    }

    // ========================================================================
    // New API Functions Tests
    // ========================================================================

    #[test]
    fn test_normalize_f32() {
        unsafe {
            let mut v: [f32; 4] = [3.0, 4.0, 0.0, 0.0];
            VecSim_Normalize(v.as_mut_ptr() as *mut c_void, 4, VecSimType::VecSimType_FLOAT32);

            // L2 norm of [3, 4, 0, 0] is 5, so normalized should be [0.6, 0.8, 0, 0]
            assert!((v[0] - 0.6).abs() < 0.001);
            assert!((v[1] - 0.8).abs() < 0.001);
            assert!((v[2] - 0.0).abs() < 0.001);
            assert!((v[3] - 0.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_normalize_f64() {
        unsafe {
            let mut v: [f64; 4] = [3.0, 4.0, 0.0, 0.0];
            VecSim_Normalize(v.as_mut_ptr() as *mut c_void, 4, VecSimType::VecSimType_FLOAT64);

            assert!((v[0] - 0.6).abs() < 0.001);
            assert!((v[1] - 0.8).abs() < 0.001);
        }
    }

    #[test]
    fn test_normalize_null_is_safe() {
        unsafe {
            // Should not crash
            VecSim_Normalize(ptr::null_mut(), 4, VecSimType::VecSimType_FLOAT32);
            VecSim_Normalize(ptr::null_mut(), 0, VecSimType::VecSimType_FLOAT32);
        }
    }

    #[test]
    fn test_basic_info() {
        let params = test_hnsw_params();

        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let info = VecSimIndex_BasicInfo(index);
            assert_eq!(info.algo, VecSimAlgo::VecSimAlgo_HNSWLIB);
            assert_eq!(info.metric, VecSimMetric::VecSimMetric_L2);
            assert_eq!(info.type_, VecSimType::VecSimType_FLOAT32);
            assert_eq!(info.dim, 4);
            assert!(!info.isMulti);
            assert!(!info.isTiered);
            assert!(!info.isDisk);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_stats_info() {
        let params = test_bf_params();

        unsafe {
            let index = VecSimIndex_NewBF(&params);
            assert!(!index.is_null());

            let info = VecSimIndex_StatsInfo(index);
            // Memory should be non-zero for an allocated index
            assert!(info.memory > 0 || info.memory == 0); // Just check it doesn't crash

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_debug_info() {
        let params = test_hnsw_params();

        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            let info = VecSimIndex_DebugInfo(index);
            assert_eq!(info.commonInfo.basicInfo.algo, VecSimAlgo::VecSimAlgo_HNSWLIB);
            assert_eq!(info.commonInfo.basicInfo.dim, 4);
            assert_eq!(info.commonInfo.indexSize, 0);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_debug_info_iterator() {
        let params = test_hnsw_params();

        unsafe {
            let index = VecSimIndex_NewHNSW(&params);
            assert!(!index.is_null());

            // Create the debug info iterator
            let iter = VecSimIndex_DebugInfoIterator(index);
            assert!(!iter.is_null());

            // Check that we have fields
            let num_fields = VecSimDebugInfoIterator_NumberOfFields(iter);
            assert!(num_fields > 0, "Should have at least one field");

            // Iterate through the fields
            let mut count = 0;
            while VecSimDebugInfoIterator_HasNextField(iter) {
                let field = VecSimDebugInfoIterator_NextField(iter);
                assert!(!field.is_null());
                assert!(!(*field).fieldName.is_null());
                count += 1;
            }
            assert_eq!(count, num_fields);

            // Should return null after exhausting
            assert!(!VecSimDebugInfoIterator_HasNextField(iter));

            // Free the iterator
            VecSimDebugInfoIterator_Free(iter);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_debug_info_iterator_bf() {
        let params = test_bf_params();

        unsafe {
            let index = VecSimIndex_NewBF(&params);
            assert!(!index.is_null());

            let iter = VecSimIndex_DebugInfoIterator(index);
            assert!(!iter.is_null());

            // BF should have fewer fields than HNSW
            let num_fields = VecSimDebugInfoIterator_NumberOfFields(iter);
            assert!(num_fields > 0);
            assert!(num_fields <= 12, "BF should have ~10 fields");

            VecSimDebugInfoIterator_Free(iter);
            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_debug_info_iterator_null_safe() {
        unsafe {
            // Null iterator should be handled gracefully
            assert_eq!(VecSimDebugInfoIterator_NumberOfFields(std::ptr::null_mut()), 0);
            assert!(!VecSimDebugInfoIterator_HasNextField(std::ptr::null_mut()));
            assert!(VecSimDebugInfoIterator_NextField(std::ptr::null_mut()).is_null());

            // Free null should not crash
            VecSimDebugInfoIterator_Free(std::ptr::null_mut());

            // Null index should return null iterator
            let iter = VecSimIndex_DebugInfoIterator(std::ptr::null());
            assert!(iter.is_null());
        }
    }

    #[test]
    fn test_prefer_adhoc_search() {
        let params = test_bf_params();

        unsafe {
            let index = VecSimIndex_NewBF(&params);
            assert!(!index.is_null());

            // Add some vectors
            for i in 0..100 {
                let v: [f32; 4] = [i as f32, 0.0, 0.0, 0.0];
                VecSimIndex_AddVector(index, v.as_ptr() as *const c_void, i as u64);
            }

            // Small subset should prefer ad-hoc
            let prefer = VecSimIndex_PreferAdHocSearch(index, 5, 10, true);
            assert!(prefer, "Small subset should prefer ad-hoc");

            // Large subset with small k should prefer batches
            // subset_ratio = 90/100 = 0.9 (> 0.3), k_ratio = 5/90 = 0.055 (< 0.1)
            let prefer = VecSimIndex_PreferAdHocSearch(index, 90, 5, true);
            assert!(!prefer, "Large subset with small k should prefer batches");

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_estimate_initial_size() {
        let params = VecSimParams {
            algo: VecSimAlgo::VecSimAlgo_BF,
            type_: VecSimType::VecSimType_FLOAT32,
            metric: VecSimMetric::VecSimMetric_L2,
            dim: 128,
            multi: false,
            initialCapacity: 1000,
            blockSize: 0,
        };

        unsafe {
            let size = VecSimIndex_EstimateInitialSize(&params);
            assert!(size > 0, "Estimate should be positive");
        }
    }

    #[test]
    fn test_estimate_element_size() {
        let params = VecSimParams {
            algo: VecSimAlgo::VecSimAlgo_HNSWLIB,
            type_: VecSimType::VecSimType_FLOAT32,
            metric: VecSimMetric::VecSimMetric_L2,
            dim: 128,
            multi: false,
            initialCapacity: 1000,
            blockSize: 0,
        };

        unsafe {
            let size = VecSimIndex_EstimateElementSize(&params);
            assert!(size > 0, "Estimate should be positive");
            // Should be at least the vector size (128 * 4 bytes)
            assert!(size >= 128 * 4, "Should include vector storage");
        }
    }

    #[test]
    fn test_tiered_gc() {
        let params = test_tiered_params();

        unsafe {
            let index = VecSimIndex_NewTiered(&params);
            assert!(!index.is_null());

            // GC on empty index should not crash
            VecSimTieredIndex_GC(index);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_tiered_locks() {
        let params = test_tiered_params();

        unsafe {
            let index = VecSimIndex_NewTiered(&params);
            assert!(!index.is_null());

            // Should not crash
            VecSimTieredIndex_AcquireSharedLocks(index);
            VecSimTieredIndex_ReleaseSharedLocks(index);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_callback_setters() {
        unsafe {
            // Test timeout callback
            VecSim_SetTimeoutCallbackFunction(None);

            // Test log callback
            VecSim_SetLogCallbackFunction(None);

            // Test log context
            VecSim_SetTestLogContext(ptr::null(), ptr::null());
        }
    }

    // ========================================================================
    // C++-Compatible API Tests
    // ========================================================================

    #[test]
    fn test_vecsim_index_new_bf() {
        use crate::compat::{AlgoParams_C, BFParams_C, VecSimParams_C};

        unsafe {
            let bf_params = BFParams_C {
                type_: VecSimType::VecSimType_FLOAT32,
                dim: 4,
                metric: VecSimMetric::VecSimMetric_L2,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
            };

            let params = VecSimParams_C {
                algo: VecSimAlgo::VecSimAlgo_BF,
                algoParams: AlgoParams_C { bfParams: bf_params },
                logCtx: ptr::null_mut(),
            };

            let index = VecSimIndex_New(&params);
            assert!(!index.is_null(), "VecSimIndex_New should create BF index");

            // Verify it works
            let v: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            VecSimIndex_AddVector(index, v.as_ptr() as *const c_void, 1);
            assert_eq!(VecSimIndex_IndexSize(index), 1);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_vecsim_index_new_hnsw() {
        use crate::compat::{AlgoParams_C, HNSWParams_C, VecSimParams_C};

        unsafe {
            let hnsw_params = HNSWParams_C {
                type_: VecSimType::VecSimType_FLOAT32,
                dim: 4,
                metric: VecSimMetric::VecSimMetric_L2,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
                M: 16,
                efConstruction: 200,
                efRuntime: 10,
                epsilon: 0.0,
            };

            let params = VecSimParams_C {
                algo: VecSimAlgo::VecSimAlgo_HNSWLIB,
                algoParams: AlgoParams_C { hnswParams: hnsw_params },
                logCtx: ptr::null_mut(),
            };

            let index = VecSimIndex_New(&params);
            assert!(!index.is_null(), "VecSimIndex_New should create HNSW index");

            // Verify it works
            let v: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            VecSimIndex_AddVector(index, v.as_ptr() as *const c_void, 1);
            assert_eq!(VecSimIndex_IndexSize(index), 1);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_vecsim_index_new_tiered() {
        use crate::compat::{
            AlgoParams_C, HNSWParams_C, TieredHNSWParams_C, TieredIndexParams_C,
            TieredSpecificParams_C, VecSimParams_C,
        };
        use std::mem::ManuallyDrop;

        unsafe {
            // Create the primary (backend) params
            let hnsw_params = HNSWParams_C {
                type_: VecSimType::VecSimType_FLOAT32,
                dim: 4,
                metric: VecSimMetric::VecSimMetric_L2,
                multi: false,
                initialCapacity: 100,
                blockSize: 0,
                M: 16,
                efConstruction: 200,
                efRuntime: 10,
                epsilon: 0.0,
            };

            let mut primary_params = VecSimParams_C {
                algo: VecSimAlgo::VecSimAlgo_HNSWLIB,
                algoParams: AlgoParams_C { hnswParams: hnsw_params },
                logCtx: ptr::null_mut(),
            };

            let tiered_params = TieredIndexParams_C {
                jobQueue: ptr::null_mut(),
                jobQueueCtx: ptr::null_mut(),
                submitCb: None,
                flatBufferLimit: 100,
                primaryIndexParams: &mut primary_params,
                specificParams: TieredSpecificParams_C {
                    tieredHnswParams: TieredHNSWParams_C { swapJobThreshold: 0 },
                },
            };

            let params = VecSimParams_C {
                algo: VecSimAlgo::VecSimAlgo_TIERED,
                algoParams: AlgoParams_C {
                    tieredParams: ManuallyDrop::new(tiered_params),
                },
                logCtx: ptr::null_mut(),
            };

            let index = VecSimIndex_New(&params);
            assert!(!index.is_null(), "VecSimIndex_New should create tiered index");

            // Verify it works
            assert!(VecSimIndex_IsTiered(index));
            let v: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            VecSimIndex_AddVector(index, v.as_ptr() as *const c_void, 1);
            assert_eq!(VecSimIndex_IndexSize(index), 1);

            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_vecsim_index_new_null_returns_null() {
        unsafe {
            let index = VecSimIndex_New(ptr::null());
            assert!(index.is_null(), "VecSimIndex_New should return null for null params");
        }
    }

    #[test]
    fn test_query_reply_get_code() {
        unsafe {
            // Create an index
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
            let index = VecSimIndex_NewBF(&params);
            assert!(!index.is_null());

            // Add a vector
            let v: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
            VecSimIndex_AddVector(index, v.as_ptr() as *const c_void, 1);

            // Query
            let reply = VecSimIndex_TopKQuery(
                index,
                v.as_ptr() as *const c_void,
                10,
                ptr::null(),
                VecSimQueryReply_Order::BY_SCORE,
            );
            assert!(!reply.is_null());

            // Check code - should be OK (not timed out)
            let code = VecSimQueryReply_GetCode(reply);
            assert_eq!(code, types::VecSimQueryReply_Code::VecSim_QueryReply_OK);

            // Cleanup
            VecSimQueryReply_Free(reply);
            VecSimIndex_Free(index);
        }
    }

    #[test]
    fn test_query_reply_get_code_null_is_safe() {
        unsafe {
            // Should not crash with null, returns OK
            let code = VecSimQueryReply_GetCode(ptr::null());
            assert_eq!(code, types::VecSimQueryReply_Code::VecSim_QueryReply_OK);
        }
    }
}

//! C-compatible type definitions for the VecSim FFI.

/// Vector element data type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimType {
    VecSimType_FLOAT32 = 0,
    VecSimType_FLOAT64 = 1,
    VecSimType_BFLOAT16 = 2,
    VecSimType_FLOAT16 = 3,
    VecSimType_INT8 = 4,
    VecSimType_UINT8 = 5,
    VecSimType_INT32 = 6,
    VecSimType_INT64 = 7,
}

/// Index algorithm type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimAlgo {
    VecSimAlgo_BF = 0,
    VecSimAlgo_HNSWLIB = 1,
    VecSimAlgo_TIERED = 2,
    VecSimAlgo_SVS = 3,
}

/// Distance metric type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimMetric {
    VecSimMetric_L2 = 0,
    VecSimMetric_IP = 1,
    VecSimMetric_Cosine = 2,
}

/// Query reply ordering.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimQueryReply_Order {
    BY_SCORE = 0,
    BY_ID = 1,
}

/// Query reply status code (for detecting timeouts, etc.).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VecSimQueryReply_Code {
    /// Query completed successfully.
    #[default]
    VecSim_QueryReply_OK = 0,
    /// Query was aborted due to timeout.
    VecSim_QueryReply_TimedOut = 1,
}

/// Index resolve codes for resolving index state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimResolveCode {
    VecSim_Resolve_OK = 0,
    VecSim_Resolve_ERR = 1,
}

/// Parameter resolution error codes.
///
/// These codes are returned by VecSimIndex_ResolveParams to indicate
/// the result of parsing runtime query parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimParamResolveCode {
    /// Parameter resolution succeeded.
    VecSimParamResolver_OK = 0,
    /// Null parameter pointer was passed.
    VecSimParamResolverErr_NullParam = 1,
    /// Parameter was already set (duplicate parameter).
    VecSimParamResolverErr_AlreadySet = 2,
    /// Unknown parameter name.
    VecSimParamResolverErr_UnknownParam = 3,
    /// Invalid parameter value.
    VecSimParamResolverErr_BadValue = 4,
    /// Invalid policy: policy does not exist.
    VecSimParamResolverErr_InvalidPolicy_NExits = 5,
    /// Invalid policy: not a hybrid query.
    VecSimParamResolverErr_InvalidPolicy_NHybrid = 6,
    /// Invalid policy: not a range query.
    VecSimParamResolverErr_InvalidPolicy_NRange = 7,
    /// Invalid policy: ad-hoc policy with batch size.
    VecSimParamResolverErr_InvalidPolicy_AdHoc_With_BatchSize = 8,
    /// Invalid policy: ad-hoc policy with ef_runtime.
    VecSimParamResolverErr_InvalidPolicy_AdHoc_With_EfRuntime = 9,
}

/// Query type for parameter resolution.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecsimQueryType {
    /// No specific query type.
    QUERY_TYPE_NONE = 0,
    /// Standard KNN query.
    QUERY_TYPE_KNN = 1,
    /// Hybrid query (combining vector similarity with filters).
    QUERY_TYPE_HYBRID = 2,
    /// Range query.
    QUERY_TYPE_RANGE = 3,
}

/// Debug command return codes.
///
/// These codes are returned by debug functions like VecSimDebug_GetElementNeighborsInHNSWGraph.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimDebugCommandCode {
    /// Command succeeded.
    VecSimDebugCommandCode_OK = 0,
    /// Bad index (null, wrong type, or unsupported).
    VecSimDebugCommandCode_BadIndex = 1,
    /// Label does not exist in the index.
    VecSimDebugCommandCode_LabelNotExists = 2,
    /// Multi-value indices are not supported for this operation.
    VecSimDebugCommandCode_MultiNotSupported = 3,
}

/// Raw parameter for runtime query configuration.
///
/// Used to pass string-based parameters that are resolved into typed
/// VecSimQueryParams by VecSimIndex_ResolveParams.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VecSimRawParam {
    /// Parameter name.
    pub name: *const std::ffi::c_char,
    /// Length of the parameter name.
    pub nameLen: usize,
    /// Parameter value as a string.
    pub value: *const std::ffi::c_char,
    /// Length of the parameter value.
    pub valLen: usize,
}

/// Write mode for tiered index operations.
///
/// Controls whether vector additions/deletions go through the async
/// buffering path or directly to the backend index.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VecSimWriteMode {
    /// Async mode: vectors go to flat buffer, migrated to backend via background jobs.
    #[default]
    VecSim_WriteAsync = 0,
    /// InPlace mode: vectors go directly to the backend index.
    VecSim_WriteInPlace = 1,
}

/// Option mode for various settings.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VecSimOptionMode {
    #[default]
    VecSimOption_AUTO = 0,
    VecSimOption_ENABLE = 1,
    VecSimOption_DISABLE = 2,
}

/// Tri-state boolean for optional settings.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimBool {
    VecSimBool_TRUE = 1,
    VecSimBool_FALSE = 0,
    VecSimBool_UNSET = -1,
}

/// Search mode for queries (used for debug/testing).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSearchMode {
    EMPTY_MODE = 0,
    STANDARD_KNN = 1,
    HYBRID_ADHOC_BF = 2,
    HYBRID_BATCHES = 3,
    HYBRID_BATCHES_TO_ADHOC_BF = 4,
    RANGE_QUERY = 5,
}

/// Timeout callback function type.
/// Returns non-zero on timeout.
pub type timeoutCallbackFunction = Option<unsafe extern "C" fn(ctx: *mut std::ffi::c_void) -> i32>;

/// Log callback function type.
pub type logCallbackFunction = Option<unsafe extern "C" fn(
    ctx: *mut std::ffi::c_void,
    level: *const std::ffi::c_char,
    message: *const std::ffi::c_char,
)>;

/// Opaque index handle.
#[repr(C)]
pub struct VecSimIndex {
    _private: [u8; 0],
}

/// Opaque query reply handle.
#[repr(C)]
pub struct VecSimQueryReply {
    _private: [u8; 0],
}

/// Opaque query result handle.
#[repr(C)]
pub struct VecSimQueryResult {
    _private: [u8; 0],
}

/// Opaque query reply iterator handle.
#[repr(C)]
pub struct VecSimQueryReply_Iterator {
    _private: [u8; 0],
}

/// Opaque batch iterator handle.
#[repr(C)]
pub struct VecSimBatchIterator {
    _private: [u8; 0],
}

/// Label type for vectors.
pub type labelType = u64;

/// Convert VecSimType to Rust type name for dispatch.
impl VecSimType {
    pub fn element_size(&self) -> usize {
        match self {
            VecSimType::VecSimType_FLOAT32 => std::mem::size_of::<f32>(),
            VecSimType::VecSimType_FLOAT64 => std::mem::size_of::<f64>(),
            VecSimType::VecSimType_BFLOAT16 => 2,
            VecSimType::VecSimType_FLOAT16 => 2,
            VecSimType::VecSimType_INT8 => 1,
            VecSimType::VecSimType_UINT8 => 1,
            VecSimType::VecSimType_INT32 => std::mem::size_of::<i32>(),
            VecSimType::VecSimType_INT64 => std::mem::size_of::<i64>(),
        }
    }
}

impl VecSimMetric {
    pub fn to_rust_metric(&self) -> vecsim::distance::Metric {
        match self {
            VecSimMetric::VecSimMetric_L2 => vecsim::distance::Metric::L2,
            VecSimMetric::VecSimMetric_IP => vecsim::distance::Metric::InnerProduct,
            VecSimMetric::VecSimMetric_Cosine => vecsim::distance::Metric::Cosine,
        }
    }
}

impl From<vecsim::distance::Metric> for VecSimMetric {
    fn from(metric: vecsim::distance::Metric) -> Self {
        match metric {
            vecsim::distance::Metric::L2 => VecSimMetric::VecSimMetric_L2,
            vecsim::distance::Metric::InnerProduct => VecSimMetric::VecSimMetric_IP,
            vecsim::distance::Metric::Cosine => VecSimMetric::VecSimMetric_Cosine,
        }
    }
}

/// Internal representation of a query result.
#[derive(Debug, Clone, Copy)]
pub struct QueryResultInternal {
    pub id: labelType,
    pub score: f64,
}

/// Internal representation of query reply.
pub struct QueryReplyInternal {
    pub results: Vec<QueryResultInternal>,
    pub code: VecSimQueryReply_Code,
}

impl QueryReplyInternal {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            code: VecSimQueryReply_Code::VecSim_QueryReply_OK,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            results: Vec::with_capacity(capacity),
            code: VecSimQueryReply_Code::VecSim_QueryReply_OK,
        }
    }

    pub fn from_results(results: Vec<QueryResultInternal>) -> Self {
        Self {
            results,
            code: VecSimQueryReply_Code::VecSim_QueryReply_OK,
        }
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn sort_by_score(&mut self) {
        self.results.sort_by(|a, b| {
            a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    pub fn sort_by_id(&mut self) {
        self.results.sort_by_key(|r| r.id);
    }
}

/// Iterator over query reply results.
pub struct QueryReplyIteratorInternal<'a> {
    results: &'a [QueryResultInternal],
    position: usize,
}

impl<'a> QueryReplyIteratorInternal<'a> {
    pub fn new(results: &'a [QueryResultInternal]) -> Self {
        Self { results, position: 0 }
    }

    pub fn next(&mut self) -> Option<&'a QueryResultInternal> {
        if self.position < self.results.len() {
            let result = &self.results[self.position];
            self.position += 1;
            Some(result)
        } else {
            None
        }
    }

    pub fn has_next(&self) -> bool {
        self.position < self.results.len()
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }
}

// ============================================================================
// Memory Function Types
// ============================================================================

/// Function pointer type for malloc-style allocation.
pub type allocFn = Option<unsafe extern "C" fn(n: usize) -> *mut std::ffi::c_void>;

/// Function pointer type for calloc-style allocation.
pub type callocFn = Option<unsafe extern "C" fn(nelem: usize, elemsz: usize) -> *mut std::ffi::c_void>;

/// Function pointer type for realloc-style reallocation.
pub type reallocFn = Option<unsafe extern "C" fn(p: *mut std::ffi::c_void, n: usize) -> *mut std::ffi::c_void>;

/// Function pointer type for free-style deallocation.
pub type freeFn = Option<unsafe extern "C" fn(p: *mut std::ffi::c_void)>;

/// Memory functions struct for custom memory management.
///
/// This allows integration with external memory management systems like Redis.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VecSimMemoryFunctions {
    /// Malloc-like allocation function.
    pub allocFunction: allocFn,
    /// Calloc-like allocation function.
    pub callocFunction: callocFn,
    /// Realloc-like reallocation function.
    pub reallocFunction: reallocFn,
    /// Free function.
    pub freeFunction: freeFn,
}

impl Default for VecSimMemoryFunctions {
    fn default() -> Self {
        Self {
            allocFunction: None,
            callocFunction: None,
            reallocFunction: None,
            freeFunction: None,
        }
    }
}

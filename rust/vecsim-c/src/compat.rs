//! C++-compatible type definitions for binary compatibility with the C++ VecSim API.
//!
//! These types use unions to match the exact memory layout of the C++ structs.
//! They are provided for drop-in compatibility with existing C++ code.
//!
//! For new Rust code, prefer the type-safe structs in `params.rs`.

use std::ffi::c_void;

use crate::types::{VecSimAlgo, VecSimMetric, VecSimOptionMode, VecSimType, VecSearchMode};

/// HNSW parameters matching C++ HNSWParams exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HNSWParams_C {
    pub type_: VecSimType,
    pub dim: usize,
    pub metric: VecSimMetric,
    pub multi: bool,
    pub initialCapacity: usize,
    pub blockSize: usize,
    pub M: usize,
    pub efConstruction: usize,
    pub efRuntime: usize,
    pub epsilon: f64,
}

/// BruteForce parameters matching C++ BFParams exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BFParams_C {
    pub type_: VecSimType,
    pub dim: usize,
    pub metric: VecSimMetric,
    pub multi: bool,
    pub initialCapacity: usize,
    pub blockSize: usize,
}

/// SVS quantization bits matching C++ VecSimSvsQuantBits.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimSvsQuantBits {
    VecSimSvsQuant_NONE = 0,
    VecSimSvsQuant_Scalar = 1,
    VecSimSvsQuant_4 = 4,
    VecSimSvsQuant_8 = 8,
    // Complex values handled as integers
}

/// SVS parameters matching C++ SVSParams exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SVSParams_C {
    pub type_: VecSimType,
    pub dim: usize,
    pub metric: VecSimMetric,
    pub multi: bool,
    pub blockSize: usize,
    pub quantBits: VecSimSvsQuantBits,
    pub alpha: f32,
    pub graph_max_degree: usize,
    pub construction_window_size: usize,
    pub max_candidate_pool_size: usize,
    pub prune_to: usize,
    pub use_search_history: VecSimOptionMode,
    pub num_threads: usize,
    pub search_window_size: usize,
    pub search_buffer_capacity: usize,
    pub leanvec_dim: usize,
    pub epsilon: f64,
}

/// Tiered HNSW specific parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TieredHNSWParams_C {
    pub swapJobThreshold: usize,
}

/// Tiered SVS specific parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TieredSVSParams_C {
    pub trainingTriggerThreshold: usize,
    pub updateTriggerThreshold: usize,
    pub updateJobWaitTime: usize,
}

/// Tiered HNSW Disk specific parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TieredHNSWDiskParams_C {
    pub _placeholder: u8,
}

/// Union of tiered-specific parameters.
#[repr(C)]
#[derive(Clone, Copy)]
pub union TieredSpecificParams_C {
    pub tieredHnswParams: TieredHNSWParams_C,
    pub tieredSVSParams: TieredSVSParams_C,
    pub tieredHnswDiskParams: TieredHNSWDiskParams_C,
}

/// Callback for submitting async jobs.
pub type SubmitCB = Option<
    unsafe extern "C" fn(
        job_queue: *mut c_void,
        index_ctx: *mut c_void,
        jobs: *mut *mut c_void,
        cbs: *mut *mut c_void,
        jobs_len: usize,
    ) -> i32,
>;

/// Tiered index parameters matching C++ TieredIndexParams.
#[repr(C)]
pub struct TieredIndexParams_C {
    pub jobQueue: *mut c_void,
    pub jobQueueCtx: *mut c_void,
    pub submitCb: SubmitCB,
    pub flatBufferLimit: usize,
    pub primaryIndexParams: *mut VecSimParams_C,
    pub specificParams: TieredSpecificParams_C,
}

/// Union of algorithm parameters matching C++ AlgoParams.
/// Note: This union cannot derive Copy because TieredIndexParams_C contains pointers.
/// Clone is implemented manually via unsafe memory copy.
#[repr(C)]
pub union AlgoParams_C {
    pub hnswParams: HNSWParams_C,
    pub bfParams: BFParams_C,
    pub tieredParams: std::mem::ManuallyDrop<TieredIndexParams_C>,
    pub svsParams: SVSParams_C,
}

impl Clone for AlgoParams_C {
    fn clone(&self) -> Self {
        // Safety: This is a C-compatible union, so we can safely copy the bytes.
        // The caller is responsible for ensuring the correct variant is used.
        unsafe {
            std::ptr::read(self)
        }
    }
}

/// VecSimParams matching C++ struct VecSimParams exactly.
#[repr(C)]
pub struct VecSimParams_C {
    pub algo: VecSimAlgo,
    pub algoParams: AlgoParams_C,
    pub logCtx: *mut c_void,
}

/// Disk context matching C++ VecSimDiskContext.
#[repr(C)]
pub struct VecSimDiskContext_C {
    pub storage: *mut c_void,
    pub indexName: *const std::ffi::c_char,
    pub indexNameLen: usize,
}

/// Disk parameters matching C++ VecSimParamsDisk.
#[repr(C)]
pub struct VecSimParamsDisk_C {
    pub indexParams: *mut VecSimParams_C,
    pub diskContext: *mut VecSimDiskContext_C,
}

/// HNSW runtime parameters matching C++ HNSWRuntimeParams.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HNSWRuntimeParams_C {
    pub efRuntime: usize,
    pub epsilon: f64,
}

/// SVS runtime parameters matching C++ SVSRuntimeParams.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SVSRuntimeParams_C {
    pub windowSize: usize,
    pub bufferCapacity: usize,
    pub searchHistory: VecSimOptionMode,
    pub epsilon: f64,
}

/// Union of runtime parameters (anonymous union in C++).
#[repr(C)]
#[derive(Clone, Copy)]
pub union RuntimeParams_C {
    pub hnswRuntimeParams: HNSWRuntimeParams_C,
    pub svsRuntimeParams: SVSRuntimeParams_C,
}

impl Default for RuntimeParams_C {
    fn default() -> Self {
        RuntimeParams_C {
            hnswRuntimeParams: HNSWRuntimeParams_C::default(),
        }
    }
}

/// Query parameters matching C++ VecSimQueryParams.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VecSimQueryParams_C {
    pub runtimeParams: RuntimeParams_C,
    pub batchSize: usize,
    pub searchMode: VecSearchMode,
    pub timeoutCtx: *mut c_void,
}

impl Default for VecSimQueryParams_C {
    fn default() -> Self {
        VecSimQueryParams_C {
            runtimeParams: RuntimeParams_C::default(),
            batchSize: 0,
            searchMode: VecSearchMode::EMPTY_MODE,
            timeoutCtx: std::ptr::null_mut(),
        }
    }
}


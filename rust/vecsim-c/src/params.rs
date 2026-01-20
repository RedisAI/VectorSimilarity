//! C-compatible parameter structs for index creation.

use crate::types::{VecSimAlgo, VecSimMetric, VecSimType};

/// Common base parameters for all index types.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VecSimParams {
    /// Algorithm type (BF or HNSW).
    pub algo: VecSimAlgo,
    /// Data type for vectors.
    pub type_: VecSimType,
    /// Distance metric.
    pub metric: VecSimMetric,
    /// Vector dimension.
    pub dim: usize,
    /// Whether this is a multi-value index.
    pub multi: bool,
    /// Initial capacity.
    pub initialCapacity: usize,
    /// Block size (0 for default).
    pub blockSize: usize,
}

impl Default for VecSimParams {
    fn default() -> Self {
        Self {
            algo: VecSimAlgo::VecSimAlgo_BF,
            type_: VecSimType::VecSimType_FLOAT32,
            metric: VecSimMetric::VecSimMetric_L2,
            dim: 0,
            multi: false,
            initialCapacity: 1024,
            blockSize: 0,
        }
    }
}

/// Parameters specific to BruteForce index.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BFParams {
    /// Common parameters.
    pub base: VecSimParams,
}

impl Default for BFParams {
    fn default() -> Self {
        Self {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_BF,
                ..VecSimParams::default()
            },
        }
    }
}

/// Parameters specific to HNSW index.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HNSWParams {
    /// Common parameters.
    pub base: VecSimParams,
    /// Maximum number of connections per element per layer (default: 16).
    pub M: usize,
    /// Size of the dynamic candidate list during construction (default: 200).
    pub efConstruction: usize,
    /// Size of the dynamic candidate list during search (default: 10).
    pub efRuntime: usize,
    /// Multiplier for epsilon (approximation factor, 0 = exact).
    pub epsilon: f64,
}

impl Default for HNSWParams {
    fn default() -> Self {
        Self {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_HNSWLIB,
                ..VecSimParams::default()
            },
            M: 16,
            efConstruction: 200,
            efRuntime: 10,
            epsilon: 0.0,
        }
    }
}

/// Parameters specific to SVS (Vamana) index.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SVSParams {
    /// Common parameters.
    pub base: VecSimParams,
    /// Maximum number of neighbors per node (R, default: 32).
    pub graphMaxDegree: usize,
    /// Alpha parameter for robust pruning (default: 1.2).
    pub alpha: f32,
    /// Beam width during construction (L, default: 200).
    pub constructionWindowSize: usize,
    /// Default beam width during search (default: 100).
    pub searchWindowSize: usize,
    /// Enable two-pass construction for better recall (default: true).
    pub twoPassConstruction: bool,
}

impl Default for SVSParams {
    fn default() -> Self {
        Self {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_SVS,
                ..VecSimParams::default()
            },
            graphMaxDegree: 32,
            alpha: 1.2,
            constructionWindowSize: 200,
            searchWindowSize: 100,
            twoPassConstruction: true,
        }
    }
}

/// Parameters specific to Tiered index.
///
/// The tiered index combines a BruteForce frontend (for fast writes) with
/// an HNSW backend (for efficient queries). Vectors are first added to the
/// flat buffer, then migrated to HNSW via flush() or when the buffer is full.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TieredParams {
    /// Common parameters (type, metric, dim, multi, etc.).
    pub base: VecSimParams,
    /// HNSW M parameter (max connections per node, default: 16).
    pub M: usize,
    /// HNSW ef_construction parameter (default: 200).
    pub efConstruction: usize,
    /// HNSW ef_runtime parameter (default: 10).
    pub efRuntime: usize,
    /// Maximum size of the flat buffer before forcing in-place writes.
    /// When flat buffer reaches this limit, new writes go directly to HNSW.
    /// Default: 10000.
    pub flatBufferLimit: usize,
    /// Write mode: 0 = Async (buffer first), 1 = InPlace (direct to HNSW).
    pub writeMode: u32,
}

impl Default for TieredParams {
    fn default() -> Self {
        Self {
            base: VecSimParams {
                algo: VecSimAlgo::VecSimAlgo_TIERED,
                ..VecSimParams::default()
            },
            M: 16,
            efConstruction: 200,
            efRuntime: 10,
            flatBufferLimit: 10000,
            writeMode: 0, // Async
        }
    }
}

/// Runtime parameters union (C++-compatible layout).
/// This union overlays HNSW and SVS runtime parameters in the same memory.
#[repr(C)]
#[derive(Clone, Copy)]
pub union RuntimeParamsUnion {
    pub hnswRuntimeParams: HNSWRuntimeParams,
    pub svsRuntimeParams: SVSRuntimeParams,
}

impl Default for RuntimeParamsUnion {
    fn default() -> Self {
        Self {
            hnswRuntimeParams: HNSWRuntimeParams::default(),
        }
    }
}

impl std::fmt::Debug for RuntimeParamsUnion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Safety: We can always read hnswRuntimeParams since it's the smaller type
        unsafe {
            f.debug_struct("RuntimeParamsUnion")
                .field("hnswRuntimeParams", &self.hnswRuntimeParams)
                .finish()
        }
    }
}

/// Query parameters (C++-compatible layout).
///
/// This struct matches the C++ VecSimQueryParams layout exactly:
/// - Union of HNSW and SVS runtime params
/// - batchSize
/// - searchMode (VecSearchMode enum)
/// - timeoutCtx
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VecSimQueryParams {
    /// Runtime parameters (union of HNSW and SVS params).
    pub runtimeParams: RuntimeParamsUnion,
    /// Batch size for batched iteration.
    pub batchSize: usize,
    /// Search mode (VecSearchMode enum from C++).
    pub searchMode: i32,  // VecSearchMode is an enum, use i32 for C compatibility
    /// Timeout callback (opaque pointer).
    pub timeoutCtx: *mut std::ffi::c_void,
}

impl Default for VecSimQueryParams {
    fn default() -> Self {
        Self {
            runtimeParams: RuntimeParamsUnion::default(),
            batchSize: 0,
            searchMode: 0,  // EMPTY_MODE
            timeoutCtx: std::ptr::null_mut(),
        }
    }
}

impl VecSimQueryParams {
    /// Get HNSW runtime parameters.
    pub fn hnsw_params(&self) -> &HNSWRuntimeParams {
        // Safety: Reading from union is safe as long as we interpret correctly
        unsafe { &self.runtimeParams.hnswRuntimeParams }
    }

    /// Get mutable HNSW runtime parameters.
    pub fn hnsw_params_mut(&mut self) -> &mut HNSWRuntimeParams {
        // Safety: Writing to union is safe
        unsafe { &mut self.runtimeParams.hnswRuntimeParams }
    }

    /// Get SVS runtime parameters.
    pub fn svs_params(&self) -> &SVSRuntimeParams {
        // Safety: Reading from union is safe as long as we interpret correctly
        unsafe { &self.runtimeParams.svsRuntimeParams }
    }

    /// Get mutable SVS runtime parameters.
    pub fn svs_params_mut(&mut self) -> &mut SVSRuntimeParams {
        // Safety: Writing to union is safe
        unsafe { &mut self.runtimeParams.svsRuntimeParams }
    }
}

/// HNSW-specific runtime parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HNSWRuntimeParams {
    /// Size of dynamic candidate list during search.
    pub efRuntime: usize,
    /// Epsilon multiplier for approximate search.
    pub epsilon: f64,
}

/// SVS-specific runtime parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SVSRuntimeParams {
    /// Search window size for SVS graph search.
    pub windowSize: usize,
    /// Search buffer capacity.
    pub bufferCapacity: usize,
    /// Whether to use search history.
    pub searchHistory: i32,
    /// Epsilon parameter for range search.
    pub epsilon: f64,
}

/// Search mode.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimSearchMode {
    STANDARD = 0,
    HYBRID = 1,
    RANGE = 2,
}

/// Hybrid policy.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimHybridPolicy {
    BATCHES = 0,
    ADHOC = 1,
}

/// Convert BFParams to Rust BruteForceParams.
impl BFParams {
    pub fn to_rust_params(&self) -> vecsim::index::BruteForceParams {
        let mut params = vecsim::index::BruteForceParams::new(
            self.base.dim,
            self.base.metric.to_rust_metric(),
        );
        params = params.with_capacity(self.base.initialCapacity);
        if self.base.blockSize > 0 {
            params = params.with_block_size(self.base.blockSize);
        }
        params
    }
}

/// Convert HNSWParams to Rust HnswParams.
impl HNSWParams {
    pub fn to_rust_params(&self) -> vecsim::index::HnswParams {
        let mut params = vecsim::index::HnswParams::new(
            self.base.dim,
            self.base.metric.to_rust_metric(),
        );
        params = params
            .with_m(self.M)
            .with_ef_construction(self.efConstruction)
            .with_ef_runtime(self.efRuntime)
            .with_capacity(self.base.initialCapacity);
        params
    }
}

/// Convert SVSParams to Rust SvsParams.
impl SVSParams {
    pub fn to_rust_params(&self) -> vecsim::index::SvsParams {
        vecsim::index::SvsParams::new(self.base.dim, self.base.metric.to_rust_metric())
            .with_graph_degree(self.graphMaxDegree)
            .with_alpha(self.alpha)
            .with_construction_l(self.constructionWindowSize)
            .with_search_l(self.searchWindowSize)
            .with_capacity(self.base.initialCapacity)
            .with_two_pass(self.twoPassConstruction)
    }
}

/// Convert TieredParams to Rust TieredParams.
impl TieredParams {
    pub fn to_rust_params(&self) -> vecsim::index::TieredParams {
        let write_mode = if self.writeMode == 0 {
            vecsim::index::tiered::WriteMode::Async
        } else {
            vecsim::index::tiered::WriteMode::InPlace
        };

        vecsim::index::TieredParams::new(self.base.dim, self.base.metric.to_rust_metric())
            .with_m(self.M)
            .with_ef_construction(self.efConstruction)
            .with_ef_runtime(self.efRuntime)
            .with_flat_buffer_limit(self.flatBufferLimit)
            .with_write_mode(write_mode)
            .with_initial_capacity(self.base.initialCapacity)
    }
}

/// Convert VecSimQueryParams to Rust QueryParams.
impl VecSimQueryParams {
    pub fn to_rust_params(&self) -> vecsim::query::QueryParams {
        let mut params = vecsim::query::QueryParams::new();
        // Safety: Reading from union - we check efRuntime which is valid for both HNSW and SVS
        let hnsw_params = self.hnsw_params();
        if hnsw_params.efRuntime > 0 {
            params = params.with_ef_runtime(hnsw_params.efRuntime);
        }
        if self.batchSize > 0 {
            params = params.with_batch_size(self.batchSize);
        }
        params
    }
}

/// Backend type for disk-based indices.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskBackend {
    /// Linear scan (exact results).
    DiskBackend_BruteForce = 0,
    /// Vamana graph (approximate, fast).
    DiskBackend_Vamana = 1,
}

impl Default for DiskBackend {
    fn default() -> Self {
        DiskBackend::DiskBackend_BruteForce
    }
}

impl DiskBackend {
    pub fn to_rust_backend(&self) -> vecsim::index::disk::DiskBackend {
        match self {
            DiskBackend::DiskBackend_BruteForce => vecsim::index::disk::DiskBackend::BruteForce,
            DiskBackend::DiskBackend_Vamana => vecsim::index::disk::DiskBackend::Vamana,
        }
    }
}

/// Parameters for disk-based index.
///
/// Disk indices store vectors in memory-mapped files for persistence.
/// They support two backends:
/// - BruteForce: Linear scan (exact results, O(n))
/// - Vamana: Graph-based approximate search (fast, O(log n))
#[repr(C)]
#[derive(Debug, Clone)]
pub struct DiskParams {
    /// Common parameters (type, metric, dim, etc.).
    pub base: VecSimParams,
    /// Path to the data file (null-terminated C string).
    pub dataPath: *const std::ffi::c_char,
    /// Backend algorithm (BruteForce or Vamana).
    pub backend: DiskBackend,
    /// Graph max degree (for Vamana backend, default: 32).
    pub graphMaxDegree: usize,
    /// Alpha parameter for Vamana (default: 1.2).
    pub alpha: f32,
    /// Construction window size for Vamana (default: 200).
    pub constructionL: usize,
    /// Search window size for Vamana (default: 100).
    pub searchL: usize,
}

impl Default for DiskParams {
    fn default() -> Self {
        Self {
            base: VecSimParams::default(),
            dataPath: std::ptr::null(),
            backend: DiskBackend::default(),
            graphMaxDegree: 32,
            alpha: 1.2,
            constructionL: 200,
            searchL: 100,
        }
    }
}

impl DiskParams {
    /// Convert to Rust DiskIndexParams.
    ///
    /// # Safety
    /// The dataPath must be a valid null-terminated C string.
    pub unsafe fn to_rust_params(&self) -> Option<vecsim::index::disk::DiskIndexParams> {
        if self.dataPath.is_null() {
            return None;
        }

        let path_cstr = std::ffi::CStr::from_ptr(self.dataPath);
        let path_str = path_cstr.to_str().ok()?;

        let params = vecsim::index::disk::DiskIndexParams::new(
            self.base.dim,
            self.base.metric.to_rust_metric(),
            path_str,
        )
        .with_backend(self.backend.to_rust_backend())
        .with_capacity(self.base.initialCapacity)
        .with_graph_degree(self.graphMaxDegree)
        .with_alpha(self.alpha)
        .with_construction_l(self.constructionL)
        .with_search_l(self.searchL);

        Some(params)
    }
}

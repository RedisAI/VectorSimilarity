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

/// Query parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VecSimQueryParams {
    /// For HNSW: ef_runtime parameter.
    pub hnswRuntimeParams: HNSWRuntimeParams,
    /// Search mode (batch vs ad-hoc).
    pub searchMode: VecSimSearchMode,
    /// Hybrid policy.
    pub hybridPolicy: VecSimHybridPolicy,
    /// Batch size for batched iteration.
    pub batchSize: usize,
    /// Timeout callback (opaque pointer).
    pub timeoutCtx: *mut std::ffi::c_void,
}

impl Default for VecSimQueryParams {
    fn default() -> Self {
        Self {
            hnswRuntimeParams: HNSWRuntimeParams::default(),
            searchMode: VecSimSearchMode::STANDARD,
            hybridPolicy: VecSimHybridPolicy::BATCHES,
            batchSize: 0,
            timeoutCtx: std::ptr::null_mut(),
        }
    }
}

/// HNSW-specific runtime parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HNSWRuntimeParams {
    /// Size of dynamic candidate list during search.
    pub efRuntime: usize,
    /// Epsilon multiplier for approximate search.
    pub epsilon: f64,
}

impl Default for HNSWRuntimeParams {
    fn default() -> Self {
        Self {
            efRuntime: 10,
            epsilon: 0.0,
        }
    }
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

/// Convert VecSimQueryParams to Rust QueryParams.
impl VecSimQueryParams {
    pub fn to_rust_params(&self) -> vecsim::query::QueryParams {
        let mut params = vecsim::query::QueryParams::new();
        if self.hnswRuntimeParams.efRuntime > 0 {
            params = params.with_ef_runtime(self.hnswRuntimeParams.efRuntime);
        }
        if self.batchSize > 0 {
            params = params.with_batch_size(self.batchSize);
        }
        params
    }
}

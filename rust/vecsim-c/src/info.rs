//! Index information and introspection functions for C FFI.

use crate::index::IndexHandle;
use crate::types::{VecSimAlgo, VecSimMetric, VecSimType};

/// Index information struct.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VecSimIndexInfo {
    /// Current number of vectors in the index.
    pub indexSize: usize,
    /// Current label count (same as indexSize for single-value indices).
    pub indexLabelCount: usize,
    /// Vector dimension.
    pub dim: usize,
    /// Data type.
    pub type_: VecSimType,
    /// Algorithm type.
    pub algo: VecSimAlgo,
    /// Distance metric.
    pub metric: VecSimMetric,
    /// Whether this is a multi-value index.
    pub isMulti: bool,
    /// Block size.
    pub blockSize: usize,
    /// Memory usage in bytes.
    pub memory: usize,
    /// HNSW-specific info (if applicable).
    pub hnswInfo: VecSimHnswInfo,
}

/// HNSW-specific index information.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct VecSimHnswInfo {
    /// M parameter.
    pub M: usize,
    /// ef_construction parameter.
    pub efConstruction: usize,
    /// ef_runtime parameter.
    pub efRuntime: usize,
    /// Maximum level in the graph.
    pub maxLevel: usize,
    /// Entry point ID.
    pub entrypoint: i64,
    /// Epsilon parameter.
    pub epsilon: f64,
}

impl Default for VecSimIndexInfo {
    fn default() -> Self {
        Self {
            indexSize: 0,
            indexLabelCount: 0,
            dim: 0,
            type_: VecSimType::VecSimType_FLOAT32,
            algo: VecSimAlgo::VecSimAlgo_BF,
            metric: VecSimMetric::VecSimMetric_L2,
            isMulti: false,
            blockSize: 0,
            memory: 0,
            hnswInfo: VecSimHnswInfo::default(),
        }
    }
}

/// Get index information.
pub fn get_index_info(handle: &IndexHandle) -> VecSimIndexInfo {
    let wrapper = &handle.wrapper;

    VecSimIndexInfo {
        indexSize: wrapper.index_size(),
        indexLabelCount: wrapper.index_size(), // For single-value, same as size
        dim: wrapper.dimension(),
        type_: handle.data_type,
        algo: handle.algo,
        metric: handle.metric,
        isMulti: handle.is_multi,
        blockSize: 0, // Not directly exposed
        memory: wrapper.memory_usage(),
        hnswInfo: VecSimHnswInfo::default(), // Would need more specific introspection
    }
}

/// Get basic index statistics.
pub fn get_index_size(handle: &IndexHandle) -> usize {
    handle.wrapper.index_size()
}

/// Get vector dimension.
pub fn get_index_dim(handle: &IndexHandle) -> usize {
    handle.wrapper.dimension()
}

/// Get data type.
pub fn get_index_type(handle: &IndexHandle) -> VecSimType {
    handle.data_type
}

/// Get metric.
pub fn get_index_metric(handle: &IndexHandle) -> VecSimMetric {
    handle.metric
}

/// Check if index is multi-value.
pub fn is_index_multi(handle: &IndexHandle) -> bool {
    handle.is_multi
}

/// Check if label exists in index.
pub fn index_contains(handle: &IndexHandle, label: u64) -> bool {
    handle.wrapper.contains(label)
}

/// Get count of vectors with given label.
pub fn get_label_count(handle: &IndexHandle, label: u64) -> usize {
    handle.wrapper.label_count(label)
}

/// Get memory usage.
pub fn get_memory_usage(handle: &IndexHandle) -> usize {
    handle.wrapper.memory_usage()
}

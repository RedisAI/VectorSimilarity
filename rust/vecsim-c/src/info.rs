//! Index information and introspection functions for C FFI.

use std::ffi::CString;

use crate::index::IndexHandle;
use crate::types::{VecSearchMode, VecSimAlgo, VecSimMetric, VecSimType};

// ============================================================================
// Debug Info Iterator Types
// ============================================================================

/// Field type for debug info fields.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSim_InfoFieldType {
    INFOFIELD_STRING = 0,
    INFOFIELD_INT64 = 1,
    INFOFIELD_UINT64 = 2,
    INFOFIELD_FLOAT64 = 3,
    INFOFIELD_ITERATOR = 4,
}

/// Union of field values.
#[repr(C)]
#[derive(Clone, Copy)]
pub union FieldValue {
    pub floatingPointValue: f64,
    pub integerValue: i64,
    pub uintegerValue: u64,
    pub stringValue: *const std::ffi::c_char,
    pub iteratorValue: *mut VecSimDebugInfoIterator,
}

impl Default for FieldValue {
    fn default() -> Self {
        FieldValue { uintegerValue: 0 }
    }
}

/// A field in the debug info iterator.
#[repr(C)]
pub struct VecSim_InfoField {
    pub fieldName: *const std::ffi::c_char,
    pub fieldType: VecSim_InfoFieldType,
    pub fieldValue: FieldValue,
}

/// Internal representation of a field with owned strings.
pub struct InfoFieldOwned {
    pub name: CString,
    pub field_type: VecSim_InfoFieldType,
    pub value: InfoFieldValue,
}

/// Internal value representation.
pub enum InfoFieldValue {
    String(CString),
    Int64(i64),
    UInt64(u64),
    Float64(f64),
    Iterator(Box<VecSimDebugInfoIterator>),
}

/// Debug info iterator - an opaque type that holds fields for iteration.
pub struct VecSimDebugInfoIterator {
    fields: Vec<InfoFieldOwned>,
    current_index: usize,
    /// Cached C representation of current field (for returning pointers).
    current_field: Option<VecSim_InfoField>,
}

impl VecSimDebugInfoIterator {
    /// Create a new iterator with the given capacity.
    pub fn new(capacity: usize) -> Self {
        VecSimDebugInfoIterator {
            fields: Vec::with_capacity(capacity),
            current_index: 0,
            current_field: None,
        }
    }

    /// Add a string field.
    pub fn add_string_field(&mut self, name: &str, value: &str) {
        if let (Ok(name_c), Ok(value_c)) = (CString::new(name), CString::new(value)) {
            self.fields.push(InfoFieldOwned {
                name: name_c,
                field_type: VecSim_InfoFieldType::INFOFIELD_STRING,
                value: InfoFieldValue::String(value_c),
            });
        }
    }

    /// Add an unsigned integer field.
    pub fn add_uint64_field(&mut self, name: &str, value: u64) {
        if let Ok(name_c) = CString::new(name) {
            self.fields.push(InfoFieldOwned {
                name: name_c,
                field_type: VecSim_InfoFieldType::INFOFIELD_UINT64,
                value: InfoFieldValue::UInt64(value),
            });
        }
    }

    /// Add a signed integer field.
    pub fn add_int64_field(&mut self, name: &str, value: i64) {
        if let Ok(name_c) = CString::new(name) {
            self.fields.push(InfoFieldOwned {
                name: name_c,
                field_type: VecSim_InfoFieldType::INFOFIELD_INT64,
                value: InfoFieldValue::Int64(value),
            });
        }
    }

    /// Add a float field.
    pub fn add_float64_field(&mut self, name: &str, value: f64) {
        if let Ok(name_c) = CString::new(name) {
            self.fields.push(InfoFieldOwned {
                name: name_c,
                field_type: VecSim_InfoFieldType::INFOFIELD_FLOAT64,
                value: InfoFieldValue::Float64(value),
            });
        }
    }

    /// Add a nested iterator field.
    pub fn add_iterator_field(&mut self, name: &str, value: VecSimDebugInfoIterator) {
        if let Ok(name_c) = CString::new(name) {
            self.fields.push(InfoFieldOwned {
                name: name_c,
                field_type: VecSim_InfoFieldType::INFOFIELD_ITERATOR,
                value: InfoFieldValue::Iterator(Box::new(value)),
            });
        }
    }

    /// Get the number of fields.
    pub fn number_of_fields(&self) -> usize {
        self.fields.len()
    }

    /// Check if there are more fields.
    pub fn has_next(&self) -> bool {
        self.current_index < self.fields.len()
    }

    /// Get the next field, returning a pointer to a cached VecSim_InfoField.
    /// The returned pointer is valid until the next call to next() or until
    /// the iterator is freed.
    pub fn next(&mut self) -> *mut VecSim_InfoField {
        if self.current_index >= self.fields.len() {
            return std::ptr::null_mut();
        }

        let field = &self.fields[self.current_index];
        self.current_index += 1;

        let field_value = match &field.value {
            InfoFieldValue::String(s) => FieldValue {
                stringValue: s.as_ptr(),
            },
            InfoFieldValue::Int64(v) => FieldValue { integerValue: *v },
            InfoFieldValue::UInt64(v) => FieldValue { uintegerValue: *v },
            InfoFieldValue::Float64(v) => FieldValue {
                floatingPointValue: *v,
            },
            InfoFieldValue::Iterator(iter) => FieldValue {
                // Return a raw pointer to the boxed iterator
                iteratorValue: iter.as_ref() as *const VecSimDebugInfoIterator
                    as *mut VecSimDebugInfoIterator,
            },
        };

        self.current_field = Some(VecSim_InfoField {
            fieldName: field.name.as_ptr(),
            fieldType: field.field_type,
            fieldValue: field_value,
        });

        self.current_field.as_mut().unwrap() as *mut VecSim_InfoField
    }
}

/// Helper function to get algorithm name as string.
pub fn algo_to_string(algo: VecSimAlgo) -> &'static str {
    match algo {
        VecSimAlgo::VecSimAlgo_BF => "FLAT",
        VecSimAlgo::VecSimAlgo_HNSWLIB => "HNSW",
        VecSimAlgo::VecSimAlgo_TIERED => "TIERED",
        VecSimAlgo::VecSimAlgo_SVS => "SVS",
    }
}

/// Helper function to get type name as string.
pub fn type_to_string(t: VecSimType) -> &'static str {
    match t {
        VecSimType::VecSimType_FLOAT32 => "FLOAT32",
        VecSimType::VecSimType_FLOAT64 => "FLOAT64",
        VecSimType::VecSimType_BFLOAT16 => "BFLOAT16",
        VecSimType::VecSimType_FLOAT16 => "FLOAT16",
        VecSimType::VecSimType_INT8 => "INT8",
        VecSimType::VecSimType_UINT8 => "UINT8",
        VecSimType::VecSimType_INT32 => "INT32",
        VecSimType::VecSimType_INT64 => "INT64",
    }
}

/// Helper function to get metric name as string.
pub fn metric_to_string(m: VecSimMetric) -> &'static str {
    match m {
        VecSimMetric::VecSimMetric_L2 => "L2",
        VecSimMetric::VecSimMetric_IP => "IP",
        VecSimMetric::VecSimMetric_Cosine => "COSINE",
    }
}

/// Helper function to get search mode as string.
pub fn search_mode_to_string(mode: VecSearchMode) -> &'static str {
    match mode {
        VecSearchMode::EMPTY_MODE => "EMPTY_MODE",
        VecSearchMode::STANDARD_KNN => "STANDARD_KNN",
        VecSearchMode::HYBRID_ADHOC_BF => "HYBRID_ADHOC_BF",
        VecSearchMode::HYBRID_BATCHES => "HYBRID_BATCHES",
        VecSearchMode::HYBRID_BATCHES_TO_ADHOC_BF => "HYBRID_BATCHES_TO_ADHOC_BF",
        VecSearchMode::RANGE_QUERY => "RANGE_QUERY",
    }
}

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

/// Index info that is static and immutable.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VecSimIndexBasicInfo {
    pub algo: VecSimAlgo,
    pub metric: VecSimMetric,
    pub type_: VecSimType,
    pub isMulti: bool,
    pub isTiered: bool,
    pub isDisk: bool,
    pub blockSize: usize,
    pub dim: usize,
}

/// Index info for statistics - thin and efficient.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VecSimIndexStatsInfo {
    pub memory: usize,
    pub numberOfMarkedDeleted: usize,
}

/// Common index information.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CommonInfo {
    pub basicInfo: VecSimIndexBasicInfo,
    pub indexSize: usize,
    pub indexLabelCount: usize,
    pub memory: u64,
    pub lastMode: VecSearchMode,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct hnswInfoStruct {
    pub M: usize,
    pub efConstruction: usize,
    pub efRuntime: usize,
    pub epsilon: f64,
    pub max_level: usize,
    pub entrypoint: usize,
    pub visitedNodesPoolSize: usize,
    pub numberOfMarkedDeletedNodes: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct bfInfoStruct {
    pub dummy: i8,
}

/// Debug information for an index.
#[repr(C)]
pub struct VecSimIndexDebugInfo {
    pub commonInfo: CommonInfo,
    // Union would be here but we'll use a simpler approach
    pub hnswInfo: hnswInfoStruct,
    pub bfInfo: bfInfoStruct,
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

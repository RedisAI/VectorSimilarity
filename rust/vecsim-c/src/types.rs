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
}

/// Index algorithm type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimAlgo {
    VecSimAlgo_BF = 0,
    VecSimAlgo_HNSWLIB = 1,
    VecSimAlgo_SVS = 2,
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

/// Index resolve codes for resolving index state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSimResolveCode {
    VecSim_Resolve_OK = 0,
    VecSim_Resolve_ERR = 1,
}

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
}

impl QueryReplyInternal {
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            results: Vec::with_capacity(capacity),
        }
    }

    pub fn from_results(results: Vec<QueryResultInternal>) -> Self {
        Self { results }
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

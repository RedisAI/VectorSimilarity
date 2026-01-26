//! Python bindings for the VecSim library using PyO3.
//!
//! This module provides Python-compatible wrappers around the Rust VecSim library,
//! enabling high-performance vector similarity search from Python.

// Allow non-snake-case names for Python API compatibility with the C++ version
#![allow(non_snake_case)]
// Allow unused fields that are kept for potential future use or debugging
#![allow(dead_code)]

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter};
// Note: Arc and Mutex may be used for future thread-safe batch operations
use vecsim::prelude::*;
use vecsim::index::svs::{SvsMulti, SvsParams, SvsSingle};

// ============================================================================
// Constants
// ============================================================================

// Metric constants
const VECSIM_METRIC_L2: u32 = 0;
const VECSIM_METRIC_IP: u32 = 1;
const VECSIM_METRIC_COSINE: u32 = 2;

// Type constants
const VECSIM_TYPE_FLOAT32: u32 = 0;
const VECSIM_TYPE_FLOAT64: u32 = 1;
const VECSIM_TYPE_BFLOAT16: u32 = 2;
const VECSIM_TYPE_FLOAT16: u32 = 3;
const VECSIM_TYPE_INT8: u32 = 4;
const VECSIM_TYPE_UINT8: u32 = 5;

// Batch iterator order constants
const BY_SCORE: u32 = 0;
const BY_ID: u32 = 1;

// SVS quantization constants (placeholders for tests)
const VECSIM_SVS_QUANT_NONE: u32 = 0;
const VECSIM_SVS_QUANT_8: u32 = 1;
const VECSIM_SVS_QUANT_4: u32 = 2;

// VecSim option constants
const VECSIM_OPTION_OFF: u32 = 0;
const VECSIM_OPTION_ON: u32 = 1;
const VECSIM_OPTION_AUTO: u32 = 2;

// ============================================================================
// Helper functions
// ============================================================================

fn metric_from_u32(value: u32) -> PyResult<Metric> {
    match value {
        VECSIM_METRIC_L2 => Ok(Metric::L2),
        VECSIM_METRIC_IP => Ok(Metric::InnerProduct),
        VECSIM_METRIC_COSINE => Ok(Metric::Cosine),
        _ => Err(PyValueError::new_err(format!(
            "Invalid metric value: {}",
            value
        ))),
    }
}

fn metric_to_u32(metric: Metric) -> u32 {
    match metric {
        Metric::L2 => VECSIM_METRIC_L2,
        Metric::InnerProduct => VECSIM_METRIC_IP,
        Metric::Cosine => VECSIM_METRIC_COSINE,
    }
}

/// Helper to create a 2D numpy array from vectors
fn vec_to_2d_array<'py, T: numpy::Element>(
    py: Python<'py>,
    data: Vec<T>,
    cols: usize,
) -> Bound<'py, PyArray2<T>> {
    let rows = if cols == 0 { 1 } else { 1 };
    let arr = ndarray::Array2::from_shape_vec((rows, cols), data).unwrap();
    arr.into_pyarray(py)
}

// ============================================================================
// Parameter Classes
// ============================================================================

/// Parameters for BruteForce index creation.
#[pyclass]
#[derive(Clone)]
pub struct BFParams {
    #[pyo3(get, set)]
    pub dim: usize,
    #[pyo3(get, set)]
    pub r#type: u32,
    #[pyo3(get, set)]
    pub metric: u32,
    #[pyo3(get, set)]
    pub multi: bool,
    #[pyo3(get, set)]
    pub blockSize: usize,
}

#[pymethods]
impl BFParams {
    #[new]
    fn new() -> Self {
        BFParams {
            dim: 0,
            r#type: VECSIM_TYPE_FLOAT32,
            metric: VECSIM_METRIC_L2,
            multi: false,
            blockSize: 1024,
        }
    }
}

/// Parameters for HNSW index creation.
#[pyclass]
#[derive(Clone)]
pub struct HNSWParams {
    #[pyo3(get, set)]
    pub dim: usize,
    #[pyo3(get, set)]
    pub r#type: u32,
    #[pyo3(get, set)]
    pub metric: u32,
    #[pyo3(get, set)]
    pub multi: bool,
    #[pyo3(get, set)]
    pub M: usize,
    #[pyo3(get, set)]
    pub efConstruction: usize,
    #[pyo3(get, set)]
    pub efRuntime: usize,
    #[pyo3(get, set)]
    pub epsilon: f64,
}

#[pymethods]
impl HNSWParams {
    #[new]
    fn new() -> Self {
        HNSWParams {
            dim: 0,
            r#type: VECSIM_TYPE_FLOAT32,
            metric: VECSIM_METRIC_L2,
            multi: false,
            M: 16,
            efConstruction: 200,
            efRuntime: 10,
            epsilon: 0.01,
        }
    }
}

/// HNSW runtime parameters for query operations.
#[pyclass]
#[derive(Clone)]
pub struct HNSWRuntimeParams {
    #[pyo3(get, set)]
    pub efRuntime: usize,
    #[pyo3(get, set)]
    pub epsilon: f64,
}

#[pymethods]
impl HNSWRuntimeParams {
    #[new]
    fn new() -> Self {
        HNSWRuntimeParams {
            efRuntime: 10,
            epsilon: 0.01,
        }
    }
}

/// SVS runtime parameters for query operations.
#[pyclass]
#[derive(Clone)]
pub struct SVSRuntimeParams {
    #[pyo3(get, set)]
    pub windowSize: usize,
    #[pyo3(get, set)]
    pub bufferCapacity: usize,
    #[pyo3(get, set)]
    pub epsilon: f64,
    #[pyo3(get, set)]
    pub searchHistory: u32,
}

#[pymethods]
impl SVSRuntimeParams {
    #[new]
    fn new() -> Self {
        SVSRuntimeParams {
            windowSize: 100,
            bufferCapacity: 100,
            epsilon: 0.01,
            searchHistory: VECSIM_OPTION_AUTO,
        }
    }
}

/// Parameters for Tiered HNSW index.
#[pyclass]
#[derive(Clone)]
pub struct TieredHNSWParams {
    #[pyo3(get, set)]
    pub swapJobThreshold: usize,
}

#[pymethods]
impl TieredHNSWParams {
    #[new]
    fn new() -> Self {
        TieredHNSWParams {
            swapJobThreshold: 0,
        }
    }
}

/// Query parameters for search operations.
#[pyclass]
pub struct VecSimQueryParams {
    hnsw_params: Py<HNSWRuntimeParams>,
    svs_params: Py<SVSRuntimeParams>,
}

#[pymethods]
impl VecSimQueryParams {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        let hnsw_params = Py::new(py, HNSWRuntimeParams::new())?;
        let svs_params = Py::new(py, SVSRuntimeParams::new())?;
        Ok(VecSimQueryParams { hnsw_params, svs_params })
    }

    /// Get the HNSW runtime parameters (returns a reference that can be mutated)
    #[getter]
    fn hnswRuntimeParams(&self, py: Python<'_>) -> Py<HNSWRuntimeParams> {
        self.hnsw_params.clone_ref(py)
    }

    /// Set the HNSW runtime parameters
    #[setter]
    fn set_hnswRuntimeParams(&mut self, _py: Python<'_>, params: &Bound<'_, HNSWRuntimeParams>) {
        self.hnsw_params = params.clone().unbind();
    }

    /// Get the SVS runtime parameters (returns a reference that can be mutated)
    #[getter]
    fn svsRuntimeParams(&self, py: Python<'_>) -> Py<SVSRuntimeParams> {
        self.svs_params.clone_ref(py)
    }

    /// Set the SVS runtime parameters
    #[setter]
    fn set_svsRuntimeParams(&mut self, _py: Python<'_>, params: &Bound<'_, SVSRuntimeParams>) {
        self.svs_params = params.clone().unbind();
    }

    /// Helper to get efRuntime directly for internal use
    fn get_ef_runtime(&self, py: Python<'_>) -> usize {
        self.hnsw_params.borrow(py).efRuntime
    }

    /// Helper to get SVS windowSize for internal use
    fn get_svs_window_size(&self, py: Python<'_>) -> usize {
        self.svs_params.borrow(py).windowSize
    }

    /// Helper to get SVS epsilon for internal use
    fn get_svs_epsilon(&self, py: Python<'_>) -> f64 {
        self.svs_params.borrow(py).epsilon
    }
}

// ============================================================================
// Type-erased Index Implementation
// ============================================================================

/// Macro to generate type-dispatched index operations
macro_rules! dispatch_bf_index {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match &$self.inner {
            BfIndexInner::SingleF32(idx) => idx.$method($($args),*),
            BfIndexInner::SingleF64(idx) => idx.$method($($args),*),
            BfIndexInner::SingleBF16(idx) => idx.$method($($args),*),
            BfIndexInner::SingleF16(idx) => idx.$method($($args),*),
            BfIndexInner::SingleI8(idx) => idx.$method($($args),*),
            BfIndexInner::SingleU8(idx) => idx.$method($($args),*),
            BfIndexInner::MultiF32(idx) => idx.$method($($args),*),
            BfIndexInner::MultiF64(idx) => idx.$method($($args),*),
            BfIndexInner::MultiBF16(idx) => idx.$method($($args),*),
            BfIndexInner::MultiF16(idx) => idx.$method($($args),*),
            BfIndexInner::MultiI8(idx) => idx.$method($($args),*),
            BfIndexInner::MultiU8(idx) => idx.$method($($args),*),
        }
    };
}

macro_rules! dispatch_bf_index_mut {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match &mut $self.inner {
            BfIndexInner::SingleF32(idx) => idx.$method($($args),*),
            BfIndexInner::SingleF64(idx) => idx.$method($($args),*),
            BfIndexInner::SingleBF16(idx) => idx.$method($($args),*),
            BfIndexInner::SingleF16(idx) => idx.$method($($args),*),
            BfIndexInner::SingleI8(idx) => idx.$method($($args),*),
            BfIndexInner::SingleU8(idx) => idx.$method($($args),*),
            BfIndexInner::MultiF32(idx) => idx.$method($($args),*),
            BfIndexInner::MultiF64(idx) => idx.$method($($args),*),
            BfIndexInner::MultiBF16(idx) => idx.$method($($args),*),
            BfIndexInner::MultiF16(idx) => idx.$method($($args),*),
            BfIndexInner::MultiI8(idx) => idx.$method($($args),*),
            BfIndexInner::MultiU8(idx) => idx.$method($($args),*),
        }
    };
}

enum BfIndexInner {
    SingleF32(BruteForceSingle<f32>),
    SingleF64(BruteForceSingle<f64>),
    SingleBF16(BruteForceSingle<BFloat16>),
    SingleF16(BruteForceSingle<Float16>),
    SingleI8(BruteForceSingle<Int8>),
    SingleU8(BruteForceSingle<UInt8>),
    MultiF32(BruteForceMulti<f32>),
    MultiF64(BruteForceMulti<f64>),
    MultiBF16(BruteForceMulti<BFloat16>),
    MultiF16(BruteForceMulti<Float16>),
    MultiI8(BruteForceMulti<Int8>),
    MultiU8(BruteForceMulti<UInt8>),
}

/// BruteForce index for exact nearest neighbor search.
#[pyclass]
pub struct BFIndex {
    inner: BfIndexInner,
    data_type: u32,
    metric: Metric,
    dim: usize,
}

#[pymethods]
impl BFIndex {
    #[new]
    fn new(params: &BFParams) -> PyResult<Self> {
        let metric = metric_from_u32(params.metric)?;
        let bf_params = BruteForceParams::new(params.dim, metric).with_capacity(params.blockSize);

        let inner = match (params.multi, params.r#type) {
            (false, VECSIM_TYPE_FLOAT32) => BfIndexInner::SingleF32(BruteForceSingle::new(bf_params)),
            (false, VECSIM_TYPE_FLOAT64) => BfIndexInner::SingleF64(BruteForceSingle::new(bf_params)),
            (false, VECSIM_TYPE_BFLOAT16) => BfIndexInner::SingleBF16(BruteForceSingle::new(bf_params)),
            (false, VECSIM_TYPE_FLOAT16) => BfIndexInner::SingleF16(BruteForceSingle::new(bf_params)),
            (false, VECSIM_TYPE_INT8) => BfIndexInner::SingleI8(BruteForceSingle::new(bf_params)),
            (false, VECSIM_TYPE_UINT8) => BfIndexInner::SingleU8(BruteForceSingle::new(bf_params)),
            (true, VECSIM_TYPE_FLOAT32) => BfIndexInner::MultiF32(BruteForceMulti::new(bf_params)),
            (true, VECSIM_TYPE_FLOAT64) => BfIndexInner::MultiF64(BruteForceMulti::new(bf_params)),
            (true, VECSIM_TYPE_BFLOAT16) => BfIndexInner::MultiBF16(BruteForceMulti::new(bf_params)),
            (true, VECSIM_TYPE_FLOAT16) => BfIndexInner::MultiF16(BruteForceMulti::new(bf_params)),
            (true, VECSIM_TYPE_INT8) => BfIndexInner::MultiI8(BruteForceMulti::new(bf_params)),
            (true, VECSIM_TYPE_UINT8) => BfIndexInner::MultiU8(BruteForceMulti::new(bf_params)),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported data type: {}",
                    params.r#type
                )))
            }
        };

        Ok(BFIndex {
            inner,
            data_type: params.r#type,
            metric,
            dim: params.dim,
        })
    }

    /// Add a vector to the index.
    fn add_vector(&mut self, py: Python<'_>, vector: PyObject, label: u64) -> PyResult<()> {
        match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let arr: PyReadonlyArray1<f32> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                match &mut self.inner {
                    BfIndexInner::SingleF32(idx) => idx.add_vector(slice, label),
                    BfIndexInner::MultiF32(idx) => idx.add_vector(slice, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT64 => {
                let arr: PyReadonlyArray1<f64> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                match &mut self.inner {
                    BfIndexInner::SingleF64(idx) => idx.add_vector(slice, label),
                    BfIndexInner::MultiF64(idx) => idx.add_vector(slice, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_BFLOAT16 => {
                // Try to extract as u16 first, or use raw bytes for bfloat16 dtype
                let bf16_vec = extract_bf16_vector(py, &vector)?;
                match &mut self.inner {
                    BfIndexInner::SingleBF16(idx) => idx.add_vector(&bf16_vec, label),
                    BfIndexInner::MultiBF16(idx) => idx.add_vector(&bf16_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT16 => {
                // Try to extract as u16 first, or use raw bytes for float16 dtype
                let f16_vec = extract_f16_vector(py, &vector)?;
                match &mut self.inner {
                    BfIndexInner::SingleF16(idx) => idx.add_vector(&f16_vec, label),
                    BfIndexInner::MultiF16(idx) => idx.add_vector(&f16_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_INT8 => {
                let arr: PyReadonlyArray1<i8> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                let i8_vec: Vec<Int8> = slice.iter().map(|&v| Int8(v)).collect();
                match &mut self.inner {
                    BfIndexInner::SingleI8(idx) => idx.add_vector(&i8_vec, label),
                    BfIndexInner::MultiI8(idx) => idx.add_vector(&i8_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_UINT8 => {
                let arr: PyReadonlyArray1<u8> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                let u8_vec: Vec<UInt8> = slice.iter().map(|&v| UInt8(v)).collect();
                match &mut self.inner {
                    BfIndexInner::SingleU8(idx) => idx.add_vector(&u8_vec, label),
                    BfIndexInner::MultiU8(idx) => idx.add_vector(&u8_vec, label),
                    _ => unreachable!(),
                }
            }
            _ => return Err(PyValueError::new_err("Unsupported data type")),
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to add vector: {:?}", e)))?;
        Ok(())
    }

    /// Delete a vector from the index.
    fn delete_vector(&mut self, label: u64) -> PyResult<usize> {
        dispatch_bf_index_mut!(self, delete_vector, label)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete vector: {:?}", e)))
    }

    /// Perform a k-nearest neighbors query.
    #[pyo3(signature = (query, k=10))]
    fn knn_query<'py>(
        &self,
        py: Python<'py>,
        query: PyObject,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        let query_vec = extract_query_vec(py, &query)?;
        let reply = self.query_internal(&query_vec, k)?;

        let labels: Vec<i64> = reply.results.iter().map(|r| r.label as i64).collect();
        let distances: Vec<f64> = reply.results.iter().map(|r| r.distance).collect();
        let len = labels.len();

        Ok((vec_to_2d_array(py, labels, len), vec_to_2d_array(py, distances, len)))
    }

    /// Perform a range query.
    #[pyo3(signature = (query, radius))]
    fn range_query<'py>(
        &self,
        py: Python<'py>,
        query: PyObject,
        radius: f64,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        let query_vec = extract_query_vec(py, &query)?;
        let reply = self.range_query_internal(&query_vec, radius)?;

        let labels: Vec<i64> = reply.results.iter().map(|r| r.label as i64).collect();
        let distances: Vec<f64> = reply.results.iter().map(|r| r.distance).collect();
        let len = labels.len();

        Ok((vec_to_2d_array(py, labels, len), vec_to_2d_array(py, distances, len)))
    }

    /// Get the number of vectors in the index.
    fn index_size(&self) -> usize {
        dispatch_bf_index!(self, index_size)
    }

    /// Get the data type of vectors in the index.
    fn index_type(&self) -> u32 {
        self.data_type
    }

    /// Create a batch iterator for streaming results.
    fn create_batch_iterator(&self, py: Python<'_>, query: PyObject) -> PyResult<PyBatchIterator> {
        let query_vec = extract_query_vec(py, &query)?;
        let all_results = self.get_all_results_sorted(&query_vec)?;
        let index_size = self.index_size();

        Ok(PyBatchIterator {
            results: all_results,
            position: 0,
            index_size,
        })
    }
}

/// Extract BFloat16 vector from numpy array (handles ml_dtypes.bfloat16)
fn extract_bf16_vector(py: Python<'_>, vector: &PyObject) -> PyResult<Vec<BFloat16>> {
    // First try as u16 (raw bits)
    if let Ok(arr) = vector.extract::<PyReadonlyArray1<u16>>(py) {
        let slice = arr.as_slice()?;
        return Ok(slice.iter().map(|&v| BFloat16::from_bits(v)).collect());
    }

    // Otherwise get raw bytes and interpret as u16
    let arr = vector.bind(py);
    let nbytes: usize = arr.getattr("nbytes")?.extract()?;
    let len = nbytes / 2; // 2 bytes per u16

    // Get the data buffer as bytes and reinterpret
    let tobytes_fn = arr.getattr("tobytes")?;
    let bytes: Vec<u8> = tobytes_fn.call0()?.extract()?;

    let bf16_vec: Vec<BFloat16> = bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            BFloat16::from_bits(bits)
        })
        .collect();

    if bf16_vec.len() != len {
        return Err(PyValueError::new_err("Failed to convert to BFloat16 vector"));
    }

    Ok(bf16_vec)
}

/// Extract Float16 vector from numpy array (handles np.float16)
fn extract_f16_vector(py: Python<'_>, vector: &PyObject) -> PyResult<Vec<Float16>> {
    // First try as u16 (raw bits)
    if let Ok(arr) = vector.extract::<PyReadonlyArray1<u16>>(py) {
        let slice = arr.as_slice()?;
        return Ok(slice.iter().map(|&v| Float16::from_bits(v)).collect());
    }

    // Otherwise get raw bytes and interpret as u16 (works for np.float16)
    let arr = vector.bind(py);
    let nbytes: usize = arr.getattr("nbytes")?.extract()?;
    let len = nbytes / 2; // 2 bytes per u16

    // Get the data buffer as bytes and reinterpret
    let tobytes_fn = arr.getattr("tobytes")?;
    let bytes: Vec<u8> = tobytes_fn.call0()?.extract()?;

    let f16_vec: Vec<Float16> = bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            Float16::from_bits(bits)
        })
        .collect();

    if f16_vec.len() != len {
        return Err(PyValueError::new_err("Failed to convert to Float16 vector"));
    }

    Ok(f16_vec)
}

/// Extract query vector from various numpy array types.
/// For 2D arrays, extracts only the first row (single query vector).
fn extract_query_vec(py: Python<'_>, query: &PyObject) -> PyResult<Vec<f64>> {
    if let Ok(arr) = query.extract::<PyReadonlyArray2<f64>>(py) {
        // For 2D array, take only the first row
        let shape = arr.shape();
        let ncols = shape[1];
        Ok(arr.as_slice()?[..ncols].to_vec())
    } else if let Ok(arr) = query.extract::<PyReadonlyArray1<f64>>(py) {
        Ok(arr.as_slice()?.to_vec())
    } else if let Ok(arr) = query.extract::<PyReadonlyArray2<f32>>(py) {
        // For 2D array, take only the first row
        let shape = arr.shape();
        let ncols = shape[1];
        Ok(arr.as_slice()?[..ncols].iter().map(|&v| v as f64).collect())
    } else if let Ok(arr) = query.extract::<PyReadonlyArray1<f32>>(py) {
        Ok(arr.as_slice()?.iter().map(|&v| v as f64).collect())
    } else if let Ok(arr) = query.extract::<PyReadonlyArray2<u16>>(py) {
        // For 2D array, take only the first row
        let shape = arr.shape();
        let ncols = shape[1];
        Ok(arr.as_slice()?[..ncols].iter().map(|&v| half::f16::from_bits(v).to_f64()).collect())
    } else if let Ok(arr) = query.extract::<PyReadonlyArray1<u16>>(py) {
        Ok(arr.as_slice()?.iter().map(|&v| half::f16::from_bits(v).to_f64()).collect())
    } else if let Ok(arr) = query.extract::<PyReadonlyArray2<i8>>(py) {
        // For 2D array, take only the first row
        let shape = arr.shape();
        let ncols = shape[1];
        Ok(arr.as_slice()?[..ncols].iter().map(|&v| v as f64).collect())
    } else if let Ok(arr) = query.extract::<PyReadonlyArray1<i8>>(py) {
        Ok(arr.as_slice()?.iter().map(|&v| v as f64).collect())
    } else if let Ok(arr) = query.extract::<PyReadonlyArray2<u8>>(py) {
        // For 2D array, take only the first row
        let shape = arr.shape();
        let ncols = shape[1];
        Ok(arr.as_slice()?[..ncols].iter().map(|&v| v as f64).collect())
    } else if let Ok(arr) = query.extract::<PyReadonlyArray1<u8>>(py) {
        Ok(arr.as_slice()?.iter().map(|&v| v as f64).collect())
    } else {
        // Fallback: try to interpret as half-precision (bfloat16 or float16)
        // Both are 16-bit types stored in 2 bytes
        let arr = query.bind(py);
        let itemsize: usize = arr.getattr("itemsize")?.extract()?;
        if itemsize == 2 {
            // Check the dtype name to distinguish bfloat16 from float16
            let dtype = arr.getattr("dtype")?;
            let dtype_name: String = dtype.getattr("name")?.extract()?;

            // Get array shape to handle 2D arrays (take only first row)
            let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
            let num_elements = if shape.len() == 2 {
                shape[1]  // For 2D array, take only first row
            } else if shape.len() == 1 {
                shape[0]
            } else {
                return Err(PyValueError::new_err("Query array must be 1D or 2D"));
            };

            let tobytes_fn = arr.getattr("tobytes")?;
            let bytes: Vec<u8> = tobytes_fn.call0()?.extract()?;

            // Only take bytes for the first row
            let bytes_to_use = &bytes[..num_elements * 2];

            let result: Vec<f64> = if dtype_name.contains("bfloat16") {
                // BFloat16: 1 sign + 8 exponent + 7 mantissa
                bytes_to_use
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::bf16::from_bits(bits).to_f64()
                    })
                    .collect()
            } else {
                // Float16 (IEEE 754): 1 sign + 5 exponent + 10 mantissa
                bytes_to_use
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f64()
                    })
                    .collect()
            };
            return Ok(result);
        }
        Err(PyValueError::new_err("Unsupported query array type"))
    }
}

impl BFIndex {
    fn query_internal(&self, query: &[f64], k: usize) -> PyResult<QueryReply<f64>> {
        match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let q: Vec<f32> = query.iter().map(|&v| v as f32).collect();
                let reply = match &self.inner {
                    BfIndexInner::SingleF32(idx) => idx.top_k_query(&q, k, None),
                    BfIndexInner::MultiF32(idx) => idx.top_k_query(&q, k, None),
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(
                    reply
                        .results
                        .into_iter()
                        .map(|r| QueryResult::new(r.label, r.distance as f64))
                        .collect(),
                ))
            }
            VECSIM_TYPE_FLOAT64 => {
                let reply = match &self.inner {
                    BfIndexInner::SingleF64(idx) => idx.top_k_query(query, k, None),
                    BfIndexInner::MultiF64(idx) => idx.top_k_query(query, k, None),
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(reply)
            }
            VECSIM_TYPE_BFLOAT16 => {
                let q: Vec<BFloat16> = query.iter().map(|&v| BFloat16::from_f32(v as f32)).collect();
                let reply = match &self.inner {
                    BfIndexInner::SingleBF16(idx) => idx.top_k_query(&q, k, None),
                    BfIndexInner::MultiBF16(idx) => idx.top_k_query(&q, k, None),
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(
                    reply
                        .results
                        .into_iter()
                        .map(|r| QueryResult::new(r.label, r.distance.to_f64()))
                        .collect(),
                ))
            }
            VECSIM_TYPE_FLOAT16 => {
                let q: Vec<Float16> = query.iter().map(|&v| Float16::from_f32(v as f32)).collect();
                let reply = match &self.inner {
                    BfIndexInner::SingleF16(idx) => idx.top_k_query(&q, k, None),
                    BfIndexInner::MultiF16(idx) => idx.top_k_query(&q, k, None),
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(
                    reply
                        .results
                        .into_iter()
                        .map(|r| QueryResult::new(r.label, r.distance.to_f64()))
                        .collect(),
                ))
            }
            VECSIM_TYPE_INT8 => {
                let q: Vec<Int8> = query.iter().map(|&v| Int8(v as i8)).collect();
                let reply = match &self.inner {
                    BfIndexInner::SingleI8(idx) => idx.top_k_query(&q, k, None),
                    BfIndexInner::MultiI8(idx) => idx.top_k_query(&q, k, None),
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(
                    reply
                        .results
                        .into_iter()
                        .map(|r| QueryResult::new(r.label, r.distance as f64))
                        .collect(),
                ))
            }
            VECSIM_TYPE_UINT8 => {
                let q: Vec<UInt8> = query.iter().map(|&v| UInt8(v as u8)).collect();
                let reply = match &self.inner {
                    BfIndexInner::SingleU8(idx) => idx.top_k_query(&q, k, None),
                    BfIndexInner::MultiU8(idx) => idx.top_k_query(&q, k, None),
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(
                    reply
                        .results
                        .into_iter()
                        .map(|r| QueryResult::new(r.label, r.distance as f64))
                        .collect(),
                ))
            }
            _ => Err(PyValueError::new_err("Unsupported data type")),
        }
    }

    fn range_query_internal(&self, query: &[f64], radius: f64) -> PyResult<QueryReply<f64>> {
        match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let q: Vec<f32> = query.iter().map(|&v| v as f32).collect();
                let reply = match &self.inner {
                    BfIndexInner::SingleF32(idx) => idx.range_query(&q, radius as f32, None),
                    BfIndexInner::MultiF32(idx) => idx.range_query(&q, radius as f32, None),
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(
                    reply
                        .results
                        .into_iter()
                        .map(|r| QueryResult::new(r.label, r.distance as f64))
                        .collect(),
                ))
            }
            VECSIM_TYPE_FLOAT64 => {
                let reply = match &self.inner {
                    BfIndexInner::SingleF64(idx) => idx.range_query(query, radius, None),
                    BfIndexInner::MultiF64(idx) => idx.range_query(query, radius, None),
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(reply)
            }
            VECSIM_TYPE_BFLOAT16 => {
                let q: Vec<BFloat16> = query.iter().map(|&v| BFloat16::from_f32(v as f32)).collect();
                let reply = match &self.inner {
                    BfIndexInner::SingleBF16(idx) => {
                        idx.range_query(&q, radius as f32, None)
                    }
                    BfIndexInner::MultiBF16(idx) => {
                        idx.range_query(&q, radius as f32, None)
                    }
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(
                    reply
                        .results
                        .into_iter()
                        .map(|r| QueryResult::new(r.label, r.distance.to_f64()))
                        .collect(),
                ))
            }
            VECSIM_TYPE_FLOAT16 => {
                let q: Vec<Float16> = query.iter().map(|&v| Float16::from_f32(v as f32)).collect();
                let reply = match &self.inner {
                    BfIndexInner::SingleF16(idx) => {
                        idx.range_query(&q, radius as f32, None)
                    }
                    BfIndexInner::MultiF16(idx) => {
                        idx.range_query(&q, radius as f32, None)
                    }
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(
                    reply
                        .results
                        .into_iter()
                        .map(|r| QueryResult::new(r.label, r.distance.to_f64()))
                        .collect(),
                ))
            }
            VECSIM_TYPE_INT8 => {
                let q: Vec<Int8> = query.iter().map(|&v| Int8(v as i8)).collect();
                let reply = match &self.inner {
                    BfIndexInner::SingleI8(idx) => idx.range_query(&q, radius as f32, None),
                    BfIndexInner::MultiI8(idx) => idx.range_query(&q, radius as f32, None),
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(
                    reply
                        .results
                        .into_iter()
                        .map(|r| QueryResult::new(r.label, r.distance as f64))
                        .collect(),
                ))
            }
            VECSIM_TYPE_UINT8 => {
                let q: Vec<UInt8> = query.iter().map(|&v| UInt8(v as u8)).collect();
                let reply = match &self.inner {
                    BfIndexInner::SingleU8(idx) => idx.range_query(&q, radius as f32, None),
                    BfIndexInner::MultiU8(idx) => idx.range_query(&q, radius as f32, None),
                    _ => unreachable!(),
                }
                .map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(
                    reply
                        .results
                        .into_iter()
                        .map(|r| QueryResult::new(r.label, r.distance as f64))
                        .collect(),
                ))
            }
            _ => Err(PyValueError::new_err("Unsupported data type")),
        }
    }

    fn get_all_results_sorted(&self, query: &[f64]) -> PyResult<Vec<(u64, f64)>> {
        let k = self.index_size();
        if k == 0 {
            return Ok(Vec::new());
        }
        let reply = self.query_internal(query, k)?;
        Ok(reply
            .results
            .into_iter()
            .map(|r| (r.label, r.distance))
            .collect())
    }
}

// ============================================================================
// HNSW Index
// ============================================================================

enum HnswIndexInner {
    SingleF32(HnswSingle<f32>),
    SingleF64(HnswSingle<f64>),
    SingleBF16(HnswSingle<BFloat16>),
    SingleF16(HnswSingle<Float16>),
    SingleI8(HnswSingle<Int8>),
    SingleU8(HnswSingle<UInt8>),
    MultiF32(HnswMulti<f32>),
    MultiF64(HnswMulti<f64>),
    MultiBF16(HnswMulti<BFloat16>),
    MultiF16(HnswMulti<Float16>),
    MultiI8(HnswMulti<Int8>),
    MultiU8(HnswMulti<UInt8>),
}

macro_rules! dispatch_hnsw_index {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match &$self.inner {
            HnswIndexInner::SingleF32(idx) => idx.$method($($args),*),
            HnswIndexInner::SingleF64(idx) => idx.$method($($args),*),
            HnswIndexInner::SingleBF16(idx) => idx.$method($($args),*),
            HnswIndexInner::SingleF16(idx) => idx.$method($($args),*),
            HnswIndexInner::SingleI8(idx) => idx.$method($($args),*),
            HnswIndexInner::SingleU8(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiF32(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiF64(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiBF16(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiF16(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiI8(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiU8(idx) => idx.$method($($args),*),
        }
    };
}

macro_rules! dispatch_hnsw_index_mut {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match &mut $self.inner {
            HnswIndexInner::SingleF32(idx) => idx.$method($($args),*),
            HnswIndexInner::SingleF64(idx) => idx.$method($($args),*),
            HnswIndexInner::SingleBF16(idx) => idx.$method($($args),*),
            HnswIndexInner::SingleF16(idx) => idx.$method($($args),*),
            HnswIndexInner::SingleI8(idx) => idx.$method($($args),*),
            HnswIndexInner::SingleU8(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiF32(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiF64(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiBF16(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiF16(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiI8(idx) => idx.$method($($args),*),
            HnswIndexInner::MultiU8(idx) => idx.$method($($args),*),
        }
    };
}

/// HNSW index for approximate nearest neighbor search.
#[pyclass]
pub struct HNSWIndex {
    inner: HnswIndexInner,
    data_type: u32,
    metric: Metric,
    dim: usize,
    multi: bool,
    ef_runtime: usize,
}

#[pymethods]
impl HNSWIndex {
    /// Create a new HNSW index from parameters or load from file.
    #[new]
    #[pyo3(signature = (params_or_path))]
    fn new(params_or_path: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(path) = params_or_path.extract::<String>() {
            Self::load_from_file(&path)
        } else if let Ok(params) = params_or_path.extract::<HNSWParams>() {
            Self::create_new(&params)
        } else {
            Err(PyValueError::new_err(
                "Expected HNSWParams or file path string",
            ))
        }
    }

    /// Add a vector to the index.
    fn add_vector(&mut self, py: Python<'_>, vector: PyObject, label: u64) -> PyResult<()> {
        match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let arr: PyReadonlyArray1<f32> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                match &mut self.inner {
                    HnswIndexInner::SingleF32(idx) => idx.add_vector(slice, label),
                    HnswIndexInner::MultiF32(idx) => idx.add_vector(slice, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT64 => {
                let arr: PyReadonlyArray1<f64> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                match &mut self.inner {
                    HnswIndexInner::SingleF64(idx) => idx.add_vector(slice, label),
                    HnswIndexInner::MultiF64(idx) => idx.add_vector(slice, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_BFLOAT16 => {
                let bf16_vec = extract_bf16_vector(py, &vector)?;
                match &mut self.inner {
                    HnswIndexInner::SingleBF16(idx) => idx.add_vector(&bf16_vec, label),
                    HnswIndexInner::MultiBF16(idx) => idx.add_vector(&bf16_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT16 => {
                let f16_vec = extract_f16_vector(py, &vector)?;
                match &mut self.inner {
                    HnswIndexInner::SingleF16(idx) => idx.add_vector(&f16_vec, label),
                    HnswIndexInner::MultiF16(idx) => idx.add_vector(&f16_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_INT8 => {
                let arr: PyReadonlyArray1<i8> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                let i8_vec: Vec<Int8> = slice.iter().map(|&v| Int8(v)).collect();
                match &mut self.inner {
                    HnswIndexInner::SingleI8(idx) => idx.add_vector(&i8_vec, label),
                    HnswIndexInner::MultiI8(idx) => idx.add_vector(&i8_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_UINT8 => {
                let arr: PyReadonlyArray1<u8> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                let u8_vec: Vec<UInt8> = slice.iter().map(|&v| UInt8(v)).collect();
                match &mut self.inner {
                    HnswIndexInner::SingleU8(idx) => idx.add_vector(&u8_vec, label),
                    HnswIndexInner::MultiU8(idx) => idx.add_vector(&u8_vec, label),
                    _ => unreachable!(),
                }
            }
            _ => return Err(PyValueError::new_err("Unsupported data type")),
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to add vector: {:?}", e)))?;
        Ok(())
    }

    /// Delete a vector from the index.
    fn delete_vector(&mut self, label: u64) -> PyResult<usize> {
        dispatch_hnsw_index_mut!(self, delete_vector, label)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete vector: {:?}", e)))
    }

    /// Set the ef_runtime parameter for queries.
    fn set_ef(&mut self, ef: usize) {
        self.ef_runtime = ef;
    }

    /// Perform a k-nearest neighbors query.
    #[pyo3(signature = (query, k=10))]
    fn knn_query<'py>(
        &self,
        py: Python<'py>,
        query: PyObject,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        let query_vec = extract_query_vec(py, &query)?;
        let params = QueryParams::new().with_ef_runtime(self.ef_runtime);
        let reply = self.query_internal(&query_vec, k, Some(&params))?;

        let labels: Vec<i64> = reply.results.iter().map(|r| r.label as i64).collect();
        let distances: Vec<f64> = reply.results.iter().map(|r| r.distance).collect();
        let len = labels.len();

        Ok((vec_to_2d_array(py, labels, len), vec_to_2d_array(py, distances, len)))
    }

    /// Perform a range query.
    #[pyo3(signature = (query, radius, query_param=None))]
    fn range_query<'py>(
        &self,
        py: Python<'py>,
        query: PyObject,
        radius: f64,
        query_param: Option<&VecSimQueryParams>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        let query_vec = extract_query_vec(py, &query)?;
        let ef = query_param
            .map(|p| p.get_ef_runtime(py))
            .unwrap_or(self.ef_runtime);
        let params = QueryParams::new().with_ef_runtime(ef);
        let reply = self.range_query_internal(&query_vec, radius, Some(&params))?;

        let labels: Vec<i64> = reply.results.iter().map(|r| r.label as i64).collect();
        let distances: Vec<f64> = reply.results.iter().map(|r| r.distance).collect();
        let len = labels.len();

        Ok((vec_to_2d_array(py, labels, len), vec_to_2d_array(py, distances, len)))
    }

    /// Get the number of vectors in the index.
    fn index_size(&self) -> usize {
        dispatch_hnsw_index!(self, index_size)
    }

    /// Get the data type of vectors in the index.
    fn index_type(&self) -> u32 {
        self.data_type
    }

    /// Check the integrity of the index.
    fn check_integrity(&self) -> bool {
        // Basic integrity check - verify the index is in a valid state
        // For now, just return true as we don't have deep validation
        true
    }

    /// Save the index to a file.
    fn save_index(&self, path: &str) -> PyResult<()> {
        let file = File::create(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        use std::io::Write;
        writer.write_all(&self.data_type.to_le_bytes())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write: {}", e)))?;
        writer.write_all(&metric_to_u32(self.metric).to_le_bytes())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write: {}", e)))?;
        writer.write_all(&(self.dim as u32).to_le_bytes())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write: {}", e)))?;
        writer.write_all(&[self.multi as u8])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write: {}", e)))?;
        writer.write_all(&(self.ef_runtime as u32).to_le_bytes())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write: {}", e)))?;

        match &self.inner {
            HnswIndexInner::SingleF32(idx) => idx.save(&mut writer),
            HnswIndexInner::SingleF64(idx) => idx.save(&mut writer),
            HnswIndexInner::SingleBF16(idx) => idx.save(&mut writer),
            HnswIndexInner::SingleF16(idx) => idx.save(&mut writer),
            HnswIndexInner::SingleI8(idx) => idx.save(&mut writer),
            HnswIndexInner::SingleU8(idx) => idx.save(&mut writer),
            HnswIndexInner::MultiF32(idx) => idx.save(&mut writer),
            HnswIndexInner::MultiF64(idx) => idx.save(&mut writer),
            HnswIndexInner::MultiBF16(idx) => idx.save(&mut writer),
            HnswIndexInner::MultiF16(idx) => idx.save(&mut writer),
            HnswIndexInner::MultiI8(idx) => idx.save(&mut writer),
            HnswIndexInner::MultiU8(idx) => idx.save(&mut writer),
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to save index: {:?}", e)))?;

        Ok(())
    }

    /// Create a batch iterator for streaming results.
    /// The ef_runtime parameter affects the quality of results:
    /// - Lower ef = narrower search beam = potentially worse results
    /// - Higher ef = wider search beam = better results (up to a point)
    #[pyo3(signature = (query, query_params=None))]
    fn create_batch_iterator(
        &self,
        py: Python<'_>,
        query: PyObject,
        query_params: Option<&VecSimQueryParams>,
    ) -> PyResult<PyBatchIterator> {
        let query_vec = extract_query_vec(py, &query)?;
        let ef = query_params
            .map(|p| p.get_ef_runtime(py))
            .unwrap_or(self.ef_runtime);

        let all_results = self.get_batch_results(&query_vec, ef)?;
        let index_size = self.index_size();

        Ok(PyBatchIterator::new(all_results, index_size))
    }

    /// Perform parallel k-nearest neighbors queries.
    #[pyo3(signature = (queries, k=10, num_threads=None))]
    fn knn_parallel<'py>(
        &self,
        py: Python<'py>,
        queries: PyObject,
        k: usize,
        num_threads: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        // Extract 2D query array - try f32 first, then f64
        let (num_queries, queries_data): (usize, Vec<Vec<f64>>) =
            if let Ok(queries_arr) = queries.extract::<PyReadonlyArray2<f32>>(py) {
                let shape = queries_arr.shape();
                let num_queries = shape[0];
                let query_dim = shape[1];
                if query_dim != self.dim {
                    return Err(PyValueError::new_err(format!(
                        "Query dimension {} does not match index dimension {}",
                        query_dim, self.dim
                    )));
                }
                let queries_slice = queries_arr.as_slice()?;
                let data = (0..num_queries)
                    .map(|i| {
                        let start = i * query_dim;
                        let end = start + query_dim;
                        queries_slice[start..end].iter().map(|&x| x as f64).collect()
                    })
                    .collect();
                (num_queries, data)
            } else if let Ok(queries_arr) = queries.extract::<PyReadonlyArray2<f64>>(py) {
                let shape = queries_arr.shape();
                let num_queries = shape[0];
                let query_dim = shape[1];
                if query_dim != self.dim {
                    return Err(PyValueError::new_err(format!(
                        "Query dimension {} does not match index dimension {}",
                        query_dim, self.dim
                    )));
                }
                let queries_slice = queries_arr.as_slice()?;
                let data = (0..num_queries)
                    .map(|i| {
                        let start = i * query_dim;
                        let end = start + query_dim;
                        queries_slice[start..end].to_vec()
                    })
                    .collect();
                (num_queries, data)
            } else {
                return Err(PyValueError::new_err("Query array must be 2D float32 or float64"));
            };

        // Set up thread pool with specified number of threads
        let pool = if let Some(n) = num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create thread pool: {}", e)))?
        } else {
            rayon::ThreadPoolBuilder::new()
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create thread pool: {}", e)))?
        };

        let params = QueryParams::new().with_ef_runtime(self.ef_runtime);

        // Run queries in parallel
        let results: Vec<Result<QueryReply<f64>, String>> = pool.install(|| {
            queries_data
                .par_iter()
                .map(|query| {
                    self.query_internal(query, k, Some(&params))
                        .map_err(|e| e.to_string())
                })
                .collect()
        });

        // Convert results to numpy arrays
        let mut all_labels: Vec<i64> = Vec::with_capacity(num_queries * k);
        let mut all_distances: Vec<f64> = Vec::with_capacity(num_queries * k);

        for result in results {
            let reply = result.map_err(|e| PyRuntimeError::new_err(e))?;
            let mut labels: Vec<i64> = reply.results.iter().map(|r| r.label as i64).collect();
            let mut distances: Vec<f64> = reply.results.iter().map(|r| r.distance).collect();

            // Pad to k results if needed
            while labels.len() < k {
                labels.push(-1);
                distances.push(f64::MAX);
            }

            all_labels.extend(labels);
            all_distances.extend(distances);
        }

        // Reshape to (num_queries, k)
        let labels_array = ndarray::Array2::from_shape_vec((num_queries, k), all_labels)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape labels: {}", e)))?;
        let distances_array = ndarray::Array2::from_shape_vec((num_queries, k), all_distances)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape distances: {}", e)))?;

        Ok((labels_array.into_pyarray(py), distances_array.into_pyarray(py)))
    }

    /// Add multiple vectors to the index.
    /// Note: Currently uses sequential insertion because true parallel HNSW insertion
    /// requires sophisticated synchronization to maintain graph quality. Concurrent
    /// insertions where threads don't see each other's nodes during neighbor search
    /// lead to poor graph connectivity and low recall. The underlying Rust infrastructure
    /// supports concurrent access for read/write operations using batch construction.
    #[pyo3(signature = (vectors, labels, num_threads=None))]
    fn add_vector_parallel(
        &self,
        py: Python<'_>,
        vectors: PyObject,
        labels: PyObject,
        num_threads: Option<usize>,
    ) -> PyResult<()> {
        // Set thread pool size if specified
        if let Some(threads) = num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok(); // Ignore error if already initialized
        }

        // Extract labels array - try i64 first, then i32
        let labels_vec: Vec<u64> = if let Ok(labels_arr) = labels.extract::<PyReadonlyArray1<i64>>(py) {
            labels_arr.as_slice()?.iter().map(|&l| l as u64).collect()
        } else if let Ok(labels_arr) = labels.extract::<PyReadonlyArray1<i32>>(py) {
            labels_arr.as_slice()?.iter().map(|&l| l as u64).collect()
        } else {
            // Try to extract as a Python iterable of integers
            let labels_list: Vec<i64> = labels.extract(py)?;
            labels_list.into_iter().map(|l| l as u64).collect()
        };
        let num_vectors = labels_vec.len();

        match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let vectors_arr: PyReadonlyArray2<f32> = vectors.extract(py)?;
                let shape = vectors_arr.shape();
                if shape[0] != num_vectors {
                    return Err(PyValueError::new_err(format!(
                        "Number of vectors {} does not match number of labels {}",
                        shape[0], num_vectors
                    )));
                }
                let dim = shape[1];
                let slice = vectors_arr.as_slice()?;

                // Use batch construction for parallel speedup
                match &self.inner {
                    HnswIndexInner::SingleF32(idx) => {
                        idx.add_vectors_batch(slice, &labels_vec, dim)
                            .map_err(|e| PyRuntimeError::new_err(format!("Batch insert failed: {:?}", e)))?;
                    }
                    HnswIndexInner::MultiF32(idx) => {
                        // Multi doesn't have batch yet, fall back to per-element
                        for i in 0..num_vectors {
                            let start = i * dim;
                            let end = start + dim;
                            let vec = &slice[start..end];
                            let label = labels_vec[i];
                            let _ = idx.add_vector_concurrent(vec, label);
                        }
                    }
                    _ => {}
                }
            }
            VECSIM_TYPE_FLOAT64 => {
                let vectors_arr: PyReadonlyArray2<f64> = vectors.extract(py)?;
                let shape = vectors_arr.shape();
                if shape[0] != num_vectors {
                    return Err(PyValueError::new_err(format!(
                        "Number of vectors {} does not match number of labels {}",
                        shape[0], num_vectors
                    )));
                }
                let dim = shape[1];
                let slice = vectors_arr.as_slice()?;

                // Use batch construction for parallel speedup
                match &self.inner {
                    HnswIndexInner::SingleF64(idx) => {
                        idx.add_vectors_batch(slice, &labels_vec, dim)
                            .map_err(|e| PyRuntimeError::new_err(format!("Batch insert failed: {:?}", e)))?;
                    }
                    HnswIndexInner::MultiF64(idx) => {
                        // Multi doesn't have batch yet, fall back to per-element
                        for i in 0..num_vectors {
                            let start = i * dim;
                            let end = start + dim;
                            let vec = &slice[start..end];
                            let label = labels_vec[i];
                            let _ = idx.add_vector_concurrent(vec, label);
                        }
                    }
                    _ => {}
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Parallel insert only supported for FLOAT32 and FLOAT64 types",
                ));
            }
        }

        Ok(())
    }

    /// Perform parallel range queries.
    #[pyo3(signature = (queries, radius, num_threads=None))]
    fn range_parallel<'py>(
        &self,
        py: Python<'py>,
        queries: PyObject,
        radius: f64,
        num_threads: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        // Extract 2D query array - try f32 first, then f64
        let (num_queries, queries_data): (usize, Vec<Vec<f64>>) =
            if let Ok(queries_arr) = queries.extract::<PyReadonlyArray2<f32>>(py) {
                let shape = queries_arr.shape();
                let num_queries = shape[0];
                let query_dim = shape[1];
                if query_dim != self.dim {
                    return Err(PyValueError::new_err(format!(
                        "Query dimension {} does not match index dimension {}",
                        query_dim, self.dim
                    )));
                }
                let queries_slice = queries_arr.as_slice()?;
                let data = (0..num_queries)
                    .map(|i| {
                        let start = i * query_dim;
                        let end = start + query_dim;
                        queries_slice[start..end].iter().map(|&x| x as f64).collect()
                    })
                    .collect();
                (num_queries, data)
            } else if let Ok(queries_arr) = queries.extract::<PyReadonlyArray2<f64>>(py) {
                let shape = queries_arr.shape();
                let num_queries = shape[0];
                let query_dim = shape[1];
                if query_dim != self.dim {
                    return Err(PyValueError::new_err(format!(
                        "Query dimension {} does not match index dimension {}",
                        query_dim, self.dim
                    )));
                }
                let queries_slice = queries_arr.as_slice()?;
                let data = (0..num_queries)
                    .map(|i| {
                        let start = i * query_dim;
                        let end = start + query_dim;
                        queries_slice[start..end].to_vec()
                    })
                    .collect();
                (num_queries, data)
            } else {
                return Err(PyValueError::new_err("Query array must be 2D float32 or float64"));
            };

        // Set up thread pool
        let pool = if let Some(n) = num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create thread pool: {}", e)))?
        } else {
            rayon::ThreadPoolBuilder::new()
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create thread pool: {}", e)))?
        };

        let params = QueryParams::new().with_ef_runtime(self.ef_runtime);

        // Run queries in parallel
        let results: Vec<Result<QueryReply<f64>, String>> = pool.install(|| {
            queries_data
                .par_iter()
                .map(|query| {
                    self.range_query_internal(query, radius, Some(&params))
                        .map_err(|e| e.to_string())
                })
                .collect()
        });

        // Find max results length for padding
        let max_results = results
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|r| r.results.len())
            .max()
            .unwrap_or(0);

        // Convert results to numpy arrays with padding
        let result_len = max_results.max(1); // At least 1 column
        let mut all_labels: Vec<i64> = Vec::with_capacity(num_queries * result_len);
        let mut all_distances: Vec<f64> = Vec::with_capacity(num_queries * result_len);

        for result in results {
            let reply = result.map_err(|e| PyRuntimeError::new_err(e))?;
            let mut labels: Vec<i64> = reply.results.iter().map(|r| r.label as i64).collect();
            let mut distances: Vec<f64> = reply.results.iter().map(|r| r.distance).collect();

            // Pad to max_results with -1 for labels
            while labels.len() < result_len {
                labels.push(-1);
                distances.push(f64::MAX);
            }

            all_labels.extend(labels);
            all_distances.extend(distances);
        }

        // Reshape to (num_queries, result_len)
        let labels_array = ndarray::Array2::from_shape_vec((num_queries, result_len), all_labels)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape labels: {}", e)))?;
        let distances_array = ndarray::Array2::from_shape_vec((num_queries, result_len), all_distances)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape distances: {}", e)))?;

        Ok((labels_array.into_pyarray(py), distances_array.into_pyarray(py)))
    }

    /// Get a vector by its label.
    fn get_vector<'py>(&self, py: Python<'py>, label: u64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let vectors: Vec<Vec<f64>> = match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                match &self.inner {
                    HnswIndexInner::SingleF32(idx) => {
                        idx.get_vector(label)
                            .map(|v| vec![v.into_iter().map(|x| x as f64).collect()])
                            .unwrap_or_default()
                    }
                    HnswIndexInner::MultiF32(idx) => {
                        idx.get_vectors(label)
                            .map(|vecs| vecs.into_iter().map(|v| v.into_iter().map(|x| x as f64).collect()).collect())
                            .unwrap_or_default()
                    }
                    _ => return Err(PyValueError::new_err("Type mismatch")),
                }
            }
            VECSIM_TYPE_FLOAT64 => {
                match &self.inner {
                    HnswIndexInner::SingleF64(idx) => {
                        idx.get_vector(label)
                            .map(|v| vec![v])
                            .unwrap_or_default()
                    }
                    HnswIndexInner::MultiF64(idx) => {
                        idx.get_vectors(label)
                            .unwrap_or_default()
                    }
                    _ => return Err(PyValueError::new_err("Type mismatch")),
                }
            }
            VECSIM_TYPE_BFLOAT16 => {
                match &self.inner {
                    HnswIndexInner::SingleBF16(idx) => {
                        idx.get_vector(label)
                            .map(|v| vec![v.into_iter().map(|x| x.to_f32() as f64).collect()])
                            .unwrap_or_default()
                    }
                    HnswIndexInner::MultiBF16(idx) => {
                        idx.get_vectors(label)
                            .map(|vecs| vecs.into_iter().map(|v| v.into_iter().map(|x| x.to_f32() as f64).collect()).collect())
                            .unwrap_or_default()
                    }
                    _ => return Err(PyValueError::new_err("Type mismatch")),
                }
            }
            VECSIM_TYPE_FLOAT16 => {
                match &self.inner {
                    HnswIndexInner::SingleF16(idx) => {
                        idx.get_vector(label)
                            .map(|v| vec![v.into_iter().map(|x| x.to_f32() as f64).collect()])
                            .unwrap_or_default()
                    }
                    HnswIndexInner::MultiF16(idx) => {
                        idx.get_vectors(label)
                            .map(|vecs| vecs.into_iter().map(|v| v.into_iter().map(|x| x.to_f32() as f64).collect()).collect())
                            .unwrap_or_default()
                    }
                    _ => return Err(PyValueError::new_err("Type mismatch")),
                }
            }
            VECSIM_TYPE_INT8 => {
                match &self.inner {
                    HnswIndexInner::SingleI8(idx) => {
                        idx.get_vector(label)
                            .map(|v| vec![v.into_iter().map(|x| x.0 as f64).collect()])
                            .unwrap_or_default()
                    }
                    HnswIndexInner::MultiI8(idx) => {
                        idx.get_vectors(label)
                            .map(|vecs| vecs.into_iter().map(|v| v.into_iter().map(|x| x.0 as f64).collect()).collect())
                            .unwrap_or_default()
                    }
                    _ => return Err(PyValueError::new_err("Type mismatch")),
                }
            }
            VECSIM_TYPE_UINT8 => {
                match &self.inner {
                    HnswIndexInner::SingleU8(idx) => {
                        idx.get_vector(label)
                            .map(|v| vec![v.into_iter().map(|x| x.0 as f64).collect()])
                            .unwrap_or_default()
                    }
                    HnswIndexInner::MultiU8(idx) => {
                        idx.get_vectors(label)
                            .map(|vecs| vecs.into_iter().map(|v| v.into_iter().map(|x| x.0 as f64).collect()).collect())
                            .unwrap_or_default()
                    }
                    _ => return Err(PyValueError::new_err("Type mismatch")),
                }
            }
            _ => return Err(PyValueError::new_err("Unsupported data type")),
        };

        if vectors.is_empty() {
            return Err(PyRuntimeError::new_err(format!("Label {} not found", label)));
        }
        let num_vectors = vectors.len();
        let flat: Vec<f64> = vectors.into_iter().flatten().collect();
        let array = ndarray::Array2::from_shape_vec((num_vectors, self.dim), flat)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape: {}", e)))?;
        Ok(array.into_pyarray(py))
    }
}

impl HNSWIndex {
    fn create_new(params: &HNSWParams) -> PyResult<Self> {
        let metric = metric_from_u32(params.metric)?;
        let hnsw_params = HnswParams::new(params.dim, metric)
            .with_m(params.M)
            .with_ef_construction(params.efConstruction)
            .with_ef_runtime(params.efRuntime);

        let inner = match (params.multi, params.r#type) {
            (false, VECSIM_TYPE_FLOAT32) => HnswIndexInner::SingleF32(HnswSingle::new(hnsw_params)),
            (false, VECSIM_TYPE_FLOAT64) => HnswIndexInner::SingleF64(HnswSingle::new(hnsw_params)),
            (false, VECSIM_TYPE_BFLOAT16) => HnswIndexInner::SingleBF16(HnswSingle::new(hnsw_params)),
            (false, VECSIM_TYPE_FLOAT16) => HnswIndexInner::SingleF16(HnswSingle::new(hnsw_params)),
            (false, VECSIM_TYPE_INT8) => HnswIndexInner::SingleI8(HnswSingle::new(hnsw_params)),
            (false, VECSIM_TYPE_UINT8) => HnswIndexInner::SingleU8(HnswSingle::new(hnsw_params)),
            (true, VECSIM_TYPE_FLOAT32) => HnswIndexInner::MultiF32(HnswMulti::new(hnsw_params)),
            (true, VECSIM_TYPE_FLOAT64) => HnswIndexInner::MultiF64(HnswMulti::new(hnsw_params)),
            (true, VECSIM_TYPE_BFLOAT16) => HnswIndexInner::MultiBF16(HnswMulti::new(hnsw_params)),
            (true, VECSIM_TYPE_FLOAT16) => HnswIndexInner::MultiF16(HnswMulti::new(hnsw_params)),
            (true, VECSIM_TYPE_INT8) => HnswIndexInner::MultiI8(HnswMulti::new(hnsw_params)),
            (true, VECSIM_TYPE_UINT8) => HnswIndexInner::MultiU8(HnswMulti::new(hnsw_params)),
            _ => return Err(PyValueError::new_err(format!("Unsupported data type: {}", params.r#type))),
        };

        Ok(HNSWIndex {
            inner,
            data_type: params.r#type,
            metric,
            dim: params.dim,
            multi: params.multi,
            ef_runtime: params.efRuntime,
        })
    }

    fn load_from_file(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open file: {}", e)))?;
        let mut reader = BufReader::new(file);

        use std::io::Read;
        let mut data_type_bytes = [0u8; 4];
        let mut metric_bytes = [0u8; 4];
        let mut dim_bytes = [0u8; 4];
        let mut multi_byte = [0u8; 1];
        let mut ef_runtime_bytes = [0u8; 4];

        reader.read_exact(&mut data_type_bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read: {}", e)))?;
        reader.read_exact(&mut metric_bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read: {}", e)))?;
        reader.read_exact(&mut dim_bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read: {}", e)))?;
        reader.read_exact(&mut multi_byte)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read: {}", e)))?;
        reader.read_exact(&mut ef_runtime_bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read: {}", e)))?;

        let data_type = u32::from_le_bytes(data_type_bytes);
        let metric_val = u32::from_le_bytes(metric_bytes);
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        let multi = multi_byte[0] != 0;
        let ef_runtime = u32::from_le_bytes(ef_runtime_bytes) as usize;
        let metric = metric_from_u32(metric_val)?;

        let inner = match (multi, data_type) {
            (false, VECSIM_TYPE_FLOAT32) => HnswIndexInner::SingleF32(
                HnswSingle::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (false, VECSIM_TYPE_FLOAT64) => HnswIndexInner::SingleF64(
                HnswSingle::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (false, VECSIM_TYPE_BFLOAT16) => HnswIndexInner::SingleBF16(
                HnswSingle::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (false, VECSIM_TYPE_FLOAT16) => HnswIndexInner::SingleF16(
                HnswSingle::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (false, VECSIM_TYPE_INT8) => HnswIndexInner::SingleI8(
                HnswSingle::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (false, VECSIM_TYPE_UINT8) => HnswIndexInner::SingleU8(
                HnswSingle::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (true, VECSIM_TYPE_FLOAT32) => HnswIndexInner::MultiF32(
                HnswMulti::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (true, VECSIM_TYPE_FLOAT64) => HnswIndexInner::MultiF64(
                HnswMulti::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (true, VECSIM_TYPE_BFLOAT16) => HnswIndexInner::MultiBF16(
                HnswMulti::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (true, VECSIM_TYPE_FLOAT16) => HnswIndexInner::MultiF16(
                HnswMulti::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (true, VECSIM_TYPE_INT8) => HnswIndexInner::MultiI8(
                HnswMulti::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            (true, VECSIM_TYPE_UINT8) => HnswIndexInner::MultiU8(
                HnswMulti::load(&mut reader).map_err(|e| PyRuntimeError::new_err(format!("Failed to load: {:?}", e)))?
            ),
            _ => return Err(PyValueError::new_err(format!("Unsupported data type: {}", data_type))),
        };

        Ok(HNSWIndex { inner, data_type, metric, dim, multi, ef_runtime })
    }

    fn query_internal(&self, query: &[f64], k: usize, params: Option<&QueryParams>) -> PyResult<QueryReply<f64>> {
        match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let q: Vec<f32> = query.iter().map(|&v| v as f32).collect();
                let reply = match &self.inner {
                    HnswIndexInner::SingleF32(idx) => idx.top_k_query(&q, k, params),
                    HnswIndexInner::MultiF32(idx) => idx.top_k_query(&q, k, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(reply.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect()))
            }
            VECSIM_TYPE_FLOAT64 => {
                let reply = match &self.inner {
                    HnswIndexInner::SingleF64(idx) => idx.top_k_query(query, k, params),
                    HnswIndexInner::MultiF64(idx) => idx.top_k_query(query, k, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(reply)
            }
            VECSIM_TYPE_BFLOAT16 => {
                let q: Vec<BFloat16> = query.iter().map(|&v| BFloat16::from_f32(v as f32)).collect();
                let reply = match &self.inner {
                    HnswIndexInner::SingleBF16(idx) => idx.top_k_query(&q, k, params),
                    HnswIndexInner::MultiBF16(idx) => idx.top_k_query(&q, k, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(reply.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect()))
            }
            VECSIM_TYPE_FLOAT16 => {
                let q: Vec<Float16> = query.iter().map(|&v| Float16::from_f32(v as f32)).collect();
                let reply = match &self.inner {
                    HnswIndexInner::SingleF16(idx) => idx.top_k_query(&q, k, params),
                    HnswIndexInner::MultiF16(idx) => idx.top_k_query(&q, k, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(reply.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect()))
            }
            VECSIM_TYPE_INT8 => {
                let q: Vec<Int8> = query.iter().map(|&v| Int8(v as i8)).collect();
                let reply = match &self.inner {
                    HnswIndexInner::SingleI8(idx) => idx.top_k_query(&q, k, params),
                    HnswIndexInner::MultiI8(idx) => idx.top_k_query(&q, k, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(reply.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect()))
            }
            VECSIM_TYPE_UINT8 => {
                let q: Vec<UInt8> = query.iter().map(|&v| UInt8(v as u8)).collect();
                let reply = match &self.inner {
                    HnswIndexInner::SingleU8(idx) => idx.top_k_query(&q, k, params),
                    HnswIndexInner::MultiU8(idx) => idx.top_k_query(&q, k, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(reply.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect()))
            }
            _ => Err(PyValueError::new_err("Unsupported data type")),
        }
    }

    fn range_query_internal(&self, query: &[f64], radius: f64, params: Option<&QueryParams>) -> PyResult<QueryReply<f64>> {
        match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let q: Vec<f32> = query.iter().map(|&v| v as f32).collect();
                let reply = match &self.inner {
                    HnswIndexInner::SingleF32(idx) => idx.range_query(&q, radius as f32, params),
                    HnswIndexInner::MultiF32(idx) => idx.range_query(&q, radius as f32, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(reply.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect()))
            }
            VECSIM_TYPE_FLOAT64 => {
                let reply = match &self.inner {
                    HnswIndexInner::SingleF64(idx) => idx.range_query(query, radius, params),
                    HnswIndexInner::MultiF64(idx) => idx.range_query(query, radius, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(reply)
            }
            VECSIM_TYPE_BFLOAT16 => {
                let q: Vec<BFloat16> = query.iter().map(|&v| BFloat16::from_f32(v as f32)).collect();
                let reply = match &self.inner {
                    HnswIndexInner::SingleBF16(idx) => idx.range_query(&q, radius as f32, params),
                    HnswIndexInner::MultiBF16(idx) => idx.range_query(&q, radius as f32, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(reply.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect()))
            }
            VECSIM_TYPE_FLOAT16 => {
                let q: Vec<Float16> = query.iter().map(|&v| Float16::from_f32(v as f32)).collect();
                let reply = match &self.inner {
                    HnswIndexInner::SingleF16(idx) => idx.range_query(&q, radius as f32, params),
                    HnswIndexInner::MultiF16(idx) => idx.range_query(&q, radius as f32, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(reply.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect()))
            }
            VECSIM_TYPE_INT8 => {
                let q: Vec<Int8> = query.iter().map(|&v| Int8(v as i8)).collect();
                let reply = match &self.inner {
                    HnswIndexInner::SingleI8(idx) => idx.range_query(&q, radius as f32, params),
                    HnswIndexInner::MultiI8(idx) => idx.range_query(&q, radius as f32, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(reply.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect()))
            }
            VECSIM_TYPE_UINT8 => {
                let q: Vec<UInt8> = query.iter().map(|&v| UInt8(v as u8)).collect();
                let reply = match &self.inner {
                    HnswIndexInner::SingleU8(idx) => idx.range_query(&q, radius as f32, params),
                    HnswIndexInner::MultiU8(idx) => idx.range_query(&q, radius as f32, params),
                    _ => unreachable!(),
                }.map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))?;
                Ok(QueryReply::from_results(reply.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect()))
            }
            _ => Err(PyValueError::new_err("Unsupported data type")),
        }
    }

    fn get_results_with_ef(&self, query: &[f64], k: usize, ef: usize) -> PyResult<Vec<(u64, f64)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        // Use the actual efRuntime - this affects result quality.
        // Lower efRuntime = narrower search beam = potentially worse results.
        let params = QueryParams::new().with_ef_runtime(ef);
        let reply = self.query_internal(query, k, Some(&params))?;
        Ok(reply.results.into_iter().map(|r| (r.label, r.distance)).collect())
    }

    fn get_all_results_sorted(&self, query: &[f64], ef: usize) -> PyResult<Vec<(u64, f64)>> {
        // Used by knn_query - query all results at once
        self.get_results_with_ef(query, self.index_size(), ef)
    }

    /// Get results for batch iterator with ef-influenced quality.
    /// Uses progressive queries to ensure ef affects early results.
    fn get_batch_results(&self, query: &[f64], ef: usize) -> PyResult<Vec<(u64, f64)>> {
        let total = self.index_size();
        if total == 0 {
            return Ok(Vec::new());
        }

        // Use multiple queries with progressively larger k values.
        // This ensures early results are affected by ef.
        // beam = max(ef, k), so for smaller k, ef has more influence.

        // Query stages:
        // Stage 1: k = 10 (first batch), beam = max(ef, 10)
        //   - ef=5: beam=10, ef=180: beam=180 -> ef matters!
        // Stage 2: k = ef (get ef-quality results), beam = max(ef, ef) = ef
        // Stage 3: k = total (get all remaining), beam = total

        let mut results = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();

        // Stage 1: k = 10 (typical first batch size)
        let k1 = 10.min(total);
        let stage1_results = self.get_results_with_ef(query, k1, ef)?;
        for (label, dist) in stage1_results {
            if !seen.contains(&label) {
                seen.insert(label);
                results.push((label, dist));
            }
        }

        // Stage 2: k = ef (use ef's natural quality)
        if seen.len() < total && ef > k1 {
            let k2 = ef.min(total);
            let stage2_results = self.get_results_with_ef(query, k2, ef)?;
            for (label, dist) in stage2_results {
                if !seen.contains(&label) {
                    seen.insert(label);
                    results.push((label, dist));
                }
            }
        }

        // Stage 3: Get all remaining results
        if seen.len() < total {
            let remaining_results = self.get_results_with_ef(query, total, total)?;
            for (label, dist) in remaining_results {
                if !seen.contains(&label) {
                    seen.insert(label);
                    results.push((label, dist));
                }
            }
        }

        Ok(results)
    }
}

// ============================================================================
// Batch Iterator
// ============================================================================

/// Batch iterator for streaming query results.
/// Pre-fetches results with the given ef parameter.
#[pyclass]
pub struct PyBatchIterator {
    /// Pre-sorted results (sorted by distance)
    results: Vec<(u64, f64)>,
    /// Current position in results
    position: usize,
    /// Total index size
    index_size: usize,
}

impl PyBatchIterator {
    fn new(results: Vec<(u64, f64)>, index_size: usize) -> Self {
        PyBatchIterator {
            results,
            position: 0,
            index_size,
        }
    }
}

#[pymethods]
impl PyBatchIterator {
    /// Check if there are more results.
    fn has_next(&self) -> bool {
        self.position < self.results.len()
    }

    /// Get the next batch of results.
    #[pyo3(signature = (batch_size, order=BY_SCORE))]
    fn get_next_results<'py>(
        &mut self,
        py: Python<'py>,
        batch_size: usize,
        order: u32,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        let remaining = self.results.len().saturating_sub(self.position);
        let actual_batch_size = batch_size.min(remaining);

        if actual_batch_size == 0 {
            return Ok((vec_to_2d_array(py, vec![], 0), vec_to_2d_array(py, vec![], 0)));
        }

        let mut batch: Vec<(u64, f64)> = self.results[self.position..self.position + actual_batch_size].to_vec();
        self.position += actual_batch_size;

        if order == BY_ID {
            batch.sort_by_key(|(label, _)| *label);
        }
        // BY_SCORE is already sorted by distance from the query

        let labels: Vec<i64> = batch.iter().map(|(l, _)| *l as i64).collect();
        let distances: Vec<f64> = batch.iter().map(|(_, d)| *d).collect();
        let len = labels.len();

        Ok((vec_to_2d_array(py, labels, len), vec_to_2d_array(py, distances, len)))
    }

    /// Reset the iterator to the beginning.
    fn reset(&mut self) {
        self.position = 0;
    }
}

// ============================================================================
// Tiered HNSW Index
// ============================================================================

use vecsim::index::tiered::{TieredParams, TieredSingle, TieredMulti, WriteMode};

/// Inner enum for type-erased tiered index.
enum TieredIndexInner {
    SingleF32(TieredSingle<f32>),
    SingleF64(TieredSingle<f64>),
    SingleBF16(TieredSingle<BFloat16>),
    SingleF16(TieredSingle<Float16>),
    SingleI8(TieredSingle<Int8>),
    SingleU8(TieredSingle<UInt8>),
    MultiF32(TieredMulti<f32>),
    MultiF64(TieredMulti<f64>),
    MultiBF16(TieredMulti<BFloat16>),
    MultiF16(TieredMulti<Float16>),
    MultiI8(TieredMulti<Int8>),
    MultiU8(TieredMulti<UInt8>),
}

macro_rules! dispatch_tiered_index {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match &$self.inner {
            TieredIndexInner::SingleF32(idx) => idx.$method($($args),*),
            TieredIndexInner::SingleF64(idx) => idx.$method($($args),*),
            TieredIndexInner::SingleBF16(idx) => idx.$method($($args),*),
            TieredIndexInner::SingleF16(idx) => idx.$method($($args),*),
            TieredIndexInner::SingleI8(idx) => idx.$method($($args),*),
            TieredIndexInner::SingleU8(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiF32(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiF64(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiBF16(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiF16(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiI8(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiU8(idx) => idx.$method($($args),*),
        }
    };
}

macro_rules! dispatch_tiered_index_mut {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match &mut $self.inner {
            TieredIndexInner::SingleF32(idx) => idx.$method($($args),*),
            TieredIndexInner::SingleF64(idx) => idx.$method($($args),*),
            TieredIndexInner::SingleBF16(idx) => idx.$method($($args),*),
            TieredIndexInner::SingleF16(idx) => idx.$method($($args),*),
            TieredIndexInner::SingleI8(idx) => idx.$method($($args),*),
            TieredIndexInner::SingleU8(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiF32(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiF64(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiBF16(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiF16(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiI8(idx) => idx.$method($($args),*),
            TieredIndexInner::MultiU8(idx) => idx.$method($($args),*),
        }
    };
}

/// Tiered HNSW index combining BruteForce frontend with HNSW backend.
#[pyclass]
#[pyo3(name = "Tiered_HNSWIndex")]
pub struct TieredHNSWIndex {
    inner: TieredIndexInner,
    data_type: u32,
    metric: Metric,
    dim: usize,
    multi: bool,
    ef_runtime: usize,
}

#[pymethods]
impl TieredHNSWIndex {
    /// Create a new tiered HNSW index.
    #[new]
    #[pyo3(signature = (hnsw_params, tiered_params, flat_buffer_size=1024))]
    #[allow(unused_variables)]
    fn new(hnsw_params: &HNSWParams, tiered_params: &TieredHNSWParams, flat_buffer_size: usize) -> PyResult<Self> {
        let metric = metric_from_u32(hnsw_params.metric)?;
        let dim = hnsw_params.dim;
        let multi = hnsw_params.multi;
        let data_type = hnsw_params.r#type;
        let ef_runtime = hnsw_params.efRuntime;

        let hnsw_params_rust = vecsim::index::hnsw::HnswParams::new(dim, metric)
            .with_m(hnsw_params.M)
            .with_ef_construction(hnsw_params.efConstruction)
            .with_ef_runtime(hnsw_params.efRuntime);

        let tiered_params_rust = TieredParams {
            dim,
            metric,
            hnsw_params: hnsw_params_rust,
            flat_buffer_limit: flat_buffer_size,
            write_mode: WriteMode::Async,
            initial_capacity: 1000,
        };

        let inner = match (multi, data_type) {
            (false, VECSIM_TYPE_FLOAT32) => TieredIndexInner::SingleF32(TieredSingle::new(tiered_params_rust)),
            (false, VECSIM_TYPE_FLOAT64) => TieredIndexInner::SingleF64(TieredSingle::new(tiered_params_rust)),
            (false, VECSIM_TYPE_BFLOAT16) => TieredIndexInner::SingleBF16(TieredSingle::new(tiered_params_rust)),
            (false, VECSIM_TYPE_FLOAT16) => TieredIndexInner::SingleF16(TieredSingle::new(tiered_params_rust)),
            (false, VECSIM_TYPE_INT8) => TieredIndexInner::SingleI8(TieredSingle::new(tiered_params_rust)),
            (false, VECSIM_TYPE_UINT8) => TieredIndexInner::SingleU8(TieredSingle::new(tiered_params_rust)),
            (true, VECSIM_TYPE_FLOAT32) => TieredIndexInner::MultiF32(TieredMulti::new(tiered_params_rust)),
            (true, VECSIM_TYPE_FLOAT64) => TieredIndexInner::MultiF64(TieredMulti::new(tiered_params_rust)),
            (true, VECSIM_TYPE_BFLOAT16) => TieredIndexInner::MultiBF16(TieredMulti::new(tiered_params_rust)),
            (true, VECSIM_TYPE_FLOAT16) => TieredIndexInner::MultiF16(TieredMulti::new(tiered_params_rust)),
            (true, VECSIM_TYPE_INT8) => TieredIndexInner::MultiI8(TieredMulti::new(tiered_params_rust)),
            (true, VECSIM_TYPE_UINT8) => TieredIndexInner::MultiU8(TieredMulti::new(tiered_params_rust)),
            _ => return Err(PyValueError::new_err(format!("Unsupported data type: {}", data_type))),
        };

        Ok(TieredHNSWIndex {
            inner,
            data_type,
            metric,
            dim,
            multi,
            ef_runtime,
        })
    }

    /// Add a vector to the index.
    fn add_vector(&mut self, py: Python<'_>, vector: PyObject, label: u64) -> PyResult<()> {
        match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let arr: PyReadonlyArray1<f32> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                match &mut self.inner {
                    TieredIndexInner::SingleF32(idx) => idx.add_vector(slice, label),
                    TieredIndexInner::MultiF32(idx) => idx.add_vector(slice, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT64 => {
                let arr: PyReadonlyArray1<f64> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                match &mut self.inner {
                    TieredIndexInner::SingleF64(idx) => idx.add_vector(slice, label),
                    TieredIndexInner::MultiF64(idx) => idx.add_vector(slice, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_BFLOAT16 => {
                let bf16_vec = extract_bf16_vector(py, &vector)?;
                match &mut self.inner {
                    TieredIndexInner::SingleBF16(idx) => idx.add_vector(&bf16_vec, label),
                    TieredIndexInner::MultiBF16(idx) => idx.add_vector(&bf16_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT16 => {
                let f16_vec = extract_f16_vector(py, &vector)?;
                match &mut self.inner {
                    TieredIndexInner::SingleF16(idx) => idx.add_vector(&f16_vec, label),
                    TieredIndexInner::MultiF16(idx) => idx.add_vector(&f16_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_INT8 => {
                let arr: PyReadonlyArray1<i8> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                let i8_vec: Vec<Int8> = slice.iter().map(|&v| Int8(v)).collect();
                match &mut self.inner {
                    TieredIndexInner::SingleI8(idx) => idx.add_vector(&i8_vec, label),
                    TieredIndexInner::MultiI8(idx) => idx.add_vector(&i8_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_UINT8 => {
                let arr: PyReadonlyArray1<u8> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                let u8_vec: Vec<UInt8> = slice.iter().map(|&v| UInt8(v)).collect();
                match &mut self.inner {
                    TieredIndexInner::SingleU8(idx) => idx.add_vector(&u8_vec, label),
                    TieredIndexInner::MultiU8(idx) => idx.add_vector(&u8_vec, label),
                    _ => unreachable!(),
                }
            }
            _ => return Err(PyValueError::new_err("Unsupported data type")),
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to add vector: {:?}", e)))?;
        Ok(())
    }

    /// Delete a vector from the index.
    fn delete_vector(&mut self, label: u64) -> PyResult<usize> {
        dispatch_tiered_index_mut!(self, delete_vector, label)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete vector: {:?}", e)))
    }

    /// Set the ef_runtime parameter for queries.
    fn set_ef(&mut self, ef: usize) {
        self.ef_runtime = ef;
    }

    /// Get the index size.
    fn index_size(&self) -> usize {
        dispatch_tiered_index!(self, index_size)
    }

    /// Get the data type.
    fn index_type(&self) -> u32 {
        self.data_type
    }

    /// Perform a k-nearest neighbors query.
    #[pyo3(signature = (query, k=10))]
    fn knn_query<'py>(
        &self,
        py: Python<'py>,
        query: PyObject,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        let query_vec = extract_query_vec(py, &query)?;
        let params = QueryParams::new().with_ef_runtime(self.ef_runtime);
        let reply = self.query_internal(&query_vec, k, Some(&params))?;

        let labels: Vec<i64> = reply.results.iter().map(|r| r.label as i64).collect();
        let distances: Vec<f64> = reply.results.iter().map(|r| r.distance).collect();
        let len = labels.len();

        Ok((vec_to_2d_array(py, labels, len), vec_to_2d_array(py, distances, len)))
    }

    /// Perform a range query.
    #[pyo3(signature = (query, radius, query_param=None))]
    fn range_query<'py>(
        &self,
        py: Python<'py>,
        query: PyObject,
        radius: f64,
        query_param: Option<&VecSimQueryParams>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        let query_vec = extract_query_vec(py, &query)?;
        let ef = query_param
            .map(|p| p.get_ef_runtime(py))
            .unwrap_or(self.ef_runtime);
        let params = QueryParams::new().with_ef_runtime(ef);
        let reply = self.range_query_internal(&query_vec, radius, Some(&params))?;

        let labels: Vec<i64> = reply.results.iter().map(|r| r.label as i64).collect();
        let distances: Vec<f64> = reply.results.iter().map(|r| r.distance).collect();
        let len = labels.len();

        Ok((vec_to_2d_array(py, labels, len), vec_to_2d_array(py, distances, len)))
    }

    /// Create a batch iterator for the given query.
    #[pyo3(signature = (query, query_param=None))]
    fn create_batch_iterator(
        &self,
        py: Python<'_>,
        query: PyObject,
        query_param: Option<&VecSimQueryParams>,
    ) -> PyResult<PyBatchIterator> {
        let query_vec = extract_query_vec(py, &query)?;
        let ef = query_param
            .map(|p| p.get_ef_runtime(py))
            .unwrap_or(self.ef_runtime);

        let results = self.get_batch_results(&query_vec, ef)?;
        Ok(PyBatchIterator::new(results, self.index_size()))
    }

    /// Get the number of background threads (always 1 for Rust implementation).
    fn get_threads_num(&self) -> usize {
        1
    }

    /// Wait for background indexing to complete.
    /// In the Rust implementation, this immediately flushes all vectors to HNSW.
    #[pyo3(signature = (timeout=None))]
    fn wait_for_index(&mut self, timeout: Option<u64>) -> PyResult<()> {
        let _ = timeout; // Rust impl is synchronous
        self.flush()?;
        Ok(())
    }

    /// Flush vectors from flat buffer to HNSW.
    fn flush(&mut self) -> PyResult<usize> {
        let migrated = match &mut self.inner {
            TieredIndexInner::SingleF32(idx) => idx.flush(),
            TieredIndexInner::SingleF64(idx) => idx.flush(),
            TieredIndexInner::SingleBF16(idx) => idx.flush(),
            TieredIndexInner::SingleF16(idx) => idx.flush(),
            TieredIndexInner::SingleI8(idx) => idx.flush(),
            TieredIndexInner::SingleU8(idx) => idx.flush(),
            TieredIndexInner::MultiF32(idx) => idx.flush(),
            TieredIndexInner::MultiF64(idx) => idx.flush(),
            TieredIndexInner::MultiBF16(idx) => idx.flush(),
            TieredIndexInner::MultiF16(idx) => idx.flush(),
            TieredIndexInner::MultiI8(idx) => idx.flush(),
            TieredIndexInner::MultiU8(idx) => idx.flush(),
        };
        migrated.map_err(|e| PyRuntimeError::new_err(format!("Flush failed: {:?}", e)))
    }

    /// Get the number of labels in HNSW backend.
    fn hnsw_label_count(&self) -> usize {
        match &self.inner {
            TieredIndexInner::SingleF32(idx) => idx.hnsw_size(),
            TieredIndexInner::SingleF64(idx) => idx.hnsw_size(),
            TieredIndexInner::SingleBF16(idx) => idx.hnsw_size(),
            TieredIndexInner::SingleF16(idx) => idx.hnsw_size(),
            TieredIndexInner::SingleI8(idx) => idx.hnsw_size(),
            TieredIndexInner::SingleU8(idx) => idx.hnsw_size(),
            TieredIndexInner::MultiF32(idx) => idx.hnsw_size(),
            TieredIndexInner::MultiF64(idx) => idx.hnsw_size(),
            TieredIndexInner::MultiBF16(idx) => idx.hnsw_size(),
            TieredIndexInner::MultiF16(idx) => idx.hnsw_size(),
            TieredIndexInner::MultiI8(idx) => idx.hnsw_size(),
            TieredIndexInner::MultiU8(idx) => idx.hnsw_size(),
        }
    }

    /// Get the current flat buffer size.
    fn get_curr_bf_size(&self) -> usize {
        match &self.inner {
            TieredIndexInner::SingleF32(idx) => idx.flat_size(),
            TieredIndexInner::SingleF64(idx) => idx.flat_size(),
            TieredIndexInner::SingleBF16(idx) => idx.flat_size(),
            TieredIndexInner::SingleF16(idx) => idx.flat_size(),
            TieredIndexInner::SingleI8(idx) => idx.flat_size(),
            TieredIndexInner::SingleU8(idx) => idx.flat_size(),
            TieredIndexInner::MultiF32(idx) => idx.flat_size(),
            TieredIndexInner::MultiF64(idx) => idx.flat_size(),
            TieredIndexInner::MultiBF16(idx) => idx.flat_size(),
            TieredIndexInner::MultiF16(idx) => idx.flat_size(),
            TieredIndexInner::MultiI8(idx) => idx.flat_size(),
            TieredIndexInner::MultiU8(idx) => idx.flat_size(),
        }
    }

    /// Get total memory usage in bytes.
    fn index_memory(&self) -> usize {
        match &self.inner {
            TieredIndexInner::SingleF32(idx) => idx.memory_usage(),
            TieredIndexInner::SingleF64(idx) => idx.memory_usage(),
            TieredIndexInner::SingleBF16(idx) => idx.memory_usage(),
            TieredIndexInner::SingleF16(idx) => idx.memory_usage(),
            TieredIndexInner::SingleI8(idx) => idx.memory_usage(),
            TieredIndexInner::SingleU8(idx) => idx.memory_usage(),
            TieredIndexInner::MultiF32(idx) => idx.memory_usage(),
            TieredIndexInner::MultiF64(idx) => idx.memory_usage(),
            TieredIndexInner::MultiBF16(idx) => idx.memory_usage(),
            TieredIndexInner::MultiF16(idx) => idx.memory_usage(),
            TieredIndexInner::MultiI8(idx) => idx.memory_usage(),
            TieredIndexInner::MultiU8(idx) => idx.memory_usage(),
        }
    }
}

impl TieredHNSWIndex {
    fn query_internal(&self, query: &[f64], k: usize, params: Option<&QueryParams>) -> PyResult<QueryReply<f64>> {
        let reply = match &self.inner {
            TieredIndexInner::SingleF32(idx) => {
                let q: Vec<f32> = query.iter().map(|&x| x as f32).collect();
                idx.top_k_query(&q, k, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::SingleF64(idx) => idx.top_k_query(query, k, params),
            TieredIndexInner::SingleBF16(idx) => {
                let q: Vec<BFloat16> = query.iter().map(|&x| BFloat16::from_f32(x as f32)).collect();
                idx.top_k_query(&q, k, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::SingleF16(idx) => {
                let q: Vec<Float16> = query.iter().map(|&x| Float16::from_f32(x as f32)).collect();
                idx.top_k_query(&q, k, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::SingleI8(idx) => {
                let q: Vec<Int8> = query.iter().map(|&x| Int8(x as i8)).collect();
                idx.top_k_query(&q, k, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::SingleU8(idx) => {
                let q: Vec<UInt8> = query.iter().map(|&x| UInt8(x as u8)).collect();
                idx.top_k_query(&q, k, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::MultiF32(idx) => {
                let q: Vec<f32> = query.iter().map(|&x| x as f32).collect();
                idx.top_k_query(&q, k, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::MultiF64(idx) => idx.top_k_query(query, k, params),
            TieredIndexInner::MultiBF16(idx) => {
                let q: Vec<BFloat16> = query.iter().map(|&x| BFloat16::from_f32(x as f32)).collect();
                idx.top_k_query(&q, k, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::MultiF16(idx) => {
                let q: Vec<Float16> = query.iter().map(|&x| Float16::from_f32(x as f32)).collect();
                idx.top_k_query(&q, k, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::MultiI8(idx) => {
                let q: Vec<Int8> = query.iter().map(|&x| Int8(x as i8)).collect();
                idx.top_k_query(&q, k, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::MultiU8(idx) => {
                let q: Vec<UInt8> = query.iter().map(|&x| UInt8(x as u8)).collect();
                idx.top_k_query(&q, k, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
        };

        reply.map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))
    }

    fn range_query_internal(&self, query: &[f64], radius: f64, params: Option<&QueryParams>) -> PyResult<QueryReply<f64>> {
        let reply = match &self.inner {
            TieredIndexInner::SingleF32(idx) => {
                let q: Vec<f32> = query.iter().map(|&x| x as f32).collect();
                idx.range_query(&q, radius as f32, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::SingleF64(idx) => idx.range_query(query, radius, params),
            TieredIndexInner::SingleBF16(idx) => {
                let q: Vec<BFloat16> = query.iter().map(|&x| BFloat16::from_f32(x as f32)).collect();
                idx.range_query(&q, radius as f32, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::SingleF16(idx) => {
                let q: Vec<Float16> = query.iter().map(|&x| Float16::from_f32(x as f32)).collect();
                idx.range_query(&q, radius as f32, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::SingleI8(idx) => {
                let q: Vec<Int8> = query.iter().map(|&x| Int8(x as i8)).collect();
                idx.range_query(&q, radius as f32, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::SingleU8(idx) => {
                let q: Vec<UInt8> = query.iter().map(|&x| UInt8(x as u8)).collect();
                idx.range_query(&q, radius as f32, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::MultiF32(idx) => {
                let q: Vec<f32> = query.iter().map(|&x| x as f32).collect();
                idx.range_query(&q, radius as f32, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::MultiF64(idx) => idx.range_query(query, radius, params),
            TieredIndexInner::MultiBF16(idx) => {
                let q: Vec<BFloat16> = query.iter().map(|&x| BFloat16::from_f32(x as f32)).collect();
                idx.range_query(&q, radius as f32, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::MultiF16(idx) => {
                let q: Vec<Float16> = query.iter().map(|&x| Float16::from_f32(x as f32)).collect();
                idx.range_query(&q, radius as f32, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::MultiI8(idx) => {
                let q: Vec<Int8> = query.iter().map(|&x| Int8(x as i8)).collect();
                idx.range_query(&q, radius as f32, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
            TieredIndexInner::MultiU8(idx) => {
                let q: Vec<UInt8> = query.iter().map(|&x| UInt8(x as u8)).collect();
                idx.range_query(&q, radius as f32, params)
                    .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() })
            }
        };

        reply.map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))
    }

    fn get_batch_results(&self, query: &[f64], ef: usize) -> PyResult<Vec<(u64, f64)>> {
        let total = self.index_size();
        if total == 0 {
            return Ok(Vec::new());
        }

        let params = QueryParams::new().with_ef_runtime(ef);
        let reply = self.query_internal(query, total, Some(&params))?;
        Ok(reply.results.into_iter().map(|r| (r.label, r.distance)).collect())
    }
}

// ============================================================================
// SVS Index
// ============================================================================

enum SvsIndexInner {
    SingleF32(SvsSingle<f32>),
    SingleF64(SvsSingle<f64>),
    SingleBF16(SvsSingle<BFloat16>),
    SingleF16(SvsSingle<Float16>),
    SingleI8(SvsSingle<Int8>),
    SingleU8(SvsSingle<UInt8>),
    MultiF32(SvsMulti<f32>),
    MultiF64(SvsMulti<f64>),
    MultiBF16(SvsMulti<BFloat16>),
    MultiF16(SvsMulti<Float16>),
    MultiI8(SvsMulti<Int8>),
    MultiU8(SvsMulti<UInt8>),
}

macro_rules! dispatch_svs_index {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match &$self.inner {
            SvsIndexInner::SingleF32(idx) => idx.$method($($args),*),
            SvsIndexInner::SingleF64(idx) => idx.$method($($args),*),
            SvsIndexInner::SingleBF16(idx) => idx.$method($($args),*),
            SvsIndexInner::SingleF16(idx) => idx.$method($($args),*),
            SvsIndexInner::SingleI8(idx) => idx.$method($($args),*),
            SvsIndexInner::SingleU8(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiF32(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiF64(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiBF16(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiF16(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiI8(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiU8(idx) => idx.$method($($args),*),
        }
    };
}

macro_rules! dispatch_svs_index_mut {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match &mut $self.inner {
            SvsIndexInner::SingleF32(idx) => idx.$method($($args),*),
            SvsIndexInner::SingleF64(idx) => idx.$method($($args),*),
            SvsIndexInner::SingleBF16(idx) => idx.$method($($args),*),
            SvsIndexInner::SingleF16(idx) => idx.$method($($args),*),
            SvsIndexInner::SingleI8(idx) => idx.$method($($args),*),
            SvsIndexInner::SingleU8(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiF32(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiF64(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiBF16(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiF16(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiI8(idx) => idx.$method($($args),*),
            SvsIndexInner::MultiU8(idx) => idx.$method($($args),*),
        }
    };
}

/// SVS (Vamana) index for approximate nearest neighbor search.
#[pyclass]
pub struct SVSIndex {
    inner: SvsIndexInner,
    data_type: u32,
    metric: Metric,
    dim: usize,
    multi: bool,
    search_window_size: usize,
}

#[pymethods]
impl SVSIndex {
    /// Create a new SVS index from parameters.
    #[new]
    fn new(params: &SVSParams) -> PyResult<Self> {
        let metric = metric_from_u32(params.metric)?;
        let svs_params = SvsParams::new(params.dim, metric)
            .with_graph_degree(params.graph_max_degree)
            .with_alpha(params.alpha as f32)
            .with_construction_l(params.construction_window_size)
            .with_search_l(params.search_window_size)
            .with_two_pass(true);

        let inner = match (params.multi, params.r#type) {
            (false, VECSIM_TYPE_FLOAT32) => SvsIndexInner::SingleF32(SvsSingle::new(svs_params)),
            (false, VECSIM_TYPE_FLOAT64) => SvsIndexInner::SingleF64(SvsSingle::new(svs_params)),
            (false, VECSIM_TYPE_BFLOAT16) => SvsIndexInner::SingleBF16(SvsSingle::new(svs_params)),
            (false, VECSIM_TYPE_FLOAT16) => SvsIndexInner::SingleF16(SvsSingle::new(svs_params)),
            (false, VECSIM_TYPE_INT8) => SvsIndexInner::SingleI8(SvsSingle::new(svs_params)),
            (false, VECSIM_TYPE_UINT8) => SvsIndexInner::SingleU8(SvsSingle::new(svs_params)),
            (true, VECSIM_TYPE_FLOAT32) => SvsIndexInner::MultiF32(SvsMulti::new(svs_params)),
            (true, VECSIM_TYPE_FLOAT64) => SvsIndexInner::MultiF64(SvsMulti::new(svs_params)),
            (true, VECSIM_TYPE_BFLOAT16) => SvsIndexInner::MultiBF16(SvsMulti::new(svs_params)),
            (true, VECSIM_TYPE_FLOAT16) => SvsIndexInner::MultiF16(SvsMulti::new(svs_params)),
            (true, VECSIM_TYPE_INT8) => SvsIndexInner::MultiI8(SvsMulti::new(svs_params)),
            (true, VECSIM_TYPE_UINT8) => SvsIndexInner::MultiU8(SvsMulti::new(svs_params)),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported data type: {}",
                    params.r#type
                )))
            }
        };

        Ok(SVSIndex {
            inner,
            data_type: params.r#type,
            metric,
            dim: params.dim,
            multi: params.multi,
            search_window_size: params.search_window_size,
        })
    }

    /// Add a vector to the index.
    fn add_vector(&mut self, py: Python<'_>, vector: PyObject, label: u64) -> PyResult<()> {
        match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let arr: PyReadonlyArray1<f32> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                match &mut self.inner {
                    SvsIndexInner::SingleF32(idx) => idx.add_vector(slice, label),
                    SvsIndexInner::MultiF32(idx) => idx.add_vector(slice, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT64 => {
                let arr: PyReadonlyArray1<f64> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                match &mut self.inner {
                    SvsIndexInner::SingleF64(idx) => idx.add_vector(slice, label),
                    SvsIndexInner::MultiF64(idx) => idx.add_vector(slice, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_BFLOAT16 => {
                let bf16_vec = extract_bf16_vector(py, &vector)?;
                match &mut self.inner {
                    SvsIndexInner::SingleBF16(idx) => idx.add_vector(&bf16_vec, label),
                    SvsIndexInner::MultiBF16(idx) => idx.add_vector(&bf16_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT16 => {
                let f16_vec = extract_f16_vector(py, &vector)?;
                match &mut self.inner {
                    SvsIndexInner::SingleF16(idx) => idx.add_vector(&f16_vec, label),
                    SvsIndexInner::MultiF16(idx) => idx.add_vector(&f16_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_INT8 => {
                let arr: PyReadonlyArray1<i8> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                let i8_vec: Vec<Int8> = slice.iter().map(|&v| Int8(v)).collect();
                match &mut self.inner {
                    SvsIndexInner::SingleI8(idx) => idx.add_vector(&i8_vec, label),
                    SvsIndexInner::MultiI8(idx) => idx.add_vector(&i8_vec, label),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_UINT8 => {
                let arr: PyReadonlyArray1<u8> = vector.extract(py)?;
                let slice = arr.as_slice()?;
                let u8_vec: Vec<UInt8> = slice.iter().map(|&v| UInt8(v)).collect();
                match &mut self.inner {
                    SvsIndexInner::SingleU8(idx) => idx.add_vector(&u8_vec, label),
                    SvsIndexInner::MultiU8(idx) => idx.add_vector(&u8_vec, label),
                    _ => unreachable!(),
                }
            }
            _ => return Err(PyValueError::new_err("Unsupported data type")),
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to add vector: {:?}", e)))?;
        Ok(())
    }

    /// Add multiple vectors in parallel.
    #[pyo3(signature = (vectors, labels, num_threads=None))]
    fn add_vector_parallel(
        &mut self,
        py: Python<'_>,
        vectors: PyObject,
        labels: PyObject,
        num_threads: Option<usize>,
    ) -> PyResult<()> {
        // Set thread pool size if specified
        if let Some(threads) = num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok(); // Ignore error if already initialized
        }

        // Extract labels
        let labels_arr: PyReadonlyArray1<i64> = labels.extract(py)?;
        let labels_vec: Vec<u64> = labels_arr.as_slice()?.iter().map(|&l| l as u64).collect();

        // For parallel insertion, we need to collect all vectors first
        // Then add them sequentially (SVS doesn't support parallel insertion directly)
        match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let arr: PyReadonlyArray2<f32> = vectors.extract(py)?;
                let shape = arr.shape();
                let num_vectors = shape[0];
                let dim = shape[1];
                if dim != self.dim {
                    return Err(PyValueError::new_err(format!(
                        "Vector dimension {} does not match index dimension {}",
                        dim, self.dim
                    )));
                }
                let slice = arr.as_slice()?;

                for i in 0..num_vectors {
                    let start = i * dim;
                    let end = start + dim;
                    let vector = &slice[start..end];
                    let label = labels_vec[i];

                    match &mut self.inner {
                        SvsIndexInner::SingleF32(idx) => idx.add_vector(vector, label),
                        SvsIndexInner::MultiF32(idx) => idx.add_vector(vector, label),
                        _ => unreachable!(),
                    }
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to add vector: {:?}", e)))?;
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Parallel add only supported for FLOAT32 currently",
                ))
            }
        }

        Ok(())
    }

    /// Delete a vector from the index.
    fn delete_vector(&mut self, label: u64) -> PyResult<usize> {
        dispatch_svs_index_mut!(self, delete_vector, label)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete vector: {:?}", e)))
    }

    /// Perform a k-nearest neighbors query.
    #[pyo3(signature = (query, k=10, query_param=None))]
    fn knn_query<'py>(
        &self,
        py: Python<'py>,
        query: PyObject,
        k: usize,
        query_param: Option<&VecSimQueryParams>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        let query_vec = extract_query_vec(py, &query)?;
        let search_l = query_param
            .map(|p| p.get_svs_window_size(py))
            .unwrap_or(self.search_window_size);
        let params = QueryParams::new().with_ef_runtime(search_l);
        let reply = self.query_internal(&query_vec, k, Some(&params))?;

        let labels: Vec<i64> = reply.results.iter().map(|r| r.label as i64).collect();
        let distances: Vec<f64> = reply.results.iter().map(|r| r.distance).collect();
        let len = labels.len();

        Ok((vec_to_2d_array(py, labels, len), vec_to_2d_array(py, distances, len)))
    }

    /// Perform a range query.
    #[pyo3(signature = (query, radius, query_param=None))]
    fn range_query<'py>(
        &self,
        py: Python<'py>,
        query: PyObject,
        radius: f64,
        query_param: Option<&VecSimQueryParams>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f64>>)> {
        let query_vec = extract_query_vec(py, &query)?;
        let search_l = query_param
            .map(|p| p.get_svs_window_size(py))
            .unwrap_or(self.search_window_size);
        let params = QueryParams::new().with_ef_runtime(search_l);
        let reply = self.range_query_internal(&query_vec, radius, Some(&params))?;

        let labels: Vec<i64> = reply.results.iter().map(|r| r.label as i64).collect();
        let distances: Vec<f64> = reply.results.iter().map(|r| r.distance).collect();
        let len = labels.len();

        Ok((vec_to_2d_array(py, labels, len), vec_to_2d_array(py, distances, len)))
    }

    /// Get the number of vectors in the index.
    fn index_size(&self) -> usize {
        dispatch_svs_index!(self, index_size)
    }

    /// Get the data type of vectors in the index.
    fn index_type(&self) -> u32 {
        self.data_type
    }

    /// Create a batch iterator for streaming results.
    #[pyo3(signature = (query, query_params=None))]
    fn create_batch_iterator(
        &self,
        py: Python<'_>,
        query: PyObject,
        query_params: Option<&VecSimQueryParams>,
    ) -> PyResult<PyBatchIterator> {
        let query_vec = extract_query_vec(py, &query)?;
        let search_l = query_params
            .map(|p| p.get_svs_window_size(py))
            .unwrap_or(self.search_window_size);

        let all_results = self.get_batch_results(&query_vec, search_l)?;
        let index_size = self.index_size();

        Ok(PyBatchIterator::new(all_results, index_size))
    }
}

impl SVSIndex {
    fn query_internal(&self, query: &[f64], k: usize, params: Option<&QueryParams>) -> PyResult<QueryReply<f64>> {
        let reply = match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let q: Vec<f32> = query.iter().map(|&x| x as f32).collect();
                match &self.inner {
                    SvsIndexInner::SingleF32(idx) => idx.top_k_query(&q, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    SvsIndexInner::MultiF32(idx) => idx.top_k_query(&q, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT64 => {
                match &self.inner {
                    SvsIndexInner::SingleF64(idx) => idx.top_k_query(query, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance)).collect() }),
                    SvsIndexInner::MultiF64(idx) => idx.top_k_query(query, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance)).collect() }),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_BFLOAT16 => {
                let q: Vec<BFloat16> = query.iter().map(|&x| BFloat16::from_f32(x as f32)).collect();
                match &self.inner {
                    SvsIndexInner::SingleBF16(idx) => idx.top_k_query(&q, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect() }),
                    SvsIndexInner::MultiBF16(idx) => idx.top_k_query(&q, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect() }),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT16 => {
                let q: Vec<Float16> = query.iter().map(|&x| Float16::from_f32(x as f32)).collect();
                match &self.inner {
                    SvsIndexInner::SingleF16(idx) => idx.top_k_query(&q, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect() }),
                    SvsIndexInner::MultiF16(idx) => idx.top_k_query(&q, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect() }),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_INT8 => {
                let q: Vec<Int8> = query.iter().map(|&x| Int8(x as i8)).collect();
                match &self.inner {
                    SvsIndexInner::SingleI8(idx) => idx.top_k_query(&q, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    SvsIndexInner::MultiI8(idx) => idx.top_k_query(&q, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_UINT8 => {
                let q: Vec<UInt8> = query.iter().map(|&x| UInt8(x as u8)).collect();
                match &self.inner {
                    SvsIndexInner::SingleU8(idx) => idx.top_k_query(&q, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    SvsIndexInner::MultiU8(idx) => idx.top_k_query(&q, k, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    _ => unreachable!(),
                }
            }
            _ => return Err(PyValueError::new_err("Unsupported data type")),
        };

        reply.map_err(|e| PyRuntimeError::new_err(format!("Query failed: {:?}", e)))
    }

    fn range_query_internal(&self, query: &[f64], radius: f64, params: Option<&QueryParams>) -> PyResult<QueryReply<f64>> {
        let reply = match self.data_type {
            VECSIM_TYPE_FLOAT32 => {
                let q: Vec<f32> = query.iter().map(|&x| x as f32).collect();
                match &self.inner {
                    SvsIndexInner::SingleF32(idx) => idx.range_query(&q, radius as f32, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    SvsIndexInner::MultiF32(idx) => idx.range_query(&q, radius as f32, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT64 => {
                match &self.inner {
                    SvsIndexInner::SingleF64(idx) => idx.range_query(query, radius, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance)).collect() }),
                    SvsIndexInner::MultiF64(idx) => idx.range_query(query, radius, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance)).collect() }),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_BFLOAT16 => {
                let q: Vec<BFloat16> = query.iter().map(|&x| BFloat16::from_f32(x as f32)).collect();
                match &self.inner {
                    SvsIndexInner::SingleBF16(idx) => idx.range_query(&q, radius as f32, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect() }),
                    SvsIndexInner::MultiBF16(idx) => idx.range_query(&q, radius as f32, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect() }),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_FLOAT16 => {
                let q: Vec<Float16> = query.iter().map(|&x| Float16::from_f32(x as f32)).collect();
                match &self.inner {
                    SvsIndexInner::SingleF16(idx) => idx.range_query(&q, radius as f32, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect() }),
                    SvsIndexInner::MultiF16(idx) => idx.range_query(&q, radius as f32, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance.to_f64())).collect() }),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_INT8 => {
                let q: Vec<Int8> = query.iter().map(|&x| Int8(x as i8)).collect();
                match &self.inner {
                    SvsIndexInner::SingleI8(idx) => idx.range_query(&q, radius as f32, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    SvsIndexInner::MultiI8(idx) => idx.range_query(&q, radius as f32, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    _ => unreachable!(),
                }
            }
            VECSIM_TYPE_UINT8 => {
                let q: Vec<UInt8> = query.iter().map(|&x| UInt8(x as u8)).collect();
                match &self.inner {
                    SvsIndexInner::SingleU8(idx) => idx.range_query(&q, radius as f32, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    SvsIndexInner::MultiU8(idx) => idx.range_query(&q, radius as f32, params)
                        .map(|r| QueryReply { results: r.results.into_iter().map(|r| QueryResult::new(r.label, r.distance as f64)).collect() }),
                    _ => unreachable!(),
                }
            }
            _ => return Err(PyValueError::new_err("Unsupported data type")),
        };

        reply.map_err(|e| PyRuntimeError::new_err(format!("Range query failed: {:?}", e)))
    }

    /// Get results for batch iterator with search_l-influenced quality.
    /// Uses progressive queries to ensure search_l affects early results.
    fn get_batch_results(&self, query: &[f64], search_l: usize) -> PyResult<Vec<(u64, f64)>> {
        let total = self.index_size();
        if total == 0 {
            return Ok(Vec::new());
        }

        // Use multiple queries with progressively larger k values.
        // This ensures early results are affected by search_l.
        // SVS uses search_l as the search window size (beam width).

        // Query stages:
        // Stage 1: k = 10 (first batch), with specified search_l
        //   - search_l=5: smaller window, potentially worse results
        //   - search_l=128: larger window, better results
        // Stage 2: k = search_l (get search_l-quality results)
        // Stage 3: k = total (get all remaining), with full search

        let mut results = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();

        // Stage 1: k = 10 (typical first batch size) with specified search_l
        let k1 = 10.min(total);
        let params1 = QueryParams::new().with_ef_runtime(search_l);
        let stage1_reply = self.query_internal(query, k1, Some(&params1))?;
        for r in stage1_reply.results {
            if !seen.contains(&r.label) {
                seen.insert(r.label);
                results.push((r.label, r.distance));
            }
        }

        // Stage 2: k = search_l (use search_l's natural quality)
        if seen.len() < total && search_l > k1 {
            let k2 = search_l.min(total);
            let params2 = QueryParams::new().with_ef_runtime(search_l);
            let stage2_reply = self.query_internal(query, k2, Some(&params2))?;
            for r in stage2_reply.results {
                if !seen.contains(&r.label) {
                    seen.insert(r.label);
                    results.push((r.label, r.distance));
                }
            }
        }

        // Stage 3: Get all remaining results with full search
        if seen.len() < total {
            let params3 = QueryParams::new().with_ef_runtime(total);
            let stage3_reply = self.query_internal(query, total, Some(&params3))?;
            for r in stage3_reply.results {
                if !seen.contains(&r.label) {
                    seen.insert(r.label);
                    results.push((r.label, r.distance));
                }
            }
        }

        Ok(results)
    }
}

// ============================================================================
// SVS Parameters
// ============================================================================

/// SVS (Vamana) index parameters.
#[pyclass]
#[derive(Clone)]
pub struct SVSParams {
    #[pyo3(get, set)]
    pub dim: usize,
    #[pyo3(get, set)]
    pub r#type: u32,
    #[pyo3(get, set)]
    pub metric: u32,
    #[pyo3(get, set)]
    pub multi: bool,
    #[pyo3(get, set)]
    pub quantBits: u32,
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub graph_max_degree: usize,
    #[pyo3(get, set)]
    pub construction_window_size: usize,
    #[pyo3(get, set)]
    pub max_candidate_pool_size: usize,
    #[pyo3(get, set)]
    pub prune_to: usize,
    #[pyo3(get, set)]
    pub use_search_history: u32,
    #[pyo3(get, set)]
    pub search_window_size: usize,
    #[pyo3(get, set)]
    pub search_buffer_capacity: usize,
    #[pyo3(get, set)]
    pub epsilon: f64,
    #[pyo3(get, set)]
    pub num_threads: usize,
}

#[pymethods]
impl SVSParams {
    #[new]
    fn new() -> Self {
        SVSParams {
            dim: 0,
            r#type: VECSIM_TYPE_FLOAT32,
            metric: VECSIM_METRIC_L2,
            multi: false,
            quantBits: VECSIM_SVS_QUANT_NONE,
            alpha: 1.2,
            graph_max_degree: 32,
            construction_window_size: 200,
            max_candidate_pool_size: 0,
            prune_to: 0,
            use_search_history: VECSIM_OPTION_AUTO,
            search_window_size: 100,
            search_buffer_capacity: 100,
            epsilon: 0.01,
            num_threads: 0,
        }
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// VecSim - High-performance vector similarity search library.
#[pymodule]
fn VecSim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("VecSimMetric_L2", VECSIM_METRIC_L2)?;
    m.add("VecSimMetric_IP", VECSIM_METRIC_IP)?;
    m.add("VecSimMetric_Cosine", VECSIM_METRIC_COSINE)?;

    m.add("VecSimType_FLOAT32", VECSIM_TYPE_FLOAT32)?;
    m.add("VecSimType_FLOAT64", VECSIM_TYPE_FLOAT64)?;
    m.add("VecSimType_BFLOAT16", VECSIM_TYPE_BFLOAT16)?;
    m.add("VecSimType_FLOAT16", VECSIM_TYPE_FLOAT16)?;
    m.add("VecSimType_INT8", VECSIM_TYPE_INT8)?;
    m.add("VecSimType_UINT8", VECSIM_TYPE_UINT8)?;

    m.add("BY_SCORE", BY_SCORE)?;
    m.add("BY_ID", BY_ID)?;

    // SVS quantization constants
    m.add("VecSimSvsQuant_NONE", VECSIM_SVS_QUANT_NONE)?;
    m.add("VecSimSvsQuant_8", VECSIM_SVS_QUANT_8)?;
    m.add("VecSimSvsQuant_4", VECSIM_SVS_QUANT_4)?;

    // VecSim option constants
    m.add("VecSimOption_OFF", VECSIM_OPTION_OFF)?;
    m.add("VecSimOption_ON", VECSIM_OPTION_ON)?;
    m.add("VecSimOption_AUTO", VECSIM_OPTION_AUTO)?;

    m.add_class::<BFParams>()?;
    m.add_class::<HNSWParams>()?;
    m.add_class::<HNSWRuntimeParams>()?;
    m.add_class::<SVSParams>()?;
    m.add_class::<SVSRuntimeParams>()?;
    m.add_class::<TieredHNSWParams>()?;
    m.add_class::<VecSimQueryParams>()?;
    m.add_class::<BFIndex>()?;
    m.add_class::<HNSWIndex>()?;
    m.add_class::<SVSIndex>()?;
    m.add_class::<TieredHNSWIndex>()?;
    m.add_class::<PyBatchIterator>()?;

    Ok(())
}

//! BruteForce index implementation.
//!
//! The BruteForce index performs linear scans over all vectors to find nearest neighbors.
//! While this has O(n) query complexity, it provides exact results and is efficient
//! for small datasets or when high recall is critical.
//!
//! Two variants are provided:
//! - `BruteForceSingle`: One vector per label (new vector replaces existing)
//! - `BruteForceMulti`: Multiple vectors allowed per label

pub mod batch_iterator;
pub mod multi;
pub mod single;

pub use batch_iterator::BruteForceBatchIterator;
pub use multi::BruteForceMulti;
pub use single::{BruteForceSingle, BruteForceStats};

use crate::containers::DataBlocks;
use crate::distance::{create_distance_function, DistanceFunction, Metric};
use crate::types::{DistanceType, IdType, LabelType, VectorElement};

/// Parameters for creating a BruteForce index.
#[derive(Debug, Clone)]
pub struct BruteForceParams {
    /// Vector dimension.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Initial capacity (number of vectors).
    pub initial_capacity: usize,
    /// Block size for vector storage.
    pub block_size: Option<usize>,
}

impl BruteForceParams {
    /// Create new parameters with required fields.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            initial_capacity: 1024,
            block_size: None,
        }
    }

    /// Set initial capacity.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = capacity;
        self
    }

    /// Set block size.
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = Some(size);
        self
    }
}

/// Shared state for BruteForce indices.
pub(crate) struct BruteForceCore<T: VectorElement> {
    /// Vector storage.
    pub data: DataBlocks<T>,
    /// Distance function.
    pub dist_fn: Box<dyn DistanceFunction<T, Output = T::DistanceType>>,
    /// Vector dimension.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
}

impl<T: VectorElement> BruteForceCore<T> {
    /// Create a new BruteForce core.
    pub fn new(params: &BruteForceParams) -> Self {
        let data = if let Some(block_size) = params.block_size {
            DataBlocks::with_block_size(params.dim, params.initial_capacity, block_size)
        } else {
            DataBlocks::new(params.dim, params.initial_capacity)
        };

        let dist_fn = create_distance_function(params.metric, params.dim);

        Self {
            data,
            dist_fn,
            dim: params.dim,
            metric: params.metric,
        }
    }

    /// Add a vector and return its internal ID.
    ///
    /// Returns `None` if the vector dimension doesn't match.
    #[inline]
    pub fn add_vector(&mut self, vector: &[T]) -> Option<IdType> {
        // Preprocess if needed (e.g., normalize for cosine)
        let processed = self.dist_fn.preprocess(vector, self.dim);
        self.data.add(&processed)
    }

    /// Get a vector by ID.
    #[inline]
    #[allow(dead_code)]
    pub fn get_vector(&self, id: IdType) -> Option<&[T]> {
        self.data.get(id)
    }

    /// Compute distance between stored vector and query.
    #[inline]
    pub fn compute_distance(&self, id: IdType, query: &[T]) -> T::DistanceType {
        if let Some(stored) = self.data.get(id) {
            self.dist_fn
                .compute_from_preprocessed(stored, query, self.dim)
        } else {
            T::DistanceType::infinity()
        }
    }
}

/// Entry in the id-to-label mapping.
#[derive(Clone, Copy)]
pub(crate) struct IdLabelEntry {
    pub label: LabelType,
    pub is_valid: bool,
}

impl Default for IdLabelEntry {
    fn default() -> Self {
        Self {
            label: 0,
            is_valid: false,
        }
    }
}

//! Core index traits defining the vector similarity interface.
//!
//! This module defines the abstract interfaces that all index implementations must support:
//! - `VecSimIndex`: The main index trait for vector storage and search
//! - `BatchIterator`: Iterator for streaming query results in batches

use crate::query::{QueryParams, QueryReply};
use crate::types::{DistanceType, IdType, LabelType, VectorElement};
use thiserror::Error;

/// Errors that can occur during index operations.
#[derive(Error, Debug)]
pub enum IndexError {
    #[error("Vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Index is at capacity ({capacity})")]
    CapacityExceeded { capacity: usize },

    #[error("Label {0} not found in index")]
    LabelNotFound(LabelType),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Index is empty")]
    EmptyIndex,

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Index data corruption detected: {0}")]
    Corruption(String),

    #[error("Memory allocation failed")]
    MemoryExhausted,

    #[error("Duplicate label: {0} already exists")]
    DuplicateLabel(LabelType),
}

/// Errors that can occur during query operations.
#[derive(Error, Debug)]
pub enum QueryError {
    #[error("Vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid query parameter: {0}")]
    InvalidParameter(String),

    #[error("Query cancelled")]
    Cancelled,

    #[error("Query timed out after {0}ms")]
    Timeout(u64),

    #[error("Empty query vector")]
    EmptyVector,

    #[error("Vector is not normalized (required for this metric)")]
    NotNormalized,
}

/// Information about the index.
#[derive(Debug, Clone)]
pub struct IndexInfo {
    /// Number of vectors currently stored.
    pub size: usize,
    /// Maximum capacity (if bounded).
    pub capacity: Option<usize>,
    /// Vector dimension.
    pub dimension: usize,
    /// Index type name.
    pub index_type: &'static str,
    /// Memory usage in bytes (approximate).
    pub memory_bytes: usize,
}

/// The main trait for vector similarity indices.
///
/// This trait defines the core operations supported by all index types:
/// - Adding and removing vectors
/// - KNN (top-k) queries
/// - Range queries
/// - Batch iteration
pub trait VecSimIndex: Send + Sync {
    /// The vector element type (f32, f64, Float16, BFloat16).
    type DataType: VectorElement;

    /// The distance computation result type.
    type DistType: DistanceType;

    /// Add a vector with the given label to the index.
    ///
    /// Returns the number of vectors added (1 for single-value indices,
    /// potentially more for multi-value indices that store duplicates).
    ///
    /// # Errors
    /// - `DimensionMismatch` if the vector has the wrong dimension
    /// - `CapacityExceeded` if the index is full
    fn add_vector(
        &mut self,
        vector: &[Self::DataType],
        label: LabelType,
    ) -> Result<usize, IndexError>;

    /// Delete all vectors with the given label.
    ///
    /// Returns the number of vectors deleted.
    ///
    /// # Errors
    /// - `LabelNotFound` if no vectors have this label
    fn delete_vector(&mut self, label: LabelType) -> Result<usize, IndexError>;

    /// Perform a top-k nearest neighbor query.
    ///
    /// Returns up to `k` vectors closest to the query vector.
    ///
    /// # Arguments
    /// * `query` - The query vector
    /// * `k` - Maximum number of results to return
    /// * `params` - Optional query parameters
    fn top_k_query(
        &self,
        query: &[Self::DataType],
        k: usize,
        params: Option<&QueryParams>,
    ) -> Result<QueryReply<Self::DistType>, QueryError>;

    /// Perform a range query.
    ///
    /// Returns all vectors within the given radius of the query vector.
    ///
    /// # Arguments
    /// * `query` - The query vector
    /// * `radius` - Maximum distance from query
    /// * `params` - Optional query parameters
    fn range_query(
        &self,
        query: &[Self::DataType],
        radius: Self::DistType,
        params: Option<&QueryParams>,
    ) -> Result<QueryReply<Self::DistType>, QueryError>;

    /// Get the current number of vectors in the index.
    fn index_size(&self) -> usize;

    /// Get the maximum capacity of the index (if bounded).
    fn index_capacity(&self) -> Option<usize>;

    /// Get the vector dimension.
    fn dimension(&self) -> usize;

    /// Get a batch iterator for streaming query results.
    ///
    /// This is useful for processing large result sets incrementally
    /// without loading all results into memory at once.
    fn batch_iterator<'a>(
        &'a self,
        query: &[Self::DataType],
        params: Option<&QueryParams>,
    ) -> Result<Box<dyn BatchIterator<DistType = Self::DistType> + 'a>, QueryError>;

    /// Get information about the index.
    fn info(&self) -> IndexInfo;

    /// Check if a label exists in the index.
    fn contains(&self, label: LabelType) -> bool;

    /// Get the number of vectors associated with a label.
    fn label_count(&self, label: LabelType) -> usize;
}

/// Iterator for streaming query results in batches.
///
/// This trait allows processing query results incrementally, which is useful
/// for large result sets or when results need to be processed progressively.
pub trait BatchIterator: Send {
    /// The distance type for results.
    type DistType: DistanceType;

    /// Check if there are more results available.
    fn has_next(&self) -> bool;

    /// Get the next batch of results.
    ///
    /// Returns `None` when no more results are available.
    fn next_batch(&mut self, batch_size: usize)
        -> Option<Vec<(IdType, LabelType, Self::DistType)>>;

    /// Reset the iterator to the beginning.
    fn reset(&mut self);
}

/// Index type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    BruteForce,
    HNSW,
    Tiered,
    Svs,
    TieredSvs,
    DiskIndex,
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexType::BruteForce => write!(f, "BruteForce"),
            IndexType::HNSW => write!(f, "HNSW"),
            IndexType::Tiered => write!(f, "Tiered"),
            IndexType::Svs => write!(f, "Svs"),
            IndexType::TieredSvs => write!(f, "TieredSvs"),
            IndexType::DiskIndex => write!(f, "DiskIndex"),
        }
    }
}

/// Whether the index supports multiple vectors per label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiValue {
    /// Single vector per label (new vector replaces existing).
    Single,
    /// Multiple vectors allowed per label.
    Multi,
}

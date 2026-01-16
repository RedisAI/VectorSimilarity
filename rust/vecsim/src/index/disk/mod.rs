//! Disk-based vector index using memory-mapped storage.
//!
//! This module provides persistent vector indices that use memory-mapped files
//! for vector storage, allowing datasets larger than RAM.
//!
//! # Backend Options
//!
//! - **BruteForce**: Linear scan over all vectors (exact results, O(n))
//! - **Vamana**: SVS graph structure for approximate search (fast, O(log n))

pub mod single;

pub use single::DiskIndexSingle;

use crate::distance::Metric;
use std::path::PathBuf;

/// Backend type for the disk index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiskBackend {
    /// Linear scan (exact results).
    #[default]
    BruteForce,
    /// Vamana graph (approximate, fast).
    Vamana,
}

/// Parameters for creating a disk-based index.
#[derive(Debug, Clone)]
pub struct DiskIndexParams {
    /// Vector dimension.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Path to the data file.
    pub data_path: PathBuf,
    /// Backend algorithm.
    pub backend: DiskBackend,
    /// Initial capacity.
    pub initial_capacity: usize,
    /// Graph max degree (for Vamana backend).
    pub graph_max_degree: usize,
    /// Alpha parameter (for Vamana backend).
    pub alpha: f32,
    /// Construction window size (for Vamana backend).
    pub construction_l: usize,
    /// Search window size (for Vamana backend).
    pub search_l: usize,
}

impl DiskIndexParams {
    /// Create new disk index parameters.
    pub fn new<P: Into<PathBuf>>(dim: usize, metric: Metric, data_path: P) -> Self {
        Self {
            dim,
            metric,
            data_path: data_path.into(),
            backend: DiskBackend::BruteForce,
            initial_capacity: 10_000,
            graph_max_degree: 32,
            alpha: 1.2,
            construction_l: 200,
            search_l: 100,
        }
    }

    /// Set the backend algorithm.
    pub fn with_backend(mut self, backend: DiskBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set initial capacity.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = capacity;
        self
    }

    /// Set graph max degree (for Vamana backend).
    pub fn with_graph_degree(mut self, degree: usize) -> Self {
        self.graph_max_degree = degree;
        self
    }

    /// Set alpha parameter (for Vamana backend).
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set construction window size (for Vamana backend).
    pub fn with_construction_l(mut self, l: usize) -> Self {
        self.construction_l = l;
        self
    }

    /// Set search window size (for Vamana backend).
    pub fn with_search_l(mut self, l: usize) -> Self {
        self.search_l = l;
        self
    }
}

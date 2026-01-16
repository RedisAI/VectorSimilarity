//! Query parameter configuration.

use crate::types::LabelType;

/// Parameters for controlling query execution.
#[derive(Default)]
pub struct QueryParams {
    /// For HNSW: the size of the dynamic candidate list during search (ef).
    /// Higher values improve recall at the cost of speed.
    /// If None, uses the index's default ef_runtime value.
    pub ef_runtime: Option<usize>,

    /// Maximum number of results to return.
    /// For batch iterators, this may be used to hint at batch sizes.
    pub batch_size: Option<usize>,

    /// Filter function to exclude certain labels from results.
    /// If Some, only vectors whose labels pass the filter are included.
    pub filter: Option<Box<dyn Fn(LabelType) -> bool + Send + Sync>>,

    /// Enable parallel query execution if supported.
    pub parallel: bool,
}

impl std::fmt::Debug for QueryParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryParams")
            .field("ef_runtime", &self.ef_runtime)
            .field("batch_size", &self.batch_size)
            .field("filter", &self.filter.as_ref().map(|_| "<filter fn>"))
            .field("parallel", &self.parallel)
            .finish()
    }
}

impl Clone for QueryParams {
    fn clone(&self) -> Self {
        Self {
            ef_runtime: self.ef_runtime,
            batch_size: self.batch_size,
            filter: None, // Filter cannot be cloned
            parallel: self.parallel,
        }
    }
}


impl QueryParams {
    /// Create new query parameters with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ef_runtime parameter for HNSW search.
    pub fn with_ef_runtime(mut self, ef: usize) -> Self {
        self.ef_runtime = Some(ef);
        self
    }

    /// Set the batch size hint.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Set a filter function.
    pub fn with_filter<F>(mut self, filter: F) -> Self
    where
        F: Fn(LabelType) -> bool + Send + Sync + 'static,
    {
        self.filter = Some(Box::new(filter));
        self
    }

    /// Enable parallel query execution.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Check if a label passes the filter (if any).
    #[inline]
    pub fn passes_filter(&self, label: LabelType) -> bool {
        self.filter.as_ref().is_none_or(|f| f(label))
    }
}

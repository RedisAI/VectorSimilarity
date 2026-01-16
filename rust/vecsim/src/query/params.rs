//! Query parameter configuration.

use crate::types::LabelType;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

    /// Timeout callback function.
    /// Returns true if the query should be cancelled.
    /// This is checked periodically during search operations.
    pub timeout_callback: Option<Box<dyn Fn() -> bool + Send + Sync>>,

    /// Query timeout duration.
    /// If set, creates an automatic timeout based on elapsed time.
    pub timeout: Option<Duration>,
}

impl std::fmt::Debug for QueryParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryParams")
            .field("ef_runtime", &self.ef_runtime)
            .field("batch_size", &self.batch_size)
            .field("filter", &self.filter.as_ref().map(|_| "<filter fn>"))
            .field("parallel", &self.parallel)
            .field(
                "timeout_callback",
                &self.timeout_callback.as_ref().map(|_| "<timeout fn>"),
            )
            .field("timeout", &self.timeout)
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
            timeout_callback: None, // Callback cannot be cloned
            timeout: self.timeout,
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

    /// Set a timeout callback function.
    ///
    /// The callback is invoked periodically during search. If it returns `true`,
    /// the search is cancelled and returns with partial results or an error.
    pub fn with_timeout_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn() -> bool + Send + Sync + 'static,
    {
        self.timeout_callback = Some(Box::new(callback));
        self
    }

    /// Set a timeout duration.
    ///
    /// The query will be cancelled if it exceeds this duration.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set a timeout in milliseconds.
    pub fn with_timeout_ms(self, ms: u64) -> Self {
        self.with_timeout(Duration::from_millis(ms))
    }

    /// Create a timeout checker that can be used during search.
    ///
    /// Returns a TimeoutChecker if a timeout duration is set.
    pub fn create_timeout_checker(&self) -> Option<TimeoutChecker> {
        TimeoutChecker::from_params(self)
    }

    /// Check if the query should be timed out.
    ///
    /// This checks both the timeout duration (if set) and the timeout callback (if set).
    #[inline]
    pub fn is_timed_out(&self, start_time: Instant) -> bool {
        // Check duration-based timeout
        if let Some(timeout) = self.timeout {
            if start_time.elapsed() >= timeout {
                return true;
            }
        }

        // Check callback-based timeout
        if let Some(ref callback) = self.timeout_callback {
            if callback() {
                return true;
            }
        }

        false
    }
}

/// Helper struct for efficient timeout checking during search.
///
/// This struct caches the start time and provides efficient timeout checking
/// with configurable check intervals to minimize overhead.
pub struct TimeoutChecker {
    start_time: Instant,
    timeout: Option<Duration>,
    check_interval: usize,
    check_counter: usize,
    timed_out: bool,
}

impl TimeoutChecker {
    /// Create a new timeout checker with a duration.
    pub fn with_duration(timeout: Duration) -> Self {
        Self {
            start_time: Instant::now(),
            timeout: Some(timeout),
            check_interval: 64, // Check every 64 iterations
            check_counter: 0,
            timed_out: false,
        }
    }

    /// Create a new timeout checker from query params.
    pub fn from_params(params: &QueryParams) -> Option<Self> {
        params.timeout.map(Self::with_duration)
    }

    /// Check if the query should time out.
    ///
    /// This method is optimized to only perform the actual check every N iterations
    /// to minimize overhead in tight loops.
    #[inline]
    pub fn check(&mut self) -> bool {
        if self.timed_out {
            return true;
        }

        self.check_counter += 1;
        if self.check_counter < self.check_interval {
            return false;
        }

        self.check_counter = 0;
        self.timed_out = self.check_now();
        self.timed_out
    }

    /// Force an immediate timeout check.
    #[inline]
    pub fn check_now(&self) -> bool {
        if let Some(timeout) = self.timeout {
            if self.start_time.elapsed() >= timeout {
                return true;
            }
        }
        false
    }

    /// Get the elapsed time since the checker was created.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Check if the timeout has already been triggered.
    pub fn is_timed_out(&self) -> bool {
        self.timed_out
    }

    /// Get the elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
}

/// A cancellation token that can be shared across threads.
///
/// This is useful for implementing query cancellation from external code.
#[derive(Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Cancel the associated operation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Create a timeout callback from this token.
    ///
    /// This can be passed to `QueryParams::with_timeout_callback`.
    pub fn as_callback(&self) -> impl Fn() -> bool + Send + Sync + 'static {
        let cancelled = Arc::clone(&self.cancelled);
        move || cancelled.load(Ordering::Acquire)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

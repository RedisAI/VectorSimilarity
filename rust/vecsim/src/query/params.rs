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

    /// Epsilon for range query search boundary expansion.
    /// Controls how far beyond the current best distance to search.
    /// E.g., 0.01 means search 1% beyond the dynamic range.
    /// If None, uses default of 0.01.
    pub epsilon: Option<f64>,
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
            epsilon: self.epsilon,
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

    /// Set epsilon for range query search boundary expansion.
    ///
    /// Controls how far beyond the current best distance to search.
    /// E.g., 0.01 means search 1% beyond the dynamic range.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = Some(epsilon);
        self
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_params_default() {
        let params = QueryParams::new();
        assert_eq!(params.ef_runtime, None);
        assert_eq!(params.batch_size, None);
        assert!(params.filter.is_none());
        assert!(!params.parallel);
        assert!(params.timeout_callback.is_none());
        assert!(params.timeout.is_none());
    }

    #[test]
    fn test_query_params_builder() {
        let params = QueryParams::new()
            .with_ef_runtime(100)
            .with_batch_size(50)
            .with_parallel(true)
            .with_timeout_ms(1000);

        assert_eq!(params.ef_runtime, Some(100));
        assert_eq!(params.batch_size, Some(50));
        assert!(params.parallel);
        assert_eq!(params.timeout, Some(Duration::from_millis(1000)));
    }

    #[test]
    fn test_query_params_with_filter() {
        let params = QueryParams::new().with_filter(|label| label % 2 == 0);

        assert!(params.filter.is_some());
        assert!(params.passes_filter(2));
        assert!(params.passes_filter(4));
        assert!(!params.passes_filter(1));
        assert!(!params.passes_filter(3));
    }

    #[test]
    fn test_query_params_passes_filter_no_filter() {
        let params = QueryParams::new();
        // Without a filter, all labels should pass
        assert!(params.passes_filter(0));
        assert!(params.passes_filter(100));
        assert!(params.passes_filter(u64::MAX));
    }

    #[test]
    fn test_query_params_with_timeout_callback() {
        let should_cancel = Arc::new(AtomicBool::new(false));
        let should_cancel_clone = Arc::clone(&should_cancel);

        let params = QueryParams::new()
            .with_timeout_callback(move || should_cancel_clone.load(Ordering::Acquire));

        let start = Instant::now();

        // Initially not timed out
        assert!(!params.is_timed_out(start));

        // Set the cancel flag
        should_cancel.store(true, Ordering::Release);

        // Now should be timed out
        assert!(params.is_timed_out(start));
    }

    #[test]
    fn test_query_params_with_timeout_duration() {
        let params = QueryParams::new().with_timeout(Duration::from_millis(10));

        let start = Instant::now();

        // Initially not timed out
        assert!(!params.is_timed_out(start));

        // Wait for timeout to expire
        std::thread::sleep(Duration::from_millis(15));

        // Now should be timed out
        assert!(params.is_timed_out(start));
    }

    #[test]
    fn test_query_params_clone() {
        let params = QueryParams::new()
            .with_ef_runtime(50)
            .with_batch_size(25)
            .with_parallel(true)
            .with_timeout_ms(500)
            .with_filter(|_| true);

        let cloned = params.clone();

        assert_eq!(cloned.ef_runtime, Some(50));
        assert_eq!(cloned.batch_size, Some(25));
        assert!(cloned.parallel);
        assert_eq!(cloned.timeout, Some(Duration::from_millis(500)));
        // Filter cannot be cloned
        assert!(cloned.filter.is_none());
    }

    #[test]
    fn test_query_params_debug() {
        let params = QueryParams::new()
            .with_ef_runtime(100)
            .with_filter(|_| true)
            .with_timeout_callback(|| false);

        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("QueryParams"));
        assert!(debug_str.contains("ef_runtime"));
        assert!(debug_str.contains("<filter fn>"));
        assert!(debug_str.contains("<timeout fn>"));
    }

    #[test]
    fn test_query_params_create_timeout_checker() {
        let params_no_timeout = QueryParams::new();
        assert!(params_no_timeout.create_timeout_checker().is_none());

        let params_with_timeout = QueryParams::new().with_timeout_ms(100);
        assert!(params_with_timeout.create_timeout_checker().is_some());
    }

    #[test]
    fn test_timeout_checker_with_duration() {
        let mut checker = TimeoutChecker::with_duration(Duration::from_millis(50));

        // Initially not timed out
        assert!(!checker.is_timed_out());
        assert!(!checker.check_now());

        // Should not timeout immediately
        for _ in 0..100 {
            if checker.check() {
                break;
            }
        }

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(60));

        // Now should timeout on check_now
        assert!(checker.check_now());

        // Force check() to do actual time check by calling enough times
        // to reach the check_interval (64)
        for _ in 0..64 {
            if checker.check() {
                break;
            }
        }
        // After enough iterations, check() will have detected timeout
        assert!(checker.is_timed_out());
    }

    #[test]
    fn test_timeout_checker_check_interval() {
        let mut checker = TimeoutChecker::with_duration(Duration::from_secs(100)); // Long timeout

        // The first N-1 checks should return false (doesn't actually check time)
        for _ in 0..63 {
            assert!(!checker.check());
        }

        // The Nth check (64th) triggers actual time check
        // Since timeout is 100s, should still be false
        assert!(!checker.check());
    }

    #[test]
    fn test_timeout_checker_elapsed() {
        let checker = TimeoutChecker::with_duration(Duration::from_secs(10));

        std::thread::sleep(Duration::from_millis(10));

        let elapsed = checker.elapsed();
        assert!(elapsed >= Duration::from_millis(10));

        let elapsed_ms = checker.elapsed_ms();
        assert!(elapsed_ms >= 10);
    }

    #[test]
    fn test_timeout_checker_from_params() {
        let params = QueryParams::new().with_timeout_ms(100);
        let checker = TimeoutChecker::from_params(&params);
        assert!(checker.is_some());

        let params_no_timeout = QueryParams::new();
        let checker_none = TimeoutChecker::from_params(&params_no_timeout);
        assert!(checker_none.is_none());
    }

    #[test]
    fn test_cancellation_token_basic() {
        let token = CancellationToken::new();

        assert!(!token.is_cancelled());

        token.cancel();

        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_clone() {
        let token = CancellationToken::new();
        let token_clone = token.clone();

        assert!(!token.is_cancelled());
        assert!(!token_clone.is_cancelled());

        // Cancel through original
        token.cancel();

        // Both should see cancellation
        assert!(token.is_cancelled());
        assert!(token_clone.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_as_callback() {
        let token = CancellationToken::new();
        let callback = token.as_callback();

        assert!(!callback());

        token.cancel();

        assert!(callback());
    }

    #[test]
    fn test_cancellation_token_with_query_params() {
        let token = CancellationToken::new();
        let params = QueryParams::new().with_timeout_callback(token.as_callback());

        let start = Instant::now();

        assert!(!params.is_timed_out(start));

        token.cancel();

        assert!(params.is_timed_out(start));
    }

    #[test]
    fn test_cancellation_token_thread_safety() {
        use std::thread;

        let token = CancellationToken::new();
        let token_clone = token.clone();

        let handle = thread::spawn(move || {
            // Wait a bit then cancel
            std::thread::sleep(Duration::from_millis(10));
            token_clone.cancel();
        });

        // Poll until cancelled
        while !token.is_cancelled() {
            std::thread::sleep(Duration::from_millis(1));
        }

        handle.join().unwrap();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_default() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_query_params_combined_timeout_check() {
        // Test with both duration and callback
        let should_cancel = Arc::new(AtomicBool::new(false));
        let should_cancel_clone = Arc::clone(&should_cancel);

        let params = QueryParams::new()
            .with_timeout(Duration::from_millis(100))
            .with_timeout_callback(move || should_cancel_clone.load(Ordering::Acquire));

        let start = Instant::now();

        // Neither triggered
        assert!(!params.is_timed_out(start));

        // Trigger callback
        should_cancel.store(true, Ordering::Release);
        assert!(params.is_timed_out(start));
    }
}

//! Query parameter and result types.
//!
//! This module provides types for configuring queries and handling results:
//! - `QueryParams`: Configuration for query execution
//! - `QueryResult`: A single result (label + distance)
//! - `QueryReply`: Collection of query results
//! - `TimeoutChecker`: Efficient timeout checking during search
//! - `CancellationToken`: Thread-safe cancellation mechanism
//! - `hybrid`: Heuristics for choosing between ad-hoc and batch search

pub mod hybrid;
pub mod params;
pub mod results;

pub use hybrid::{
    determine_search_mode, prefer_adhoc_brute_force, prefer_adhoc_hnsw, prefer_adhoc_svs,
    HybridPolicy, HybridSearchParams, QueryResultOrder, SearchMode,
};
pub use params::{CancellationToken, QueryParams, TimeoutChecker};
pub use results::{QueryReply, QueryResult};

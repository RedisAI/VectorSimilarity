//! Query parameter and result types.
//!
//! This module provides types for configuring queries and handling results:
//! - `QueryParams`: Configuration for query execution
//! - `QueryResult`: A single result (label + distance)
//! - `QueryReply`: Collection of query results
//! - `TimeoutChecker`: Efficient timeout checking during search
//! - `CancellationToken`: Thread-safe cancellation mechanism

pub mod params;
pub mod results;

pub use params::{CancellationToken, QueryParams, TimeoutChecker};
pub use results::{QueryReply, QueryResult};

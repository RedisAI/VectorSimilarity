//! Query parameter and result types.
//!
//! This module provides types for configuring queries and handling results:
//! - `QueryParams`: Configuration for query execution
//! - `QueryResult`: A single result (label + distance)
//! - `QueryReply`: Collection of query results

pub mod params;
pub mod results;

pub use params::QueryParams;
pub use results::{QueryReply, QueryResult};

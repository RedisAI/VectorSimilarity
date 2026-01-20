//! Utility types and functions.
//!
//! This module provides utility data structures used throughout the library:
//! - Priority queues (max-heap and min-heap) for KNN search
//! - Memory prefetching utilities for cache optimization

pub mod heap;
pub mod prefetch;

pub use heap::{MaxHeap, MinHeap};

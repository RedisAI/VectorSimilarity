//! Utility types and functions.
//!
//! This module provides utility data structures used throughout the library:
//! - Priority queues (max-heap and min-heap) for KNN search

pub mod heap;

pub use heap::{MaxHeap, MinHeap};

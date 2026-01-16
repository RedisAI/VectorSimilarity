//! Container types for vector storage.
//!
//! This module provides efficient data structures for storing vectors:
//! - `DataBlocks`: Block-based storage with SIMD-aligned memory

pub mod data_blocks;

pub use data_blocks::DataBlocks;

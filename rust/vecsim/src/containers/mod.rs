//! Container types for vector storage.
//!
//! This module provides efficient data structures for storing vectors:
//! - `DataBlocks`: Block-based storage with SIMD-aligned memory
//! - `Sq8DataBlocks`: Block-based storage for SQ8-quantized vectors
//! - `MmapDataBlocks`: Memory-mapped storage for disk-based indices

pub mod data_blocks;
pub mod mmap_blocks;
pub mod sq8_blocks;

pub use data_blocks::DataBlocks;
pub use mmap_blocks::MmapDataBlocks;
pub use sq8_blocks::Sq8DataBlocks;

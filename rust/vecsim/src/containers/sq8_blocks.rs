//! Block-based storage for SQ8-quantized vectors.
//!
//! This module provides `Sq8DataBlocks`, a container optimized for storing
//! SQ8-quantized vectors with per-vector metadata for dequantization and
//! asymmetric distance computation.

use crate::quantization::{Sq8Codec, Sq8VectorMeta};
use crate::types::{IdType, INVALID_ID};
use std::collections::HashSet;

/// Default block size (number of vectors per block).
const DEFAULT_BLOCK_SIZE: usize = 1024;

/// A single block of SQ8-quantized vector data.
struct Sq8DataBlock {
    /// Quantized vector data (u8 values).
    data: Vec<u8>,
    /// Per-vector metadata.
    metadata: Vec<Sq8VectorMeta>,
    /// Number of vectors this block can hold.
    capacity: usize,
    /// Vector dimension.
    dim: usize,
}

impl Sq8DataBlock {
    /// Create a new SQ8 data block.
    fn new(num_vectors: usize, dim: usize) -> Self {
        Self {
            data: vec![0u8; num_vectors * dim],
            metadata: vec![Sq8VectorMeta::default(); num_vectors],
            capacity: num_vectors,
            dim,
        }
    }

    /// Check if an index is within bounds.
    #[inline]
    fn is_valid_index(&self, index: usize) -> bool {
        index < self.capacity
    }

    /// Get quantized data and metadata for a vector.
    #[inline]
    fn get_vector(&self, index: usize) -> Option<(&[u8], &Sq8VectorMeta)> {
        if !self.is_valid_index(index) {
            return None;
        }
        let start = index * self.dim;
        let end = start + self.dim;
        Some((&self.data[start..end], &self.metadata[index]))
    }

    /// Get raw pointer to quantized data (for SIMD operations).
    #[inline]
    fn get_data_ptr(&self, index: usize) -> Option<*const u8> {
        if !self.is_valid_index(index) {
            return None;
        }
        Some(unsafe { self.data.as_ptr().add(index * self.dim) })
    }

    /// Write quantized vector data and metadata.
    #[inline]
    fn write_vector(&mut self, index: usize, data: &[u8], meta: Sq8VectorMeta) -> bool {
        if !self.is_valid_index(index) || data.len() != self.dim {
            return false;
        }
        let start = index * self.dim;
        self.data[start..start + self.dim].copy_from_slice(data);
        self.metadata[index] = meta;
        true
    }
}

/// Block-based storage for SQ8-quantized vectors.
///
/// Stores vectors in quantized form with per-vector metadata for efficient
/// asymmetric distance computation.
pub struct Sq8DataBlocks {
    /// The blocks storing quantized vector data.
    blocks: Vec<Sq8DataBlock>,
    /// Number of vectors per block.
    vectors_per_block: usize,
    /// Vector dimension.
    dim: usize,
    /// Total number of vectors stored (excluding deleted).
    count: usize,
    /// Free slots from deleted vectors.
    free_slots: HashSet<IdType>,
    /// High water mark: highest ID ever allocated + 1.
    high_water_mark: usize,
    /// SQ8 codec for encoding/decoding.
    codec: Sq8Codec,
}

impl Sq8DataBlocks {
    /// Create a new Sq8DataBlocks container.
    ///
    /// # Arguments
    /// * `dim` - Vector dimension
    /// * `initial_capacity` - Initial number of vectors to allocate
    pub fn new(dim: usize, initial_capacity: usize) -> Self {
        let vectors_per_block = DEFAULT_BLOCK_SIZE;
        let num_blocks = initial_capacity.div_ceil(vectors_per_block);

        let blocks: Vec<_> = (0..num_blocks.max(1))
            .map(|_| Sq8DataBlock::new(vectors_per_block, dim))
            .collect();

        Self {
            blocks,
            vectors_per_block,
            dim,
            count: 0,
            free_slots: HashSet::new(),
            high_water_mark: 0,
            codec: Sq8Codec::new(dim),
        }
    }

    /// Create with a custom block size.
    pub fn with_block_size(dim: usize, initial_capacity: usize, block_size: usize) -> Self {
        let vectors_per_block = block_size;
        let num_blocks = initial_capacity.div_ceil(vectors_per_block);

        let blocks: Vec<_> = (0..num_blocks.max(1))
            .map(|_| Sq8DataBlock::new(vectors_per_block, dim))
            .collect();

        Self {
            blocks,
            vectors_per_block,
            dim,
            count: 0,
            free_slots: HashSet::new(),
            high_water_mark: 0,
            codec: Sq8Codec::new(dim),
        }
    }

    /// Get the vector dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Get the number of vectors stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the total capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.blocks.len() * self.vectors_per_block
    }

    /// Get the SQ8 codec.
    #[inline]
    pub fn codec(&self) -> &Sq8Codec {
        &self.codec
    }

    /// Convert an internal ID to block and offset indices.
    #[inline]
    fn id_to_indices(&self, id: IdType) -> (usize, usize) {
        let id = id as usize;
        (id / self.vectors_per_block, id % self.vectors_per_block)
    }

    /// Add an f32 vector (will be quantized) and return its internal ID.
    pub fn add(&mut self, vector: &[f32]) -> Option<IdType> {
        if vector.len() != self.dim {
            return None;
        }

        // Quantize the vector
        let (quantized, meta) = self.codec.encode_from_f32_slice(vector);

        // Find a slot
        let slot = if let Some(&id) = self.free_slots.iter().next() {
            self.free_slots.remove(&id);
            id
        } else {
            let next_slot = self.high_water_mark;
            if next_slot >= self.capacity() {
                self.blocks
                    .push(Sq8DataBlock::new(self.vectors_per_block, self.dim));
            }
            self.high_water_mark += 1;
            next_slot as IdType
        };

        let (block_idx, offset) = self.id_to_indices(slot);
        if self.blocks[block_idx].write_vector(offset, &quantized, meta) {
            self.count += 1;
            Some(slot)
        } else {
            // Write failed, restore state
            if slot as usize == self.high_water_mark - 1 {
                self.high_water_mark -= 1;
            } else {
                self.free_slots.insert(slot);
            }
            None
        }
    }

    /// Add a pre-quantized vector with metadata.
    pub fn add_quantized(&mut self, quantized: &[u8], meta: Sq8VectorMeta) -> Option<IdType> {
        if quantized.len() != self.dim {
            return None;
        }

        // Find a slot
        let slot = if let Some(&id) = self.free_slots.iter().next() {
            self.free_slots.remove(&id);
            id
        } else {
            let next_slot = self.high_water_mark;
            if next_slot >= self.capacity() {
                self.blocks
                    .push(Sq8DataBlock::new(self.vectors_per_block, self.dim));
            }
            self.high_water_mark += 1;
            next_slot as IdType
        };

        let (block_idx, offset) = self.id_to_indices(slot);
        if self.blocks[block_idx].write_vector(offset, quantized, meta) {
            self.count += 1;
            Some(slot)
        } else {
            if slot as usize == self.high_water_mark - 1 {
                self.high_water_mark -= 1;
            } else {
                self.free_slots.insert(slot);
            }
            None
        }
    }

    /// Check if an ID is valid and not deleted.
    #[inline]
    pub fn is_valid(&self, id: IdType) -> bool {
        if id == INVALID_ID {
            return false;
        }
        let id_usize = id as usize;
        id_usize < self.high_water_mark && !self.free_slots.contains(&id)
    }

    /// Get quantized data and metadata by ID.
    #[inline]
    pub fn get(&self, id: IdType) -> Option<(&[u8], &Sq8VectorMeta)> {
        if !self.is_valid(id) {
            return None;
        }
        let (block_idx, offset) = self.id_to_indices(id);
        if block_idx >= self.blocks.len() {
            return None;
        }
        self.blocks[block_idx].get_vector(offset)
    }

    /// Get raw pointer to quantized data.
    #[inline]
    pub fn get_data_ptr(&self, id: IdType) -> Option<*const u8> {
        if !self.is_valid(id) {
            return None;
        }
        let (block_idx, offset) = self.id_to_indices(id);
        if block_idx >= self.blocks.len() {
            return None;
        }
        self.blocks[block_idx].get_data_ptr(offset)
    }

    /// Get metadata only (for precomputed values).
    #[inline]
    pub fn get_metadata(&self, id: IdType) -> Option<&Sq8VectorMeta> {
        self.get(id).map(|(_, meta)| meta)
    }

    /// Decode a vector back to f32.
    pub fn decode(&self, id: IdType) -> Option<Vec<f32>> {
        let (quantized, meta) = self.get(id)?;
        Some(self.codec.decode_from_u8_slice(quantized, meta))
    }

    /// Mark a slot as deleted.
    pub fn mark_deleted(&mut self, id: IdType) -> bool {
        if id == INVALID_ID {
            return false;
        }
        let id_usize = id as usize;
        if id_usize >= self.high_water_mark || self.free_slots.contains(&id) {
            return false;
        }
        self.free_slots.insert(id);
        self.count = self.count.saturating_sub(1);
        true
    }

    /// Update a vector at the given ID.
    pub fn update(&mut self, id: IdType, vector: &[f32]) -> bool {
        if vector.len() != self.dim || !self.is_valid(id) {
            return false;
        }

        let (quantized, meta) = self.codec.encode_from_f32_slice(vector);
        let (block_idx, offset) = self.id_to_indices(id);

        if block_idx >= self.blocks.len() {
            return false;
        }

        self.blocks[block_idx].write_vector(offset, &quantized, meta)
    }

    /// Clear all vectors.
    pub fn clear(&mut self) {
        self.count = 0;
        self.free_slots.clear();
        self.high_water_mark = 0;
    }

    /// Reserve space for additional vectors.
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.count + additional;
        let current_capacity = self.capacity();

        if needed > current_capacity {
            let additional_blocks = (needed - current_capacity).div_ceil(self.vectors_per_block);
            for _ in 0..additional_blocks {
                self.blocks
                    .push(Sq8DataBlock::new(self.vectors_per_block, self.dim));
            }
        }
    }

    /// Iterate over all valid vector IDs.
    pub fn iter_ids(&self) -> impl Iterator<Item = IdType> + '_ {
        (0..self.high_water_mark as IdType).filter(move |&id| !self.free_slots.contains(&id))
    }

    /// Get the number of deleted slots.
    #[inline]
    pub fn deleted_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Get the fragmentation ratio.
    #[inline]
    pub fn fragmentation(&self) -> f64 {
        if self.high_water_mark == 0 {
            0.0
        } else {
            self.free_slots.len() as f64 / self.high_water_mark as f64
        }
    }

    /// Memory usage in bytes (approximate).
    pub fn memory_usage(&self) -> usize {
        let data_size = self.blocks.len() * self.vectors_per_block * self.dim;
        let meta_size = self.blocks.len()
            * self.vectors_per_block
            * std::mem::size_of::<Sq8VectorMeta>();
        let overhead = self.free_slots.len() * std::mem::size_of::<IdType>();
        data_size + meta_size + overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_blocks_basic() {
        let mut blocks = Sq8DataBlocks::new(4, 10);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![5.0, 6.0, 7.0, 8.0];

        let id1 = blocks.add(&v1).unwrap();
        let id2 = blocks.add(&v2).unwrap();

        assert_eq!(blocks.len(), 2);
        assert!(blocks.is_valid(id1));
        assert!(blocks.is_valid(id2));

        // Decode and verify
        let decoded1 = blocks.decode(id1).unwrap();
        let decoded2 = blocks.decode(id2).unwrap();

        for (orig, dec) in v1.iter().zip(decoded1.iter()) {
            assert!((orig - dec).abs() < 0.1);
        }
        for (orig, dec) in v2.iter().zip(decoded2.iter()) {
            assert!((orig - dec).abs() < 0.1);
        }
    }

    #[test]
    fn test_sq8_blocks_delete_reuse() {
        let mut blocks = Sq8DataBlocks::new(4, 10);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![5.0, 6.0, 7.0, 8.0];
        let v3 = vec![9.0, 10.0, 11.0, 12.0];

        let id1 = blocks.add(&v1).unwrap();
        let _id2 = blocks.add(&v2).unwrap();

        // Delete first
        assert!(blocks.mark_deleted(id1));
        assert_eq!(blocks.len(), 1);
        assert!(!blocks.is_valid(id1));

        // Add new - should reuse slot
        let id3 = blocks.add(&v3).unwrap();
        assert_eq!(id3, id1);
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_sq8_blocks_metadata() {
        let mut blocks = Sq8DataBlocks::new(4, 10);

        let v = vec![0.0, 0.5, 1.0, 0.25];
        let id = blocks.add(&v).unwrap();

        let meta = blocks.get_metadata(id).unwrap();
        assert_eq!(meta.min, 0.0);
        assert!((meta.delta - (1.0 / 255.0)).abs() < 0.0001);
    }

    #[test]
    fn test_sq8_blocks_update() {
        let mut blocks = Sq8DataBlocks::new(4, 10);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![10.0, 20.0, 30.0, 40.0];

        let id = blocks.add(&v1).unwrap();
        assert!(blocks.update(id, &v2));

        let decoded = blocks.decode(id).unwrap();
        for (orig, dec) in v2.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.5);
        }
    }

    #[test]
    fn test_sq8_blocks_memory() {
        let blocks = Sq8DataBlocks::new(128, 1000);
        let mem = blocks.memory_usage();

        // Should be approximately: 1 block * 1024 vectors * (128 bytes + 16 bytes meta)
        assert!(mem > 0);
    }
}

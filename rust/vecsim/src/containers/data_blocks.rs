//! Block-based vector storage with SIMD-aligned memory.
//!
//! This module provides `DataBlocks`, a container optimized for storing
//! vectors in contiguous, cache-friendly blocks with proper SIMD alignment.

use crate::distance::simd::optimal_alignment;
use crate::types::{IdType, VectorElement, INVALID_ID};
use std::alloc::{self, Layout};
use std::ptr::NonNull;

/// Default block size (number of vectors per block).
const DEFAULT_BLOCK_SIZE: usize = 1024;

/// A single block of vector data with aligned memory.
struct DataBlock<T: VectorElement> {
    /// Pointer to aligned memory.
    data: NonNull<T>,
    /// Layout used for allocation.
    layout: Layout,
    /// Number of elements (not vectors) in this block.
    capacity: usize,
}

impl<T: VectorElement> DataBlock<T> {
    /// Create a new data block with capacity for `num_vectors` vectors of `dim` elements.
    fn new(num_vectors: usize, dim: usize) -> Self {
        let num_elements = num_vectors * dim;
        let alignment = optimal_alignment().max(std::mem::align_of::<T>());
        let size = num_elements * std::mem::size_of::<T>();

        let layout = Layout::from_size_align(size.max(1), alignment)
            .expect("Invalid layout for DataBlock");

        let data = if size == 0 {
            NonNull::dangling()
        } else {
            let ptr = unsafe { alloc::alloc(layout) };
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            NonNull::new(ptr as *mut T).expect("Allocation returned null")
        };

        Self {
            data,
            layout,
            capacity: num_elements,
        }
    }

    /// Get a pointer to the vector at the given index.
    #[inline]
    fn get_vector_ptr(&self, index: usize, dim: usize) -> *const T {
        debug_assert!(index * dim < self.capacity);
        unsafe { self.data.as_ptr().add(index * dim) }
    }

    /// Get a mutable pointer to the vector at the given index.
    #[inline]
    fn get_vector_ptr_mut(&mut self, index: usize, dim: usize) -> *mut T {
        debug_assert!(index * dim < self.capacity);
        unsafe { self.data.as_ptr().add(index * dim) }
    }

    /// Get a slice to the vector at the given index.
    #[inline]
    fn get_vector(&self, index: usize, dim: usize) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.get_vector_ptr(index, dim), dim) }
    }

    /// Write a vector at the given index.
    #[inline]
    fn write_vector(&mut self, index: usize, dim: usize, data: &[T]) {
        debug_assert_eq!(data.len(), dim);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.get_vector_ptr_mut(index, dim), dim);
        }
    }
}

impl<T: VectorElement> Drop for DataBlock<T> {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe {
                alloc::dealloc(self.data.as_ptr() as *mut u8, self.layout);
            }
        }
    }
}

// Safety: DataBlock contains raw pointers but they are owned and not shared.
unsafe impl<T: VectorElement> Send for DataBlock<T> {}
unsafe impl<T: VectorElement> Sync for DataBlock<T> {}

/// Block-based storage for vectors with SIMD-aligned memory.
///
/// Vectors are stored in contiguous blocks for cache efficiency.
/// Each vector is accessed by its internal ID.
pub struct DataBlocks<T: VectorElement> {
    /// The blocks storing vector data.
    blocks: Vec<DataBlock<T>>,
    /// Number of vectors per block.
    vectors_per_block: usize,
    /// Vector dimension.
    dim: usize,
    /// Total number of vectors stored.
    count: usize,
    /// Free slots from deleted vectors (for reuse).
    free_slots: Vec<IdType>,
}

impl<T: VectorElement> DataBlocks<T> {
    /// Create a new DataBlocks container.
    ///
    /// # Arguments
    /// * `dim` - Vector dimension
    /// * `initial_capacity` - Initial number of vectors to allocate
    pub fn new(dim: usize, initial_capacity: usize) -> Self {
        let vectors_per_block = DEFAULT_BLOCK_SIZE;
        let num_blocks = (initial_capacity + vectors_per_block - 1) / vectors_per_block;

        let blocks: Vec<_> = (0..num_blocks.max(1))
            .map(|_| DataBlock::new(vectors_per_block, dim))
            .collect();

        Self {
            blocks,
            vectors_per_block,
            dim,
            count: 0,
            free_slots: Vec::new(),
        }
    }

    /// Create with a custom block size.
    pub fn with_block_size(dim: usize, initial_capacity: usize, block_size: usize) -> Self {
        let vectors_per_block = block_size;
        let num_blocks = (initial_capacity + vectors_per_block - 1) / vectors_per_block;

        let blocks: Vec<_> = (0..num_blocks.max(1))
            .map(|_| DataBlock::new(vectors_per_block, dim))
            .collect();

        Self {
            blocks,
            vectors_per_block,
            dim,
            count: 0,
            free_slots: Vec::new(),
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

    /// Get the total capacity (number of vector slots).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.blocks.len() * self.vectors_per_block
    }

    /// Convert an internal ID to block and offset indices.
    #[inline]
    fn id_to_indices(&self, id: IdType) -> (usize, usize) {
        let id = id as usize;
        (id / self.vectors_per_block, id % self.vectors_per_block)
    }

    /// Convert block and offset indices to an internal ID.
    #[inline]
    #[allow(dead_code)]
    fn indices_to_id(&self, block: usize, offset: usize) -> IdType {
        (block * self.vectors_per_block + offset) as IdType
    }

    /// Add a vector and return its internal ID.
    pub fn add(&mut self, vector: &[T]) -> IdType {
        debug_assert_eq!(vector.len(), self.dim);

        // Try to reuse a free slot first
        if let Some(id) = self.free_slots.pop() {
            let (block_idx, offset) = self.id_to_indices(id);
            self.blocks[block_idx].write_vector(offset, self.dim, vector);
            self.count += 1;
            return id;
        }

        // Find the next available slot
        let total_slots = self.blocks.len() * self.vectors_per_block;
        let next_slot = self.count;

        if next_slot >= total_slots {
            // Need to allocate a new block
            self.blocks
                .push(DataBlock::new(self.vectors_per_block, self.dim));
        }

        let (block_idx, offset) = self.id_to_indices(next_slot as IdType);
        self.blocks[block_idx].write_vector(offset, self.dim, vector);
        self.count += 1;

        next_slot as IdType
    }

    /// Get a vector by its internal ID.
    #[inline]
    pub fn get(&self, id: IdType) -> Option<&[T]> {
        if id == INVALID_ID {
            return None;
        }
        let (block_idx, offset) = self.id_to_indices(id);
        if block_idx >= self.blocks.len() {
            return None;
        }
        Some(self.blocks[block_idx].get_vector(offset, self.dim))
    }

    /// Get a raw pointer to a vector (for SIMD operations).
    #[inline]
    pub fn get_ptr(&self, id: IdType) -> *const T {
        let (block_idx, offset) = self.id_to_indices(id);
        self.blocks[block_idx].get_vector_ptr(offset, self.dim)
    }

    /// Mark a slot as free for reuse.
    ///
    /// Note: This doesn't actually clear the data, just marks the slot as available.
    pub fn mark_deleted(&mut self, id: IdType) {
        if id != INVALID_ID && (id as usize) < self.capacity() {
            self.free_slots.push(id);
            self.count = self.count.saturating_sub(1);
        }
    }

    /// Update a vector at the given ID.
    pub fn update(&mut self, id: IdType, vector: &[T]) {
        debug_assert_eq!(vector.len(), self.dim);
        let (block_idx, offset) = self.id_to_indices(id);
        self.blocks[block_idx].write_vector(offset, self.dim, vector);
    }

    /// Clear all vectors, resetting to empty state.
    ///
    /// This keeps the allocated blocks but marks them as empty.
    pub fn clear(&mut self) {
        self.count = 0;
        self.free_slots.clear();
    }

    /// Reserve space for additional vectors.
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.count + additional;
        let current_capacity = self.capacity();

        if needed > current_capacity {
            let additional_blocks =
                (needed - current_capacity + self.vectors_per_block - 1) / self.vectors_per_block;

            for _ in 0..additional_blocks {
                self.blocks
                    .push(DataBlock::new(self.vectors_per_block, self.dim));
            }
        }
    }

    /// Iterate over all valid vector IDs.
    ///
    /// Note: This iterates over all slots, not just active vectors.
    /// Use with the label mapping to get only active vectors.
    pub fn iter_ids(&self) -> impl Iterator<Item = IdType> + '_ {
        (0..self.capacity() as IdType)
            .filter(move |&id| !self.free_slots.contains(&id) || id as usize >= self.count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_blocks_basic() {
        let mut blocks = DataBlocks::<f32>::new(4, 10);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![5.0, 6.0, 7.0, 8.0];

        let id1 = blocks.add(&v1);
        let id2 = blocks.add(&v2);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks.get(id1), Some(v1.as_slice()));
        assert_eq!(blocks.get(id2), Some(v2.as_slice()));
    }

    #[test]
    fn test_data_blocks_reuse() {
        let mut blocks = DataBlocks::<f32>::new(4, 10);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![5.0, 6.0, 7.0, 8.0];
        let v3 = vec![9.0, 10.0, 11.0, 12.0];

        let id1 = blocks.add(&v1);
        let _id2 = blocks.add(&v2);
        assert_eq!(blocks.len(), 2);

        // Delete first vector
        blocks.mark_deleted(id1);
        assert_eq!(blocks.len(), 1);

        // Add new vector - should reuse slot
        let id3 = blocks.add(&v3);
        assert_eq!(id3, id1); // Reused the same slot
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks.get(id3), Some(v3.as_slice()));
    }

    #[test]
    fn test_data_blocks_grow() {
        let mut blocks = DataBlocks::<f32>::with_block_size(4, 2, 2);

        // Fill initial capacity
        for i in 0..2 {
            blocks.add(&vec![i as f32; 4]);
        }
        assert_eq!(blocks.len(), 2);

        // Should trigger new block allocation
        blocks.add(&vec![99.0; 4]);
        assert_eq!(blocks.len(), 3);
        assert!(blocks.capacity() >= 3);
    }
}

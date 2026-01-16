//! Block-based vector storage with SIMD-aligned memory.
//!
//! This module provides `DataBlocks`, a container optimized for storing
//! vectors in contiguous, cache-friendly blocks with proper SIMD alignment.

use crate::distance::simd::optimal_alignment;
use crate::types::{IdType, VectorElement, INVALID_ID};
use std::alloc::{self, Layout};
use std::collections::HashSet;
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

    /// Check if an index is within bounds.
    #[inline]
    fn is_valid_index(&self, index: usize, dim: usize) -> bool {
        index * dim + dim <= self.capacity
    }

    /// Get a pointer to the vector at the given index.
    ///
    /// # Safety
    /// Caller must ensure the index is valid (use `is_valid_index` first).
    #[inline]
    unsafe fn get_vector_ptr_unchecked(&self, index: usize, dim: usize) -> *const T {
        self.data.as_ptr().add(index * dim)
    }

    /// Get a mutable pointer to the vector at the given index.
    ///
    /// # Safety
    /// Caller must ensure the index is valid (use `is_valid_index` first).
    #[inline]
    unsafe fn get_vector_ptr_mut_unchecked(&mut self, index: usize, dim: usize) -> *mut T {
        self.data.as_ptr().add(index * dim)
    }

    /// Get a slice to the vector at the given index.
    ///
    /// Returns `None` if the index is out of bounds.
    #[inline]
    fn get_vector(&self, index: usize, dim: usize) -> Option<&[T]> {
        if !self.is_valid_index(index, dim) {
            return None;
        }
        // SAFETY: We just verified the index is valid.
        unsafe {
            Some(std::slice::from_raw_parts(
                self.get_vector_ptr_unchecked(index, dim),
                dim,
            ))
        }
    }

    /// Write a vector at the given index.
    ///
    /// Returns `false` if the index is out of bounds or data length doesn't match dim.
    #[inline]
    fn write_vector(&mut self, index: usize, dim: usize, data: &[T]) -> bool {
        if data.len() != dim || !self.is_valid_index(index, dim) {
            return false;
        }
        // SAFETY: We just verified the index is valid and data length matches.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.get_vector_ptr_mut_unchecked(index, dim),
                dim,
            );
        }
        true
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
    /// Total number of vectors stored (excluding deleted).
    count: usize,
    /// Free slots from deleted vectors (for reuse). Uses HashSet for O(1) lookup.
    free_slots: HashSet<IdType>,
    /// High water mark: the highest ID ever allocated + 1.
    /// Used to determine which slots are valid vs never-allocated.
    high_water_mark: usize,
}

impl<T: VectorElement> DataBlocks<T> {
    /// Create a new DataBlocks container.
    ///
    /// # Arguments
    /// * `dim` - Vector dimension
    /// * `initial_capacity` - Initial number of vectors to allocate
    pub fn new(dim: usize, initial_capacity: usize) -> Self {
        let vectors_per_block = DEFAULT_BLOCK_SIZE;
        let num_blocks = initial_capacity.div_ceil(vectors_per_block);

        let blocks: Vec<_> = (0..num_blocks.max(1))
            .map(|_| DataBlock::new(vectors_per_block, dim))
            .collect();

        Self {
            blocks,
            vectors_per_block,
            dim,
            count: 0,
            free_slots: HashSet::new(),
            high_water_mark: 0,
        }
    }

    /// Create with a custom block size.
    pub fn with_block_size(dim: usize, initial_capacity: usize, block_size: usize) -> Self {
        let vectors_per_block = block_size;
        let num_blocks = initial_capacity.div_ceil(vectors_per_block);

        let blocks: Vec<_> = (0..num_blocks.max(1))
            .map(|_| DataBlock::new(vectors_per_block, dim))
            .collect();

        Self {
            blocks,
            vectors_per_block,
            dim,
            count: 0,
            free_slots: HashSet::new(),
            high_water_mark: 0,
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
    ///
    /// Returns `None` if the vector dimension doesn't match the container's dimension.
    pub fn add(&mut self, vector: &[T]) -> Option<IdType> {
        if vector.len() != self.dim {
            return None;
        }

        // Try to reuse a free slot first
        if let Some(&id) = self.free_slots.iter().next() {
            self.free_slots.remove(&id);
            let (block_idx, offset) = self.id_to_indices(id);
            if self.blocks[block_idx].write_vector(offset, self.dim, vector) {
                self.count += 1;
                return Some(id);
            }
            // Write failed (shouldn't happen), put the slot back
            self.free_slots.insert(id);
            return None;
        }

        // Find the next available slot using high water mark
        let next_slot = self.high_water_mark;
        let total_slots = self.blocks.len() * self.vectors_per_block;

        if next_slot >= total_slots {
            // Need to allocate a new block
            self.blocks
                .push(DataBlock::new(self.vectors_per_block, self.dim));
        }

        let (block_idx, offset) = self.id_to_indices(next_slot as IdType);
        if self.blocks[block_idx].write_vector(offset, self.dim, vector) {
            self.count += 1;
            self.high_water_mark += 1;
            Some(next_slot as IdType)
        } else {
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
        // Must be within allocated range and not deleted
        id_usize < self.high_water_mark && !self.free_slots.contains(&id)
    }

    /// Get a vector by its internal ID.
    ///
    /// Returns `None` if the ID is invalid, out of bounds, or the vector was deleted.
    #[inline]
    pub fn get(&self, id: IdType) -> Option<&[T]> {
        if !self.is_valid(id) {
            return None;
        }
        let (block_idx, offset) = self.id_to_indices(id);
        if block_idx >= self.blocks.len() {
            return None;
        }
        self.blocks[block_idx].get_vector(offset, self.dim)
    }

    /// Get a raw pointer to a vector (for SIMD operations).
    ///
    /// Returns `None` if the ID is invalid, out of bounds, or the vector was deleted.
    #[inline]
    pub fn get_ptr(&self, id: IdType) -> Option<*const T> {
        if !self.is_valid(id) {
            return None;
        }
        let (block_idx, offset) = self.id_to_indices(id);
        if block_idx >= self.blocks.len() {
            return None;
        }
        let block = &self.blocks[block_idx];
        if !block.is_valid_index(offset, self.dim) {
            return None;
        }
        // SAFETY: We verified the index is valid above.
        unsafe { Some(block.get_vector_ptr_unchecked(offset, self.dim)) }
    }

    /// Mark a slot as free for reuse.
    ///
    /// Returns `true` if the slot was successfully marked as deleted,
    /// `false` if the ID is invalid, already deleted, or out of bounds.
    ///
    /// Note: This doesn't actually clear the data, just marks the slot as available.
    pub fn mark_deleted(&mut self, id: IdType) -> bool {
        if id == INVALID_ID {
            return false;
        }
        let id_usize = id as usize;
        // Check bounds and ensure not already deleted
        if id_usize >= self.high_water_mark || self.free_slots.contains(&id) {
            return false;
        }
        self.free_slots.insert(id);
        self.count = self.count.saturating_sub(1);
        true
    }

    /// Update a vector at the given ID.
    ///
    /// Returns `true` if the update was successful, `false` if the ID is invalid,
    /// deleted, out of bounds, or the vector dimension doesn't match.
    pub fn update(&mut self, id: IdType, vector: &[T]) -> bool {
        if vector.len() != self.dim || !self.is_valid(id) {
            return false;
        }
        let (block_idx, offset) = self.id_to_indices(id);
        if block_idx >= self.blocks.len() {
            return false;
        }
        self.blocks[block_idx].write_vector(offset, self.dim, vector)
    }

    /// Clear all vectors, resetting to empty state.
    ///
    /// This keeps the allocated blocks but marks them as empty.
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
            let additional_blocks =
                (needed - current_capacity).div_ceil(self.vectors_per_block);

            for _ in 0..additional_blocks {
                self.blocks
                    .push(DataBlock::new(self.vectors_per_block, self.dim));
            }
        }
    }

    /// Iterate over all valid (non-deleted) vector IDs.
    pub fn iter_ids(&self) -> impl Iterator<Item = IdType> + '_ {
        (0..self.high_water_mark as IdType).filter(move |&id| !self.free_slots.contains(&id))
    }

    /// Compact the storage by removing gaps from deleted vectors.
    ///
    /// Returns a mapping from old IDs to new IDs. Vectors that were deleted
    /// will not appear in the mapping.
    ///
    /// After compaction:
    /// - All vectors are contiguous starting from ID 0
    /// - `free_slots` is empty
    /// - `high_water_mark` equals `count`
    /// - Unused blocks may be deallocated if `shrink` is true
    ///
    /// # Arguments
    /// * `shrink` - If true, deallocate unused blocks after compaction
    pub fn compact(&mut self, shrink: bool) -> std::collections::HashMap<IdType, IdType> {
        use std::collections::HashMap;

        if self.free_slots.is_empty() {
            // No gaps to fill, just return identity mapping
            return (0..self.high_water_mark as IdType)
                .map(|id| (id, id))
                .collect();
        }

        let mut id_mapping = HashMap::with_capacity(self.count);
        let mut new_id: IdType = 0;

        // Collect valid vectors and their data
        let valid_ids: Vec<IdType> = (0..self.high_water_mark as IdType)
            .filter(|id| !self.free_slots.contains(id))
            .collect();

        // Copy vectors to temporary storage
        let vectors: Vec<Vec<T>> = valid_ids
            .iter()
            .filter_map(|&id| self.get(id).map(|v| v.to_vec()))
            .collect();

        // Clear and rebuild
        self.free_slots.clear();
        self.high_water_mark = 0;
        self.count = 0;

        // Re-add vectors in order
        for (old_id, vector) in valid_ids.into_iter().zip(vectors.into_iter()) {
            if let Some(added_id) = self.add(&vector) {
                id_mapping.insert(old_id, added_id);
                new_id = added_id + 1;
            }
        }

        // Shrink blocks if requested
        if shrink {
            let needed_blocks = new_id as usize / self.vectors_per_block + 1;
            if self.blocks.len() > needed_blocks {
                self.blocks.truncate(needed_blocks);
            }
        }

        id_mapping
    }

    /// Get the number of deleted (free) slots.
    #[inline]
    pub fn deleted_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Get the fragmentation ratio (deleted / total allocated).
    ///
    /// Returns 0.0 if no vectors have been allocated.
    #[inline]
    pub fn fragmentation(&self) -> f64 {
        if self.high_water_mark == 0 {
            0.0
        } else {
            self.free_slots.len() as f64 / self.high_water_mark as f64
        }
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

        let id1 = blocks.add(&v1).unwrap();
        let id2 = blocks.add(&v2).unwrap();

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

        let id1 = blocks.add(&v1).unwrap();
        let _id2 = blocks.add(&v2).unwrap();
        assert_eq!(blocks.len(), 2);

        // Delete first vector
        assert!(blocks.mark_deleted(id1));
        assert_eq!(blocks.len(), 1);

        // Verify deleted vector is not accessible
        assert!(blocks.get(id1).is_none());

        // Add new vector - should reuse slot
        let id3 = blocks.add(&v3).unwrap();
        assert_eq!(id3, id1); // Reused the same slot
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks.get(id3), Some(v3.as_slice()));
    }

    #[test]
    fn test_data_blocks_grow() {
        let mut blocks = DataBlocks::<f32>::with_block_size(4, 2, 2);

        // Fill initial capacity
        for i in 0..2 {
            blocks.add(&vec![i as f32; 4]).unwrap();
        }
        assert_eq!(blocks.len(), 2);

        // Should trigger new block allocation
        blocks.add(&vec![99.0; 4]).unwrap();
        assert_eq!(blocks.len(), 3);
        assert!(blocks.capacity() >= 3);
    }

    #[test]
    fn test_data_blocks_double_delete() {
        let mut blocks = DataBlocks::<f32>::new(4, 10);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let id1 = blocks.add(&v1).unwrap();

        // First delete should succeed
        assert!(blocks.mark_deleted(id1));
        assert_eq!(blocks.len(), 0);

        // Second delete should fail (already deleted)
        assert!(!blocks.mark_deleted(id1));
        assert_eq!(blocks.len(), 0);
    }

    #[test]
    fn test_data_blocks_invalid_dimension() {
        let mut blocks = DataBlocks::<f32>::new(4, 10);

        // Wrong dimension should fail
        let wrong_dim = vec![1.0, 2.0, 3.0]; // 3 instead of 4
        assert!(blocks.add(&wrong_dim).is_none());

        // Correct dimension should succeed
        let correct_dim = vec![1.0, 2.0, 3.0, 4.0];
        assert!(blocks.add(&correct_dim).is_some());
    }

    #[test]
    fn test_data_blocks_bounds_checking() {
        let blocks = DataBlocks::<f32>::new(4, 10);

        // Invalid ID should return None
        assert!(blocks.get(INVALID_ID).is_none());
        assert!(blocks.get_ptr(INVALID_ID).is_none());

        // Out of bounds ID should return None
        assert!(blocks.get(999).is_none());
        assert!(blocks.get_ptr(999).is_none());
    }

    #[test]
    fn test_data_blocks_update() {
        let mut blocks = DataBlocks::<f32>::new(4, 10);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![5.0, 6.0, 7.0, 8.0];
        let id1 = blocks.add(&v1).unwrap();

        // Update should succeed
        assert!(blocks.update(id1, &v2));
        assert_eq!(blocks.get(id1), Some(v2.as_slice()));

        // Update with wrong dimension should fail
        let wrong_dim = vec![1.0, 2.0, 3.0];
        assert!(!blocks.update(id1, &wrong_dim));

        // Update deleted vector should fail
        blocks.mark_deleted(id1);
        assert!(!blocks.update(id1, &v1));
    }

    #[test]
    fn test_data_blocks_compact() {
        let mut blocks = DataBlocks::<f32>::new(4, 10);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![5.0, 6.0, 7.0, 8.0];
        let v3 = vec![9.0, 10.0, 11.0, 12.0];
        let v4 = vec![13.0, 14.0, 15.0, 16.0];

        let id1 = blocks.add(&v1).unwrap();
        let id2 = blocks.add(&v2).unwrap();
        let id3 = blocks.add(&v3).unwrap();
        let id4 = blocks.add(&v4).unwrap();

        // Delete vectors 1 and 3 (creating gaps)
        blocks.mark_deleted(id2);
        blocks.mark_deleted(id3);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks.deleted_count(), 2);
        assert!((blocks.fragmentation() - 0.5).abs() < 0.01);

        // Compact
        let mapping = blocks.compact(false);

        // Should have 2 vectors now, contiguous
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks.deleted_count(), 0);
        assert!((blocks.fragmentation() - 0.0).abs() < 0.01);

        // Check mapping
        assert!(mapping.contains_key(&id1));
        assert!(mapping.contains_key(&id4));
        assert!(!mapping.contains_key(&id2)); // Deleted
        assert!(!mapping.contains_key(&id3)); // Deleted

        // Verify data integrity
        let new_id1 = mapping[&id1];
        let new_id4 = mapping[&id4];
        assert_eq!(blocks.get(new_id1), Some(v1.as_slice()));
        assert_eq!(blocks.get(new_id4), Some(v4.as_slice()));
    }

    #[test]
    fn test_data_blocks_compact_no_deletions() {
        let mut blocks = DataBlocks::<f32>::new(4, 10);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![5.0, 6.0, 7.0, 8.0];

        let id1 = blocks.add(&v1).unwrap();
        let id2 = blocks.add(&v2).unwrap();

        // Compact with no deletions should return identity mapping
        let mapping = blocks.compact(false);

        assert_eq!(mapping[&id1], id1);
        assert_eq!(mapping[&id2], id2);
        assert_eq!(blocks.len(), 2);
    }
}

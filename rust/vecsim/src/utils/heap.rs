//! Priority queue implementations for KNN search.
//!
//! This module provides specialized heap implementations optimized for
//! vector similarity search operations:
//! - `MaxHeap`: Used to maintain top-k results (evict largest distance)
//! - `MinHeap`: Used for candidate exploration (process smallest distance first)

use crate::types::{DistanceType, IdType};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// An entry in the priority queue containing an ID and distance.
#[derive(Debug, Clone, Copy)]
pub struct HeapEntry<D: DistanceType> {
    pub id: IdType,
    pub distance: D,
}

impl<D: DistanceType> HeapEntry<D> {
    #[inline]
    pub fn new(id: IdType, distance: D) -> Self {
        Self { id, distance }
    }
}

/// Wrapper for max-heap ordering (largest distance at top).
#[derive(Debug, Clone, Copy)]
struct MaxHeapEntry<D: DistanceType>(HeapEntry<D>);

impl<D: DistanceType> PartialEq for MaxHeapEntry<D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance.partial_cmp(&other.0.distance) == Some(Ordering::Equal)
    }
}

impl<D: DistanceType> Eq for MaxHeapEntry<D> {}

impl<D: DistanceType> PartialOrd for MaxHeapEntry<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D: DistanceType> Ord for MaxHeapEntry<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural ordering for max-heap: larger distances come first
        self.0
            .distance
            .partial_cmp(&other.0.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Wrapper for min-heap ordering (smallest distance at top).
#[derive(Debug, Clone, Copy)]
struct MinHeapEntry<D: DistanceType>(HeapEntry<D>);

impl<D: DistanceType> PartialEq for MinHeapEntry<D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance.partial_cmp(&other.0.distance) == Some(Ordering::Equal)
    }
}

impl<D: DistanceType> Eq for MinHeapEntry<D> {}

impl<D: DistanceType> PartialOrd for MinHeapEntry<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D: DistanceType> Ord for MinHeapEntry<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap: smaller distances come first
        other
            .0
            .distance
            .partial_cmp(&self.0.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// A max-heap that keeps track of the k smallest elements by evicting the largest.
///
/// This is used to maintain the current top-k results during KNN search.
/// The largest distance is at the top, making it efficient to check if a new
/// candidate should replace an existing result.
#[derive(Debug)]
pub struct MaxHeap<D: DistanceType> {
    heap: BinaryHeap<MaxHeapEntry<D>>,
    capacity: usize,
}

impl<D: DistanceType> MaxHeap<D> {
    /// Create a new max-heap with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity + 1),
            capacity,
        }
    }

    /// Get the number of elements in the heap.
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if the heap is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Check if the heap is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.capacity
    }

    /// Get the largest distance in the heap (top element).
    #[inline]
    pub fn top_distance(&self) -> Option<D> {
        self.heap.peek().map(|e| e.0.distance)
    }

    /// Get the top entry without removing it.
    #[inline]
    pub fn peek(&self) -> Option<HeapEntry<D>> {
        self.heap.peek().map(|e| e.0)
    }

    /// Try to insert an element. Returns true if inserted.
    ///
    /// If the heap is not full, always inserts.
    /// If full, only inserts if the distance is smaller than the current maximum.
    #[inline]
    pub fn try_insert(&mut self, id: IdType, distance: D) -> bool {
        if self.heap.len() < self.capacity {
            self.heap.push(MaxHeapEntry(HeapEntry::new(id, distance)));
            true
        } else if let Some(mut top) = self.heap.peek_mut() {
            if distance < top.0.distance {
                // Replace in-place using PeekMut - avoids separate pop+push operations
                // PeekMut::pop() removes the top element efficiently, then we push the new one
                // This is more efficient than pop() + push() because it avoids redundant sift operations
                *top = MaxHeapEntry(HeapEntry::new(id, distance));
                // Drop the PeekMut to trigger sift_down
                drop(top);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Insert an element unconditionally, maintaining capacity.
    #[inline]
    pub fn insert(&mut self, id: IdType, distance: D) {
        self.heap.push(MaxHeapEntry(HeapEntry::new(id, distance)));
        if self.heap.len() > self.capacity {
            self.heap.pop();
        }
    }

    /// Pop the largest element.
    #[inline]
    pub fn pop(&mut self) -> Option<HeapEntry<D>> {
        self.heap.pop().map(|e| e.0)
    }

    /// Convert to a sorted vector (smallest distance first).
    pub fn into_sorted_vec(self) -> Vec<HeapEntry<D>> {
        // Pre-allocate with exact capacity to avoid reallocations
        let len = self.heap.len();
        let mut entries = Vec::with_capacity(len);
        entries.extend(self.heap.into_iter().map(|e| e.0));
        entries.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        entries
    }

    /// Convert to a sorted vector of (id, distance) pairs.
    /// This is optimized to minimize allocations by reusing the heap's buffer.
    #[inline]
    pub fn into_sorted_pairs(self) -> Vec<(IdType, D)> {
        // Use into_sorted_iter from BinaryHeap which pops in sorted order
        let len = self.heap.len();
        let mut result = Vec::with_capacity(len);
        // BinaryHeap::into_sorted_vec() uses into_iter + sort, we do the same
        // but map directly to the output format
        let mut entries: Vec<_> = self.heap.into_vec();
        entries.sort_by(|a, b| {
            a.0.distance
                .partial_cmp(&b.0.distance)
                .unwrap_or(Ordering::Equal)
        });
        result.extend(entries.into_iter().map(|e| (e.0.id, e.0.distance)));
        result
    }

    /// Convert to a vector (unordered).
    pub fn into_vec(self) -> Vec<HeapEntry<D>> {
        self.heap.into_iter().map(|e| e.0).collect()
    }

    /// Iterate over entries (unordered).
    pub fn iter(&self) -> impl Iterator<Item = HeapEntry<D>> + '_ {
        self.heap.iter().map(|e| e.0)
    }

    /// Clear the heap.
    pub fn clear(&mut self) {
        self.heap.clear();
    }
}

/// A min-heap for processing candidates in order of increasing distance.
///
/// This is used for the candidate list during HNSW graph traversal,
/// where we want to process the closest candidates first.
#[derive(Debug)]
pub struct MinHeap<D: DistanceType> {
    heap: BinaryHeap<MinHeapEntry<D>>,
}

impl<D: DistanceType> MinHeap<D> {
    /// Create a new empty min-heap.
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    /// Create a new min-heap with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
        }
    }

    /// Get the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Get the smallest distance without removing.
    #[inline]
    pub fn top_distance(&self) -> Option<D> {
        self.heap.peek().map(|e| e.0.distance)
    }

    /// Peek at the smallest entry.
    #[inline]
    pub fn peek(&self) -> Option<HeapEntry<D>> {
        self.heap.peek().map(|e| e.0)
    }

    /// Insert an element.
    #[inline]
    pub fn push(&mut self, id: IdType, distance: D) {
        self.heap.push(MinHeapEntry(HeapEntry::new(id, distance)));
    }

    /// Pop the smallest element.
    #[inline]
    pub fn pop(&mut self) -> Option<HeapEntry<D>> {
        self.heap.pop().map(|e| e.0)
    }

    /// Clear the heap.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Convert to a sorted vector (smallest distance first).
    pub fn into_sorted_vec(mut self) -> Vec<HeapEntry<D>> {
        let mut result = Vec::with_capacity(self.heap.len());
        while let Some(entry) = self.pop() {
            result.push(entry);
        }
        result
    }
}

impl<D: DistanceType> Default for MinHeap<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_heap_topk() {
        let mut heap = MaxHeap::<f32>::new(3);

        heap.try_insert(1, 5.0);
        heap.try_insert(2, 3.0);
        heap.try_insert(3, 7.0);
        assert!(heap.is_full());
        assert_eq!(heap.top_distance(), Some(7.0));

        // This should replace 7.0
        assert!(heap.try_insert(4, 2.0));
        assert_eq!(heap.top_distance(), Some(5.0));

        // This should not be inserted (8.0 > 5.0)
        assert!(!heap.try_insert(5, 8.0));

        let sorted = heap.into_sorted_vec();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].id, 4); // distance 2.0
        assert_eq!(sorted[1].id, 2); // distance 3.0
        assert_eq!(sorted[2].id, 1); // distance 5.0
    }

    #[test]
    fn test_min_heap() {
        let mut heap = MinHeap::<f32>::new();

        heap.push(1, 5.0);
        heap.push(2, 3.0);
        heap.push(3, 7.0);
        heap.push(4, 1.0);

        // Should come out in ascending order
        assert_eq!(heap.pop().unwrap().id, 4); // 1.0
        assert_eq!(heap.pop().unwrap().id, 2); // 3.0
        assert_eq!(heap.pop().unwrap().id, 1); // 5.0
        assert_eq!(heap.pop().unwrap().id, 3); // 7.0
        assert!(heap.pop().is_none());
    }
}

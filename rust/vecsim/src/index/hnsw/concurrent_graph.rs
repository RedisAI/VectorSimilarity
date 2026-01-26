//! Lock-free concurrent graph storage for HNSW.
//!
//! This module provides a concurrent graph structure that allows multiple threads
//! to read and write graph data without blocking each other. It uses a segmented
//! approach where each segment is a fixed-size array of graph elements.
//!
//! The key insight from the C++ HNSW implementation is that the graph array itself
//! doesn't need locks - only individual node modifications need synchronization,
//! which is handled by the per-node locks in `ElementGraphData`.

use super::ElementGraphData;
use crate::types::IdType;
use parking_lot::RwLock;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// Flags for element status (matches C++ Flags enum).
#[allow(dead_code)]
mod flags {
    /// Element is logically deleted but still exists in the graph.
    pub const DELETE_MARK: u8 = 0x1;
    /// Element is being inserted into the graph.
    pub const IN_PROCESS: u8 = 0x2;
}

/// Default segment size (number of elements per segment).
const SEGMENT_SIZE: usize = 4096;

/// A segment of graph elements.
///
/// Each slot uses `UnsafeCell` to allow interior mutability. Thread safety
/// is ensured by:
/// 1. Each element has its own lock (`ElementGraphData.lock`)
/// 2. Element initialization is atomic (write once, then read-only structure)
/// 3. Neighbor modifications use the per-element lock
struct GraphSegment {
    data: Box<[UnsafeCell<Option<ElementGraphData>>]>,
}

impl GraphSegment {
    /// Create a new segment with the given size.
    fn new(size: usize) -> Self {
        let mut vec = Vec::with_capacity(size);
        for _ in 0..size {
            vec.push(UnsafeCell::new(None));
        }
        Self {
            data: vec.into_boxed_slice(),
        }
    }

    /// Get a reference to an element.
    ///
    /// # Safety
    /// Caller must ensure no mutable reference exists to this element.
    #[inline]
    unsafe fn get(&self, index: usize) -> Option<&ElementGraphData> {
        if index >= self.data.len() {
            return None;
        }
        (*self.data[index].get()).as_ref()
    }

    /// Set an element.
    ///
    /// # Safety
    /// Caller must ensure no other thread is writing to this exact index.
    #[inline]
    unsafe fn set(&self, index: usize, value: ElementGraphData) {
        if index < self.data.len() {
            *self.data[index].get() = Some(value);
        }
    }

    /// Get the number of slots in this segment.
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
}

// Safety: GraphSegment is safe to share across threads because:
// 1. The Box<[UnsafeCell<...>]> is never reallocated after creation
// 2. Individual elements use UnsafeCell for interior mutability
// 3. Thread safety is ensured by callers using per-element locks
unsafe impl Send for GraphSegment {}
unsafe impl Sync for GraphSegment {}

/// A concurrent graph structure for HNSW.
///
/// This structure allows lock-free reads and writes to graph elements.
/// Growth (adding new segments) uses a read-write lock but this is rare
/// since segments are large (4096 elements by default).
pub struct ConcurrentGraph {
    /// Segments of graph elements.
    /// RwLock is only acquired for growth (very rare).
    segments: RwLock<Vec<GraphSegment>>,
    /// Number of elements per segment.
    segment_size: usize,
    /// Total number of initialized elements (approximate, may be slightly stale).
    len: AtomicUsize,
    /// Atomic flags for each element (DELETE_MARK, IN_PROCESS).
    /// Stored separately for O(1) access without acquiring segment lock.
    /// This matches the C++ idToMetaData approach.
    element_flags: RwLock<Vec<AtomicU8>>,
}

impl ConcurrentGraph {
    /// Create a new concurrent graph with initial capacity.
    pub fn new(initial_capacity: usize) -> Self {
        let segment_size = SEGMENT_SIZE;
        let num_segments = initial_capacity.div_ceil(segment_size).max(1);

        let segments: Vec<_> = (0..num_segments)
            .map(|_| GraphSegment::new(segment_size))
            .collect();

        // Pre-allocate flags array for initial capacity
        let flags: Vec<AtomicU8> = (0..initial_capacity)
            .map(|_| AtomicU8::new(0))
            .collect();

        Self {
            segments: RwLock::new(segments),
            segment_size,
            len: AtomicUsize::new(0),
            element_flags: RwLock::new(flags),
        }
    }

    /// Get the segment and offset for an ID.
    #[inline]
    fn id_to_indices(&self, id: IdType) -> (usize, usize) {
        let id = id as usize;
        (id / self.segment_size, id % self.segment_size)
    }

    /// Get a reference to an element by ID.
    ///
    /// Returns `None` if the ID is out of bounds or the element is not initialized.
    #[inline]
    pub fn get(&self, id: IdType) -> Option<&ElementGraphData> {
        let (seg_idx, offset) = self.id_to_indices(id);
        let segments = self.segments.read();
        if seg_idx >= segments.len() {
            return None;
        }
        // Safety: We hold the read lock on segments, ensuring the segment exists.
        // The returned reference is safe because:
        // 1. Segments are never removed or reallocated
        // 2. Individual elements use their own lock for modifications
        // 3. We're returning an immutable reference to the ElementGraphData
        unsafe {
            // Transmute lifetime to 'static then back to our lifetime.
            // This is safe because segments are never removed and elements
            // are never moved once created.
            let segment = &segments[seg_idx];
            let result = segment.get(offset);
            std::mem::transmute::<Option<&ElementGraphData>, Option<&ElementGraphData>>(result)
        }
    }

    /// Set an element at the given ID.
    ///
    /// This method ensures the graph has capacity for the ID before setting.
    /// It's safe to call from multiple threads for different IDs.
    pub fn set(&self, id: IdType, value: ElementGraphData) {
        let (seg_idx, offset) = self.id_to_indices(id);

        // Ensure we have enough segments
        self.ensure_capacity(id);

        // Set the element
        let segments = self.segments.read();
        // Safety: We've ensured capacity above, and each ID is only written once
        // during insertion. The ElementGraphData's own lock handles concurrent
        // modifications to neighbors.
        unsafe {
            segments[seg_idx].set(offset, value);
        }

        // Update length (best-effort, may be slightly stale)
        let id_plus_one = (id as usize) + 1;
        loop {
            let current = self.len.load(Ordering::Relaxed);
            if id_plus_one <= current {
                break;
            }
            if self
                .len
                .compare_exchange_weak(current, id_plus_one, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Ensure the graph has capacity for the given ID.
    fn ensure_capacity(&self, id: IdType) {
        let (seg_idx, _) = self.id_to_indices(id);

        // Fast path - check with read lock
        {
            let segments = self.segments.read();
            if seg_idx < segments.len() {
                return;
            }
        }

        // Slow path - need to grow
        let mut segments = self.segments.write();
        // Double-check after acquiring write lock
        while seg_idx >= segments.len() {
            segments.push(GraphSegment::new(self.segment_size));
        }
    }

    /// Ensure the flags array has capacity for the given ID.
    fn ensure_flags_capacity(&self, id: IdType) {
        let id_usize = id as usize;

        // Fast path - check with read lock
        {
            let flags = self.element_flags.read();
            if id_usize < flags.len() {
                return;
            }
        }

        // Slow path - need to grow
        let mut flags = self.element_flags.write();
        // Double-check after acquiring write lock
        let needed = id_usize + 1;
        let current_len = flags.len();
        if needed > current_len {
            // Grow by at least one segment worth
            let new_capacity = needed.max(current_len + self.segment_size);
            let additional = new_capacity - current_len;
            flags.reserve(additional);
            for _ in 0..additional {
                flags.push(AtomicU8::new(0));
            }
        }
    }

    /// Check if an element is marked as deleted (O(1) lock-free after initial read lock).
    #[inline]
    pub fn is_marked_deleted(&self, id: IdType) -> bool {
        let id_usize = id as usize;
        let flags = self.element_flags.read();
        if id_usize < flags.len() {
            flags[id_usize].load(Ordering::Acquire) & flags::DELETE_MARK != 0
        } else {
            false
        }
    }

    /// Mark an element as deleted atomically.
    pub fn mark_deleted(&self, id: IdType) {
        self.ensure_flags_capacity(id);
        let flags = self.element_flags.read();
        let id_usize = id as usize;
        if id_usize < flags.len() {
            flags[id_usize].fetch_or(flags::DELETE_MARK, Ordering::Release);
        }
    }

    /// Get the approximate number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Check if the graph is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the total capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.segments.read().len() * self.segment_size
    }

    /// Iterate over all initialized elements.
    ///
    /// Note: This provides a snapshot view. Elements may be added during iteration.
    pub fn iter(&self) -> impl Iterator<Item = (IdType, &ElementGraphData)> {
        let len = self.len();
        (0..len as IdType).filter_map(move |id| self.get(id).map(|e| (id, e)))
    }

    /// Clear all elements from the graph.
    ///
    /// This resets the graph to empty state while keeping allocated memory.
    pub fn clear(&self) {
        // Reset length
        self.len.store(0, Ordering::Release);

        // Clear all segments by resetting each slot to None
        let segments = self.segments.read();
        for segment in segments.iter() {
            for i in 0..segment.len() {
                // Safety: We're the only writer during clear()
                unsafe {
                    *segment.data[i].get() = None;
                }
            }
        }
    }

    /// Replace the graph contents with a new vector of elements.
    ///
    /// This is used during compaction to rebuild the graph.
    pub fn replace(&self, new_elements: Vec<Option<ElementGraphData>>) {
        // Reset length
        self.len.store(0, Ordering::Release);

        // Clear existing segments
        let segments = self.segments.read();
        for segment in segments.iter() {
            for i in 0..segment.len() {
                unsafe {
                    *segment.data[i].get() = None;
                }
            }
        }
        drop(segments);

        // Reset flags array (clear all flags)
        {
            let mut flags = self.element_flags.write();
            // Clear existing flags
            for flag in flags.iter() {
                flag.store(0, Ordering::Release);
            }
            // Resize if needed
            let needed = new_elements.len();
            let current_len = flags.len();
            if needed > current_len {
                let additional = needed - current_len;
                flags.reserve(additional);
                for _ in 0..additional {
                    flags.push(AtomicU8::new(0));
                }
            }
        }

        // Set new elements and update flags for deleted elements
        for (id, element) in new_elements.into_iter().enumerate() {
            if let Some(data) = element {
                // If the element is marked as deleted in metadata, update flags
                if data.meta.deleted {
                    let flags = self.element_flags.read();
                    if id < flags.len() {
                        flags[id].fetch_or(flags::DELETE_MARK, Ordering::Release);
                    }
                }
                self.set(id as IdType, data);
            }
        }
    }

    /// Get an iterator over indices with their elements (for compaction).
    ///
    /// Returns (index, &ElementGraphData) for each initialized slot.
    pub fn indexed_iter(&self) -> impl Iterator<Item = (usize, &ElementGraphData)> {
        let len = self.len();
        (0..len).filter_map(move |idx| self.get(idx as IdType).map(|e| (idx, e)))
    }
}

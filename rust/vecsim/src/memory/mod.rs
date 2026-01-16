//! Custom memory allocator interface for tracking and managing memory.
//!
//! This module provides a custom memory allocator that:
//! - Tracks total allocated memory via atomic counters
//! - Supports aligned allocations
//! - Allows custom memory functions for integration with external systems
//! - Provides RAII-style scoped memory management
//!
//! ## Usage
//!
//! ```rust,ignore
//! use vecsim::memory::{VecSimAllocator, AllocatorRef};
//!
//! // Create an allocator
//! let allocator = VecSimAllocator::new();
//!
//! // Allocate memory
//! let ptr = allocator.allocate(1024);
//!
//! // Check allocation size
//! println!("Allocated: {} bytes", allocator.allocation_size());
//!
//! // Deallocate
//! allocator.deallocate(ptr, 1024);
//! ```

use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Function pointer type for malloc-style allocation.
pub type AllocFn = fn(usize) -> *mut u8;

/// Function pointer type for calloc-style allocation.
pub type CallocFn = fn(usize, usize) -> *mut u8;

/// Function pointer type for realloc-style reallocation.
pub type ReallocFn = fn(*mut u8, usize, usize) -> *mut u8;

/// Function pointer type for free-style deallocation.
pub type FreeFn = fn(*mut u8, usize);

/// Custom memory functions for integration with external systems.
#[derive(Clone, Copy)]
pub struct MemoryFunctions {
    /// Allocation function (malloc-style).
    pub alloc: AllocFn,
    /// Zero-initialized allocation function (calloc-style).
    pub calloc: CallocFn,
    /// Reallocation function (takes old_size for proper deallocation).
    pub realloc: ReallocFn,
    /// Deallocation function (takes size for proper deallocation).
    pub free: FreeFn,
}

impl Default for MemoryFunctions {
    fn default() -> Self {
        Self {
            alloc: default_alloc,
            calloc: default_calloc,
            realloc: default_realloc,
            free: default_free,
        }
    }
}

// Default memory functions using Rust's global allocator
fn default_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    unsafe {
        let layout = Layout::from_size_align_unchecked(size, 8);
        alloc(layout)
    }
}

fn default_calloc(count: usize, size: usize) -> *mut u8 {
    let total = count.saturating_mul(size);
    if total == 0 {
        return std::ptr::null_mut();
    }
    unsafe {
        let layout = Layout::from_size_align_unchecked(total, 8);
        alloc_zeroed(layout)
    }
}

fn default_realloc(ptr: *mut u8, old_size: usize, new_size: usize) -> *mut u8 {
    if ptr.is_null() {
        return default_alloc(new_size);
    }
    if new_size == 0 {
        default_free(ptr, old_size);
        return std::ptr::null_mut();
    }

    let new_ptr = default_alloc(new_size);
    if !new_ptr.is_null() {
        let copy_size = old_size.min(new_size);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, new_ptr, copy_size);
        }
        default_free(ptr, old_size);
    }
    new_ptr
}

fn default_free(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    unsafe {
        let layout = Layout::from_size_align_unchecked(size, 8);
        dealloc(ptr, layout);
    }
}

/// Header stored before each allocation for tracking.
/// This header stores information needed for deallocation.
#[repr(C)]
struct AllocationHeader {
    /// Original raw pointer from allocation (for deallocation).
    raw_ptr: *mut u8,
    /// Total size of the raw allocation.
    total_size: u64,
    /// User-requested size.
    user_size: u64,
}

impl AllocationHeader {
    const SIZE: usize = std::mem::size_of::<Self>();
    const ALIGN: usize = std::mem::align_of::<Self>();
}

/// Thread-safe memory allocator with tracking capabilities.
///
/// This allocator tracks total memory allocated and supports custom memory functions
/// for integration with external systems like Redis.
pub struct VecSimAllocator {
    /// Total bytes currently allocated.
    allocated: AtomicU64,
    /// Custom memory functions.
    mem_functions: MemoryFunctions,
}

impl VecSimAllocator {
    /// Create a new allocator with default memory functions.
    pub fn new() -> Arc<Self> {
        Self::with_memory_functions(MemoryFunctions::default())
    }

    /// Create a new allocator with custom memory functions.
    pub fn with_memory_functions(mem_functions: MemoryFunctions) -> Arc<Self> {
        Arc::new(Self {
            allocated: AtomicU64::new(0),
            mem_functions,
        })
    }

    /// Get the total bytes currently allocated.
    pub fn allocation_size(&self) -> u64 {
        self.allocated.load(Ordering::Relaxed)
    }

    /// Allocate memory with tracking.
    ///
    /// Returns a pointer to the allocated memory, or None if allocation failed.
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        self.allocate_aligned(size, 8)
    }

    /// Allocate memory with specific alignment.
    pub fn allocate_aligned(&self, size: usize, alignment: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            return None;
        }

        // Ensure alignment is at least header alignment
        let alignment = alignment.max(AllocationHeader::ALIGN);

        // Calculate total size: header + padding for alignment + user data
        // We need enough space so that after placing header, we can align the user pointer
        let header_size = AllocationHeader::SIZE;
        let total_size = header_size + alignment + size;

        // Allocate raw memory
        let raw_ptr = (self.mem_functions.alloc)(total_size);
        if raw_ptr.is_null() {
            return None;
        }

        // Calculate where to place user data (aligned)
        // User data starts at: raw_ptr + header_size, then aligned up
        let user_ptr_unaligned = unsafe { raw_ptr.add(header_size) };
        let offset = user_ptr_unaligned.align_offset(alignment);
        let user_ptr = unsafe { user_ptr_unaligned.add(offset) };

        // Place header just before user pointer
        let header_ptr = unsafe { user_ptr.sub(header_size) } as *mut AllocationHeader;
        unsafe {
            (*header_ptr).raw_ptr = raw_ptr;
            (*header_ptr).total_size = total_size as u64;
            (*header_ptr).user_size = size as u64;
        }

        // Track allocation
        self.allocated.fetch_add(total_size as u64, Ordering::Relaxed);

        NonNull::new(user_ptr)
    }

    /// Allocate zero-initialized memory.
    pub fn callocate(&self, size: usize) -> Option<NonNull<u8>> {
        let ptr = self.allocate(size)?;
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, size);
        }
        Some(ptr)
    }

    /// Reallocate memory to a new size.
    ///
    /// This copies data from the old allocation to the new one.
    pub fn reallocate(&self, ptr: NonNull<u8>, new_size: usize) -> Option<NonNull<u8>> {
        // Get old info from header
        let header_ptr =
            unsafe { ptr.as_ptr().sub(AllocationHeader::SIZE) } as *const AllocationHeader;
        let old_user_size = unsafe { (*header_ptr).user_size } as usize;

        // Allocate new memory
        let new_ptr = self.allocate(new_size)?;

        // Copy data
        let copy_size = old_user_size.min(new_size);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), copy_size);
        }

        // Free old memory
        self.deallocate(ptr);

        Some(new_ptr)
    }

    /// Deallocate memory.
    pub fn deallocate(&self, ptr: NonNull<u8>) {
        // Get header
        let header_ptr =
            unsafe { ptr.as_ptr().sub(AllocationHeader::SIZE) } as *const AllocationHeader;
        let raw_ptr = unsafe { (*header_ptr).raw_ptr };
        let total_size = unsafe { (*header_ptr).total_size } as usize;

        // Free the raw allocation
        (self.mem_functions.free)(raw_ptr, total_size);

        // Update tracking
        self.allocated.fetch_sub(total_size as u64, Ordering::Relaxed);
    }

    /// Allocate with RAII wrapper that automatically deallocates on drop.
    pub fn allocate_scoped(self: &Arc<Self>, size: usize) -> Option<ScopedAllocation> {
        let ptr = self.allocate(size)?;
        Some(ScopedAllocation {
            ptr,
            size,
            allocator: Arc::clone(self),
        })
    }

    /// Allocate aligned memory with RAII wrapper.
    pub fn allocate_aligned_scoped(
        self: &Arc<Self>,
        size: usize,
        alignment: usize,
    ) -> Option<ScopedAllocation> {
        let ptr = self.allocate_aligned(size, alignment)?;
        Some(ScopedAllocation {
            ptr,
            size,
            allocator: Arc::clone(self),
        })
    }
}

impl Default for VecSimAllocator {
    fn default() -> Self {
        Self {
            allocated: AtomicU64::new(0),
            mem_functions: MemoryFunctions::default(),
        }
    }
}

/// Reference-counted allocator handle.
pub type AllocatorRef = Arc<VecSimAllocator>;

/// RAII wrapper for scoped allocations.
///
/// The memory is automatically deallocated when this struct is dropped.
pub struct ScopedAllocation {
    ptr: NonNull<u8>,
    size: usize,
    allocator: Arc<VecSimAllocator>,
}

impl ScopedAllocation {
    /// Get the raw pointer.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the NonNull pointer.
    pub fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
    }

    /// Get the size of the allocation.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a slice view of the allocation.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get a mutable slice view of the allocation.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }
}

impl Drop for ScopedAllocation {
    fn drop(&mut self) {
        self.allocator.deallocate(self.ptr);
    }
}

/// Trait for types that can provide their allocator.
pub trait HasAllocator {
    /// Get a reference to the allocator.
    fn allocator(&self) -> &Arc<VecSimAllocator>;

    /// Get the total memory allocated by this allocator.
    fn memory_usage(&self) -> u64 {
        self.allocator().allocation_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_basic() {
        let allocator = VecSimAllocator::new();

        // Initial allocation should be 0
        assert_eq!(allocator.allocation_size(), 0);

        // Allocate some memory
        let ptr = allocator.allocate(1024).expect("allocation failed");

        // Should have tracked the allocation (with header overhead)
        assert!(allocator.allocation_size() > 1024);

        // Deallocate
        allocator.deallocate(ptr);

        // Should be back to 0
        assert_eq!(allocator.allocation_size(), 0);
    }

    #[test]
    fn test_allocator_aligned() {
        let allocator = VecSimAllocator::new();

        // Allocate with 64-byte alignment
        let ptr = allocator
            .allocate_aligned(1024, 64)
            .expect("allocation failed");

        // Check alignment
        assert_eq!(ptr.as_ptr() as usize % 64, 0);

        allocator.deallocate(ptr);
        assert_eq!(allocator.allocation_size(), 0);
    }

    #[test]
    fn test_allocator_callocate() {
        let allocator = VecSimAllocator::new();

        let ptr = allocator.callocate(1024).expect("allocation failed");

        // Check that memory is zeroed
        unsafe {
            let slice = std::slice::from_raw_parts(ptr.as_ptr(), 1024);
            assert!(slice.iter().all(|&b| b == 0));
        }

        allocator.deallocate(ptr);
        assert_eq!(allocator.allocation_size(), 0);
    }

    #[test]
    fn test_scoped_allocation() {
        let allocator = VecSimAllocator::new();

        {
            let alloc = allocator.allocate_scoped(1024).expect("allocation failed");
            assert!(allocator.allocation_size() > 0);
            assert_eq!(alloc.size(), 1024);
        }

        // Should be deallocated after scope ends
        assert_eq!(allocator.allocation_size(), 0);
    }

    #[test]
    fn test_multiple_allocations() {
        let allocator = VecSimAllocator::new();

        let ptr1 = allocator.allocate(1024).expect("allocation failed");
        let ptr2 = allocator.allocate(2048).expect("allocation failed");
        let ptr3 = allocator.allocate(512).expect("allocation failed");

        // Total should be sum of all allocations (plus overhead)
        assert!(allocator.allocation_size() > 1024 + 2048 + 512);

        allocator.deallocate(ptr1);
        allocator.deallocate(ptr2);
        allocator.deallocate(ptr3);

        assert_eq!(allocator.allocation_size(), 0);
    }

    #[test]
    fn test_reallocate() {
        let allocator = VecSimAllocator::new();

        let ptr = allocator.allocate(1024).expect("allocation failed");

        // Write some data
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), 1024);
            for (i, byte) in slice.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }
        }

        // Reallocate to larger size
        let new_ptr = allocator.reallocate(ptr, 2048).expect("reallocation failed");

        // Check data was preserved
        unsafe {
            let slice = std::slice::from_raw_parts(new_ptr.as_ptr(), 1024);
            for (i, &byte) in slice.iter().enumerate() {
                assert_eq!(byte, (i % 256) as u8);
            }
        }

        allocator.deallocate(new_ptr);
        assert_eq!(allocator.allocation_size(), 0);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let allocator = VecSimAllocator::new();

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let alloc = Arc::clone(&allocator);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let ptr = alloc.allocate(1024).expect("allocation failed");
                        alloc.deallocate(ptr);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(allocator.allocation_size(), 0);
    }

    #[test]
    fn test_scoped_slice_access() {
        let allocator = VecSimAllocator::new();

        let mut alloc = allocator.allocate_scoped(256).expect("allocation failed");

        // Write via mutable slice
        {
            let slice = alloc.as_mut_slice();
            for (i, byte) in slice.iter_mut().enumerate() {
                *byte = i as u8;
            }
        }

        // Read via immutable slice
        {
            let slice = alloc.as_slice();
            for (i, &byte) in slice.iter().enumerate() {
                assert_eq!(byte, i as u8);
            }
        }
    }
}

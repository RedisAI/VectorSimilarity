//! Memory prefetching utilities for optimizing cache behavior.
//!
//! This module provides platform-specific memory prefetching hints to help
//! hide memory latency during graph traversal in HNSW search.

/// Prefetch data into L1 cache for reading.
///
/// This is a hint to the processor that the data at the given pointer
/// will be needed soon. On x86_64, this uses the `_mm_prefetch` intrinsic.
/// On other architectures (including aarch64), this is currently a no-op
/// as the aarch64 prefetch intrinsics are not yet stable in Rust.
#[inline]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        // Safety: _mm_prefetch is safe to call with any pointer.
        // If the pointer is invalid, the prefetch is simply ignored.
        unsafe {
            use std::arch::x86_64::*;
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        }
    }

    // Note: aarch64 prefetch intrinsics (_prefetch) are unstable as of Rust 1.75+.
    // When they become stable, we can add:
    // #[cfg(target_arch = "aarch64")]
    // unsafe {
    //     use std::arch::aarch64::*;
    //     _prefetch(ptr as *const i8, _PREFETCH_READ, _PREFETCH_LOCALITY3);
    // }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr; // Suppress unused warning
    }
}

/// Prefetch a slice of data into cache.
///
/// This prefetches the beginning of the slice, which is typically
/// sufficient for vector data that fits in a few cache lines.
#[inline]
pub fn prefetch_slice<T>(slice: &[T]) {
    if !slice.is_empty() {
        prefetch_read(slice.as_ptr());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_read_null() {
        // Should not crash on null pointer
        prefetch_read(std::ptr::null::<u8>());
    }

    #[test]
    fn test_prefetch_read_valid() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        prefetch_read(data.as_ptr());
    }

    #[test]
    fn test_prefetch_slice_empty() {
        let empty: &[f32] = &[];
        prefetch_slice(empty);
    }

    #[test]
    fn test_prefetch_slice_valid() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        prefetch_slice(&data);
    }
}


//! Memory prefetching utilities for optimizing cache behavior.
//!
//! This module provides platform-specific memory prefetching hints to help
//! hide memory latency during graph traversal in HNSW search.

/// Prefetch data into L1 cache for reading.
///
/// This is a hint to the processor that the data at the given pointer
/// will be needed soon. On x86_64, this uses the `_mm_prefetch` intrinsic.
/// On aarch64, this uses inline assembly with the `prfm pldl1keep` instruction.
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

    #[cfg(target_arch = "aarch64")]
    {
        // Use inline assembly for aarch64 prefetch (prfm pldl1keep)
        // PLDL1KEEP = Prefetch for Load, L1 cache, temporal (keep in cache)
        // Safety: prfm is a hint instruction that is safe to call with any pointer.
        // If the pointer is invalid or unmapped, the prefetch is silently ignored.
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags)
            );
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr; // Suppress unused warning on other architectures
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

/// Configuration for adaptive prefetching based on vector characteristics.
#[derive(Debug, Clone, Copy)]
pub struct PrefetchConfig {
    /// Number of vectors to prefetch ahead in the search path.
    pub prefetch_depth: usize,
    /// Whether to also prefetch graph structure.
    pub prefetch_graph: bool,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            prefetch_depth: 2,
            prefetch_graph: true,
        }
    }
}

impl PrefetchConfig {
    /// Create config optimized for the given vector size.
    ///
    /// Smaller vectors benefit from more aggressive prefetching since more
    /// can fit in cache. Larger vectors need less prefetching to avoid
    /// cache pollution.
    pub fn for_vector_size(dim: usize, element_size: usize) -> Self {
        let vector_bytes = dim * element_size;

        // Adjust prefetch depth based on vector size
        // Smaller vectors: prefetch more aggressively
        // Larger vectors: prefetch less to avoid cache pollution
        let prefetch_depth = if vector_bytes <= 256 {
            4 // Small vectors (e.g., dim=64 f32): prefetch 4 ahead
        } else if vector_bytes <= 1024 {
            2 // Medium vectors (e.g., dim=256 f32): prefetch 2 ahead
        } else if vector_bytes <= 4096 {
            1 // Large vectors (e.g., dim=1024 f32): prefetch 1 ahead
        } else {
            0 // Very large vectors: no prefetch (would pollute cache)
        };

        // Only prefetch graph structure for smaller vectors
        let prefetch_graph = vector_bytes <= 512;

        Self {
            prefetch_depth,
            prefetch_graph,
        }
    }

    /// Create config for a specific vector type.
    pub fn for_type<T>(dim: usize) -> Self {
        Self::for_vector_size(dim, std::mem::size_of::<T>())
    }
}

/// Prefetch multiple cache lines of a vector.
///
/// For larger vectors spanning multiple cache lines, this prefetches
/// all cache lines to ensure the entire vector is in cache.
#[inline]
pub fn prefetch_vector<T>(data: &[T]) {
    if data.is_empty() {
        return;
    }

    let bytes = std::mem::size_of_val(data);
    let cache_line_size = 64;
    let cache_lines = (bytes + cache_line_size - 1) / cache_line_size;
    let ptr = data.as_ptr() as *const u8;

    // Prefetch up to 8 cache lines (512 bytes)
    for i in 0..cache_lines.min(8) {
        prefetch_read(unsafe { ptr.add(i * cache_line_size) } as *const T);
    }
}

/// Prefetch data for multiple neighbors ahead in the search path.
///
/// This function prefetches vector data for neighbors that will be
/// visited in upcoming iterations, hiding memory latency.
#[inline]
pub fn prefetch_neighbors<'a, T, F>(
    neighbors: &[u32],
    current_idx: usize,
    config: &PrefetchConfig,
    data_getter: &F,
) where
    T: 'a,
    F: Fn(u32) -> Option<&'a [T]>,
{
    if config.prefetch_depth == 0 {
        return;
    }

    // Prefetch data for neighbors ahead in the iteration
    for offset in 1..=config.prefetch_depth {
        let prefetch_idx = current_idx + offset;
        if prefetch_idx < neighbors.len() {
            if let Some(data) = data_getter(neighbors[prefetch_idx]) {
                prefetch_vector(data);
            }
        }
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

    #[test]
    fn test_prefetch_config_default() {
        let config = PrefetchConfig::default();
        assert_eq!(config.prefetch_depth, 2);
        assert!(config.prefetch_graph);
    }

    #[test]
    fn test_prefetch_config_for_vector_size() {
        // Small vectors (256 bytes = 64 f32)
        let small = PrefetchConfig::for_vector_size(64, 4);
        assert_eq!(small.prefetch_depth, 4);
        assert!(small.prefetch_graph);

        // Medium vectors (1024 bytes = 256 f32)
        let medium = PrefetchConfig::for_vector_size(256, 4);
        assert_eq!(medium.prefetch_depth, 2);
        assert!(!medium.prefetch_graph); // 1024 > 512, so no graph prefetch

        // Large vectors (4096 bytes = 1024 f32)
        let large = PrefetchConfig::for_vector_size(1024, 4);
        assert_eq!(large.prefetch_depth, 1);
        assert!(!large.prefetch_graph);

        // Very large vectors (8192 bytes = 2048 f32)
        let very_large = PrefetchConfig::for_vector_size(2048, 4);
        assert_eq!(very_large.prefetch_depth, 0);
        assert!(!very_large.prefetch_graph);
    }

    #[test]
    fn test_prefetch_config_for_type() {
        let config = PrefetchConfig::for_type::<f32>(128);
        // 128 * 4 = 512 bytes
        assert_eq!(config.prefetch_depth, 2);
        assert!(config.prefetch_graph);
    }

    #[test]
    fn test_prefetch_vector_multiple_cache_lines() {
        // Vector spanning multiple cache lines
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect(); // 1024 bytes = 16 cache lines
        prefetch_vector(&data);
    }

    #[test]
    fn test_prefetch_vector_empty() {
        let empty: &[f32] = &[];
        prefetch_vector(empty);
    }
}


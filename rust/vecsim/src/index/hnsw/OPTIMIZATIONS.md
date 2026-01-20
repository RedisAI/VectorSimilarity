# HNSW Performance Optimization Opportunities

This document analyzes specific optimization opportunities for the Rust HNSW implementation,
with code examples and implementation suggestions based on analysis of the current codebase.

## Table of Contents

1. [Batch Distance Computation](#1-batch-distance-computation)
2. [Memory Layout Improvements](#2-memory-layout-improvements)
3. [Product Quantization Integration](#3-product-quantization-integration)
4. [Parallel Search Improvements](#4-parallel-search-improvements)
5. [Adaptive Parameters](#5-adaptive-parameters)
6. [Graph Construction Optimizations](#6-graph-construction-optimizations)

---

## 1. Batch Distance Computation

### Current Implementation Analysis

**Location**: `rust/vecsim/src/index/hnsw/search.rs:172-193`

The current `search_layer` function computes distances one at a time in the inner loop:

```rust
for neighbor in element.iter_neighbors(level) {
    if visited.visit(neighbor) {
        continue;
    }
    // ...
    if let Some(data) = data_getter(neighbor) {
        let dist = dist_fn.compute(data, query, dim);  // One distance at a time
        // ...
    }
}
```

### Optimization Opportunity

**Expected Impact**: 15-30% speedup in search operations  
**Complexity**: Medium

Batch multiple neighbors for SIMD-efficient distance computation. Instead of computing
distances one at a time, collect neighbor vectors and compute distances in batches.

### Suggested Implementation

```rust
// New batch distance function in rust/vecsim/src/distance/mod.rs
pub trait BatchDistanceFunction<T: VectorElement> {
    /// Compute distances from query to multiple vectors.
    /// Returns distances in the same order as input vectors.
    fn compute_batch(
        &self,
        query: &[T],
        vectors: &[&[T]],  // Multiple vectors
        dim: usize,
    ) -> Vec<T::DistanceType>;
}

// Example AVX2 implementation for L2 (rust/vecsim/src/distance/simd/avx2.rs)
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_squared_batch_f32_avx2(
    query: *const f32,
    vectors: &[*const f32],  // Pointers to multiple vectors
    dim: usize,
) -> Vec<f32> {
    let mut results = Vec::with_capacity(vectors.len());
    
    // Process in groups of 4 vectors (optimal for AVX2 256-bit registers)
    for chunk in vectors.chunks(4) {
        // Interleave loading from 4 vectors for better memory bandwidth
        let mut sums = [_mm256_setzero_ps(); 4];
        
        for d in (0..dim).step_by(8) {
            let q = _mm256_loadu_ps(query.add(d));
            
            for (i, &vec_ptr) in chunk.iter().enumerate() {
                let v = _mm256_loadu_ps(vec_ptr.add(d));
                let diff = _mm256_sub_ps(q, v);
                sums[i] = _mm256_fmadd_ps(diff, diff, sums[i]);
            }
        }
        
        // Horizontal sum and store results
        for (i, sum) in sums.iter().take(chunk.len()).enumerate() {
            results.push(hsum256_ps(*sum));
        }
    }
    results
}
```

### Modified search_layer

```rust
// In rust/vecsim/src/index/hnsw/search.rs
pub fn search_layer_batched<'a, T, D, F, P, G>(/* ... */) -> SearchResult<D> {
    // ... setup ...
    
    while let Some(candidate) = candidates.pop() {
        if let Some(element) = graph.get(candidate.id) {
            // Collect non-visited neighbors
            let mut batch_ids: Vec<IdType> = Vec::with_capacity(32);
            let mut batch_data: Vec<&[T]> = Vec::with_capacity(32);
            
            for neighbor in element.iter_neighbors(level) {
                if visited.visit(neighbor) {
                    continue;
                }
                if let Some(data) = data_getter(neighbor) {
                    batch_ids.push(neighbor);
                    batch_data.push(data);
                }
            }
            
            // Batch distance computation
            if !batch_data.is_empty() {
                let distances = dist_fn.compute_batch(query, &batch_data, dim);
                
                for (i, dist) in distances.into_iter().enumerate() {
                    let neighbor = batch_ids[i];
                    // ... rest of distance processing ...
                }
            }
        }
    }
}
```

---

## 2. Memory Layout Improvements

### Current Implementation Analysis

**Location**: `rust/vecsim/src/index/hnsw/graph.rs:42-52, 194-222`

Current `LevelLinks` structure:

```rust
pub struct LevelLinks {
    neighbors: Vec<AtomicU32>,    // Dynamic allocation, pointer indirection
    count: AtomicU32,             // 4 bytes
    capacity: usize,              // 8 bytes
}

pub struct ElementGraphData {
    pub meta: ElementMetaData,    // 10+ bytes (label u64 + level u8 + deleted bool)
    pub levels: Vec<LevelLinks>,  // Dynamic Vec, pointer indirection
    pub lock: Mutex<()>,          // ~40+ bytes (parking_lot Mutex)
}
```

### Issues

1. **Cache Misses**: Multiple pointer indirections (ElementGraphData → Vec → LevelLinks → Vec)
2. **Memory Fragmentation**: Small allocations scattered across heap
3. **Lock Overhead**: Per-element mutex is ~40 bytes overhead per element

### Optimization Opportunity

**Expected Impact**: 10-25% speedup in graph traversal  
**Complexity**: Hard

### Suggested Implementation: Cache-Line Aligned Compact Structure

```rust
// rust/vecsim/src/index/hnsw/graph_compact.rs

/// Compact neighbor list that fits in cache lines.
/// For M=16, this is exactly 64 bytes (one cache line).
#[repr(C, align(64))]
pub struct CompactLevelLinks {
    /// Neighbor IDs stored inline (no pointer indirection)
    neighbors: [u32; 15],    // 60 bytes - supports up to M=15
    count: u8,               // 1 byte
    capacity: u8,            // 1 byte  
    _padding: [u8; 2],       // 2 bytes padding for alignment
}

/// For level 0 with M_max_0=32, use two cache lines
#[repr(C, align(64))]
pub struct CompactLevelLinks0 {
    neighbors: [u32; 31],    // 124 bytes
    count: u8,               // 1 byte
    capacity: u8,            // 1 byte
    _padding: [u8; 2],       // 2 bytes
}  // Total: 128 bytes = 2 cache lines

impl CompactLevelLinks {
    #[inline]
    pub fn iter_neighbors(&self) -> impl Iterator<Item = u32> + '_ {
        self.neighbors[..self.count as usize].iter().copied()
    }
}
```

### Structure-of-Arrays Layout for Vector Data

**Location**: `rust/vecsim/src/containers/data_blocks.rs`

Consider a Structure-of-Arrays (SoA) layout for better SIMD efficiency:

```rust
// Current: Array of Structures (AoS)
// vectors[0] = [x0, y0, z0, w0]
// vectors[1] = [x1, y1, z1, w1]
// ...

// Proposed: Structure of Arrays (SoA) for specific dimensions
pub struct SoAVectorBlock {
    x: Vec<f32>,  // All x components contiguous
    y: Vec<f32>,  // All y components contiguous
    z: Vec<f32>,  // All z components contiguous
    w: Vec<f32>,  // All w components contiguous
}

// This enables computing distances to 8 vectors simultaneously with AVX2:
// Load 8 x-components, compute 8 differences, etc.
```

### Compressed Neighbor IDs

For indices with < 65536 vectors, use u16 IDs:

```rust
/// Adaptive ID type based on index size
pub enum CompressedLinks {
    /// For small indices (< 65536 elements)
    Small(SmallLinks),
    /// For large indices
    Large(LargeLinks),
}

#[repr(C, align(64))]
pub struct SmallLinks {
    neighbors: [u16; 30],  // 60 bytes - double the capacity!
    count: u8,
    capacity: u8,
    _padding: [u8; 2],
}  // Still fits in one cache line, but 2x capacity
```

---

## 3. Product Quantization (PQ) Integration

### Current Implementation Analysis

**Location**: `rust/vecsim/src/quantization/mod.rs`

The codebase already has quantization support:
- `SQ8`: Scalar quantization to 8-bit
- `LVQ`: Learned Vector Quantization (4-bit/8-bit)
- `LeanVec`: Dimension reduction with two-level quantization

### Optimization Opportunity

**Expected Impact**: 2-4x memory reduction, 20-50% faster search
**Complexity**: Hard

### Suggested PQ Implementation

```rust
// rust/vecsim/src/quantization/pq.rs

/// Product Quantization codec
pub struct PQCodec {
    /// Number of subspaces (typically dim/4 to dim/8)
    m: usize,
    /// Bits per subspace (typically 8 for 256 centroids)
    nbits: usize,
    /// Dimension of original vectors
    dim: usize,
    /// Dimension of each subspace
    dsub: usize,
    /// Centroids for each subspace: [m][2^nbits][dsub]
    centroids: Vec<Vec<Vec<f32>>>,
}

impl PQCodec {
    /// Encode a vector to PQ codes
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.m);

        for (i, chunk) in vector.chunks(self.dsub).enumerate() {
            // Find nearest centroid for this subspace
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;

            for (j, centroid) in self.centroids[i].iter().enumerate() {
                let dist = l2_distance(chunk, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            codes.push(best_idx as u8);
        }
        codes
    }

    /// Asymmetric distance computation using precomputed lookup tables
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        // Precompute distance tables: for each subspace, distance to all centroids
        let tables: Vec<Vec<f32>> = (0..self.m)
            .map(|i| {
                let query_sub = &query[i * self.dsub..(i + 1) * self.dsub];
                self.centroids[i]
                    .iter()
                    .map(|c| l2_squared(query_sub, c))
                    .collect()
            })
            .collect();

        // Sum distances from lookup tables (very fast!)
        codes.iter()
            .enumerate()
            .map(|(i, &code)| tables[i][code as usize])
            .sum()
    }
}
```

### HNSW Integration Points

**Location**: `rust/vecsim/src/index/hnsw/mod.rs:716-724`

```rust
// In HnswCore::compute_distance
#[inline]
fn compute_distance(&self, id: IdType, query: &[T]) -> T::DistanceType {
    // Option 1: Two-stage search with PQ
    if let Some(pq) = &self.pq_codec {
        // Fast PQ distance for initial filtering
        let pq_codes = self.pq_data.get(id);
        let approx_dist = pq.asymmetric_distance(query, pq_codes);

        // Only compute exact distance if promising
        if approx_dist < self.rerank_threshold {
            if let Some(data) = self.data.get(id) {
                return self.dist_fn.compute(data, query, self.params.dim);
            }
        }
        return T::DistanceType::from_f64(approx_dist as f64);
    }

    // Original exact distance
    if let Some(data) = self.data.get(id) {
        self.dist_fn.compute(data, query, self.params.dim)
    } else {
        T::DistanceType::infinity()
    }
}
```

---

## 4. Parallel Search Improvements

### Current Implementation Analysis

**Location**: `rust/vecsim/src/index/hnsw/concurrent_graph.rs:85-93`

Current locking structure:
```rust
pub struct ConcurrentGraph {
    segments: RwLock<Vec<GraphSegment>>,  // Global lock for growth
    segment_size: usize,
    len: AtomicUsize,
}
```

**Location**: `rust/vecsim/src/index/hnsw/graph.rs:194-202`

Per-element lock:
```rust
pub struct ElementGraphData {
    pub lock: Mutex<()>,  // Per-element lock for neighbor modification
}
```

### Optimization Opportunities

**Expected Impact**: 2-4x throughput for concurrent queries
**Complexity**: Medium to Hard

### 4.1 Lock-Free Read Path

The current read path acquires a read lock on segments. For search-only workloads,
we can use epoch-based reclamation:

```rust
// rust/vecsim/src/index/hnsw/concurrent_graph_lockfree.rs
use crossbeam_epoch::{self as epoch, Atomic, Owned, Shared};

pub struct LockFreeGraph {
    /// Segments using atomic pointer
    segments: Atomic<Vec<GraphSegment>>,
    segment_size: usize,
    len: AtomicUsize,
}

impl LockFreeGraph {
    /// Lock-free read
    #[inline]
    pub fn get(&self, id: IdType) -> Option<&ElementGraphData> {
        let guard = epoch::pin();
        let segments = unsafe { self.segments.load(Ordering::Acquire, &guard).deref() };

        let (seg_idx, offset) = self.id_to_indices(id);
        if seg_idx >= segments.len() {
            return None;
        }

        unsafe { segments[seg_idx].get(offset) }
    }
}
```

### 4.2 Batch Query Parallelization

```rust
// rust/vecsim/src/index/hnsw/single.rs

impl<T: VectorElement> HnswSingle<T> {
    /// Process multiple queries in parallel
    pub fn batch_search(
        &self,
        queries: &[Vec<T>],
        k: usize,
        ef: usize,
    ) -> Vec<Vec<(IdType, T::DistanceType)>> {
        use rayon::prelude::*;

        queries
            .par_iter()
            .map(|query| {
                self.core.search(query, k, ef, None)
            })
            .collect()
    }

    /// Optimized batch search with shared state
    pub fn batch_search_optimized(
        &self,
        queries: &[Vec<T>],
        k: usize,
        ef: usize,
    ) -> Vec<Vec<(IdType, T::DistanceType)>> {
        use rayon::prelude::*;

        // Pre-allocate visited handlers for all queries
        let num_queries = queries.len();
        let handlers: Vec<_> = (0..rayon::current_num_threads())
            .map(|_| self.core.visited_pool.get())
            .collect();

        queries
            .par_iter()
            .enumerate()
            .map(|(i, query)| {
                let thread_id = rayon::current_thread_index().unwrap_or(0);
                // Reuse handler for this thread
                self.core.search_with_handler(query, k, ef, None, &handlers[thread_id])
            })
            .collect()
    }
}
```

### 4.3 SIMD-Parallel Candidate Evaluation

```rust
/// Evaluate multiple candidates in parallel using SIMD
fn evaluate_candidates_simd<T: VectorElement>(
    candidates: &[(IdType, &[T])],
    query: &[T],
    dist_fn: &dyn DistanceFunction<T, Output = T::DistanceType>,
    dim: usize,
) -> Vec<(IdType, T::DistanceType)> {
    // Process 4 candidates at a time with AVX2
    let mut results = Vec::with_capacity(candidates.len());

    for chunk in candidates.chunks(4) {
        let distances = compute_distances_4way(
            query,
            chunk.iter().map(|(_, v)| *v).collect::<Vec<_>>().as_slice(),
            dim,
        );

        for (i, (id, _)) in chunk.iter().enumerate() {
            results.push((*id, distances[i]));
        }
    }

    results
}
```

---

## 5. Adaptive Parameters

### Current Implementation Analysis

**Location**: `rust/vecsim/src/index/hnsw/mod.rs:100-120`

Current static parameters:
```rust
pub struct HnswParams {
    pub dim: usize,
    pub m: usize,
    pub m_max_0: usize,
    pub ef_construction: usize,
    pub ef_runtime: usize,
    // ...
}
```

### 5.1 Dynamic ef Adjustment

**Expected Impact**: Better quality-latency tradeoff
**Complexity**: Easy

```rust
// rust/vecsim/src/index/hnsw/adaptive.rs

/// Adaptive ef controller based on result quality
pub struct AdaptiveEfController {
    /// Minimum ef value
    min_ef: usize,
    /// Maximum ef value
    max_ef: usize,
    /// Current ef value
    current_ef: AtomicUsize,
    /// Quality threshold (e.g., recall@k target)
    quality_threshold: f32,
    /// Moving average of observed quality
    quality_ema: AtomicU32,  // Stored as fixed-point
}

impl AdaptiveEfController {
    /// Update ef based on measured quality
    pub fn update(&self, measured_recall: f32) {
        let current = self.current_ef.load(Ordering::Relaxed);

        if measured_recall < self.quality_threshold {
            // Increase ef to improve quality
            let new_ef = (current * 12 / 10).min(self.max_ef);
            self.current_ef.store(new_ef, Ordering::Relaxed);
        } else if measured_recall > self.quality_threshold + 0.05 {
            // Decrease ef to reduce latency
            let new_ef = (current * 9 / 10).max(self.min_ef);
            self.current_ef.store(new_ef, Ordering::Relaxed);
        }

        // Update EMA
        self.update_quality_ema(measured_recall);
    }

    /// Get current adaptive ef value
    pub fn get_ef(&self) -> usize {
        self.current_ef.load(Ordering::Relaxed)
    }
}
```

### 5.2 Adaptive Prefetch Based on Vector Size (SVS Approach)

**Location**: `rust/vecsim/src/index/hnsw/search.rs:82-98`

Reference: The SVS (Scalable Vector Search) paper suggests adapting prefetch
distance based on vector size to hide memory latency effectively.

```rust
// rust/vecsim/src/utils/prefetch.rs

/// Adaptive prefetch parameters based on vector characteristics
pub struct PrefetchConfig {
    /// Number of vectors to prefetch ahead
    pub prefetch_depth: usize,
    /// Whether to prefetch graph structure
    pub prefetch_graph: bool,
}

impl PrefetchConfig {
    /// Create config based on vector size and cache characteristics
    pub fn for_vector_size(dim: usize, element_size: usize) -> Self {
        let vector_bytes = dim * element_size;
        let l1_cache_size = 32 * 1024;  // 32KB typical L1D
        let cache_lines = (vector_bytes + 63) / 64;  // 64-byte cache lines

        // Prefetch more aggressively for smaller vectors
        let prefetch_depth = if vector_bytes <= 256 {
            4  // Small vectors: prefetch 4 ahead
        } else if vector_bytes <= 1024 {
            2  // Medium vectors: prefetch 2 ahead
        } else {
            1  // Large vectors: prefetch 1 ahead
        };

        Self {
            prefetch_depth,
            prefetch_graph: vector_bytes <= 512,
        }
    }
}

/// Enhanced prefetch for multiple cache lines
#[inline]
pub fn prefetch_vector<T>(data: &[T], prefetch_config: &PrefetchConfig) {
    let bytes = std::mem::size_of_val(data);
    let cache_lines = (bytes + 63) / 64;
    let ptr = data.as_ptr() as *const i8;

    for i in 0..cache_lines.min(8) {  // Max 8 cache lines
        prefetch_read(unsafe { ptr.add(i * 64) } as *const T);
    }
}
```

### 5.3 Early Termination Heuristics

```rust
// In search_layer function

/// Check if we can terminate early based on distance distribution
fn should_terminate_early<D: DistanceType>(
    results: &MaxHeap<D>,
    candidates: &MinHeap<D>,
    early_termination_factor: f32,
) -> bool {
    if !results.is_full() {
        return false;
    }

    let worst_result = results.top_distance().unwrap();
    if let Some(best_candidate) = candidates.peek() {
        // If best remaining candidate is significantly worse than
        // our worst result, we can stop
        let threshold = worst_result.to_f64() * (1.0 + early_termination_factor as f64);
        best_candidate.distance.to_f64() > threshold
    } else {
        true
    }
}
```

---

## 6. Graph Construction Optimizations

### Current Implementation Analysis

**Location**: `rust/vecsim/src/index/hnsw/mod.rs:286-466`

Current insertion flow:
1. Generate random level
2. Find entry point via greedy search
3. For each level: search_layer → select_neighbors → mutually_connect

### 6.1 Parallel Index Building

**Expected Impact**: Near-linear speedup with cores
**Complexity**: Medium

```rust
// rust/vecsim/src/index/hnsw/single.rs

impl<T: VectorElement> HnswSingle<T> {
    /// Build index from vectors in parallel
    pub fn build_parallel(
        params: HnswParams,
        vectors: &[T],
        labels: &[LabelType],
    ) -> Result<Self, IndexError> {
        let dim = params.dim;
        let num_vectors = vectors.len() / dim;

        // Phase 1: Sequential insertion of first few elements (establish structure)
        let mut index = Self::new(params.clone());
        let bootstrap_count = (num_vectors / 100).max(100).min(num_vectors);

        for i in 0..bootstrap_count {
            let vec_start = i * dim;
            let vector = &vectors[vec_start..vec_start + dim];
            index.add_vector(vector, labels[i])?;
        }

        // Phase 2: Parallel insertion of remaining elements
        let remaining: Vec<(usize, LabelType)> = (bootstrap_count..num_vectors)
            .map(|i| (i, labels[i]))
            .collect();

        remaining.par_iter().try_for_each(|&(i, label)| {
            let vec_start = i * dim;
            let vector = &vectors[vec_start..vec_start + dim];
            index.add_vector_concurrent(vector, label)
        })?;

        Ok(index)
    }
}
```

### 6.2 Incremental Neighbor Selection

Instead of recomputing all neighbors, incrementally update:

```rust
/// Incrementally update neighbor selection when a new vector is added
fn update_neighbors_incremental(
    &self,
    existing_neighbors: &[(IdType, T::DistanceType)],
    new_candidate: (IdType, T::DistanceType),
    max_neighbors: usize,
) -> Vec<IdType> {
    if existing_neighbors.len() < max_neighbors {
        // Simply add if below capacity
        let mut result: Vec<_> = existing_neighbors.iter().map(|(id, _)| *id).collect();
        result.push(new_candidate.0);
        return result;
    }

    // Check if new candidate should replace worst existing neighbor
    let worst_existing = existing_neighbors
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    if new_candidate.1 < worst_existing.1 {
        // Replace worst with new candidate
        existing_neighbors
            .iter()
            .filter(|(id, _)| *id != worst_existing.0)
            .map(|(id, _)| *id)
            .chain(std::iter::once(new_candidate.0))
            .collect()
    } else {
        // Keep existing neighbors
        existing_neighbors.iter().map(|(id, _)| *id).collect()
    }
}
```

### 6.3 Lazy Re-linking

Defer expensive neighbor updates to background thread:

```rust
// rust/vecsim/src/index/hnsw/lazy_relink.rs

pub struct LazyRelinkQueue {
    /// Queue of pending relink operations
    pending: Mutex<VecDeque<RelinkTask>>,
    /// Background worker handle
    worker: Option<JoinHandle<()>>,
}

struct RelinkTask {
    node_id: IdType,
    level: usize,
    new_neighbors: Vec<IdType>,
}

impl LazyRelinkQueue {
    /// Schedule a relink operation (non-blocking)
    pub fn schedule_relink(&self, node_id: IdType, level: usize, neighbors: Vec<IdType>) {
        self.pending.lock().push_back(RelinkTask {
            node_id,
            level,
            new_neighbors: neighbors,
        });
    }

    /// Process pending relinks in background
    fn process_pending(&self, graph: &ConcurrentGraph) {
        while let Some(task) = self.pending.lock().pop_front() {
            if let Some(element) = graph.get(task.node_id) {
                let _lock = element.lock.lock();
                element.set_neighbors(task.level, &task.new_neighbors);
            }
        }
    }
}
```

---

## Summary: Prioritized Implementation Roadmap

| Optimization | Impact | Complexity | Priority |
|--------------|--------|------------|----------|
| Adaptive Prefetch | 5-15% | Easy | High |
| Early Termination | 5-10% | Easy | High |
| Batch Distance Computation | 15-30% | Medium | High |
| Compact Memory Layout | 10-25% | Hard | Medium |
| Parallel Index Building | 3-8x build | Medium | Medium |
| Dynamic ef Adjustment | Quality++ | Easy | Medium |
| Lock-Free Read Path | 2-4x QPS | Hard | Low |
| PQ Integration | 2-4x memory | Hard | Low |
| Lazy Re-linking | 10-20% build | Medium | Low |

### Recommended Implementation Order

1. **Quick wins** (1-2 days each):
   - Adaptive prefetch parameters
   - Early termination heuristics
   - Dynamic ef adjustment

2. **Medium effort** (1-2 weeks each):
   - Batch distance computation
   - Parallel index building improvements

3. **Major refactoring** (2-4 weeks each):
   - Compact memory layout
   - PQ integration
   - Lock-free concurrent graph


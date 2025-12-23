# HNSWDisk Multi-Threading Architecture

## Overview

This document describes the architectural changes introduced in the `dorer-disk-poc-add-delete-mt` branch compared to the original `disk-poc` branch. The focus is on multi-threading, synchronization, concurrency in writing to disk, and performance enhancements.

## Key Architectural Changes

### 1. Insertion Mode

**Previous single threaded approach:** Vectors were accumulated in batches before being written to disk, requiring complex coordination between threads.


**Current approach:** Each insert job is self-contained and can write directly to disk upon completion, optimized for workloads where disk writes are cheap but neighbor searching (reads from disk) is the bottleneck.

```
┌──────────────────────────────────────────────────────────────┐
│                    HNSWDiskSingleInsertJob                   │
├──────────────────────────────────────────────────────────────┤
│ - vectorId                                                   │
│ - elementMaxLevel                                            │
│ - rawVectorData (copied into job - no external references)   │
│ - processedVectorData (quantized vector for distance calc)   │
└──────────────────────────────────────────────────────────────┘
```

### 2. Segmented Neighbor Cache

To reduce lock contention in multi-threaded scenarios, the neighbor changes cache is partitioned into **64 independent segments**:

```cpp
static constexpr size_t NUM_CACHE_SEGMENTS = 64;  // Power of 2 for efficient modulo

struct alignas(64) CacheSegment {
    std::shared_mutex guard;                              // Per-segment lock
    std::unordered_map<uint64_t, std::vector<idType>> cache;  // Neighbor lists
    std::unordered_set<uint64_t> dirty;                   // Nodes needing disk write
    std::unordered_set<uint64_t> newNodes;                // Nodes never written to disk
};
```
Note:
NUM_CACHE_SEGMENTS can be changes which will be cause better separation of the cache,
but will require more RAM usage.

**Key benefits:**
- Threads accessing different segments proceed in parallel.
- Cache-line alignment (`alignas(64)`) prevents false sharing.
- Hash-based segment distribution.

### 3. Lock Hierarchy

| Lock | Type | Protects | Notes |
|------|------|----------|------|
| `indexDataGuard` | `shared_mutex` | `entrypointNode`, `maxLevel`, `idToMetaData`, `labelToIdMap` | Metadata access during graph construction |
| `vectorsGuard` | `shared_mutex` | Vectors container (prevents resize during access) | Vectors access during graph construction |
| `rawVectorsGuard` | `shared_mutex` | `rawVectorsInRAM` map | Raw vectors access during graph construction |
| `stagedUpdatesGuard` | `shared_mutex` | Staged graph updates for deletions | 
| `diskWriteGuard` | `mutex` | Serializes global flush operations |
| `cacheSegments_[i].guard` | `shared_mutex` | Per-segment cache, dirty set, newNodes |

### 4. Atomic Variables

```cpp
std::atomic<size_t> curElementCount;      // Thread-safe element counting
std::atomic<size_t> totalDirtyCount_{0};  // Fast threshold check without locking
std::atomic<size_t> pendingSingleInsertJobs_{0};  // Track pending async jobs
```
Note:
We can think of more atomics that can be added to further improve performance.

## Concurrency Patterns

### Swap-and-Flush Pattern (Critical for Preventing Lost Updates)

The `writeDirtyNodesToDisk` and `flushDirtyNodesToDisk` methods implement a two-phase approach:

```
Phase 1: Read cache data under SHARED lock (concurrent writers allowed)
         - Read neighbor lists
         - Build RocksDB WriteBatch

Phase 2: Clear dirty flags under EXCLUSIVE lock (brief, per-segment)
         - Atomically swap dirty set contents
         - Release lock immediately

Phase 3: Write to disk (NO locks held)
         - Any new inserts after Phase 2 will re-add to dirty set
         - On failure: re-add nodes to dirty set for retry
```

This prevents the "Lost Update" race condition where a concurrent insert's changes could be lost.

### Atomic Check-and-Add for Neighbor Lists

```cpp
bool tryAddNeighborToCacheIfCapacity(nodeId, level, newNeighborId, maxCapacity) {
    // Under exclusive lock:
    // 1. Check if already present (avoid duplicates)
    // 2. Check capacity
    // 3. If space available: add neighbor, mark dirty
    // 4. If full: return false (caller must use heuristic)
}
```

This atomic operation prevents race conditions when multiple threads add neighbors to the same node.

### Read-Copy-Update Pattern for Cache Access

```cpp
// Read path (getNeighborsFromCache):
1. Try shared lock → read from cache
2. If miss: read from disk (no lock held during I/O) - rocksdb is thread-safe
3. Acquire exclusive lock → insert to cache (double-check)

// Write path (addNeighborToCache):
1. Acquire exclusive lock
2. Load from disk if needed (release/re-acquire around I/O) - rocksdb is thread-safe
3. Add neighbor, mark dirty
```

## Disk Write Batching

Controlled by `diskWriteBatchThreshold`:

| Value | Behavior |
|-------|----------|
| `0` | Flush after every insert (no batching) |
| `>0` | Accumulate until `totalDirtyCount_ >= threshold`, then flush |

```cpp
if (diskWriteBatchThreshold == 0) {
    writeDirtyNodesToDisk(modifiedNodes, rawVectorData, vectorId);
} else if (totalDirtyCount_ >= diskWriteBatchThreshold) {
    flushDirtyNodesToDisk();  // Flush ALL dirty nodes
}
```

## Job Queue Integration

### Async Processing Flow

```
addVector()
    │
    ├── Single-threaded path (no job queue):
    │   └── executeGraphInsertionCore() [inline]
    │
    └── Multi-threaded path (with job queue):
        ├── Create HNSWDiskSingleInsertJob (copies vector data)
        ├── Submit via SubmitJobsToQueue callback
        └── Worker thread executes:
            └── executeSingleInsertJob()
                └── executeGraphInsertionCore()
```

### Job Structure

```cpp
struct HNSWDiskSingleInsertJob : public AsyncJob {
    idType vectorId;
    size_t elementMaxLevel;
    std::string rawVectorData;       // Copied - no external references
    std::string processedVectorData; // Quantized data for distance calc
};
```

Copying vector data into the job eliminates race conditions with the caller's buffer.

## Data Flow During Insert

```
1. addVector()
   ├── Atomically allocate vectorId (curElementCount.fetch_add)
   ├── Store raw vector in rawVectorsInRAM (for other jobs to access)
   ├── Preprocess vector (quantization)
   ├── Store processed vector in vectors container
   └── Store metadata (label, topLevel)

2. executeGraphInsertionCore()
   ├── insertElementToGraph()
   │   ├── greedySearchLevel() [levels > element_max_level]
   │   └── For each level:
   │       ├── searchLayer() → find neighbors
   │       └── mutuallyConnectNewElement()
   │           ├── setNeighborsInCache() [new node's neighbors]
   │           └── For each neighbor:
   │               ├── tryAddNeighborToCacheIfCapacity()
   │               └── or revisitNeighborConnections() [if full]
   └── Handle disk write (based on threshold)

3. Disk Write (when triggered)
   ├── writeDirtyNodesToDisk() [per-insert path]
   └── or flushDirtyNodesToDisk() [batch flush path]
```

## Performance Optimizations

### 1. Lock-Free Hot Paths

```cpp
// Fast deleted check without acquiring indexDataGuard
template <Flags FLAG>
bool isMarkedAsUnsafe(idType internalId) const {
    return __atomic_load_n(&idToMetaData[internalId].flags, 0) & FLAG;
}
```

Used in `processCandidate()` to avoid lock contention during search.

### 2. Atomic Counters for Fast Threshold Checks

```cpp
// No locking needed to check if flush is required
if (totalDirtyCount_.load(std::memory_order_relaxed) >= diskWriteBatchThreshold) {
    flushDirtyNodesToDisk();
}
```

### 3. newNodes Tracking

Nodes created in the current batch are tracked in `cacheSegment.newNodes`:
- Avoids disk lookups for vectors that haven't been written yet
- Cleared after successful flush to disk

### 4. Raw Vectors in RAM

Raw vectors are kept in `rawVectorsInRAM` until flushed to disk:
- Allows concurrent jobs to access vectors before disk write
- Eliminates redundant disk reads during graph construction
- Protected by `rawVectorsGuard` (shared_mutex)

## Thread Safety Summary

| Operation | Thread Safety | Notes |
|-----------|---------------|-------|
| `addVector()` | ✅ Safe | Atomic ID allocation, locked metadata access |
| `topKQuery()` | ✅ Safe | Read-only with lock-free deleted checks |
| Cache read | ✅ Safe | Shared lock per segment |
| Cache write | ✅ Safe | Exclusive lock per segment |
| Disk flush | ✅ Safe | Swap-and-flush pattern |

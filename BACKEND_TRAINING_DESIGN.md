# Tiered SQ8 accumulation with an always-present backend

This alternative targets PR #1029 at `5079b7a7`. Its changes belong in #1029;
PR #1035 can remain unchanged.

## Ownership

```text
TieredHNSWIndex<DataType, DistType>
  |-- FLAT frontend: buffered, full-precision vectors
  |-- pending insertion jobs: migration scheduling
  |-- optional SQAccumulationState: threshold and double running sum
  `-- HNSWIndex<DataType, DistType>: always exists
        |-- existing single/multi implementation
        |-- SQ8 preprocessor: mean values
        `-- distance calculator: mean sum of squares
```

The existing HNSW classes and templates serve both NONE and SQ8. The factory selects
quantization components using `quantType`. Tiered owns accumulation because it owns the
FLAT vectors and decides when they migrate. HNSW receives only the calculated mean through
`setQuantizationMean()`; each component exposes a small setter for its own data.

## Where state is checked

There are two accumulation checks, at the entry to tiered `addVector()` and `deleteVector()`.
Within the accumulation add path, FLAT's vector count is compared with the threshold.
Reads, statistics, GC, and batch iterators use the backend normally:

```cpp
size_t getNumMarkedDeleted() const override {
    return getHNSWIndex()->getNumMarkedDeleted();
}
```

An empty backend naturally returns zero. Its pointer stays unchanged throughout the index's
lifetime. No training query or backend publication state is needed in the shared tiered base.

## Creation and transition

| Configuration | Backend at construction | Tiered accumulation state |
| --- | --- | --- |
| `VecSimQuant_NONE` | Ordinary HNSW | Absent, regardless of threshold |
| SQ8, threshold 0, no mean | SQ8 without centering | Absent |
| SQ8, threshold 0, supplied mean | SQ8 with supplied mean | Absent |
| SQ8, positive threshold | Empty SQ8 with its final mean layout, initially zero | Sum and threshold |

A positive threshold overrides a supplied mean and is capped at
`MAX_QUANT_NORMALIZATION_SET_SIZE`. The tiered factory supplies an initial zero mean through
the existing HNSW factory API. Memory estimates include the empty backend and temporary sum.

During accumulation, additions stay in FLAT even with write-in-place mode or a full buffer.
The running sum uses FLAT's stored values, so cosine normalization happens before summing.
FP16 values are widened to FP32 and accumulated into doubles. Single-value overwrites subtract
the old vector first; multi-value additions count vectors, not labels. Deletes subtract all
vectors for the label and update pending job IDs after FLAT swaps.

At the threshold, `calculateQuantizationMean()` in tiered divides the sum by FLAT's vector
count. Finalization snapshots the jobs, takes the exclusive main lock, installs the mean in
HNSW, and clears the optional accumulation state. It then submits the jobs, or executes them
synchronously in write-in-place mode. Deleting every vector later does not restart accumulation.

## Synchronization

The existing tiered contract serializes writers. No background insertion job is submitted
before finalization. Queries access backend components under the shared main lock, so the
exclusive finalization lock protects the mean update. Installing the mean does not allocate
or replace components; both cached distance dispatches retain the same calculator context.
Stored-to-stored IP requires updating that context's mean sum of squares as well as the
preprocessor's mean.

Batch iterators retain the main lock while their backend iterator is live. Empty backend
iterators are destroyed before unlocking; reset creates a fresh iterator. SQ8 `getDataByLabel`
returns no values during accumulation and after migration.

## Review order

1. [TieredHNSWIndex](src/VecSim/algorithms/hnsw/hnsw_tiered.h): state, accumulation writes, mean calculation, and migration.
2. [HNSWIndex](src/VecSim/algorithms/hnsw/hnsw.h): installing the mean in existing components.
3. [Tiered factory](src/VecSim/index_factories/tiered_factory.cpp): eager construction and memory estimates.
4. [Shared tiered base](src/VecSim/vec_sim_tiered_index.h): ordinary backend access.
5. [Tiered tests](tests/unit/test_hnsw_tiered.cpp) and [SQ8 tests](tests/unit/test_hnsw_sq8.cpp): transition and query coverage.

## Validation

On x86_64 with GCC 13.3 and SVS v0.3.2 enabled, all 529 selected Debug tests passed:
`test_hnsw` (341), `test_hnsw_sq8` (129), `test_components` (52), and `test_allocator` (7).
Coverage includes NONE ignoring the threshold, exact initial memory estimates, overwrites and
deletes during accumulation, mean/calculator agreement, component stability, migration,
iterator transitions, and concurrent queries.

The Release `VectorSimilarity` library also built with `VECSIM_BUILD_TESTS=OFF`.
Changed-line formatting and `git diff --check` passed.

Set `ROOT` to the source directory when running the tests; serialization tests require it.

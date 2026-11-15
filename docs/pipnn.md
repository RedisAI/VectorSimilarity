# PiPNN backend

PiPNN is an opt-in, explicitly built graph index inspired by
[the PiPNN paper](https://arxiv.org/abs/2602.21247).
It does not replace HNSW or change existing index defaults.

This implementation supports finite FLOAT32 vectors, squared L2 distance, and
one vector per label. Other data types, metrics, multi-value indexes, and tiered
indexes are rejected. It is a VectorSimilarity library backend; this change
does not add a Redis Search schema option.

## Lifecycle

```cpp
VecSimParams params{};
params.algo = VecSimAlgo_PIPNN;
params.algoParams.pipnnParams.type = VecSimType_FLOAT32;
params.algoParams.pipnnParams.dim = 128;
params.algoParams.pipnnParams.metric = VecSimMetric_L2;
VecSimIndex *index = VecSimIndex_New(&params);
// Check index != nullptr; insert 128-component vectors with VecSimIndex_AddVector.
// Check insertion return values: 1 inserted, 0 replaced, -1 failed.
if (index && VecSimIndex_BuildGraph(index)) {
    // TopKQuery now searches the graph. Check replies for nullptr before use.
}
if (index) VecSimIndex_Free(index);
```

Before building, TopK queries use exact search. Successful insertions,
overwrites, deletions, updates, and relabels invalidate the graph. Queries then
use exact search until the caller explicitly invokes `VecSimIndex_BuildGraph`.
No query rebuilds the index. A failed build leaves vector storage available
for exact queries. `VecSimIndex_IsGraphReady` exposes the current state.
An allocation failure during a mutation can also invalidate the graph while
preserving stored vectors and labels; check readiness after a failed operation.

Range queries and batch iteration always use exact search. A dirty index can
therefore have substantially higher query latency than a built graph. This
lifecycle is intended for bulk ingestion followed by read-heavy operation;
frequent writes need an explicit rebuild policy at the integration layer.

Callers must serialize **all** operations on an index, including queries:
inherited search-mode bookkeeping is mutable. Do not mutate or free an index
while a batch iterator is live. There is no background builder or tiered
integration in this version.

## Construction and search

The builder partitions vectors using sampled leaders and overlapping buckets,
collects two local neighbors per point per leaf, merges candidates by residual
direction hashes, and applies distance-based pruning. Candidate reservoirs and
final graph rows have explicit size limits. A directed successor cycle reserves
one edge per vertex to preserve graph reachability, including duplicate data.

This is a native implementation, not a vendored copy of the paper's software.
It uses existing VectorSimilarity distance kernels, Gaussian projections,
overlap at the first two partition levels, bounded recursion, and deterministic
fallback leaves for non-shrinking partitions. It currently has no GEMM-based
leaf kernel or parallel construction. These choices require independent recall,
latency, and build-time evaluation; upstream PoC measurements do not establish
this backend's performance. A fixed seed reproduces construction within the same
standard-library and numerical environment; cross-platform bitwise identity is
not promised.

TopK uses bounded beam search. Set `pipnnRuntimeParams.efRuntime` in
`VecSimQueryParams`, or resolve `EF_RUNTIME` through the existing parameter API.
Higher values usually improve recall at increased query cost. Graph results
are approximate; exact fallback, range, and batch results are exhaustive subject
to timeout. Timeouts use the existing query reply code.

## Parameters

Zero selects the default for optional fields, including seed.

| Parameter | Default | Constraint |
| --- | ---: | --- |
| `maxDegree` | 32 | At least 2 |
| `leafSize` | 128 | At least 2 |
| `fanout` | 2 | 2–32 |
| `hashBits` | 12 | 1–32 |
| `candidateLimit` | 80 | At least `maxDegree` |
| `alpha` | 1.1 | Finite, at least 1 |
| `efRuntime` | 100 | Positive after default resolution |
| `seed` | 100 | Nonzero to override |
| `blockSize` | 1024 | Positive after default resolution |

Internal IDs and count are bounded by the existing 32-bit ID representation.
Parameter-dependent allocation products are checked. Memory estimates include
the final graph but not peak build scratch: candidate storage is proportional
to `count * min(candidateLimit, count - 1)`, with additional projection and
partition storage. Large fanout and leaf sizes can substantially increase work.

## Persistence

`VecSimIndex_SavePiPNN` writes a versioned little-endian file containing resolved
parameters, labels, vectors, graph readiness, and graph data. It writes a
temporary file in the destination directory, flushes it, and renames it over
the destination only after successful writing. Parent-directory fsync is not
performed, so this is not a promise of rename durability after power loss.

`VecSimIndex_LoadPiPNN` validates lengths, parameters, finite vectors, unique
labels, graph bounds, duplicate edges, connectivity edges, and a corruption
checksum. It returns null on failure. The checksum is not authentication.
Dirty indexes stay dirty after loading; built indexes retain their graph.
Files are specific to this backend and cannot be loaded as HNSW files.

## Validation

Build and run `test_pipnn` with the repository's normal CMake test configuration.
The suite covers exact fallback, graph recall, mutations, label width, range and
batch queries, timeouts, invalid inputs, duplicate vectors, allocation cleanup,
and persistence corruption. Production rollout also requires workload-specific
recall/latency/build-memory validation and integration tests in the embedding
application.

ARM validation on 2026-11-15 passed 245 selected Common, BruteForce, Allocator,
and PiPNN CTest cases with GCC 12.3 and SVS enabled. All 25 PiPNN cases also
passed Clang 21 ASan/UBSan with leak detection. Function-type sanitization was
disabled because the existing CPU-feature C/C++ dispatch triggers that check.
Clang was used after GCC 12 hit a sanitizer compiler error in an existing BF16
SVE kernel.
The SVS-enabled GCC Release library also builds with both `VECSIM_BUILD_TESTS`
and `VECSIM_BUILD_TEST_HELPERS` disabled.

The [initial native ARM screen](benchmarks/pipnn/README.md) found faster builds but
slower queries than HNSW at comparable recall on both tested datasets. It does not
establish this backend as a production HNSW replacement.

Build the optional native comparison runner with
`cmake --build <build-dir> --target pipnn_comparison -j4`, then run:

```sh
<build-dir>/benchmark/pipnn_comparison base.fbin query.fbin truth.ibin pipnn 50,100,200,400,800 1000
<build-dir>/benchmark/pipnn_comparison base.fbin query.fbin truth.ibin hnsw 50,100,200,400,800 1000
```

The matrix files have two little-endian uint32 dimensions followed by row-major
FLOAT32 vectors or uint32 ground-truth IDs. Ground truth must contain at least
10 neighbors per query. The runner uses PiPNN defaults and HNSW M=16,
efConstruction=100, seed=100. Build timing includes index creation, insertion,
and PiPNN graph construction. Peak build RSS includes the input matrix; index
bytes come from allocator accounting. Compare latency at matched recall, not
merely at the same ef value. Run each backend in a separate process on an idle
machine and repeat before making a performance decision.

# Backend-owned SQ8 training

This is an alternative implementation on top of PR #1029 at `5079b7a7`, whose base is
PR #1035 at `4c2f7509`. All changes belong in #1029. #1035 can remain unchanged.

## Objects and responsibilities

```text
TieredHNSWIndex<DataType, DistType>           existing class for NONE and SQ8
  |-- FLAT frontend                        owns the buffered full-precision vectors
  |-- insertion jobs                       tiered owns scheduling and migration
  `-- HNSWIndex<DataType, DistType>          always exists; existing single/multi subclasses
        |-- preprocessor                   final storage/query layout from construction
        |-- distance calculator            cached dispatch context has a stable address
        `-- optional QuantizationTrainer
              `-- SQ8QuantizationTrainer   running sum, vector count, threshold, mean calculation
```

There is no new quantized tiered subclass, dummy graph, backend pointer replacement, or
publication atomic. The trainer is owned by HNSW; it references the existing SQ8 components.
The small trainer interface allows a later quantizer to implement its own collection and
finalization logic.

## Creation

| Configuration | Backend components | Trainer |
| --- | --- | --- |
| `VecSimQuant_NONE` | Ordinary HNSW | Absent |
| SQ8, threshold 0, no mean | SQ8 without centering | Absent |
| SQ8, threshold 0, supplied mean | SQ8 with supplied mean | Absent |
| SQ8, positive threshold | SQ8 with mean storage allocated and values initially zero | SQ8 trainer |

A positive training threshold takes precedence over a supplied mean, preserving #1029's
parameter behavior. The threshold is capped at `MAX_QUANT_NORMALIZATION_SET_SIZE`.
Memory estimates include the empty backend, its mean, and the temporary trainer and sum.

## Write flow

`TieredHNSWIndex::addVector` checks `backend->needsTraining()` once at entry. When false,
it follows the existing insertion path. When true:

1. Hold the FLAT lock and insert into FLAT, even with write-in-place mode or a full buffer.
2. On a single-value overwrite, report removal of the old stored vector to the trainer first.
   Reuse its unsubmitted insertion job.
3. Report the new FLAT-stored vector to the trainer. Cosine vectors have already been
   normalized by FLAT; FP16 values are widened before accumulating into a double sum.
4. Keep the insertion job in the tiered job map without submitting it.
5. If the trainer is ready, snapshot the pending jobs, acquire the exclusive tiered main lock,
   and ask HNSW to finalize training.
6. The SQ8 trainer computes `mean[d] = runningSum[d] / vectorCount`, writes it into the
   existing preprocessor, and updates the calculator's `mean_sum_squares` in place.
7. HNSW destroys the trainer. Release the main lock, then submit the saved jobs, or execute
   them synchronously in write-in-place mode.

`deleteVector` has the other `needsTraining()` check. During training it subtracts every
stored vector for the label, deletes its unsubmitted jobs, removes the FLAT data, and fixes
the remaining jobs' internal IDs after FLAT swaps. Relabeling changes labels and job labels;
it does not change the sum or vector count.

Deleting all vectors after training does not create a new trainer. An empty graph and an
untrained quantizer are different states.

## Reads and synchronization

Reads never ask whether training is needed. For example:

```cpp
size_t getNumMarkedDeleted() const override {
    return getHNSWIndex()->getNumMarkedDeleted();
}
```

During training the real empty backend returns zero. Top-k and range queries merge FLAT
results with an empty backend result using the ordinary tiered paths. Debug information
reports the actual SQ8 backend. SQ8 `getDataByLabel` reports no values in every phase.

The existing tiered contract serializes add/delete writers. Only that writer accesses the
trainer. No insertion job reaches the backend before finalization. Queries access backend
components while holding the shared main lock; finalization holds it exclusively. Batch
iterators retain that lock while an HNSW iterator remains live. An empty backend iterator
is destroyed before releasing the lock, and reset creates a new iterator.

Mean finalization allocates no memory and requires an empty graph. Both cached distance
dispatches continue to point at the same calculator context. In particular, updating only
the preprocessor would be incorrect for stored-to-stored IP distances.

## Review order

1. [QuantizationTrainer](src/VecSim/spaces/computer/quantization_trainer.h): notification/finalization interface.
2. [SQ8QuantizationTrainer](src/VecSim/spaces/computer/sq8_quantization_trainer.h): complete SQ8 math and state.
3. [HNSWIndex](src/VecSim/algorithms/hnsw/hnsw.h): ownership and one-time finalization.
4. [TieredHNSWIndex](src/VecSim/algorithms/hnsw/hnsw_tiered.h): training writes, deferred jobs, migration.
5. [Component factory](src/VecSim/index_factories/components/components_factory.h),
   [HNSW factory](src/VecSim/index_factories/hnsw_factory.cpp), and
   [tiered factory](src/VecSim/index_factories/tiered_factory.cpp): eager construction and memory estimates.
6. [VecSimTieredIndex](src/VecSim/vec_sim_tiered_index.h): removal of nullable-backend read paths.
7. [Tiered lifecycle tests](tests/unit/test_hnsw_tiered.cpp) and
   [SQ8 tests](tests/unit/test_hnsw_sq8.cpp): lifecycle and regressions.

The cost is allocating the empty backend from the start and retaining one optional trainer
pointer in HNSW. Ordinary HNSW writes make the two lifecycle entry checks; read paths incur
no training-state checks. The shared SVS retrieval behavior from #1029 is preserved.

## Quantization type versus training state

HNSW retains the existing `isQuantized` flag exposed through `usesQuantizedStorage()`.
The factory sets it when creating SQ8 components. Finishing training leaves that flag true
and consumes only the trainer, so `needsTraining()` becomes false.

This implementation specializes the preprocessor and trainer, without adding a `QuantType`
parameter to the whole HNSW class. Such a parameter could remove quantization branches in
specialized graph code, but eliminating tiered branches would also require carrying that
specialization into the tiered layer. SQ8's transition from collecting to trained would
still require runtime state. SVS specializes its backend storage types, while its tiered
wrapper remains templated only on `DataType`.

## Validation

Validated on x86_64 with GCC 13.3, SVS enabled, and the v0.3.2 dependency.

| Debug test executable | Passed |
| --- | ---: |
| `test_hnsw` | 339 |
| `test_hnsw_sq8` | 129 |
| `test_components` | 52 |
| `test_allocator` | 7 |
| Total | 527 |

The Release `VectorSimilarity` library also built with `VECSIM_BUILD_TESTS=OFF`.
Changed-line formatting and `git diff --check` passed.

Run the test executables with `ROOT` pointing at the source directory; the serialization
tests require that environment variable. For example, from the source directory:

```sh
ROOT="$PWD" build/unit_tests/test_hnsw
ROOT="$PWD" build/unit_tests/test_hnsw_sq8
ROOT="$PWD" build/unit_tests/test_components
ROOT="$PWD" build/unit_tests/test_allocator
```

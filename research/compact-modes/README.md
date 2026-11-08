# TQ2/TQ3 HNSW integration

This draft adds smaller total HNSW index choices, with recall and query latency
as explicit tradeoffs. It is based on PR #1101 (`codex/quant-storage-type`,
`217026b84aa123ae3bc0f483998a0a85b9d47b25`). It does not include SQ4 from
#1099/#1103 and does not change the default quantization mode.

## Shared StorageType pipeline

SQ8, TQ2 and TQ3 use the same `CreateQuantizedIndexComponents`,
`QuantPreprocessor` and `QuantizedCalculator`. A StorageType selects its typed
context, which owns encoding, prepared queries, distance dispatch, validation,
mean installation and fitted-state persistence. The scalar-family default
context retains the existing SQ8 arithmetic and SIMD selection; TQ contexts
retain the current rotation, packed-code scoring and correction policies.

HNSW and tiered code have no TQ flags, casts or codec access. Tiered construction
requests a backend prepared for regular mean training without interpreting
quantizer parameters. The backend forwards mean publication and validation to
its storage context. Scoring still uses the cached distance dispatch, without
adding a per-candidate virtual call. Components own the shared context; the
index retains a non-owning alias. Context/control allocations are counted in
the memory estimator, and allocated query/storage blobs retain their allocator.

This common foundation belongs in #1101. The TQ-specific layer adds StorageType
registration, contexts, codec/kernels and their tests. SQ4's existing scalar
layout/code hooks can continue through the scalar-family default context.

## One regular training phase

TQ2/TQ3 reuse the existing tiered quantization training phase:

1. Initial vectors stay in the ordinary FP32 FLAT buffer. The existing accumulator
   tracks the sum and count of live, preprocessed rows, including updates/deletes.
2. At `QuantNormalizationSetSize`, it calculates one mean vector (the average).
3. The backend installs that mean, then buffered vectors migrate into HNSW.

There is no separate TQ training state, accumulator, training-count metadata or
`isTurboQuantTraining()` API. The tiered layer follows one accumulation and
finalization path for SQ8 and TQ. Only installation of the mean is codec-specific
inside the backend. Job callbacks are prepared before publishing the mean, and
failed preparation leaves the existing accumulator available for retry.
TQ's one-time mean installation allocates its fitted buffers and rotates the mean
under the exclusive backend guard, so queries can wait during that finalization.

This training learns the mean only. The rotation seed remains fixed (42 by
default); training does not search for seeds, learn a rotation or fit code levels.
After migration the mean stays frozen. Standalone HNSW does not train implicitly;
a zero threshold also skips tiered training and uses the construction-time mean.

## Representation and API

TQ2/TQ3 support FP32 IP and cosine. Each record has packed 2/3-bit codes plus two
FP32 values (8 bytes) for reconstruction scale and mean-error correction. The
dense orthogonal rotation and scalar centroids are shared. Projection-match
scaling and mean-error correction are experimental extensions to the baseline
TurboQuant estimator, not an unchanged implementation of the paper.

```cpp
VecSimTurboQuantParams tq{.mean = nullptr, .seed = 42};
HNSWParams params{.type = VecSimType_FLOAT32,
                  .dim = 768,
                  .metric = VecSimMetric_IP,
                  .quantType = VecSimQuant_TQ2,
                  .quantParams = &tq};
```

The factory copies configuration/mean values. Null `quantParams` selects zero
mean and seed 42; explicit seed zero is respected. Positive tiered training
thresholds replace any supplied mean with the mean from the regular training
phase. The tiered cosine frontend normalizes finite, nonzero inputs. Standalone
cosine creation/loading requires explicit `is_normalized=true` in the C++ factory
and normalized queries/inserts; the ordinary standalone factory rejects raw
cosine. IP also accepts zero. L2 and other element types are rejected.

TQ backend graph/state use encoding V6, preserving the actual rotation and mean.
NONE/SQ8 writers retain their existing formats. New-state and graph validation
precede graph loading. Backend serialization does not capture the tiered FLAT
buffer or pending jobs; saving an empty backend during accumulation is not a
snapshot of the whole tiered index. Serialization remains gated by the existing
test-helper build options. Enum value 2 is reserved for the independent SQ4 stack;
TQ2/TQ3 use 3/4.

## Complete costs

At dimension 768, TQ2/TQ3 vector records are 200/296 bytes. Total index memory
also includes graph links, labels, capacity, allocator headers, components and
the shared transform. Queries score packed codes without a retained FP32
reranking copy, but need FP32 query preparation and temporary workspace.

The dense transform costs O(d²) shared memory and query preparation, with cubic
initialization work. Regular training retains the existing FLAT buffer until
migration: 10,240 rows at dimension 768 alone have 30 MiB of FP32 payload. The
harness reports allocator totals, query/workspace payloads and selected build
peaks separately; these are not process RSS or system malloc bookkeeping.

## Build and check

Initialize the repository's existing dependencies, then:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVECSIM_BUILD_TESTS=ON
cmake --build build --target test_hnsw_turboquant bm_selected_modes
ROOT="$PWD" build/unit_tests/test_hnsw_turboquant
build/benchmark/bm_selected_modes --stage micro --codec tq2 --output /tmp/tq2-micro
```

Tests use normal CTest discovery. The bounded harness accepts `tq2` and `tq3`;
stages are `micro`, `tiny`, `representation` and `hnsw`. Output directories must
be new. Real-index stages use ordinary public tiered training/query/mutation
paths and backend reload checks. Exhaustive codec scans are diagnostic, not a
supported quantized FLAT product mode. Run micros, then a tiny index check,
before a bounded graph sweep.

The checkpoint fixture is a million-row FP32 DBpedia cosine HNSW-V3 file with
SHA-256 `b59f5311786cb48d7f697d11722a4042d953eb174254f0655851737d639cd379`.
Verify this checksum externally before/after execution; the driver records the
expected hash but does not hash the file itself. Queries begin at row 500,000.

## Before merge

[validation.md](validation.md) distinguishes current checks from historical
performance evidence. Remaining requirements are:

- Settle error/status translation for C and foreign callers. Invalid inputs can
  currently throw C++ exceptions through C-linkage entry points.
- Add Python selection/configuration and binding tests if required for release.
- Run sanitizer/x86 qualification, scalar/native equivalence and format checks.
- Qualify representative million-vector datasets with independent held-out
  queries, matched recall targets and full training/query/storage costs.
- Rebase onto #1101's merged successor and reconcile enum/format integration
  with the independent SQ4 stack.

Qualify TQ3 first, then TQ2 as the smaller 85–90% tier where validated.

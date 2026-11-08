# TQ2/TQ3 integration and optional RaBitQ1 prototype

This draft explores smaller **total HNSW indexes**, with recall and query latency
as explicit tradeoffs. It is based on PR #1101 (`codex/quant-storage-type`,
`217026b84aa123ae3bc0f483998a0a85b9d47b25`). It does not include the SQ4 changes
from #1099 or #1103 and does not change the default quantization mode.

## What is implemented

TQ2 and TQ3 are FP32 HNSW modes for inner product and cosine. They use an immutable
dense orthogonal rotation, fixed scalar centroids, projection-match reconstruction
scaling, and a mean-error correction. These last two policies are experimental
extensions to the baseline TurboQuant implementation; this is not an unmodified
implementation of the paper. Each stored vector has packed codes plus two FP32
values (8 bytes). Queries are prepared once and score packed codes directly;
there is no retained FP32 reranking copy.

The public C++ factory path uses `VecSimQuant_TQ2` or `VecSimQuant_TQ3` in
`HNSWParams::quantType`. `quantParams` may point to `VecSimTurboQuantParams`:

```cpp
VecSimTurboQuantParams tq{.mean = nullptr, .seed = 42};
HNSWParams params{.type = VecSimType_FLOAT32,
                  .dim = 768,
                  .metric = VecSimMetric_IP,
                  .quantType = VecSimQuant_TQ2,
                  .quantParams = &tq};
```

The factory copies configuration and mean values. Null `quantParams` chooses seed
42 and zero mean; an explicit seed of zero is respected. Standalone HNSW does not
train implicitly. Tiered HNSW with positive `QuantNormalizationSetSize` fits one
mean from live preprocessed FLAT rows, then freezes it before migration. Retries,
updates, multivalue labels, publication and asynchronous job preparation have
focused coverage. The tiered cosine frontend accepts raw finite, nonzero rows and
normalizes them. Standalone cosine construction and loading require the C++
factory's explicit `is_normalized=true` argument and normalized queries/inserts;
the ordinary standalone factory rejects raw cosine. IP also accepts zero. Only
FP32 is supported; L2 and other element types are rejected.

TQ state and graph use encoding V6, including the actual transform and fitted
state. Legacy NONE/SQ8 writers retain their versions. Loading validates the new
state and graph before passing them to the existing graph loader. Helpers and
serialization remain controlled by the repository's existing build options.
Enum value 2 is left free for the independent SQ4 branch; TQ2/TQ3 use 3/4.

RaBitQ1 is a separate RAM-only research adapter in namespace
`vecsim::experimental`. It has no public quantization enum, tiered training
contract or persistence. It accepts externally normalized FP32 IP vectors,
dimension 64..65536, pads to a multiple of 32, and uses the official seeded
FHT/Kac rotation and encoder/scorer. Its 1-bit record contains sign codes and
three FP32 factors (12 bytes). The common adapter also has 2/3-bit test coverage;
those are not proposed public modes here.

## Costs that remain visible

At dimension 768, TQ2/TQ3 records are 200/296 bytes, and RaBitQ1 records are
108 bytes. These are **vector record sizes**, not total index memory. The graph,
labels, capacity, allocator headers, shared transforms and components all remain.
TQ's dense matrix takes O(d²) shared memory, O(d²) query preparation, and cubic
initialization work. A faster packed decoder does not remove those costs.

TQ preprocessing also needs a prepared FP32 query plus a correction value and
temporary FP32 workspace. Mean training retains the existing FLAT buffer until
migration; 10,240 vectors at dimension 768 alone occupy 30 MiB of FP32 payload.
RaBitQ has shared rotation signs, padded query state, encoding workspace and a
transient blob deleter allocation outside the VecSim allocator. The harness
reports these requested external payloads separately; system malloc overhead
and process RSS are not claimed to equal allocator totals.

## Build and run

TQ tests are registered with ordinary CTest discovery. RaBitQ stays off by default:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVECSIM_BUILD_TESTS=ON \
  -DVECSIM_RABITQ_RESEARCH=ON
cmake --build build --target test_hnsw_turboquant test_rabitq1 bm_selected_modes
ROOT="$PWD" build/unit_tests/test_hnsw_turboquant
ROOT="$PWD" build/unit_tests/test_rabitq1
build/benchmark/bm_selected_modes --stage micro --codec tq2 --output /tmp/tq2-micro
```

Initialize the repository's existing dependencies first. Optional RaBitQ requires
CMake 3.20 and downloads the official MIT-licensed library at commit
`4c91ef02bdfaa531b81fe679605acae31952d26a` using FetchContent. For offline builds,
set `FETCHCONTENT_SOURCE_DIR_RABITQ` to a verified checkout at that commit. Its
bundled Eigen has its own license notices. No modified SPLIT2 selector is used.
Index, FastScan, OpenMP, upstream tests/samples and native-machine tuning are off.
TQ itself introduces no RaBitQ dependency in a normal build.

The bounded harness accepts only `tq2`, `tq3`, and `rq1`. Its stages are `micro`,
`tiny`, `representation`, and `hnsw`; output directories must be new. Real-data
stages take a repository-format FP32 HNSW fixture via `--fixture` and compare
against exact FP32 top-k over the selected base rows. Run microchecks, then a
small real-index check, before a bounded graph sweep. The harness includes
native-versus-reference scoring, tail dimensions, accounting, mutation and
TQ reload checks. RaBitQ graphs use the research adapter, not a public factory.
The checkpoint fixture is the million-row DBpedia cosine HNSW-V3 file with
SHA-256 `b59f5311786cb48d7f697d11722a4042d953eb174254f0655851737d639cd379`.
Verify that checksum externally before and after a run; the driver records the
expected checksum but does not hash the file itself. Queries begin at row 500,000.

## Evidence and merge decisions

See [validation.md](validation.md) for the fresh port checks and the scope of the
earlier 100k evidence. Historical results are useful for deciding which options
to pursue, but do not qualify this port or a million-vector deployment.

Before making this ready to merge:

- Settle the public error contract. Invalid TQ rows currently throw C++ exceptions
  through entry points declared with C linkage (`VecSimIndex_AddVector` and query
  paths); construction can also throw. C++ tests exercise that behavior, but C or
  other foreign callers need an agreed translation/status policy. Do not advertise
  the draft as a safe foreign-language API until this is resolved.
- Add Python enum/configuration exposure and binding tests if Python selection is
  part of the release scope; it is absent here.
- Run sanitizer and x86 qualification, including scalar/native equivalence and
  serializer compatibility. The ARM check does not establish cross-platform
  reproducibility or portability.
- Decide whether to ship RaBitQ at all. Its optional tests need an enabled build
  job; default CI does not exercise that adapter. Review the dependency/license
  packaging and define persistence/training/public API only if promoting it.
- Confirm memory, recall and latency on representative million-vector datasets
  and independent held-out queries. Tune search effort per recall target and
  count graph, training peaks, workspace and all external allocations.
- Rebase onto #1101's merged successor and reconcile enum/format changes with
  the independent SQ4 stack before merge. Review the additional HNSW validation
  hooks and training ownership as part of that integration.

Recommended order: qualify **TQ3** as the less aggressive compact option, then
**TQ2** for the smaller 85–90% tier where the dataset supports it. Keep **RaBitQ1**
experimental until its lower measured recall is acceptable for a specific use.

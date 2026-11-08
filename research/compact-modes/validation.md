# Validation and historical evidence

## Current StorageType revision

Passed on user-selected `dorer-arm-big` (ARM64, Neoverse V2, 16-byte SVE),
with quiet-host/resource guards and four compile jobs:

- Release test/benchmark build and ordinary library build with tests/helpers disabled.
- TQ2/TQ3 repository microchecks, independent scalar references and packed tails.
  Query kernel: SVE2-128 with NEON fallback; stored-pair kernel: NEON.
- **2,007 unit tests:** TQ 27, existing SQ8 209, common 153, components 61,
  distance spaces 1,557. Coverage includes SQ8 byte/kernel equivalence, allocation
  estimates, failed combined preprocessing, buffer lifetime and cached dispatch after training.
- Two public DBpedia real-index checks, each 1,000 vectors at dimension 768 and
  five development queries, after the micro/unit gates. They cover shared training,
  public/diagnostic record, score and kernel parity, mutation, query and backend reload.
- Ten accepted stage receipts; no competing workload recorded. Source, dependency,
  test/benchmark binary and runtime manifests checked.
- The SQ8-only foundation on #1101 built separately and passed its 61 component
  and 209 SQ8 tests, with its own source/binary/runtime pins and quiet-host guards.

[qualification.json](qualification.json) records 29 source/build-file hashes,
stage hashes, counts and configuration. Source/build inputs match the native snapshot
byte-for-byte; merging the shared foundation did not change those bytes. The separate
foundation source export matches commit `6a60e408`. Documentation/receipts were finalized
afterward. Earlier revision receipts do not qualify this StorageType refactor.

Local archive: `storage-context-evidence.tar.gz`, SHA-256
`4fe094fc53c940eb260d07de7b7da0d310e0f803b497a8f39a17c4b415d9ed19`, preserved in
`storage-context-evidence-20261108/` beside the worktree. It contains commands,
logs, source exports, manifests and receipts. Native binaries/external dependency
bytes are not bundled; this is not a public CI download or complete offline replay package.

These are correctness and small-index checks, not a new 100k/million-vector
performance comparison, sanitizer/x86 run or foreign-language API qualification.
The one-time exclusive lock duration during mean installation was not measured.
Allocator totals include context/control allocations. Process RSS and libc
bookkeeping are excluded; helper-only SQ8 serialization keeps its legacy separate
mean mirror outside the VecSim allocator (one dimension-sized FP32 vector).

## Historical 100k checkpoint (different source tree)

Measured on the research implementation based on `fc096df5`, including the
independent SQ4/SIMD stack, before this port onto #1101: ARM Neoverse V2,
100,000 normalized 768-dimensional DBpedia vectors, 500 development queries,
cosine/IP, HNSW M=16 and efConstruction=100. Recall@10 uses exact FP32 top-k
within those 100k base rows. No FP32 reranking store or fitMemory was used.
Memory is the populated index allocator total; process RSS was not measured.

| Mode | Total index MiB | Recall@10 | Mean query ms | efRuntime |
|---|---:|---:|---:|---:|
| SQ4 reference | 67.672 | 87.82% | 0.164833 | 40 |
| TQ2, 10,240 training rows | 50.720 | 86.32% | 0.332501 | 80 |
| TQ3, 10,240 training rows | 60.021 | 88.70% | 0.258606 | 40 |
| TQ3, 10,240 training rows | 60.021 | 91.36% | 0.389152 | 80 |

These illustrate tradeoffs, not equal-recall speed comparisons. TQ2 saved about
25% total memory versus SQ4 at the shown 85–90% tier; TQ3 saved about 11%.
Both were slower than the SQ4 reference at those operating points. TQ's query
preparation micros were around 62–63 us; that is not a whole-query profile.

Queries served development/policy selection, not independent held-out evaluation.
This dataset/size does not establish million-vector, other-dataset or x86
behavior. The StorageType refactor preserves the intended mean calculation
and encoding policies, but these historical measurements are not new performance
results for its implementation.

The preserved historical archive is outside this PR:
`selected-modes-evidence-20261108.tar.gz`, SHA-256
`499ca3076b492997c3b62653ef4a9f8c10876ec1220ee74751d31256aed275d9`, originally in
the research worktree's `research/selected-modes-checkpoint/final/` directory.
Raw inputs, archives and binaries are not committed here. This is a local
research receipt, not a public download or an offline replay guarantee.

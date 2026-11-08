# Validation and historical evidence

## Current shared-training revision

Passed on user-selected `dorer-arm-big` (ARM64, Neoverse V2, 16-byte SVE),
with quiet-host/resource guards and four compile jobs:

- Release test/benchmark build and a separate ordinary library build with tests
  and test helpers disabled.
- TQ2/TQ3 repository micros, including independent scalar score references and
  packed-tail checks. Selected query kernel: SVE2-128 with NEON fallback;
  stored-pair kernel: NEON.
- **387 unit tests passed:** TQ 25, existing SQ8 209, common 153. The TQ callback failure/retry
  case also exercises SQ8, and query-publication/multivalue/update coverage remains.
- Two real-index smokes, each 1,000 DBpedia vectors and five development queries
  at dimension 768, after the micro/unit gates. They check ordinary public mean
  training, record/score parity, mutation, queries and backend reload.
- All eight stage receipts accepted; no competing workload recorded. Source,
  dependency, benchmark/test binary and runtime manifests were verified.

[qualification.json](qualification.json) records the 17 changed source/build-file
hashes, counts, kernel selection and stage hashes. All source/build inputs match
the native snapshot; documentation/receipt were finalized afterward. Earlier
revision receipts are superseded and do not qualify this refactor.

Local evidence: `shared-training-evidence.tar.gz`, SHA-256
`940470e2fa0eb08fa92e7c457c2d17574e979e8f95902573ee2108e20a00c138`, preserved beside the worktree in
`tq-shared-training-evidence-20261108/`. It contains commands, logs, manifests and
receipts, not all native binaries or external dependency bytes. It is not a
public CI download or a complete offline replay package.

This is correctness/smoke qualification, not a new 100k/million-vector performance
comparison, sanitizer/x86 run or foreign-language API qualification. The one-time
exclusive lock duration during TQ mean installation was not measured.

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
behavior. The shared-training refactor preserves the intended mean calculation
and encoding policies, but these historical measurements are not new performance
results for its implementation.

The preserved historical archive is outside this PR:
`selected-modes-evidence-20261108.tar.gz`, SHA-256
`499ca3076b492997c3b62653ef4a9f8c10876ec1220ee74751d31256aed275d9`, originally in
the research worktree's `research/selected-modes-checkpoint/final/` directory.
Raw inputs, archives and binaries are not committed here. This is a local
research receipt, not a public download or an offline replay guarantee.

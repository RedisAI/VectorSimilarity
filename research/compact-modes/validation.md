# Validation and historical evidence

## Fresh port qualification

Passed on the user-selected `dorer-arm-big` (ARM64, Neoverse V2, 16-byte SVE),
with four compile jobs, quiet-host checks and disk/effective-memory guards:

- Release build with tests and optional RaBitQ enabled; a separate ordinary
  library build with tests, test helpers and RaBitQ disabled.
- Repository codec micros for TQ2, TQ3 and RaBitQ1, including independent scalar
  query/graph score references and packed-tail checks. The TQ selector reported
  SVE2-128 query scoring with NEON fallback and NEON graph scoring.
- **395 unit tests passed:** TQ 25, RaBitQ 8, existing SQ8 209, common 153.
  TQ includes both new cosine reload normalization-contract regressions.
- Three real-index smoke cases, each 1,000 DBpedia vectors and five development
  queries at dimension 768, after the micro/unit gates. TQ uses public training,
  mutation, query and reload operations; RaBitQ uses its disclosed research graph.
- All 11 stage receipts accepted; no competing workload recorded. Exact source,
  dependency, binary and runtime hashes were checked by the qualification runner.

[qualification.json](qualification.json) records the 19 changed source/build-file
hashes, exact counts, selected kernels and stage receipt hashes. All those source
and build files still match the native snapshot; documentation and the receipt
were finalized afterward. The official RaBitQ dependency's 568 selected source,
header and CMake files match commit `4c91ef02bdfaa531b81fe679605acae31952d26a`
without research extensions. Offline qualification supplied that verified subset
via FetchContent's source override; automatic network fetching was not exercised.

The local `port-validation-evidence.tar.gz` archive has SHA-256
`ac0f51d2f7b1dc73ff03b6815033c003155fe015d2fc9666004b6ec05b271bc2` and is preserved
beside the PR worktree in `tq23-rabitq-draft-evidence-20261108/`. It contains commands,
logs, manifests and receipts, not all native binaries/external dependency bytes;
it is not a public CI download or a fully offline replay package.

These are port correctness/smoke checks, **not a fresh 100k or million-vector
performance comparison**. Sanitizers, x86, Python bindings and foreign-language
error handling remain unqualified.

## Historical 100k checkpoint (different source tree)

Measured on the research implementation based on `fc096df5` (including the
independent SQ4/SIMD stack), before this port onto #1101. ARM Neoverse V2;
100,000 normalized 768-dimensional DBpedia vectors; 500 development queries;
cosine/IP; HNSW M=16 and efConstruction=100. Recall is recall@10 against exact
FP32 top-k within those 100k base rows. No FP32 reranking store or fitMemory was
used. Memory is the populated index allocator total, with the research adapter's
requested external rotation payload included; process RSS is not measured here.

| Mode | Total index MiB | Recall@10 | Mean query ms | efRuntime |
|---|---:|---:|---:|---:|
| SQ4 reference | 67.672 | 87.82% | 0.164833 | 40 |
| TQ2, 10,240 training rows | 50.720 | 86.32% | 0.332501 | 80 |
| TQ3, 10,240 training rows | 60.021 | 88.70% | 0.258606 | 40 |
| TQ3, 10,240 training rows | 60.021 | 91.36% | 0.389152 | 80 |
| RaBitQ1 research | 39.330 | 75.72% | 0.426683 | 160 |

These are illustrative operating points, not equal-recall speed comparisons.
TQ2 saved about 25% total memory versus SQ4 at the shown 85–90% tier; TQ3 saved
about 11%. Both were slower than the SQ4 reference at these points. RaBitQ1 saved
about 42%, but did not reach 85% recall. Increasing search effort cannot be assumed
to repair its representation loss: exhaustive quantized scoring reached 76.10%
for RaBitQ1 in that checkpoint. Query-preparation micros were around 62–63 us for
TQ versus 4 us for RaBitQ; this does not attribute a full graph query profile.

The queries served development and policy selection, so this is not an independent
held-out evaluation. This dataset and size do not establish million-vector,
other-dataset or x86 behavior. Current optional RaBitQ uses the unmodified official
library; the earlier snapshot contained a separate unused 2-bit selector extension.

The full preserved historical evidence archive is outside this PR:
`selected-modes-evidence-20261108.tar.gz`, SHA-256
`499ca3076b492997c3b62653ef4a9f8c10876ec1220ee74751d31256aed275d9`.
Its original location is the research worktree's
`research/selected-modes-checkpoint/final/`. Raw inputs, archives and generated
binaries are deliberately not committed as source. The archive is a local research
receipt, not a remotely downloadable CI artifact or an offline replay guarantee.

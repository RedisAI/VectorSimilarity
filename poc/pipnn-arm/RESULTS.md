# PiPNN versus VectorSimilarity HNSW on ARM

Initial decision: continue investigating PiPNN for bulk construction, but do not
replace HNSW on this evidence. The low-dimensional workload is promising. The
high-dimensional workload trades faster construction for slower queries at 95%
recall, and the experiment does not cover online insertion/deletion.

## Scope and comparison boundaries

Measurements use the user-selected `dorer-arm-big` (Neoverse-V2), one build thread
and serial queries, FP32 L2, and 100,000 vectors per dataset. DEEP is 96D with
999 held-out queries; normalized DBpedia/OpenAI embeddings are 1536D with 1000
queries. The two implementations use exactly the same prepared inputs and
recomputed subset ground truth. See [methodology](README.md) and data manifests
under `evidence/data/` for exclusion and normalization details.

These are comparisons of specific builds, not isolated algorithm speedups:
PiPNN uses native architecture and fast-math flags; VecSim has its normal ARM
SIMD compilation without global fast-math. Its selected runtime distance function
was not instrumented. Exact revisions and flags are in [metadata](metadata.txt).

**Build seconds have different boundaries.** PiPNN measures the full builder
process, including input loading and graph serialization, but excludes the
subsequent graph/vector loading by the separate query process. HNSW measures
input loading and construction of an immediately queryable in-memory index,
without serialization. Ratios below are a construction-screening proxy, not
equivalent end-to-end ready-to-query speedups. Common offline normalization and
ground-truth computation are excluded from both.

Each configuration has two repeated runs with fixed/default randomness. These
measure timing repeatability, not variation across random graph seeds. DEEP's
PiPNN graph hashes match between repeats. Query tables use median run p95 at a
measured search effort that reaches the target recall in both runs; no recall
interpolation. Only search efforts present in both runs qualify. The first DEEP
M32 sweep was coarser, so this rule limits its candidate settings. Two runs are
not a confidence interval. Raw observations and ranges are in
[summary.json](evidence/summary.json).

## Measurements

Medians of two runs; memory is decimal MB. Peak is the maximum observed build
process RSS. Resident memory is the loaded query process before query/truth input.

| Dataset | Engine | Construction seconds* | p95 ≥95% recall (µs) | p95 ≥99% recall (µs) | Resident MB | Peak build MB |
|---|---|---:|---:|---:|---:|---:|
| deep100k | HNSW M16-efC200 | 23.77 | 90.1 | 190.8 | 77.6 | 117.8 |
| deep100k | HNSW M32-efC200 | 29.56 | 128.9 | 226.0 | 89.1 | 130.4 |
| deep100k | PiPNN | 11.38 | 71.5 | 158.0 | 74.5 | 291.8 |
| dbpedia100k | HNSW M16-efC200 | 119.23 | 633.5 | 2233.0 | 653.9 | 1268.9 |
| dbpedia100k | HNSW M32-efC200 | 147.27 | 794.5 | 2235.4 | 665.0 | 1280.8 |
| dbpedia100k | PiPNN | 65.08 | 765.4 | 2078.1 | 649.1 | 1232.3 |

*Construction boundaries differ as explained above.

Against M16, DEEP has a 2.09x construction-time ratio, with 20.6% lower p95
at ≥95% recall and 17.2% lower p95 at ≥99% recall. Its peak build RSS is 2.48x
HNSW (292 versus 118 MB). DBpedia has a 1.83x construction-time ratio, but p95
is 20.8% higher at ≥95% recall and 6.9% lower at ≥99% recall. Resident memory
is slightly lower with PiPNN on both datasets.

| Dataset / engine | Effort at ≥95% | Measured recall | Effort at ≥99% | Measured recall |
|---|---:|---:|---:|---:|
| dbpedia100k / hnsw M16-efC200 | 60 | 0.95750 | 300 | 0.99200 |
| dbpedia100k / hnsw M32-efC200 | 50 | 0.96190 | 200 | 0.99100 |
| dbpedia100k / pipnn alpha1.2-trees1 | 50 | 0.95460 | 200 | 0.99240 |
| deep100k / hnsw M16-efC200 | 40 | 0.96196 | 100 | 0.99159 |
| deep100k / hnsw M32-efC200 | 40 | 0.97818 | 80 | 0.99379 |
| deep100k / pipnn alpha1.2-trees1 | 30 | 0.95045 | 80 | 0.99159 |

Effort is efRuntime for HNSW and beam size for PiPNN; equal numbers are not
equivalent work budgets. Recall is the minimum across the two observations.

## Interpretation and next experiment

HNSW M16/efConstruction=200 is the primary baseline: it builds faster and has
lower measured query latency than the tested M32 alternative. PiPNN uses one
partition tree, alpha=1.2, final maximum degree 64, EigenKNN leaves, and no
quantization. This is one candidate configuration, not a tuned optimum for
either implementation. Full commands are recorded with the raw run evidence.

The screening gate was at least 2x construction improvement, no more than 10%
p95 query regression at the required recall, and no more than 25% resident
memory growth. The mixed high-dimensional result does not clear that gate
across workloads. DEEP also has substantially higher peak construction memory,
despite slightly lower resident query-process memory. Peak figures include
input buffers and runtime allocations and are not production lower bounds.

The next experiment should tune PiPNN pruning/partition parameters and HNSW
construction parameters against the 1536D recall/latency frontier, align compiler
arithmetic settings, and measure a common in-memory ready-to-query boundary.
If that improves the tradeoff, expand to million-vector and multithreaded builds.
Dynamic updates, deletion, filtering, range queries, persistence integration,
quantization, and concurrent serving remain untested. The paper's bulk graph
construction result alone does not establish a drop-in HNSW replacement.

## Validation and artifacts

- ARM distance checks compared L2, inner product, and norm across dimensions
  1–100 with a double-precision reference.
- A fully connected graph with 128 vectors and 20 independent queries reached
  recall@10=1.0 through the PiPNN query driver, with bounds/uniqueness checks.
- VecSim's 200-vector/12-query smoke test reached recall@10=1.0.
- Both drivers warm 100 queries per effort and time query execution separately
  from recall evaluation. QPS is inverse mean serial query latency.
- Build patches, harnesses, input preparation, runner, raw logs, and the summary
  script are retained here. The full remote experiment is
  `dorer-arm-big:/home/ubuntu/pipnn-poc-20261114`.

Cleanup removed old compiled artifacts only from three inactive build trees,
freeing roughly 16 GiB physically. Disk availability increased from 4.6 GiB to
21 GiB and is about 20 GiB after the PoC. Repositories, source changes, datasets,
logs, and reports were preserved; no production HNSW implementation was changed.

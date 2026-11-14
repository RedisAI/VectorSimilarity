# PiPNN ARM feasibility experiment

This directory is an experimental benchmark bundle, not a VectorSimilarity
backend or a production HNSW change. Start with [phase 2 findings](PHASE2_RESULTS.md)
for the current conclusion, then [initial findings](RESULTS.md) for both datasets.
Build instructions are in [BASELINE_BUILD.md](BASELINE_BUILD.md) and
[PIPNN_BUILD.md](PIPNN_BUILD.md). Builds use pinned revisions, not this PR's base.

The runner and data preparation scripts preserve the original ARM host paths.
To rerun elsewhere, adapt those paths and provide the datasets referenced in
`prepare_data.py`; vectors, binaries and graph files are not included. The JSON
summary can be regenerated locally from the included text logs without datasets.
`evidence/` preserves phase 1; `phase2-evidence/` is the combined snapshot, so
some raw logs deliberately appear in both. See [upstream attribution](PIPNN_LICENSE.txt)
for the PiPNN patch and its source.

Status: ARM build and correctness smoke checks passed. Initial 100k-vector
screening is complete; see [results and limitations](RESULTS.md).
The follow-up [1536D parameter screen](PHASE2_RESULTS.md) identifies a useful
high-recall PiPNN setting and a cheaper HNSW baseline.
This experiment evaluates a candidate replacement for VectorSimilarity HNSW.
Static build/search results cannot establish support for online updates/deletes.

## Host and isolation

- User-selected host: `dorer-arm-big`, 16 Neoverse-V2 cores, 123 GiB RAM.
- Remote experiment: `/home/ubuntu/pipnn-poc-20261114`.
- PiPNN upstream: `443a328dc71a3b7be753d52ca2b3c9a4ba77f3fd`.
- VectorSimilarity: clean detached worktree at `54a841ae841828d53a482a826d6ace7753bc5450`.
- Main working checkout and its existing modifications are untouched.
- Baseline uses FP32 L2, single-value HNSW, Release build, fixed C API seed 100.

Before setup, root disk had 4.6 GiB available. Cleanup removed only ELF binaries
and static archives from these unused build trees, following process-reference
checks, retaining repositories, sources, datasets, logs, and reports:

- `/home/ubuntu/wt/hnsw-search-prefetch/bin`
- `/home/ubuntu/wt/mod-18279/bin`
- `/home/ubuntu/sq4-gate-1103/bin`

Available disk increased to 21 GiB. Logical removed file sizes total 18,708,642,004
bytes; this differs from physical disk reclaimed because of sparse files and
filesystem accounting. New experiment setup subsequently consumes some space.

## Inputs and measurement boundaries

`prepare_data.py` prepares 100k DEEP vectors (96 dimensions) and 100k DBpedia
embeddings (1536 dimensions) from existing host datasets. DBpedia is normalized
to FP32 unit vectors and searched with squared L2, approximately preserving cosine
ranking (rounding can affect close ties). Ground truth uses these FP32 L2 inputs.
Raw and normalized exact-content overlaps are excluded. Near-identical queries
and ambiguous top-10 distance ties are excluded and counted in each manifest.
DEEP has 999 evaluation queries; DBpedia has 1000. Ground truth is recomputed
against the actual subset using an exhaustive float64 scan and direct-distance
reranking of shortlisted candidates.

Initial screening uses one build thread and one query thread for both algorithms.
Inputs are prepared FP32 fbin files; common offline data preparation and ground
truth generation are excluded from measured build costs. Build timing, file
loading, and graph serialization must be reported with their respective scope.
Query timing excludes recall evaluation; reported QPS is inverse mean serial
query time, not concurrent batch throughput. Both drivers warm the first 100
queries before each search-effort sweep.

HNSW reports tracked index allocation and resident process memory after releasing
its extra raw input copy, before loading queries/truth. PiPNN reports resident
process memory after loading graph/vectors, before loading queries/truth. These
RSS values include runtime overhead; graph file size is not resident index size.

Compare actual operating points attaining at least 95% and 99% recall@10, never
interpolate unattained targets. Initial proposed gate per dataset/target: at least
2x faster construction, at most 10% p95 query-latency regression and at most 25%
resident-index-process-memory increase. Report measured outcomes separately and
do not call a static winner a production-ready HNSW replacement. Repeat promising
configurations; parameter choices and random graph variation must be disclosed.

## Sources

- Paper: https://arxiv.org/abs/2602.21247
- Authors' implementation: https://github.com/ParAlg/PiPNN
- Follow-up research (not implemented here): https://arxiv.org/abs/2611.05690

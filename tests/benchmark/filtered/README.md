# Filtered vector benchmark foundation

This directory owns the dataset format, fixed filter bank, exact filtered oracle,
and scorer shared by RAM and disk benchmark runners. It does not implement a
search algorithm or emulate Redis's AUTO policy. Compare real query execution
through the RediSearch runner; native VecSim timing is a separate follow-up.

## Quick start

```sh
python3 -m venv /tmp/filtered-bench-venv
/tmp/filtered-bench-venv/bin/pip install -r tests/benchmark/filtered/requirements.txt
PYTHON=/tmp/filtered-bench-venv/bin/python
BENCH=tests/benchmark/filtered/filtered_bench.py
$PYTHON "$BENCH" prepare --output /tmp/filtered-smoke \
  --documents 10000 --dimensions 128 --queries 32 --seed 20221
$PYTHON "$BENCH" oracle /tmp/filtered-smoke
$PYTHON -m unittest discover -s tests/benchmark/filtered -v
```

Point a candidate runner at this bundle, then score the resulting JSONL:

```sh
$PYTHON "$BENCH" score /tmp/filtered-smoke /tmp/candidate/results.jsonl \
  --policies AUTO BATCHES ADHOC_BF --repeats 3 \
  --target-recall 0.95 --fail-below-target --output /tmp/candidate/report.json
```

Bundle, oracle, and report outputs refuse to overwrite existing files. Keep large
arrays, generated oracles, and run outputs outside Git. Preparation depends on the
NumPy version, which is recorded; distribute the frozen bundle rather than
regenerating it on each candidate branch. Consumers verify every file checksum.

## Datasets

`datasets.json` records existing raw dataset references, shapes, and intended
roles. Asset URLs are discovery references, not verified immutable downloads.
Import records SHA-256 hashes of both source files. Pin and distribute those
hashes with the benchmark release before comparing candidates. Existing HNSW
snapshots and unfiltered ground truth are not a filtered-search oracle.

Use the synthetic 10K fixture for pipeline checks, DBpedia 100K/1M × 768 cosine
for selection, and Deep 1M/10M × 96 squared L2 for disk scale. Start with 32–128
development queries for inexpensive iteration, then the fixed 2K development
split. Reserve source query rows `[2000, 10000)` for the final comparison; do not
tune using that split. Source query arrays must contain genuine separate queries;
the importer checks file identity and shape, not the dataset publisher's split
provenance or semantic independence.

Example after downloading the DBpedia FP32 base and query arrays:

```sh
$PYTHON "$BENCH" import --output /tmp/dbpedia-100k-dev \
  --vectors /data/dbpedia-cosine-dim768-1M-vectors.raw \
  --queries /data/dbpedia-cosine-dim768-test_vectors.raw \
  --format raw --documents 1000000 --query-count 10000 --dimensions 768 \
  --limit-documents 100000 --limit-queries 128 --metric cosine \
  --source dbpedia-1m-fp32 --seed 20221
```

For Deep, use `--format fbin --documents 10000000 --query-count 10000
--dimensions 96 --metric l2`, with `--limit-documents 1000000` for the small
variant. The importer requires an exact little-endian `(count, dimensions)` fbin
header and exact payload length. For a held-out bundle, select
`--query-start 2000 --limit-queries 8000`. The source row interval is recorded.

## Fixed workload

The default matrix contains independent, exact-cardinality filters at 100%, 10%,
1%, 0.1%, and 0.01%, with `k=10`. At 10% and 1%, `k=100` is also included when
at least 200 documents qualify. Counts round down, with a minimum of one; semantic
cases explicitly cover zero, one, `k-1`, `k`, and `k+1` eligible documents.

Up to eight query anchors each receive fixed near/far filters at 10%, 1%, and
0.1%. Their memberships are determined once using the exact distance to the
anchor, producing deliberately enriched/depleted neighborhoods. Each anchor is
queried against its own filters; these are controlled extremes, not a claim about
natural customer predicates. All filters coexist as static metadata during a
run. No index or filter is rebuilt between policies. The oracle records each
case's matching rate among the nearest 1,000 unfiltered documents (or the whole
corpus when smaller), making the realized correlation visible.

Real TAG/NUMERIC/TEXT predicates and customer application replay belong in the
engine adapters. Customer-specific schemas and query definitions stay in their
private source repository. Updates/churn need separately versioned document
snapshots and oracles; this static oracle is invalid after changing vectors or
filter membership.

## Artifact and result contract

`load_bundle(path)` returns `Bundle` with `manifest`, NumPy `vectors` and
`queries`, `filters`, `cases`, and `bundle_id` (the manifest SHA-256). Inputs are:

| File | Content |
| --- | --- |
| `manifest.json` | Version, metric (`cosine`/`l2`), dimensions, document/query counts, seed, provenance, four file checksums |
| `vectors.f32`, `queries.f32` | Row-major little-endian FP32; external document/query IDs are zero-based rows |
| `filters.json` | Filter name to sorted, unique eligible document IDs |
| `cases.json` | Objects with `case_id`, `query_id`, `filter_id`, `k`, `cohort` |

The oracle uses float64 arithmetic over original FP32 inputs, independently of
index quantization or returned engine scores. It calculates the corpus distance
array once per query in bounded blocks, then selects the requested top-k for each
filter. Memory is one float64 corpus-distance array plus a 4,096-vector block,
filter metadata, and selection temporaries; it never allocates a Q×N matrix.
L2 means squared Euclidean distance. Cosine rejects zero vectors.

`ground_truth.jsonl` is bound to the bundle and has a checksum sidecar. It records
eligible count, target count `min(k, eligible_count)`, canonical IDs/distances,
strictly closer IDs, and IDs tied with the kth neighbor. Ties use absolute and
relative tolerances of `1e-7`. Any boundary-tied document can fill a boundary slot;
returning extra tied documents cannot compensate for a missing strictly closer
document. Each document counts once. Empty truth has null recall and separately
requires a successful empty response.

Each runner writes one JSONL record per case, policy, and repeat:

```json
{"bundle_id":"manifest-sha256","case_id":"q0:independent_1:k10","repeat":0,"requested_policy":"AUTO","effective_policy":null,"status":"ok","ids":[4,9],"distances":[0.1,0.2],"latency_ms":0.7}
```

`status` is `ok`, `error`, or `timeout`. Errors still need empty `ids`/`distances`
and a measured finite nonnegative latency. Repeat IDs are zero-based.
`effective_policy` stays null unless actually observed; requesting AUTO does not
prove which policy ran. An optional `error` describes failures. Candidate scores
are checked for finiteness but recall is computed from IDs, since a quantized
backend's scores need not equal full-precision distances.

The scorer requires the complete expected policy/repeat matrix. It rejects mixed
bundles and duplicate rows, marks duplicate/ineligible document IDs as invalid,
and keeps missing responses, errors, and timeouts in the recall denominator.
Reports include every query plus policy/repeat/cohort summaries, incomplete result
counts, recall, and successful-query latency percentiles. Missing/error latency
is never presented as successful latency. Exit status is nonzero for invalid,
missing, incomplete, error, or timeout results; `--fail-below-target` also enforces
the chosen recall target.

## Comparing POCs

Create the benchmark base before branching algorithms. Pin the exact
VectorSimilarity, RediSearch, and Enterprise commits together; an Enterprise
checkout's nested dependencies must agree with the benchmark branches. Record the
bundle hash, candidate SHAs and dirty state, compiler/build mode, CPU, RAM limit,
storage, thread/client counts, index parameters, graph/snapshot hash, efRuntime,
policy, and timing scope in each runner's `run.json`.

For traversal-only candidates, load the same serialized graph and keep insertion
order fixed. If a candidate changes construction, report build cost and graph
differences separately. Run each candidate at least three times and compare
latency at the same recall target per selectivity/correlation cohort, including
failures and variance. Sweeping efRuntime must produce separate run directories.
Do not select on an average that hides failure in a selective cohort.

RAM and disk need separate timing labels. Disk comparisons additionally record
the total process/memory budget, block-cache size, warmup, and whether OS/device
caches were actually controlled. A restarted process alone does not establish a
cold disk cache. Benchmark timings are invalid when run alongside builds, tests,
or other benchmark jobs. Native kernel adapters, concurrency/churn workloads,
and application-level business ranking remain separate additions; this foundation
provides static filtered-vector recall and a shared comparison format.

# Using the benchmark base for algorithm POCs

Start algorithm selection in VecSim, then validate the surviving candidates through
Redis and the customer application. Keep the benchmark tooling and frozen inputs
identical across candidates. NaviX and ACORN implementations are not included.

## Artifacts and ownership

| Artifact | Repository / draft | What it measures |
| --- | --- | --- |
| Frozen FP32 vectors, queries, filter memberships, exact oracle, scorer, native runner | [VectorSimilarity draft](https://github.com/RedisAI/VectorSimilarity/pull/1093), `tests/benchmark/filtered/` | Algorithm recall and C++ search latency with precomputed eligibility |
| Redis command runner | [RediSearch draft](https://github.com/RediSearch/RediSearch/pull/12157), `tests/benchmarks/filtered-vector/` | Actual `FT.SEARCH` round trips, including filtering, with AUTO / BATCHES / ADHOC_BF |
| Redis Flex wrapper | [Enterprise draft](https://github.com/redislabsdev/RediSearchEnterprise/pull/1372), `benchmarks/filtered-vector/` | Disk command execution, reranking, memory/configuration observations and explicit cache protocols |
| Seeded customer replay and frozen RESP inputs | [Private customer-workload draft](https://github.com/RediSearch/search-workload-repro/pull/1), `docs/BENCHMARK-BASE.md` | One production-derived workload: read-only KNN, updates, and application fan-out |

The common branch name is `omer-filtered-vector-bench-base`. Resolve and record
its exact commit in each repository before branching; a moving branch name is
not a reproducible version. Benchmark tooling paths can point to separate
checkouts using `--shared-root` and `--redisearch-root`. No engine dependency
upgrade is required merely to use the scripts.

## First native comparison

From the VectorSimilarity benchmark branch, using Python 3.12 or newer:

```sh
cmake -S . -B /tmp/vecsim-filtered-build -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SVS=OFF -DVECSIM_BUILD_TESTS=OFF -DVECSIM_BUILD_FILTERED_BENCHMARK=ON
cmake --build /tmp/vecsim-filtered-build --target native_filtered_bench -j 4
python3 -m venv /tmp/filtered-bench-venv
/tmp/filtered-bench-venv/bin/pip install --only-binary=:all: --require-hashes -r tests/benchmark/filtered/requirements.txt
PYTHON=/tmp/filtered-bench-venv/bin/python
BENCH=tests/benchmark/filtered
BINARY=/tmp/vecsim-filtered-build/filtered_benchmark/native_filtered_bench

$PYTHON "$BENCH/filtered_bench.py" prepare --output /tmp/filtered-smoke \
  --documents 10000 --dimensions 128 --queries 32 --seed 20221 --metric cosine
$PYTHON "$BENCH/filtered_bench.py" oracle /tmp/filtered-smoke
$PYTHON "$BENCH/native_run.py" build --bundle /tmp/filtered-smoke \
  --binary "$BINARY" --output /tmp/native-graph --m 16 --ef-construction 200
$PYTHON "$BENCH/native_run.py" run --bundle /tmp/filtered-smoke \
  --binary "$BINARY" --graph /tmp/native-graph --output /tmp/native-ef200 \
  --policies NATIVE_SCAN NATIVE_BATCHES --repeats 3 \
  --ef-runtime 200 --batch-size 100 --timeout-ms 10000
$PYTHON "$BENCH/filtered_bench.py" score /tmp/filtered-smoke \
  /tmp/native-ef200/results.jsonl --policies NATIVE_SCAN NATIVE_BATCHES \
  --repeats 3 --target-recall .95 --output /tmp/native-ef200/report.json
```

`NATIVE_SCAN` scans eligible documents. `NATIVE_BATCHES` requests neighbors from
the actual HNSW batch iterator and intersects them with the eligible set. Neither
is Redis AUTO or a measurement of Redis's filter iterator costs. The oracle
remains independent: float64 arithmetic over the original FP32 inputs.

Use new output directories for every run. Examine per-query and per-cohort
recall, failures, incomplete results and latency before comparing speed. Add
`--fail-below-target` when requiring **every nonempty query** to reach the target;
the default scorer still fails on errors, invalid/missing or incomplete results.
The strict per-query gate is stronger than a mean-recall target.

## Move from smoke to selection

Use the DBpedia downloader and import commands in the [README](README.md) to
freeze a 100K-document, 32-query development bundle. Retain download provenance,
bundle manifest, oracle and their checksums. Compute the oracle once, then build
the baseline graph once. The generated files stay outside Git.

Run EF_RUNTIME 10, 50, 100 and 200 in separate output directories, with three
repetitions each. Keep batch size and construction parameters fixed unless that
parameter is the experiment. Compare latency at matched recall **within each
selectivity/correlation/k cohort**; a faster run that loses recall is not a win.
Record each repetition, including failures. These native runs are single-threaded
search measurements, not concurrent service throughput tests.

For stronger candidates expand to DBpedia 1M with the fixed development query
split, then use held-out query rows `[2000, 10000)` for final evaluation. For disk
pressure add Deep 1M/10M. Do not choose the algorithm from synthetic vectors or
from one small development run on an uncontrolled workstation.

## Plug in NaviX

1. Branch from the pinned VectorSimilarity benchmark commit. Keep the artifact
   loader, oracle, scorer and baseline implementations unchanged.
2. Add the implementation behind the C++ candidate adapter documented in the
   [native runner section](README.md). Register an explicit candidate label such
   as `NAVIX`; the unmodified runner must reject that label until an adapter exists.
3. Build the candidate executable in a separate build directory. Run it against
   the **same bundle and saved graph directory**, requesting its actual candidate
   label. Score those rows using that same label and oracle. Keep the baseline
   and candidate binary hashes, source revisions and build settings with results.
4. Sweep relevant search parameters and compare matched-recall curves. Report
   actual counters only; returned candidates are not graph visits or distance
   computations. Eligibility generation is excluded from native search timing.
5. If the candidate changes graph construction or cannot load the baseline graph,
   run it as a separate construction experiment. Record build time, memory and
   graph size; do not describe it as a traversal-only comparison.

## Validate inside Redis

After native selection, wire the implementation into the real Redis query path.
RQE currently produces filter eligibility; the public VecSim batch iterator does
not take that filter. An algorithm that uses eligibility during traversal needs
an actual interface and engine integration. Adding `NAVIX` to a Python argument
list alone cannot exercise it.

Expose an explicit strategy route and an observable marker/counter proving the
candidate ran. Then extend the command runner to request and verify it. AUTO may
otherwise choose ADHOC and bypass the implementation. Use the RediSearch README
for module provenance, loading, readiness, graph reuse and scoring commands.

For Flex, follow the Enterprise README: the supported policies are BATCHES and
ADHOC_BF; disk AUTO is unavailable. Use a fixed total memory budget, explicit
rerank setting and documented warm/cold-start preparation. The wrapper records
cache evidence but does not evict caches. Native RAM HNSW measurements do not
establish disk performance.

Finally run the private customer profiles on the supported RAM setup. There is
one workload with three entity types, using synthetic documents and vectors.
Its grouped/business-ranked output is an application regression gate, not the
filtered top-k oracle. Customer replay on Flex and vector-stage recall extraction
need additional adapters. Restore the pristine snapshot before every churn run.

## Keep with each comparison

- Exact tooling and engine revisions; executable/module hashes and build settings.
- Download provenance, frozen bundle ID, oracle checksum and graph/snapshot hash.
- Query split, policies, construction/search parameters, repetitions and timeouts.
- Raw `results.jsonl`, `run.json`, scored reports and unmodified error records.
- Hardware, thread/client counts, memory limits, cache procedure and concurrent
  system activity. Do not benchmark while building or running another benchmark.

The initial suite covers static single-value FP32 eligibility and command-level
RAM/disk execution. Rich numeric/text predicates, multi-value vectors, automated
cache restoration, instrumented disk I/O, concurrent throughput and mutation-aware
recall remain separate extensions. Keep those distinctions in any selection report.

# Initial native development baseline

Recorded on 2026-11-02 using source `2de821bd185805000e871181737bd3be90a58d77`,
GCC 13.3.0, Release, SVS disabled. This is a development baseline on an
uncontrolled workstation, useful for verifying the comparison pipeline. It is
not an algorithm-selection verdict or a Redis service latency measurement.

The [manifest](native-development-manifest.json) pins source, executable, graph,
bundle, download and result hashes, CPU details and construction measurements.
The [cohort CSV](native-development-cohorts.csv) preserves each repetition's
summary, including failures, completeness, recall and p50/p95 search latency.
Large input/graph files and raw run outputs are outside Git; their local paths
and hashes are recorded in the manifest. Download/import/build/run commands are
in the [POC guide](../POC-GUIDE.md) and [README](../README.md).

## Executed matrix

- Synthetic 10K × 128 cosine, 32 queries, 245 cases, 2 policies × 3 repetitions:
  1,470 searches, no failures or incomplete results.
- DBpedia fixed 100K × 768 cosine, development query rows `[0,32)`, 277 cases,
  2 policies × 3 repetitions × EF_RUNTIME 10/50/100/200: 6,648 searches, no
  failures or incomplete results.
- Each dataset uses one saved HNSW graph across all policies/settings, M=16,
  EF_CONSTRUCTION=200, ascending insertion order. Batch size is 10 throughout.
- A separate DBpedia scan run with a 1 ms deadline retained 112 timeout rows out
  of 277 cases. Both runner and scorer returned nonzero status as expected.

DBpedia graph construction took 54.218 seconds, followed by 0.166 seconds of
serialization; VecSim reported 339,133,536 index bytes. That allocation statistic
is not peak process RSS. No CPU pinning, frequency control, cache eviction or
dedicated memory limit was applied. Policies rotate by repetition; there is no
implicit warmup. Builds and other benchmark jobs were not run concurrently.

## DBpedia overview

These values combine heterogeneous filter cohorts. Use the CSV to compare
within the same cohort; the overall percentiles below are descriptive only.
Latency intervals are the minimum and maximum of the three repetitions'
percentiles, not confidence intervals. Recall was identical across repetitions.

| Native policy | EF | Mean recall | Minimum recall | p50 ms across repetitions | p95 ms across repetitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scan | all | 1.00000 | 1.00 | 0.537–0.585 | 15.697–16.368 |
| HNSW batches | 10 | 0.98467 | 0.70 | 3.034–3.290 | 66.940–67.440 |
| HNSW batches | 50 | 0.99384 | 0.90 | 3.230–3.251 | 79.043–79.649 |
| HNSW batches | 100 | 0.99645 | 0.90 | 3.449–3.683 | 103.680–104.707 |
| HNSW batches | 200 | 0.99851 | 0.90 | 4.427–4.625 | 166.767–179.544 |

The 95% **per-query** gate fails for every batch setting, despite their high
mean recalls. This is why the scorer reports the worst queries and separate
cohorts. Do not infer an AUTO-policy recommendation from these native loops:
they exclude Redis filter generation and policy switching, and the batch
baseline never switches to an eligible-document scan.

For the next POC, load the same graph and bundle with the implemented adapter,
capture its rows, and compare latency at matched cohort recall. Expand the
development query count and dataset size before evaluating the held-out split.

# DBpedia100k parameter screen on ARM

Decision: retain PiPNN alpha=1.1 with one tree as a high-recall research candidate.
Do not replace HNSW. None of the screened PiPNN settings meets the 2x construction
screening target against the cheaper HNSW configuration. Pruning tuning did not
remove the 95%-recall latency disadvantage against HNSW efConstruction=200.

This phase changes only parameters on the same ARM host, binaries, prepared
100k x 1536 FP32 inputs and 1000 queries. It adds seven complete runs: four new
PiPNN configurations, a second run of the selected alpha=1.1 configuration, and
two HNSW M16/efConstruction=100 runs. The existing evidence remains unchanged in
`evidence/`; the combined snapshot is in `phase2-evidence/`.

## Measured tradeoff

| Configuration | Runs | Build s | p95 >=95% recall (ms) | p95 >=99% recall (ms) |
|---|---:|---:|---:|---:|
| hnsw M16-efC100 | 2 | 59.41 | 0.715 | 4.566 |
| hnsw M16-efC200 | 2 | 119.23 | 0.633 | 2.233 |
| hnsw M32-efC200 | 2 | 147.27 | 0.794 | 2.235 |
| pipnn alpha1.0-trees1 | 1 | 60.04 | 0.820 | 2.424 |
| pipnn alpha1.1-trees1 | 2 | 62.58 | 0.768 | 1.850 |
| pipnn alpha1.2-trees1 | 2 | 65.08 | 0.765 | 2.078 |
| pipnn alpha1.2-trees2 | 1 | 126.73 | 0.822 | 1.854 |
| pipnn alpha1.4-trees1 | 1 | 66.53 | 0.817 | 2.264 |

Construction seconds are an imperfect proxy: PiPNN includes graph serialization
but excludes loading that graph for queries; HNSW is immediately queryable after
its reported construction. These are not common end-to-end readiness times.
PiPNN also retains native/fast-math build flags unlike the VecSim core. Results
compare these builds, not algorithms under identical compiler arithmetic policy.

Query values are the median of per-run p95 at a measured search effort meeting
the recall threshold in every available repeat. A single-run screen is labeled
explicitly. The sweep is finite: notably, HNSW efConstruction=100 reaches only
98.9% recall at efRuntime=400, then 99.34% at 800. Untested efforts in between
could reduce its reported >=99% latency. Do not treat these sampled points as
the exact optimal frontier. Recall also differs slightly among qualifying points.

## What changed our understanding

- Reducing HNSW efConstruction from 200 to 100 roughly halves construction time.
  That removes the general PiPNN build-speed advantage on this dataset, but costs
  substantial query latency at the tested 99%-recall point.
- Stronger PiPNN pruning (alpha=1.0) saves a few build seconds but requires wider
  searches. Less aggressive pruning (alpha=1.4) adds edges without improving the
  measured latency. Alpha=1.1 gives a better measured 99%-recall tradeoff.
- Two trees improve candidate coverage and attain 99% recall with a smaller beam.
  However, their approximately 127-second build is about twice the single-tree
  cost, and the selected alpha=1.1 single-tree candidate has similar query latency.
- Reported query-process residency is about
  649 MB for PiPNN versus 653–654 MB for HNSW M16. Peak construction RSS is about
  1232 MB versus 1268–1269 MB. These include runtime/input-buffer overhead.
  Every PiPNN setting reports exactly the same RSS values despite differing
  graph file sizes. This phase did not isolate the reason for that equality;
  do not use these snapshots to rank memory efficiency among pruning settings.

Selected alpha=1.1 repeats built in 62.45–62.71 seconds. At beam 200, both reached recall 0.9901; p95 ranged 1.839–1.861 ms. Compared with HNSW efConstruction=100, its median construction proxy is 5.3% longer, its >=95% p95 is 7.4% higher, and its >=99% p95 is 59.5% lower at the tested efforts. Compared with HNSW efConstruction=200, its construction proxy is 47.5% lower and its >=99% p95 is 17.1% lower, but its >=95% p95 is 21.3% higher.

## Verification and next step

The runner refused overwrites and competing known benchmark/build processes.
The host had about 20 GiB free and was idle before the screen. Source inspection
confirmed alpha and replica forwarding to the builder; output reports the tree
count and graph degrees. Both drivers check result uniqueness and compute recall
against the existing exhaustive reference. The PiPNN driver also checks returned
ID bounds; the VecSim driver checks ground-truth ID bounds. Fixed/default graph randomness
and two repeats establish timing repeatability only, not seed robustness.

Parameter selection reused the evaluation queries, so there is no unseen-query
validation claim. The plan's fallback construction budget also failed after the
cheaper HNSW result; this is recorded in [the plan](PHASE2_PLAN.md). The selected
PiPNN setting was repeated to quantify the near-equal-build high-recall tradeoff,
not to relabel that gate as passed.

Recommend the next validation target be >=99% recall: use untouched queries,
denser search-effort sweeps near the threshold, aligned arithmetic flags and a
common ready-to-query boundary before moving to million-vector/multithreaded
measurements. Online updates/deletes and production integration remain untested.
No production HNSW source was changed.

Reproduce the combined summary with:

```sh
python3 poc/pipnn-arm/summarize.py --evidence poc/pipnn-arm/phase2-evidence
```

Exact commands, return codes, timings and query observations are saved in
`phase2-evidence/results/`; executable/input hashes are in its `metadata.json`.

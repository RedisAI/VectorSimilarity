# HNSW auxiliary capacity growth experiment

The minimal capacity-policy patch reduced measured insertion-loop time by
12.7–15.6% for a populated 10M-vector HNSW index, using 135.60 MiB (3.13%) more
index allocation. These measurements cover the final implementation, including
its original insertion locking and allocation ordering.

## Workload and provenance

- Measured on 2026-09-27.
- Baseline: `fcdeb37b1c264c00ad2e8176d91cc2c0d071625a`.
- Minimal patch: `9a3ca0a55e5331afd13382678c04db5602ec2ee9`.
- Generated float32 vectors, 32 dimensions, L2; input seed 42, graph seed 100.
- M=16, efConstruction=100, efRuntime=100, block size 1,024.
- GCC 13.3, `-O3`, assertions enabled, Intel Xeon Platinum 8375C.
- One pinned logical CPU on a shared 16-logical-CPU host; environment sampling
  used a different core.
- Four fresh processes, sequentially: baseline, patch, patch, baseline.
- Each process built monotonically from zero to 10M using a dedicated Google
  Benchmark case. Input generation, index construction, checkpoint queries,
  checkpoint RSS collection and output were excluded from manual insertion-loop
  wall time. Equal loop bookkeeping and selected insertion timing probes were
  included.

Both builds used the same harness and resize-helper timer. Tracked source changes
against each pinned revision, including staged changes, were checked: the only
instrumentation edit added that timer to `resizeIndexCommon`. Both executable
hashes were unchanged after all runs. Both builds used an identical external
CMake hook setting `HAVE_SVS_LVQ=0`. That build setting is not part of the patch;
these measurements do not validate the default SVS dependency configuration.

The patch doubles auxiliary capacity when exhausted and reclaims it at
one-quarter utilization. Graph and vector storage still grow in blocks. It
retains the original shared resize helper, including exact-fit vector
reclamation, but calls it less frequently. Existing allocator accounting
includes the spare capacity.

## Results

| Pair order | Baseline insertion loop | Patch insertion loop | Observed reduction |
|---|---:|---:|---:|
| Baseline then patch | 3,135.455 s | 2,645.502 s | 15.6% |
| Patch then baseline | 3,038.529 s | 2,651.770 s | 12.7% |

The patch insertion loop took 44.09–44.20 minutes versus 50.64–52.26 minutes
for baseline.
The earlier, larger prototype on base `ec835be8c` showed 14.6–16.5% reductions
with the same C++ workload. The minimal patch retains a similar measured benefit;
these separate campaigns do not establish identical performance.

| Metric at 10M | Baseline | Minimal patch |
|---|---:|---:|
| Auxiliary resize calls | 9,766 | 15 |
| Total resize-helper time, run range | 403.229–407.532 s | 1.383–1.384 s |
| Largest helper call across both runs | 788.265 ms | 701.739 ms |
| Index allocator bytes | 4,547,576,356 | 4,689,765,236 |

The extra index allocation was 142,188,880 bytes, or 135.60 MiB (3.13%). This is
allocator accounting at 10M, not process RSS or peak allocation during resizing.
Sampled query signatures at 1M, 2M, 4M, 8M and 10M matched across all four runs;
each checkpoint used 32 self queries. Every process exited successfully. The
artifact validator checked all four runs, their expected resize counts and probes,
query signatures, and agreement between checkpoint and Google Benchmark timings.

At the selected insertion crossing 8,388,608 vectors, median complete insertion
time was 71.919 ms for baseline and 702.122 ms for the patch (two samples each).
At 32 nearby block boundaries per run, median complete insertion time was
71.350 ms and 0.282 ms respectively (64 samples each). Less frequent resizing
therefore does not eliminate large individual growth events.

## Validation and limits

Before measurement, both instrumented variants passed a 32,769-vector smoke run
with matching queries and an 8,193-vector Valgrind run with zero errors and no
definitely, indirectly or possibly lost blocks. Each retained 329 bytes in five
reachable blocks. Two unchanged-baseline controls at 100K took 10.646 s and
10.500 s; these short controls do not bound noise in the 10M measurements.

| Run | Recorded one-minute host load, median / maximum |
|---|---:|
| First baseline | 1.81 / 2.45 |
| First patch | 1.79 / 3.16 |
| Second patch | 1.71 / 2.44 |
| Second baseline | 1.60 / 2.51 |

No swap activity was recorded during the runs. Host load does not measure
contention on the pinned CPU. These are descriptive observations from two runs
per variant on a shared host, not a confidence interval or a causal estimate of
every saved second.

This generated single-index workload does not reproduce a representative sharded
deployment. Selected insertion durations and resize-helper time do not measure
Redis command latency, reader wait time or exclusive-lock duration. Matching
sampled queries do not establish full ANN recall.

Before marking the draft ready, measure concurrent-query latency during ingestion
at representative local index sizes and vector dimensions, and confirm standard
CI with the default SVS dependency configuration. Larger speculative allocations
can fail earlier under memory pressure; allocation-failure recovery is not
addressed by this capacity-policy change.

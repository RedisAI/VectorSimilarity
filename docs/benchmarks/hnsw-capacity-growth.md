# HNSW auxiliary capacity growth experiment

An earlier geometric-growth prototype reduced measured insertion-loop time for
one populated 10M-vector HNSW index. These timings come from that prototype;
the final draft on its newer base has not yet been benchmarked at this scale.

## Workload

- Base revision: `ec835be8c`.
- Generated float32 vectors, 32 dimensions, L2; input seed 42, graph seed 100.
- M=16, efConstruction=100, efRuntime=100, block size 1,024.
- GCC 13.3, `-O3`, assertions enabled, Intel Xeon Platinum 8375C.
- One pinned logical CPU on a shared 16-logical-CPU host.
- Four fresh processes: baseline, prototype, prototype, baseline.
- Each process built monotonically from zero to 10M using a dedicated Google
  Benchmark case. Input generation, query validation and output were excluded
  from manual insertion-loop wall time. Equal loop bookkeeping and selected
  insertion timing probes were included.

The measured prototype doubled auxiliary capacity when exhausted and reclaimed
it at one-quarter utilization. Graph and vector storage still grew in blocks.
Both the prototype and this draft retain the original shared resize helper,
including its exact-fit vector reclamation, but call it less frequently. The
draft changes the capacity policy without changing allocation-failure handling
or insertion locking. Existing allocator accounting includes the spare capacity.

## Results

| Pair | Baseline insertion loop | Prototype insertion loop | Observed reduction |
|---|---:|---:|---:|
| Baseline then prototype | 3,049.748 s | 2,603.703 s | 14.6% |
| Prototype then baseline | 3,168.437 s | 2,645.137 s | 16.5% |

| Metric at 10M | Baseline | Prototype |
|---|---:|---:|
| Auxiliary resize calls | 9,766 | 15 |
| Total resize-helper time, run range | 413.559–427.886 s | 1.366–1.374 s |
| Largest helper call across both runs | 771.362 ms | 706.320 ms |
| Index allocator bytes | 4,547,576,356 | 4,689,765,236 |

The extra index allocation was 135.60 MiB (3.13%). Sampled query signatures at
1M, 2M, 4M, 8M and 10M matched across all four runs; each checkpoint used 32
self queries. Both executable hashes stayed fixed, and every process exited
successfully. Both corrected harness variants had passed a small Valgrind run
with zero errors before measurement.

At the selected insertion crossing 8,388,608 vectors, median complete insertion
time was 74.8 ms for baseline and 707.0 ms for prototype (two samples each).
At 32 nearby block boundaries per run, median complete insertion time was
74.1 ms and 0.291 ms respectively (64 samples each). Less frequent resizing
therefore does not eliminate large individual growth events.

## Limits and follow-up

The final baseline encountered a temporary host-load spike. Its recorded
one-minute load median/maximum was 1.55/13.87, compared with 1.05/1.96 for the
first baseline. These are descriptive observations from two runs per variant,
not a confidence interval or a causal estimate of every saved second.

This generated single-index workload does not reproduce a representative sharded
deployment. Selected insertion durations and resize-helper time do not measure
Redis command latency, reader wait time or exclusive-lock duration. Matching
sampled queries do not establish full ANN recall.

Before marking the draft ready, rerun performance on the final implementation
and measure concurrent-query latency during ingestion at representative local
index sizes and vector dimensions. Larger speculative allocations can fail
earlier under memory pressure; allocation-failure recovery is not addressed by
this capacity-policy change.

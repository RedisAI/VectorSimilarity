# Native PiPNN ARM screen — 2026-11-15

This screen measures the native backend in this change, not the upstream PiPNN
executable used in [the earlier PoC](https://github.com/RedisAI/VectorSimilarity/pull/1122).
The native implementation builds faster in these runs, but HNSW gives materially
lower query latency at comparable recall. These results do **not** support
replacing HNSW for query-heavy workloads.

## Conditions

- `dorer-arm-big`: aarch64, 16 CPUs, Neoverse-V2; GCC 12.3.0.
- CMake `RelWithDebInfo` (`-O2 -g -DNDEBUG`), normal SVS-enabled build;
  both backends use the same library and distance dispatch.
- One build and one query thread; backends run serially with no competing build.
- One build per backend/dataset, fixed seed 100. This is a screening measurement,
  not a repeated or statistically characterized benchmark.
- PiPNN defaults: degree 32, leaf 128, fanout 2, hash bits 12, candidate limit 80,
  alpha 1.1. HNSW: M=16, efConstruction=100.
- 100,000 vectors per dataset. DEEP has 96 dimensions and 999 queries; normalized
  DBpedia has 1,536 dimensions and 1,000 queries. Inputs and exact ground truth are
  the prepared datasets described in the earlier PoC. DBpedia unit normalization
  makes squared L2 approximately preserve cosine ranking, subject to FP32 rounding.
- Build time includes creation and insertion, plus explicit graph construction
  for PiPNN. File I/O and ground-truth preparation are excluded.
- Each ef sweep warms up the first 100 queries. p95 measures the query call,
  excluding result inspection and recall computation.

## Results

These are the lowest measured ef points exceeding 95% recall in the coarse sweep.
Recall is not identical between rows; no interpolation is used.

| Dataset | Backend | Build (s) | ef | Recall@10 | p95 (ms) | Index (MB) | Peak build RSS (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DBpedia 1536D | PiPNN | 21.326 | 800 | 96.17% | 3.180 | 635.6 | 1362.4 |
| DBpedia 1536D | HNSW | 60.003 | 100 | 96.92% | 0.847 | 647.7 | 1268.4 |
| DEEP 96D | PiPNN | 2.587 | 400 | 97.61% | 0.831 | 57.5 | 210.0 |
| DEEP 96D | HNSW | 12.628 | 50 | 96.92% | 0.109 | 69.8 | 117.6 |

MB is decimal. Index memory is allocator accounting; peak RSS includes the input
matrix, process overhead, and temporary build storage. PiPNN's smaller final
index does not imply smaller peak build memory.

The complete ef sweeps, including 99% recall points where reached, are in
[the raw output](arm-native-2026-11-15.txt). PiPNN did not reach 99% on DBpedia
within this sweep (maximum ef 1600 reached 98.24%).

For reproduction, build `pipnn_comparison` and follow the commands in
[the backend documentation](../../pipnn.md). Use ef values
`50,100,200,400,800,1600` and the query counts above. Before considering deployment,
repeat the screen on the target workload and assess the complete rebuild lifecycle,
peak memory, and serialized-call integration requirement. No Redis Search schema
or tiered integration is included in this change.

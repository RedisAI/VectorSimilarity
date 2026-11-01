# End-to-end HNSW policy comparison

Two sequential experiments on one c7i.4xlarge compare actual `FT.SEARCH` calls through two-shard OSS coordinators. Each experiment uses 146,373 real DBpedia embeddings, split 73,187/73,186, converted from FLOAT32 input to JSON FLOAT64, cosine distance, dimension 768, M=16, EF_CONSTRUCTION=100, and K=200. This is a customer-like proxy; the customer's embeddings and complete schema are unavailable.

## Revisions and fixtures

| Experiment | Candidate RediSearch / VecSim | Queries | Hole fixture max document IDs |
| --- | --- | --- | --- |
| `first-candidate` | `50b3cb6d667494982941dc09ae4fac65c2287174` / `2d27f83f0cdb23f060983c5ca7eb74139fb05a93` | 2000–2019 | 1,152,500 / 1,058,100 |
| `narrowed` | `f8f3945d7fcf29097602dd0a33e5078e064ba449` / `3fbc21974bd0c0d73ac7205604b01900b4a73751` | 4000–4019 | 1,000,000 / 1,000,000 |

Both use baseline RediSearch `1fbe5191c925c710945a7ed526293d552e8e346a` / VecSim `1db74bef518d58318d53bc930027d511edcc8e49`, Redis `2dfc1e4a990b6b1c85172d3aa65700ffad82d38c`, and RedisJSON `d2cad90ddbbc4dfc7cf692ddcb5a1a97596d72ac`. Builds use the same GCC 13.3/Rust 1.94 toolchain, RelWithDebInfo, and assertions disabled. The Redis runtime is not ASan-instrumented.

The narrowed candidate preserves BF for initial estimates at or below 17.5%. A capped full estimate preserves BF below one million estimated document IDs. The first experiment exposed why the original capped ratio was insufficient: fresh sparse negative filters regressed, while expensive scans over holes benefited. The second experiment tests the chosen lower bound on new reserved queries; it does not reuse the first query set for validation.

Each experiment includes a fresh-index fixture. Hole fixtures interleave creation/deletion of non-vector JSON documents while preserving vector insertion order. All three indexing worker settings are zero during loading; measurement uses two search workers and zero minimum operation/maintenance workers. Before timing, indexing must settle, both graphs must contain identical live counts, and no HNSW deletion may remain pending. Exact and forced-batches keys **and scores** must agree across builds. Saved controls also allow comparison across histories.

## Measurement contract

The full matrix contains random and first-coordinate-correlated membership; 1%, 5%, 10%, 17.5%, 26%, 50%, 95%, and 100% passing fractions; equivalent positive TAG and three-negative-TAG predicates; EF_RUNTIME 10, 400, and 800; 20 queries; three repeats; and four strategies. Jobs are shuffled and strategy order rotates. Only one request runs at a time. Profiles are collected separately from unprofiled timings.

The headline comparison is **baseline AUTO versus candidate AUTO on the identical original predicate**. Forced BATCHES uses that predicate too. The exact ADHOC_BF reference adds an all-document TAG guard, since an all-missing negative filter otherwise simplifies to unfiltered KNN. That guard can change NOT traversal cost: its timing is not the customer's raw BF comparator. The 100% negative case optimizes to STANDARD_KNN on this master and is an unchanged control, not hybrid-policy validation.

Each completed experiment contains 46,080 latency rows and 192 groups with 60 samples per strategy/group. Query repeats are correlated observations, not 60 independent queries. Recall is intersection with exact filtered top-200, with tie-adjusted recall also recorded. A complete result count does not imply exact recall.

## Results

The first candidate regressed 17/96 fresh groups by more than 10% at the median, all sparse negative filters; worst median slowdown was 3.64×. On the ticket-scale hole fixture, no median exceeded 1.043×, but one sparse group had a 1.159× p95 slowdown. At 26% negative filters/EF10, hole-fixture medians fell from 237–242 ms to 8.6–9.7 ms, with mean recall 99.6% and minimum query recall 97.5–98.5%. These are **first-candidate** measurements.

The **final narrowed candidate** completed all 46,080 measurements with no errors or incomplete replies; all controls and cross-history equality checks passed. Customer-like 26% negative-filter results at exactly one million document IDs per shard:

| Membership | EF | Baseline p50 ms | Candidate p50 ms | Baseline p95 ms | Candidate p95 ms | Mean recall@200 | Minimum query recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Projection | 10 | 203.39 | 9.78 | 220.03 | 17.46 | 99.375% | 98% |
| Random | 10 | 207.91 | 10.34 | 227.14 | 11.28 | 99.425% | 97% |
| Projection | 400 | 204.68 | 12.33 | 228.81 | 19.34 | 99.8% | 99% |
| Random | 400 | 209.50 | 12.85 | 228.44 | 14.37 | 99.825% | 99.5% |
| Projection | 800 | 207.21 | 14.83 | 228.98 | 30.92 | 99.9% | 99% |
| Random | 800 | 206.57 | 15.35 | 229.41 | 16.68 | 99.9% | 99.5% |

Baseline AUTO recall in this table is 100%. These are proxy-query observations, not customer recall guarantees.

The final policy removed the large fresh-index regressions: 0/96 fresh groups exceed 10% median slowdown, and the worst median is +3.6%. Fresh negative 26% stays on BF (~22.5 ms, exact recall), deliberately giving up the earlier fresh-case speedup. The remaining trade-off is **8/192 total groups above 10% median slowdown**, all 1%/5% negative filters at the one-million-ID boundary. Their medians increased by roughly 15–20 ms; worst +14.0% (142.52→162.42 ms). The worst p95 regression is +16.1%.

Per-query results matter too: 10/1,920 fresh and 178/1,920 hole-fixture query/configuration combinations exceed 10% slowdown using the median of three repeats. A correlated 26% **positive** filter at EF800 has the worst fresh query, 18.03→36.23 ms (2.01×); that group's median improves 42%, but p95 regresses 14.8%. The same query is 1.87× slower with holes. Across the full matrix, the minimum candidate recall is 95.5% (95% positive filters, EF10). These limits remain visible in the draft PRs; the change is not regression-free.

See [all workload results](narrowed/summary.json), [per-query results](narrowed/query-summary.json), and [the consistency audit](narrowed/audit.json).

## Inspect and reproduce

`summary.json` retains every group, including regressions. `query-summary.json` retains each query/configuration median across its three repeats. `audit.json` checks all expected pairs, result completeness, stable IDs across repeats, exact/BATCHES reference agreement, graph state, and cross-history control equality. Large JSON/JSONL files are losslessly gzip-compressed with deterministic headers. Artifact hashes identify binaries; binaries and embeddings are not committed.

To recalculate a summary, expand the selected experiment's compressed files, then run its original analyzer:

```sh
python3 -m pip install numpy==2.5.4 redis==8.1.0
gzip -dk first-candidate/results/*/*.gz
python3 first-candidate/analyze_current.py first-candidate
python3 audit_results.py first-candidate
python3 analyze_queries.py first-candidate
```

Use the native harness's `auto_measure extract` with the [DBpedia serialized source graph](https://dev.cto.redis.s3.amazonaws.com/VectorSimilarity/dbpedia-cosine-dim768-M64-efc512.hnsw_v3), count 300000, and seed 18264; take the first 146373 vectors from that shuffled raw output. The archived native harness uses `std::mt19937` and `std::shuffle`; match the supplied input SHA256, since shuffle implementations can differ. Extract the specified 20-vector slice from [the disjoint test vectors](https://dev.cto.redis.s3.amazonaws.com/VectorSimilarity/dbpedia-cosine-dim768-test_vectors.raw). Both inputs are little-endian FLOAT32, dimension 768. Each run's data manifest records byte sizes and hashes.

The run/build scripts are retained as executed and have host-specific paths. Adapt those paths and use a dedicated benchmark machine, two empty two-shard clusters, identical runtime/dependencies, and no concurrent build or benchmark. The Python harness also accepts four explicit ports, root directory, and history; `load` refuses nonempty databases. The launcher's exact `rsv.sh` is retained in `narrowed/`; it comes from the repository's drive-redisearch skill, adapted to accept the selected checkout root.

The measured domain does not establish behavior on every graph, CPU, datatype, correlation, or concurrency level. The one-million estimate is a conservative measured scan-cost proxy, not a query-time bound. Native tests cover additional index shapes, but these end-to-end runs cover one schema and K=200. Approximate-search recall remains data-dependent.

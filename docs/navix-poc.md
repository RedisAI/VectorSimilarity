# NaviX filtered batches POC

The RAM HNSW batch iterator accepts an optional `VecSimQueryParams.navix` callback.
Zero means definitely excluded; nonzero means MAYBE. The callback receives an external
label, not an internal graph ID. Zero-initialize query parameters to keep existing behavior.
This extends the C struct ABI: rebuild callers with the matching headers and library.

The caller owns the callback context and optional cumulative counters until the iterator
is freed, including resets. Keep the index/label/filter view stable while iterating and
serialize access to the counters. The callback must not mutate the index, perform I/O,
throw, or acquire graph locks. It runs after adjacency locks have been released.

## Traversal

This adapts [NaviX](https://arxiv.org/html/2506.23397v1) to a resumable stream with
conservative eligibility. It is not the paper's exact membership-set implementation.

- Keep the existing unfiltered upper-layer descent and its single base-layer seed.
- Estimate local selectivity over all direct neighbors before visited suppression.
- Use `D = M0 = 2*M`: one hop for `s > 0.5`, directed two hop for
  `s*D*(D+1) >= 3*D`, blind two hop otherwise. `forceBlind` always selects blind.
- Admit direct matches first; stop after `D` unique newly admitted candidates across
  both hops. Directed mode scores rejected first-hop branches to choose their order.
- Rejected nodes can supply second-hop links; they never enter the result heap. A
  rejected initial seed can navigate without tightening the result bound.
- Retain the frontier/extras/visited list across batches. Bound ties remain on the
  frontier for the next batch so equal-distance candidates are not silently dropped.
- Reuse O(D) scratch space, copy one adjacency list under one node lock, and poll
  the query timeout in both hops. There is no membership bitmap, extra seed,
  disconnected-component recovery, or fallback scan.

The consumer must apply its exact filter and request more batches when MAYBE candidates
fail it. Exhaustion can return fewer than k, even when more eligible documents exist.
Increasing ef cannot repair all disconnected filtered neighborhoods.

The tiered brute-force frontend applies the same callback before scoring. Counters cover
base-layer expansion and frontend scoring; distance counts exclude upper-layer descent
and the initial seed distance. Counters accumulate across calls and resets. Top-k and
range APIs do not activate this POC; use the batch API. The integration validates only
unquantized, single-value FLOAT32 RAM HNSW.

`test_navix` covers rejected seeds/branches, shared second-hop nodes, equal-distance
batch continuation, strategy selection, timeout/reset, and frontend filtering/reset.

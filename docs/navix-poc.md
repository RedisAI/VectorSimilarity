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
- Cache NO/MAYBE callback decisions by internal ID across batches. The cache grows
  through the highest checked ID using the iterator allocator: one byte per ID slot,
  plus vector capacity slack (one million slots hold 1 MB of decisions). Reset invalidates
  decisions and retains capacity for reuse; freeing the iterator releases the cache.
  Ordinary HNSW queries do not allocate it. New backend nodes from background migration
  are handled without reading the concurrently changing index size.
- Reuse O(D) scratch space, copy one adjacency list under one node lock, and poll
  the query timeout in both hops. There is no membership bitmap, extra seed,
  disconnected-component recovery, or fallback scan.

The consumer must apply its exact filter and request more batches when MAYBE candidates
fail it. Exhaustion can return fewer than k, even when more eligible documents exist.
Increasing ef cannot repair all disconnected filtered neighborhoods.

The tiered brute-force frontend applies the same callback before scoring. Counters cover
base-layer expansion and frontend scoring. HNSW filter checks/rejections count actual
callback evaluations, excluding cache hits; distance counts exclude upper-layer descent
and the initial seed distance. Counters accumulate across calls and resets. Top-k and
range APIs do not activate this POC; use the batch API. The integration validates only
unquantized, single-value FLOAT32 RAM HNSW.

`test_navix` covers rejected seeds/branches, shared second-hop nodes, equal-distance
batch continuation, strategy selection, timeout/reset, frontend filtering/reset, cached
NO/MAYBE decisions, reset invalidation, isolation between query iterators, and backend growth.

## Traversal experiments

The default and `forceBlind` behavior are unchanged. Three opt-in controls can be
combined, except that `bridgeRejected` takes precedence over the two-hop controls:

- `skipScanned`: skip second-hop adjacency lists already processed completely by this
  iterator, including lists processed as direct neighbors. Partial lists remain eligible.
  An additional byte per encountered internal-ID slot is retained across batches and
  cleared on reset. As with visited nodes, completed lists are not reconsidered for
  concurrent graph changes; compare these policies on a fully built, stable graph.
- `secondHopBudget`: maximum second-hop neighbor inspections per expansion (zero is
  unlimited). Already-visited and rejected neighbors consume the budget too. This bounds
  inspections, not total query work. Budget stops can reduce recall or exhaust the frontier;
  there is no automatic retry/fallback. Partial lists restart from their beginning on a
  later visit, so small budgets can repeatedly scan the same prefix.
- `bridgeRejected`: enqueue every direct neighbor not yet visited, including rejected nodes,
  and disable second-hop expansion. Only MAYBE nodes enter result heaps and affect the
  result bound. This uses one-hop graph navigation with filtered result admission;
  it can cross longer rejected chains, but does not guarantee exact recall or recovery
  from genuinely disconnected graph components. With no matches it may scan the whole
  reachable graph, subject to the query timeout. The existing NaviX stopping rule is
  retained: distance ties at a full result bound wait for the next batch (`>=`), unlike
  the strict `>` comparison in ordinary HNSW. This can defer a rejected bridge and a closer
  match behind it when the consumer stops after the first batch.

New counters report skipped complete second-hop lists, expansions stopped by the
inspection budget, and bridge-mode expansions. A budget exhaustion is counted only if
an additional inspection was prevented. The parameter/stat struct additions change the
C ABI; rebuild consumers with the matching headers and library. The flat frontend still
uses its existing callback filter and has no hop policy.

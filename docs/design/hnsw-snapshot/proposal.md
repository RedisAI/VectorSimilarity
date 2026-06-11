# Add a consistent, lock-free snapshot iterator for HNSW vector search

> Status: **draft proposal** for maintainer review. No code yet. A standalone
> design exploration and two working prototypes back this proposal — see
> [`docs/design/vecsim-hnsw-snapshot.md`](design-note.md)
> and `prototypes/`.

## Why

VecSim batch iteration over an HNSW index has two coupled problems in
multi-threaded mode:

1. **Long reads block writes.** A query holds the index read lock for its
   *entire* duration (`VecSimTieredIndex::acquireSharedLocks`, held across the
   whole `topKQuery`/batch iteration). For a long KNN scan — and especially a
   cursor-paginated aggregation (`FT.AGGREGATE … WITHCURSOR`) over a vector
   index, which can stay open across many `FT.CURSOR READ` calls — the
   background indexer and concurrent writers are stalled the whole time. This is
   a real throughput and tail-latency problem under mixed read/write load.

2. **Results are not point-in-time consistent.** VecSim explicitly documents
   that a query "may include vectors that were added after the query was
   submitted, with no guarantees." Results are therefore non-reproducible under
   concurrency, which complicates testing, debugging, and any read-your-writes
   reasoning.

The reason the lock is held for the whole query is that lock-free traversal
during writes is currently unsafe: the block backbone can be reallocated by a
concurrent insert, and slot reuse (SWAP) and in-place link rewrites can pull
data out from under a reader.

## What Changes

Introduce a **snapshot iterator** for HNSW: an iterator that is **constructed
under the read lock, then releases it**, and iterates **lock-free while
insertions and deletions proceed concurrently**, returning results **consistent
with the index state at construction time**.

User-visible surface:

- **Consistency guarantee strengthens.** A vector query / batch iterator (KNN,
  hybrid, and cursor-paginated vector aggregations) reflects the graph as of
  *query start*: vectors added after the query started are not included, and a
  vector deleted after start keeps its slot (the slot is not recycled while the
  snapshot is live), so a *different* element never appears in its place. One
  nuance: the deleted/in-process **flags** are weakly consistent — a delete that
  lands after start may or may not be observed (the same best-effort the live
  index already has); the strengthening is about *which set of vectors is
  visible* and *slot stability*, not strict deletion timing. The current "may see
  newer vectors" wording no longer applies when snapshotting is enabled.
- **Writers are no longer blocked for the duration of a vector read.** Inserts
  and deletes proceed while a query/cursor iterates, improving write throughput
  and tail latency under mixed load.
- **New configuration** *(planned, not yet built)* (tiered-HNSW scoped, mirroring
  `swapJobThreshold`) to enable snapshot iteration and to **bound snapshot memory
  / lifetime**, since a live snapshot retains superseded graph versions. (The
  vecsim layer currently exposes the opt-in as a per-query param; the config + cap
  are the remaining production surface.)
- **`FT.INFO` / debug exposure** *(planned, not yet built)* of snapshot state
  (active snapshots, retained snapshot memory) for operability.

### Non-goals

- **No change to the flat (brute-force) index** — its batch iterator already
  materializes scores at construction and is unaffected.
- **No change to non-vector iterators** (inverted index, numeric, tag).
- **No cross-shard snapshot coordination** in cluster mode beyond per-shard
  consistency; each shard snapshots independently, as it already executes
  independently.
- **No exact-recall guarantee change** — results remain approximate; the change
  is about *which set of vectors* is visible and *when the lock is held*, not
  about HNSW recall.

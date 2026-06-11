# Design: HNSW snapshot iterator

> Backed by the standalone design note
> [`docs/design/vecsim-hnsw-snapshot.md`](design-note.md)
> and two compiled/run prototypes under `prototypes/`. Numbers cited
> here come from those prototypes.

## Overview

A snapshot iterator captures an immutable, point-in-time view of the HNSW graph
under the index read lock, then iterates without holding any lock. Concurrent
writers never mutate **graph** data a snapshot can see — they create *new*
versions via copy-on-write, and old graph versions are reclaimed automatically
once no snapshot references them. This is a **persistent (rooted) copy-on-write**
structure with refcount-driven reclamation.

COW covers the graph container only. A slot's *vector* and *metadata* live in
separate, un-versioned containers that SWAP overwrites in place, so they need a
*second*, complementary mechanism — snapshot-gated slot reclamation — described
in its own section below. Getting this split right is the crux of the design:
**refcount COW for graph versions, horizon-gated deferral for slot recycling.**

Why not a cheaper trick (doc-id watermark, append-only edge log, prefix reads):
HNSW is **lossy on write** — inserting a node can evict an edge between two
existing nodes (`revisitNeighborConnections` → `getNeighborsByHeuristic2` →
`setLinks`), and delete/repair rewrites neighbor lists in place. A point-in-time
view cannot be reconstructed by filtering the live graph; the old bytes must be
**retained**. COW is the retention mechanism.

## Data structure

Replace the graph block storage

```
vecsim_stl::vector<DataBlock> graphDataBlocks;     // today
```

with a rooted, refcounted structure:

```
shared_ptr< vector< Block > >  root;          // contiguous header vector
   where  Block { shared_ptr<Buffer> data; }  // per-block refcounted buffer
```

Prototype B established the exact shape by measuring a dependent random walk
(1M elements, dim-128 distance per visit):

| layout | lookup-only vs flat | notes |
|---|---|---|
| `vector<DataBlock>` (today) | baseline | |
| `shared_ptr<vector<shared_ptr<Block>>>` | **+58–88%** | block headers scattered on heap |
| **`shared_ptr<vector<Block{shared_ptr buf}>>`** | **~0% (noise)** | contiguous headers, refcounted buffer |

The cost of the naive nested-`shared_ptr` form is the **per-block heap scatter**,
not the root indirection (which hoists out of the loop). Contiguous headers +
a refcounted data buffer recover flat-speed reads, so the chosen structure
imposes **no read regression** while keeping per-block COW granularity.

## Operations

- **Snapshot capture (under read lock):** `snap = root` — an O(1) `shared_ptr`
  copy. Release the lock. Iterate through `snap` with no lock.
- **No snapshot active:** `root`/buffers are unshared (`use_count == 1`) →
  writers mutate in place. **Zero overhead — identical to today.**
- **First write after a snapshot:** the backbone (header vector) must be
  preserved before its slots are repointed. Decide by **generation tag, not
  refcount**: each backbone carries `gen`; clone it once iff
  `max_live_snapshot_id >= backbone.gen`, stamp the clone with the current gen,
  swap `root`.
- **Content write to a block:** each block buffer carries `gen`; clone it iff
  `max_live_snapshot_id >= block.gen`, stamp the clone with the current gen,
  repoint its header slot, mutate the copy. A block written *after* all live
  snapshots (`block.gen > max_live_id`) is mutated in place — including repeated
  writes within the same generation (a freshly-cloned block carries the current
  gen, so subsequent writes to it stay in place).
- **Reclamation stays refcount-driven.** `shared_ptr` on the backbone + block
  buffers frees a superseded version automatically once the last snapshot
  referencing it drops. So the two halves split cleanly: **the generation tag
  decides *when to clone* (`max` live id); the refcount decides *when to free*
  (`min` live id, implicitly).** Same max/min duality as "Snapshot identity".

**Locking (as implemented).** The plain index takes writes under the exclusive
lock, so `max_live_snapshot_id` is trivially stable there. The **tiered** index,
however, wires concurrent inserts into the live graph under the *shared*
`mainIndexGuard` + per-node mutexes — so two things moved from the original
sketch: **capture runs under the *exclusive* lock** (a brief quiescence point at
which no writer is mid-insert, so the backbone fork + generation handout can't
race an in-flight write), and **per-block COW publishes the new buffer with an
atomic compare-exchange** (`std::atomic_*` free functions on the block handle —
`std::atomic<shared_ptr>` is GCC-12+), so two inserts COW'ing the same block
can't tear the handle; the generation stamp makes the CAS idempotent (the loser
adopts the winner's fork). The backbone is forked once **at capture** and stamped
the current generation, so it is never re-forked under the shared lock and stays
structurally stable for readers. After capture, the reader touches only its
private immutable handle — **no per-node locks**, and the only atomic on the read
path is the block-handle load, taken **only while a snapshot is live** (gated by
`anySnapshotLive()`; with none live the read is a plain pointer deref).

> **Why not `use_count` for the clone decision.** A *container-level*
> `root.use_count()` check goes **stale after the first backbone COW**: once the
> live backbone is cloned, `root.use_count() == 1` even while a snapshot still
> holds the old backbone — so a naive check wrongly mutates in place and corrupts
> the snapshot. A *per-block-buffer* `use_count` is technically sound (a snapshot
> transitively holds each buffer's `shared_ptr`), but it couples correctness to
> refcount discipline and false-positives on any transient buffer copy. The
> generation tag is refcount-independent and immune to both, which is why the
> clone decision uses it and refcount is reserved for reclamation.

> **What pins a block version (and why blocks don't need a horizon).** A snapshot
> captures the **whole backbone** (`root` = `shared_ptr<vector<Block>>`), and that
> vector holds a `shared_ptr<Buffer>` for **every** block. A block is therefore
> pinned by the *captured backbone*, **not** by the reader traversing to it — a
> block the snapshot never reaches is still held alive. This is correct only under
> two invariants, both worth a test:
> 1. **capture is eager and whole-backbone** (the handle holds `root`, not a
>    lazily-grown set of touched nodes); and
> 2. **backbone COW clones the header vector before repointing any slot**, so the
>    captured backbone is never mutated in place.
>
> Violate either and it becomes a use-after-free: a block overwritten before the
> reader reaches it loses its only ref and is freed. Given the invariants,
> `shared_ptr` is sufficient for graph blocks and **no horizon is needed for
> them** — and a horizon wouldn't help anyway (a consistent snapshot inherently
> pins all blocks as of T).
>
> The horizon **is** needed for the **vector + metadata** containers: those are
> not refcounted, so a snapshot holds **no** `shared_ptr` on a slot's embedding or
> flags — nothing pins them. That is exactly why slot reclamation is a separate,
> deferral/horizon-gated mechanism (see "Slot reclamation"), not a refcount one.

Prototype A demonstrated correctness: a snapshot stayed bit-identical
(checksum invariant) through 5.8M concurrent writes with no lock during
iteration, the live index diverged, and dropping the snapshot freed exactly the
COW'd-away versions.

## Snapshot identity: the generation id

Every snapshot carries a **monotonically increasing id** (a generation), handed
out from one global counter at capture time and never reused, plus the
**`curElementCount` it captured**. The index's `SnapshotRegistry` tracks the live
set. The clone and reclaim decisions use two different keys:

- **`newestLive()` = max(live ids)** is the **clone threshold**: a block/backbone
  version stamped `g` must be preserved (cloned on write) iff `newestLive() >= g`
  (some live snapshot predates, and so can still see, it).
- **`maxVisibleCount()` = max captured `curElementCount` over live snapshots** is
  the **reclaim horizon** for slot recycling. A snapshot only ever reads ids
  `< its curElementCount`, so a SWAP that overwrites internal id `d` is invisible
  to *every* live snapshot iff `d >= maxVisibleCount()` — then it recycles now;
  below it, the slot is deferred. (This is the count-based form that shipped. An
  earlier sketch keyed the horizon on `min(live ids)` + a per-slot
  free-generation; the curElementCount form is simpler and exact — what a snapshot
  can reach is defined by its count, not by id ordering.)

`graphSnapshotActive()` is the **degenerate gate**: "is the live set non-empty?"
(`newestLive() != 0`). It is what the boolean MVP used to defer *all* swaps; the
`maxVisibleCount()` horizon refines the reclaim end so compaction keeps flowing
under a long-lived cursor (high-id churn recycles immediately).

The id and the count are both maintained from the first branch that adds the
handle, so the clone decision (`newestLive`) and the horizon (`maxVisibleCount`)
drop in without reworking the capture API. The clone decision is **generation-tag
based, not `use_count`** (see the callout above); `use_count` is reserved purely
for reclamation.

## Slot reclamation — a separate problem COW does *not* solve

A snapshot must give a consistent view of an internal `id` (a "slot"), and a
slot's state is spread across **three parallel containers**, all indexed by the
same id (`block = id/blockSize`, `offset = id%blockSize`):

| state | container | COW'd by this design? |
|---|---|---|
| graph data (`ElementGraphData`: links, neighbors, mutex) | `graphDataBlocks` → the rooted `shared_ptr<vector<Block>>` | **yes** (above) |
| raw vector (the embedding) | `this->vectors` (`DataBlocksContainer`) | **no** |
| per-id metadata (`ElementMetaData`: label + flags incl. marked-deleted) | `idToMetaData` (flat vector) | **no** |

The refcounted COW versions only the **graph-data** container. The vector and
metadata containers are single-version and mutated **in place**. The dangerous
operation is **SWAP** (`SwapLastIdWithDeletedId`): to compact a delete it copies
the last element (`curElementCount-1`) into the deleted id's slot —

```
graph:    memcpy(getGraphDataByInternalIdForWrite(id), last_element, ...)  // COW-aware, safe
vector:   memcpy(getDataByInternalId(id),              last_data, ...)     // IN PLACE
metadata: idToMetaData[id] = idToMetaData[curElementCount]                 // IN PLACE
```

So a snapshot holding `id < snapshot.curElementCount` would read the *right*
links but the *wrong embedding* and a *wrong deleted-flag* — a different element
was moved into that slot underneath it. **Graph COW alone is insufficient for
slot safety.**

### Fix: defer the swap, don't COW the other two containers

COW-ing the vector + metadata containers as well would version the **largest**
data in the index (the embeddings) to protect against a comparatively rare event
(SWAP) — the wrong trade. The clean fix is to **not recycle a slot while a
snapshot references it**: leave the deleted slot as a tombstone and defer the
SWAP. A deferred slot keeps all three containers intact for the snapshot with
**zero extra COW**.

This is the job of the **horizon** — distinct from, and complementary to, the
refcount COW:

| problem | mechanism |
|---|---|
| graph-data version lifetime (link rewrites that don't move a slot) | **`shared_ptr` refcount COW** — automatic |
| slot/id recycling (SWAP overwrites vector + metadata in place) | **snapshot-gated SWAP deferral** — boolean now, horizon later |

Two levels, shipped in order:

1. **Boolean gate (MVP):** gate `removeAndSwap` / `executeReadySwapJobs` on
   `graphSnapshotActive()` — while *any* snapshot is live, defer all swaps
   (deletes become tombstones). Correct and trivial; its only cost is that one
   long-lived cursor stalls *all* compaction.
2. **Horizon refinement (when measured to matter):** stamp each freed slot with
   the generation at which it became free; recycle it once the **oldest live
   snapshot is newer than that stamp** (head of the owner list). This keeps
   compaction flowing under long-lived cursors instead of blocking it globally.
   Worth the owner-list + per-slot free-generation bookkeeping only if tombstone
   buildup under open cursors proves to be a real problem. (For the clone
   decision the *newest* live snapshot — the list tail — is the relevant end; for
   slot reclamation it is the *oldest* — the head.)

### Deleted-flag consistency (decide explicitly)

The metadata deleted-flag is flipped **in place** by every `markDelete`, not just
by SWAP. So even with swap deferral, a snapshot can observe an element that was
live at capture as *deleted* (a concurrent delete flipped its shared flag). That
is acceptable under a "no *new* vectors after T, crash-safe" contract, but it is
**not** strict as-of-capture consistency. Choosing strict would require
versioning/snapshotting the flag too. This is a deliberate contract decision —
see Open questions.

## Subsystems touched

1. **Block storage & graph data.** *(implemented)*
   - `graph_data.h`: `ElementGraphData` is made **deep-copyable** via
     `ElementGraphData::copyTo` + `ElementLevelData::copyInto`: a field-wise deep
     copy that re-initializes a fresh `neighborsGuard` mutex on the copy (lock
     state never aliased), copies the flexible-array link lists by `recordSize`,
     deep-copies each level's `incomingUnidirectionalEdges`, and walks the
     separately-allocated `others` upper levels. The mutex **stays in the struct**;
     the "side array keyed by id" idea was spiked and rejected (it fixes lock
     identity but not the non-COW'd write — see design-note cost #2).
   - `graph_data_blocks.h` *(new file)*: `RootedCowStore<T>` (the reusable
     `shared_ptr` root + generation tag + `SnapshotRegistry`), `GraphBlockBuffer`
     (the refcounted per-block buffer), `GraphDataBlocks` (the contiguous header
     vector + COW write paths, with the atomic CAS-fork in `cowBlock`), and the
     `HNSWGraphSnapshot` handle. `MetaDataStore` (in `hnsw.h`) is built on the
     same `RootedCowStore<T>` so the per-id metadata grows by COW instead of
     reallocating, and is read through an atomically-published pointer.
   - `hnsw.h`: `getGraphDataByInternalId` resolves through the root;
     `getGraphDataByInternalIdForWrite` COWs backbone+block before returning a
     writable pointer. The delete/**repair** path is the subtle one:
     `mutuallyUpdateForRepairedNode` **COWs every node it will touch up front,
     before taking any per-node lock**, so a held lock's buffer can't be forked
     out from under it (the mutex-identity hazard). This is what lets repair run
     **while a snapshot is live** rather than being deferred.
2. **Batch iterator.** *(implemented — plain + tiered, opt-in)* Rather than
   branch the hot live iterator, a **dedicated `HNSWSnapshot_BatchIterator`**
   (+ single/multi subclasses) reads entirely through the captured handle with no
   per-node locks; `hnsw_batch_iterator.h` is left untouched (zero perf/risk to
   the default path). `newBatchIterator` captures the snapshot (under the lock,
   released immediately) and dispatches to it **when the opt-in
   `useGraphSnapshotIterator` query param is set**; otherwise the live iterator is
   unchanged.
3. **Tiered index.** *(implemented)* The cursor takes `mainIndexGuard`
   *exclusively* only to capture the backend snapshot, then **releases it and
   iterates lock-free** — ingestion/GC are no longer blocked for the cursor's
   life. SWAP slot reuse (`executeReadySwapJobs`) is gated by the
   **`maxVisibleCount()` reclaim horizon** (the boolean `graphSnapshotActive()`
   gate was the MVP; the horizon shipped): a freed id `>=` the horizon recycles
   immediately, below it defers. Repair runs under the snapshot (COW-before-lock,
   item 1), so the horizon is actually reached rather than stalled behind globally
   deferred repairs. The vector + metadata containers are still overwritten in
   place — the horizon, not COW, is what keeps a deferred slot valid.
4. **RediSearch boundary.** *(not yet implemented)* `src/iterators/hybrid_reader.c`
   (and the vecsim wrappers) would construct the iterator under the lock and
   release it for iteration, and hold the snapshot across `FT.CURSOR READ`. Also
   needs a string param-resolver token so `FT.SEARCH` can flip
   `useGraphSnapshotIterator` (today it is only settable programmatically).
5. **Config & info.** *(not yet implemented)* enable flag + snapshot
   memory/lifetime cap, and `FT.INFO`/`FT.DEBUG` exposure of active-snapshot count
   and retained memory.

## Edge cases

- **Index grows during iteration** (`growByBlock` appends a block): the snapshot
  references the old header vector / `curElementCount` at T, so new ids are
  simply outside its view. No new vectors appear in results — exactly the
  intended consistency.
- **Element deleted during iteration:** marked-deleted; the snapshot still holds
  the old buffer, so the vector remains visible (consistent with T). Physical
  removal (SWAP) is deferred until no snapshot references the slot.
- **Long-lived snapshot (open cursor) pins memory:** retained versions
  accumulate. Bounded by a configurable snapshot lifetime / memory cap; on
  breach, either refuse new snapshots or fail the cursor with a clear error
  (decision deferred to review — see Open questions).
- **Snapshot outlives the index drop:** the `shared_ptr` graph keeps the
  referenced blocks alive until the last iterator is freed; index teardown must
  not assume blocks are gone.
- **Multiple concurrent snapshots at different generations:** each holds its own
  root; refcounts naturally keep every still-referenced version alive.

## Alternatives considered

- **Doc-id watermark / "traverse edges added before T".** Rejected: HNSW evicts
  and rewrites old nodes' edges in place, so filtering the live graph cannot
  reconstruct the topology at T.
- **Append-only edge list + prefix read.** Rejected: HNSW is bounded-degree and
  prunes; making edges append-only either breaks the algorithm or degenerates
  into a versioned log with O(log-length) replay on the hot read path.
- **Mutate-in-place + per-snapshot overlay.** Rejected: a lock-free reader that
  misses the overlay and reads the live block can *race* a writer that
  inserts-then-mutates that block, yielding a newer-than-T read. Only immutable
  versions are cleanly lock-free.
- **`shared_ptr<vector<shared_ptr<Block>>>` (per-block pointers).** Rejected as
  the storage layout: +23% end-to-end read regression from heap scatter
  (prototype B). Kept the per-block COW idea but moved to contiguous headers +
  refcounted buffers.
- **Eager full copy at construction / `fork()`.** O(N) per query, or
  process-fork semantics. `fork()` remains the right tool for *batch*-scoped
  snapshots (GC/RDB) and is explicitly out of scope here (this is per-query).
- **Safety-only weak consistency** (gate reclamation, no COW): a viable, much
  cheaper fallback if the trivially-copyable-block refactor proves too invasive.
  It is lock-free and crash-safe but does not isolate edge drift among existing
  nodes. Recorded as the fallback in the design note.

## Testing strategy

- **C++ unit (`tests/cpptests`):** COW correctness (snapshot invariant under
  concurrent writes), automatic reclamation (no leak after snapshot drop),
  backbone-growth-during-snapshot, delete-during-snapshot visibility.
- **Concurrency / sanitizers:** the snapshot-vs-writer tests under
  `SAN=address` and TSan to prove the lock-free read path is race-free.
- **Python end-to-end (`tests/pytests`):** KNN / hybrid / `WITHCURSOR` over a
  vector index returning consistent results while a writer adds/deletes
  concurrently; writer progress is observable during a long read (lock released).
- **Benchmarks (`microbenchmark` label):** read-path regression check (must stay
  ~flat per prototype B) and mixed read/write throughput improvement.

## Open questions (resolved during implementation, except where noted)

1. **Opt-in vs default.** → **Resolved: opt-in.** A query param
   (`useGraphSnapshotIterator`, default off) selects the snapshot iterator; the
   default path is byte-for-byte unchanged.
2. **Memory-cap policy on breach.** → **Still open (not implemented).** No cap
   yet bounds a long-lived cursor's retained memory; a stuck cursor pins its
   as-of-capture memory and defers below-horizon slots indefinitely. This is the
   main "trust it in production" gap.
3. **Granularity of COW.** → **Resolved: per-block buffer**, as proposed. No
   block-copy churn observed in the unit/concurrency tests; revisit only if a
   real workload shows it.
4. **Deleted-flag consistency level.** → **Resolved: weak** — "no new vectors
   after T, crash-safe, slot-stable." Strict deleted-flags (versioning the
   metadata flag) was deliberately not built. See `spec-delta.md`.
5. **Slot deferral granularity.** → **Resolved: horizon shipped.** The boolean
   `graphSnapshotActive()` gate was the MVP; the shipped reclaim horizon is
   `maxVisibleCount()` (max captured `curElementCount` over live snapshots), so
   one open cursor no longer stalls all compaction. (Note the shipped horizon is
   count-based, not the min-live-id + per-slot-free-generation form sketched
   earlier — see "Snapshot identity".)

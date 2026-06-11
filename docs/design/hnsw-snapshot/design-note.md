# Design: Consistent lock-free snapshot iterator for HNSW (VecSim)

Status: **draft / prototype**. This is a design exploration, not a committed plan.

## Problem & requirement

A query/batch iterator over an HNSW index should be:

1. **Constructed under the index read lock**, then **release the lock**, and
2. **iterate while concurrent insertions/deletions run**, ideally
3. returning results **consistent with the index state at construction time**.

Today the multi-threaded path (tiered index `acquireSharedLocks`,
`hnsw_tiered.h`) holds the shared lock for the *whole* query precisely because
lock-free traversal during writes is unsafe: the block backbone can be
reallocated, and slot reuse (SWAP) / in-place link rewrites can pull data out
from under a reader. Releasing the lock is exactly the guarantee we must
replace.

## Why "filter the live graph" cannot work

HNSW is **lossy on write**: inserting a node can *evict* an edge between two
existing nodes (`revisitNeighborConnections` → `getNeighborsByHeuristic2` →
`setLinks`), and delete/repair rewrites neighbor lists in place
(`removeLink`, swap-remove). So a point-in-time view cannot be reconstructed by
reading the live structure and filtering (by doc-id watermark, by prefix of an
"append-only" list, etc.) — the old information is destroyed. **A snapshot must
retain the old bytes.** The only real choice is the retention mechanism.

## Chosen mechanism: persistent (rooted) copy-on-write

Store the graph blocks behind a **shared, rooted structure**. Prototype B
(below) settled the exact shape: block headers must stay **contiguous** (a
`vector<shared_ptr<DataBlock>>` scatters them across the heap and costs +25–60%
on traversal). The structure that keeps flat-speed reads *and* per-block COW is
contiguous headers each holding a **refcounted data buffer**:

```
shared_ptr< vector< Block > >          // root over a CONTIGUOUS header vector
   where  Block { shared_ptr<Buffer> data; }   // per-block refcounted buffer
```

- **root** (`shared_ptr<...>`) — swapped rarely (once per generation that writes).
- **header vector** — contiguous; copied (headers only, cheap) on backbone COW.
- **block buffer** (`shared_ptr<Buffer>`) — the edges; COW'd per block on write.

Reads are `root` (hoisted) → contiguous `vec[b]` → `data.get()` → buffer — the
same access shape as today's flat `vector<DataBlock>`.

### Operations

- **Snapshot capture** (under the *write*/exclusive lock — see "Locking model"):
  hand out a generation id, **fork the backbone** (copy the header vector,
  `N/blockSize` pointers, so the live index keeps mutating a private copy while
  the snapshot owns the captured one), and snapshot the vector-block base
  pointers. O(#blocks); no element/vector data is copied. Release the lock, then
  iterate through the captured handle with **no lock**.
- **No snapshot active**: every version's generation post-dates all (zero) live
  snapshots → writers mutate in place. **Zero overhead — current behavior.**
- **Backbone after capture**: the fork is stamped the *current* generation, so a
  concurrent writer's backbone-COW check is a no-op — the live backbone is never
  re-forked under the shared lock and stays structurally stable for lock-free
  readers. Thereafter only per-block buffers are COW'd.
- **Content write to a shared block**: clone the block buffer (deep-copy via
  `copyTo`) iff a live snapshot can still see it, repoint its slot, mutate the
  copy. Under concurrent tiered inserts (which mutate the live graph under the
  *shared* lock + per-node mutexes) the slot is repointed with an **atomic
  compare-exchange**, so two writers racing to COW the same block can't tear the
  handle or lose a fork (the generation stamp makes the CAS idempotent — the
  loser adopts the winner's fork).
- **Cleanup**: **automatic** via refcounts. When the snapshot is dropped, old
  blocks/backbones with no remaining references free themselves. No
  hand-rolled epoch/horizon/retire-queue.

**Clone decision: a generation tag, not `use_count`.** The initial sketch keyed
the clone on `shared_ptr` `use_count` (`> 1` ⇒ shared ⇒ clone). That breaks: a
container-level `use_count` goes **stale after the first COW** — the live root is
unshared again while a snapshot still holds the *old* one — so it would wrongly
mutate in place and corrupt the snapshot. Instead every version carries the
**generation it was last written in**; a write clones iff `newestLive() >= gen`
(some live snapshot predates it) and stamps the clone with the current
generation. `use_count` is reserved purely for **reclamation**.

**Locking model (revised for concurrency).** The first design assumed all writes
run under the *exclusive* lock and capture under the *shared* lock (mutually
exclusive), so the clone decision would need no atomics. That holds for the plain
index, but the **tiered** index wires concurrent inserts into the live graph
under the *shared* `mainIndexGuard` + per-node mutexes. So the implementation
moved two things: **capture runs under the *exclusive* lock** (a brief quiescence
point — no writer is mid-insert — where the backbone fork + generation handout
can't race an in-flight insert), and **per-block COW publishes via an atomic
CAS** so concurrent inserts COW'ing the same block are safe. After capture the
reader still touches only its private immutable handle — no per-node locks, and
no atomics on the snapshot read path.

## Consistency guarantee

The shipped guarantee is **strict for graph topology + membership + vectors**,
**weak (best-effort) for per-element metadata flags**:

- **Strict as-of-construction** for the graph the snapshot traverses: the set of
  ids that existed at capture and their links/levels, plus each visible id's
  vector bytes, are frozen — concurrent inserts COW, so they never perturb the
  captured backbone/buffers. This is the property the requirement asks for and
  the one the lock-free reader depends on.
- **Weak** for the deleted / in-process **flags**: `markDelete` flips the flag in
  place on the single-version metadata, so a delete that lands after capture MAY
  or MAY NOT be observed by the reader (it tolerates either value). This matches
  the live index's existing weak flag consistency; making it strict would mean
  versioning the metadata flag too, which we chose not to do (see
  `spec-delta.md`, "Deleted-flag consistency").

A cheaper **safety-only** variant (drop COW entirely, just capture the backbone +
gate slot reuse: "no crashes, no new nodes after T, but edges among old nodes may
drift") was the recorded fallback; the implementation took the consistent COW
version above.

## Costs & open problems

1. **Read-path indirection.** `root → backbone[i] → block → data` adds one
   pointer hop vs today's flat `vector<DataBlock>` offset math, and the block
   headers stop being contiguous. *The element-data locality is unchanged*
   (it already lives in a separate `DataBlock::data` heap buffer). Net cost must
   be measured — see prototype B.
2. **Block must be deep-copyable for block-COW.** `ElementGraphData` embeds a
   `std::mutex` (can't be copied) and a pointer to separately-allocated upper
   levels. **Resolved by an explicit `copyTo` deep-copy** (re-inits a fresh mutex
   on the copy, deep-copies each level's incoming-edges, walks the `others` upper
   levels) — the mutex **stays in the struct**. The originally-sketched "move the
   mutexes into a side array keyed by id" was implemented as a spike and
   **rejected**: a stable side-array lock fixes lock *identity* but repair still
   writes non-COW'd buffers, so it was neither necessary nor sufficient — the real
   requirement is to **COW every touched block before taking any per-node lock**
   (so a held lock's buffer can't be forked out from under it). With COW-before-
   lock in place, deep-copy + in-struct mutex is sufficient and churns zero call
   sites. (The mutex-identity hazard, and why the side array doesn't solve it, is
   the central finding of the delete/repair work — see `design.md` "Subsystems".)
3. **Pinned snapshot → retained memory.** A long-lived snapshot (open
   `FT.CURSOR`) keeps old blocks alive; COW'd blocks double until it drops.
   Mitigation = cap snapshot lifetime. (Same issue under any retention scheme.)
4. **Granularity.** Block-COW copies a whole ~200 KB block to preserve one
   node's edges. Fine if writes touch few blocks during a snapshot; approaches a
   full copy if writes are spread across the whole index.

## Feasibility: PROVEN (end-to-end POC)

`prototypes/snapshot_poc.cpp` is a minimal graph index built on the **real
VecSim `ElementGraphData`/`ElementLevelData`** (it `#include`s the production
`graph_data.h`) and the **real `copyTo` primitive** added in Phase 1, with the
rooted-COW block storage. It demonstrates the full requirement end to end:
construct a snapshot under a read lock, release it, and traverse lock-free while
a writer thread adds nodes and rewires edges (COW-ing real blocks via `copyTo`).

Results (compiled against the real headers; `memory/vecsim_malloc.cpp` +
`memory/vecsim_base.cpp` linked):

- Snapshot query stayed **bit-identical** across thousands of lock-free reads
  during ~1–2.4M concurrent writer ops; the live graph provably diverged.
- **ThreadSanitizer: clean** (no data race) — the lock-free read path is sound.
- **ASan/UBSan/LeakSan: clean** — the `copyTo` FAM logic and the refcounted
  buffer deleter are memory-safe, no leaks.
- Allocator bytes grew while the snapshot was held (pinned COW versions) and
  **dropped on release** (automatic reclamation).

This answers "is it implementable against VecSim's actual node representation?"
— yes. The remaining work is integration scope (wiring COW through the real
HNSW read/write paths, the batch iterator, the tiered index, and RediSearch),
not a feasibility unknown.

## Prototype findings

Prototypes are in `prototypes/` (standalone). A/B/C use `g++ -O2 -std=c++20`;
the POC additionally links two VecSim `memory/*.cpp` files (see its header).

- **C — `block_deepcopy.cpp`:** validates the `copyTo` algorithm (FAM links +
  fresh mutex + independent incoming-edges) on a multi-level node. PASSED.

**A — `cow_snapshot.cpp` (mechanism): PASSED.** O(1) capture; a snapshot stayed
bit-identical (checksum invariant) through **2.4M concurrent writes** with no
lock during iteration; the live index diverged; dropping the snapshot freed
exactly the COW'd-away versions automatically (276 → 212 live blocks). Confirms
the refcount-driven COW is correct, lock-free for readers, and self-cleaning.

**B — `read_indirection_bench.cpp` (read cost): layout matters a lot.**
Dependent random walk, 1M elements, dim-128 distance per visit:

| layout | lookup-only | vs flat |
|---|---|---|
| `vector<DataBlock>` (today, flat) | ~9.9 ns | — |
| `shared_ptr<vector<Block>>` (blocks by value) | ~8.2 ns | ~0% |
| **`shared_ptr<vector<Block{shared_ptr buf}>>`** (chosen) | ~9.0 ns | **~0%** |
| `shared_ptr<vector<shared_ptr<Block>>>` | ~15.7 ns | **+58%** |

With a realistic distance calc per visit, the scattered `shared_ptr<Block>`
layout still cost **+23%** end-to-end. **Conclusion: the cost is the per-block
`shared_ptr` heap scatter, not the root indirection.** Contiguous headers +
refcounted buffer recover flat-speed reads, so the chosen structure imposes
**no read regression** while keeping per-block COW.

### Decision recorded: why not a "mutate-in-place + per-snapshot overlay"

An overlay (keep the live index flat, copy a block into a side map just before a
writer mutates it in place) would also avoid the read tax. **Rejected:** a
lock-free snapshot reader that misses the overlay and reads the live block can
**race** a writer that inserts-then-mutates that same block, yielding a
*newer-than-T* read. In-place mutation is not cleanly lock-free. The rooted COW
wins because snapshots read **immutable** versions — writers only ever create
*new* blocks, never touch a block a snapshot holds. This is the property that
makes "release the lock and iterate during writes" actually safe.

## Fallback

If the trivially-copyable-block refactor (cost #2) proves too invasive, fall
back to **safety-only weak consistency**: keep the flat layout, capture the
backbone + `curElementCount` under the lock, gate slot reuse by an owner
horizon. Lock-free and crash-safe, excludes nodes added after T, but does not
isolate edge drift among existing nodes.

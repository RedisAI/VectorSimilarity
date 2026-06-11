# HNSW snapshot iterator — design

Design exploration for a **consistent, lock-free snapshot iterator** over HNSW:
construct an iterator under the index read lock, release the lock, and iterate
while concurrent inserts/deletes proceed — with results consistent as of
construction time. Mechanism: persistent (rooted) copy-on-write where the
**clone decision is generation-tag-driven** and **reclamation is refcount-driven**
(superseded versions free themselves when the last snapshot referencing them
drops); slot recycling (SWAP) is gated by a separate reclaim horizon.

The foundational change (the block deep-copy primitive `ElementGraphData::copyTo`
/ `ElementLevelData::copyInto`) is implemented in
`src/VecSim/algorithms/hnsw/graph_data.h` on this branch.

## Contents

| file | what it is |
|---|---|
| `proposal.md`   | Why + what (the user-visible surface). No code. |
| `design.md`     | Implementation plan: subsystems touched, data model, alternatives considered/rejected, testing strategy, open questions. |
| `design-note.md`| Deeper technical exploration: why "filter the live graph" fails, the chosen `shared_ptr<vector<Block{shared_ptr buf}>>` structure, prototype + benchmark + POC findings, costs. |
| `tasks.md`      | Phased implementation checklist (phase 1 = the `copyTo` primitive, already started). |
| `spec-delta.md` | Behavior spec: requirements + WHEN/THEN scenarios. |
| `prototypes/`   | Standalone C++ programs (below). |

## Prototypes

All build with `g++ -O2 -std=c++20 -pthread`. `snapshot_poc.cpp` additionally
links two `src/VecSim/memory/*.cpp` files (see its header comment).

| prototype | proves |
|---|---|
| `cow_snapshot.cpp`          | rooted-COW snapshot mechanism: O(1) capture, consistent lock-free reads under concurrent writes, automatic cleanup |
| `read_indirection_bench.cpp`| read-path cost of candidate layouts — the per-block `shared_ptr` scatter is the cost; contiguous headers + refcounted buffer read at flat speed |
| `block_deepcopy.cpp`        | the `copyTo` deep-copy algorithm (FAM links + fresh mutex + independent incoming-edges) on a multi-level node |
| `snapshot_poc.cpp`          | **end-to-end feasibility** using the REAL `ElementGraphData` + `copyTo`: snapshot stays consistent through millions of concurrent writes; clean under TSan/ASan/UBSan |

## Status

The **vecsim integration is implemented and validated** (a stacked series of
branches, `snapshot/01`–`10`): the COW block storage + generation-tag clone, the
O(1)-ish snapshot capture, the lock-free KNN/batch iterator (plain **and**
tiered, opt-in), concurrent-insert safety (atomic block-COW), the swap reclaim
horizon, and COW-before-lock repair so deletes' graph repair runs under live
cursors. Full unit suite green; the concurrent insert+delete+repair+cursor repro
is clean under ASan. The original prototypes (below) proved feasibility first.

**Not yet built** (tracked in `tasks.md` Phases 5–7): RediSearch wiring
(`hybrid_reader.c`, holding the snapshot across `FT.CURSOR READ`, a string
param-resolver token), the snapshot memory/lifetime cap + `FT.INFO`/`FT.DEBUG`
observability, range-query snapshot support, and production-scale benchmarks.
`spec-delta.md` uses the OpenSpec requirement/scenario shape from the RediSearch
repo, which is framework-neutral.

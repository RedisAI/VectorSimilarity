# Tasks: HNSW snapshot iterator

> Each top-level task is roughly one reviewable PR/commit. Phases 1–2 are the
> risky structural work; later phases depend on them. Test tasks are explicit: a
> change is not done until new behavior is covered and the suites are green.

## 1. Make the graph block deep-copyable (`deps/VectorSimilarity`)

> **Decision (changed during implementation):** instead of moving the mutex out
> to a side array (original 1.1), the block is made *deep-copyable* via an
> explicit `copyTo` that re-initializes a fresh mutex on the copy and deep-copies
> each level's incoming-edges vector. Rationale: the side-array refactor touches
> every `lockNodeLinks`/`getElementLevelData` call site (large blast radius, high
> miscompile risk), whereas value-semantic deep copy achieves the same goal —
> a block buffer that can be COW'd without sharing mutable state — with **zero**
> call-site churn. The mutex stays in `ElementGraphData`; the copy never aliases
> its lock state (COW runs under the exclusive write lock, before any
> per-node lock is taken). Algorithm validated in
> `docs/design/prototypes/block_deepcopy.cpp`.

- [x] 1.1 Add `ElementGraphData::copyTo` + `ElementLevelData::copyInto` deep-copy
      primitives (fresh mutex, independent incoming-edges, FAM-aware) in
      `graph_data.h`; compiles against the real headers (`-fsyntax-only`)
- [x] 1.2 Handle upper levels (`others`) in `copyTo`; the
      `incomingUnidirectionalEdges` copy uses the index's allocator
- [x] 1.3 C++ unit test in `tests/unit` exercising `copyTo` on a
      multi-level node (contents equal, storage independent, source-mutation
      isolation)
- [x] 1.4 Build VecSim unit suite green with the primitive in place

## 2. Rooted copy-on-write block storage

- [x] 2.1 Refcounted `GraphBlockBuffer` + the rooted `shared_ptr<vector<Block>>`
      storage (`RootedCowStore<T>` + `GraphDataBlocks` in `graph_data_blocks.h`)
- [x] 2.2 Resolve `getGraphDataByInternalId` / `getElement` through `root`
- [x] 2.3 Backbone COW (forked once at capture; never re-forked under the shared
      lock — see 4.6 / design.md "Operations")
- [x] 2.4 Per-block buffer COW in the write paths (insert + repair), with the
      atomic CAS-fork for concurrent inserts
- [x] 2.5 C++ unit test: writes COW only when a snapshot shares the block;
      in-place when unshared (`cowStorageSnapshotIsolation`,
      `cowDecisionByGenerationNotUseCount`)
- [~] 2.6 Read-path cost: the `blockData` raw-pointer return keeps the no-snapshot
      path a plain deref (gated on `anySnapshotLive()`); informal microbench shows
      ~flat at realistic dims. A formal `microbenchmark`-label run at production
      dimensions is still outstanding.

## 3. Snapshot handle + capture under the lock

- [x] 3.1 `HNSWGraphSnapshot` handle (captured `root`, entry point, `maxLevel`,
      `curElementCount`, captured vector-block pointers + metadata root)
- [x] 3.2 `captureGraphSnapshot()` captures under the lock; the caller releases it
- [x] 3.3 A **dedicated** `HNSWSnapshot_BatchIterator` resolves all node access
      through the handle (the live `hnsw_batch_iterator.h` is left untouched)
- [x] 3.4 C++ unit tests: capture + iteration touch only the snapshot
      (`snapshotHandleCapture`, `snapshotIterator_*`)
- [x] 3.5 **Give the snapshot a monotonic generation id** (+ its captured
      `curElementCount`). Handed out from one counter at capture (never reused);
      the `SnapshotRegistry` tracks the live set. Clone threshold = `newestLive()`
      = max(live ids); reclaim horizon = `maxVisibleCount()` = max captured
      `curElementCount` (NOT `min(live ids)` — see design.md "Snapshot identity").

## 4. Lock-free iteration & reclamation

> **Status: IMPLEMENTED (vecsim layer), across branches `snapshot/06`–`10`.**
> Phases 1–3 landed the storage mechanism (deep-copy primitive, rooted COW block
> storage, O(1) snapshot handle + `captureGraphSnapshot()`). Phase 4 then took the
> production lock-free iterator the whole way: plain + tiered, opt-in, with
> concurrent inserts/deletes/repair/swap all proceeding during a live cursor. The
> concurrent insert+delete+repair+cursor repro
> (`parallelBatchIteratorSearchSnapshot`) is clean under ASan.
>
> **The hard-won finding that shaped it.** An early attempt to make the production
> iterator capture a snapshot deadlocked under `parallelBatchIteratorSearch`: the
> tiered insert path mutates the live graph **in place under a *shared* lock**,
> relying on stable node addresses + per-node `neighborsGuard` mutexes. With a
> snapshot live, a concurrent insert COWs a block under that shared lock — moving
> a node (and its mutex) to a new buffer while a holder is mid-critical-section,
> so `lockNodeLinks(old)`/`unlock(new)` resolve to different mutexes (lock leak →
> deadlock; torn reads → crash). The resolution had two parts, and notably it is
> **NOT** "serialize all writers under the exclusive lock":
> - **Capture runs under the *exclusive* lock** — a brief quiescence point where
>   no writer is mid-insert — and **forks the backbone** there, so the live
>   backbone is never re-forked under the shared lock (stays structurally stable
>   for readers). Concurrent inserts keep running under the *shared* lock.
> - **Per-block COW publishes via an atomic compare-exchange** (generation-tag
>   idempotent), so two inserts racing to COW the same block are safe without a
>   global writer serialization.
> - **Repair COWs every touched node's block BEFORE taking any per-node lock**
>   (`mutuallyUpdateForRepairedNode`), so a held lock's buffer can't be forked out
>   from under it — this is what lets repair run *under* a live snapshot instead of
>   being deferred.
> - **Reader reads the immutable snapshot with NO per-node locks** via a dedicated
>   `HNSWSnapshot_BatchIterator` (the live `hnsw_batch_iterator.h` is untouched).
> - **SWAP slot reuse gated** by the `maxVisibleCount()` reclaim horizon, so a
>   freed id a snapshot can still see stays valid (and the vectors container needs
>   no COW).
> - **All gated behind the opt-in query param** so the default path is unchanged.
>
> Validated under **ASan** (memory safety on the concurrent path). A fully clean
> **TSan** run is not claimed: pre-existing relaxed flag-byte races (4.4a) remain.

- [~] 4.1 Lock-free snapshot read path. **Done (core capability):**
      `HNSWIndex::topKFromSnapshot(snap, query, k, params)` runs the full two-phase
      HNSW KNN (greedy upper-level descent + ef-bounded level-0 best-first) reading
      **all** graph access through the captured `HNSWGraphSnapshot` with **no
      per-node locks** and a local visited set — it touches no live graph or mutex.
      Proven by `HNSWTest.lockFreeSnapshotQuery`: on the same graph it returns
      results identical to the live `topKQuery`, and run with no lock it stays
      point-in-time consistent while a writer copy-on-writes the live graph
      underneath (TSan-clean, 0 races). The snapshot also captures the **vectors**
      container's per-block base pointers (`HNSWGraphSnapshot::getVectorData`), so
      the query reads vector data without touching the live (reallocating) vectors
      container — TSan confirmed this removes the vector-data race.
      **Third per-id container now closed (Stage A):** `idToMetaData`
      (deleted-flags + labels) used to be a flat vector that *fully* reallocs on
      growth (`resizeIndexCommon`); the query read it via `isMarkedDeleted` /
      `isInProcess` / `getExternalLabel`, so a query during a concurrent **insert**
      raced there (TSan-confirmed: `isMarkedAs` vs `resizeIndexCommon`). It is now
      wrapped in `MetaDataStore`, backed by the same rooted/generation-tagged COW
      as the graph (see consolidation note below): capture is O(1), growth COWs
      instead of reallocating, and `topKFromSnapshot` reads flags + labels from the
      **captured** metadata (`snap.metaData`), not the live index. Proven by
      `HNSWTest.lockFreeSnapshotQueryDuringInserts` — a writer doing real
      `addVector`s (growing all three per-id containers) cannot perturb a
      lock-free snapshot read; TSan-clean.
      **Done:** wired into the production batch iterator via the dedicated
      `HNSWSnapshot_BatchIterator` (plain, branch 06) and the tiered cursor that
      captures-then-releases the main lock (branch 07), behind the opt-in
      `useGraphSnapshotIterator` param.

      **COW consolidation (Stage A).** The rooted-COW + generation-tag mechanism
      that was specific to the graph backbone is now a single reusable
      `RootedCowStore<T>` (in `graph_data_blocks.h`): `shared_ptr<T>` root +
      generation tag + `SnapshotRegistry`, exposing `capture()` (O(1)),
      `cowForWrite(cloneFn)` (clone iff `newestLive() >= gen`, stamp with
      `currentGeneration()`), `initRoot`/`reset`/`mustClone`. Both the graph
      backbone (`GraphDataBlocks`) and the metadata (`MetaDataStore`) are built on
      it. The vectors container stays on the captured-per-block-pointer scheme
      (append-only blocks keep stable addresses; gated SWAP keeps ids stable), which
      `lockFreeSnapshotQueryDuringInserts` confirms is race-free — so it is not
      migrated to `RootedCowStore<T>`.
- [x] 4.2 SWAP-reuse gating primitive (`graphSnapshotActive()`), now backed by
      the live-id registry (not `root.use_count()`, which goes stale after the
      first COW) and **wired into `executeReadySwapJobs`** (see 4.7a).
- [x] 4.3 Verify automatic reclamation: dropping the snapshot frees superseded
      versions (covered by `cowStorageSnapshotIsolation` + `snapshotLockFree-
      Consistency`)
- [~] 4.4 Concurrency test under **TSan** (clang; g++ libtsan segfaults in this
      sandbox). **Run and clean** for the snapshot paths — 0 data races in
      `snapshotPreservedUnderConcurrentInserts` (real concurrent inserts + a
      lock-free snapshot read), `snapshotLockFreeConsistency`,
      `cowDecisionByGenerationNotUseCount`, `snapshotPinsUntraversedBlock`, and
      `swapDeferredWhileSnapshotHeld`. Two findings, neither in snapshot code:
      (a) **pre-existing**: `parallelBatchIteratorSearch` reports 26 data races in
      the existing tiered concurrent insert+search path (e.g. a non-atomic read of
      `ElementMetaData::flags` IN_PROCESS in `processCandidate` vs the atomic write
      in `unmarkInProcess`) — all in `hnsw.h`/`hnsw_tiered.h`, none touching
      `graph_data_blocks.h` / `SnapshotRegistry` / the gen tags; never TSan-checked
      before. `swapJobBasic` (same delete/repair path, no snapshot) is TSan-clean.
      (b) the delete/**repair** path was not COW-safe under a live snapshot — now
      **fixed** (4.5, branch 10: COW-before-lock). The full production path
      (concurrent insert + delete + repair + lock-free cursors,
      `parallelBatchIteratorSearchSnapshot`) is now validated **clean under ASan**.
      Still outstanding: a fully clean **TSan** run — blocked only by the
      pre-existing (a) relaxed flag-byte races, not by snapshot code.
- [x] 4.5 Convert all graph-mutation sites to COW-aware (COW-before-reference).
      **Insert path** via `getElementLevelDataForWrite` /
      `getGraphDataByInternalIdForWrite` in `mutuallyConnectNewElement`,
      `revisitNeighborConnections`, `mutuallyRemoveNeighborAtPos` (proven by
      `snapshotPreservedUnderConcurrentInserts`). **Delete/repair path** completed
      in branch 10: `mutuallyUpdateForRepairedNode` COWs **every** node in
      `nodes_to_update` **before** taking any per-node lock, so the held lock's
      buffer can't be forked out from under it (the mutex-identity hazard the
      earlier TSan run surfaced as "unlock of an unlocked mutex"). With that,
      **repair runs under a live snapshot** (no longer deferred);
      `removeVectorInPlace` captures the locked candidate id so lock/unlock can't
      target different ids. Proven by `tieredSnapshotRepairRunsHorizonRecycles` +
      `parallelBatchIteratorSearchSnapshot` (ASan-clean).
- [x] 4.6 ~~Make COW-ing writes take the exclusive lock when a snapshot is
      active.~~ **Superseded.** Instead of serializing all writers under the
      exclusive lock, the implementation keeps concurrent inserts on the *shared*
      lock and makes per-block COW publish via an **atomic compare-exchange**
      (generation-tag idempotent); only **capture** takes the exclusive lock (a
      brief quiescence point that forks the backbone). This preserves write
      concurrency, which serializing would have killed.
- [ ] 4.7 **Slot reclamation — the vector+metadata problem (distinct from graph
      COW).** A slot spans three parallel containers; COW only versions the
      graph. SWAP overwrites the **vector** (`getDataByInternalId`) and
      **metadata** (`idToMetaData[id]`) **in place**, so a snapshot-referenced
      `id` would read a *wrong embedding + wrong deleted-flag*. Fix = **defer the
      SWAP**, do NOT COW the vector/metadata containers:
      - [x] 4.7a Wired `graphSnapshotActive()` into `executeReadySwapJobs`: while
        a snapshot is live the ready swap jobs are deferred (tombstoned) and
        recycled by a later GC pass once no snapshot references them.
      - [x] 4.7b Test `HNSWTieredIndexTest.swapDeferredWhileSnapshotHeld`:
        single-writer delete while holding a snapshot — the deleted id's slot
        keeps its **vector bytes** and **identity** (label), and shows its **own**
        deleted-flag rather than a live element wrongly occupying the recycled
        slot; releasing the snapshot lets the deferred swap proceed.
      - [x] 4.7c Horizon refinement **shipped** (branch 09), as a *count-based*
        horizon rather than the originally-sketched min-live-id + per-slot
        free-generation: recycle a freed id `d` once `d >= maxVisibleCount()` (max
        captured `curElementCount` over live snapshots) — no live snapshot can read
        an id at/above its own count. So one long-lived cursor no longer stalls all
        compaction (high-id churn recycles immediately). Proven by
        `tieredSnapshotRepairRunsHorizonRecycles`.
- [x] 4.8 **Deleted-flag consistency decision.** `markDelete` flips the flag in
      place, so even with SWAP deferral a snapshot can observe an element that was
      live at capture as *deleted*. Default contract: accept this weak
      consistency ("no *new* vectors after T, crash-safe"). Strict as-of-capture
      deleted-flags would require versioning the metadata flag (separate work).
      Record the chosen contract in `spec-delta.md`.
- [x] 4.9 **COW trigger = generation tag, not `use_count`.** Stamp the backbone
      and each block buffer with the generation they were last written in; in the
      backbone-COW check and `getGraphDataByInternalIdForWrite`, clone iff
      `max_live_snapshot_id >= gen`, NOT `use_count() > 1`. Rationale:
      container-level `root.use_count()` goes **stale after the first backbone
      COW** (reports unshared while a snapshot still holds the old backbone →
      in-place mutation corrupts the snapshot — the same staleness 4.2 hit for the
      gate); per-buffer `use_count` is sound but refcount-coupled and
      false-positive-prone. Keep `shared_ptr` purely for **reclamation**.
      `max_live_snapshot_id` = tail of the 3.5 live-id set. **Done:** backbone +
      `GraphBlockBuffer` carry `gen`; `SnapshotRegistry` exposes lock-free
      `newestLive()` / `currentGeneration()` (atomics on the write hot path, mutex
      only for the rare min() query); `cowBackbone`/`cowBlock` clone iff
      `newestLive() >= gen` and stamp the clone with `currentGeneration()`;
      `shared_ptr` retained purely for reclamation. Tests
      `HNSWTest.cowDecisionByGenerationNotUseCount` (write block A → backbone
      clones, then a write to block B still COWs) and
      `HNSWTest.snapshotPinsUntraversedBlock` (a never-traversed block is pinned by
      the captured backbone — read-order independence). NOTE: capturing a snapshot
      now requires a registered generation (`captureGraphSnapshot()`); a bare
      `captureRoot()` no longer triggers COW, so `cowStorageSnapshotIsolation` was
      updated to capture via the handle.

- [ ] 4.10 **Snapshot that outlives `VecSimIndex_Free` corrupts the heap.**
      Discovered via 4.9's tests: if a snapshot handle is still alive when the
      index is destroyed (its captured backbone/buffers then outlive the index),
      teardown corrupts the heap (`malloc_consolidate: unaligned fastbin chunk`).
      Dropping the handle before `VecSimIndex_Free` is clean, which is the normal
      usage (a batch iterator is always freed before the index), and all snapshot
      unit tests now release the handle first. But design.md "Edge cases" lists
      "snapshot outlives the index drop" as a supported scenario, so this is a real
      teardown/allocator-ordering bug to root-cause separately (likely the index's
      allocator vs. the buffers' allocator refs during `~HNSWIndex` +
      `operator delete`). Out of 4.9's scope (the gen-tag itself is correct — the
      tests pass once the handle is released before the free).

## 5. RediSearch integration

- [ ] 5.1 `src/iterators/hybrid_reader.c` + vecsim wrappers: construct the
      iterator under the lock, release it for iteration
- [ ] 5.2 Ensure cursor-paginated vector aggregation holds the snapshot across
      `FT.CURSOR READ` calls and frees it on cursor close/timeout
- [ ] 5.3 Python end-to-end: KNN / hybrid / `WITHCURSOR` return consistent
      results while a concurrent writer adds and deletes
- [ ] 5.4 Python end-to-end: writer makes observable progress during a long
      vector read (lock released)

## 6. Configuration, limits & observability

- [ ] 6.1 Add a tiered-HNSW config to enable snapshot iteration
- [ ] 6.2 Add a snapshot memory / lifetime cap; define on-breach behavior
      (resolve Open question #2)
- [ ] 6.3 Expose active-snapshot count and retained snapshot memory via
      `FT.INFO` and/or `FT.DEBUG`
- [ ] 6.4 Python end-to-end: cap is enforced (long-lived cursor cannot grow
      retained memory without bound)

## 7. Docs & spec delta

- [ ] 7.1 Update the delta spec under
      `specs/vector-snapshot-iteration/spec.md` to match what shipped
- [ ] 7.2 Document the new config and the strengthened consistency guarantee
- [ ] 7.3 Confirm the PR's release-notes checkbox reflects the behavior change

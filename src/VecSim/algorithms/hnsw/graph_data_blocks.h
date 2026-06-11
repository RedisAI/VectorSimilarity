/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/algorithms/hnsw/graph_data.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

// Rooted, refcounted copy-on-write storage for the HNSW graph blocks.
//
// This replaces the old `vecsim_stl::vector<DataBlock> graphDataBlocks` with the
// structure the snapshot design settled on:
//
//     shared_ptr< vector< Block > >  root          // contiguous header vector
//        where  Block { shared_ptr<GraphBlockBuffer> data; }
//
// The point is to make a point-in-time *snapshot* of the graph cheap to capture
// (an O(1) `shared_ptr` copy of `root`) and safe to read lock-free while writers
// keep mutating the live index: writers never touch a buffer a snapshot can see;
// instead they copy-on-write a new version. Old versions free themselves once no
// snapshot references them (refcount-driven reclamation), so there is no
// hand-rolled epoch/retire machinery.
//
// When no snapshot is active every refcount is 1, so `cowBackbone`/`cowBlock` are
// no-ops and all mutations happen in place: behavior and cost are identical to
// the previous flat storage. Snapshot capture itself is added in a later phase;
// this phase introduces the storage + COW primitives and routes the read/grow/
// shrink/serialize paths through them.

// Tracks the live snapshot generation ids for one index. Snapshots get a unique,
// monotonically increasing id at capture (`acquire`) and are removed at release.
// Held via shared_ptr by both the index and the snapshot handles, so it safely
// outlives the index if a snapshot does.
//
// It drives the two generation-based decisions (see design.md "Operations"):
//   - `newestLive()` = max(live ids) = the **clone threshold**: a backbone/block
//     version stamped `gen` must be preserved (cloned on write) iff
//     `newestLive() >= gen` — some live snapshot can still see it.
// `currentGeneration()` = the next id to be handed out = strictly greater than
// every already-captured id; freshly-written/cloned versions are stamped with it,
// so writes that post-date all live snapshots stay in place.
//
// The hot path (clone decision, read on every COW-aware write) uses only the
// lock-free atomics; the mutex guards the ordered set on capture/release.
class SnapshotRegistry {
public:
    uint64_t acquire() {
        uint64_t g = nextGeneration_.fetch_add(1);
        std::lock_guard<std::mutex> lk(guard_);
        live_.insert(g);
        maxLive_.store(*live_.rbegin());
        return g;
    }
    void release(uint64_t generation) {
        std::lock_guard<std::mutex> lk(guard_);
        live_.erase(generation);
        maxLive_.store(live_.empty() ? 0 : *live_.rbegin());
    }
    // "is any snapshot live?" — the degenerate gate (used by the SWAP deferral).
    bool anyLive() const { return maxLive_.load() != 0; }
    // max(live ids) — the clone threshold; 0 when none live. Lock-free.
    uint64_t newestLive() const { return maxLive_.load(); }
    // next id to be handed out (> all captured ids). Lock-free.
    uint64_t currentGeneration() const { return nextGeneration_.load(); }

private:
    mutable std::mutex guard_;
    std::set<uint64_t> live_;            // unique, monotonic ids; begin=min, rbegin=max
    std::atomic<uint64_t> nextGeneration_{1}; // 0 reserved for "no/invalid generation"
    std::atomic<uint64_t> maxLive_{0};   // cached max(live_) for the lock-free hot path
};

// Generation-tagged, copy-on-write holder for a single `shared_ptr<T>` version
// (the reusable core of the snapshot mechanism). A snapshot captures the current
// version with capture() — an O(1) shared_ptr bump that pins it. Before a write,
// cowForWrite() clones the version iff a live snapshot can still see it
// (newestLive() >= gen), swaps in the clone, and stamps it with the current
// generation; a version written after all live snapshots is mutated in place.
// The clone operation is supplied by the caller (trivial copy for the backbone /
// metadata; a deep copyTo for the per-block graph buffers), so the same gen-tag
// machinery serves every per-id container. Reclamation stays refcount-driven:
// the old version frees itself once the last snapshot referencing it drops.
//
// `gen` is intentionally NOT use_count: a container-level use_count goes stale
// after the first clone (the live root is unshared again while a snapshot still
// holds the old one), and a per-buffer use_count couples correctness to refcount
// discipline. The generation tag is immune to both.
template <typename T>
class RootedCowStore {
public:
    void setRegistry(std::shared_ptr<SnapshotRegistry> registry) {
        registry_ = std::move(registry);
    }
    bool hasRoot() const { return static_cast<bool>(root_); }
    const std::shared_ptr<T> &root() const { return root_; }
    std::shared_ptr<T> capture() const { return root_; } // O(1) pin for a snapshot

    // Install the initial version, stamped with the current generation.
    void initRoot(std::shared_ptr<T> root) {
        root_ = std::move(root);
        gen_ = currentGeneration();
    }
    // Drop the live version (a live snapshot keeps its own copy alive). The next
    // initRoot re-stamps a fresh generation.
    void reset() {
        root_.reset();
        gen_ = 0;
    }
    // Generation to stamp freshly-written/cloned versions with (> every live id).
    uint64_t currentGeneration() const {
        return registry_ ? registry_->currentGeneration() : 1;
    }
    // True iff a live snapshot can still see the current version (so a write must
    // preserve it by cloning). Reusable for finer-grained (per-buffer) decisions.
    bool mustClone(uint64_t gen) const {
        return (registry_ ? registry_->newestLive() : 0) >= gen;
    }
    bool mustCloneRoot() const { return mustClone(gen_); }

    // Copy-on-write: if a live snapshot can still see this version, replace it with
    // `cloneFn(*root_)` and re-stamp; otherwise leave it (subsequent writes mutate
    // in place). cloneFn(const T&) -> std::shared_ptr<T>.
    template <class CloneFn> void cowForWrite(CloneFn &&cloneFn) {
        if (mustCloneRoot()) {
            root_ = cloneFn(static_cast<const T &>(*root_));
            gen_ = currentGeneration();
        }
    }

private:
    std::shared_ptr<SnapshotRegistry> registry_;
    std::shared_ptr<T> root_;
    uint64_t gen_ = 0;
};

// One block's raw bytes: `block_size` ElementGraphData records laid out
// contiguously, plus the count of live records. Held only behind a shared_ptr;
// when the last reference drops, the destructor releases each live element's
// owned resources (incoming-edges vectors + the `others` upper-level allocation)
// and frees the raw buffer. A superseded (COW'd-away) buffer thus cleans itself
// up the moment the last snapshot that pinned it is released.
//
// `gen` is the generation the buffer was last written in (stamped on creation and
// on each copy-on-write clone); the clone decision compares it against
// `SnapshotRegistry::newestLive()` rather than the shared_ptr use_count.
class GraphBlockBuffer : public VecsimBaseObject {
public:
    GraphBlockBuffer(size_t blockSize, size_t elementBytes, size_t levelDataSize, uint64_t gen,
                     std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator), length(0), element_bytes(elementBytes),
          level_data_size(levelDataSize), block_size(blockSize), gen_(gen) {
        data = static_cast<char *>(this->allocator->callocate(block_size * element_bytes));
        if (data == nullptr) {
            throw std::runtime_error("VecSim index low memory error");
        }
    }

    // Generation this buffer's contents were last written in (set on creation and
    // on each COW clone). Drives the clone decision via SnapshotRegistry.
    uint64_t gen() const { return gen_; }

    ~GraphBlockBuffer() noexcept override {
        if (data == nullptr) {
            return;
        }
        for (size_t i = 0; i < length; i++) {
            reinterpret_cast<ElementGraphData *>(data + i * element_bytes)
                ->destroy(level_data_size, allocator);
        }
        allocator->free_allocation(data);
    }

    // Held only via shared_ptr; copying is done explicitly through copyLiveInto.
    GraphBlockBuffer(const GraphBlockBuffer &) = delete;
    GraphBlockBuffer &operator=(const GraphBlockBuffer &) = delete;

    char *getElement(size_t offset) const { return data + offset * element_bytes; }
    size_t getLength() const { return length; }

    // Append a bitwise copy of an already-constructed ElementGraphData record
    // (its owned pointers are moved into this buffer's slot). Returns the slot.
    char *addElement(const void *element_record) {
        assert(length < block_size);
        char *slot = data + length * element_bytes;
        memcpy(slot, element_record, element_bytes);
        length++;
        return slot;
    }

    // Drop the last record from the live count and return it (the caller either
    // moves it elsewhere or has already released its resources). Mirrors the old
    // DataBlock::removeAndFetchLastElement.
    char *removeAndFetchLastElement() {
        assert(length > 0);
        return data + (--length) * element_bytes;
    }

    // Deep-copy this buffer's live elements into a freshly-allocated buffer using
    // the Phase-1 ElementGraphData::copyTo primitive, so the copy shares no
    // mutable state (fresh mutex, independent incoming-edges, own `others`).
    void copyLiveInto(GraphBlockBuffer &dst) const {
        assert(dst.block_size == block_size && dst.element_bytes == element_bytes);
        for (size_t i = 0; i < length; i++) {
            reinterpret_cast<const ElementGraphData *>(data + i * element_bytes)
                ->copyTo(reinterpret_cast<ElementGraphData *>(dst.data + i * element_bytes),
                         level_data_size, element_bytes, allocator);
        }
        dst.length = length;
    }

private:
    // `allocator` is provided by VecsimBaseObject.
    size_t length;
    size_t element_bytes;
    size_t level_data_size;
    size_t block_size;
    char *data;
    uint64_t gen_; // generation last written in (clone decision; not the use_count)
};

// A backbone slot: a refcounted handle to one block's buffer. Trivially/cheaply
// copyable, which is what keeps backbone copy-on-write (path-copying the header
// vector) cheap — only refcounts are bumped, buffers are shared.
struct GraphBlock {
    std::shared_ptr<GraphBlockBuffer> data;
};

class GraphDataBlocks {
public:
    using Backbone = vecsim_stl::vector<GraphBlock>;
    using Root = std::shared_ptr<Backbone>;

    explicit GraphDataBlocks(std::shared_ptr<VecSimAllocator> allocator)
        : allocator(std::move(allocator)), element_bytes(0), level_data_size(0), block_size(0) {
        // The backbone is created lazily on the first block, so an empty index
        // allocates nothing here (matching the previous flat storage and the
        // initial-size estimation).
    }

    // Set the per-element/per-block geometry and the shared snapshot registry.
    // Must be called once, before any block is added, after the index has computed
    // elementGraphDataSize / levelDataSize (which depend on M / M0). The registry
    // supplies the live-snapshot generation bounds that drive the clone decision.
    void configure(size_t blockSize, size_t elementBytes, size_t levelDataSize,
                   std::shared_ptr<SnapshotRegistry> registry) {
        block_size = blockSize;
        element_bytes = elementBytes;
        level_data_size = levelDataSize;
        backbone_.setRegistry(std::move(registry));
    }

    // ---- sizing ----
    size_t size() const { return backbone_.hasRoot() ? backbone_.root()->size() : 0; }
    size_t capacity() const { return backbone_.hasRoot() ? backbone_.root()->capacity() : 0; }
    void reserve(size_t n) {
        ensureRoot();
        backbone_.root()->reserve(n);
    }
    void shrink_to_fit() {
        if (!backbone_.hasRoot()) {
            return;
        }
        if (backbone_.root()->empty()) {
            // Release the backbone entirely so an emptied index returns to its
            // initial footprint (a live snapshot, if any, keeps its own copy
            // alive via its captured root).
            backbone_.reset();
            return;
        }
        cowBackbone();
        backbone_.root()->shrink_to_fit();
    }

    // ---- read path ----
    char *getElement(size_t id) const {
        return (*backbone_.root())[id / block_size].data->getElement(id % block_size);
    }
    size_t blockLength(size_t blockIdx) const {
        return (*backbone_.root())[blockIdx].data->getLength();
    }
    char *getElementInBlock(size_t blockIdx, size_t offset) const {
        return (*backbone_.root())[blockIdx].data->getElement(offset);
    }

    // ---- write path (copy-on-write aware; must run under the index write lock) ----

    // Resolve an element for mutation: ensure neither the backbone nor the block
    // holding `id` is shared with a snapshot, then return a writable pointer.
    char *getElementForWrite(size_t id) {
        cowBackbone();
        cowBlock(id / block_size);
        return (*backbone_.root())[id / block_size].data->getElement(id % block_size);
    }

    // Append an empty block (graph growth). Path-copies the backbone first if a
    // snapshot is sharing it.
    void addBlock() {
        ensureRoot();
        cowBackbone();
        backbone_.root()->push_back(GraphBlock{makeBuffer()});
    }

    // Append a new element record to the last block. COWs the last block if it is
    // shared. Returns a writable pointer to the stored record.
    char *addElement(const void *element_record) {
        cowBackbone();
        cowBlock(backbone_.root()->size() - 1);
        return backbone_.root()->back().data->addElement(element_record);
    }

    // Remove (decrement) the last record of the last block and return it. COWs
    // the last block if shared.
    char *removeAndFetchLastElement() {
        cowBackbone();
        cowBlock(backbone_.root()->size() - 1);
        return backbone_.root()->back().data->removeAndFetchLastElement();
    }

    // Drop the last (empty) block. Path-copies the backbone first if shared.
    void popLastBlock() {
        cowBackbone();
        backbone_.root()->pop_back();
    }

    // ---- snapshot support ----
    // O(1) capture of the current immutable view. Callers take this under the
    // index read lock; reads through the returned Root are then lock-free.
    Root captureRoot() const { return backbone_.capture(); }

private:
    void ensureRoot() {
        if (!backbone_.hasRoot()) {
            // Allocate the backbone vector object through the index allocator (so
            // it is accounted); the shared_ptr control block uses global new and
            // is intentionally not routed through the VecSimAllocator, keeping the
            // tracked footprint expressible via sizeof for the memory tests.
            backbone_.initRoot(Root(new (allocator) Backbone(allocator)));
        }
    }

    std::shared_ptr<GraphBlockBuffer> makeBuffer() const {
        return std::shared_ptr<GraphBlockBuffer>(new (allocator) GraphBlockBuffer(
            block_size, element_bytes, level_data_size, backbone_.currentGeneration(), allocator));
    }

    // Backbone COW via the shared generation-tag store: clone the header vector
    // (copying the GraphBlock handles, sharing the buffers) iff a live snapshot can
    // still see this version.
    void cowBackbone() {
        backbone_.cowForWrite([this](const Backbone &old) {
            Root fresh(new (allocator) Backbone(allocator));
            *fresh = old;
            return fresh;
        });
    }

    // Per-block COW (same generation-tag rule, finer granularity): clone the block
    // buffer (deep-copy via copyTo) iff a live snapshot can still see this version.
    // A per-buffer use_count would be misleading before the backbone is cloned (a
    // shared backbone holds a single shared_ptr per block, so use_count == 1 even
    // though a snapshot sees it) — the generation tag is reliable.
    void cowBlock(size_t blockIdx) {
        GraphBlock &blk = (*backbone_.root())[blockIdx];
        if (backbone_.mustClone(blk.data->gen())) {
            auto fresh = makeBuffer(); // carries currentGeneration()
            blk.data->copyLiveInto(*fresh);
            blk.data = std::move(fresh);
        }
    }

    std::shared_ptr<VecSimAllocator> allocator;
    RootedCowStore<Backbone> backbone_;
    size_t element_bytes;
    size_t level_data_size;
    size_t block_size;
};

// An immutable, point-in-time view of the HNSW graph topology. It is captured by
// copying the backbone root (an O(1) shared_ptr bump that pins every block buffer
// referenced at capture time) together with the scalar entry-point / level /
// count state. Because writers copy-on-write rather than mutate shared buffers,
// the snapshot keeps observing the graph exactly as it was at capture, even as
// the live index diverges. Reads through it resolve node access without touching
// the live `graphDataBlocks` member.
//
// `generation` is the snapshot's id from the index's SnapshotRegistry; `liveToken`
// is an opaque handle whose deleter removes that id from the registry when the
// last copy of this snapshot is destroyed (registration lifetime == snapshot
// lifetime, independent of the copy-on-write `root` refcount).
struct HNSWGraphSnapshot {
    GraphDataBlocks::Root root;
    idType entrypointNode;
    size_t maxLevel;
    size_t curElementCount;
    size_t blockSize;
    size_t levelDataSize;
    size_t elementGraphDataSize;
    uint64_t generation = 0;
    std::shared_ptr<void> liveToken;

    // Captured per-block base pointers of the raw vector data (one per block at
    // capture time), so a lock-free query reads vector data WITHOUT touching the
    // live vectors container (whose block-header vector reallocs on insert). The
    // pointers stay valid while the snapshot is live: the underlying data buffers
    // don't move when the index grows (only the header vector reallocs; the buffer
    // pointer is preserved), and SWAP/shrink reuse is deferred while a snapshot is
    // held. Shared so handle copies don't re-capture. (Capture is O(#blocks)
    // rather than strictly O(1); rooting the vectors container would restore O(1).)
    std::shared_ptr<std::vector<const char *>> vectorBlocks;
    size_t vectorElementBytes = 0;

    // Captured per-id metadata version (the flat label+flags vector), type-erased
    // because this header is below ElementMetaData's definition. It pins the
    // as-of-capture metadata so a lock-free query reads its flags + labels without
    // racing a concurrent insert that reallocates the live metadata. The reader
    // (HNSWIndex::topKFromSnapshot) casts it back to vecsim_stl::vector<ElementMetaData>.
    std::shared_ptr<void> metaData;

    bool valid() const { return root != nullptr; }

    ElementGraphData *getGraphData(idType id) const {
        return reinterpret_cast<ElementGraphData *>(
            (*root)[id / blockSize].data->getElement(id % blockSize));
    }
    ElementLevelData &getLevelData(idType id, size_t level) const {
        return getGraphData(id)->getElementLevelData(level, levelDataSize);
    }
    const char *getVectorData(idType id) const {
        return (*vectorBlocks)[id / blockSize] +
               static_cast<size_t>(id % blockSize) * vectorElementBytes;
    }
};

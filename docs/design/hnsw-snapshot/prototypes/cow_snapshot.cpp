// Prototype A: persistent rooted copy-on-write snapshot for an HNSW-like block
// store. Proves:
//   1. O(1) snapshot capture (clone the root shared_ptr).
//   2. Lock-free reads from a snapshot stay CONSISTENT while a writer thread
//      concurrently mutates edges, COWs blocks, and grows the backbone.
//   3. Automatic refcount-driven cleanup: dropping the snapshot frees the old
//      blocks/backbones with no remaining references.
//
// Build: g++ -O2 -std=c++20 -pthread cow_snapshot.cpp -o cow_snapshot
//
// This is a standalone model of the structure
//   shared_ptr< vector< shared_ptr<Block> > >
// described in docs/design/vecsim-hnsw-snapshot.md. It is NOT wired into VecSim;
// it exists to validate the mechanism and its concurrency story in isolation.

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>

// ----- Block: models a DataBlock holding `blockSize` elements, each with an
// edge list. We track live instances to prove cleanup.
struct Block {
  static inline std::atomic<int64_t> live{0};

  // elems[e] is the link list (neighbors) of element e in this block.
  std::vector<std::vector<uint32_t>> elems;

  explicit Block(size_t blockSize) : elems(blockSize) { live.fetch_add(1); }
  // Deep copy ctor — this is what block-COW invokes.
  Block(const Block& o) : elems(o.elems) { live.fetch_add(1); }
  ~Block() { live.fetch_sub(1); }
};

using Backbone = std::vector<std::shared_ptr<Block>>;
using Root = std::shared_ptr<Backbone>;

// ----- The index: a root pointer guarded by a shared_mutex, mirroring VecSim's
// indexDataGuard. Writers take it exclusive; snapshots take it shared just long
// enough to clone the root, then iterate lock-free.
class CowIndex {
 public:
  CowIndex(size_t nBlocks, size_t blockSize) : blockSize_(blockSize) {
    auto bb = std::make_shared<Backbone>();
    for (size_t i = 0; i < nBlocks; i++) {
      auto blk = std::make_shared<Block>(blockSize);
      for (size_t e = 0; e < blockSize; e++)
        blk->elems[e] = {uint32_t(i), uint32_t(e)};  // some initial edges
      bb->push_back(std::move(blk));
    }
    root_ = std::move(bb);
  }

  // O(1) snapshot: clone the root under a shared lock, then it's detached.
  Root snapshot() {
    std::shared_lock lk(guard_);
    return root_;  // shared_ptr copy: bumps backbone refcount, O(1)
  }

  size_t blockSize() const { return blockSize_; }

  // A write: mutate the edge list of (block b, element e). Performs generational
  // backbone-COW then per-block COW, all under the exclusive lock.
  void mutateEdges(size_t b, size_t e, uint32_t tag) {
    std::unique_lock lk(guard_);
    cowBackboneIfShared();
    auto& slot = (*root_)[b];
    if (slot.use_count() > 1)               // shared with a live snapshot
      slot = std::make_shared<Block>(*slot);  // deep-copy this block
    slot->elems[e] = {tag, tag, tag};       // mutate the private copy
  }

  // Grow: append a new block (the rare backbone-membership change).
  void addBlock() {
    std::unique_lock lk(guard_);
    cowBackboneIfShared();
    auto blk = std::make_shared<Block>(blockSize_);
    root_->push_back(std::move(blk));
  }

  size_t numBlocks() {
    std::shared_lock lk(guard_);
    return root_->size();
  }

 private:
  // Path-copy the backbone once per generation: only if a snapshot shares it.
  void cowBackboneIfShared() {
    if (root_.use_count() > 1 || /* the backbone object itself */ false) {
      // Copy the vector of block shared_ptrs (cheap: N/blockSize pointers).
      root_ = std::make_shared<Backbone>(*root_);
    }
  }

  std::shared_mutex guard_;
  Root root_;
  size_t blockSize_;
};

// Checksum over an entire snapshot's edges — used to assert consistency.
static uint64_t checksum(const Root& snap) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a
  for (auto& blk : *snap)
    for (auto& edges : blk->elems)
      for (uint32_t v : edges) {
        h ^= v;
        h *= 1099511628211ull;
      }
  return h;
}

int main() {
  const size_t nBlocks = 64, blockSize = 1024;
  CowIndex idx(nBlocks, blockSize);

  std::printf("initial live blocks: %lld (expect %zu)\n",
              (long long)Block::live.load(), nBlocks);

  // Capture a snapshot and its baseline checksum (lock-free reads hereafter).
  Root snap = idx.snapshot();
  const uint64_t baseline = checksum(snap);
  std::printf("snapshot captured, baseline checksum = %016llx\n",
              (unsigned long long)baseline);

  // Writer thread: hammer the index with edge mutations + block growth.
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> writes{0};
  std::thread writer([&] {
    std::mt19937 rng(12345);
    while (!stop.load(std::memory_order_relaxed)) {
      size_t nb = idx.numBlocks();
      size_t b = rng() % nb;
      size_t e = rng() % blockSize;
      idx.mutateEdges(b, e, 0xDEADBEEF);
      if ((writes++ & 0x3FFF) == 0) idx.addBlock();  // occasionally grow
    }
  });

  // Main thread: repeatedly re-read the SNAPSHOT lock-free and verify it never
  // changes, while the live index is being mutated underneath.
  bool consistent = true;
  for (int i = 0; i < 2000; i++) {
    if (checksum(snap) != baseline) {
      consistent = false;
      break;
    }
    std::this_thread::yield();
  }

  stop.store(true);
  writer.join();

  // The live index HAS changed (sanity: the snapshot diverges from live).
  Root liveNow = idx.snapshot();
  uint64_t liveChecksum = 0;
  // Only compare the original blocks (live has more after growth).
  {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < nBlocks; i++)
      for (auto& edges : (*liveNow)[i]->elems)
        for (uint32_t v : edges) { h ^= v; h *= 1099511628211ull; }
    liveChecksum = h;
  }

  std::printf("writes performed: %llu\n", (unsigned long long)writes.load());
  std::printf("snapshot consistent during writes: %s\n",
              consistent ? "YES" : "NO");
  std::printf("snapshot checksum still = %016llx\n",
              (unsigned long long)checksum(snap));
  std::printf("live   checksum now     = %016llx (differs: %s)\n",
              (unsigned long long)liveChecksum,
              liveChecksum != baseline ? "YES" : "no");

  int64_t whileSnapHeld = Block::live.load();
  std::printf("live blocks while snapshot held: %lld (>= %zu: COW copies + growth)\n",
              (long long)whileSnapHeld, nBlocks);

  // Drop the snapshot -> automatic cleanup of versions only it referenced.
  snap.reset();
  liveNow.reset();
  int64_t afterDrop = Block::live.load();
  std::printf("live blocks after dropping snapshot: %lld\n", (long long)afterDrop);
  std::printf("(should equal current live backbone size = %zu)\n", idx.numBlocks());

  bool cleanupOk = (afterDrop == (int64_t)idx.numBlocks());
  std::printf("automatic cleanup correct: %s\n", cleanupOk ? "YES" : "NO");

  bool pass = consistent && cleanupOk && (liveChecksum != baseline);
  std::printf("\n==== PROTOTYPE A %s ====\n", pass ? "PASSED" : "FAILED");
  return pass ? 0 : 1;
}

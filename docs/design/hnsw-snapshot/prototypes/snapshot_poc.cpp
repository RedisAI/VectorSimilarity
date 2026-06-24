// POC: lock-free consistent snapshot iteration over a graph built from the REAL
// VecSim node structures (`ElementGraphData`/`ElementLevelData` from
// graph_data.h) and the REAL `copyTo` deep-copy primitive added in Phase 1.
//
// It is a minimal (level-0) graph index — not full HNSW search quality — whose
// only purpose is to answer "is the snapshot mechanism implementable against
// VecSim's actual node representation?" It demonstrates, end to end:
//   * rooted COW block storage  shared_ptr<vector<Block{shared_ptr buf}>>
//   * snapshot capture under a read lock, then release (O(1) shared_ptr copy)
//   * a graph traversal run lock-free from the snapshot
//   * a concurrent writer that adds nodes + rewires edges, COW-ing real blocks
//     via ElementGraphData::copyTo
//   * the snapshot's query result staying invariant while the live graph diverges
//   * automatic refcount cleanup (allocator bytes return to the live baseline)
//
// Build (from this dir):
//   g++ -O2 -std=c++20 -pthread -I ../../../../src \
//       snapshot_poc.cpp ../../../../src/VecSim/memory/vecsim_malloc.cpp \
//       -o snapshot_poc

#include "VecSim/algorithms/hnsw/graph_data.h"
#include "VecSim/memory/vecsim_malloc.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <queue>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>

using Alloc = std::shared_ptr<VecSimAllocator>;
static constexpr idType INVALID = (idType)-1;

struct Block {
  std::shared_ptr<char[]> data;  // blockSize * elementGraphDataSize bytes
};
using Backbone = std::vector<Block>;
using Root = std::shared_ptr<Backbone>;

// Immutable snapshot handle: captured under the read lock, then used lock-free.
struct Snapshot {
  Root root;
  idType entry;
  size_t count, blockSize, egds, lds;
  ElementGraphData *at(idType id) const {
    return (ElementGraphData *)((*root)[id / blockSize].data.get() +
                                (id % blockSize) * egds);
  }
};

class SnapGraph {
 public:
  SnapGraph(size_t M = 16, size_t M0 = 32, size_t blockSize = 64)
      : M_(M), M0_(M0), blockSize_(blockSize),
        alloc_(VecSimAllocator::newVecsimAllocator()) {
    lds_ = sizeof(ElementLevelData) + sizeof(idType) * M_;
    egds_ = sizeof(ElementGraphData) + sizeof(idType) * M0_;
    root_ = std::make_shared<Backbone>();
  }

  size_t allocatedBytes() { return alloc_->getAllocationSize(); }

  Snapshot snapshot() {
    std::shared_lock lk(guard_);
    return Snapshot{root_, entry_, count_, blockSize_, egds_, lds_};
  }

  // Append a node (no links yet). Returns its id.
  idType addNode() {
    std::unique_lock lk(guard_);
    idType id = (idType)count_;
    if (id % blockSize_ == 0) {  // need a new block
      cowBackbone();
      root_->push_back(makeBlock());
    }
    count_++;
    if (entry_ == INVALID) entry_ = id;
    return id;
  }

  // Add directed edge a -> b at level 0 (COW-aware).
  void connect(idType a, idType b) {
    std::unique_lock lk(guard_);
    mutateNode(a, [&](ElementLevelData &ld) {
      if (ld.getNumLinks() < (linkListSize)M0_) ld.appendLink(b);
    });
  }

  // Replace a's level-0 links with `neighbors` (COW-aware) — used to diverge.
  void rewire(idType a, const std::vector<idType> &neighbors) {
    std::unique_lock lk(guard_);
    mutateNode(a, [&](ElementLevelData &ld) {
      vecsim_stl::vector<idType> v(alloc_);
      for (idType n : neighbors)
        if (v.size() < M0_) v.push_back(n);
      ld.setLinks(v);
    });
  }

  size_t liveCount() {
    std::shared_lock lk(guard_);
    return count_;
  }

 private:
  ElementGraphData *at(Root &r, idType id) {
    return (ElementGraphData *)((*r)[id / blockSize_].data.get() +
                                (id % blockSize_) * egds_);
  }

  // Backbone COW: copy the header vector once per generation (cheap; shares the
  // block buffers by refcount). Only when a snapshot is sharing it.
  void cowBackbone() {
    if (root_.use_count() > 1) root_ = std::make_shared<Backbone>(*root_);
  }

  // Per-block COW via the REAL ElementGraphData::copyTo. Only when a snapshot
  // shares this block's buffer.
  void cowBlock(idType id) {
    Block &blk = (*root_)[id / blockSize_];
    if (blk.data.use_count() > 1) {
      char *nraw = (char *)alloc_->allocate(blockSize_ * egds_);
      for (size_t i = 0; i < blockSize_; i++) {
        auto *src = (ElementGraphData *)(blk.data.get() + i * egds_);
        auto *dst = (ElementGraphData *)(nraw + i * egds_);
        src->copyTo(dst, lds_, egds_, alloc_);  // <-- exercises Phase 1 primitive
      }
      blk.data = makeBufferOwner(nraw);
    }
  }

  template <typename F>
  void mutateNode(idType id, F &&fn) {
    cowBackbone();
    cowBlock(id);
    fn(at(root_, id)->getElementLevelData(0, lds_));
  }

  std::shared_ptr<char[]> makeBufferOwner(char *raw) {
    Alloc a = alloc_;
    size_t bs = blockSize_, e = egds_, l = lds_;
    return std::shared_ptr<char[]>(raw, [a, bs, e, l](char *p) {
      for (size_t i = 0; i < bs; i++)
        ((ElementGraphData *)(p + i * e))->destroy(l, a);
      a->free_allocation(p);
    });
  }

  Block makeBlock() {
    char *raw = (char *)alloc_->allocate(blockSize_ * egds_);
    for (size_t i = 0; i < blockSize_; i++)
      new (raw + i * egds_) ElementGraphData(0, lds_, alloc_);  // level-0 only
    return Block{makeBufferOwner(raw)};
  }

  size_t M_, M0_, blockSize_, lds_, egds_;
  Alloc alloc_;
  std::shared_mutex guard_;
  Root root_;
  idType entry_ = INVALID;
  size_t count_ = 0;
};

// Lock-free traversal of a snapshot: BFS from the entry over level-0 links,
// collect reachable ids, return an order-independent hash of the closest `k` to
// `q` (distance = |id - q|). Result depends only on the snapshot's topology.
static uint64_t query(const Snapshot &s, idType q, size_t k) {
  if (s.count == 0) return 0;
  std::vector<char> seen(s.count, 0);
  std::queue<idType> bfs;
  std::vector<idType> reached;
  bfs.push(s.entry);
  seen[s.entry] = 1;
  while (!bfs.empty()) {
    idType cur = bfs.front();
    bfs.pop();
    reached.push_back(cur);
    ElementLevelData &ld = s.at(cur)->getElementLevelData(0, s.lds);
    for (linkListSize j = 0; j < ld.getNumLinks(); j++) {
      idType nb = ld.getLinkAtPos(j);
      if (nb < s.count && !seen[nb]) {
        seen[nb] = 1;
        bfs.push(nb);
      }
    }
  }
  auto dist = [&](idType id) {
    return id > q ? (uint64_t)(id - q) : (uint64_t)(q - id);
  };
  std::sort(reached.begin(), reached.end(),
            [&](idType a, idType b) { return dist(a) < dist(b); });
  if (reached.size() > k) reached.resize(k);
  std::sort(reached.begin(), reached.end());  // order-independent
  uint64_t h = 1469598103934665603ull;
  for (idType id : reached) {
    h ^= id;
    h *= 1099511628211ull;
  }
  return h;
}

int main() {
  SnapGraph g;
  const size_t N = 2000;
  std::mt19937 rng(2024);

  // Build a connected graph: ring + random forward edges so BFS reaches a lot.
  for (size_t i = 0; i < N; i++) g.addNode();
  for (idType i = 0; i < N; i++) {
    g.connect(i, (i + 1) % N);
    for (int e = 0; e < 6; e++) g.connect(i, rng() % N);
  }

  size_t baseBytes = g.allocatedBytes();
  Snapshot snap = g.snapshot();
  const idType q = 1000;
  const uint64_t baseline = query(snap, q, 32);
  std::printf("baseline snapshot query = %016llx\n", (unsigned long long)baseline);

  // Concurrent writer: rewire random nodes (changes reachability) + add nodes.
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> ops{0};
  std::thread writer([&] {
    std::mt19937 wr(777);
    while (!stop.load(std::memory_order_relaxed)) {
      size_t n = g.liveCount();
      idType a = wr() % n;
      std::vector<idType> nb;
      for (int e = 0; e < 8; e++) nb.push_back(wr() % n);
      g.rewire(a, nb);
      if ((ops++ & 0xFFF) == 0) {
        idType x = g.addNode();
        g.connect(x, wr() % g.liveCount());
      }
    }
  });

  // Read the snapshot lock-free, repeatedly, while writes happen.
  bool consistent = true;
  for (int i = 0; i < 5000; i++) {
    if (query(snap, q, 32) != baseline) { consistent = false; break; }
    std::this_thread::yield();
  }

  stop.store(true);
  writer.join();

  // Deterministically diverge the LIVE graph: point the entry node only at
  // itself, so a live BFS reaches just {entry}. The snapshot must be unaffected.
  g.rewire(0, {0});
  Snapshot liveNow = g.snapshot();
  uint64_t liveResult = query(liveNow, q, 32);
  uint64_t snapResult = query(snap, q, 32);

  std::printf("writer ops: %llu\n", (unsigned long long)ops.load());
  std::printf("snapshot stayed consistent during writes: %s\n",
              consistent ? "YES" : "NO");
  std::printf("snapshot query still = %016llx (== baseline: %s)\n",
              (unsigned long long)snapResult, snapResult == baseline ? "YES" : "NO");
  std::printf("live query after edit = %016llx (diverged: %s)\n",
              (unsigned long long)liveResult, liveResult != baseline ? "YES" : "no");

  size_t heldBytes = g.allocatedBytes();
  snap.root.reset();
  liveNow.root.reset();
  size_t afterBytes = g.allocatedBytes();
  std::printf("allocator bytes: baseline=%zu, while-held=%zu, after-drop=%zu\n",
              baseBytes, heldBytes, afterBytes);
  bool reclaimed = afterBytes < heldBytes;  // COW versions freed on snapshot drop

  bool pass = consistent && (snapResult == baseline) && (liveResult != baseline) &&
              reclaimed;
  std::printf("\n==== SNAPSHOT POC %s ====\n", pass ? "PASSED" : "FAILED");
  return pass ? 0 : 1;
}

// Prototype C: validate the block deep-copy algorithm that Phase 1 of the
// snapshot work needs. The real ElementGraphData has three features that make a
// copy non-trivial, and this models all three:
//   1. a flexible-array-member link list (`idType links[]`) sized per level,
//   2. an embedded std::mutex that must NOT be copied (fresh on the copy),
//   3. a heap-allocated incoming-edges vector that must be deep-copied so the
//      snapshot never shares a mutable vector with the live index.
//
// Build: g++ -O2 -std=c++20 -pthread block_deepcopy.cpp -o block_deepcopy
//
// This mirrors deps/VectorSimilarity/.../graph_data.h closely enough to prove
// the copyTo logic before applying it to the real (FAM + allocator) header.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

using idType = uint32_t;
using linkListSize = uint16_t;

// --- mirrors ElementLevelData (FAM links + incoming-edges pointer) ---
struct ElementLevelData {
  std::vector<idType>* incomingUnidirectionalEdges;
  linkListSize numLinks;
  idType links[];  // flexible array member
};

// --- mirrors ElementGraphData (mutex + others + inline level0) ---
struct ElementGraphData {
  size_t toplevel;
  std::mutex neighborsGuard;
  ElementLevelData* others;
  ElementLevelData level0;
};

static size_t levelDataSize(size_t M) {
  return sizeof(ElementLevelData) + sizeof(idType) * M;
}
static size_t elementGraphDataSize(size_t M0) {
  return sizeof(ElementGraphData) + sizeof(idType) * M0;
}

// Copy one level record of `recordSize` bytes (fixed part + links FAM), giving
// the destination an independent incoming-edges vector.
static void copyLevelInto(ElementLevelData* dst, const ElementLevelData* src,
                          size_t recordSize) {
  std::memcpy(dst, src, recordSize);  // numLinks + links FAM (+ stale ptr)
  dst->incomingUnidirectionalEdges =
      new std::vector<idType>(*src->incomingUnidirectionalEdges);  // deep copy
}

// Deep-copy `src` graph data into raw zeroed memory `dst`. Fresh mutex.
static void copyTo(ElementGraphData* dst, const ElementGraphData* src, size_t M,
                   size_t M0) {
  const size_t lds = levelDataSize(M);
  const size_t egds = elementGraphDataSize(M0);
  dst->toplevel = src->toplevel;
  new (&dst->neighborsGuard) std::mutex();  // fresh; never copy lock state
  const size_t level0Size =
      egds - (sizeof(ElementGraphData) - sizeof(ElementLevelData));
  copyLevelInto(&dst->level0, &src->level0, level0Size);
  if (src->toplevel > 0) {
    dst->others = (ElementLevelData*)std::calloc(src->toplevel, lds);
    for (size_t i = 0; i < src->toplevel; i++) {
      auto* s = (const ElementLevelData*)((const char*)src->others + i * lds);
      auto* d = (ElementLevelData*)((char*)dst->others + i * lds);
      copyLevelInto(d, s, lds);
    }
  } else {
    dst->others = nullptr;
  }
}

// Helpers to build/destroy a node the way the real ctor/destroy do.
static ElementGraphData* makeNode(size_t toplevel, size_t M, size_t M0) {
  void* mem = std::calloc(1, elementGraphDataSize(M0));
  auto* gd = (ElementGraphData*)mem;
  gd->toplevel = toplevel;
  new (&gd->neighborsGuard) std::mutex();
  gd->level0.incomingUnidirectionalEdges = new std::vector<idType>();
  gd->others = toplevel ? (ElementLevelData*)std::calloc(toplevel, levelDataSize(M))
                        : nullptr;
  for (size_t i = 0; i < toplevel; i++) {
    auto* ld = (ElementLevelData*)((char*)gd->others + i * levelDataSize(M));
    ld->incomingUnidirectionalEdges = new std::vector<idType>();
  }
  return gd;
}
static ElementLevelData& levelOf(ElementGraphData* gd, size_t lvl, size_t M) {
  return lvl == 0 ? gd->level0
                  : *(ElementLevelData*)((char*)gd->others +
                                         (lvl - 1) * levelDataSize(M));
}

int main() {
  const size_t M = 16, M0 = 32, toplevel = 3;  // node present on levels 0..3
  auto* src = makeNode(toplevel, M, M0);

  // Populate links + incoming edges on each level with recognizable data.
  for (size_t lvl = 0; lvl <= toplevel; lvl++) {
    auto& ld = levelOf(src, lvl, M);
    size_t cap = (lvl == 0) ? M0 : M;
    ld.numLinks = cap / 2;
    for (size_t j = 0; j < ld.numLinks; j++) ld.links[j] = uint32_t(lvl * 1000 + j);
    for (size_t j = 0; j < lvl + 2; j++)
      ld.incomingUnidirectionalEdges->push_back(uint32_t(lvl * 100 + j));
  }

  // Deep copy.
  void* mem = std::calloc(1, elementGraphDataSize(M0));
  auto* dst = (ElementGraphData*)mem;
  copyTo(dst, src, M, M0);

  // Verify: identical contents, but INDEPENDENT storage.
  bool ok = (dst->toplevel == src->toplevel);
  for (size_t lvl = 0; lvl <= toplevel && ok; lvl++) {
    auto& s = levelOf(src, lvl, M);
    auto& d = levelOf(dst, lvl, M);
    if (d.numLinks != s.numLinks) { ok = false; break; }
    if (std::memcmp(d.links, s.links, s.numLinks * sizeof(idType))) { ok = false; break; }
    if (*d.incomingUnidirectionalEdges != *s.incomingUnidirectionalEdges) { ok = false; break; }
    // independent pointers (not shared)
    if (d.incomingUnidirectionalEdges == s.incomingUnidirectionalEdges) { ok = false; break; }
    if (lvl > 0 && d.links == s.links) { ok = false; break; }
  }
  std::printf("contents equal + storage independent: %s\n", ok ? "YES" : "NO");

  // Mutate the SOURCE after copy; destination must be unaffected (isolation).
  levelOf(src, 0, M).links[0] = 0xFFFFFFFF;
  src->level0.incomingUnidirectionalEdges->push_back(0xABCD);
  bool isolated = (levelOf(dst, 0, M).links[0] == 0) &&  // dst kept 0*1000+0 == 0
                  (dst->level0.incomingUnidirectionalEdges->back() != 0xABCD);
  std::printf("snapshot isolated from later source mutation: %s\n",
              isolated ? "YES" : "NO");

  // Both mutexes are usable (fresh, unlocked).
  bool locks_ok = src->neighborsGuard.try_lock() && dst->neighborsGuard.try_lock();
  if (locks_ok) { src->neighborsGuard.unlock(); dst->neighborsGuard.unlock(); }
  std::printf("both mutexes fresh & lockable: %s\n", locks_ok ? "YES" : "NO");

  bool pass = ok && isolated && locks_ok;
  std::printf("\n==== PROTOTYPE C %s ====\n", pass ? "PASSED" : "FAILED");
  return pass ? 0 : 1;
}

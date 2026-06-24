// Prototype B: quantify the read-path cost of the rooted-COW layout vs today's
// flat layout, in an HNSW-traversal-like dependent random walk.
//
//   FLAT     : vector<Block>                          (today: blocks[id/bs].data)
//   INDIRECT : shared_ptr<vector<shared_ptr<Block>>>  (proposed rooted COW)
//
// Both store edges identically in a per-block heap buffer, so the ONLY
// difference measured is the container indirection (extra pointer hop + the
// block headers no longer being contiguous). Element-data locality is identical
// in both, matching the real DataBlock (data lives in a separate heap buffer).
//
// Build: g++ -O2 -std=c++20 cow_snapshot.cpp ... no; standalone:
//        g++ -O2 -std=c++20 read_indirection_bench.cpp -o read_indirection_bench

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

static constexpr size_t M = 16;            // links per element (HNSW default)
static constexpr size_t BS = 1024;         // block size
static constexpr size_t N = 1'000'000;     // elements
static constexpr size_t NB = (N + BS - 1) / BS;
static constexpr size_t STEPS = 20'000'000;  // walk length
static constexpr size_t DIM = 128;         // vector dim for the distance calc

struct Block {
  uint32_t* data;  // BS * M link ids, like ElementLevelData::links[] per elem
  Block() { data = new uint32_t[BS * M]; }
  ~Block() { delete[] data; }
  Block(const Block&) = delete;
};

int main() {
  std::mt19937 rng(98765);

  // ---- FLAT layout: vector<Block> (headers contiguous, data in heap buffers)
  std::vector<Block> flat(NB);
  // ---- INDIRECT layout: shared_ptr<vector<shared_ptr<Block>>>
  auto bb = std::make_shared<std::vector<std::shared_ptr<Block>>>();
  bb->reserve(NB);
  for (size_t i = 0; i < NB; i++) bb->push_back(std::make_shared<Block>());
  auto root = std::make_shared<decltype(bb)::element_type>(*bb);

  // Same random edge data in both layouts.
  for (size_t b = 0; b < NB; b++)
    for (size_t k = 0; k < BS * M; k++) {
      uint32_t v = rng() % N;
      flat[b].data[k] = v;
      (*root)[b]->data[k] = v;
    }

  // Shared vector data (one copy, used by both layouts) for a realistic
  // distance computation per visit. This is the dominant per-visit cost in real
  // HNSW and is identical between layouts; only link lookup differs.
  std::vector<float> vecs(N * DIM);
  for (auto& f : vecs) f = float(rng()) / float(rng.max());
  std::vector<float> query(DIM);
  for (auto& f : query) f = float(rng()) / float(rng.max());
  auto l2 = [&](uint32_t id) {
    const float* v = vecs.data() + size_t(id) * DIM;
    float s = 0;
    for (size_t d = 0; d < DIM; d++) {
      float diff = v[d] - query[d];
      s += diff * diff;
    }
    return s;
  };

  auto walk_flat = [&](uint32_t start) {
    uint32_t cur = start;
    uint64_t acc = 0;
    for (size_t s = 0; s < STEPS; s++) {
      size_t b = cur / BS, e = cur % BS;
      const uint32_t* links = flat[b].data + e * M;
      uint32_t nxt = links[cur % M];
      acc += nxt;
      cur = nxt;
    }
    return acc;
  };
  auto walk_indirect = [&](uint32_t start) {
    const auto& R = *root;
    uint32_t cur = start;
    uint64_t acc = 0;
    for (size_t s = 0; s < STEPS; s++) {
      size_t b = cur / BS, e = cur % BS;
      const uint32_t* links = R[b]->data + e * M;  // extra hop: R[b]-> ...
      uint32_t nxt = links[cur % M];
      acc += nxt;
      cur = nxt;
    }
    return acc;
  };

  auto bench = [&](const char* name, auto fn) {
    volatile uint64_t sink = 0;
    double best = 1e30;
    for (int trial = 0; trial < 3; trial++) {
      auto t0 = std::chrono::steady_clock::now();
      sink = fn(123457u);
      auto t1 = std::chrono::steady_clock::now();
      double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
      best = std::min(best, ns / STEPS);
    }
    std::printf("%-10s : %.3f ns/visit\n", name, best);
    (void)sink;
    return best;
  };

  // Realistic variants: same walk, but compute a distance per visit.
  auto walk_flat_dist = [&](uint32_t start) {
    uint32_t cur = start;
    double acc = 0;
    for (size_t s = 0; s < STEPS; s++) {
      size_t b = cur / BS, e = cur % BS;
      const uint32_t* links = flat[b].data + e * M;
      uint32_t nxt = links[cur % M];
      acc += l2(nxt);
      cur = nxt;
    }
    return uint64_t(acc);
  };
  auto walk_indirect_dist = [&](uint32_t start) {
    const auto& R = *root;
    uint32_t cur = start;
    double acc = 0;
    for (size_t s = 0; s < STEPS; s++) {
      size_t b = cur / BS, e = cur % BS;
      const uint32_t* links = R[b]->data + e * M;
      uint32_t nxt = links[cur % M];
      acc += l2(nxt);
      cur = nxt;
    }
    return uint64_t(acc);
  };

  // Third layout: shared_ptr<vector<Block>> — blocks BY VALUE in the vector
  // (the original "shared_ptr<vector<DataBlock>>" idea). Aliases `flat` (no-op
  // deleter) so it reads the same data. Backbone access is a contiguous-array
  // load like FLAT; isolates whether the cost is the per-block shared_ptr.
  std::shared_ptr<std::vector<Block>> rootVal(&flat, [](std::vector<Block>*) {});
  auto walk_val = [&](uint32_t start) {
    const auto& R = *rootVal;
    uint32_t cur = start;
    uint64_t acc = 0;
    for (size_t s = 0; s < STEPS; s++) {
      size_t b = cur / BS, e = cur % BS;
      const uint32_t* links = R[b].data + e * M;  // contiguous header, by value
      uint32_t nxt = links[cur % M];
      acc += nxt;
      cur = nxt;
    }
    return acc;
  };

  // Fourth layout: shared_ptr<vector<BlockSP>> where each BlockSP holds a
  // REFCOUNTED data buffer (shared_ptr<uint32_t[]>). Headers stay contiguous
  // (fast reads) AND each block's buffer can be COW'd independently (cheap
  // writes). This is the candidate "best of both" structure.
  struct BlockSP {
    std::shared_ptr<uint32_t[]> data;
  };
  auto rootSP = std::make_shared<std::vector<BlockSP>>(NB);
  for (size_t b = 0; b < NB; b++) {
    (*rootSP)[b].data = std::shared_ptr<uint32_t[]>(new uint32_t[BS * M]);
    for (size_t k = 0; k < BS * M; k++) (*rootSP)[b].data[k] = flat[b].data[k];
  }
  auto walk_sp = [&](uint32_t start) {
    const auto& R = *rootSP;
    uint32_t cur = start;
    uint64_t acc = 0;
    for (size_t s = 0; s < STEPS; s++) {
      size_t b = cur / BS, e = cur % BS;
      const uint32_t* links = R[b].data.get() + e * M;  // contiguous hdr -> buf
      uint32_t nxt = links[cur % M];
      acc += nxt;
      cur = nxt;
    }
    return acc;
  };

  std::printf("walk of %zu steps over %zu elements (%zu blocks), dim=%zu\n",
              STEPS, N, NB, DIM);
  std::printf("\n-- link lookup only (isolates the indirection) --\n");
  double f = bench("FLAT", walk_flat);
  double v = bench("VAL-vec", walk_val);
  double sp = bench("VAL+spbuf", walk_sp);
  double i = bench("INDIRECT", walk_indirect);
  std::printf("VAL+spbuf (contiguous hdr + refcounted buffer) vs FLAT: %+.1f%%\n",
              100.0 * (sp - f) / f);
  std::printf("shared_ptr<vector<Block>>  overhead vs FLAT: %+.1f%%\n",
              100.0 * (v - f) / f);
  std::printf("shared_ptr<vec<shared_ptr>> overhead vs FLAT: %+.1f%% (%.3f ns)\n",
              100.0 * (i - f) / f, i - f);

  std::printf("\n-- with distance calc (realistic per-visit work) --\n");
  double fd = bench("FLAT+dist", walk_flat_dist);
  double id = bench("INDIR+dist", walk_indirect_dist);
  std::printf("indirection overhead: %+.1f%% (%.3f ns/visit)\n",
              100.0 * (id - fd) / fd, id - fd);
  return 0;
}

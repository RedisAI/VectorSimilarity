#include "abstract.hpp"
#include "with_files_directly.hpp"
#include "with_rocksdb.hpp"

#include <benchmark/benchmark.h>

static void BM_BFSWithFiles(benchmark::State &state) {
    BFSWithFiles bfs(state.range(0), state.range(1));
    for (auto _ : state) {
        bfs.scanGraph();
    }
}

static void BM_BFSWithRocksDB(benchmark::State &state) {
    BFSWithRocksDB bfs(state.range(0), state.range(1));
    for (auto _ : state) {
        bfs.scanGraph();
    }
}

const std::vector<std::pair<int64_t, int64_t>> ranges = {
    {1L << 10, 1L << 20},
    {1L, 100L},
};

#define BM_ARGS Ranges(ranges)->Unit(benchmark::kMillisecond)

BENCHMARK(BM_BFSWithFiles)->BM_ARGS;
BENCHMARK(BM_BFSWithRocksDB)->BM_ARGS;

BENCHMARK_MAIN();

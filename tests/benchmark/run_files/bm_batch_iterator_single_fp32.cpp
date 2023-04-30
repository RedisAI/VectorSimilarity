#include "benchmark/bm_batch_iterator.h"

bool BM_VecSimGeneral::is_multi = false;

// Global benchmark data
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::block_size = 1024;

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

JobQueue BM_VecSimGeneral::jobQ{};
const size_t BM_VecSimGeneral::thread_pool_size = MIN(8, std::thread::hardware_concurrency());
std::vector<std::thread> BM_VecSimGeneral::thread_pool{};
std::mutex BM_VecSimGeneral::queue_guard{};
std::condition_variable BM_VecSimGeneral::queue_cond{};
bool BM_VecSimGeneral::run_threads = false;

#define BM_FUNC_NAME(bm_func, algo) algo##_##bm_func##_Single

#include "benchmark/bm_initialization/bm_batch_initialize_fp32.h"

BENCHMARK_MAIN();

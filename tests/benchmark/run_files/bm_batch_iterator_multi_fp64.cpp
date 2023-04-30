#include "benchmark/bm_batch_iterator.h"

bool BM_VecSimGeneral::is_multi = true;

// Global benchmark data
size_t BM_VecSimGeneral::n_vectors = 1111025;
size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 512;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::block_size = 1024;

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/fp64_fashion_images_multi_value-M64-efc512.hnsw_v2";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/fashion_images_multi_test_vecs_fp64.raw";

JobQueue BM_VecSimGeneral::jobQ{};
const size_t BM_VecSimGeneral::thread_pool_size = MIN(8, std::thread::hardware_concurrency());
std::vector<std::thread> BM_VecSimGeneral::thread_pool{};
std::mutex BM_VecSimGeneral::queue_guard{};
std::condition_variable BM_VecSimGeneral::queue_cond{};
bool BM_VecSimGeneral::run_threads = false;

#define BM_FUNC_NAME(bm_func, algo) algo##_##bm_func##_Multi

#include "benchmark/bm_initialization/bm_batch_initialize_fp64.h"

BENCHMARK_MAIN();

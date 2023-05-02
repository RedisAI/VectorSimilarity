#include "benchmark/bm_vecsim_basics.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"

/**************************************
  Basic tests for single value index with fp32 data type.
***************************************/

bool BM_VecSimGeneral::is_multi = false;

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;

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

#define BM_FUNC_NAME(bm_func, algo) bm_func##_##algo##_Single
#define BM_ADD_LABEL                AddLabel_Single

DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, BF), fp32_index_t, BruteForceIndex_Single, float,
                    float, VecSimAlgo_BF)
DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, HNSW), fp32_index_t, HNSWIndex_Single, float, float,
                    VecSimAlgo_HNSWLIB)
DEFINE_DELETE_LABEL(DeleteLabel_Tiered, fp32_index_t, TieredHNSWIndex, float, float,
                    VecSimAlgo_TIERED)
#include "benchmark/bm_initialization/bm_basics_initialize_fp32.h"
BENCHMARK_MAIN();

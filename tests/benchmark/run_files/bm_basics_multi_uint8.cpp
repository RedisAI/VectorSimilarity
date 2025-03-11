#include "benchmark/bm_vecsim_basics.h"
#include "VecSim/algorithms/brute_force/brute_force_multi.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"

/**************************************
  Basic tests for multi value index with uint8 data type.
***************************************/

bool BM_VecSimGeneral::is_multi = true;

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::dim = 1024;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
tieredIndexMock BM_VecSimGeneral::mock_thread_pool{};

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/wipedia_multi_uint8-cosine-dim1024-M64-efc512-uint8.hnsw_v3";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/wipedia_multi_uint8-cosine-dim1024-uint8-test_vectors.raw";

#define BM_FUNC_NAME(bm_func, algo) bm_func##_##algo##_Multi
#define BM_ADD_LABEL                AddLabel_Multi
#define BM_ADD_LABEL_ASYNC          AddLabel_Async_Multi
#define BM_DELETE_LABEL_ASYNC       DeleteLabel_Async_Multi

DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, BF), uint8_index_t, BruteForceIndex_Multi, uint8_t,
                    float, VecSimAlgo_BF)
DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, HNSW), uint8_index_t, HNSWIndex_Multi, uint8_t, float,
                    VecSimAlgo_HNSWLIB)
DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, Tiered), uint8_index_t, TieredHNSWIndex, uint8_t, float,
                    VecSimAlgo_TIERED)
#include "benchmark/bm_initialization/bm_basics_initialize_uint8.h"

BENCHMARK_MAIN();

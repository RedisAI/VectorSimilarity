#include "benchmark/bm_vecsim_basics.h"
#include "VecSim/algorithms/brute_force/brute_force_multi.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"

/**************************************
  Basic tests for multi value index.
***************************************/

bool BM_VecSimGeneral::is_multi = true;

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::n_vectors = 1111025;
size_t BM_VecSimGeneral::dim = 512;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/fp64_fashion_images_multi_value-M64-efc512.hnsw_v2";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/fashion_images_multi_test_vecs_fp64.raw";

#define BM_FUNC_NAME(bm_func, algo) bm_func##_##algo##_Multi
#define BM_ADD_LABEL                AddLabel_Multi

DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, BF), fp64_index_t, BruteForceIndex_Multi, double,
                    double, VecSimAlgo_BF)
DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, HNSW), fp64_index_t, HNSWIndex_Multi, double, double,
                    VecSimAlgo_HNSWLIB)
#include "benchmark/bm_initialization/bm_basics_initialize_fp64.h"

BENCHMARK_MAIN();

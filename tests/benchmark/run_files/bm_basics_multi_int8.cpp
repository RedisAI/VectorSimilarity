#include "benchmark/bm_vecsim_basics.h"
#include "VecSim/algorithms/brute_force/brute_force_multi.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"

/**************************************
  Basic tests for multi value index with int8 data type.
***************************************/

bool BM_VecSimGeneral::is_multi = true;

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::dim = 1024;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
tieredIndexMock BM_VecSimGeneral::mock_thread_pool{};

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/wipedia_multi-cosine-dim1024-M64-efc512-int8.hnsw_v3";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/wipedia_multi-cosine-dim1024-int8-test_vectors.raw";

#define BM_FUNC_NAME(bm_func, algo) bm_func##_##algo##_Multi
#define BM_ADD_LABEL                AddLabel_Multi
#define BM_ADD_LABEL_ASYNC          AddLabel_Async_Multi
#define BM_DELETE_LABEL_ASYNC       DeleteLabel_Async_Multi

// Larger Range query values are required for wikipedia dataset.
// Current values in bm_vecsim_basics.h gives 0 results
#define REGISTER_Range_INT8_HNSW(BM_FUNC)                                                          \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->Args({50, 1})                                                                            \
        ->Args({50, 10})                                                                           \
        ->Args({50, 100})                                                                          \
        ->Args({65, 1})                                                                            \
        ->Args({65, 10})                                                                           \
        ->Args({65, 100})                                                                          \
        ->Args({80, 1})                                                                            \
        ->Args({80, 10})                                                                           \
        ->Args({80, 100})                                                                          \
        ->ArgNames({"radiusX100", "epsilonX1000"})                                                 \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

#define REGISTER_Range_INT8_BF(BM_FUNC)                                                            \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->Arg(50)                                                                                  \
        ->Arg(65)                                                                                  \
        ->Arg(80)                                                                                  \
        ->ArgName("radiusX100")                                                                    \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, BF), int8_index_t, BruteForceIndex_Multi, int8_t,
                    float, VecSimAlgo_BF)
DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, HNSW), int8_index_t, HNSWIndex_Multi, int8_t, float,
                    VecSimAlgo_HNSWLIB)
DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, Tiered), int8_index_t, TieredHNSWIndex, int8_t, float,
                    VecSimAlgo_TIERED)
#include "benchmark/bm_initialization/bm_basics_initialize_int8.h"

BENCHMARK_MAIN();

#include "benchmark/bm_special_index.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "utils/mock_thread_pool.h"

/**************************************
  Basic tests for single value index with fp32 data type.
***************************************/

bool BM_VecSimGeneral::is_multi = false;

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
tieredIndexMock BM_VecSimGeneral::mock_thread_pool(false);

const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/dbpedia-cosine-dim768-test_vectors.raw";
const char *BM_VecSimGeneral::hnsw_index_file = NULL;
template <>
const char *BM_VecSimSpecialIndex<fp16_index_t>::raw_vectors_file =
    "tests/benchmark/data/dbpedia-cosine-dim768-vectors.raw";

#define BM_FUNC_NAME(bm_func, algo) bm_func##_##algo##_Single_fp16

#include "benchmark/bm_initialization/bm_basics_initialize_fp16.h"
BENCHMARK_MAIN();

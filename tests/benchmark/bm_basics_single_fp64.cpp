#include "bm_basics.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
/**************************************
  Basic tests for single value index with fp64 data type.
***************************************/

bool BM_VecSimGeneral::is_multi = false;

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::n_vectors = 1000000;

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

DEFINE_DELETE_VECTOR(DeleteVector_BF_FP64, fp64_index_t, BruteForceIndex_Single, double, double,
                     VecSimAlgo_BF)
DEFINE_DELETE_VECTOR(DeleteVector_HNSW_FP64, fp64_index_t, HNSWIndex_Single, double, double,
                     VecSimAlgo_HNSWLIB)
#include "bm_basics_define_n_register_fp64.h"

BENCHMARK_MAIN();

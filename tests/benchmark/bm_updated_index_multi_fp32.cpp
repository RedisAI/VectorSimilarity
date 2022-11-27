
#include "bm_updated_index.h"

/**************************************
  Basic tests for updated multi value index.
***************************************/
bool BM_VecSimGeneral::is_multi = true;

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 65;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::n_vectors = 500000;

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512.hnsw";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

template <>
const char *BM_VecSimUpdatedIndex<fp32_index_t>::updated_hnsw_index_file =
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512-updated.hnsw";

#include "bm_updated_define_n_register_fp32.h"
BENCHMARK_MAIN();
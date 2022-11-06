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

std::vector<const char *> BM_VecSimGeneral::hnsw_index_files = {
    // TODO serialize both with v2
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512_MULTI.hnsw",
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512_MULTI_FP64.hnsw_v2"};

std::vector<const char *> BM_VecSimGeneral::test_vectors_files = {
    // TODO create test vector for FP64
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw",
    "tests/benchmark/data/DBpedia-test_vectors-n10k_FP64.raw"};

template <typename index_type_t>
std::vector<const char *> BM_VecSimUpdatedIndex<index_type_t>::updated_hnsw_index_files = {
    // TODO create updated index files for FP64
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512-updated.hnsw",
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512_FP64 -updated_.hnsw"};

BENCHMARK_MAIN();

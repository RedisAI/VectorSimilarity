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

template <typename index_type_t>
const std::vector<const char *> BM_VecSimIndex<index_type_t>::GetIndexFiles() {
    static const std::vector<const char *> v = {
        "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1",
        "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1"};

    return v;
}

template <typename index_type_t>
const std::vector<const char *> BM_VecSimIndex<index_type_t>::GetTestFiles() {
    static const std::vector<const char *> v = {
        "tests/benchmark/data/DBpedia-test_vectors-n10k.raw",
        "tests/benchmark/data/DBpedia-test_vectors-n10k.raw"};

    return v;
}

template <typename index_type_t>
const std::vector<const char *> BM_VecSimUpdatedIndex<index_type_t>::GetUpdatedIndexFiles() {
    static const std::vector<const char *> v = {
        "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512-updated.hnsw",
        "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512-updated.hnsw"};

    return v;
}
BENCHMARK_MAIN();

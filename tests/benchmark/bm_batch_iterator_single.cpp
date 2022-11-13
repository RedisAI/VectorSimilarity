#include "bm_batch_iterator.h"

bool BM_VecSimGeneral::is_multi = false;

// Global benchmark data
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::block_size = 1024;

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
BENCHMARK_MAIN();

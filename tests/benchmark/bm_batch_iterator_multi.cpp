
#include "bm_batch_iterator.h"

bool BM_VecSimGeneral::is_multi = true;

// Global benchmark data
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::block_size = 1024;

size_t BM_VecSimGeneral::ref_count = 0;

std::vector<const char *> BM_VecSimGeneral::hnsw_index_files = {
    // TODO serialize both with v2
    "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1",
    "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512_FP64.hnsw_v2"};

std::vector<const char *> BM_VecSimGeneral::test_vectors_files = {
    // TODO create test vector for FP64
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw",
    "tests/benchmark/data/DBpedia-test_vectors-n10k_FP64.raw"};
BENCHMARK_MAIN();

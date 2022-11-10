
#include "bm_batch_iterator.h"



bool BM_VecSimGeneral::is_multi = true;

// Global benchmark data
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::block_size = 1024;

std::vector<const char *> init_index_files() { 
    std::vector<const char *> v = {"tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1", 
    "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1"};

    return v;
}

std::vector<const char *> init_test_files() { 
    std::vector<const char *> v = {"tests/benchmark/data/DBpedia-test_vectors-n10k.raw", 
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw"};

    return v;
}
template <typename index_type_t>
BM_BatchIterator<index_type_t>::BM_BatchIterator() :  BM_VecSimIndex<index_type_t>(init_index_files(), init_test_files())  {}
BENCHMARK_MAIN();
                                                                   
 

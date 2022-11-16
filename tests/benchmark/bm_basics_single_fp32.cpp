#include "bm_basics.h"

/**************************************
  Basic tests for single value index with fp32 data type.
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

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::AddVector(benchmark::State &st) {
    // Add a new vector from the test vectors in every iteration.
    size_t iter = 0;
    size_t new_id = VecSimIndex_IndexSize(INDICES[st.range(0)]);
    size_t memory_delta = 0;
    for (auto _ : st) {
        memory_delta +=
            VecSimIndex_AddVector(INDICES[st.range(0)], QUERIES[iter % N_QUERIES].data(), new_id++);
        iter++;
    }
    st.counters["memory"] = (double)memory_delta / (double)iter;

    // Clean-up.
    size_t new_index_size = VecSimIndex_IndexSize(INDICES[st.range(0)]);
    for (size_t id = N_VECTORS; id < new_index_size; id++) {
        VecSimIndex_DeleteVector(INDICES[st.range(0)], id);
    }
}

#include "bm_basics_define_n_register_fp32.h"

BENCHMARK_MAIN();

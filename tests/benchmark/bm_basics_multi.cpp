#include "bm_basics.h"

/**************************************
  Basic tests for single value index.
***************************************/

bool BM_VecSimGeneral::is_multi = true;

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::n_vectors = 1000000;

std::vector<const char *> BM_VecSimGeneral::hnsw_index_files = {
    // TODO serialize both with v2
    "tests/benchmark/data/MULTI_FP32.hnsw_v2", "tests/benchmark/data/MULTI_FP64.hnsw_v2"};

std::vector<const char *> BM_VecSimGeneral::test_vectors_files = {
    // TODO create test vector for FP64
    "tests/benchmark/data/test_vectorsMULTI_FP32.raw",
    "tests/benchmark/data/test_vectorsMULTI_FP64.raw"};

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::AddVector(benchmark::State &st) {
    // TODO write
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

// AddVector BM
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, AddVector_fp32, fp32_index_t)
(benchmark::State &st) { AddVector(st); }

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, AddVector_fp64, fp64_index_t)
(benchmark::State &st) { AddVector(st); }

BENCHMARK_REGISTER_F(BM_VecSimBasics, AddVector_fp32)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)
    ->Arg(VecSimAlgo_HNSWLIB);
BENCHMARK_REGISTER_F(BM_VecSimBasics, AddVector_fp64)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)
    ->Arg(VecSimAlgo_HNSWLIB);

BENCHMARK_MAIN();

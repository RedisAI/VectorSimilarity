#pragma once
#include "bm_common.h"

template <typename index_type_t>
class BM_VecSimBasics : public BM_VecSimCommon<index_type_t> {
public:
    using data_t = typename index_type_t::data_t;

    BM_VecSimBasics() = default;
    ~BM_VecSimBasics() = default;

    // Different implementation for multi and single.
    static void AddVector(benchmark::State &st);

    // we Pass a specific index pointer instead of VecSimIndex * so we can use getDataByInternalId
    // which is not known to VecSimIndex class.
    template <typename algo_t>
    static void DeleteVector(algo_t *index, benchmark::State &st);

    static void Range_BF(benchmark::State &st);
    static void Range_HNSW(benchmark::State &st);
};

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::AddVector(benchmark::State &st) {
    // Add a new vector from the test vectors in every iteration.
    size_t iter = 0;

    size_t index_size = N_VECTORS;
    size_t initial_label_count = (INDICES[st.range(0)])->indexLabelCount();

    // In a single vector per label index, index size should equal label count.
    size_t vec_per_label = index_size % initial_label_count == 0
                               ? index_size / initial_label_count
                               : index_size / initial_label_count + 1;
    size_t vec_id = initial_label_count * vec_per_label;
    size_t memory_delta = 0;

    for (auto _ : st) {
        memory_delta += VecSimIndex_AddVector(
            INDICES[st.range(0)], QUERIES[iter % N_QUERIES].data(), vec_id / vec_per_label);
        vec_id++;
        iter++;
    }
    st.counters["memory"] = (double)memory_delta / (double)iter;

    assert(VecSimIndex_IndexSize(INDICES[st.range(0)]) == N_VECTORS + iter);

    // Clean-up all the new vectors to restore the index size to its original value.

    size_t new_label_count = (INDICES[st.range(0)])->indexLabelCount();
    for (size_t id = initial_label_count; id < new_label_count; id++) {
        VecSimIndex_DeleteVector(INDICES[st.range(0)], id);
    }

    assert(VecSimIndex_IndexSize(INDICES[st.range(0)]) == N_VECTORS);
}

template <typename index_type_t>
template <typename algo_t>
void BM_VecSimBasics<index_type_t>::DeleteVector(algo_t *index, benchmark::State &st) {
    // Remove a different vector in every execution.
    std::vector<std::vector<data_t>> blobs;
    size_t id_to_remove = 0;
    double memory_delta = 0;
    size_t iter = 0;

    for (auto _ : st) {
        st.PauseTiming();
        auto removed_vec = std::vector<data_t>(DIM);
        memcpy(removed_vec.data(), index->getDataByInternalId(id_to_remove), DIM * sizeof(data_t));
        blobs.push_back(removed_vec);
        st.ResumeTiming();

        iter++;
        auto delta = (double)VecSimIndex_DeleteVector(index, id_to_remove++);
        memory_delta += delta;
    }
    st.counters["memory"] = memory_delta / (double)iter;

    // Restore index state.
    for (size_t i = 0; i < blobs.size(); i++) {
        VecSimIndex_AddVector(index, blobs[i].data(), i);
    }
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::Range_BF(benchmark::State &st) {
    double radius = (1.0 / 100.0) * (double)st.range(0);
    size_t iter = 0;
    size_t total_res = 0;

    for (auto _ : st) {
        auto res = VecSimIndex_RangeQuery(INDICES[VecSimAlgo_BF], QUERIES[iter % N_QUERIES].data(),
                                          radius, nullptr, BY_ID);
        total_res += VecSimQueryResult_Len(res);
        iter++;
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::Range_HNSW(benchmark::State &st) {
    double radius = (1.0 / 100.0) * (double)st.range(0);
    double epsilon = (1.0 / 1000.0) * (double)st.range(1);
    size_t iter = 0;
    size_t total_res = 0;
    size_t total_res_bf = 0;
    HNSWRuntimeParams hnswRuntimeParams = {.epsilon = epsilon};
    auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);

    for (auto _ : st) {
        auto hnsw_results =
            VecSimIndex_RangeQuery(INDICES[VecSimAlgo_HNSWLIB], QUERIES[iter % N_QUERIES].data(),
                                   radius, &query_params, BY_ID);
        st.PauseTiming();
        total_res += VecSimQueryResult_Len(hnsw_results);

        // Measure recall:
        auto bf_results = VecSimIndex_RangeQuery(
            INDICES[VecSimAlgo_BF], QUERIES[iter % N_QUERIES].data(), radius, nullptr, BY_ID);
        total_res_bf += VecSimQueryResult_Len(bf_results);

        VecSimQueryResult_Free(bf_results);
        VecSimQueryResult_Free(hnsw_results);
        iter++;
        st.ResumeTiming();
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
    st.counters["Recall"] = (float)total_res / total_res_bf;
}

#define UNIT_AND_ITERATIONS                                                                        \
    Unit(benchmark::kMillisecond)->Iterations((long)BM_VecSimGeneral::block_size)

// The actual radius will be the given arg divided by 100, since arg must be an integer.
#define REGISTER_Range_BF(BM_FUNC)                                                                 \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->Arg(20)                                                                                  \
        ->Arg(35)                                                                                  \
        ->Arg(50)                                                                                  \
        ->ArgName("radiusX100")                                                                    \
        ->Unit(benchmark::kMillisecond)

// {radius*100, epsilon*1000}
// The actual radius will be the given arg divided by 100, and the actual epsilon values
// will be the given arg divided by 1000.
#define REGISTER_Range_HNSW(BM_FUNC)                                                               \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->Args({20, 1})                                                                            \
        ->Args({20, 10})                                                                           \
        ->Args({20, 100})                                                                          \
        ->Args({35, 1})                                                                            \
        ->Args({35, 10})                                                                           \
        ->Args({35, 100})                                                                          \
        ->Args({50, 1})                                                                            \
        ->Args({50, 10})                                                                           \
        ->Args({50, 100})                                                                          \
        ->ArgNames({"radiusX100", "epsilonX1000"})                                                 \
        ->Iterations(100)                                                                          \
        ->Unit(benchmark::kMillisecond)

#define REGISTER_AddVector(BM_FUNC, VecSimAlgo)                                                    \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo)                                                     \
        ->ArgName(#VecSimAlgo)

#define DEFINE_DELETE_VECTOR(BM_FUNC, INDEX_TYPE, INDEX_NAME, DATA_TYPE, DIST_TYPE, VecSimAlgo)    \
    BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_FUNC, INDEX_TYPE)(benchmark::State & st) {     \
        DeleteVector<INDEX_NAME<DATA_TYPE, DIST_TYPE>>(                                            \
            reinterpret_cast<INDEX_NAME<DATA_TYPE, DIST_TYPE> *>(                                  \
                BM_VecSimIndex<INDEX_TYPE>::indices[VecSimAlgo]),                                  \
            st);                                                                                   \
    }
#define DEFINE_DELETE_VECTOR_BF(FP_TYPE, DATA_TYPE, DIST_TYPE)                                     \
    DEFINE_DELETE_VECTOR(DeleteVector_BF_##FP_TYPE, FP_TYPE##_index_t, BruteForceIndex, DATA_TYPE, \
                         DIST_TYPE, VecSimAlgo_BF)
#define DEFINE_DELETE_VECTOR_HNSW(FP_TYPE, DATA_TYPE, DIST_TYPE)                                   \
    DEFINE_DELETE_VECTOR(DeleteVector_HNSW_##FP_TYPE, FP_TYPE##_index_t, HNSWIndex, DATA_TYPE,     \
                         DIST_TYPE, VecSimAlgo_HNSWLIB)
#define REGISTER_DeleteVector(BM_FUNC)                                                             \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)->UNIT_AND_ITERATIONS

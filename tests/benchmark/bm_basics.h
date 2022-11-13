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

// DeleteVector BM
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, DeleteVector_fp32, fp32_index_t)
(benchmark::State &st) {
    if (VecSimAlgo_BF == st.range(0)) {
        DeleteVector<BruteForceIndex<float, float>>(
            reinterpret_cast<BruteForceIndex<float, float> *>(
                BM_VecSimIndex<fp32_index_t>::indices[VecSimAlgo_BF]),
            st);
    } else if (VecSimAlgo_HNSWLIB == st.range(0)) {
        DeleteVector<HNSWIndex<float, float>>(
            reinterpret_cast<HNSWIndex<float, float> *>(
                BM_VecSimIndex<fp32_index_t>::indices[VecSimAlgo_HNSWLIB]),
            st);
    }
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, DeleteVector_fp64, fp64_index_t)
(benchmark::State &st) {
    if (VecSimAlgo_BF == st.range(0)) {
        DeleteVector<BruteForceIndex<double, double>>(
            reinterpret_cast<BruteForceIndex<double, double> *>(
                BM_VecSimIndex<fp64_index_t>::indices[VecSimAlgo_BF]),
            st);
    } else if (VecSimAlgo_HNSWLIB == st.range(0)) {
        DeleteVector<HNSWIndex<double, double>>(
            reinterpret_cast<HNSWIndex<double, double> *>(
                BM_VecSimIndex<fp64_index_t>::indices[VecSimAlgo_HNSWLIB]),
            st);
    }
}

#define UNIT_AND_ITERATIONS                                                                        \
    Unit(benchmark::kMillisecond)->Iterations((long)BM_VecSimGeneral::block_size)

BENCHMARK_REGISTER_F(BM_VecSimBasics, DeleteVector_fp32)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)
    ->Arg(VecSimAlgo_HNSWLIB);
BENCHMARK_REGISTER_F(BM_VecSimBasics, DeleteVector_fp64)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)
    ->Arg(VecSimAlgo_HNSWLIB);

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Range_BF_fp32, fp32_index_t)
(benchmark::State &st) { Range_BF(st); }

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Range_BF_fp64, fp64_index_t)
(benchmark::State &st) { Range_BF(st); }

// The actual radius will be the given arg divided by 100, since arg must be an integer.
#define REGISTER_Range_BF(BM_FUNC)                                                                 \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->Arg(20)                                                                                  \
        ->ArgName("radiusX100")                                                                    \
        ->Arg(35)                                                                                  \
        ->ArgName("radiusX100")                                                                    \
        ->Arg(50)                                                                                  \
        ->ArgName("radiusX100")                                                                    \
        ->Unit(benchmark::kMillisecond)

REGISTER_Range_BF(Range_BF_fp32);
REGISTER_Range_BF(Range_BF_fp64);

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Range_HNSW_fp32, fp32_index_t)
(benchmark::State &st) { Range_HNSW(st); }

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Range_HNSW_fp64, fp64_index_t)
(benchmark::State &st) { Range_HNSW(st); }

#define HNSW_RANGE_ARGS(radius, epsilon)                                                           \
    Args({radius, epsilon})->ArgNames({"radiusX100", "epsilonX1000"})
// {radius*100, epsilon*1000}
// The actual radius will be the given arg divided by 100, and the actual epsilon values
// will be the given arg divided by 1000.
#define REGISTER_Range_HNSW(BM_FUNC)                                                               \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->HNSW_RANGE_ARGS(20, 1)                                                                   \
        ->HNSW_RANGE_ARGS(20, 10)                                                                  \
        ->HNSW_RANGE_ARGS(20, 100)                                                                 \
        ->HNSW_RANGE_ARGS(35, 1)                                                                   \
        ->HNSW_RANGE_ARGS(35, 10)                                                                  \
        ->HNSW_RANGE_ARGS(35, 100)                                                                 \
        ->HNSW_RANGE_ARGS(50, 1)                                                                   \
        ->HNSW_RANGE_ARGS(50, 10)                                                                  \
        ->HNSW_RANGE_ARGS(50, 100)                                                                 \
        ->Iterations(100)                                                                          \
        ->Unit(benchmark::kMillisecond)

REGISTER_Range_HNSW(Range_HNSW_fp32);
REGISTER_Range_HNSW(Range_HNSW_fp64);

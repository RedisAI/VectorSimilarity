#include "bm_utils.h"

template <VecSimType type, typename DataType, typename DistType = DataType>
struct IndexType {
    static VecSimType get_index_type() { return type; }
    typedef DataType data_t;
    typedef DistType dist_t;
};
using fp32_index_t = IndexType<VecSimType_FLOAT32, float, float>;
using fp64_index_t = IndexType<VecSimType_FLOAT64, double, double>;
template <>
size_t BM_VecSimBasics<false>::n_vectors = 1000000;
template <>
size_t BM_VecSimBasics<false>::n_queries = 10000;
template <>
size_t BM_VecSimBasics<false>::dim = 768;
template <>
size_t BM_VecSimBasics<false>::M = 64;
template <>
size_t BM_VecSimBasics<false>::EF_C = 512;
template <>
size_t BM_VecSimBasics<false>::block_size = 1024;
template <>
const char *BM_VecSimBasics<false>::hnsw_index_file =
    "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1";
template <>
const char *BM_VecSimBasics<false>::test_vectors_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";
template <>
size_t BM_VecSimBasics<false>::ref_count = 0;

template <typename index_type_t>
class BM_VecSimBasics_Single : public BM_VecSimBasics<false> {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    BM_VecSimBasics_Single() : BM_VecSimBasics<false>(index_type_t::get_index_type()){};

    static std::vector<std::vector<data_t>> queries;

    void AddVector(benchmark::State &st);

    // we Pass a specific index pointer instead of VecSimIndex * so we can use getDataByInternalId
    // which is not known to VecSimIndex class.
    template <typename algo_t>
    void DeleteVector(algo_t *index, benchmark::State &st);

    void TopK_BF(size_t k, benchmark::State &st);
    void TopK_HNSW(size_t ef, size_t k, benchmark::State &st);

    void Range_BF(double radius, benchmark::State &st);
    void Range_HNSW(double radius, double epsilon, benchmark::State &st);

    void Memory_FLAT(benchmark::State &st);
    void Memory_HNSW(benchmark::State &st);

    static std::vector<VecSimIndex *> indices;

    // Functions that are used by BM_VecSimBasics::Initialize()
    virtual void InitializeIndicesVector(VecSimIndex *bf_index, VecSimIndex *hnsw_index) override;
    virtual void InsertToQueries(std::ifstream &input) override;
    virtual void LoadHNSWIndex(std::string location) override;
    virtual inline char *GetHNSWDataByInternalId(size_t id) const override {
        return CastToHNSW(indices[VecSimAlgo_HNSWLIB])->getDataByInternalId(id);
    }
    virtual inline VecSimIndex *GetBF() override { return indices[VecSimAlgo_BF]; }

    ~BM_VecSimBasics_Single();

private:
    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) const {
        return reinterpret_cast<HNSWIndex<data_t, dist_t> *>(index);
    }

    inline std::vector<data_t> NewQuery() {
        std::vector<data_t> ret(dim);
        return ret;
    }
};

template <typename index_type_t>
std::vector<VecSimIndex *>
    BM_VecSimBasics_Single<index_type_t>::indices = std::vector<VecSimIndex *>();
template <typename index_type_t>
std::vector<std::vector<typename index_type_t::data_t>>
    BM_VecSimBasics_Single<index_type_t>::queries =
        std::vector<std::vector<typename index_type_t::data_t>>();

template <typename index_type_t>
BM_VecSimBasics_Single<index_type_t>::~BM_VecSimBasics_Single() {
    ref_count--;
    if (ref_count == 0) {
        VecSimIndex_Free(indices[VecSimAlgo_BF]);
        VecSimIndex_Free(indices[VecSimAlgo_HNSWLIB]);
    }
}

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::InitializeIndicesVector(VecSimIndex *bf_index,
                                                                   VecSimIndex *hnsw_index) {
    indices.push_back(bf_index);
    indices.push_back(hnsw_index);
}

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::InsertToQueries(std::ifstream &input) {
    for (size_t i = 0; i < BM_VecSimBasics::n_queries; i++) {
        auto query = NewQuery();
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::LoadHNSWIndex(std::string location) {
    auto *hnsw_index = CastToHNSW(indices[VecSimAlgo_HNSWLIB]);
    hnsw_index->loadIndex(location);
    size_t ef_r = 10;
    hnsw_index->setEf(ef_r);
}

// AddVector BM
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, AddVector_fp32, fp32_index_t)
(benchmark::State &st) { AddVector(st); }

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, AddVector_fp64, fp64_index_t)
(benchmark::State &st) { AddVector(st); }

// DeleteVector BM
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, DeleteVector_fp32, fp32_index_t)
(benchmark::State &st) {
    if (VecSimAlgo_BF == st.range(0)) {
        DeleteVector<BruteForceIndex<float, float>>(
            reinterpret_cast<BruteForceIndex<float, float> *>(indices[VecSimAlgo_BF]), st);
    } else if (VecSimAlgo_HNSWLIB == st.range(0)) {
        DeleteVector<HNSWIndex<float, float>>(
            reinterpret_cast<HNSWIndex<float, float> *>(indices[VecSimAlgo_HNSWLIB]), st);
    }
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, DeleteVector_fp64, fp64_index_t)
(benchmark::State &st) {
    if (VecSimAlgo_BF == st.range(0)) {
        DeleteVector<BruteForceIndex<double, double>>(
            reinterpret_cast<BruteForceIndex<double, double> *>(indices[VecSimAlgo_BF]), st);
    } else if (VecSimAlgo_HNSWLIB == st.range(0)) {
        DeleteVector<HNSWIndex<double, double>>(
            reinterpret_cast<HNSWIndex<double, double> *>(indices[VecSimAlgo_HNSWLIB]), st);
    }
}

// TopK search BM
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, TopK_BF_fp32, fp32_index_t)
(benchmark::State &st) {
    size_t k = st.range(0);
    TopK_BF(k, st);
}
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, TopK_BF_fp64, fp64_index_t)
(benchmark::State &st) {
    size_t k = st.range(0);
    TopK_BF(k, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, TopK_HNSW_fp32, fp32_index_t)
(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    TopK_HNSW(ef, k, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, TopK_HNSW_fp64, fp64_index_t)
(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    TopK_HNSW(ef, k, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, Range_BF_fp32, fp32_index_t)
(benchmark::State &st) {
    double radius = (1.0 / 100.0) * (double)st.range(0);
    Range_BF(radius, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, Range_BF_fp64, fp64_index_t)
(benchmark::State &st) {
    double radius = (1.0 / 100.0) * (double)st.range(0);
    Range_BF(radius, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, Range_HNSW_fp32, fp32_index_t)
(benchmark::State &st) {
    double radius = (1.0 / 100.0) * (double)st.range(0);
    double epsilon = (1.0 / 1000.0) * (double)st.range(1);
    Range_HNSW(radius, epsilon, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, Range_HNSW_fp64, fp64_index_t)
(benchmark::State &st) {
    double radius = (1.0 / 100.0) * (double)st.range(0);
    double epsilon = (1.0 / 1000.0) * (double)st.range(1);
    Range_HNSW(radius, epsilon, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, Memory_FLAT_fp32, fp32_index_t)
(benchmark::State &st) { Memory_FLAT(st); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, Memory_FLAT_fp64, fp64_index_t)
(benchmark::State &st) { Memory_FLAT(st); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, Memory_HNSW_fp32, fp32_index_t)
(benchmark::State &st) { Memory_HNSW(st); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, Memory_HNSW_fp64, fp64_index_t)
(benchmark::State &st) { Memory_HNSW(st); }

#define UNIT_AND_ITERATIONS                                                                        \
    Unit(benchmark::kMillisecond)->Iterations((long)BM_VecSimBasics<false>::block_size)

BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, AddVector_fp32)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)
    ->Arg(VecSimAlgo_HNSWLIB);
BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, AddVector_fp64)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)
    ->Arg(VecSimAlgo_HNSWLIB);
BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, DeleteVector_fp32)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)
    ->Arg(VecSimAlgo_HNSWLIB);
BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, DeleteVector_fp64)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)
    ->Arg(VecSimAlgo_HNSWLIB);

#define REGISTER_TopK_BF(BM_FUNC)                                                                  \
    BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, BM_FUNC)                                          \
        ->Arg(10)                                                                                  \
        ->ArgName("k")                                                                             \
        ->Arg(100)                                                                                 \
        ->ArgName("k")                                                                             \
        ->Arg(500)                                                                                 \
        ->ArgName("k")                                                                             \
        ->Unit(benchmark::kMillisecond)

REGISTER_TopK_BF(TopK_BF_fp32);
REGISTER_TopK_BF(TopK_BF_fp64);

// {ef_runtime, k} (recall that always ef_runtime >= k)
#define REGISTER_TopK_HNSW(BM_FUNC)                                                                \
    BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, BM_FUNC)                                          \
        ->HNSW_TOP_K_ARGS(10, 10)                                                                  \
        ->HNSW_TOP_K_ARGS(200, 10)                                                                 \
        ->HNSW_TOP_K_ARGS(100, 100)                                                                \
        ->HNSW_TOP_K_ARGS(200, 100)                                                                \
        ->HNSW_TOP_K_ARGS(500, 500)                                                                \
        ->Iterations(100)                                                                          \
        ->Unit(benchmark::kMillisecond)

REGISTER_TopK_BF(TopK_HNSW_fp32);
REGISTER_TopK_BF(TopK_HNSW_fp64);

// The actual radius will be the given arg divided by 100, since arg must be an integer.
#define REGISTER_Range_BF(BM_FUNC)                                                                 \
    BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, BM_FUNC)                                          \
        ->Arg(20)                                                                                  \
        ->ArgName("radiusX100")                                                                    \
        ->Arg(35)                                                                                  \
        ->ArgName("radiusX100")                                                                    \
        ->Arg(50)                                                                                  \
        ->ArgName("radiusX100")                                                                    \
        ->Unit(benchmark::kMillisecond)

REGISTER_Range_BF(Range_BF_fp32);
REGISTER_Range_BF(Range_BF_fp64);
#define HNSW_RANGE_ARGS(radius, epsilon)                                                           \
    Args({radius, epsilon})->ArgNames({"radiusX100", "epsilonX1000"})
// {radius*100, epsilon*1000}
// The actual radius will be the given arg divided by 100, and the actual epsilon values
// will be the given arg divided by 1000.
#define REGISTER_Range_HNSW(BM_FUNC)                                                               \
    BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, BM_FUNC)                                          \
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

BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, Memory_FLAT_fp32)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, Memory_FLAT_fp64)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, Memory_HNSW_fp32)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, Memory_HNSW_fp64)->Iterations(1);

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::AddVector(benchmark::State &st) {
    // Add a new vector from the test vectors in every iteration.
    size_t iter = 0;
    size_t new_id = VecSimIndex_IndexSize(indices[st.range(0)]);
    size_t memory_delta = 0;
    for (auto _ : st) {
        memory_delta +=
            VecSimIndex_AddVector(indices[st.range(0)], queries[iter % n_queries].data(), new_id++);
        iter++;
    }
    st.counters["memory"] = (double)memory_delta / (double)iter;

    // Clean-up.
    size_t new_index_size = VecSimIndex_IndexSize(indices[st.range(0)]);
    for (size_t id = n_vectors; id < new_index_size; id++) {
        VecSimIndex_DeleteVector(indices[st.range(0)], id);
    }
}

template <typename index_type_t>
template <typename algo_t>
void BM_VecSimBasics_Single<index_type_t>::DeleteVector(algo_t *index, benchmark::State &st) {
    // Remove a different vector in every execution.
    std::vector<std::vector<data_t>> blobs;
    size_t id_to_remove = 0;
    double memory_delta = 0;
    size_t iter = 0;

    for (auto _ : st) {
        st.PauseTiming();
        auto removed_vec = std::vector<data_t>(dim);
        memcpy(removed_vec.data(), index->getDataByInternalId(id_to_remove), dim * sizeof(data_t));
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
void BM_VecSimBasics_Single<index_type_t>::TopK_BF(size_t k, benchmark::State &st) {
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(indices[VecSimAlgo_BF], queries[iter % n_queries].data(), k, nullptr,
                              BY_SCORE);
        iter++;
    }
}

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::TopK_HNSW(size_t ef, size_t k, benchmark::State &st) {
    size_t correct = 0;
    size_t iter = 0;
    for (auto _ : st) {
        RunTopK_HNSW(st, ef, iter, k, correct, indices[VecSimAlgo_HNSWLIB], indices[VecSimAlgo_BF],
                     queries);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
}

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::Range_BF(double radius, benchmark::State &st) {
    size_t iter = 0;
    size_t total_res = 0;

    for (auto _ : st) {
        auto res = VecSimIndex_RangeQuery(indices[VecSimAlgo_BF], queries[iter % n_queries].data(),
                                          radius, nullptr, BY_ID);
        total_res += VecSimQueryResult_Len(res);
        iter++;
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
}

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::Range_HNSW(double radius, double epsilon,
                                                      benchmark::State &st) {
    size_t iter = 0;
    size_t total_res = 0;
    size_t total_res_bf = 0;
    auto query_params =
        VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.epsilon = epsilon}};

    for (auto _ : st) {
        auto hnsw_results =
            VecSimIndex_RangeQuery(indices[VecSimAlgo_HNSWLIB], queries[iter % n_queries].data(),
                                   radius, &query_params, BY_ID);
        st.PauseTiming();
        total_res += VecSimQueryResult_Len(hnsw_results);

        // Measure recall:
        auto bf_results = VecSimIndex_RangeQuery(
            indices[VecSimAlgo_BF], queries[iter % n_queries].data(), radius, nullptr, BY_ID);
        total_res_bf += VecSimQueryResult_Len(bf_results);

        VecSimQueryResult_Free(bf_results);
        VecSimQueryResult_Free(hnsw_results);
        iter++;
        st.ResumeTiming();
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
    st.counters["Recall"] = (float)total_res / total_res_bf;
}

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::Memory_FLAT(benchmark::State &st) {

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_Info(indices[VecSimAlgo_BF]).bfInfo.memory;
}
template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::Memory_HNSW(benchmark::State &st) {

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_Info(indices[VecSimAlgo_HNSWLIB]).hnswInfo.memory;
}

BENCHMARK_MAIN();

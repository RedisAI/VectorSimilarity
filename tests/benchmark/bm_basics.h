#pragma once

#include "bm_utils.h"
#include "bm_common.h"

std::vector<VecSimIndex *> BM_VecSimUtils::indices = std::vector<VecSimIndex *>();

size_t BM_VecSimUtils::n_queries = 10000;
size_t BM_VecSimUtils::dim = 768;
size_t BM_VecSimUtils::block_size = 1024;

const char *BM_VecSimUtils::test_vectors_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";
size_t BM_VecSimUtils::ref_count = 0;

template <typename index_type_t>
class BM_VecSimBasics : public BM_VecSimUtils {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    BM_VecSimBasics() : BM_VecSimUtils(index_type_t::get_index_type(), false){};
    ~BM_VecSimBasics() = default;

    static std::vector<std::vector<data_t>> queries;

    void TopK_BF(size_t k, benchmark::State &st, Offset_t index_offset = 0);
    void TopK_HNSW(size_t ef, size_t k, benchmark::State &st, Offset_t index_offset = 0);

    void Memory_FLAT(benchmark::State &st, Offset_t index_offset = 0);
    void Memory_HNSW(benchmark::State &st, Offset_t index_offset = 0);

    // Functions that are used by BM_VecSimUtils::Initialize()
    virtual void InsertToQueries(std::ifstream &input) override;
    virtual void LoadHNSWIndex(std::string location, Offset_t index_offset = 0) override;
    virtual inline char *GetHNSWDataByInternalId(size_t id,
                                                 Offset_t index_offset = 0) const override {
        return CastToHNSW(indices[VecSimAlgo_HNSWLIB + index_offset])->getDataByInternalId(id);
    }

protected:
    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) const {
        return reinterpret_cast<HNSWIndex<data_t, dist_t> *>(index);
    }
};

template <typename index_type_t>
std::vector<std::vector<typename index_type_t::data_t>> BM_VecSimBasics<index_type_t>::queries =
    std::vector<std::vector<typename index_type_t::data_t>>();

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::InsertToQueries(std::ifstream &input) {
    for (size_t i = 0; i < BM_VecSimUtils::n_queries; i++) {
        std::vector<data_t> query(dim);
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::LoadHNSWIndex(std::string location, Offset_t index_offset) {
    auto *hnsw_index = CastToHNSW(indices[VecSimAlgo_HNSWLIB + index_offset]);
    hnsw_index->loadIndex(location);
    size_t ef_r = 10;
    hnsw_index->setEf(ef_r);
}

// TopK search BM
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, TopK_BF_fp32, fp32_index_t)
(benchmark::State &st) {
    size_t k = st.range(0);
    TopK_BF(k, st);
}
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, TopK_BF_fp64, fp64_index_t)
(benchmark::State &st) {
    size_t k = st.range(0);
    TopK_BF(k, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, TopK_HNSW_fp32, fp32_index_t)
(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    TopK_HNSW(ef, k, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, TopK_HNSW_fp64, fp64_index_t)
(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    TopK_HNSW(ef, k, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Memory_FLAT_fp32, fp32_index_t)
(benchmark::State &st) { Memory_FLAT(st); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Memory_FLAT_fp64, fp64_index_t)
(benchmark::State &st) { Memory_FLAT(st); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Memory_HNSW_fp32, fp32_index_t)
(benchmark::State &st) { Memory_HNSW(st); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Memory_HNSW_fp64, fp64_index_t)
(benchmark::State &st) { Memory_HNSW(st); }

#define REGISTER_TopK_BF(BM_CLASS, BM_FUNC)                                                        \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Arg(10)                                                                                  \
        ->ArgName("k")                                                                             \
        ->Arg(100)                                                                                 \
        ->ArgName("k")                                                                             \
        ->Arg(500)                                                                                 \
        ->ArgName("k")                                                                             \
        ->Unit(benchmark::kMillisecond)

REGISTER_TopK_BF(BM_VecSimBasics, TopK_BF_fp32);
REGISTER_TopK_BF(BM_VecSimBasics, TopK_BF_fp64);

// {ef_runtime, k} (recall that always ef_runtime >= k)
#define REGISTER_TopK_HNSW(BM_CLASS, BM_FUNC)                                                      \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->HNSW_TOP_K_ARGS(10, 10)                                                                  \
        ->HNSW_TOP_K_ARGS(200, 10)                                                                 \
        ->HNSW_TOP_K_ARGS(100, 100)                                                                \
        ->HNSW_TOP_K_ARGS(200, 100)                                                                \
        ->HNSW_TOP_K_ARGS(500, 500)                                                                \
        ->Iterations(100)                                                                          \
        ->Unit(benchmark::kMillisecond)

REGISTER_TopK_HNSW(BM_VecSimBasics, TopK_HNSW_fp32);
REGISTER_TopK_HNSW(BM_VecSimBasics, TopK_HNSW_fp64);

BENCHMARK_REGISTER_F(BM_VecSimBasics, Memory_FLAT_fp32)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimBasics, Memory_FLAT_fp64)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimBasics, Memory_HNSW_fp32)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimBasics, Memory_HNSW_fp64)->Iterations(1);

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::TopK_BF(size_t k, benchmark::State &st, Offset_t index_offset) {
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(indices[VecSimAlgo_BF + index_offset],
                              queries[iter % n_queries].data(), k, nullptr, BY_SCORE);
        iter++;
    }
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::TopK_HNSW(size_t ef, size_t k, benchmark::State &st,
                                              Offset_t index_offset) {
    size_t correct = 0;
    size_t iter = 0;
    for (auto _ : st) {
        RunTopK_HNSW(st, ef, iter, k, correct, queries, index_offset);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::Memory_FLAT(benchmark::State &st, Offset_t index_offset) {

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] =
        (double)VecSimIndex_Info(indices[VecSimAlgo_BF + index_offset]).bfInfo.memory;
}
template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::Memory_HNSW(benchmark::State &st, Offset_t index_offset) {

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] =
        (double)VecSimIndex_Info(indices[VecSimAlgo_HNSWLIB + index_offset]).hnswInfo.memory;
}

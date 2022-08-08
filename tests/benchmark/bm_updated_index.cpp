#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/algorithms/hnsw/serialization.h"
#include "bm_utils.h"

// Global benchmark data
size_t BM_VecSimBasics::n_vectors = 500000;
size_t BM_VecSimBasics::n_queries = 10000;
size_t BM_VecSimBasics::dim = 768;
VecSimIndex *BM_VecSimBasics::bf_index;
VecSimIndex *BM_VecSimBasics::hnsw_index;
std::vector<std::vector<float>> *BM_VecSimBasics::queries;
size_t HNSW_M = 65;
size_t HNSW_EF_C = 512;
const char *hnsw_index_file = "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512.hnsw";
const char *updated_hnsw_index_file =
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512-updated.hnsw";
const char *test_vectors_file = "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

size_t BM_VecSimBasics::ref_count = 0;

class BM_VecSimUpdatedIndex : public BM_VecSimBasics {
protected:
    static VecSimIndex *bf_index_updated;
    static VecSimIndex *hnsw_index_updated;

    BM_VecSimUpdatedIndex() : BM_VecSimBasics() {
        if (ref_count == 1) {
            // Initialize the updated indexes as well, if this is the first instance.
            Initialize(HNSW_M, HNSW_EF_C, updated_hnsw_index_file);
        }
    }
    static void Initialize(size_t M_, size_t ef_c_, const char *updated_hnsw_index_path) {

        VecSimParams bf_params = {.algo = VecSimAlgo_BF,
                                  .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                                       .dim = dim,
                                                       .metric = VecSimMetric_Cosine,
                                                       .initialCapacity = n_vectors}};
        bf_index_updated = VecSimIndex_New(&bf_params);

        // Initially, load all the vectors to the updated bf index (before we override it).
        for (size_t i = 0; i < n_vectors; ++i) {
            char *blob =
                reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex()->getDataByInternalId(i);
            size_t label =
                reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex()->getExternalLabel(i);
            VecSimIndex_AddVector(bf_index_updated, blob, label);
        }

        // Initialize and populate an HNSW index where all vectors have been updated.
        VecSimParams params = {.algo = VecSimAlgo_HNSWLIB,
                               .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                        .dim = dim,
                                                        .metric = VecSimMetric_Cosine,
                                                        .initialCapacity = n_vectors,
                                                        .M = M_,
                                                        .efConstruction = ef_c_}};
        hnsw_index_updated = VecSimIndex_New(&params);
        load_HNSW_index(updated_hnsw_index_path, hnsw_index_updated);

        // Add the same vectors to the *updated* FLAT index (override the previous vectors).
        for (size_t i = 0; i < n_vectors; ++i) {
            char *blob = reinterpret_cast<HNSWIndex *>(hnsw_index_updated)
                             ->getHNSWIndex()
                             ->getDataByInternalId(i);
            size_t label = reinterpret_cast<HNSWIndex *>(hnsw_index_updated)
                               ->getHNSWIndex()
                               ->getExternalLabel(i);
            VecSimIndex_AddVector(bf_index_updated, blob, label);
        }
    }

public:
    ~BM_VecSimUpdatedIndex() {
        if (ref_count == 1) {
            VecSimIndex_Free(hnsw_index_updated);
            VecSimIndex_Free(bf_index_updated);
        }
    }
};

VecSimIndex *BM_VecSimUpdatedIndex::bf_index_updated;
VecSimIndex *BM_VecSimUpdatedIndex::hnsw_index_updated;

BENCHMARK_DEFINE_F(BM_VecSimUpdatedIndex, TopK_BF)(benchmark::State &st) {
    size_t k = st.range(0);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(bf_index, (*queries)[iter % n_queries].data(), k, nullptr, BY_SCORE);
        iter++;
    }
}

BENCHMARK_DEFINE_F(BM_VecSimUpdatedIndex, TopK_BF_Updated)(benchmark::State &st) {
    size_t k = st.range(0);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(bf_index_updated, (*queries)[iter % n_queries].data(), k, nullptr,
                              BY_SCORE);
        iter++;
    }
}

BENCHMARK_DEFINE_F(BM_VecSimUpdatedIndex, TopK_HNSW)(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    size_t correct = 0;
    size_t iter = 0;
    for (auto _ : st) {
        RunTopK_HNSW(st, ef, iter, k, correct, hnsw_index, bf_index);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
}

BENCHMARK_DEFINE_F(BM_VecSimUpdatedIndex, TopK_HNSW_Updated)(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    size_t correct = 0;
    size_t iter = 0;
    for (auto _ : st) {
        RunTopK_HNSW(st, ef, iter, k, correct, hnsw_index_updated, bf_index_updated);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
}

BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, TopK_BF)
    ->Arg(10)
    ->ArgName("k")
    ->Arg(100)
    ->ArgName("k")
    ->Arg(500)
    ->ArgName("k")
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, TopK_HNSW)
    // {ef_runtime, k} (recall that always ef_runtime >= k)
    ->Args({10, 10})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({200, 10})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({100, 100})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({200, 100})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({500, 500})
    ->ArgNames({"ef_runtime", "k"})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, TopK_BF_Updated)
    ->Arg(10)
    ->ArgName("k")
    ->Arg(100)
    ->ArgName("k")
    ->Arg(500)
    ->ArgName("k")
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, TopK_HNSW_Updated)
    // {ef_runtime, k} (recall that always ef_runtime >= k)
    ->Args({10, 10})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({200, 10})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({100, 100})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({200, 100})
    ->ArgNames({"ef_runtime", "k"})
    ->Args({500, 500})
    ->ArgNames({"ef_runtime", "k"})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

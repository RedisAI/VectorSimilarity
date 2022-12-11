/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/serializer.h"
#include "bm_utils.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"

// Global benchmark data
size_t BM_VecSimBasics::n_vectors = 500000;
size_t BM_VecSimBasics::n_queries = 10000;
size_t BM_VecSimBasics::dim = 768;
VecSimIndex *BM_VecSimBasics::hnsw_index;
VecSimIndex *BM_VecSimBasics::bf_index;
std::vector<std::vector<float>> BM_VecSimBasics::queries;
size_t BM_VecSimBasics::M = 65;
size_t BM_VecSimBasics::EF_C = 512;
size_t BM_VecSimBasics::block_size = 1024;
const char *BM_VecSimBasics::hnsw_index_file =
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512.hnsw";
const char *BM_VecSimBasics::test_vectors_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

size_t BM_VecSimBasics::ref_count = 0;

const char *updated_hnsw_index_file =
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512-updated.hnsw";

class BM_VecSimUpdatedIndex : public BM_VecSimBasics {
protected:
    static VecSimIndex *bf_index_updated;
    static VecSimIndex *hnsw_index_updated;

    BM_VecSimUpdatedIndex() : BM_VecSimBasics() {
        if (ref_count == 1) {
            // Initialize the updated indexes as well, if this is the first instance.
            Initialize();
        }
    }
    static void Initialize() {

        VecSimParams bf_params = {.algo = VecSimAlgo_BF,
                                  .bfParams =
                                      BFParams{.type = VecSimType_FLOAT32,
                                               .dim = BM_VecSimBasics::dim,
                                               .metric = VecSimMetric_Cosine,
                                               .initialCapacity = BM_VecSimBasics::n_vectors}};
        BM_VecSimUpdatedIndex::bf_index_updated = VecSimIndex_New(&bf_params);

        auto hnsw_index_casted = reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index);
        // Initially, load all the vectors to the updated bf index (before we override it).
        for (size_t i = 0; i < BM_VecSimBasics::n_vectors; ++i) {
            const char *blob = hnsw_index_casted->getDataByInternalId(i);
            size_t label = hnsw_index_casted->getExternalLabel(i);
            VecSimIndex_AddVector(bf_index_updated, blob, label);
        }

        // Generate the updated index from file.
        // HNSWParams is required to load v1 index
        HNSWParams params = {.type = VecSimType_FLOAT32,
                             .dim = BM_VecSimBasics::dim,
                             .metric = VecSimMetric_Cosine,
                             .multi = false,
                             .blockSize = BM_VecSimBasics::block_size};

        // Generate index from file.
        hnsw_index_updated = HNSWFactory::NewIndex(
            BM_VecSimBasics::GetSerializedIndexLocation(updated_hnsw_index_file), &params);

        auto hnsw_index_updated_casted =
            reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index_updated);

        if (!hnsw_index_updated_casted->checkIntegrity().valid_state) {
            throw std::runtime_error("The loaded HNSW index is corrupted. Exiting...");
        }
        // Add the same vectors to the *updated* FLAT index (override the previous vectors).
        for (size_t i = 0; i < n_vectors; ++i) {
            const char *blob = hnsw_index_updated_casted->getDataByInternalId(i);
            size_t label = hnsw_index_updated_casted->getExternalLabel(i);
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
        VecSimIndex_TopKQuery(bf_index, queries[iter % n_queries].data(), k, nullptr, BY_SCORE);
        iter++;
    }
}

BENCHMARK_DEFINE_F(BM_VecSimUpdatedIndex, TopK_BF_Updated)(benchmark::State &st) {
    size_t k = st.range(0);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(bf_index_updated, queries[iter % n_queries].data(), k, nullptr,
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

// Index memory metrics - run only once.
BENCHMARK_DEFINE_F(BM_VecSimUpdatedIndex, Memory_FLAT_before)(benchmark::State &st) {
    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_Info(bf_index).bfInfo.memory;
}

BENCHMARK_DEFINE_F(BM_VecSimUpdatedIndex, Memory_FLAT_updated)(benchmark::State &st) {
    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_Info(bf_index_updated).bfInfo.memory;
}

BENCHMARK_DEFINE_F(BM_VecSimUpdatedIndex, Memory_HNSW_before)(benchmark::State &st) {
    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_Info(hnsw_index).hnswInfo.memory;
}

BENCHMARK_DEFINE_F(BM_VecSimUpdatedIndex, Memory_HNSW_updated)(benchmark::State &st) {
    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_Info(hnsw_index_updated).hnswInfo.memory;
}

BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, Memory_FLAT_before)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, Memory_FLAT_updated)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, Memory_HNSW_before)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, Memory_FLAT_updated)->Iterations(1);

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
HNSW_TOP_K_ARGS(10, 10)
HNSW_TOP_K_ARGS(200, 10)
HNSW_TOP_K_ARGS(100, 100)
HNSW_TOP_K_ARGS(200, 100)
HNSW_TOP_K_ARGS(500, 500)->Iterations(100)->Unit(benchmark::kMillisecond);

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
HNSW_TOP_K_ARGS(10, 10)
HNSW_TOP_K_ARGS(200, 10)
HNSW_TOP_K_ARGS(100, 100)
HNSW_TOP_K_ARGS(200, 100)
HNSW_TOP_K_ARGS(500, 500)->Iterations(100)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

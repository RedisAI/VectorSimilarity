#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/hnsw/serialization.h"

static void GetHNSWIndex(VecSimIndex *hnsw_index, const char *file_name) {

    // Load the index file, if it exists in the expected path.
    auto location = std::string(getenv("ROOT"));
    auto full_file_name = location + "/tests/benchmark/data/" + std::string(file_name);
    // auto full_file_name = "/home/alon/Code/VectorSimilarity/tests/benchmark/data/" +
    // std::string(file_name);
    auto serializer =
        hnswlib::HNSWIndexSerializer(reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex());
    std::ifstream input(full_file_name, std::ios::binary);
    if (input.is_open()) {
        serializer.loadIndex(full_file_name,
                             reinterpret_cast<HNSWIndex *>(hnsw_index)->getSpace().get());
        if (!serializer.checkIntegrity().valid_state) {
            throw std::runtime_error("The loaded HNSW index is corrupted. Exiting...");
        }
    } else {
        throw std::runtime_error("HNSW index file was not found in path. Exiting...");
    }
}

class BM_VecSimUpdatedIndex : public benchmark::Fixture {
protected:
    VecSimIndex *bf_index;
    VecSimIndex *hnsw_index;
    VecSimIndex *bf_index_updated;
    VecSimIndex *hnsw_index_updated;
    size_t dim;
    size_t n_vectors;
    std::vector<std::vector<float>> *queries;
    size_t n_queries;

    // We use this class as a singleton for every test case, so we won't hold several indices (to
    // reduce memory consumption).
    static BM_VecSimUpdatedIndex *instance;
    static size_t ref_count;

    BM_VecSimUpdatedIndex() {
        dim = 768;
        n_vectors = 500000;
        n_queries = 10000;
        ref_count++;
        if (instance != nullptr) {
            // Use the same indices and query vectors for every instance.
            queries = instance->queries;
            bf_index = instance->bf_index;
            hnsw_index = instance->hnsw_index;
            bf_index_updated = instance->bf_index_updated;
            hnsw_index_updated = instance->hnsw_index_updated;
        } else {
            // Initialize and load HNSW index for the first half of DBPedia data set.
            size_t M = 65;
            size_t ef_c = 512;
            VecSimParams params = {.algo = VecSimAlgo_HNSWLIB,
                                   .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                            .dim = dim,
                                                            .metric = VecSimMetric_Cosine,
                                                            .initialCapacity = n_vectors,
                                                            .M = M,
                                                            .efConstruction = ef_c}};

            // Load pre-generated HNSW index.
            hnsw_index = VecSimIndex_New(&params);
            GetHNSWIndex(hnsw_index, "DBpedia-n500K-cosine-d768-M65-EFC512.hnsw");
            size_t ef_r = 10;
            reinterpret_cast<HNSWIndex *>(hnsw_index)->setEf(ef_r);

            // Create two FLAT indexes.
            VecSimParams bf_params = {.algo = VecSimAlgo_BF,
                                      .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                                           .dim = dim,
                                                           .metric = VecSimMetric_Cosine,
                                                           .initialCapacity = n_vectors}};
            bf_index = VecSimIndex_New(&bf_params);
            bf_index_updated = VecSimIndex_New(&bf_params);

            // Add the same vector to the FLAT indexes.
            for (size_t i = 0; i < n_vectors; ++i) {
                char *blob = reinterpret_cast<HNSWIndex *>(hnsw_index)
                                 ->getHNSWIndex()
                                 ->getDataByInternalId(i);
                size_t label =
                    reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex()->getExternalLabel(i);
                VecSimIndex_AddVector(bf_index, blob, label);
                VecSimIndex_AddVector(bf_index_updated, blob, label);
            }

            // Load pre-generated *updated* HNSW index.
            hnsw_index_updated = VecSimIndex_New(&params);
            GetHNSWIndex(hnsw_index_updated, "DBpedia-n500K-cosine-d768-M65-EFC512-updated.hnsw");
            reinterpret_cast<HNSWIndex *>(hnsw_index_updated)->setEf(ef_r);

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

            // Load the test query vectors.
            queries = new std::vector<std::vector<float>>(n_queries);
            auto location = std::string(std::string(getenv("ROOT")));
            auto file_name = location + "/tests/benchmark/data/DBpedia-test_vectors-n10k.raw";
            // auto file_name =
            // "/home/alon/Code/VectorSimilarity/tests/benchmark/data/DBpedia-test_vectors-n10k.raw";
            std::ifstream input(file_name, std::ios::binary);
            input.seekg(0, std::ifstream::beg);
            for (size_t i = 0; i < n_queries; i++) {
                std::vector<float> query(dim);
                input.read((char *)query.data(), dim * sizeof(float));
                (*queries)[i] = query;
            }

            instance = this;
        }
    }

public:
    ~BM_VecSimUpdatedIndex() {
        ref_count--;
        if (ref_count == 0) {
            VecSimIndex_Free(hnsw_index);
            VecSimIndex_Free(hnsw_index_updated);
            VecSimIndex_Free(bf_index);
            VecSimIndex_Free(bf_index_updated);
            delete queries;
        }
    }
};

size_t BM_VecSimUpdatedIndex::ref_count = 0;
BM_VecSimUpdatedIndex *BM_VecSimUpdatedIndex::instance = nullptr;

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
    size_t correct = 0.0f;
    size_t iter = 0;
    for (auto _ : st) {
        auto query_params =
            VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.efRuntime = ef}};
        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, (*queries)[iter % n_queries].data(),
                                                  k, &query_params, BY_SCORE);
        st.PauseTiming();

        // Measure recall:
        auto bf_results = VecSimIndex_TopKQuery(bf_index, (*queries)[iter % n_queries].data(), k,
                                                nullptr, BY_SCORE);
        auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
        while (VecSimQueryResult_IteratorHasNext(hnsw_it)) {
            auto hnsw_res_item = VecSimQueryResult_IteratorNext(hnsw_it);
            auto bf_it = VecSimQueryResult_List_GetIterator(bf_results);
            while (VecSimQueryResult_IteratorHasNext(bf_it)) {
                auto bf_res_item = VecSimQueryResult_IteratorNext(bf_it);
                if (VecSimQueryResult_GetId(hnsw_res_item) ==
                    VecSimQueryResult_GetId(bf_res_item)) {
                    correct++;
                    break;
                }
            }
            VecSimQueryResult_IteratorFree(bf_it);
        }
        VecSimQueryResult_IteratorFree(hnsw_it);

        VecSimQueryResult_Free(bf_results);
        VecSimQueryResult_Free(hnsw_results);
        iter++;
        st.ResumeTiming();
    }
    st.counters["Recall"] = (float)correct / (k * iter);
}

BENCHMARK_DEFINE_F(BM_VecSimUpdatedIndex, TopK_HNSW_Updated)(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    size_t correct = 0.0f;
    size_t iter = 0;
    for (auto _ : st) {
        auto query_params =
            VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.efRuntime = ef}};
        auto hnsw_results = VecSimIndex_TopKQuery(
            hnsw_index_updated, (*queries)[iter % n_queries].data(), k, &query_params, BY_SCORE);
        st.PauseTiming();

        // Measure recall:
        auto bf_results = VecSimIndex_TopKQuery(
            bf_index_updated, (*queries)[iter % n_queries].data(), k, nullptr, BY_SCORE);
        auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
        while (VecSimQueryResult_IteratorHasNext(hnsw_it)) {
            auto hnsw_res_item = VecSimQueryResult_IteratorNext(hnsw_it);
            auto bf_it = VecSimQueryResult_List_GetIterator(bf_results);
            while (VecSimQueryResult_IteratorHasNext(bf_it)) {
                auto bf_res_item = VecSimQueryResult_IteratorNext(bf_it);
                if (VecSimQueryResult_GetId(hnsw_res_item) ==
                    VecSimQueryResult_GetId(bf_res_item)) {
                    correct++;
                    break;
                }
            }
            VecSimQueryResult_IteratorFree(bf_it);
        }
        VecSimQueryResult_IteratorFree(hnsw_it);

        VecSimQueryResult_Free(bf_results);
        VecSimQueryResult_Free(hnsw_results);
        iter++;
        st.ResumeTiming();
    }
    st.counters["Recall"] = (float)correct / (k * iter);
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

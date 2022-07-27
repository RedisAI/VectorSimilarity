#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/hnsw/serialization.h"

static void GetHNSWIndex(VecSimIndex *hnsw_index) {

    // Load the index file, if it exists in the expected path.
    auto location = std::string(std::string(getenv("ROOT")));
    auto file_name = location + "/tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1";
    auto serializer =
        hnswlib::HNSWIndexSerializer(reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex());
    std::ifstream input(file_name, std::ios::binary);
    if (input.is_open()) {
        serializer.loadIndex(file_name,
                             reinterpret_cast<HNSWIndex *>(hnsw_index)->getSpace().get());
        if (!serializer.checkIntegrity().valid_state) {
            throw std::runtime_error("The loaded HNSW index is corrupted. Exiting...");
        }
    } else {
        throw std::runtime_error("HNSW index file was not found in path. Exiting...");
    }
}

static void GetTestVectors(std::vector<std::vector<float>> &queries, size_t n_queries, size_t dim) {
    auto location = std::string(std::string(getenv("ROOT")));
    auto file_name = location + "/tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

    std::ifstream input(file_name, std::ios::binary);

    queries.reserve(n_queries);
    if (input.is_open()) {
        input.seekg(0, std::ifstream::beg);
        for (size_t i = 0; i < n_queries; i++) {
            std::vector<float> query(dim);
            input.read((char *)query.data(), dim * sizeof(float));
            queries[i] = query;
        }
    } else {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
}

class BM_VecSimBasics : public benchmark::Fixture {
protected:
    VecSimIndex *bf_index;
    VecSimIndex *hnsw_index;
    size_t dim;
    size_t n_vectors;
    std::vector<std::vector<float>> *queries;
    size_t n_queries;

    // We use this class as a singleton for every test case, so we won't hold several indices (to
    // reduce memory consumption).
    static BM_VecSimBasics *instance;
    static size_t ref_count;

    BM_VecSimBasics() {
        dim = 768;
        n_vectors = 1000000;
        n_queries = 10000;
        ref_count++;
        if (instance != nullptr) {
            // Use the same indices and query vectors for every instance.
            queries = instance->queries;
            bf_index = instance->bf_index;
            hnsw_index = instance->hnsw_index;
        } else {
            // Initialize and load HNSW index for DBPedia data set.
            size_t M = 64;
            size_t ef_c = 512;
            VecSimParams params = {.algo = VecSimAlgo_HNSWLIB,
                                   .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                            .dim = dim,
                                                            .metric = VecSimMetric_Cosine,
                                                            .initialCapacity = n_vectors,
                                                            .M = M,
                                                            .efConstruction = ef_c}};
            hnsw_index = VecSimIndex_New(&params);

            // Load pre-generated HNSW index.
            GetHNSWIndex(hnsw_index);
            size_t ef_r = 10;
            reinterpret_cast<HNSWIndex *>(hnsw_index)->setEf(ef_r);

            VecSimParams bf_params = {.algo = VecSimAlgo_BF,
                                      .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                                           .dim = dim,
                                                           .metric = VecSimMetric_Cosine,
                                                           .initialCapacity = n_vectors}};
            bf_index = VecSimIndex_New(&bf_params);

            // Add the same vectors to Flat index.
            for (size_t i = 0; i < n_vectors; ++i) {
                char *blob = reinterpret_cast<HNSWIndex *>(hnsw_index)
                                 ->getHNSWIndex()
                                 ->getDataByInternalId(i);
                VecSimIndex_AddVector(bf_index, blob, i);
            }
            // Load the test query vectors form file.
            queries = new std::vector<std::vector<float>>;
            GetTestVectors(*queries, n_queries, dim);
            instance = this;
        }
    }

public:
    ~BM_VecSimBasics() {
        ref_count--;
        if (ref_count == 0) {
            VecSimIndex_Free(hnsw_index);
            VecSimIndex_Free(bf_index);
            delete queries;
        }
    }
};

size_t BM_VecSimBasics::ref_count = 0;
BM_VecSimBasics *BM_VecSimBasics::instance = nullptr;

BENCHMARK_DEFINE_F(BM_VecSimBasics, AddVectorHNSW)(benchmark::State &st) {
    // Add a new vector from the test vectors in every iteration.
    size_t iter = 0;
    size_t new_id = VecSimIndex_IndexSize(hnsw_index);
    for (auto _ : st) {
        VecSimIndex_AddVector(hnsw_index, (*queries)[(iter % n_queries)].data(), new_id++);
        iter++;
    }
    // Clean-up.
    size_t new_index_size = VecSimIndex_IndexSize(hnsw_index);
    for (size_t id = n_vectors; id < new_index_size; id++) {
        VecSimIndex_DeleteVector(hnsw_index, id);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, DeleteVectorHNSW)(benchmark::State &st) {
    // Remove a different vector in every execution.
    std::vector<std::vector<float>> blobs;
    size_t id_to_remove = 0;

    for (auto _ : st) {
        st.PauseTiming();
        auto removed_vec = std::vector<float>(dim);
        memcpy(removed_vec.data(),
               reinterpret_cast<HNSWIndex *>(hnsw_index)
                   ->getHNSWIndex()
                   ->getDataByInternalId(id_to_remove),
               dim * sizeof(float));
        blobs.push_back(removed_vec);
        st.ResumeTiming();
        VecSimIndex_DeleteVector(hnsw_index, id_to_remove++);
    }

    // Restore index state.
    for (size_t i = 0; i < blobs.size(); i++) {
        VecSimIndex_AddVector(hnsw_index, blobs[i].data(), i);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, TopK_BF)(benchmark::State &st) {
    size_t k = st.range(0);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(bf_index, (*queries)[iter % n_queries].data(), k, nullptr, BY_SCORE);
        iter++;
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, TopK_HNSW)(benchmark::State &st) {
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
        if (iter == queries->size())
            break;
        st.ResumeTiming();
    }
    st.counters["Recall"] = (float)correct / (k * iter);
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, Range_BF)(benchmark::State &st) {
    float radius = (1.0f / 100.0f) * (float)st.range(0);
    size_t iter = 0;
    size_t total_res = 0;

    for (auto _ : st) {
        auto res = VecSimIndex_RangeQuery(bf_index, (*queries)[iter % n_queries].data(), radius,
                                          nullptr, BY_ID);
        total_res += VecSimQueryResult_Len(res);
        iter++;
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
}

// Register the function as a benchmark
BENCHMARK_REGISTER_F(BM_VecSimBasics, Range_BF)
    // The actual radius will be the given arg divided by 100, since arg must be an integer.
    ->Arg(20)
    ->Arg(35)
    ->Arg(50)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, AddVectorHNSW)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, DeleteVectorHNSW)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, TopK_BF)
    ->Arg(10)
    ->Arg(100)
    ->Arg(500)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, TopK_HNSW)
    // {ef_runtime, k} (recall that always ef_runtime >= k)
    ->Args({10, 10})
    ->Args({200, 10})
    ->Args({100, 100})
    ->Args({200, 100})
    ->Args({500, 500})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

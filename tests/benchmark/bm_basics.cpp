#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/hnsw/serialization.h"

static void GetHNSWIndex(VecSimIndex *hnsw_index) {

    // Load the index file, if it exists in the expected path.
    char *location = getcwd(nullptr, 0);
    auto file_name =
        std::string(location) + "/tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1";
    auto serializer =
        hnswlib::HNSWIndexSerializer(reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex());
    std::ifstream input(file_name, std::ios::binary);
    if (input.is_open()) {
        serializer.loadIndex(file_name,
                             reinterpret_cast<HNSWIndex *>(hnsw_index)->getSpace().get());
        if (!serializer.checkIntegrity().valid_state) {
            std::cerr << "The loaded HNSW index is corrupted. Exiting..." << std::endl;
            exit(1);
        }
    } else {
        std::cerr << "HNSW index file was not found in path. Exiting..." << std::endl;
        exit(1);
    }
    free(location);
}

class BM_VecSimBasics : public benchmark::Fixture {
protected:
    std::mt19937 rng;
    VecSimIndex *bf_index;
    VecSimIndex *hnsw_index;
    size_t dim;
    size_t n_vectors;
    std::vector<float> query;

    BM_VecSimBasics() {
        dim = 768;
        n_vectors = 1000000;
        query.reserve(dim);
        rng.seed(47);
    }

public:
    void SetUp(const ::benchmark::State &state) {

        // Initialize and load HNSW index for DBPedia data set.
        size_t M = 64;
        size_t ef_c = 512;
        VecSimParams params = {.algo = VecSimAlgo_HNSWLIB,
                               .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                        .dim = dim,
                                                        .metric = VecSimMetric_Cosine,
                                                        .initialCapacity = n_vectors + 1,
                                                        .M = M,
                                                        .efConstruction = ef_c}};
        hnsw_index = VecSimIndex_New(&params);

        // Load pre-generated HNSW index.
        GetHNSWIndex(hnsw_index);
        size_t ef_r = 500;
        reinterpret_cast<HNSWIndex *>(hnsw_index)->setEf(ef_r);

        VecSimParams bf_params = {.algo = VecSimAlgo_BF,
                                  .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                                       .dim = dim,
                                                       .metric = VecSimMetric_Cosine,
                                                       .initialCapacity = n_vectors}};
        bf_index = VecSimIndex_New(&bf_params);

        // Add the same vectors to Flat index.
        for (size_t i = 0; i < n_vectors; ++i) {
            char *blob =
                reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex()->getDataByInternalId(i);
            VecSimIndex_AddVector(bf_index, blob, i);
        }
    }

    void TearDown(const ::benchmark::State &state) {
        VecSimIndex_Free(hnsw_index);
        VecSimIndex_Free(bf_index);
    }

    ~BM_VecSimBasics() {}
};

BENCHMARK_DEFINE_F(BM_VecSimBasics, AddVectorHNSW)(benchmark::State &st) {
    // Save the first 500 vector, remove then from the index, and add a different vector in every
    // execution.
    size_t n_vectors_to_add = 500;
    float *blobs[n_vectors_to_add];
    for (size_t i = 0; i < n_vectors_to_add; i++) {
        blobs[i] = new float[dim];
        memcpy(blobs[i],
               reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex()->getDataByInternalId(i),
               dim * sizeof(float));
        VecSimIndex_DeleteVector(hnsw_index, i);
    }
    for (auto _ : st) {
        size_t id = n_vectors - VecSimIndex_IndexSize(hnsw_index) - 1;
        VecSimIndex_AddVector(hnsw_index, query.data(), id);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, DeleteVectorHNSW)(benchmark::State &st) {
    // Remove a different vector in every execution.
    for (auto _ : st) {
        size_t id = n_vectors - VecSimIndex_IndexSize(hnsw_index);
        VecSimIndex_DeleteVector(hnsw_index, id);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, TopK_BF)(benchmark::State &st) {
    size_t k = st.range(0);
    for (auto _ : st) {
        st.PauseTiming();
        // Generate random query vector before test.
        std::uniform_real_distribution<double> distrib(-1.0, 1.0);
        for (size_t i = 0; i < dim; ++i) {
            query[i] = (float)distrib(rng);
        }
        st.ResumeTiming();
        VecSimIndex_TopKQuery(bf_index, query.data(), k, nullptr, BY_SCORE);
    }
}

BENCHMARK_DEFINE_F(BM_VecSimBasics, TopK_HNSW)(benchmark::State &st) {
    size_t k = st.range(0);
    size_t correct = 0.0f;
    size_t iter = 0;
    for (auto _ : st) {
        st.PauseTiming();
        // Generate random query vector before test.
        std::uniform_real_distribution<double> distrib(-1.0, 1.0);
        for (size_t i = 0; i < dim; ++i) {
            query[i] = (float)distrib(rng);
        }
	    st.ResumeTiming();
	    auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, query.data(), k, nullptr, BY_SCORE);
		st.PauseTiming();

        // Measure recall:
	    auto bf_results = VecSimIndex_TopKQuery(bf_index, query.data(), k, nullptr, BY_SCORE);
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

BENCHMARK_DEFINE_F(BM_VecSimBasics, Range_BF)(benchmark::State &st) {
    float radius = (1.0f / 100.0f) * (float)st.range(0);
    size_t iter = 0;
    size_t total_res = 0;
    for (auto _ : st) {
        st.PauseTiming();
        // Generate random query vector before test.
	    std::uniform_real_distribution<double> distrib(-1.0, 1.0);
	    for (size_t i = 0; i < dim; ++i) {
            query[i] = (float)distrib(rng);
        }
	    st.ResumeTiming();
	    auto res = VecSimIndex_RangeQuery(bf_index, query.data(), radius, nullptr, BY_ID);
	    st.PauseTiming();
	    total_res += VecSimQueryResult_Len(res);
        iter++;
	    st.ResumeTiming();
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
}

// Register the function as a benchmark
BENCHMARK_REGISTER_F(BM_VecSimBasics, Range_BF)
	// The actual radius will the given arg divided by 100, since arg must me and integer.
    ->Arg(92)
    ->Arg(95)
    ->Arg(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, AddVectorHNSW)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, DeleteVectorHNSW)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, TopK_BF)
    ->Arg(10)
    ->Arg(100)
    ->Arg(500)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_VecSimBasics, TopK_HNSW)
    ->Arg(10)
    ->Arg(100)
    ->Arg(500)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

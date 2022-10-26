/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include <istream>
#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_interface.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "bm_common.h"

class BM_VecSimUtils : public benchmark::Fixture {
public:
    static size_t dim;
    static size_t n_vectors;
    static size_t block_size;

    static const char *test_vectors_file;
    static size_t n_queries;

    // We use this class as a singleton for every test case, so we won't hold several indices (to
    // reduce memory consumption).
    static size_t ref_count;

    static std::vector<VecSimIndex *> indices;

    BM_VecSimUtils(VecSimType type, bool is_multi);
    virtual ~BM_VecSimUtils();

    template <typename data_t>
    static void RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                             size_t &correct, const std::vector<std::vector<data_t>> &queries,
                             Offset_t index_offset = 0);

    static const char *hnsw_index_file;
    static inline std::string AttachRootPath(const char *file_name) {
        return std::string(getenv("ROOT")) + "/" + file_name;
    }

    template <typename IndexParams>
    static inline VecSimIndex *CreateNewIndex(IndexParams &index_params) {
        VecSimParams params = CreateParams(index_params);
        return VecSimIndex_New(&params);
    }

private:
    void Initialize(VecSimType type, bool is_multi);

    static inline VecSimParams CreateParams(const HNSWParams &hnsw_params) {
        VecSimParams params{.algo = VecSimAlgo_HNSWLIB, .hnswParams = hnsw_params};
        return params;
    }

    static inline VecSimParams CreateParams(const BFParams &bf_params) {
        VecSimParams params{.algo = VecSimAlgo_BF, .bfParams = bf_params};
        return params;
    }

    void loadTestVectors();

protected:
    virtual void InsertToQueries(std::ifstream &input) = 0;
    virtual void LoadHNSWIndex(std::string location, Offset_t index_offset = 0) = 0;
    virtual char *GetHNSWDataByInternalId(size_t id, Offset_t index_offset = 0) const = 0;
};

BM_VecSimUtils::~BM_VecSimUtils() {
    ref_count--;
    if (ref_count == 0) {
        VecSimIndex_Free(indices[VecSimAlgo_BF]);
        VecSimIndex_Free(indices[VecSimAlgo_HNSWLIB]);
    }
}

BM_VecSimUtils::BM_VecSimUtils(VecSimType type, bool is_multi) {
    if (ref_count == 0) {
        // Initialize the static members.
        Initialize(type, is_multi);
    }
    ref_count++;
}

void BM_VecSimUtils::Initialize(VecSimType type, bool is_multi) {

    // Initialize and load HNSW index for DBPedia data set.
    HNSWParams hnsw_params = {.type = type,
                              .dim = dim,
                              .metric = VecSimMetric_Cosine,
                              .multi = is_multi,
                              .initialCapacity = n_vectors,
                              .blockSize = block_size};

    BFParams bf_params = {.type = type,
                          .dim = dim,
                          .metric = VecSimMetric_Cosine,
                          .initialCapacity = n_vectors,
                          .blockSize = block_size};

    indices.push_back(CreateNewIndex(bf_params));
    indices.push_back(CreateNewIndex(hnsw_params));

    // Load pre-generated HNSW index. Index file path is relative to repository root dir.
    LoadHNSWIndex(AttachRootPath(hnsw_index_file));

    // Add the same vectors to Flat index.
    for (size_t i = 0; i < n_vectors; ++i) {
        char *blob = GetHNSWDataByInternalId(i);
        VecSimIndex_AddVector(indices[VecSimAlgo_BF], blob, i);
    }

    // Load the test query vectors form file. Index file path is relative to repository root dir.
    loadTestVectors();
}

void BM_VecSimUtils::loadTestVectors() {
    auto location = std::string(std::string(getenv("ROOT")));
    auto file_name = location + "/" + test_vectors_file;

    std::ifstream input(file_name, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);
    InsertToQueries(input);
}

template <typename data_t>
void BM_VecSimUtils::RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                                  size_t &correct, const std::vector<std::vector<data_t>> &queries,
                                  Offset_t index_offset) {
    auto query_params = VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.efRuntime = ef}};
    auto hnsw_results =
        VecSimIndex_TopKQuery(indices[VecSimAlgo_HNSWLIB + index_offset],
                              queries[iter % n_queries].data(), k, &query_params, BY_SCORE);
    st.PauseTiming();

    // Measure recall:
    auto bf_results = VecSimIndex_TopKQuery(indices[VecSimAlgo_BF + index_offset],
                                            queries[iter % n_queries].data(), k, nullptr, BY_SCORE);
    auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
    while (VecSimQueryResult_IteratorHasNext(hnsw_it)) {
        auto hnsw_res_item = VecSimQueryResult_IteratorNext(hnsw_it);
        auto bf_it = VecSimQueryResult_List_GetIterator(bf_results);
        while (VecSimQueryResult_IteratorHasNext(bf_it)) {
            auto bf_res_item = VecSimQueryResult_IteratorNext(bf_it);
            if (VecSimQueryResult_GetId(hnsw_res_item) == VecSimQueryResult_GetId(bf_res_item)) {
                correct++;
                break;
            }
        }
        VecSimQueryResult_IteratorFree(bf_it);
    }
    VecSimQueryResult_IteratorFree(hnsw_it);

    VecSimQueryResult_Free(bf_results);
    VecSimQueryResult_Free(hnsw_results);
    st.ResumeTiming();
}
/*
 *  Populate the given queries vector with the serialized raw vectors data in
 *  the file which is located in the given path.

*/
#define HNSW_TOP_K_ARGS(ef_runtime, k) Args({ef_runtime, k})->ArgNames({"ef_runtime", "k"})

#define FIXED_BATCH_SIZE_ARGS(batch_size, num_batches)                                             \
    ->Args({batch_size, num_batches})->ArgNames({"batch size", "number of batches"})

#define VARIABLE_BATCH_SIZE_ARGS(initial_batch_size, num_batches)                                  \
    ->Args({initial_batch_size, num_batches})->ArgNames({"batch initial size", "number of batches"})

#define BATCHES_TO_ADHOC_ARGS(step, num_batches)                                                   \
    ->Args({step, num_batches})->ArgNames({"step", "number of batches"})

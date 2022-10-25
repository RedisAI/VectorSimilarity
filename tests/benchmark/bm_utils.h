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

class BM_VecSimGeneral : public benchmark::Fixture {
public:
    static size_t block_size;

protected:
    static size_t dim;
    static size_t n_vectors;
    static size_t M;
    static size_t EF_C;
    static size_t block_size;

    static const char *test_vectors_file;
    static size_t n_queries;

    static size_t ref_count;

    static std::vector<VecSimIndex *> indices;

    static const char *hnsw_index_file;

    BM_VecSimGeneral() = default;
    virtual ~BM_VecSimGeneral();

    static void MeasureRecall(VecSimQueryResult_List hnsw_results,
                              VecSimQueryResult_List bf_results, size_t &correct);

    template <typename IndexParams>
    static inline VecSimIndex *CreateNewIndex(IndexParams &index_params) {
        VecSimParams params = CreateParams(index_params);
        return VecSimIndex_New(&params);
    }

    static inline std::string AttachRootPath(const char *file_name) {
        return std::string(getenv("ROOT")) + "/" + file_name;
    }

private:
    static inline VecSimParams CreateParams(const HNSWParams &hnsw_params) {
        VecSimParams params{.algo = VecSimAlgo_HNSWLIB, .hnswParams = hnsw_params};
        return params;
    }

    static inline VecSimParams CreateParams(const BFParams &bf_params) {
        VecSimParams params{.algo = VecSimAlgo_BF, .bfParams = bf_params};
        return params;
    }
};

template <bool is_multi>
BM_VecSimBasics<is_multi>::BM_VecSimBasics(VecSimType type) {
    if (ref_count == 0) {
        // Initialize the static members.
        Initialize(type);
    }
    ref_count++;
}

template <bool is_multi>
void BM_VecSimBasics<is_multi>::Initialize(VecSimType type) {

    // Initialize and load HNSW index for DBPedia data set.
    HNSWParams hnsw_params = {.type = type,
                              .dim = BM_VecSimBasics::dim,
                              .metric = VecSimMetric_Cosine,
                              .multi = is_multi,
                              .initialCapacity = BM_VecSimBasics::n_vectors,
                              .blockSize = BM_VecSimBasics::block_size,
                              .M = BM_VecSimBasics::M,
                              .efConstruction = BM_VecSimBasics::EF_C};

    BFParams bf_params = {.type = type,
                          .dim = BM_VecSimBasics::dim,
                          .metric = VecSimMetric_Cosine,
                          .initialCapacity = BM_VecSimBasics::n_vectors,
                          .blockSize = BM_VecSimBasics::block_size};

    InitializeIndicesVector(CreateNewIndex(bf_params), CreateNewIndex(hnsw_params));

    // Load pre-generated HNSW index. Index file path is relative to repository root dir.
    LoadHNSWIndex(GetSerializedIndexLocation(BM_VecSimBasics::hnsw_index_file));

    // Add the same vectors to Flat index.
    for (size_t i = 0; i < n_vectors; ++i) {
        char *blob = GetHNSWDataByInternalId(i);
        VecSimIndex_AddVector(GetBF(), blob, i);
    }

    // Load the test query vectors form file. Index file path is relative to repository root dir.
    load_test_vectors();
}

template <bool is_multi>
void BM_VecSimBasics<is_multi>::load_test_vectors() {
    auto location = std::string(std::string(getenv("ROOT")));
    auto file_name = location + "/" + BM_VecSimBasics::test_vectors_file;

    std::ifstream input(file_name, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);
    InsertToQueries(input);
}

template <bool is_multi>
template <typename data_t>
void BM_VecSimBasics<is_multi>::RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                                             size_t &correct, VecSimIndex *hnsw_index_,
                                             VecSimIndex *bf_index_,
                                             const std::vector<std::vector<data_t>> &queries) {
    auto query_params = VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.efRuntime = ef}};
    auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index_, queries[iter % n_queries].data(), k,
                                              &query_params, BY_SCORE);
    st.PauseTiming();

    // Measure recall:
    auto bf_results =
        VecSimIndex_TopKQuery(bf_index_, queries[iter % n_queries].data(), k, nullptr, BY_SCORE);
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
void load_test_vectors(const char *path, std::vector<std::vector<float>> &queries, size_t n_queries,
                       size_t dim);

    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);
    InsertToQueries(input);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::InsertToQueries(std::ifstream &input) {
    for (size_t i = 0; i < BM_VecSimIndex::n_queries; i++) {
        std::vector<data_t> query(dim);
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}

#define VARIABLE_BATCH_SIZE_ARGS(initial_batch_size, num_batches)                                  \
    ->Args({initial_batch_size, num_batches})->ArgNames({"batch initial size", "number of batches"})

#define BATCHES_TO_ADHOC_ARGS(step, num_batches)                                                   \
    ->Args({step, num_batches})->ArgNames({"step", "number of batches"})

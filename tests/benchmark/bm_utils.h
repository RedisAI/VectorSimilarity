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

BM_VecSimGeneral::~BM_VecSimGeneral() {
    ref_count--;
    if (ref_count == 0) {
        VecSimIndex_Free(indices[VecSimAlgo_BF]);
        VecSimIndex_Free(indices[VecSimAlgo_HNSWLIB]);
    }
}

void BM_VecSimGeneral::MeasureRecall(VecSimQueryResult_List hnsw_results,
                                     VecSimQueryResult_List bf_results, size_t &correct) {
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
}

std::vector<VecSimIndex *> BM_VecSimGeneral::indices = std::vector<VecSimIndex *>();
template <typename index_type_t>
class BM_VecSimIndex : public BM_VecSimGeneral {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    static std::vector<std::vector<data_t>> queries;

    BM_VecSimIndex(bool is_multi);
    virtual ~BM_VecSimIndex() = default;

protected:
    static inline HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex<data_t, dist_t> *>(index);
    }
    static inline char *GetHNSWDataByInternalId(size_t id, Offset_t index_offset = 0) {
        return CastToHNSW(INDICES[VecSimAlgo_HNSWLIB + index_offset])->getDataByInternalId(id);
    }

    static void LoadHNSWIndex(const std::string &location, Offset_t index_offset = 0);

private:
    static void Initialize(bool is_multi);
    static void loadTestVectors();
    static void InsertToQueries(std::ifstream &input);
};

template <typename index_type_t>
std::vector<std::vector<typename index_type_t::data_t>> BM_VecSimIndex<index_type_t>::queries =
    std::vector<std::vector<typename index_type_t::data_t>>();

template <typename index_type_t>
BM_VecSimIndex<index_type_t>::BM_VecSimIndex(bool is_multi) {
    if (ref_count == 0) {
        // Initialize the static members.
        Initialize(is_multi);
    }
    ref_count++;
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::Initialize(bool is_multi) {


    BFParams bf_params = {.type = index_type_t::get_index_type(),
                          .dim = dim,
                          .metric = VecSimMetric_Cosine,
                          .initialCapacity = n_vectors,
                          .blockSize = block_size};

    INDICES.push_back(CreateNewIndex(bf_params));
    
    // Initialize and load HNSW index for DBPedia data set.
    INDICES.push_back(HNSWFactory::NewIndex(
            GetSerializedIndexLocation(hnsw_index_file), index_type_t::get_index_type(), false));

    // Add the same vectors to Flat index.
    for (size_t i = 0; i < n_vectors; ++i) {
        char *blob = GetHNSWDataByInternalId(i);
        VecSimIndex_AddVector(INDICES[VecSimAlgo_BF], blob, i);
    }

    // Load the test query vectors form file. Index file path is relative to repository root dir.
    loadTestVectors();
}
template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::LoadHNSWIndex(const std::string &location,
                                                 Offset_t index_offset) {
    auto *hnsw_index = CastToHNSW(INDICES[VecSimAlgo_HNSWLIB + index_offset]);

    hnsw_index->loadIndex(location);
    size_t ef_r = 10;
    hnsw_index->setEf(ef_r);
}
template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::loadTestVectors() {
    auto location = std::string(std::string(getenv("ROOT")));
    auto file_name = location + "/" + test_vectors_file;

    std::ifstream input(file_name, std::ios::binary);

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

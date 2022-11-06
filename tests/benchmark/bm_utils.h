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
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "bm_definitions.h"

class BM_VecSimGeneral : public benchmark::Fixture {
public:
    static size_t block_size;

protected:
    static size_t dim;
    static size_t M;
    static size_t EF_C;
    static size_t n_vectors;

    static bool is_multi;
    static std::vector<const char *> test_vectors_files;
    static size_t n_queries;

    static size_t ref_count;

    static std::vector<VecSimIndex *> indices;

    static std::vector<const char *> hnsw_index_files;

    BM_VecSimGeneral() = default;
    virtual ~BM_VecSimGeneral();

    static void MeasureRecall(VecSimQueryResult_List hnsw_results,
                              VecSimQueryResult_List bf_results, size_t &correct);

    template <typename IndexParams>
    static inline VecSimIndex *CreateNewIndex(IndexParams &index_params) {
        VecSimParams params = CreateParams(index_params);
        return VecSimIndex_New(&params);
    }

    static inline std::string AttachRootPath(std::string file_name) {
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

template <typename index_type_t>
class BM_VecSimIndex : public BM_VecSimGeneral {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    static std::vector<std::vector<data_t>> queries;

    BM_VecSimIndex() = default;
    virtual ~BM_VecSimIndex() = default;

protected:
    static inline HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex<data_t, dist_t> *>(index);
    }
    static inline char *GetHNSWDataByInternalId(size_t id, Offset_t index_offset = 0) {
        return CastToHNSW(INDICES[VecSimAlgo_HNSWLIB + index_offset])->getDataByInternalId(id);
    }

private:
    static void Initialize();
    static void loadTestVectors(VecSimType type);
    static void InsertToQueries(std::ifstream &input);
};

template <typename index_type_t>
std::vector<std::vector<typename index_type_t::data_t>> BM_VecSimIndex<index_type_t>::queries =
    std::vector<std::vector<typename index_type_t::data_t>>();

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::Initialize() {

    VecSimType type = index_type_t::get_index_type();

    BFParams bf_params = {.type = type,
                          .dim = dim,
                          .metric = VecSimMetric_Cosine,
                          .multi = IS_MULTI,
                          .initialCapacity = n_vectors,
                          .blockSize = block_size};

    INDICES.push_back(CreateNewIndex(bf_params));

    HNSWParams params = {.type = type,
                         .dim = DIM,
                         .metric = VecSimMetric_Cosine,
                         .multi = IS_MULTI,
                         .initialCapacity = N_VECTORS,
                         .blockSize = BM_VecSimGeneral::block_size,
                         .M = BM_VecSimGeneral::M,
                         .efConstruction = BM_VecSimGeneral::EF_C};

    // Initialize and load HNSW index for DBPedia data set.
    INDICES.push_back(HNSWFactory::NewIndex(AttachRootPath(hnsw_index_files[type]), &params));

    auto *hnsw_index = CastToHNSW(INDICES[VecSimAlgo_HNSWLIB]);
    size_t ef_r = 10;
    hnsw_index->setEf(ef_r);

    // Add the same vectors to Flat index.
    for (size_t i = 0; i < n_vectors; ++i) {
        char *blob = GetHNSWDataByInternalId(i);
        VecSimIndex_AddVector(INDICES[VecSimAlgo_BF], blob, i);
    }

    // Load the test query vectors form file. Index file path is relative to repository root dir.
    loadTestVectors(type);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::loadTestVectors(VecSimType type) {
    auto file_name = AttachRootPath(test_vectors_files[type]);

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

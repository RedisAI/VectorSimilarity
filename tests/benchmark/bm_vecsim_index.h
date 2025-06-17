/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include "bm_vecsim_general.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"

template <typename index_type_t>
class BM_VecSimIndex : public BM_VecSimGeneral {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    static size_t ref_count;

    static std::vector<std::vector<data_t>> queries;

    static std::array<VecSimIndex *, MAX_INDEX_COUNT> indices;

    BM_VecSimIndex();

    virtual ~BM_VecSimIndex();

protected:
    static inline HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        return dynamic_cast<HNSWIndex<data_t, dist_t> *>(index);
    }
    static inline const char *GetHNSWDataByInternalId(size_t id, unsigned short index_offset = 0) {
        return CastToHNSW(indices[INDEX_VecSimAlgo_HNSWLIB + index_offset])->getDataByInternalId(id);
    }

private:
    static void Initialize();
    static void InsertToQueries(std::ifstream &input);
    static void loadTestVectors(const std::string &test_file, VecSimType type);
};

template <typename index_type_t>
size_t BM_VecSimIndex<index_type_t>::ref_count = 0;

// Needs to be explicitly initalized
template <>
std::vector<std::vector<float>> BM_VecSimIndex<fp32_index_t>::queries{};

template <>
std::vector<std::vector<double>> BM_VecSimIndex<fp64_index_t>::queries{};

template <>
std::vector<std::vector<vecsim_types::bfloat16>> BM_VecSimIndex<bf16_index_t>::queries{};

template <>
std::vector<std::vector<vecsim_types::float16>> BM_VecSimIndex<fp16_index_t>::queries{};

template <>
std::vector<std::vector<int8_t>> BM_VecSimIndex<int8_index_t>::queries{};

template <>
std::vector<std::vector<uint8_t>> BM_VecSimIndex<uint8_index_t>::queries{};

template <>
std::array<VecSimIndex *, MAX_INDEX_COUNT> BM_VecSimIndex<fp32_index_t>::indices{};

template <>
std::array<VecSimIndex *, MAX_INDEX_COUNT> BM_VecSimIndex<fp64_index_t>::indices{};

template <>
std::array<VecSimIndex *, MAX_INDEX_COUNT> BM_VecSimIndex<bf16_index_t>::indices{};

template <>
std::array<VecSimIndex *, MAX_INDEX_COUNT> BM_VecSimIndex<fp16_index_t>::indices{};

template <>
std::array<VecSimIndex *, MAX_INDEX_COUNT> BM_VecSimIndex<int8_index_t>::indices{};

template <>
std::array<VecSimIndex *, MAX_INDEX_COUNT> BM_VecSimIndex<uint8_index_t>::indices{};

template <typename index_type_t>
BM_VecSimIndex<index_type_t>::~BM_VecSimIndex() {
    ref_count--;
    if (ref_count == 0) {
        VecSimIndex_Free(indices[VecSimAlgo_BF]);
        /* Note that VecSimAlgo_HNSW will be destroyed as part of the tiered index release, and
         * the VecSimAlgo_Tiered index ptr will be deleted when the mock thread pool ctx object is
         * destroyed.
         */
    }
}

template <typename index_type_t>
BM_VecSimIndex<index_type_t>::BM_VecSimIndex() {
    if (ref_count == 0) {
        // Initialize the static members.
        Initialize();
    }
    ref_count++;
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::Initialize() {

    VecSimType type = index_type_t::get_index_type();
    // dim, block_size, M, EF_C, n_vectors, is_multi, n_queries, hnsw_index_file and
    // test_queries_file are BM_VecSimGeneral static data members that are defined for a specific
    // index type benchmarks.
    BFParams bf_params = {.type = type,
                          .dim = dim,
                          .metric = VecSimMetric_Cosine,
                          .multi = is_multi,
                          .blockSize = block_size};

    if (IndexTypeFlags::INDEX_TYPE_BF & enabled_index_types) {
        // Create a new BF index.
        indices[INDEX_VecSimAlgo_BF] = CreateNewIndex(bf_params);
    }

    if (IndexTypeFlags::INDEX_TYPE_HNSW & enabled_index_types) {
    // Initialize and load HNSW index for DBPedia data set.
        indices[INDEX_VecSimAlgo_HNSWLIB] = (HNSWFactory::NewIndex(AttachRootPath(hnsw_index_file)));
    }
    if (IndexTypeFlags::INDEX_TYPE_TIERED_HNSW & enabled_index_types) {
        // Initialize and load HNSW index for DBPedia data set.
        auto *hnsw_index = CastToHNSW(indices[INDEX_VecSimAlgo_HNSWLIB]);
        size_t ef_r = 10;
        hnsw_index->setEf(ef_r);

        // Create tiered index from the loaded HNSW index.
        auto &mock_thread_pool = BM_VecSimGeneral::mock_thread_pool;
        TieredIndexParams tiered_params = {.jobQueue = &BM_VecSimGeneral::mock_thread_pool.jobQ,
                                        .jobQueueCtx = mock_thread_pool.ctx,
                                        .submitCb = tieredIndexMock::submit_callback,
                                        .flatBufferLimit = block_size,
                                        .primaryIndexParams = nullptr,
                                        .specificParams = {TieredHNSWParams{.swapJobThreshold = 0}}};

        auto *tiered_index =
            TieredFactory::TieredHNSWFactory::NewIndex<data_t, dist_t>(&tiered_params, hnsw_index);
        mock_thread_pool.ctx->index_strong_ref.reset(tiered_index);
        indices[INDEX_VecSimAlgo_TIERED_HNSW] = tiered_index;
    }

    // Launch the BG threads loop that takes jobs from the queue and executes them.

    // Add the same vectors to Flat index.
    // for (size_t i = 0; i < n_vectors; ++i) {
    //     const char *blob = GetHNSWDataByInternalId(i);
    //     // Fot multi value indices, the internal id is not necessarily equal the label.
    //     size_t label = CastToHNSW(indices[VecSimAlgo_HNSWLIB])->getExternalLabel(i);
    //     VecSimIndex_AddVector(indices[VecSimAlgo_BF], blob, label);
    // }
    if constexpr (std::is_same_v<data_t, float>) {
    if (IndexTypeFlags::INDEX_TYPE_SVS & enabled_index_types) {
        // Initialize and load HNSW index for DBPedia data set.
        SVSParams svs_params = {
            .type = type,
            .dim = dim,
            .metric = VecSimMetric_Cosine,
            /* SVS-Vamana specifics */
            .quantBits = VecSimSvsQuant_NONE,
            .graph_max_degree = CastToHNSW(indices[INDEX_VecSimAlgo_HNSWLIB])->getM(),
            .construction_window_size = CastToHNSW(indices[INDEX_VecSimAlgo_HNSWLIB])->getEf(),
    };
        indices[INDEX_VecSimAlgo_SVS] = CreateNewIndex(svs_params);
    }

    
    if (IndexTypeFlags::INDEX_TYPE_TIERED_SVS & enabled_index_types) {
        // Create SVS parameters for the tiered index
        SVSParams svs_params = {
            .type = type,
            .dim = dim,
            .metric = VecSimMetric_Cosine,
            .quantBits = VecSimSvsQuant_NONE,
            .graph_max_degree = 16, // Default value or based on your requirements
            .construction_window_size = 200, // Default value or based on your requirements
        };

        VecSimParams primary_params = {
            .algo = VecSimAlgo_SVS,
            .algoParams = {.svsParams = svs_params}
        };

        // Create tiered index parameters with proper SVS primary index params
        TieredIndexParams tiered_svs_params = {
            .jobQueue = &BM_VecSimGeneral::mock_thread_pool.jobQ,
            .jobQueueCtx = mock_thread_pool.ctx,
            .submitCb = tieredIndexMock::submit_callback,
            .flatBufferLimit = block_size,
            .primaryIndexParams = &primary_params,
            .specificParams = {.tieredSVSParams = {
                .trainingTriggerThreshold = 0,
                .updateTriggerThreshold = 0,
                .updateJobWaitTime = 0,
            }}};

        if (!indices[INDEX_VecSimAlgo_SVS]) {
            indices[INDEX_VecSimAlgo_TIERED_SVS] = TieredFactory::TieredSVSFactory::NewIndex(&tiered_svs_params);
        }
        else {
            indices[INDEX_VecSimAlgo_TIERED_SVS] = TieredFactory::TieredSVSFactory::NewIndex(&tiered_svs_params,
                                                                                 indices[INDEX_VecSimAlgo_SVS]);
        }

        mock_thread_pool.ctx->index_strong_ref.reset(indices[INDEX_VecSimAlgo_TIERED_SVS]);
    }
}

    // Add the SVS index to the indices vector.
    

    // Load the test query vectors form file. Index file path is relative to repository root dir.
    loadTestVectors(AttachRootPath(test_queries_file), type);

    VecSim_SetLogCallbackFunction(nullptr);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::loadTestVectors(const std::string &test_file, VecSimType type) {

    std::ifstream input(test_file, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);

    InsertToQueries(input);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::InsertToQueries(std::ifstream &input) {
    for (size_t i = 0; i < n_queries; i++) {
        std::vector<data_t> query(dim);
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}

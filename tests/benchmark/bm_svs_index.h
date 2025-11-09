/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include <benchmark/benchmark.h>
#include "gtest/gtest.h"
#include "VecSim/utils/serializer.h"
#include "bm_common.h"
#include "bm_utils.h"
#include "bm_vecsim_general.h"
#include "VecSim/index_factories/svs_factory.h"
#include <filesystem>
#include <cstdlib>
#include <string>
/**************************************
  Basic tests for updated single value index.
***************************************/

template <typename index_type_t>
class BM_VecSimSVSdIndex : public BM_VecSimCommon<index_type_t> {
public:
    using data_t = typename index_type_t::data_t;

    BM_VecSimSVSdIndex() {
        if (!is_initialized) {
            initialize();
            is_initialized = true;
        }
    }

    void AddLabelInPlace(benchmark::State &st);
    void AddLabelBatches(benchmark::State &st);
    void AddLabelAsync(benchmark::State &st);

private:
    static const char *svs_index_tar_file;
    static bool is_initialized;

    static void initialize();
    static void extractTarGz(const std::string &filename, const std::string &destination) {

        // Extract tar.gz
        std::string command = "tar -xzf " + filename + " -C " + destination;
        int result = system(command.c_str());
        if (result != 0) {
            throw std::runtime_error("Failed to extract tar.gz file");
        }
    }

    SVSIndexBase *GetSVSIndex() const {
        assert(is_initialized);
        auto result = dynamic_cast<SVSIndexBase *>(INDICES[INDEX_SVS].get());
        assert(result);
        return result;
    }

    TieredSVSIndex<data_t> *GetTieredSVSIndex() const {
        assert(is_initialized);
        auto result = dynamic_cast<TieredSVSIndex<data_t> *>(INDICES[INDEX_TIERED_SVS].get());
        assert(result);
        return result;
    }
};

template <typename index_type_t>
void BM_VecSimSVSdIndex<index_type_t>::initialize() {
    extractTarGz(BM_VecSimGeneral::AttachRootPath(svs_index_tar_file),
                 BM_VecSimGeneral::AttachRootPath("tests/benchmark/data"));
    if (BM_VecSimGeneral::enabled_index_types & IndexTypeFlags::INDEX_MASK_SVS) {
        auto &mock_thread_pool = *BM_VecSimGeneral::mock_thread_pool;
        SVSParams svs_params = {.type = index_type_t::get_index_type(),
                                .dim = DIM,
                                .metric = VecSimMetric_Cosine,
                                .multi = IS_MULTI,
                                .graph_max_degree = BM_VecSimGeneral::M,
                                .construction_window_size = BM_VecSimGeneral::EF_C,
                                .num_threads =
                                    BM_VecSimGeneral::mock_thread_pool->thread_pool_size};
        VecSimParams params = BM_VecSimGeneral::CreateParams(svs_params);
        // TODO: in tiered should be is_normalized = true
        INDICES[INDEX_SVS] = IndexPtr(SVSFactory::NewIndex(
            BM_VecSimGeneral::AttachRootPath("tests/benchmark/data/dbpedia_svs_none"), &params));
        TieredIndexParams tiered_params =
            test_utils::CreateTieredSVSParams(params, mock_thread_pool, 1000, 10000);
        auto *svs_index =
            static_cast<VecSimIndexAbstract<data_t, float> *>(INDICES[INDEX_SVS].get());
        auto *tiered_index =
            TieredFactory::TieredSVSFactory::NewIndex<data_t>(&tiered_params, svs_index);
        assert(tiered_index);

        INDICES[INDEX_TIERED_SVS] = IndexPtr(tiered_index);
        // make sure we don't override an existing index reference
        ASSERT_EQ(mock_thread_pool.ctx->index_strong_ref, nullptr);
        mock_thread_pool.ctx->index_strong_ref = INDICES[INDEX_TIERED_SVS].get_shared();
        // Release HNSW ownership since tiered will free it (sets owns_ptr=false)
        INDICES[INDEX_SVS].release_ownership();

        // Launch the BG threads loop that takes jobs from the queue and executes them.
        mock_thread_pool.init_threads();
    }
}

template <typename index_type_t>
void BM_VecSimSVSdIndex<index_type_t>::AddLabelInPlace(benchmark::State &st) {
    auto original_mode = VecSimIndexInterface::asyncWriteMode;
    auto original_num_threads = this->GetSVSIndex()->getNumThreads();
    VecSim_SetWriteMode(VecSim_WriteInPlace);
    this->GetSVSIndex()->setNumThreads(1);
    auto index = GET_INDEX(st.range(0));
    BM_VecSimBasics<index_type_t>::AddLabel(st);
    ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes,
              BM_VecSimGeneral::block_size);
    index->runGC();
    ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes, 0);
    // Restore original write mode
    ASSERT_EQ(VecSimIndexInterface::asyncWriteMode, VecSim_WriteInPlace);
    VecSim_SetWriteMode(original_mode);

    // Restore original num threads
    ASSERT_EQ(this->GetSVSIndex()->getNumThreads(), 1);
    this->GetSVSIndex()->setNumThreads(original_num_threads);
}

template <typename index_type_t>
void BM_VecSimSVSdIndex<index_type_t>::AddLabelBatches(benchmark::State &st) {
    auto index = GET_INDEX(INDEX_SVS);

    size_t batch_size = st.range(0);
    size_t num_threads = st.range(1);
    this->GetSVSIndex()->setNumThreads(num_threads);
    // Ensure we have enough vectors to add
    ASSERT_GE(QUERIES.size(), batch_size);
    size_t first_label = index->indexLabelCount();
    std::vector<size_t> ids(batch_size);
    std::iota(ids.begin(), ids.end(), first_label);

    std::vector<data_t> flattened_data;
    flattened_data.reserve(batch_size * DIM);
    for (size_t i = 0; i < batch_size; ++i) {
        flattened_data.insert(flattened_data.end(), QUERIES[i].begin(), QUERIES[i].end());
    }

    size_t memory_delta = index->getAllocationSize();
    std::cout << "start " << std::endl;
    for (auto _ : st) {
        // Add batch directly to svs
        this->GetSVSIndex()->addVectors(flattened_data.data(), ids.data(), batch_size);
        std::cout << "done add vectors starting cleanup " << std::endl;

        st.PauseTiming();
        size_t index_size_after = VecSimIndex_IndexSize(index);
        ASSERT_EQ(index_size_after, N_VECTORS + batch_size);
        memory_delta = index->getAllocationSize() - memory_delta;
        size_t expected_threads = BM_VecSimGeneral::mock_thread_pool->thread_pool_size;

        ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes,
                  0);
        // Remove new vectors
        this->GetSVSIndex()->deleteVectors(ids.data(), batch_size);
        ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes,
                  batch_size);
        index->runGC();
        ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes,
                  0);
        ASSERT_EQ(VecSimIndex_IndexSize(GET_INDEX(INDEX_SVS)), N_VECTORS);

        st.ResumeTiming();
    }

    st.counters["memory_per_vector"] =
        benchmark::Counter((double)memory_delta / (double)batch_size, benchmark::Counter::kDefaults,
                           benchmark::Counter::OneK::kIs1024);

    ASSERT_EQ(VecSimIndex_IndexSize(index), N_VECTORS);
}

template <typename index_type_t>
void BM_VecSimSVSdIndex<index_type_t>::AddLabelAsync(benchmark::State &st) {
    ASSERT_EQ(VecSimIndexInterface::asyncWriteMode, VecSim_WriteAsync);
    auto index = GET_INDEX(INDEX_TIERED_SVS);
    size_t update_trigger_threshold = st.range(0);
    this->GetTieredSVSIndex()->setUpdateTriggerThreshold(update_trigger_threshold);
    ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_TIERED_SVS))
                  .tieredInfo.specificTieredBackendInfo.svsTieredInfo.updateTriggerThreshold,
              update_trigger_threshold);

    size_t num_threads = st.range(1);
    this->GetSVSIndex()->setThreadPoolCapacity(num_threads);
    // Ensure we have enough vectors to add
    ASSERT_GE(QUERIES.size(), update_trigger_threshold);
    labelType initial_label_count = index->indexLabelCount();

    size_t memory_delta = index->getAllocationSize();
    std::cout << "start " << std::endl;
    for (auto _ : st) {
        // Add batch directly to svs
        for (labelType i = 0; i < update_trigger_threshold; ++i) {
            VecSimIndex_AddVector(index, QUERIES[i].data(), initial_label_count + i);
        }

        BM_VecSimGeneral::mock_thread_pool->thread_pool_wait();
        std::cout << "done add vectors starting cleanup " << std::endl;

        st.PauseTiming();
        size_t index_size_after = VecSimIndex_IndexSize(GET_INDEX(INDEX_SVS));
        ASSERT_EQ(index_size_after, N_VECTORS + update_trigger_threshold);
        memory_delta = index->getAllocationSize() - memory_delta;

        ASSERT_NO_FATAL_FAILURE(
            test_utils::verifyNumThreads(GetTieredSVSIndex(), num_threads, num_threads));

        ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes,
                  0);
        // Remove new vectors
        for (labelType i = 0; i < update_trigger_threshold; ++i) {
            VecSimIndex_DeleteVector(GET_INDEX(INDEX_SVS), initial_label_count + i);
        }
        ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes,
                  update_trigger_threshold);
        index->runGC();
        ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes,
                  0);
        ASSERT_EQ(VecSimIndex_IndexSize(GET_INDEX(INDEX_SVS)), N_VECTORS);
        std::cout << "done cleanup " << std::endl;

        st.ResumeTiming();
    }

    st.counters["memory_per_vector"] =
        benchmark::Counter((double)memory_delta / (double)update_trigger_threshold,
                           benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1024);

    ASSERT_EQ(VecSimIndex_IndexSize(index), N_VECTORS);
}

template <typename index_type_t>
bool BM_VecSimSVSdIndex<index_type_t>::is_initialized = false;

#define REGISTER_AddLabelSVS(BM_FUNC, VecSimAlgo)                                                  \
    BENCHMARK_REGISTER_F(BM_VecSimSVSdIndex, BM_FUNC)                                              \
        ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo)                                                     \
        ->ArgName(#VecSimAlgo)

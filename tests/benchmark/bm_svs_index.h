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
            extractTarGz(this->AttachRootPath(svs_index_tar_file),
                         this->AttachRootPath("tests/benchmark/data"));
            if (BM_VecSimGeneral::enabled_index_types & IndexTypeFlags::INDEX_MASK_SVS) {
                auto &mock_thread_pool = *BM_VecSimGeneral::mock_thread_pool;
                SVSParams svs_params = {
                    .type = index_type_t::get_index_type(),
                    .dim = DIM,
                    .metric = VecSimMetric_Cosine,
                    .multi = IS_MULTI,
                    .graph_max_degree = BM_VecSimGeneral::M,
                    .construction_window_size = BM_VecSimGeneral::EF_C,
                };
                VecSimParams params = this->CreateParams(svs_params);
                // TODO: in tiered should be is_normalized = true
                INDICES[INDEX_SVS] = IndexPtr(SVSFactory::NewIndex(
                    this->AttachRootPath("tests/benchmark/data/dbpedia_svs_none"), &params));
                TieredIndexParams tiered_params =
                    test_utils::CreateTieredSVSParams(params, mock_thread_pool, 0, 0);
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
            is_initialized = true;
        }
    }
    static void AddLabel(benchmark::State &st);
    static void AddLabelBatches(benchmark::State &st);

private:
    static const char *svs_index_tar_file;
    static bool is_initialized;

    static void extractTarGz(const std::string &filename, const std::string &destination) {

        // Extract tar.gz
        std::string command = "tar -xzf " + filename + " -C " + destination;
        int result = system(command.c_str());
        if (result != 0) {
            throw std::runtime_error("Failed to extract tar.gz file");
        }
    }
};

template <typename index_type_t>
void BM_VecSimSVSdIndex<index_type_t>::AddLabel(benchmark::State &st) {
    auto original_mode = VecSimIndexInterface::asyncWriteMode;
    VecSim_SetWriteMode(VecSim_WriteInPlace);
    auto index = GET_INDEX(st.range(0));
    BM_VecSimBasics<index_type_t>::AddLabel(st);
    ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes,
              BM_VecSimGeneral::block_size);
    index->runGC();
    ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes, 0);
    Restore original write mode ASSERT_EQ(VecSimIndexInterface::asyncWriteMode,
                                          VecSim_WriteInPlace);
    VecSim_SetWriteMode(original_mode);
}

template <typename index_type_t>
void BM_VecSimSVSdIndex<index_type_t>::AddLabelBatches(benchmark::State &st) {
    ASSERT_EQ(VecSimIndexInterface::asyncWriteMode, VecSim_AsyncWrite);
    auto index = GET_INDEX(st.range(0));
    ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes,
              BM_VecSimGeneral::block_size);
    index->runGC();
    ASSERT_EQ(VecSimIndex_DebugInfo(GET_INDEX(INDEX_SVS)).svsInfo.numberOfMarkedDeletedNodes, 0);
    Restore original write mode
}

template <typename index_type_t>
bool BM_VecSimSVSdIndex<index_type_t>::is_initialized = false;

#define REGISTER_AddLabelSVS(BM_FUNC, VecSimAlgo)                                                  \
    BENCHMARK_REGISTER_F(BM_VecSimSVSdIndex, BM_FUNC)                                              \
        ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo)                                                     \
        ->ArgName(#VecSimAlgo)

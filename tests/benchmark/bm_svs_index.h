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
#include "VecSim/utils/serializer.h"
#include "bm_common.h"
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
    BM_VecSimSVSdIndex() {
        if (!is_initialized) {
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
                extractTarGz(this->AttachRootPath(svs_index_tar_file),
                             this->AttachRootPath("tests/benchmark/data"));
                INDICES[INDEX_SVS] = IndexPtr(SVSFactory::NewIndex(
                    this->AttachRootPath("tests/benchmark/data/dbpedia_svs_none"), &params));
                mock_thread_pool.ctx->index_strong_ref = INDICES[INDEX_SVS].get_shared();
            }

            is_initialized = true;
        }
    }
    static void AddLabel(benchmark::State &st) {
        BM_VecSimBasics<index_type_t>::AddLabel(st);
        ASSERT_EQ(VecSimIndex_IndexSize(GET_INDEX(INDEX_SVS)),
                  N_VECTORS + BM_VecSimGeneral::block_size);
    }

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
bool BM_VecSimSVSdIndex<index_type_t>::is_initialized = false;

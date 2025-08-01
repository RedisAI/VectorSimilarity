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
#include <random>
#include <unistd.h>
#include <istream>
#include <thread>
#include <condition_variable>
#include <bitset>
#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_interface.h"
#include "VecSim/vec_sim_tiered_index.h"
#include "VecSim/query_results.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "bm_definitions.h"
#include "bm_macros.h"
#include "utils/mock_thread_pool.h"

// This class includes every static data member that is:
// 1. Common for all data type data sets.
// or
// 2. In use for all benchmark types, if this type
// is defined in a separate compilation unit.
class BM_VecSimGeneral : public benchmark::Fixture {
public:
    // block_size is public because it is used to define the number of iterations on some test cases
    static size_t block_size;

protected:
    static size_t dim;
    static size_t M;
    static size_t EF_C;
    static size_t n_vectors;

    static bool is_multi;
    // Bitmask controlling which index types to include in benchmarks (uses IndexTypeFlags)
    static uint32_t enabled_index_types;
    static tieredIndexMock *mock_thread_pool;

    static size_t n_queries;
    static const char *hnsw_index_file;
    static const char *test_queries_file;

    BM_VecSimGeneral() = default;
    virtual ~BM_VecSimGeneral() {
        if (mock_thread_pool) {
            delete mock_thread_pool;
            mock_thread_pool = nullptr;
        }
    };

    // Updates @correct according to the number of search results in @hnsw_results
    // that appear also in the flat algorithm results list.
    static void MeasureRecall(VecSimQueryReply *hnsw_results, VecSimQueryReply *bf_results,
                              std::atomic_int &correct);

protected:
    static inline VecSimQueryParams CreateQueryParams(const HNSWRuntimeParams &RuntimeParams) {
        VecSimQueryParams QueryParams = {.hnswRuntimeParams = RuntimeParams};
        return QueryParams;
    }

    static inline VecSimParams CreateParams(const HNSWParams &hnsw_params) {
        VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                            .algoParams = {.hnswParams = HNSWParams{hnsw_params}}};
        return params;
    }

    static inline VecSimParams CreateParams(const BFParams &bf_params) {
        VecSimParams params{.algo = VecSimAlgo_BF, .algoParams = {.bfParams = BFParams{bf_params}}};
        return params;
    }

    // Gets HNSWParams or BFParams parameters struct, and creates new VecSimIndex.
    template <typename IndexParams>
    static inline VecSimIndex *CreateNewIndex(IndexParams &index_params) {
        VecSimParams params = CreateParams(index_params);
        return VecSimIndex_New(&params);
    }

    // Adds the library's root path to @file_name
    static inline std::string AttachRootPath(std::string file_name) {
        return std::string(getenv("ROOT")) + "/" + file_name;
    }
};

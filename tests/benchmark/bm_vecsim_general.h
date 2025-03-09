/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
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
#include "utils/mock_thread_pool.h"

#define EXPAND(x) x
#define EXPAND2(x) EXPAND(x)
// Helper for raw concatenation with varying arguments
#define BM_FUNC_NAME_HELPER1_2(a, b) a ## _ ## b
#define BM_FUNC_NAME_HELPER1_3(a, b, c) a ## _ ## b ## _ ## c
#define BM_FUNC_NAME_HELPER1_4(a, b, c, d) a ## _ ## b ## _ ## c ## _ ## d
#define BM_FUNC_NAME_HELPER1_5(a, b, c, d, e) a ## _ ## b ## _ ## c ## _ ## d ## _ ## e

// Force expansion of macro arguments
#define BM_FUNC_NAME_HELPER_2(a, b) BM_FUNC_NAME_HELPER1_2(a, b)
#define BM_FUNC_NAME_HELPER_3(a, b, c) BM_FUNC_NAME_HELPER1_3(a, b, c)
#define BM_FUNC_NAME_HELPER_4(a, b, c, d) BM_FUNC_NAME_HELPER1_4(a, b, c, d)
#define BM_FUNC_NAME_HELPER_5(a, b, c, d, e) BM_FUNC_NAME_HELPER1_5(a, b, c, d, e)

// Determine the number of arguments and select the appropriate helper
#define COUNT_ARGS(...) COUNT_ARGS_(__VA_ARGS__, 6, 5, 4, 3, 2, 1)
#define COUNT_ARGS_(_1, _2, _3, _4, _5, _6, N, ...) N

// Concatenate BM_FUNC_NAME_HELPER with the number of arguments
#define CONCAT_HELPER(a, b) a ## _ ## b
#define CONCAT(a, b) CONCAT_HELPER(a, b)

// Main macro that selects the appropriate helper based on argument count
#define CONCAT_WITH_UNDERSCORE(...) EXPAND2(CONCAT(BM_FUNC_NAME_HELPER, EXPAND2(COUNT_ARGS(__VA_ARGS__)))(__VA_ARGS__))
// Modify this macro to account for the extra BENCHMARK_ARCH parameter
#define CONCAT_WITH_UNDERSCORE_ARCH(...) CONCAT_WITH_UNDERSCORE(__VA_ARGS__, BENCHMARK_ARCH)

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
    static tieredIndexMock mock_thread_pool;

    static size_t n_queries;
    static const char *hnsw_index_file;
    static const char *test_queries_file;

    BM_VecSimGeneral() = default;
    virtual ~BM_VecSimGeneral() = default;

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

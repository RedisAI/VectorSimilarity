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
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "bm_definitions.h"

using JobQueue = std::queue<AsyncJob *>;
static const size_t MAX_POOL_SIZE = 16;

void inline MarkExecuteInProcess(std::bitset<MAX_POOL_SIZE> &executions_status,
                                 size_t thread_index) {
    executions_status.set(thread_index);
}

void inline MarkExecuteDone(std::bitset<MAX_POOL_SIZE> &executions_status, size_t thread_index) {
    executions_status.reset(thread_index);
}

// This class includes every static data member that is:
// 1. Common for fp32 and fp64 data sets.
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

    static size_t n_queries;
    static const char *hnsw_index_file;
    static const char *test_queries_file;

    // Tiered index mock attributes
    static JobQueue jobQ;
    static const size_t thread_pool_size;
    static std::vector<std::thread> thread_pool;
    static std::mutex queue_guard;
    static std::condition_variable queue_cond;
    static bool run_threads;
    // A bit set to help determine whether all jobs are done by checking
    // that the job queue is empty and the all the bits are 0.
    // Each thread is associated with a bit position in the bit set.
    // The thread's corresponding bit should be set to before the job is popped
    // from the queue and the execution starts.
    // We turn the bit off after the execute callback returns to mark the job is done.
    static std::bitset<MAX_POOL_SIZE> executions_status;

    BM_VecSimGeneral() = default;
    virtual ~BM_VecSimGeneral() = default;

    // Updates @correct according to the number of search results in @hnsw_results
    // that appear also in the flat algorithm results list.
    static void MeasureRecall(VecSimQueryResult_List hnsw_results,
                              VecSimQueryResult_List bf_results, size_t &correct);

    /** Mock tiered index callbacks. **/

    static int submit_callback(void *job_queue, void *index_ctx, AsyncJob **jobs, JobCallback *CBs,
                               size_t len);
    static void thread_main_loop(int thread_id);
    static void thread_pool_join();
    static void thread_pool_wait();

protected:
    static inline VecSimQueryParams CreateQueryParams(const HNSWRuntimeParams &RuntimeParams) {
        VecSimQueryParams QueryParams = {.hnswRuntimeParams = RuntimeParams};
        return QueryParams;
    }

    static inline VecSimParams CreateParams(const HNSWParams &hnsw_params) {
        VecSimParams params{.algo = VecSimAlgo_HNSWLIB, .hnswParams = hnsw_params};
        return params;
    }

    static inline VecSimParams CreateParams(const BFParams &bf_params) {
        VecSimParams params{.algo = VecSimAlgo_BF, .bfParams = bf_params};
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

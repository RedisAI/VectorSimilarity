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

    BM_VecSimGeneral() = default;
    virtual ~BM_VecSimGeneral() = default;

    // Updates @correct according to the number of search results in @hnsw_results
    // that appear also in the flat algorithm results list.
    static void MeasureRecall(VecSimQueryResult_List hnsw_results,
                              VecSimQueryResult_List bf_results, size_t &correct);

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
        //        return std::string("/home/alon/Code/VectorSimilarity/") + file_name;
    }

    /** Mock tiered index callbacks. **/

    static int submit_callback(void *job_queue, void *index_ctx, AsyncJob **jobs, JobCallback *CBs,
                               size_t len) {
        {
            std::unique_lock<std::mutex> lock(queue_guard);
            for (size_t i = 0; i < len; i++) {
                static_cast<JobQueue *>(job_queue)->push(jobs[i]);
            }
        }
        if (len == 1) {
            queue_cond.notify_one();
        } else {
            queue_cond.notify_all();
        }
        return VecSim_OK;
    }

    // Main loop for background worker threads that execute the jobs form the job queue.
    // run_thread uses as a signal to the thread that indicates whether it should keep running or
    // stop and terminate the thread.
    static void thread_main_loop() {
        while (run_threads) {
            std::unique_lock<std::mutex> lock(queue_guard);
            // Wake up and acquire the lock (atomically) ONLY if the job queue is not empty at that
            // point, or if the thread should not run anymore (and quit in that case).
            queue_cond.wait(lock, []() { return !jobQ.empty() || !run_threads; });
            if (!run_threads)
                return;
            auto job = jobQ.front();
            jobQ.pop();
            lock.unlock();
            job->Execute(job);
        }
    }

    static void thread_pool_join() {
        // Check every 10 ms if queue is empty, and if so, terminate the threads loop.
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::unique_lock<std::mutex> lock(queue_guard);
            if (jobQ.empty()) {
                run_threads = false;
                queue_cond.notify_all();
                break;
            }
        }
        for (size_t i = 0; i < thread_pool_size; i++) {
            thread_pool[i].join();
        }
        thread_pool.clear();
    }

    static void thread_pool_wait() {
        thread_pool_join(); // wait for all threads to finish
        // Recreate the thread pool
        run_threads = true;
        for (size_t i = 0; i < thread_pool_size; i++) {
            thread_pool.emplace_back(thread_main_loop);
        }
    }
};

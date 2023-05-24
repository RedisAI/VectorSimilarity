/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <thread>
#include <condition_variable>
#include <bitset>

#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "pybind11/pybind11.h"

namespace tiered_index_mock {

std::mutex queue_guard;
std::condition_variable queue_cond;
std::vector<std::thread> thread_pool;
typedef struct RefManagedJob {
    AsyncJob *job;
    std::weak_ptr<VecSimIndex> index_weak_ref;
} RefManagedJob;

using JobQueue = std::queue<RefManagedJob>;
int submit_callback(void *job_queue, void *index_ctx, AsyncJob **jobs, JobCallback *CBs,
                    JobCallback *freeCBs, size_t len);

typedef struct IndexExtCtx {
    std::shared_ptr<VecSimIndex> index_strong_ref;
    ~IndexExtCtx() {}
} IndexExtCtx;

static const size_t MAX_POOL_SIZE = 16;
static const size_t THREAD_POOL_SIZE = MIN(MAX_POOL_SIZE, std::thread::hardware_concurrency());

class ThreadParams {
public:
    bool &run_thread;
    // A bit set to help determine whether all jobs are done by checking
    // that the job queue is empty and the all the bits are 0.
    // Each thread is associated with a bit position in the bit set.
    // The thread's corresponding bit should be set to before the job is popped
    // from the queue and the execution starts.
    // We turn the bit off after the execute callback returns to mark the job is done.
    std::bitset<MAX_POOL_SIZE> &executions_status;
    const unsigned int thread_index;
    JobQueue &jobQ;
    ThreadParams(bool &run_thread, std::bitset<MAX_POOL_SIZE> &executions_status,
                 const unsigned int thread_index, JobQueue &jobQ)
        : run_thread(run_thread), executions_status(executions_status), thread_index(thread_index),
          jobQ(jobQ) {}

    ThreadParams(const ThreadParams &other) = default;
};

void inline MarkExecuteInProcess(std::bitset<MAX_POOL_SIZE> &executions_status,
                                 size_t thread_index) {
    executions_status.set(thread_index);
}

void inline MarkExecuteDone(std::bitset<MAX_POOL_SIZE> &executions_status, size_t thread_index) {
    executions_status.reset(thread_index);
}

void thread_main_loop(ThreadParams params) {
    while (params.run_thread) {
        std::unique_lock<std::mutex> lock(queue_guard);
        // Wake up and acquire the lock (atomically) ONLY if the job queue is not empty at that
        // point, or if the thread should not run anymore (and quit in that case).
        queue_cond.wait(lock, [&params]() { return !(params.jobQ.empty()) || !params.run_thread; });
        if (!params.run_thread)
            return;
        auto managed_job = params.jobQ.front();
        MarkExecuteInProcess(params.executions_status, params.thread_index);
        params.jobQ.pop();

        lock.unlock();
        // Upgrade the index weak reference to a strong ref while we run the job over the index.
        if (auto temp_ref = managed_job.index_weak_ref.lock()) {
            managed_job.job->Execute(managed_job.job);
            MarkExecuteDone(params.executions_status, params.thread_index);
        }
    }
}

int submit_callback(void *job_queue, void *index_ctx, AsyncJob **jobs, JobCallback *CBs,
                    size_t len) {
    {
        std::unique_lock<std::mutex> lock(queue_guard);
        for (size_t i = 0; i < len; i++) {
            // Wrap the job with a struct that contains a weak reference to the related index.
            auto owned_job = RefManagedJob{
                .job = jobs[i],
                .index_weak_ref = reinterpret_cast<IndexExtCtx *>(index_ctx)->index_strong_ref};
            static_cast<JobQueue *>(job_queue)->push(owned_job);
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

void thread_pool_terminate(JobQueue &jobQ, bool &run_thread) {
    // Check every 10 ms if queue is empty, and if so, terminate the threads loop.
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::unique_lock<std::mutex> lock(queue_guard);
        if (jobQ.empty()) {
            run_thread = false;
            queue_cond.notify_all();
            break;
        }
    }
    for (size_t i = 0; i < THREAD_POOL_SIZE; i++) {
        thread_pool[i].join();
    }
    thread_pool.clear();
}
} // namespace tiered_index_mock

#pragma once

#include <functional>
#include <cmath>
#include <exception>
#include <thread>
#include <condition_variable>
#include <bitset>

#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"

namespace tiered_index_mock {

static const size_t MAX_POOL_SIZE = 16;

void inline MarkExecuteInProcess(std::bitset<MAX_POOL_SIZE> &executions_status,
                                 size_t thread_index) {
    executions_status.set(thread_index);
}

void inline MarkExecuteDone(std::bitset<MAX_POOL_SIZE> &executions_status, size_t thread_index) {
    executions_status.reset(thread_index);
}

typedef struct RefManagedJob {
    AsyncJob *job;
    std::weak_ptr<VecSimIndex> index_weak_ref;
} RefManagedJob;

struct SearchJobMock : public AsyncJob {
    void *query; // The query vector. ownership is passed to the job in the constructor.
    size_t k;    // The number of results to return.
    size_t n;    // The number of vectors in the index (might be useful for the mock)
    size_t dim;  // The dimension of the vectors in the index (might be useful for the mock)
    std::atomic_int &successful_searches; // A reference to a shared counter that counts the number
                                          // of successful searches.
    SearchJobMock(std::shared_ptr<VecSimAllocator> allocator, JobCallback searchCB,
                  VecSimIndex *index_, void *query_, size_t k_, size_t n_, size_t dim_,
                  std::atomic_int &successful_searches_)
        : AsyncJob(allocator, HNSW_SEARCH_JOB, searchCB, index_), query(query_), k(k_), n(n_),
          dim(dim_), successful_searches(successful_searches_) {}
    ~SearchJobMock() { this->allocator->free_allocation(query); }
};

struct JobQueue : public std::queue<RefManagedJob> {
    // Pops and destroys the job at the front of the queue.
    inline void kick() {
        delete this->front().job;
        this->pop();
    }
};

int submit_callback(void *job_queue, void *index_ctx, AsyncJob **jobs, JobCallback *CBs,
                    size_t jobs_len);

typedef struct IndexExtCtx {
    std::shared_ptr<VecSimIndex> index_strong_ref;
} IndexExtCtx;

static const size_t THREAD_POOL_SIZE = MIN(16, std::thread::hardware_concurrency());
extern std::vector<std::thread> thread_pool;
extern std::mutex queue_guard;
extern std::condition_variable queue_cond;

// A bit set to help determine whether all jobs are done by checking
// that the job queue is empty and the all the bits are 0.
// Each thread is associated with a bit position in the bit set.
// The thread's corresponding bit should be set to before the job is popped
// from the queue and the execution starts.
// We turn the bit off after the execute callback returns to mark the job is done.
extern std::bitset<MAX_POOL_SIZE> executions_status;

// A single iteration of the thread main loop.
void thread_iteration(JobQueue &jobQ, int thread_id = 0, const bool *run_thread = nullptr);
void thread_main_loop(JobQueue &jobQ, bool &run_thread, int thread_id);

void thread_pool_join(JobQueue &jobQ, bool &run_thread);
void thread_pool_wait(JobQueue &jobQ);

} // namespace tiered_index_mock

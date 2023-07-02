#include "mock_thread_pool.h"

/*
 * Mock callbacks and synchronization primitives for testing async tiered index.
 * We use a simple std::queue to simulate the job queue.
 */

size_t tieredIndexMock::THREAD_POOL_SIZE = MIN(8, std::thread::hardware_concurrency());
std::mutex tieredIndexMock::queue_guard;
std::condition_variable tieredIndexMock::queue_cond;
std::vector<std::thread> tieredIndexMock::thread_pool;
std::bitset<tieredIndexMock::MAX_POOL_SIZE> tieredIndexMock::executions_status;
tieredIndexMock::JobQueue tieredIndexMock::jobQ;
bool tieredIndexMock::run_thread = false;
tieredIndexMock::IndexExtCtx *tieredIndexMock::ctx = nullptr;

int tieredIndexMock::submit_callback(void *job_queue, void *index_ctx, AsyncJob **jobs,
                                     JobCallback *CBs, size_t len) {
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

// If `run_thread` is null, treat it as `true`.
void tieredIndexMock::thread_iteration(int thread_id, const bool *run_thread_ptr) {
    std::unique_lock<std::mutex> lock(queue_guard);
    // Wake up and acquire the lock (atomically) ONLY if the job queue is not empty at that
    // point, or if the thread should not run anymore (and quit in that case).
    queue_cond.wait(lock, [&]() { return !jobQ.empty() || (run_thread_ptr && !*run_thread_ptr); });
    if (run_thread_ptr && !*run_thread_ptr)
        return;
    auto managed_job = jobQ.front();
    MarkExecuteInProcess(thread_id);
    jobQ.pop();
    lock.unlock();
    // Upgrade the index weak reference to a strong ref while we run the job over the index.
    if (auto temp_ref = managed_job.index_weak_ref.lock()) {
        managed_job.job->Execute(managed_job.job);
    }
    MarkExecuteDone(thread_id);
}

// Main loop for background worker threads that execute the jobs form the job queue.
// run_thread uses as a signal to the thread that indicates whether it should keep running or
// stop and terminate the thread.
void tieredIndexMock::thread_main_loop(int thread_id) {
    while (run_thread) {
        tieredIndexMock::thread_iteration(thread_id, &run_thread);
    }
}

void tieredIndexMock::thread_pool_join() {
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

void tieredIndexMock::thread_pool_wait() {
    while (true) {
        if (jobQ.empty() && executions_status.count() == 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

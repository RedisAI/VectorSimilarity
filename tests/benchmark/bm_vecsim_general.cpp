
/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "bm_vecsim_general.h"

void BM_VecSimGeneral::MeasureRecall(VecSimQueryResult_List hnsw_results,
                                     VecSimQueryResult_List bf_results, size_t &correct) {
    auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
    while (VecSimQueryResult_IteratorHasNext(hnsw_it)) {
        auto hnsw_res_item = VecSimQueryResult_IteratorNext(hnsw_it);
        auto bf_it = VecSimQueryResult_List_GetIterator(bf_results);
        while (VecSimQueryResult_IteratorHasNext(bf_it)) {
            auto bf_res_item = VecSimQueryResult_IteratorNext(bf_it);
            if (VecSimQueryResult_GetId(hnsw_res_item) == VecSimQueryResult_GetId(bf_res_item)) {
                correct++;
                break;
            }
        }
        VecSimQueryResult_IteratorFree(bf_it);
    }
    VecSimQueryResult_IteratorFree(hnsw_it);
}

int BM_VecSimGeneral::submit_callback(void *job_queue, void *index_ctx, AsyncJob **jobs,
                                      JobCallback *CBs, size_t len) {
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
void BM_VecSimGeneral::thread_main_loop(int thread_id) {
    while (run_threads) {
        std::unique_lock<std::mutex> lock(queue_guard);
        // Wake up and acquire the lock (atomically) ONLY if the job queue is not empty at that
        // point, or if the thread should not run anymore (and quit in that case).
        queue_cond.wait(lock, []() { return !jobQ.empty() || !run_threads; });
        if (!run_threads)
            return;
        auto job = jobQ.front();
        MarkExecuteInProcess(executions_status, thread_id);
        jobQ.pop();
        lock.unlock();
        job->Execute(job);
        MarkExecuteDone(executions_status, thread_id);
    }
}

void BM_VecSimGeneral::thread_pool_join() {
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

void BM_VecSimGeneral::thread_pool_wait() {
    while (true) {
        if (jobQ.empty() && executions_status.count() == 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

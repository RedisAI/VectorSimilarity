#include <functional>

namespace tiered_index_mock {
using JobQueue = std::queue<void *>;
int submit_callback(void *job_queue, void **jobs, size_t len) {
    for (size_t i = 0; i < len; i++) {
        static_cast<JobQueue *>(job_queue)->push(jobs[i]);
    }
    return VecSim_OK;
}
int update_mem_callback(void *mem_ctx, size_t mem) {
    *(size_t *)mem_ctx = mem;
    return VecSim_OK;
}
} // namespace tiered_index_mock

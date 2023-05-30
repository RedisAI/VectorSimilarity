#include "benchmark/bm_vecsim_general.h"

JobQueue BM_VecSimGeneral::jobQ{};
const size_t BM_VecSimGeneral::thread_pool_size = MIN(8, std::thread::hardware_concurrency());
std::vector<std::thread> BM_VecSimGeneral::thread_pool{};
std::mutex BM_VecSimGeneral::queue_guard{};
std::condition_variable BM_VecSimGeneral::queue_cond{};
bool BM_VecSimGeneral::run_threads = false;
std::bitset<MAX_POOL_SIZE> BM_VecSimGeneral::executions_status{};

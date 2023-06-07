#include "benchmark/bm_vecsim_general.h"

tiered_index_mock::JobQueue BM_VecSimGeneral::jobQ{};
const size_t BM_VecSimGeneral::thread_pool_size = MIN(8, std::thread::hardware_concurrency());
bool BM_VecSimGeneral::run_threads = false;
tiered_index_mock::IndexExtCtx BM_VecSimGeneral::ctx{};

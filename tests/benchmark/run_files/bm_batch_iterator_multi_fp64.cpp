/*
* Copyright (c) 2006-Present, Redis Ltd.
* All rights reserved.
*
* Licensed under your choice of the Redis Source Available License 2.0
* (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
* GNU Affero General Public License v3 (AGPLv3).
*/


#include "benchmark/bm_batch_iterator.h"

bool BM_VecSimGeneral::is_multi = true;

// Global benchmark data
size_t BM_VecSimGeneral::n_vectors = 1111025;
size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 512;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::block_size = 1024;
tieredIndexMock BM_VecSimGeneral::mock_thread_pool{};

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/fashion_images_multi_value-cosine-dim512-M64-efc512-fp64.hnsw_v3";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/fashion_images_multi_value-cosine-dim512-fp64-test_vectors.raw";

#define BM_FUNC_NAME(bm_func, algo) CONCAT_WITH_UNDERSCORE_ARCH(algo, bm_func, Multi)

#include "benchmark/bm_initialization/bm_batch_initialize_fp64.h"

BENCHMARK_MAIN();

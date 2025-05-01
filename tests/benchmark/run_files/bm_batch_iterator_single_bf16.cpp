/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "benchmark/bm_batch_iterator.h"

bool BM_VecSimGeneral::is_multi = false;

// Global benchmark data
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::block_size = 1024;
tieredIndexMock BM_VecSimGeneral::mock_thread_pool{};

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/dbpedia-cosine-dim768-M64-efc512-bf16.hnsw_v3";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/dbpedia-cosine-dim768-bf16-test_vectors.raw";

#define BM_FUNC_NAME(bm_func, algo) CONCAT_WITH_UNDERSCORE_ARCH(algo, bm_func, Single)

#include "benchmark/bm_initialization/bm_batch_initialize_bf16.h"

BENCHMARK_MAIN();

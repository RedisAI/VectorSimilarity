/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once
/**************************************
  Define and register tests
  NOTE: benchmarks' tests order can affect their results. Please add new benchmarks at the end of
the file.
***************************************/
#define BM_FUNC_NAME(bm_func, algo) CONCAT_WITH_UNDERSCORE_ARCH(bm_func, algo, Single)

// build
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_FUNC_NAME(Build, SVS), fp32_index_t)
(benchmark::State &st) { Build_SVS(st); }
BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC_NAME(Build, SVS))
    ->Iterations(1)
    ->Unit(benchmark::kSecond);

// memory
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, SVS), fp32_index_t)
(benchmark::State &st) { Memory_SVS(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, SVS))->Iterations(1);

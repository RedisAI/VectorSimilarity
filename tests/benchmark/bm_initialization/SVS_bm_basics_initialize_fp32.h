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

// Memory
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, SVS), fp32_index_t)
(benchmark::State &st) { Memory_SVS(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, SVS))->Iterations(1);

// AddLabel
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_ADD_LABEL, fp32_index_t)
(benchmark::State &st) { AddLabel_SVS(st); }
REGISTER_AddLabel(BM_ADD_LABEL, INDEX_VecSimAlgo_SVS);

REGISTER_DeleteLabel(BM_FUNC_NAME(DeleteLabel, SVS));

// TopK SVS
// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(TopK, SVS), fp32_index_t)
// (benchmark::State &st) { TopK_SVS(st); }
// REGISTER_TopK_SVS(BM_VecSimCommon, BM_FUNC_NAME(TopK, SVS));

// // // Range SVS
// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_FUNC_NAME(Range, SVS), fp32_index_t)
// (benchmark::State &st) { Range_SVS(st); }
// REGISTER_Range_SVS(BM_FUNC_NAME(Range, SVS), fp32_index_t);

REGISTER_AddLabel(BM_ADD_LABEL, INDEX_VecSimAlgo_TIERED_SVS);
REGISTER_DeleteLabel(BM_FUNC_NAME(DeleteLabel, Tiered));

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_ADD_LABEL_ASYNC, fp32_index_t)
(benchmark::State &st) { AddLabel_AsyncIngest_SVS(st); }
BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_ADD_LABEL_ASYNC)
    ->UNIT_AND_ITERATIONS->Arg(INDEX_VecSimAlgo_TIERED_SVS)
    ->ArgName("INDEX_VecSimAlgo_TIERED_SVS");

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_DELETE_LABEL_ASYNC, fp32_index_t)
(benchmark::State &st) { DeleteLabel_AsyncRepair_SVS(st); }
BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_DELETE_LABEL_ASYNC)
    ->UNIT_AND_ITERATIONS->Arg(1)
    ->Arg(100)
    ->Arg(BM_VecSimGeneral::block_size)
    ->ArgName("SwapJobsThreshold");

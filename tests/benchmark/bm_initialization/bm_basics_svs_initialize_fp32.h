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
// deleteLabel one by one
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVS, BM_FUNC_NAME(BM_RunGC), DATA_TYPE_INDEX_T)
(benchmark::State &st) { RunGC(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVS, BM_FUNC_NAME(BM_RunGC))
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Arg(50)
    ->Arg(100)
    ->Arg(250)
    ->Arg(500)
    ->ArgName("num_deletions");

// AddLabel one by one
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVS, BM_FUNC_NAME(BM_AddLabelOneByOne), DATA_TYPE_INDEX_T)
(benchmark::State &st) { AddLabel(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVS, BM_FUNC_NAME(BM_AddLabelOneByOne))
    ->Unit(benchmark::kMillisecond)
    ->Iterations(BM_VecSimGeneral::block_size);

// Add vectors in batches via tiered index
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVS, BM_FUNC_NAME(BM_TriggerUpdateTiered), DATA_TYPE_INDEX_T)
(benchmark::State &st) { TriggerUpdateTiered(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVS, BM_FUNC_NAME(BM_TriggerUpdateTiered))
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->ArgsProduct({{static_cast<long int>(BM_VecSimGeneral::block_size), 5000,
                    static_cast<long int>(10 * BM_VecSimGeneral::block_size)},
                   {2, 4, 8}})
    ->ArgNames({"update_threshold", "thread_count"})
    ->MeasureProcessCPUTime();

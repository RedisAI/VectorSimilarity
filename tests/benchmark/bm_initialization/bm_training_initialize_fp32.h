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
#define UNIT_AND_ITERATIONS Unit(benchmark::kMillisecond)->Iterations(5)

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVSTrain, BM_Train, fp32_index_t)
(benchmark::State &st) { Train(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVSTrain, BM_Train)
    ->UNIT_AND_ITERATIONS->Arg(BM_VecSimGeneral::block_size)
    ->Arg(5 * BM_VecSimGeneral::block_size)
    ->Arg(10000)
    ->ArgName("training_threshold");

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVSTrain, BM_TrainAsync, fp32_index_t)
(benchmark::State &st) { TrainAsync(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVSTrain, BM_TrainAsync)
    ->UNIT_AND_ITERATIONS
    ->ArgsProduct({{static_cast<long int>(BM_VecSimGeneral::block_size), 5000, 10000, 50000,
                    100000},
                   {2, 4, 8, 16}})
    ->ArgNames({"training_threshold", "thread_count"})
    ->MeasureProcessCPUTime();

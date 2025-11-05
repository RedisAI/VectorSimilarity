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

#if HAVE_SVS_LVQ
#define QUANT_BITS_ARGS {VecSimSvsQuant_8, VecSimSvsQuant_4x8_LeanVec}
#else
#define QUANT_BITS_ARGS {VecSimSvsQuant_8}
#endif

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVSTrain, BM_TrainNoCompression, fp16_index_t)
(benchmark::State &st) { Train(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVSTrain, BM_TrainNoCompression)
    ->UNIT_AND_ITERATIONS
    ->Args({VecSimSvsQuant_NONE, static_cast<long int>(BM_VecSimGeneral::block_size)})
    ->ArgNames({"quant_bits", "training_threshold"});

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVSTrain, BM_TrainAsyncNoCompression, fp16_index_t)
(benchmark::State &st) { TrainAsync(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVSTrain, BM_TrainAsyncNoCompression)
    ->UNIT_AND_ITERATIONS
    ->ArgsProduct({{VecSimSvsQuant_NONE},
                   {static_cast<long int>(BM_VecSimGeneral::block_size)},
                   {4, 8, 16}})
    ->ArgNames({"quant_bits", "training_threshold", "thread_count"})
    ->MeasureProcessCPUTime();

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVSTrain, BM_TrainCompressed, fp16_index_t)
(benchmark::State &st) { Train(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVSTrain, BM_TrainCompressed)
    ->UNIT_AND_ITERATIONS
    ->ArgsProduct({QUANT_BITS_ARGS,
                   {static_cast<long int>(BM_VecSimGeneral::block_size), 5000, 10000}})
    ->ArgNames({"quant_bits", "training_threshold"});

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVSTrain, BM_TrainCompressedAsync, fp16_index_t)
(benchmark::State &st) { TrainAsync(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVSTrain, BM_TrainCompressedAsync)
    ->UNIT_AND_ITERATIONS
    ->ArgsProduct({QUANT_BITS_ARGS,
                   {static_cast<long int>(BM_VecSimGeneral::block_size), 5000, 10000, 50000},
                   {4, 8, 16}})
    ->ArgNames({"quant_bits", "training_threshold", "thread_count"})
    ->MeasureProcessCPUTime();

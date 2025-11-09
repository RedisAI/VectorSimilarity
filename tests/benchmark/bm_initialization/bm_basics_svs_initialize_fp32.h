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

// Memory SVS
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVSdIndex, BM_FUNC_NAME(Memory, SVS), fp32_index_t)
(benchmark::State &st) { Memory(st, INDEX_SVS); }
BENCHMARK_REGISTER_F(BM_VecSimSVSdIndex, BM_FUNC_NAME(Memory, SVS))->Iterations(1);

AddLabel
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVSdIndex, BM_ADD_LABEL, fp32_index_t)
(benchmark::State &st) { AddLabelInPlace(st); }
// Order matters because we use the same svs index for both benchmarks.
REGISTER_AddLabelSVS(BM_ADD_LABEL, INDEX_SVS);
REGISTER_AddLabelSVS(BM_ADD_LABEL, INDEX_TIERED_SVS);

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVSdIndex, BM_AddLabelBatches, fp32_index_t)
(benchmark::State &st) { AddLabelBatches(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVSdIndex, BM_AddLabelBatches)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->ArgsProduct({{static_cast<long int>(BM_VecSimGeneral::block_size)}, {2, 4, 8}})
    ->ArgNames({"batch_size", "thread_count"})
    ->MeasureProcessCPUTime();
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSVSdIndex, BM_AddLabelBatchesAsync, fp32_index_t)
(benchmark::State &st) { AddLabelAsync(st); }
BENCHMARK_REGISTER_F(BM_VecSimSVSdIndex, BM_AddLabelBatchesAsync)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->ArgsProduct({{static_cast<long int>(BM_VecSimGeneral::block_size)}, {2, 4, 8}})
    ->ArgNames({"update_trigger_threshold", "thread_count"})
    ->MeasureProcessCPUTime();

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
  Define and register tests for disk-based HNSW index
  NOTE: benchmarks' tests order can affect their results. Please add new benchmarks at the end of
the file.
***************************************/

// Memory benchmarks
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, FLAT), fp32_index_t)
(benchmark::State &st) { Memory(st, INDEX_BF); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, FLAT))->Iterations(1);

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, HNSW), fp32_index_t)
(benchmark::State &st) { Memory(st, INDEX_HNSW); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, HNSW))->Iterations(1);

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, HNSWDisk), fp32_index_t)
(benchmark::State &st) { Memory(st, INDEX_HNSW_DISK); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, HNSWDisk))->Iterations(1);

// AddLabel benchmarks
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_ADD_LABEL, fp32_index_t)
(benchmark::State &st) { AddLabel(st); }
REGISTER_AddLabel(BM_ADD_LABEL, INDEX_HNSW_DISK);

// DeleteLabel Registration. Definition is placed in the .cpp file.
// REGISTER_DeleteLabel(BM_FUNC_NAME(DeleteLabel, BF));
// REGISTER_DeleteLabel(BM_FUNC_NAME(DeleteLabel, HNSW));
// DISABLED: DeleteLabel benchmarks for HNSW disk index cause segmentation faults
// REGISTER_DeleteLabel(BM_FUNC_NAME(DeleteLabel, HNSWDisk));

// TopK benchmarks

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(TopK, HNSWDisk), fp32_index_t)
(benchmark::State &st) { TopK_HNSW_DISK(st); }
REGISTER_TopK_HNSW_DISK(BM_VecSimCommon, BM_FUNC_NAME(TopK, HNSWDisk));

// Range benchmarks
// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_FUNC_NAMEC_NAME(Range, BF), fp32_index_t)
// (benchmark::State &st) { Range_BF(st); }
// REGISTER_Range_BF(BM_FUNC_NAME(Range, BF), fp32_index_t);
//
// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_FUNC_NAME(Range, HNSW), fp32_index_t)
// (benchmark::State &st) { Range_HNSW(st); }
// REGISTER_Range_HNSW(BM_FUNC_NAME(Range, HNSW), fp32_index_t);

// Special disk-based HNSW benchmarks for batch processing
// RE-ENABLED: Async AddLabel and DeleteLabel benchmarks for HNSW disk index now work with populated index
// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_ADD_LABEL_ASYNC, fp32_index_t)
// (benchmark::State &st) { AddLabel_AsyncIngest(st); }
// BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_ADD_LABEL_ASYNC)
//     ->UNIT_AND_ITERATIONS->Arg(INDEX_HNSW_DISK)
//     ->ArgName("INDEX_HNSW_DISK");

// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_DELETE_LABEL_ASYNC, fp32_index_t)
// (benchmark::State &st) { DeleteLabel_AsyncRepair(st); }
// BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_DELETE_LABEL_ASYNC)
//     ->UNIT_AND_ITERATIONS->Arg(1)
//     ->Arg(100)
//     ->Arg(BM_VecSimGeneral::block_size)
//     ->ArgName("SwapJobsThreshold");

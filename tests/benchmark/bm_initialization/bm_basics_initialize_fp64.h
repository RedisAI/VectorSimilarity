#pragma once
/**************************************
  Define and register tests
  NOTE: benchmarks' tests order can affect their results. Please add new benchmarks at the end of
the file.
***************************************/

// Memory BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, FLAT), fp64_index_t)
(benchmark::State &st) { Memory_FLAT(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, FLAT))->Iterations(1);

// Memory HNSW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, HNSW), fp64_index_t)
(benchmark::State &st) { Memory_HNSW(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, HNSW))->Iterations(1);

// Memory Tiered
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, Tiered), fp64_index_t)
(benchmark::State &st) { Memory_Tiered(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, Tiered))->Iterations(1);

// AddLabel
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_ADD_LABEL, fp64_index_t)
(benchmark::State &st) { AddLabel(st); }
REGISTER_AddLabel(BM_ADD_LABEL, VecSimAlgo_BF);
REGISTER_AddLabel(BM_ADD_LABEL, VecSimAlgo_HNSWLIB);

// DeleteLabel Registration. Definition is placed in the .cpp file.
REGISTER_DeleteLabel(BM_FUNC_NAME(DeleteLabel, BF));
REGISTER_DeleteLabel(BM_FUNC_NAME(DeleteLabel, HNSW));

// TopK BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(TopK, BF), fp64_index_t)
(benchmark::State &st) { TopK_BF(st); }
REGISTER_TopK_BF(BM_VecSimCommon, BM_FUNC_NAME(TopK, BF));

// TopK HNSW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(TopK, HNSW), fp64_index_t)
(benchmark::State &st) { TopK_HNSW(st); }
REGISTER_TopK_HNSW(BM_VecSimCommon, BM_FUNC_NAME(TopK, HNSW));

// TopK Tiered HNSW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(TopK, Tiered), fp64_index_t)
(benchmark::State &st) { TopK_Tiered(st); }
REGISTER_TopK_Tiered(BM_VecSimCommon, BM_FUNC_NAME(TopK, Tiered));

// Range BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_FUNC_NAME(Range, BF), fp64_index_t)
(benchmark::State &st) { Range_BF(st); }
REGISTER_Range_BF(BM_FUNC_NAME(Range, BF));

// Range HSNW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_FUNC_NAME(Range, HNSW), fp64_index_t)
(benchmark::State &st) { Range_HNSW(st); }
REGISTER_Range_HNSW(BM_FUNC_NAME(Range, HNSW));

// Tiered HNSW add/delete benchmarks
REGISTER_AddLabel(BM_ADD_LABEL, VecSimAlgo_TIERED);
REGISTER_DeleteLabel(BM_FUNC_NAME(DeleteLabel, Tiered));

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_ADD_LABEL_ASYNC, fp64_index_t)
(benchmark::State &st) { AddLabel_AsyncIngest(st); }
BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_ADD_LABEL_ASYNC)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_TIERED)
    ->ArgName("VecSimAlgo_TIERED");

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_DELETE_LABEL_ASYNC, fp64_index_t)
(benchmark::State &st) { DeleteLabel_AsyncRepair(st); }
BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_DELETE_LABEL_ASYNC)
    ->UNIT_AND_ITERATIONS->Arg(1)
    ->Arg(100)
    ->Arg(BM_VecSimGeneral::block_size)
    ->ArgName("SwapJobsThreshold");

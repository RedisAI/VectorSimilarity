#pragma once

/**************************************
  Define and register tests for updated index
  NOTE: benchmarks' tests order can affect their results. Please add new benchmarks at the end of
the file.
***************************************/
// Memory BF before
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_BEFORE_FUNC_NAME(Memory, FLAT), fp32_index_t)
(benchmark::State &st) { Memory_FLAT(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_BEFORE_FUNC_NAME(Memory, FLAT))->Iterations(1);

// Updated memory BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, BM_UPDATED_FUNC_NAME(Memory, FLAT), fp32_index_t)
(benchmark::State &st) { Memory_FLAT(st, updated_index_offset); }
BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, BM_UPDATED_FUNC_NAME(Memory, FLAT))->Iterations(1);

// Memory HSNW before
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_BEFORE_FUNC_NAME(Memory, HNSW), fp32_index_t)
(benchmark::State &st) { Memory_HNSW(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_BEFORE_FUNC_NAME(Memory, HNSW))->Iterations(1);

// Updated memory HSNW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, BM_UPDATED_FUNC_NAME(Memory, HNSW), fp32_index_t)
(benchmark::State &st) { Memory_HNSW(st, updated_index_offset); }
BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, BM_UPDATED_FUNC_NAME(Memory, HNSW))->Iterations(1);

// TopK BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_BEFORE_FUNC_NAME(TopK, FLAT), fp32_index_t)
(benchmark::State &st) { TopK_BF(st); }
REGISTER_TopK_BF(BM_VecSimCommon, BM_BEFORE_FUNC_NAME(TopK, FLAT));

// TopK HNSW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_BEFORE_FUNC_NAME(TopK, HNSW), fp32_index_t)
(benchmark::State &st) { TopK_HNSW(st); }
REGISTER_TopK_HNSW(BM_VecSimCommon, BM_BEFORE_FUNC_NAME(TopK, HNSW));

// TopK updated BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, BM_UPDATED_FUNC_NAME(TopK, FLAT), fp32_index_t)
(benchmark::State &st) { TopK_BF(st, updated_index_offset); }
REGISTER_TopK_BF(BM_VecSimUpdatedIndex, BM_UPDATED_FUNC_NAME(TopK, FLAT));

// TopK updated HNSW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, BM_UPDATED_FUNC_NAME(TopK, HNSW), fp32_index_t)
(benchmark::State &st) { TopK_HNSW(st, updated_index_offset); }
REGISTER_TopK_HNSW(BM_VecSimUpdatedIndex, BM_UPDATED_FUNC_NAME(TopK, HNSW));

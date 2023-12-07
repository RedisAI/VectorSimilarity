#pragma once
/**************************************
  Define and register tests
  NOTE: benchmarks' tests order can affect their results. Please add new benchmarks at the end of
the file.
***************************************/

// TopK BF sanity
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSpecialIndex, BM_FUNC_NAME(TopK, BF), fp16_index_t)
(benchmark::State &st) { TopK_BF(st); }
REGISTER_TopK_BF(BM_VecSimSpecialIndex, BM_FUNC_NAME(TopK, BF));

// TopK BF bf16
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSpecialIndex, BM_FUNC_NAME(TopK, FLAT_FP16), fp16_index_t)
(benchmark::State &st) { TopK_BF_special(st, special_index_offset); }
REGISTER_TopK_BF(BM_VecSimSpecialIndex, BM_FUNC_NAME(TopK, FLAT_FP16));

// TopK HNSW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSpecialIndex, BM_FUNC_NAME(TopK, HNSW), fp16_index_t)
(benchmark::State &st) { TopK_HNSW(st); }
REGISTER_TopK_HNSW(BM_VecSimSpecialIndex, BM_FUNC_NAME(TopK, HNSW));

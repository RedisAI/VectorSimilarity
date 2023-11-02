#pragma once
/**************************************
  Define and register tests
  NOTE: benchmarks' tests order can affect their results. Please add new benchmarks at the end of
the file.
***************************************/

// Memory BF GT
// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, FLAT), bf16_index_t)
// (benchmark::State &st) { Memory_FLAT(st); }
// BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, FLAT))->Iterations(1);

// // Memory BF bf16
// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBF16dIndex, BM_FUNC_NAME(Memory, FLAT_BF16), bf16_index_t)
// (benchmark::State &st) { Memory_FLAT(st, bf16_index_offset); }
// BENCHMARK_REGISTER_F(BM_VecSimBF16dIndex, BM_FUNC_NAME(Memory, FLAT_BF16))->Iterations(1);

// // Memory HNSW
// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBF16dIndex, BM_FUNC_NAME(Memory, HNSW), bf16_index_t)
// (benchmark::State &st) { Memory_HNSW(st); }
// BENCHMARK_REGISTER_F(BM_VecSimBF16dIndex, BM_FUNC_NAME(Memory, HNSW))->Iterations(1);

// TopK BF sanity
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBF16dIndex, BM_FUNC_NAME(TopK, BF), bf16_index_t)
(benchmark::State &st) { TopK_BF(st); }
REGISTER_TopK_BF(BM_VecSimBF16dIndex, BM_FUNC_NAME(TopK, BF));

// TopK BF bf16
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBF16dIndex, BM_FUNC_NAME(TopK, FLAT_BF16), bf16_index_t)
(benchmark::State &st) { TopK_BF_bf16(st, bf16_index_offset); }
REGISTER_TopK_BF(BM_VecSimBF16dIndex, BM_FUNC_NAME(TopK, FLAT_BF16));

// // TopK HNSW
// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBF16dIndex, BM_FUNC_NAME(TopK, HNSW), bf16_index_t)
// (benchmark::State &st) { TopK_HNSW(st); }
// REGISTER_TopK_HNSW(BM_VecSimBF16dIndex, BM_FUNC_NAME(TopK, HNSW));

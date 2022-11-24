/**************************************
  Define and register tests for updated index
  NOTE: benchmarks' tests order can affect their results. Please add new benchmarks at the end of
the file.
***************************************/
// Memory BF before
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, Memory_FLAT_fp32, fp32_index_t)
(benchmark::State &st) { Memory_FLAT(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, Memory_FLAT_fp32)->Iterations(1);

// Updated memory BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, Memory_FLAT_updated_fp32, fp32_index_t)
(benchmark::State &st) { Memory_FLAT(st, updated_index_offset); }
BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, Memory_FLAT_updated_fp32)->Iterations(1);

// Memory HSNW before
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, Memory_HNSW_fp32, fp32_index_t)
(benchmark::State &st) { Memory_HNSW(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, Memory_HNSW_fp32)->Iterations(1);

// Updated memory HSNW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, Memory_HNSW_updated_fp32, fp32_index_t)
(benchmark::State &st) { Memory_HNSW(st, updated_index_offset); }
BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, Memory_HNSW_updated_fp32)->Iterations(1);

// TopK BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, TopK_BF_fp32, fp32_index_t)
(benchmark::State &st) { TopK_BF(st); }
REGISTER_TopK_BF(BM_VecSimCommon, TopK_BF_fp32);

// TopK HNSW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, TopK_HNSW_fp32, fp32_index_t)
(benchmark::State &st) { TopK_HNSW(st); }
REGISTER_TopK_HNSW(BM_VecSimCommon, TopK_HNSW_fp32);

// TopK updated BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, TopK_BF_Updated_fp32, fp32_index_t)
(benchmark::State &st) { TopK_BF(st, updated_index_offset); }
REGISTER_TopK_BF(BM_VecSimUpdatedIndex, TopK_BF_Updated_fp32);

// TopK updated HNSW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, TopK_HNSW_Updated_fp32, fp32_index_t)
(benchmark::State &st) { TopK_HNSW(st, updated_index_offset); }
REGISTER_TopK_HNSW(BM_VecSimUpdatedIndex, TopK_HNSW_Updated_fp32);

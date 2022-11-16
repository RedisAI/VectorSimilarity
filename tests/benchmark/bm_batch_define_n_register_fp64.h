
/**************************************
  Define and register benchmarks for batch itertor with index of data type fp64
  NOTE: benchmarks tests order is important. Please add new benchmarks at the end of the file.
***************************************/

// Fixed size batch BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_BatchIterator, BF_FixedBatchSize_fp64, fp64_index_t)
(benchmark::State &st) { BF_FixedBatchSize(st); }
REGISTER_FixedBatchSize(BF_FixedBatchSize_fp64);

// Variable size batch BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_BatchIterator, BF_VariableBatchSize_fp64, fp64_index_t)
(benchmark::State &st) { BF_VariableBatchSize(st); }
REGISTER_VariableBatchSize(BF_VariableBatchSize_fp64);

// Batches to hadoc BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_BatchIterator, BF_BatchesToAdhocBF_fp64, fp64_index_t)
(benchmark::State &st) { BF_BatchesToAdhocBF(st); }
REGISTER_BatchesToAdhocBF(BF_BatchesToAdhocBF_fp64);

// Fixed size batch HNSW
BENCHMARK_TEMPLATE_DEFINE_F(BM_BatchIterator, HNSW_FixedBatchSize_fp64, fp64_index_t)
(benchmark::State &st) { HNSW_FixedBatchSize(st); }
REGISTER_HNSW_FixedBatchSize(HNSW_FixedBatchSize_fp64);

// Variable size batch BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_BatchIterator, HNSW_VariableBatchSize_fp64, fp64_index_t)
(benchmark::State &st) { HNSW_VariableBatchSize(st); }
REGISTER_HNSW_VariableBatchSize(HNSW_VariableBatchSize_fp64);

// Batches to hadoc HSNW
BENCHMARK_TEMPLATE_DEFINE_F(BM_BatchIterator, HNSW_BatchesToAdhocBF_fp64, fp64_index_t)
(benchmark::State &st) { HNSW_BatchesToAdhocBF(st); }
REGISTER_HNSW_BatchesToAdhocBF(HNSW_BatchesToAdhocBF_fp64);

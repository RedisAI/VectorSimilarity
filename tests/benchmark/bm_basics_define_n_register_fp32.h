
/**************************************
  Define and register tests
  NOTE: benchmarks' tests order can affect their results. Please add new benchmarks at the end of
the file.
***************************************/

// Memory BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, Memory_FLAT_fp32, fp32_index_t)
(benchmark::State &st) { Memory_FLAT(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, Memory_FLAT_fp32)->Iterations(1);

// Memory HSNW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, Memory_HNSW_fp32, fp32_index_t)
(benchmark::State &st) { Memory_HNSW(st); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, Memory_HNSW_fp32)->Iterations(1);

// AddVector
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, AddVector_fp32, fp32_index_t)
(benchmark::State &st) { AddVector(st); }
BENCHMARK_REGISTER_F(BM_VecSimBasics, AddVector_fp32)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)
    ->Arg(VecSimAlgo_HNSWLIB);

// DeleteVector
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, DeleteVector_fp32, fp32_index_t)
(benchmark::State &st) {
    if (VecSimAlgo_BF == st.range(0)) {
        DeleteVector<BruteForceIndex<float, float>>(
            reinterpret_cast<BruteForceIndex<float, float> *>(
                BM_VecSimIndex<fp32_index_t>::indices[VecSimAlgo_BF]),
            st);
    } else if (VecSimAlgo_HNSWLIB == st.range(0)) {
        DeleteVector<HNSWIndex<float, float>>(
            reinterpret_cast<HNSWIndex<float, float> *>(
                BM_VecSimIndex<fp32_index_t>::indices[VecSimAlgo_HNSWLIB]),
            st);
    }
}

BENCHMARK_REGISTER_F(BM_VecSimBasics, DeleteVector_fp32)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)
    ->Arg(VecSimAlgo_HNSWLIB);

// TopK BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, TopK_BF_fp32, fp32_index_t)
(benchmark::State &st) { TopK_BF(st); }
REGISTER_TopK_BF(BM_VecSimCommon, TopK_BF_fp32);

// TopK HNSW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, TopK_HNSW_fp32, fp32_index_t)
(benchmark::State &st) { TopK_HNSW(st); }
REGISTER_TopK_HNSW(BM_VecSimCommon, TopK_HNSW_fp32);

// Range BF
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Range_BF_fp32, fp32_index_t)
(benchmark::State &st) { Range_BF(st); }
REGISTER_Range_BF(Range_BF_fp32);

// Range HSNW
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, Range_HNSW_fp32, fp32_index_t)
(benchmark::State &st) { Range_HNSW(st); }
REGISTER_Range_HNSW(Range_HNSW_fp32);
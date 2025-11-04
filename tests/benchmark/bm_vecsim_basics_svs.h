/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once
#include "bm_vecsim_general.h"
#include "bm_macros.h"
#include "VecSim/algorithms/svs/svs_tiered.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "gtest/gtest.h"

template <typename index_type_t>
class BM_VecSimSVSTrain : public BM_VecSimGeneral {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    BM_VecSimSVSTrain()
        : quantBits(VecSimSvsQuant_NONE), data_type(index_type_t::get_index_type()) {
        if (!is_initialized) {
            VecSim_SetLogCallbackFunction(nullptr);
            loadTestVectors(AttachRootPath(test_queries_file));
            is_initialized = true;
        }
    }
    ~BM_VecSimSVSTrain() = default;

    void Train(benchmark::State &st);
    void TrainAsync(benchmark::State &st);

private:
    // Each test instance will have its own quantization bits.
    VecSimSvsQuantBits quantBits;
    VecSimType data_type;

    // Initialize test vectors once
    static bool is_initialized;
    static std::vector<std::vector<data_t>> test_vectors;

    static void InsertToQueries(std::ifstream &input);
    static void loadTestVectors(const std::string &test_file);

    template <bool is_async>
    void runTrainBMIteration(benchmark::State &st, tieredIndexMock &mock_thread_pool,
                             size_t training_threshold);

    static TieredIndexParams
    CreateTieredSVSParams(VecSimParams &svs_params, tieredIndexMock &mock_thread_pool,
                          size_t training_threshold, size_t update_threshold,
                          size_t update_job_wait_time = SVS_DEFAULT_UPDATE_JOB_WAIT_TIME) {
        return TieredIndexParams{
            .jobQueue = &mock_thread_pool.jobQ,
            .jobQueueCtx = mock_thread_pool.ctx,
            .submitCb = tieredIndexMock::submit_callback,
            .primaryIndexParams = &svs_params,
            .specificParams = {.tieredSVSParams =
                                   TieredSVSParams{.trainingTriggerThreshold = training_threshold,
                                                   .updateTriggerThreshold = update_threshold,
                                                   .updateJobWaitTime = update_job_wait_time}}};
    }
    static void verifyNumThreads(TieredSVSIndex<data_t> *tiered_index, size_t expected_num_threads,
                                 size_t expected_capcity) {
        ASSERT_EQ(tiered_index->GetSVSIndex()->getNumThreads(), expected_num_threads)
            << "last reserved threads size mismatch";
        ASSERT_EQ(tiered_index->GetSVSIndex()->getThreadPoolCapacity(), expected_capcity)
            << "thread pool capacity mismatch";
    }
    TieredSVSIndex<data_t> *
    CreateTieredSVSIndex(tieredIndexMock &mock_thread_pool, size_t training_threshold,
                         size_t update_threshold = SVS_VAMANA_DEFAULT_UPDATE_THRESHOLD) {
        SVSParams svs_params = {
            .type = this->data_type,
            .dim = BM_VecSimGeneral::dim,
            .metric = VecSimMetric_Cosine,
            .quantBits = this->quantBits,
            .graph_max_degree = BM_VecSimGeneral::M,
            .construction_window_size = BM_VecSimGeneral::EF_C,
            .num_threads = mock_thread_pool.thread_pool_size,
        };
        VecSimParams params{.algo = VecSimAlgo_SVS, .algoParams = {.svsParams = svs_params}};

        TieredIndexParams tiered_params =
            CreateTieredSVSParams(params, mock_thread_pool, training_threshold, update_threshold);
        auto *tiered_index =
            reinterpret_cast<TieredSVSIndex<data_t> *>(TieredFactory::NewIndex(&tiered_params));
        assert(tiered_index);
        // Set the created tiered index in the index external context (it will take ownership over
        // the index, and we'll need to release the ctx at the end of the test.
        mock_thread_pool.ctx->index_strong_ref.reset(tiered_index);
        // Set numThreads to 1 by default to allow direct calls to SVS addVector() API,
        // which requires exactly 1 thread. When using tiered index addVector API,
        // the thread count is managed internally according to the operation and threadpool
        // capacity, so testing parallelism remains intact.
        size_t params_threadpool_size =
            tiered_params.primaryIndexParams->algoParams.svsParams.num_threads;
        size_t num_threads =
            params_threadpool_size ? params_threadpool_size : mock_thread_pool.thread_pool_size;
        tiered_index->GetSVSIndex()->setNumThreads(num_threads);
        verifyNumThreads(tiered_index, num_threads, num_threads);

        return tiered_index;
    }
};

template <typename index_type_t>
bool BM_VecSimSVSTrain<index_type_t>::is_initialized = false;

// Needs to be explicitly initalized
template <>
std::vector<std::vector<float>> BM_VecSimSVSTrain<fp32_index_t>::test_vectors{};

template <typename index_type_t>
void BM_VecSimSVSTrain<index_type_t>::loadTestVectors(const std::string &test_file) {

    std::ifstream input(test_file, std::ios::binary);
    std::cout << "loading test vectors from " << test_file << std::endl;

    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);

    InsertToQueries(input);
}

template <typename index_type_t>
void BM_VecSimSVSTrain<index_type_t>::InsertToQueries(std::ifstream &input) {

    std::vector<data_t> query(dim);
    for (size_t i = 0; i < N_QUERIES; ++i) {
        ASSERT_TRUE(input.read((char *)query.data(), dim * sizeof(data_t)));
        test_vectors.push_back(query);
    }
    std::cout << "loaded " << test_vectors.size() << " test vectors" << std::endl;
}
template <typename index_type_t>
template <bool is_async>
void BM_VecSimSVSTrain<index_type_t>::runTrainBMIteration(benchmark::State &st,
                                                          tieredIndexMock &mock_thread_pool,
                                                          size_t training_threshold) {
    this->quantBits = static_cast<VecSimSvsQuantBits>(st.range(0));
    auto *tiered_index = CreateTieredSVSIndex(mock_thread_pool, training_threshold);

    VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(tiered_index);
#if HAVE_SVS_LVQ
    ASSERT_EQ(info.tieredInfo.backendInfo.svsInfo.quantBits, this->quantBits);
#else
    if (this->quantBits == VecSimSvsQuant_NONE) {
        ASSERT_EQ(info.tieredInfo.backendInfo.svsInfo.quantBits, this->quantBits);
    } else {
        ASSERT_EQ(info.tieredInfo.backendInfo.svsInfo.quantBits, VecSimSvsQuant_Scalar);
    }
#endif

    auto verify_index_size = [&](size_t expected_tiered_index_size, size_t expected_frontend_size,
                                 size_t expected_backend_size, std::string msg = "") {
        VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(tiered_index);
        auto backend_info = info.tieredInfo.backendCommonInfo;
        auto frontend_info = info.tieredInfo.frontendCommonInfo;
        ASSERT_EQ(info.commonInfo.indexSize, expected_tiered_index_size) << msg;
        ASSERT_EQ(backend_info.indexSize, expected_backend_size) << msg;
        ASSERT_EQ(frontend_info.indexSize, expected_frontend_size) << msg;
    };

    // insert just below training threshold vectors
    for (size_t i = 0; i < training_threshold - 1; ++i) {
        VecSimIndex_AddVector(tiered_index, test_vectors[i].data(), i);
    }
    // Expect frontend index size is (training_threshold - 1) and backend index size is 0
    verify_index_size(training_threshold - 1, training_threshold - 1, 0,
                      (std::ostringstream() << "added training_threshold - 1 ("
                                            << (training_threshold - 1) << ") vectors")
                          .str());

    // start threads
    if constexpr (is_async)
        mock_thread_pool.init_threads();

    // Start timer
    st.ResumeTiming();

    // add one more vector
    VecSimIndex_AddVector(tiered_index, test_vectors[training_threshold - 1].data(),
                          training_threshold - 1);
    if constexpr (is_async)
        mock_thread_pool.thread_pool_wait();

    // Stop timer
    st.PauseTiming();
    // expect backend index size is training_threshold and frontend index size is 0
    verify_index_size(training_threshold, 0, training_threshold,
                      (std::ostringstream()
                       << "added the training_threshold'th (" << training_threshold << ") vector")
                          .str());
    if constexpr (is_async)
        verifyNumThreads(tiered_index, mock_thread_pool.thread_pool_size,
                         mock_thread_pool.thread_pool_size);

    // Resume for next iteration
    st.ResumeTiming();
}

template <typename index_type_t>
void BM_VecSimSVSTrain<index_type_t>::Train(benchmark::State &st) {
    // set write mode to inplace
    auto original_mode = VecSimIndexInterface::asyncWriteMode;
    VecSim_SetWriteMode(VecSim_WriteInPlace);

    auto training_threshold = st.range(1);

    // Ensure we have enough vectors to train.
    ASSERT_GE(N_QUERIES, training_threshold);
    for (auto _ : st) {
        st.PauseTiming();
        // In each iteration create a new index
        auto mock_thread_pool = tieredIndexMock();
        runTrainBMIteration<false>(st, mock_thread_pool, training_threshold);
    }
    // Restore original write mode
    ASSERT_EQ(VecSimIndexInterface::asyncWriteMode, VecSim_WriteInPlace);
    VecSim_SetWriteMode(original_mode);
}

template <typename index_type_t>
void BM_VecSimSVSTrain<index_type_t>::TrainAsync(benchmark::State &st) {
    // ensure mode is async
    ASSERT_EQ(VecSimIndexInterface::asyncWriteMode, VecSim_WriteAsync);

    auto training_threshold = st.range(1);
    int unsigned num_threads = st.range(2);

    if (num_threads > std::thread::hardware_concurrency()) {
        GTEST_SKIP() << "Not enough threads available, skipping test...";
    }

    // Ensure we have enough vectors to train.
    ASSERT_GE(N_QUERIES, training_threshold);
    size_t iter = 0;
    for (auto _ : st) {
        st.PauseTiming();
        // In each iteration create a new index
        auto mock_thread_pool = tieredIndexMock(num_threads);
        ASSERT_EQ(mock_thread_pool.thread_pool_size, num_threads);
        runTrainBMIteration<true>(st, mock_thread_pool, training_threshold);
    }
}

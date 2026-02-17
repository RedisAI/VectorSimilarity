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
#include "VecSim/types/float16.h"
#include "bm_utils.h"

template <typename index_type_t>
class BM_VecSimSVS : public BM_VecSimGeneral {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    BM_VecSimSVS() : data_type(index_type_t::get_index_type()) {
        if (!is_initialized) {
            VecSim_SetLogCallbackFunction(nullptr);
            loadTestVectors(AttachRootPath(test_queries_file));
            if (svs_index_tar_file) {
                base_path = AttachRootPath("tests/benchmark/data");
                extractTarGz(BM_VecSimGeneral::AttachRootPath(svs_index_tar_file), base_path);
            }
            is_initialized = true;
        }
    }
    ~BM_VecSimSVS() = default;

    // Training benchmarks
    void Train(benchmark::State &st);
    void TrainAsync(benchmark::State &st);

    // Loaded index benchmarks

    // Add label directly to svs index one by one
    void AddLabel(benchmark::State &st);
    // Add vector to reach update threshold in tiered index. Measure time to move vectors to svs
    // index.
    void TriggerUpdateTiered(benchmark::State &st);

    // Add vectors to reach training threshold in tiered index, and then add more vectors in
    // parallel to backend training job. Measure time to add new vectors in this scenario.
    void AddVectorsDuringTraining(benchmark::State &st);

    // Deletes an amount of labels from the index that triggers inplace consolidation.
    void RunGC(benchmark::State &st);

private:
    static const char *svs_index_tar_file;
    static std::string base_path;

    // Quant bits can be controlled by the benchmark framework, or by changed by a running benchmark
    // if it explicitly sets it.
    static VecSimSvsQuantBits quantBits;
    VecSimType data_type;

    // Initialize test vectors once
    static bool is_initialized;
    static std::vector<std::vector<data_t>> test_vectors;

    static void InsertToQueries(std::ifstream &input);
    static void loadTestVectors(const std::string &test_file);
    static void extractTarGz(const std::string &filename, const std::string &destination) {

        // Extract tar.gz
        std::string command = "tar -xzf " + filename + " -C " + destination;
        int result = system(command.c_str());
        if (result != 0) {
            throw std::runtime_error("Failed to extract tar.gz file");
        }
    }

    template <bool is_async>
    void runTrainBMIteration(benchmark::State &st, tieredIndexMock &mock_thread_pool,
                             size_t training_threshold);

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

        TieredIndexParams tiered_params = test_utils::CreateTieredSVSParams(
            params, mock_thread_pool, training_threshold, update_threshold);
        auto *tiered_index =
            reinterpret_cast<TieredSVSIndex<data_t> *>(TieredFactory::NewIndex(&tiered_params));
        assert(tiered_index);
        verifyQuantBits(tiered_index, "CreateTieredSVSIndex");

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
        test_utils::verifyNumThreads(tiered_index, num_threads, num_threads,
                                     std::string("CreateTieredSVSIndex"));

        return tiered_index;
    }

    VecSimIndexAbstract<data_t, float> *CreateSVSIndexFromFile(size_t update_threshold,
                                                               size_t num_threads = 1) {
        SVSParams svs_params = {.type = this->data_type,
                                .dim = BM_VecSimGeneral::dim,
                                .metric = VecSimMetric_Cosine,
                                .quantBits = this->quantBits,
                                .graph_max_degree = BM_VecSimGeneral::M,
                                .construction_window_size = BM_VecSimGeneral::EF_C,
                                .num_threads = num_threads};
        VecSimParams params{.algo = VecSimAlgo_SVS, .algoParams = {.svsParams = svs_params}};

        // Load svs index - construct path based on quantBits
        std::string index_path = base_path;
        if (this->quantBits == VecSimSvsQuant_NONE) {
            index_path += "/dbpedia_svs_none";
        } else if (this->quantBits == VecSimSvsQuant_8) {
            index_path += "/dbpedia_svs_8";
        }
        auto *svs_index = reinterpret_cast<VecSimIndexAbstract<data_t, float> *>(
            SVSFactory::NewIndex(index_path, &params));

        return svs_index;
    }
    // training_threshold is not relevant in this case, as the loaded index is already trained.
    TieredSVSIndex<data_t> *CreateTieredSVSIndexFromFile(tieredIndexMock &mock_thread_pool,
                                                         size_t update_threshold) {
        SVSParams svs_params = {.type = this->data_type,
                                .dim = BM_VecSimGeneral::dim,
                                .metric = VecSimMetric_Cosine,
                                .quantBits = this->quantBits,
                                .graph_max_degree = BM_VecSimGeneral::M,
                                .construction_window_size = BM_VecSimGeneral::EF_C,
                                .num_threads = mock_thread_pool.thread_pool_size};
        VecSimParams params{.algo = VecSimAlgo_SVS, .algoParams = {.svsParams = svs_params}};

        // Load svs index
        auto *svs_index =
            CreateSVSIndexFromFile(update_threshold, mock_thread_pool.thread_pool_size);
        TieredIndexParams tiered_params = test_utils::CreateTieredSVSParams(
            params, mock_thread_pool, update_threshold, update_threshold);

        auto *tiered_index = reinterpret_cast<TieredSVSIndex<data_t> *>(
            TieredFactory::TieredSVSFactory::NewIndex<data_t>(&tiered_params, svs_index));
        assert(tiered_index);
        verifyQuantBits(tiered_index, "CreateTieredSVSIndexFromFile");
        // Set the created tiered index in the index external context (it will take ownership over
        // the index, and we'll need to release the ctx at the end of the test.
        mock_thread_pool.ctx->index_strong_ref.reset(tiered_index);
        size_t num_threads = mock_thread_pool.thread_pool_size;
        test_utils::verifyNumThreads(tiered_index, num_threads, num_threads,
                                     std::string("CreateTieredSVSIndexFromFile"));

        return tiered_index;
    }

    void verifyQuantBits(VecSimIndex *index, std::string msg) {
        VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(index);
#if HAVE_SVS_LVQ
        ASSERT_EQ(info.svsInfo.quantBits, this->quantBits) << msg;
#else
        if (this->quantBits == VecSimSvsQuant_NONE) {
            ASSERT_EQ(info.svsInfo.quantBits, this->quantBits) << msg;
        } else {
            ASSERT_EQ(info.svsInfo.quantBits, VecSimSvsQuant_Scalar) << msg;
        }
#endif
    }
};

template <typename index_type_t>
bool BM_VecSimSVS<index_type_t>::is_initialized = false;
template <typename index_type_t>
VecSimSvsQuantBits BM_VecSimSVS<index_type_t>::quantBits = VecSimSvsQuant_NONE;
template <typename index_type_t>
std::string BM_VecSimSVS<index_type_t>::base_path = "tests/benchmark/data";

// Needs to be explicitly initalized
template <>
std::vector<std::vector<float>> BM_VecSimSVS<fp32_index_t>::test_vectors{};
template <>
std::vector<std::vector<vecsim_types::float16>> BM_VecSimSVS<fp16_index_t>::test_vectors{};

template <typename index_type_t>
void BM_VecSimSVS<index_type_t>::loadTestVectors(const std::string &test_file) {

    std::ifstream input(test_file, std::ios::binary);
    std::cout << "loading test vectors from " << test_file << std::endl;

    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);

    InsertToQueries(input);
}

template <typename index_type_t>
void BM_VecSimSVS<index_type_t>::InsertToQueries(std::ifstream &input) {

    std::vector<data_t> query(dim);
    for (size_t i = 0; i < N_QUERIES; ++i) {
        ASSERT_TRUE(input.read((char *)query.data(), dim * sizeof(data_t)));
        test_vectors.push_back(query);
    }
    std::cout << "loaded " << test_vectors.size() << " test vectors" << std::endl;
}

template <typename index_type_t>
template <bool is_async>
void BM_VecSimSVS<index_type_t>::runTrainBMIteration(benchmark::State &st,
                                                     tieredIndexMock &mock_thread_pool,
                                                     size_t training_threshold) {
    this->quantBits = static_cast<VecSimSvsQuantBits>(st.range(0));
    auto *tiered_index = CreateTieredSVSIndex(mock_thread_pool, training_threshold);

    auto verify_index_size = [&](size_t expected_tiered_index_size, size_t expected_frontend_size,
                                 size_t expected_backend_size, std::string msg = "") {
        VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(tiered_index);
        auto backend_info = info.tieredInfo.backendCommonInfo;
        auto frontend_info = info.tieredInfo.frontendCommonInfo;
        ASSERT_EQ(info.commonInfo.indexSize, expected_tiered_index_size) << msg;
        ASSERT_EQ(backend_info.indexSize, expected_backend_size) << msg;
        ASSERT_EQ(frontend_info.indexSize, expected_frontend_size) << msg;
    };

    // Phase 1: Accumulate vectors in the flat buffer (frontend index) without triggering training.
    // Add (training_threshold - 1) vectors to stay below the training threshold.
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

    // Phase 2: Trigger training and backend index initialization.
    // Adding this final vector reaches the training threshold, which triggers:
    // 1. Training of the SVS backend index using all accumulated vectors
    // 2. Transfer of all vectors from flat buffer to the to build the index.
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
        test_utils::verifyNumThreads(tiered_index, mock_thread_pool.thread_pool_size,
                                     mock_thread_pool.thread_pool_size,
                                     std::string("runTrainBMIteration"));

    // Resume for next iteration
    st.ResumeTiming();
}

template <typename index_type_t>
void BM_VecSimSVS<index_type_t>::Train(benchmark::State &st) {
    // set write mode to inplace
    auto original_mode = VecSimIndexInterface::asyncWriteMode;
    VecSim_SetWriteMode(VecSim_WriteInPlace);

    auto training_threshold = st.range(1);

    // Ensure we have enough vectors to train.
    ASSERT_GE(N_QUERIES, training_threshold);
    for (auto _ : st) {
        st.PauseTiming();
        // In each iteration create a new index
        auto mock_thread_pool = tieredIndexMock(1);
        runTrainBMIteration<false>(st, mock_thread_pool, training_threshold);
    }
    // Restore original write mode
    ASSERT_EQ(VecSimIndexInterface::asyncWriteMode, VecSim_WriteInPlace);
    VecSim_SetWriteMode(original_mode);
}

template <typename index_type_t>
void BM_VecSimSVS<index_type_t>::TrainAsync(benchmark::State &st) {
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

template <typename index_type_t>
void BM_VecSimSVS<index_type_t>::AddLabel(benchmark::State &st) {

    size_t label = 0;
    auto index = CreateSVSIndexFromFile(1, 1);
    size_t memory_delta = index->getAllocationSize();
    for (auto _ : st) {
        VecSimIndex_AddVector(index, test_vectors[label].data(), label + N_VECTORS);
        label++;
    };
    memory_delta = index->getAllocationSize() - memory_delta;

    ASSERT_EQ(VecSimIndex_IndexSize(index), N_VECTORS + this->block_size);
    st.counters["memory_per_vector"] =
        benchmark::Counter((double)memory_delta / (double)(this->block_size),
                           benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1024);
}

template <typename index_type_t>
void BM_VecSimSVS<index_type_t>::TriggerUpdateTiered(benchmark::State &st) {
    // ensure mode is async
    ASSERT_EQ(VecSimIndexInterface::asyncWriteMode, VecSim_WriteAsync);

    auto update_threshold = st.range(0);
    int unsigned num_threads = st.range(1);

    if (num_threads > std::thread::hardware_concurrency()) {
        GTEST_SKIP() << "Not enough threads available, skipping test...";
    }

    // Ensure we have enough vectors to update.
    ASSERT_GE(N_QUERIES, update_threshold);

    // In each iteration create a new index
    auto mock_thread_pool = tieredIndexMock(num_threads);
    ASSERT_EQ(mock_thread_pool.thread_pool_size, num_threads);
    auto *tiered_index = CreateTieredSVSIndexFromFile(mock_thread_pool, update_threshold);
    VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(tiered_index);

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
    verify_index_size(
        N_VECTORS, 0, N_VECTORS,
        (std::ostringstream() << "Loaded svs index from file with " << N_VECTORS << " vectors")
            .str());
    for (size_t i = 0; i < update_threshold - 1; ++i) {
        int ret = VecSimIndex_AddVector(tiered_index, test_vectors[i].data(), i + N_VECTORS);
        ASSERT_EQ(ret, 1);
    }
    mock_thread_pool.init_threads();
    for (auto _ : st) {
        // add one more vector to trigger the update
        int ret = VecSimIndex_AddVector(tiered_index, test_vectors[update_threshold - 1].data(),
                                        update_threshold - 1 + N_VECTORS);
        ASSERT_EQ(ret, 1);

        mock_thread_pool.thread_pool_wait();
    }
    verify_index_size(N_VECTORS + update_threshold, 0, N_VECTORS + update_threshold,
                      (std::ostringstream()
                       << "added the update_threshold'th (" << update_threshold << ") vector")
                          .str());
}

template <typename index_type_t>
inline void BM_VecSimSVS<index_type_t>::AddVectorsDuringTraining(benchmark::State &st) {
    // ensure mode is async
    ASSERT_EQ(VecSimIndexInterface::asyncWriteMode, VecSim_WriteAsync);

    auto training_threshold = st.range(0);
    int unsigned num_threads = st.range(1);

    if (num_threads > std::thread::hardware_concurrency()) {
        GTEST_SKIP() << "Not enough threads available, skipping test...";
    }

    // Ensure we have enough vectors to train.
    ASSERT_GE(N_QUERIES, training_threshold);

    // In each iteration create a new index
    auto mock_thread_pool = tieredIndexMock(num_threads);
    ASSERT_EQ(mock_thread_pool.thread_pool_size, num_threads);
    auto *tiered_index = CreateTieredSVSIndex(
        mock_thread_pool, training_threshold,
        1 << 30); // set very high update threshold to avoid updates during this test

    // Add vectors to reach training threshold and trigger training.
    for (size_t i = 0; i < training_threshold; ++i) {
        VecSimIndex_AddVector(tiered_index, test_vectors[i].data(), i);
    }
    mock_thread_pool.init_threads();
    size_t label = training_threshold;

    for (auto _ : st) {
        // While the backend is training, keep adding vectors in parallel to the training job.
        for (size_t i = 0; i < 1000; i++, label++) {
            VecSimIndex_AddVector(tiered_index, test_vectors[label].data(), label);
        }
    }
    mock_thread_pool.thread_pool_join();
}

template <typename index_type_t>
void BM_VecSimSVS<index_type_t>::RunGC(benchmark::State &st) {

    size_t num_deletions = st.range(0);
    auto mock_thread_pool = tieredIndexMock(st.range(1));
    ASSERT_EQ(mock_thread_pool.thread_pool_size, st.range(1));
    auto *tiered_index = CreateTieredSVSIndexFromFile(mock_thread_pool, 1);

    // start threadpool
    mock_thread_pool.init_threads();

    // Delete vectors, not yet triggering consolidation.
    for (size_t i = 0; i < num_deletions; ++i) {
        int ret = VecSimIndex_DeleteVector(tiered_index, i);
        ASSERT_EQ(ret, 1);
    }
    // Num deleted should equal num_deletions.
    VecSimIndexDebugInfo info = VecSimIndex_DebugInfo(tiered_index);
    ASSERT_EQ(info.svsInfo.numberOfMarkedDeletedNodes, num_deletions);
    for (auto _ : st) {
        VecSimTieredIndex_GC(tiered_index);
        mock_thread_pool.thread_pool_wait();
    };
    // num deleted should be 0
    info = VecSimIndex_DebugInfo(tiered_index);
    ASSERT_EQ(info.svsInfo.numberOfMarkedDeletedNodes, 0);

    ASSERT_EQ(VecSimIndex_IndexSize(tiered_index), N_VECTORS - num_deletions);
}

#define UNIT_AND_ITERATIONS Unit(benchmark::kMillisecond)->Iterations(2)

#if HAVE_SVS_LVQ
#define QUANT_BITS_ARGS {VecSimSvsQuant_8, VecSimSvsQuant_4x8_LeanVec}
#define COMPRESSED_TRAINING_THRESHOLD_ARGS                                                         \
    {static_cast<long int>(BM_VecSimGeneral::block_size), 5000, 10000}
#define COMPRESSED_ASYNC_TRAINING_THRESHOLD_ARGS                                                   \
    {static_cast<long int>(BM_VecSimGeneral::block_size), 5000, 10000, 50000}
#else
#define QUANT_BITS_ARGS {VecSimSvsQuant_8}
// Using smaller training TH to avoid long test times without LVQ
#define COMPRESSED_TRAINING_THRESHOLD_ARGS                                                         \
    {static_cast<long int>(BM_VecSimGeneral::block_size), 5000}
#define COMPRESSED_ASYNC_TRAINING_THRESHOLD_ARGS                                                   \
    {static_cast<long int>(BM_VecSimGeneral::block_size), 5000}

#endif

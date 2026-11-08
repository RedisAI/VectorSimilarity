/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 * Licensed under your choice of RSALv2, SSPLv1, or AGPLv3.
 */
#include "gtest/gtest.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/quantization/turboquant_components.h"
#include "VecSim/vec_sim.h"
#include "mock_thread_pool.h"
#include "unit_test_utils.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

namespace {
using namespace vecsim::quantization;
using IndexOwner = std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)>;
class TurboQuantTest : public ::testing::TestWithParam<unsigned> {};

HNSWParams parameters(unsigned bits, size_t dim, const VecSimTurboQuantParams *config = nullptr) {
    return {.type = VecSimType_FLOAT32, .dim = dim, .metric = VecSimMetric_IP,
            .multi = false, .blockSize = 8, .M = 8, .efConstruction = 64, .efRuntime = 128,
            .quantType = bits == 2 ? VecSimQuant_TQ2 : VecSimQuant_TQ3, .quantParams = config};
}
std::shared_ptr<TurboQuant> codec(const std::shared_ptr<VecSimAllocator> &allocator, unsigned bits,
                                 std::span<const float> mean,
                                 TurboQuantExecution execution = TurboQuantExecution::Dispatch) {
    return std::allocate_shared<TurboQuant>(VecsimSTLAllocator<TurboQuant>(allocator), allocator,
        mean.size(), bits, 42, TurboQuantVariant::MSE, execution,
        TurboQuantScalePolicy::ProjectionMatch, TurboQuantBiasPolicy::MeanError, mean);
}
float unalignedFloat(const void *data, size_t offset) {
    float value; std::memcpy(&value, static_cast<const char *>(data) + offset, sizeof(value));
    return value;
}
// An independent bit reader and decoded-score reference, including the full mean error.
std::vector<double> decoded(const TurboQuant &quantizer, const void *blob) {
    const auto *bytes = static_cast<const unsigned char *>(blob);
    std::vector<double> values(quantizer.dim());
    const float scale = unalignedFloat(blob, (quantizer.dim() * quantizer.bits() + 7) / 8);
    for (size_t i = 0; i < values.size(); ++i) {
        unsigned code = 0;
        for (unsigned b = 0; b < quantizer.bits(); ++b) {
            const size_t bit = i * quantizer.bits() + b;
            code |= ((bytes[bit / 8] >> (bit % 8)) & 1) << b;
        }
        values[i] = double(scale) * quantizer.centroids()[code];
    }
    return values;
}
double queryReference(const TurboQuant &quantizer, const void *blob, const float *query) {
    const auto values = decoded(quantizer, blob);
    double score = 1 + unalignedFloat(blob, quantizer.storageBytes() - sizeof(float));
    for (size_t i = 0; i < quantizer.dim(); ++i) {
        double rotated = 0;
        for (size_t j = 0; j < quantizer.dim(); ++j)
            rotated += double(quantizer.rotation()[i * quantizer.dim() + j]) * query[j];
        score -= values[i] * rotated;
        score -= double(quantizer.biasMean()[i]) * query[i];
    }
    return score;
}
double graphReference(const TurboQuant &quantizer, const void *a, const void *b) {
    const auto x = decoded(quantizer, a), y = decoded(quantizer, b);
    double inner = 0, mean_squared = 0;
    for (size_t i = 0; i < quantizer.dim(); ++i) {
        double rotated_mean = 0;
        for (size_t j = 0; j < quantizer.dim(); ++j)
            rotated_mean += double(quantizer.rotation()[i * quantizer.dim() + j]) *
                quantizer.biasMean()[j];
        inner += x[i] * y[i] + (x[i] + y[i]) * rotated_mean;
        mean_squared += double(quantizer.biasMean()[i]) * quantizer.biasMean()[i];
    }
    return 1 - (inner + mean_squared) +
        unalignedFloat(a, quantizer.storageBytes() - sizeof(float)) +
        unalignedFloat(b, quantizer.storageBytes() - sizeof(float));
}
std::string path(unsigned bits, const char *suffix) {
    return ::testing::TempDir() + "vecsim_tq" + std::to_string(bits) + "_" + suffix + ".hnsw";
}

TEST_P(TurboQuantTest, IndependentEncodingAndScoreReferencesWithUnalignedTails) {
    const unsigned bits = GetParam();
    for (size_t dim : {2, 7, 8, 9, 17}) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        std::vector<float> mean(dim), x(dim), y(dim), query(dim), residual(dim);
        for (size_t i = 0; i < dim; ++i) {
            mean[i] = 0.12f * std::sin(float(i + 1));
            x[i] = 0.3f * std::cos(float(i + 2));
            y[i] = 0.2f * std::sin(float(i + 3));
            query[i] = 0.1f * std::cos(float(i + 4));
            residual[i] = x[i] - mean[i];
        }
        auto native = codec(allocator, bits, mean);
        auto scalar = codec(allocator, bits, mean, TurboQuantExecution::Scalar);
        TurboQuantPreprocessors pp(allocator, native), sp(allocator, scalar);
        TurboQuantCalculator calculator(allocator, native), reference(allocator, scalar);
        auto a = pp.preprocessForStorage(x.data(), dim * sizeof(float));
        auto b = pp.preprocessForStorage(y.data(), dim * sizeof(float));
        auto q = pp.preprocessQuery(query.data(), dim * sizeof(float));
        auto sa = sp.preprocessForStorage(x.data(), dim * sizeof(float));
        auto sb = sp.preprocessForStorage(y.data(), dim * sizeof(float));
        auto sq = sp.preprocessQuery(query.data(), dim * sizeof(float));
        EXPECT_EQ(native->storageBytes(), (dim * bits + 7) / 8 + 8);
        EXPECT_EQ(pp.queryBytes(), 4 * dim + 4);
        EXPECT_EQ(std::memcmp(a.get(), sa.get(), native->storageBytes()), 0);
        // Independently recover centroid choices and projection-match scale from raw residual.
        float norm = 0;
        for (float value : residual) norm = std::hypot(norm, value);
        double projection = 0, code_mean = 0, residual_mean = 0;
        std::vector<unsigned> expected_codes(dim);
        for (size_t i = 0; i < dim; ++i) {
            float rotated = 0;
            for (size_t j = 0; j < dim; ++j)
                rotated += native->rotation()[i * dim + j] * (residual[j] / norm);
            unsigned expected_code = 0;
            for (size_t c = 1; c < native->centroids().size(); ++c)
                if (rotated > (native->centroids()[c - 1] + native->centroids()[c]) * 0.5f)
                    expected_code = c;
            expected_codes[i] = expected_code;
            projection += double(rotated) * native->centroids()[expected_code];
            code_mean += double(native->centroids()[expected_code]) * native->biasRotatedMean()[i];
            residual_mean += double(residual[i]) * mean[i];
        }
        const size_t code_bytes = (dim * bits + 7) / 8;
        const float expected_scale = float(double(norm) / projection);
        EXPECT_FLOAT_EQ(unalignedFloat(a.get(), code_bytes), expected_scale);
        EXPECT_FLOAT_EQ(unalignedFloat(a.get(), code_bytes + 4),
                        float(double(expected_scale) * code_mean - residual_mean));
        const auto decoded_a = decoded(*native, a.get());
        for (size_t i = 0; i < dim; ++i)
            EXPECT_DOUBLE_EQ(decoded_a[i], double(expected_scale) * native->centroids()[expected_codes[i]]);
        // A one-byte offset exercises the real unaligned record layout and full stride.
        std::vector<unsigned char> unaligned(native->storageBytes() + 1);
        std::memcpy(unaligned.data() + 1, a.get(), native->storageBytes());
        EXPECT_NEAR(calculator.calcDistanceForQuery(unaligned.data() + 1, q.get(), dim),
                    queryReference(*native, a.get(), query.data()), 3e-6);
        EXPECT_NEAR(calculator.calcDistance(a.get(), b.get(), dim),
                    graphReference(*native, a.get(), b.get()), 3e-6);
        EXPECT_NEAR(calculator.calcDistanceForQuery(a.get(), q.get(), dim),
                    reference.calcDistanceForQuery(sa.get(), sq.get(), dim), 3e-6);
        EXPECT_NEAR(calculator.calcDistance(a.get(), b.get(), dim),
                    reference.calcDistance(sa.get(), sb.get(), dim), 3e-6);
        auto zero_residual = pp.preprocessForStorage(mean.data(), dim * sizeof(float));
        EXPECT_EQ(unalignedFloat(zero_residual.get(), code_bytes), 0);
        EXPECT_NEAR(calculator.calcDistanceForQuery(zero_residual.get(), q.get(), dim),
                    queryReference(*native, zero_residual.get(), query.data()), 3e-6);
    }
}

TEST_P(TurboQuantTest, PublicFactoryMetadataRangePersistenceMutationAndOwnership) {
    const unsigned bits = GetParam();
    constexpr size_t dim = 9;
    std::vector<float> mean(dim, 0.1f), query(dim, 0.05f);
    VecSimTurboQuantParams config{.mean = mean.data(), .seed = 0};
    auto hp = parameters(bits, dim, &config);
    auto params = CreateParams(hp);
    IndexOwner index(VecSimIndex_New(&params), VecSimIndex_Free);
    ASSERT_NE(index, nullptr);
    auto *hnsw = dynamic_cast<HNSWIndex<float, float> *>(index.get());
    ASSERT_NE(hnsw, nullptr);
    auto allocator = hnsw->getAllocator();
    auto retained_codec = hnsw->getTurboQuant();
    EXPECT_EQ(retained_codec->seed(), 0);
    EXPECT_EQ(hnsw->getStoredDataSize(), (dim * bits + 7) / 8 + 8);
    const auto info = hnsw->debugInfo();
    EXPECT_EQ(info.hnswInfo.turboQuantType, hp.quantType);
    EXPECT_EQ(info.hnswInfo.quantizedVectorBytes, hnsw->getStoredDataSize());
    EXPECT_EQ(info.commonInfo.memory, allocator->getAllocationSize());
    EXPECT_GE(VecSimIndex_EstimateInitialSize(&params), index->getAllocationSize());
    EXPECT_GE(VecSimIndex_EstimateElementSize(&params), hnsw->getStoredDataSize());
    std::array<std::array<float, dim>, 12> rows{};
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < dim; ++j) rows[i][j] = 0.2f * std::sin(float(i * dim + j + 1));
        EXPECT_EQ(VecSimIndex_AddVector(index.get(), rows[i].data(), i), 1);
    }
    const double expected = queryReference(*retained_codec, hnsw->getDataByInternalId(0), query.data());
    EXPECT_NEAR(VecSimIndex_GetDistanceFrom_Unsafe(index.get(), 0, query.data()), expected, 3e-6);
    for (double radius : {expected - 1e-3, expected + 1e-3}) {
        auto *reply = VecSimIndex_RangeQuery(index.get(), query.data(), radius, nullptr, BY_SCORE);
        auto *it = VecSimQueryReply_GetIterator(reply);
        bool found = false;
        while (VecSimQueryReply_IteratorHasNext(it)) {
            const auto *item = VecSimQueryReply_IteratorNext(it);
            if (VecSimQueryResult_GetId(item) == 0) found = true;
            EXPECT_LE(VecSimQueryResult_GetScore(item), radius + 1e-6);
        }
        EXPECT_EQ(found, radius > expected);
        VecSimQueryReply_IteratorFree(it); VecSimQueryReply_Free(reply);
    }
    const auto filename = path(bits, "roundtrip");
    hnsw->saveIndex(filename);
    { std::ifstream in(filename, std::ios::binary);
      EXPECT_EQ(HNSWSerializer::ReadVersion(in), HNSWSerializer::EncodingVersion::V6); }
    IndexOwner restored(HNSWFactory::NewIndex(filename), VecSimIndex_Free);
    ASSERT_NE(restored, nullptr);
    auto *loaded = dynamic_cast<HNSWIndex<float, float> *>(restored.get());
    ASSERT_NE(loaded, nullptr);
    for (size_t i = 0; i < rows.size(); ++i) {
        EXPECT_EQ(std::memcmp(hnsw->getDataByInternalId(i), loaded->getDataByInternalId(i),
                              hnsw->getStoredDataSize()), 0);
        EXPECT_FLOAT_EQ(VecSimIndex_GetDistanceFrom_Unsafe(index.get(), i, query.data()),
                        VecSimIndex_GetDistanceFrom_Unsafe(restored.get(), i, query.data()));
    }
    for (auto getter : {&TurboQuant::rotation, &TurboQuant::centroids, &TurboQuant::biasMean,
                        &TurboQuant::biasRotatedMean}) {
        const auto a = (retained_codec.get()->*getter)();
        const auto b = (loaded->getTurboQuant().get()->*getter)();
        EXPECT_EQ(a.size(), b.size());
        EXPECT_TRUE(std::equal(a.begin(), a.end(), b.begin()));
    }
    EXPECT_EQ(VecSimIndex_AddVector(restored.get(), rows[0].data(), 99), 1);
    EXPECT_EQ(restored->relabelVector(99, 100), VecSimRelabel_OK);
    EXPECT_EQ(VecSimIndex_DeleteVector(restored.get(), 100), 1);
    EXPECT_TRUE(loaded->checkIntegrity().valid_state);
    std::remove(filename.c_str());
    index.reset();
    EXPECT_GT(allocator->getAllocationSize(), sizeof(VecSimAllocator));
    retained_codec.reset();
    EXPECT_EQ(allocator->getAllocationSize(), sizeof(VecSimAllocator));
}

TEST_P(TurboQuantTest, CosinePersistenceRequiresNormalizedInputOptIn) {
    const unsigned bits = GetParam();
    constexpr size_t dim = 9;
    auto hp = parameters(bits, dim);
    hp.metric = VecSimMetric_Cosine;
    auto params = CreateParams(hp);
    IndexOwner index(HNSWFactory::NewIndex(&params, true), VecSimIndex_Free);
    ASSERT_NE(index, nullptr);
    auto *backend = dynamic_cast<HNSWIndex<float, float> *>(index.get());
    ASSERT_NE(backend, nullptr);
    std::array<float, dim> row{3, 4}, query{4, 3};
    VecSim_Normalize(row.data(), dim, VecSimType_FLOAT32);
    VecSim_Normalize(query.data(), dim, VecSimType_FLOAT32);
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), row.data(), 0), 1);
    const auto filename = path(bits, "cosine");
    backend->saveIndex(filename);
    IndexOwner default_loaded(HNSWFactory::NewIndex(filename), VecSimIndex_Free);
    EXPECT_EQ(default_loaded, nullptr);
    IndexOwner raw_loaded(HNSWFactory::NewIndex(filename, false), VecSimIndex_Free);
    EXPECT_EQ(raw_loaded, nullptr);
    IndexOwner restored(HNSWFactory::NewIndex(filename, true), VecSimIndex_Free);
    ASSERT_NE(restored, nullptr);
    auto *loaded = dynamic_cast<HNSWIndex<float, float> *>(restored.get());
    ASSERT_NE(loaded, nullptr);

    // Opting in promises normalized queries and future inserts; the backend does not
    // normalize either input itself. Use the same normalized rows before and after load.
    row = {4, -3};
    VecSim_Normalize(row.data(), dim, VecSimType_FLOAT32);
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), row.data(), 1), 1);
    ASSERT_EQ(VecSimIndex_AddVector(restored.get(), row.data(), 1), 1);
    for (size_t label : {0U, 1U}) {
        EXPECT_EQ(std::memcmp(backend->getDataByInternalId(label),
                              loaded->getDataByInternalId(label), backend->getStoredDataSize()), 0);
        EXPECT_FLOAT_EQ(VecSimIndex_GetDistanceFrom_Unsafe(index.get(), label, query.data()),
                        VecSimIndex_GetDistanceFrom_Unsafe(restored.get(), label, query.data()));
    }
    const auto *stored = loaded->getDataByInternalId(0);
    const double normalized_score = queryReference(*loaded->getTurboQuant(), stored, query.data());
    EXPECT_NEAR(VecSimIndex_GetDistanceFrom_Unsafe(restored.get(), 0, query.data()),
                normalized_score, 3e-6);
    auto *reply = VecSimIndex_TopKQuery(restored.get(), query.data(), 2, nullptr, BY_SCORE);
    EXPECT_EQ(VecSimQueryReply_GetCode(reply), VecSim_QueryReply_OK);
    auto *iterator = VecSimQueryReply_GetIterator(reply);
    size_t found = 0;
    while (VecSimQueryReply_IteratorHasNext(iterator)) {
        const auto *item = VecSimQueryReply_IteratorNext(iterator);
        EXPECT_FLOAT_EQ(VecSimQueryResult_GetScore(item), VecSimIndex_GetDistanceFrom_Unsafe(
            restored.get(), VecSimQueryResult_GetId(item), query.data()));
        ++found;
    }
    EXPECT_EQ(found, 2);
    VecSimQueryReply_IteratorFree(iterator);
    VecSimQueryReply_Free(reply);
    for (auto &value : query) value *= 2;
    const double raw_score = queryReference(*loaded->getTurboQuant(), stored, query.data());
    EXPECT_GT(std::abs(raw_score - normalized_score), 0.1);
    EXPECT_NEAR(VecSimIndex_GetDistanceFrom_Unsafe(restored.get(), 0, query.data()), raw_score, 3e-6);
    EXPECT_TRUE(loaded->checkIntegrity().valid_state);
    std::remove(filename.c_str());
}

TEST_P(TurboQuantTest, UnsupportedConfigurationsAndUnsafeRowsAreRejected) {
    const unsigned bits = GetParam();
    auto hp = parameters(bits, 9);
    for (auto type : {VecSimType_FLOAT16, VecSimType_FLOAT64, VecSimType_INT8}) {
        hp.type = type; auto params = CreateParams(hp);
        EXPECT_EQ(VecSimIndex_New(&params), nullptr);
        EXPECT_EQ(VecSimIndex_EstimateInitialSize(&params), SIZE_MAX);
        EXPECT_EQ(VecSimIndex_EstimateElementSize(&params), SIZE_MAX);
    }
    hp.type = VecSimType_FLOAT32; hp.metric = VecSimMetric_L2;
    auto invalid = CreateParams(hp);
    EXPECT_EQ(VecSimIndex_New(&invalid), nullptr);
    EXPECT_EQ(VecSimIndex_EstimateInitialSize(&invalid), SIZE_MAX);
    hp.metric = VecSimMetric_IP;
    auto params = CreateParams(hp);
    IndexOwner index(VecSimIndex_New(&params), VecSimIndex_Free);
    std::vector<float> row(9, 0.1f);
    row[0] = std::numeric_limits<float>::infinity();
    EXPECT_THROW(VecSimIndex_AddVector(index.get(), row.data(), 0), std::invalid_argument);
    EXPECT_EQ(index->indexSize(), 0);
    row[0] = std::numeric_limits<float>::max();
    EXPECT_THROW(VecSimIndex_AddVector(index.get(), row.data(), 0), std::overflow_error);
    EXPECT_EQ(index->indexSize(), 0);
}

struct WriteModeGuard {
    VecSimWriteMode previous = VecSimIndexInterface::asyncWriteMode;
    ~WriteModeGuard() { VecSim_SetWriteMode(previous); }
};
struct TestPool : tieredIndexMock {
    TestPool() : tieredIndexMock(0) {}
    ~TestPool() {
        while (!jobQ.empty()) thread_iteration();
        reset_ctx(); // The mock destructor otherwise dereferences an empty index context.
    }
};

TEST_P(TurboQuantTest, TieredLiveMeanTrainingRetriesAndMigratesInBothWriteModes) {
    const unsigned bits = GetParam();
    WriteModeGuard mode_guard;
    for (auto mode : {VecSim_WriteAsync, VecSim_WriteInPlace}) {
        for (auto metric : {VecSimMetric_IP, VecSimMetric_Cosine}) {
            VecSim_SetWriteMode(mode);
            TestPool pool;
            constexpr size_t dim = 9;
            auto hp = parameters(bits, dim);
            hp.metric = metric;
            auto primary = CreateParams(hp);
            TieredIndexParams tiered_params{
                .jobQueue = &pool.jobQ, .jobQueueCtx = pool.ctx,
                .submitCb = tieredIndexMock::submit_callback, .flatBufferLimit = 1,
                .primaryIndexParams = &primary,
                .specificParams = {TieredHNSWParams{.QuantNormalizationSetSize = 4}}};
            auto *base = TieredFactory::NewIndex(&tiered_params);
            pool.ctx->index_strong_ref = std::shared_ptr<VecSimIndex>(base, VecSimIndex_Free);
            ASSERT_NE(base, nullptr);
            auto *tiered = dynamic_cast<TieredHNSWIndex<float, float> *>(base);
            ASSERT_NE(tiered, nullptr);
            auto *backend = tiered->getBackendForTests();
            ASSERT_TRUE(backend->isTurboQuantTraining());
            std::array<std::array<float, dim>, 5> rows{};
            for (size_t i = 0; i < rows.size(); ++i)
                for (size_t j = 0; j < dim; ++j) rows[i][j] = 0.3f * std::sin(float(i * dim + j + 1));
            VecSimIndex_AddVector(base, rows[0].data(), 10);
            VecSimIndex_AddVector(base, rows[1].data(), 20);
            EXPECT_EQ(VecSimIndex_AddVector(base, rows[2].data(), 10), 0);
            EXPECT_EQ(VecSimIndex_DeleteVector(base, 20), 1);
            EXPECT_EQ(base->relabelVector(10, 11), VecSimRelabel_OK);
            EXPECT_EQ(backend->turboQuantTrainingCount(), 1);
            EXPECT_EQ(backend->indexSize(), 0);
            EXPECT_TRUE(pool.jobQ.empty());
            EXPECT_THROW(backend->saveIndex(path(bits, "untrained")), std::runtime_error);
            VecSimIndex_AddVector(base, rows[1].data(), 20);
            VecSimIndex_AddVector(base, rows[3].data(), 30);
            tiered->setBeforeQuantizationFinalizationHook([] { throw std::bad_alloc(); });
            EXPECT_THROW(VecSimIndex_AddVector(base, rows[4].data(), 40), std::bad_alloc);
            EXPECT_TRUE(backend->isTurboQuantTraining());
            EXPECT_EQ(backend->turboQuantTrainingCount(), 4);
            EXPECT_EQ(backend->indexSize(), 0);
            EXPECT_TRUE(pool.jobQ.empty());
            for (float value : backend->getTurboQuant()->biasMean()) EXPECT_EQ(value, 0);
            // A failed preparation/publication attempt is retried by the next writer.
            tiered->setBeforeQuantizationFinalizationHook({});
            EXPECT_EQ(VecSimIndex_AddVector(base, rows[4].data(), 40), 0);
            EXPECT_FALSE(backend->isTurboQuantTraining());
            if (mode == VecSim_WriteAsync) {
                EXPECT_EQ(pool.jobQ.size(), 4);
                while (!pool.jobQ.empty()) pool.thread_iteration();
            } else {
                EXPECT_TRUE(pool.jobQ.empty());
            }
            ASSERT_EQ(backend->indexSize(), 4);
            EXPECT_EQ(base->statisticInfo().flatBufferSize, 0);
            std::vector<double> expected_mean(dim);
            for (size_t i : {size_t(1), size_t(2), size_t(3), size_t(4)}) {
                auto row = rows[i];
                if (metric == VecSimMetric_Cosine) VecSim_Normalize(row.data(), dim, VecSimType_FLOAT32);
                for (size_t j = 0; j < dim; ++j) expected_mean[j] += double(row[j]) / 4;
            }
            for (size_t j = 0; j < dim; ++j)
                EXPECT_NEAR(backend->getTurboQuant()->biasMean()[j], expected_mean[j], 3e-8);
            EXPECT_TRUE(backend->checkIntegrity().valid_state);
            const auto saved_mean = std::vector<float>(backend->getTurboQuant()->biasMean().begin(),
                                                       backend->getTurboQuant()->biasMean().end());
            // Removing every row and adding again never restarts a fitted backend.
            for (size_t label : {11, 20, 30, 40}) VecSimIndex_DeleteVector(base, label);
            while (!pool.jobQ.empty()) pool.thread_iteration();
            VecSimIndex_AddVector(base, rows[0].data(), 50);
            while (!pool.jobQ.empty()) pool.thread_iteration();
            EXPECT_FALSE(backend->isTurboQuantTraining());
            EXPECT_TRUE(std::equal(saved_mean.begin(), saved_mean.end(),
                                   backend->getTurboQuant()->biasMean().begin()));
        }
    }
}

TEST_P(TurboQuantTest, MultiValueTrainingAndSubsequentUpdatesKeepOneFittedMean) {
    const unsigned bits = GetParam();
    WriteModeGuard guard; VecSim_SetWriteMode(VecSim_WriteInPlace);
    TestPool pool;
    constexpr size_t dim = 7;
    auto hp = parameters(bits, dim); hp.multi = true;
    auto primary = CreateParams(hp);
    TieredIndexParams tp{.jobQueue = &pool.jobQ, .jobQueueCtx = pool.ctx,
        .submitCb = tieredIndexMock::submit_callback, .flatBufferLimit = 1,
        .primaryIndexParams = &primary,
        .specificParams = {TieredHNSWParams{.QuantNormalizationSetSize = 3}}};
    auto *base = TieredFactory::NewIndex(&tp);
    pool.ctx->index_strong_ref = std::shared_ptr<VecSimIndex>(base, VecSimIndex_Free);
    ASSERT_NE(base, nullptr);
    auto *tiered = dynamic_cast<TieredHNSWIndex<float, float> *>(base);
    auto *backend = tiered->getBackendForTests();
    std::array<float, dim> a{}, b{}, c{};
    for (size_t i = 0; i < dim; ++i) { a[i] = 0.1f * i; b[i] = -0.1f * i; c[i] = 0.03f * i; }
    VecSimIndex_AddVector(base, a.data(), 10);
    VecSimIndex_AddVector(base, b.data(), 10);
    EXPECT_EQ(VecSimIndex_DeleteVector(base, 10), 2);
    EXPECT_EQ(backend->turboQuantTrainingCount(), 0);
    VecSimIndex_AddVector(base, a.data(), 11);
    std::array<float, 2 * dim> replacement{};
    std::copy(b.begin(), b.end(), replacement.begin());
    std::copy(c.begin(), c.end(), replacement.begin() + dim);
    EXPECT_EQ(VecSimIndex_UpdateVectors(base, 11, replacement.data(), 2), VecSimUpdate_OK);
    EXPECT_EQ(backend->turboQuantTrainingCount(), 2);
    EXPECT_EQ(backend->indexSize(), 0);
    VecSimIndex_AddVector(base, a.data(), 12);
    EXPECT_FALSE(backend->isTurboQuantTraining());
    EXPECT_EQ(backend->indexSize(), 3);
    for (size_t i = 0; i < dim; ++i)
        EXPECT_NEAR(backend->getTurboQuant()->biasMean()[i], (double(a[i]) + b[i] + c[i]) / 3, 2e-8);
    EXPECT_EQ(VecSimIndex_UpdateVectors(base, 11, a.data(), 1), VecSimUpdate_OK);
    EXPECT_EQ(base->indexSize(), 2);
    EXPECT_TRUE(backend->checkIntegrity().valid_state);
}

TEST_P(TurboQuantTest, QueriesRemainIsolatedDuringFittedMeanPublication) {
    const unsigned bits = GetParam();
    WriteModeGuard guard; VecSim_SetWriteMode(VecSim_WriteInPlace);
    TestPool pool;
    constexpr size_t dim = 9;
    auto hp = parameters(bits, dim);
    auto primary = CreateParams(hp);
    TieredIndexParams tp{.jobQueue = &pool.jobQ, .jobQueueCtx = pool.ctx,
        .submitCb = tieredIndexMock::submit_callback, .flatBufferLimit = 1,
        .primaryIndexParams = &primary,
        .specificParams = {TieredHNSWParams{.QuantNormalizationSetSize = 2}}};
    auto *base = TieredFactory::NewIndex(&tp);
    pool.ctx->index_strong_ref = std::shared_ptr<VecSimIndex>(base, VecSimIndex_Free);
    ASSERT_NE(base, nullptr);
    auto *tiered = dynamic_cast<TieredHNSWIndex<float, float> *>(base);
    std::array<float, dim> a{}, b{};
    for (size_t i = 0; i < dim; ++i) { a[i] = 0.2f * std::sin(float(i)); b[i] = -a[i] / 2; }
    VecSimIndex_AddVector(base, a.data(), 1);
    // Force an actual query to overlap preparation, then exercise independent query buffers
    // from two readers while the writer publishes and migrates. Useful under TSan as well.
    std::atomic<bool> stop{false};
    std::atomic<size_t> calls{0}, failures{0};
    auto query = [&] {
        while (!stop.load()) {
            auto *reply = VecSimIndex_TopKQuery(base, a.data(), 2, nullptr, BY_SCORE);
            if (VecSimQueryReply_GetCode(reply) != VecSim_QueryReply_OK) ++failures;
            auto *it = VecSimQueryReply_GetIterator(reply);
            while (VecSimQueryReply_IteratorHasNext(it))
                if (!std::isfinite(VecSimQueryResult_GetScore(VecSimQueryReply_IteratorNext(it))))
                    ++failures;
            VecSimQueryReply_IteratorFree(it); VecSimQueryReply_Free(reply); ++calls;
        }
    };
    std::thread first(query), second(query);
    tiered->setBeforeQuantizationFinalizationHook([&] { while (calls.load() < 10) std::this_thread::yield(); });
    VecSimIndex_AddVector(base, b.data(), 2);
    stop.store(true); first.join(); second.join();
    EXPECT_GE(calls.load(), 10); EXPECT_EQ(failures.load(), 0);
    auto *backend = tiered->getBackendForTests();
    EXPECT_EQ(backend->indexSize(), 2);
    EXPECT_TRUE(backend->checkIntegrity().valid_state);
}

TEST_P(TurboQuantTest, RejectsTruncatedOrCorruptFittedStateAndGraph) {
    const unsigned bits = GetParam();
    auto hp = parameters(bits, 9); auto params = CreateParams(hp);
    IndexOwner index(VecSimIndex_New(&params), VecSimIndex_Free);
    auto *backend = dynamic_cast<HNSWIndex<float, float> *>(index.get());
    ASSERT_NE(backend, nullptr);
    std::array<float, 9> row{}; row[0] = 0.5f;
    VecSimIndex_AddVector(index.get(), row.data(), 1);
    const auto filename = path(bits, "corrupt");
    backend->saveIndex(filename);
    std::ifstream source(filename, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
    const size_t fitted = sizeof(HNSWSerializer::EncodingVersion) + sizeof(VecSimAlgo) +
        sizeof(size_t) + sizeof(VecSimType) + sizeof(VecSimMetric) + sizeof(size_t) +
        sizeof(bool) + sizeof(size_t) + sizeof(VecSimQuantType);
    auto reject = [&](std::string corrupt) {
        { std::ofstream out(filename, std::ios::binary | std::ios::trunc);
          out.write(corrupt.data(), corrupt.size()); }
        EXPECT_THROW({ IndexOwner invalid(HNSWFactory::NewIndex(filename), VecSimIndex_Free); },
                     std::runtime_error);
    };
    for (size_t length : {size_t(0), fitted, fitted + 16, bytes.size() / 2, bytes.size() - 1})
        reject(bytes.substr(0, length));
    auto unknown = bytes; uint32_t bad_policy = 999;
    std::memcpy(unknown.data() + fitted + 3 * sizeof(uint32_t), &bad_policy, sizeof(bad_policy));
    reject(unknown);
    auto bad_length = bytes; uint64_t huge = UINT64_MAX;
    std::memcpy(bad_length.data() + fitted + 5 * sizeof(uint32_t) + sizeof(uint64_t), &huge, sizeof(huge));
    reject(bad_length);
    auto nonfinite = bytes; const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(nonfinite.data() + fitted + 5 * sizeof(uint32_t) + 2 * sizeof(uint64_t), &nan, sizeof(nan));
    reject(nonfinite);
    const size_t dim = 9;
    const size_t fields = fitted + 5 * sizeof(uint32_t) + 5 * sizeof(uint64_t) +
        (dim * dim + (size_t(1) << bits) + 2 * dim + 1) * sizeof(float);
    const size_t count_offset = fields + 4 * sizeof(size_t) + 2 * sizeof(double);
    const size_t max_offset = count_offset + 2 * sizeof(size_t);
    const size_t entry_offset = max_offset + sizeof(size_t);
    const size_t metadata = entry_offset + sizeof(idType);
    auto bad_max = bytes;
    size_t saved_max = 0; std::memcpy(&saved_max, bytes.data() + max_offset, sizeof(saved_max));
    ++saved_max; std::memcpy(bad_max.data() + max_offset, &saved_max, sizeof(saved_max));
    reject(bad_max);
    auto bad_flags = bytes; bad_flags[metadata + sizeof(labelType)] = IN_PROCESS;
    reject(bad_flags);
    // Build a two-node fixture whose high-level outgoing/incoming edge refers to
    // a level-zero node. All IDs and list sizes are valid, so level checks are essential.
    for (bool incoming_edge : {false, true}) {
        auto graph = bytes.substr(0, metadata);
        auto patch = [&](size_t offset, const auto &value) {
            std::memcpy(graph.data() + offset, &value, sizeof(value));
        };
        patch(count_offset, size_t(2)); patch(max_offset, size_t(1)); patch(entry_offset, idType(0));
        auto append = [&](const auto &value) {
            graph.append(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        for (labelType label : {labelType(0), labelType(1)}) { append(label); append(elementFlags(0)); }
        const auto record = bytes.substr(metadata + sizeof(labelType) + sizeof(elementFlags),
                                         backend->getStoredDataSize());
        graph += record; graph += record;
        append(unsigned(2));
        append(size_t(1)); // node 0 owns levels 0 and 1
        append(linkListSize(1)); append(idType(1)); append(unsigned(0)); // level 0
        if (incoming_edge) { append(linkListSize(0)); append(unsigned(1)); append(idType(1)); }
        else { append(linkListSize(1)); append(idType(1)); append(unsigned(0)); }
        append(size_t(0)); // node 1 owns level 0 only
        append(linkListSize(1)); append(idType(0)); append(unsigned(0));
        reject(graph);
        if (!incoming_edge) {
            // Keep the same node levels, remove its high edge, and choose the level-zero
            // node as the advertised level-one entrypoint instead.
            const size_t high_level = metadata + 2 * (sizeof(labelType) + sizeof(elementFlags)) +
                2 * backend->getStoredDataSize() + sizeof(unsigned) + sizeof(size_t) +
                sizeof(linkListSize) + sizeof(idType) + sizeof(unsigned);
            graph.erase(high_level + sizeof(linkListSize), sizeof(idType));
            const linkListSize no_links = 0;
            std::memcpy(graph.data() + high_level, &no_links, sizeof(no_links));
            const idType wrong_entrypoint = 1;
            std::memcpy(graph.data() + entry_offset, &wrong_entrypoint, sizeof(wrong_entrypoint));
            reject(graph);
        }
    }
    std::remove(filename.c_str());
}

// Invalid rows at either position must be rejected before a multi label is removed.
TEST_P(TurboQuantTest, ValidatesEveryReplacementBeforeDeletingExistingRows) {
    const unsigned bits = GetParam();
    constexpr size_t dim = 7;
    const auto original = std::array<float, dim>{0.1f, 0.2f, -0.3f, 0.4f, 0.2f, -0.1f, 0.3f};
    auto hp = parameters(bits, dim); hp.multi = true;
    auto params = CreateParams(hp);
    IndexOwner standalone(VecSimIndex_New(&params), VecSimIndex_Free);
    ASSERT_NE(standalone, nullptr);
    VecSimIndex_AddVector(standalone.get(), original.data(), 10);
    auto *native = dynamic_cast<HNSWIndex<float, float> *>(standalone.get());
    const auto stored = native->getStoredVectorDataByLabel(10);
    auto check = [&](VecSimIndex *target, const std::function<void()> &unchanged) {
        for (size_t bad_row : {size_t(0), size_t(1)}) {
            for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::max()}) {
                std::array<float, 2 * dim> replacement{};
                std::copy(original.begin(), original.end(), replacement.begin());
                std::copy(original.begin(), original.end(), replacement.begin() + dim);
                replacement[bad_row * dim] = invalid;
                EXPECT_ANY_THROW(VecSimIndex_UpdateVectors(target, 10, replacement.data(), 2));
                unchanged();
            }
        }
        EXPECT_THROW(VecSimIndex_UpdateVectors(target, 10, nullptr, 2), std::invalid_argument);
        unchanged();
        EXPECT_THROW(VecSimIndex_UpdateVectors(target, 10, original.data(), SIZE_MAX), std::length_error);
        unchanged();
    };
    check(standalone.get(), [&] {
        EXPECT_EQ(standalone->indexSize(), 1);
        EXPECT_EQ(native->getStoredVectorDataByLabel(10), stored);
    });
    WriteModeGuard guard; VecSim_SetWriteMode(VecSim_WriteInPlace);
    TestPool pool;
    auto primary = CreateParams(hp);
    TieredIndexParams tp{.jobQueue = &pool.jobQ, .jobQueueCtx = pool.ctx,
        .submitCb = tieredIndexMock::submit_callback, .flatBufferLimit = 1,
        .primaryIndexParams = &primary,
        .specificParams = {TieredHNSWParams{.QuantNormalizationSetSize = 3}}};
    auto *base = TieredFactory::NewIndex(&tp);
    pool.ctx->index_strong_ref = std::shared_ptr<VecSimIndex>(base, VecSimIndex_Free);
    ASSERT_NE(base, nullptr);
    auto *tiered = dynamic_cast<TieredHNSWIndex<float, float> *>(base);
    auto *backend = tiered->getBackendForTests();
    VecSimIndex_AddVector(base, original.data(), 10);
    const double original_score = VecSimIndex_GetDistanceFrom_Unsafe(base, 10, original.data());
    check(base, [&] {
        EXPECT_EQ(base->indexSize(), 1);
        EXPECT_EQ(backend->turboQuantTrainingCount(), 1);
        EXPECT_EQ(backend->indexSize(), 0);
        EXPECT_DOUBLE_EQ(VecSimIndex_GetDistanceFrom_Unsafe(base, 10, original.data()),
                         original_score);
    });
    // Completing training proves that no failed update modified the live running sum.
    VecSimIndex_AddVector(base, original.data(), 20);
    VecSimIndex_AddVector(base, original.data(), 30);
    EXPECT_FALSE(backend->isTurboQuantTraining());
    for (size_t i = 0; i < dim; ++i) EXPECT_FLOAT_EQ(backend->getTurboQuant()->biasMean()[i], original[i]);
}

TEST_P(TurboQuantTest, RejectsZeroCosineBeforeBufferOrOverwriteMutation) {
    const unsigned bits = GetParam();
    WriteModeGuard guard;
    for (auto mode : {VecSim_WriteAsync, VecSim_WriteInPlace}) {
      for (bool multi : {false, true}) {
        VecSim_SetWriteMode(mode);
        TestPool pool;
        constexpr size_t dim = 7;
        auto hp = parameters(bits, dim); hp.metric = VecSimMetric_Cosine; hp.multi = multi;
        auto primary = CreateParams(hp);
        TieredIndexParams tp{.jobQueue = &pool.jobQ, .jobQueueCtx = pool.ctx,
            .submitCb = tieredIndexMock::submit_callback, .flatBufferLimit = 1,
            .primaryIndexParams = &primary,
            .specificParams = {TieredHNSWParams{.QuantNormalizationSetSize = 2}}};
        auto *base = TieredFactory::NewIndex(&tp);
        pool.ctx->index_strong_ref = std::shared_ptr<VecSimIndex>(base, VecSimIndex_Free);
        ASSERT_NE(base, nullptr);
        auto *tiered = dynamic_cast<TieredHNSWIndex<float, float> *>(base);
        auto *backend = tiered->getBackendForTests();
        std::array<float, dim> row{}, zero{}; row[0] = 0.5f; row[2] = -0.3f;
        EXPECT_THROW(VecSimIndex_AddVector(base, zero.data(), 10), std::invalid_argument);
        EXPECT_EQ(base->indexSize(), 0); EXPECT_EQ(backend->turboQuantTrainingCount(), 0);
        EXPECT_EQ(VecSimIndex_AddVector(base, row.data(), 10), 1);
        EXPECT_THROW(VecSimIndex_AddVector(base, zero.data(), 10), std::invalid_argument);
        EXPECT_THROW(VecSimIndex_UpdateVectors(base, 10, zero.data(), 1), std::invalid_argument);
        EXPECT_EQ(base->indexSize(), 1); EXPECT_EQ(backend->turboQuantTrainingCount(), 1);
        if (multi) {
            for (size_t zero_row : {size_t(0), size_t(1)}) {
                std::array<float, 2 * dim> replacement{};
                std::copy(row.begin(), row.end(), replacement.begin());
                std::copy(row.begin(), row.end(), replacement.begin() + dim);
                std::fill(replacement.begin() + zero_row * dim,
                          replacement.begin() + (zero_row + 1) * dim, 0.0f);
                EXPECT_THROW(VecSimIndex_UpdateVectors(base, 10, replacement.data(), 2), std::invalid_argument);
                EXPECT_EQ(base->indexSize(), 1); EXPECT_EQ(backend->turboQuantTrainingCount(), 1);
            }
        }
        EXPECT_EQ(VecSimIndex_DeleteVector(base, 10), 1);
        EXPECT_EQ(backend->turboQuantTrainingCount(), 0);
        VecSimIndex_AddVector(base, row.data(), 10);
        VecSimIndex_AddVector(base, row.data(), 20);
        while (!pool.jobQ.empty()) pool.thread_iteration();
        auto normalized = row; VecSim_Normalize(normalized.data(), dim, VecSimType_FLOAT32);
        for (size_t i = 0; i < dim; ++i) EXPECT_FLOAT_EQ(backend->getTurboQuant()->biasMean()[i], normalized[i]);
        const auto before = backend->getStoredVectorDataByLabel(10);
        EXPECT_THROW(VecSimIndex_AddVector(base, zero.data(), 10), std::invalid_argument);
        EXPECT_EQ(backend->getStoredVectorDataByLabel(10), before);
        EXPECT_EQ(base->indexSize(), 2);
      }
    }
    // A zero IP vector has a valid distance and never passes through cosine normalization.
    auto hp = parameters(bits, 7); auto params = CreateParams(hp);
    IndexOwner ip(VecSimIndex_New(&params), VecSimIndex_Free);
    std::array<float, 7> zero{};
    EXPECT_EQ(VecSimIndex_AddVector(ip.get(), zero.data(), 10), 1);
    EXPECT_EQ(VecSimIndex_GetDistanceFrom_Unsafe(ip.get(), 10, zero.data()), 1);
}

struct CallbackPreparationProbe {
    tieredIndexMock::IndexExtCtx *pool_context;
    VecSimIndex *index = nullptr;
    size_t memory_after_publication = 0;
    size_t calls = 0;
    static int submit(void *queue, void *context, AsyncJob **jobs, JobCallback *callbacks, size_t count) {
        auto &probe = *static_cast<CallbackPreparationProbe *>(context);
        EXPECT_EQ(probe.index->getAllocationSize(), probe.memory_after_publication);
        for (size_t i = 0; i < count; ++i) EXPECT_EQ(callbacks[i], jobs[i]->Execute);
        ++probe.calls;
        return tieredIndexMock::submit_callback(queue, probe.pool_context, jobs, callbacks, count);
    }
};
TEST_P(TurboQuantTest, AsyncCallbacksArePreparedBeforeFittedStatePublication) {
    const unsigned bits = GetParam();
    WriteModeGuard guard; VecSim_SetWriteMode(VecSim_WriteAsync);
    TestPool pool;
    CallbackPreparationProbe probe{.pool_context = pool.ctx};
    auto hp = parameters(bits, 7); auto primary = CreateParams(hp);
    TieredIndexParams tp{.jobQueue = &pool.jobQ, .jobQueueCtx = &probe,
        .submitCb = CallbackPreparationProbe::submit, .flatBufferLimit = 1,
        .primaryIndexParams = &primary,
        .specificParams = {TieredHNSWParams{.QuantNormalizationSetSize = 2}}};
    auto *base = TieredFactory::NewIndex(&tp); probe.index = base;
    pool.ctx->index_strong_ref = std::shared_ptr<VecSimIndex>(base, VecSimIndex_Free);
    ASSERT_NE(base, nullptr);
    auto *tiered = dynamic_cast<TieredHNSWIndex<float, float> *>(base);
    tiered->setAfterQuantizationFinalizationHook([&] { probe.memory_after_publication = base->getAllocationSize(); });
    std::array<float, 7> row{}; row[0] = 0.5f;
    VecSimIndex_AddVector(base, row.data(), 10);
    tiered->setBeforeTurboQuantCallbacksPreparationHook([] { throw std::bad_alloc(); });
    EXPECT_THROW(VecSimIndex_AddVector(base, row.data(), 20), std::bad_alloc);
    auto *backend = tiered->getBackendForTests();
    EXPECT_TRUE(backend->isTurboQuantTraining());
    EXPECT_EQ(backend->turboQuantTrainingCount(), 2);
    EXPECT_EQ(base->indexSize(), 2); EXPECT_EQ(backend->indexSize(), 0);
    EXPECT_EQ(probe.calls, 0); EXPECT_TRUE(pool.jobQ.empty());
    for (float value : backend->getTurboQuant()->biasMean()) EXPECT_EQ(value, 0);
    tiered->setBeforeTurboQuantCallbacksPreparationHook({});
    EXPECT_EQ(VecSimIndex_AddVector(base, row.data(), 20), 0);
    EXPECT_FALSE(backend->isTurboQuantTraining());
    EXPECT_EQ(probe.calls, 1);
    EXPECT_EQ(pool.jobQ.size(), 2);
    while (!pool.jobQ.empty()) pool.thread_iteration();
    EXPECT_EQ(base->statisticInfo().flatBufferSize, 0);
}

TEST_P(TurboQuantTest, BackendMetadataUsesAtomicTrainingCountDuringFlatWrites) {
    const unsigned bits = GetParam();
    WriteModeGuard guard; VecSim_SetWriteMode(VecSim_WriteInPlace);
    TestPool pool;
    auto hp = parameters(bits, 7); auto primary = CreateParams(hp);
    TieredIndexParams tp{.jobQueue = &pool.jobQ, .jobQueueCtx = pool.ctx,
        .submitCb = tieredIndexMock::submit_callback, .flatBufferLimit = 1,
        .primaryIndexParams = &primary,
        .specificParams = {TieredHNSWParams{.QuantNormalizationSetSize = 1000}}};
    auto *base = TieredFactory::NewIndex(&tp);
    pool.ctx->index_strong_ref = std::shared_ptr<VecSimIndex>(base, VecSimIndex_Free);
    ASSERT_NE(base, nullptr);
    auto *backend = dynamic_cast<TieredHNSWIndex<float, float> *>(base)->getBackendForTests();
    std::array<float, 7> row{}; row[0] = 0.5f;
    std::atomic<bool> stop{false}; std::atomic<size_t> reads{0}, invalid{0};
    std::thread reader([&] {
        while (!stop.load()) {
            auto *info = backend->debugInfoIterator();
            while (info->hasNext()) {
                const auto *field = info->next();
                if (std::strcmp(field->fieldName, "QUANT_TRAINING_COUNT") == 0 &&
                    field->fieldValue.uintegerValue > 1) ++invalid;
            }
            delete info; ++reads;
        }
    });
    while (reads.load() < 10) std::this_thread::yield();
    for (size_t i = 0; i < 200; ++i) {
        VecSimIndex_AddVector(base, row.data(), 10);
        VecSimIndex_DeleteVector(base, 10);
    }
    stop.store(true); reader.join();
    EXPECT_EQ(invalid.load(), 0); EXPECT_GE(reads.load(), 10);
    EXPECT_EQ(backend->turboQuantTrainingCount(), 0);
}

TEST(TurboQuantCompatibility, ExistingWriterVersionsAndMetadataRemainUnchanged) {
    for (auto type : {VecSimQuant_NONE, VecSimQuant_SQ8}) {
        auto hp = parameters(2, 9); hp.quantType = type;
        auto params = CreateParams(hp);
        IndexOwner index(VecSimIndex_New(&params), VecSimIndex_Free);
        ASSERT_NE(index, nullptr);
        auto *backend = dynamic_cast<HNSWIndex<float, float> *>(index.get());
        ASSERT_NE(backend, nullptr);
        const auto filename = path(unsigned(type), "legacy");
        backend->saveIndex(filename);
        { std::ifstream in(filename, std::ios::binary);
          EXPECT_EQ(HNSWSerializer::ReadVersion(in), type == VecSimQuant_NONE
              ? HNSWSerializer::EncodingVersion::V4 : HNSWSerializer::EncodingVersion::V5); }
        IndexOwner restored(HNSWFactory::NewIndex(filename), VecSimIndex_Free);
        ASSERT_NE(restored, nullptr);
        auto *info = index->debugInfoIterator();
        while (info->hasNext()) EXPECT_STRNE(info->next()->fieldName, "QUANTIZATION");
        delete info;
        std::remove(filename.c_str());
    }
}

INSTANTIATE_TEST_SUITE_P(TQ2AndTQ3, TurboQuantTest, ::testing::Values(2U, 3U));
} // namespace

/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "VecSim/vec_sim.h"
#include "unit_test_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <unistd.h>
#include <vector>

namespace {

using IndexPtr = std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)>;
using ReplyPtr = std::unique_ptr<VecSimQueryReply, decltype(&VecSimQueryReply_Free)>;
using BatchPtr = std::unique_ptr<VecSimBatchIterator, decltype(&VecSimBatchIterator_Free)>;
using Point = std::array<float, 4>;

VecSimParams pipnnParams(size_t beam = 256) {
    VecSimParams params{};
    params.algo = VecSimAlgo_PIPNN;
    auto &p = params.algoParams.pipnnParams;
    p.type = VecSimType_FLOAT32;
    p.dim = Point{}.size();
    p.metric = VecSimMetric_L2;
    p.blockSize = 8;
    p.maxDegree = 16;
    p.leafSize = 8;
    p.fanout = 4;
    p.hashBits = 4;
    p.candidateLimit = 128;
    p.alpha = 1.2;
    p.efRuntime = beam;
    p.seed = 42;
    return params;
}

IndexPtr newIndex(const VecSimParams &params) {
    return IndexPtr(VecSimIndex_New(&params), VecSimIndex_Free);
}

IndexPtr newExactIndex() {
    BFParams bf{.type = VecSimType_FLOAT32, .dim = Point{}.size(), .metric = VecSimMetric_L2};
    return newIndex(CreateParams(bf));
}

std::vector<Point> randomPoints(size_t count) {
    std::mt19937 generator(73);
    std::uniform_real_distribution<float> distribution(-10, 10);
    std::vector<Point> points(count);
    for (auto &point : points) {
        for (auto &coordinate : point) {
            coordinate = distribution(generator);
        }
    }
    return points;
}

std::vector<std::pair<size_t, double>> topK(VecSimIndex *index, const Point &query, size_t k) {
    std::vector<std::pair<size_t, double>> results;
    runTopKSearchTest(index, query.data(), k, [&](size_t label, double score, size_t) {
        results.emplace_back(label, score);
    });
    return results;
}

void expectNearest(VecSimIndex *index, const Point &query, size_t expected_label) {
    runTopKSearchTest(index, query.data(), 1, [&](size_t label, double score, size_t) {
        EXPECT_EQ(label, expected_label);
        EXPECT_DOUBLE_EQ(score, 0);
    });
}

// Each file owns a unique path so persistence tests can run concurrently.
struct TemporaryFile {
    std::string path;

    TemporaryFile() {
        char name[] = "/tmp/vecsim-pipnn-test-XXXXXX";
        int descriptor = mkstemp(name);
        if (descriptor >= 0) {
            close(descriptor);
            path = name;
        }
    }
    ~TemporaryFile() {
        if (!path.empty()) {
            unlink(path.c_str());
        }
    }
};

struct TimeoutGuard {
    TimeoutGuard() {
        VecSim_SetTimeoutCallbackFunction([](void *) { return 1; });
    }
    ~TimeoutGuard() {
        VecSim_SetTimeoutCallbackFunction([](void *) { return 0; });
    }
};

// These wrappers remain compatible with allocations made before and after an injection.
class FailNthAllocation {
    static inline thread_local size_t failAt = 0;
    static inline thread_local size_t calls = 0;
    static inline thread_local bool failed = false;
    static inline thread_local bool trackPointers = false;
    static inline thread_local std::unordered_set<uintptr_t> livePointers;

    static bool shouldFail() {
        if (++calls != failAt) {
            return false;
        }
        failed = true;
        return true;
    }
    static void *remember(void *pointer) {
        if (trackPointers && pointer) {
            livePointers.insert(reinterpret_cast<uintptr_t>(pointer));
        }
        return pointer;
    }
    static void *allocate(size_t size) {
        return shouldFail() ? nullptr : remember(std::malloc(size));
    }
    static void *callocate(size_t count, size_t size) {
        return shouldFail() ? nullptr : remember(std::calloc(count, size));
    }
    static void *reallocate(void *pointer, size_t size) {
        if (shouldFail()) {
            return nullptr;
        }
        const auto old_address = reinterpret_cast<uintptr_t>(pointer);
        void *resized = std::realloc(pointer, size);
        if (trackPointers && (resized || size == 0)) {
            livePointers.erase(old_address);
        }
        return remember(resized);
    }
    static void release(void *pointer) {
        if (trackPointers && pointer) {
            livePointers.erase(reinterpret_cast<uintptr_t>(pointer));
        }
        std::free(pointer);
    }

public:
    explicit FailNthAllocation(size_t position, bool track = false) {
        failAt = position;
        calls = 0;
        failed = false;
        trackPointers = track;
        livePointers.clear();
        VecSim_SetMemoryFunctions({.allocFunction = allocate,
                                   .callocFunction = callocate,
                                   .reallocFunction = reallocate,
                                   .freeFunction = release});
    }
    ~FailNthAllocation() { restore(); }
    bool wasInjected() const { return failed; }
    size_t liveAllocationCount() const { return livePointers.size(); }
    static void restore() {
        trackPointers = false;
        VecSim_SetMemoryFunctions({.allocFunction = std::malloc,
                                   .callocFunction = std::calloc,
                                   .reallocFunction = std::realloc,
                                   .freeFunction = std::free});
    }
};

class PiPNNAllocationFailureTest : public ::testing::Test {
protected:
    static constexpr size_t storedCount = 16;
    static constexpr size_t maxFailurePosition = 2048;
    const std::vector<Point> points = randomPoints(storedCount + 1);
    std::vector<std::shared_ptr<VecSimAllocator>> allocators;

    IndexPtr preparedIndex(bool build = true) {
        auto index = newIndex(pipnnParams());
        if (!index) {
            return index;
        }
        allocators.push_back(index->getAllocator());
        for (size_t label = 0; label < storedCount; ++label) {
            EXPECT_EQ(VecSimIndex_AddVector(index.get(), points[label].data(), label), 1);
        }
        if (build) {
            EXPECT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
        }
        return index;
    }

    void verifyStoredVectors(VecSimIndex *index, bool relabeled = false) {
        for (size_t label = 0; label < storedCount; ++label) {
            expectNearest(index, points[label], relabeled && label == 3 ? 99 : label);
        }
    }

    void TearDown() override {
        FailNthAllocation::restore();
        // Kept alive until teardown so fatal assertions also verify allocator cleanup.
        for (const auto &allocator : allocators) {
            EXPECT_EQ(allocator->getAllocationSize(), sizeof(VecSimAllocator));
        }
    }
};

} // namespace

TEST(PiPNNTest, SupportsOnlySingleValueFloat32L2) {
    auto params = pipnnParams();
    auto index = newIndex(params);
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(VecSimIndex_IndexSize(index.get()), 0);

    for (auto type : {VecSimType_FLOAT64, VecSimType_FLOAT16, VecSimType_BFLOAT16, VecSimType_INT8,
                      VecSimType_UINT8}) {
        auto unsupported = params;
        unsupported.algoParams.pipnnParams.type = type;
        EXPECT_EQ(newIndex(unsupported), nullptr);
    }
    for (auto metric : {VecSimMetric_IP, VecSimMetric_Cosine}) {
        auto unsupported = params;
        unsupported.algoParams.pipnnParams.metric = metric;
        EXPECT_EQ(newIndex(unsupported), nullptr);
    }
    params.algoParams.pipnnParams.multi = true;
    EXPECT_EQ(newIndex(params), nullptr);
    params.algoParams.pipnnParams.multi = false;
    params.algoParams.pipnnParams.dim = 0;
    EXPECT_EQ(newIndex(params), nullptr);
}

TEST(PiPNNTest, ZeroAlgorithmParametersUseDefaults) {
    VecSimParams params{};
    params.algo = VecSimAlgo_PIPNN;
    params.algoParams.pipnnParams.type = VecSimType_FLOAT32;
    params.algoParams.pipnnParams.dim = Point{}.size();
    params.algoParams.pipnnParams.metric = VecSimMetric_L2;
    auto index = newIndex(params);
    ASSERT_NE(index, nullptr);
    Point point{1, 2, 3, 4};
    EXPECT_EQ(VecSimIndex_AddVector(index.get(), point.data(), 7), 1);
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    EXPECT_TRUE(VecSimIndex_IsGraphReady(index.get()));
    expectNearest(index.get(), point, 7);
}

TEST(PiPNNTest, PublicInfoEstimatesAndRuntimeParameters) {
    constexpr size_t count = 64;
    auto params = pipnnParams(1);
    const size_t initial_estimate = VecSimIndex_EstimateInitialSize(&params);
    const size_t element_estimate = VecSimIndex_EstimateElementSize(&params);
    EXPECT_GT(initial_estimate, 0);
    EXPECT_LT(initial_estimate, SIZE_MAX);
    EXPECT_GT(element_estimate, Point{}.size() * sizeof(float));
    EXPECT_LT(element_estimate, SIZE_MAX);
    auto invalid = params;
    invalid.algoParams.pipnnParams.metric = VecSimMetric_Cosine;
    EXPECT_EQ(VecSimIndex_EstimateInitialSize(&invalid), SIZE_MAX);
    EXPECT_EQ(VecSimIndex_EstimateElementSize(&invalid), SIZE_MAX);

    auto index = newIndex(params);
    auto exact = newExactIndex();
    ASSERT_NE(index, nullptr);
    ASSERT_NE(exact, nullptr);
    auto points = randomPoints(count);
    for (size_t label = 0; label < count; ++label) {
        VecSimIndex_AddVector(index.get(), points[label].data(), label);
        VecSimIndex_AddVector(exact.get(), points[label].data(), label);
    }
    auto check_info = [&](bool ready) {
        EXPECT_EQ(VecSimIndex_BasicInfo(index.get()).algo, VecSimAlgo_PIPNN);
        auto info = VecSimIndex_DebugInfo(index.get());
        EXPECT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_PIPNN);
        EXPECT_EQ(info.commonInfo.indexSize, count);
        std::unique_ptr<VecSimDebugInfoIterator, decltype(&VecSimDebugInfoIterator_Free)> iterator(
            VecSimIndex_DebugInfoIterator(index.get()), VecSimDebugInfoIterator_Free);
        ASSERT_NE(iterator, nullptr);
        bool found_ready = false;
        while (VecSimDebugInfoIterator_HasNextField(iterator.get())) {
            auto field = VecSimDebugInfoIterator_NextField(iterator.get());
            if (std::string(field->fieldName) == "GRAPH_READY") {
                found_ready = true;
                ASSERT_EQ(field->fieldType, INFOFIELD_UINT64);
                EXPECT_EQ(field->fieldValue.uintegerValue, ready);
            }
            if (std::string(field->fieldName) == "ALGORITHM") {
                ASSERT_EQ(field->fieldType, INFOFIELD_STRING);
                EXPECT_STREQ(field->fieldValue.stringValue, "PIPNN");
            }
        }
        EXPECT_TRUE(found_ready);
    };
    check_info(false);
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    check_info(true);

    VecSimQueryParams runtime{};
    VecSimRawParam beam{"EF_RUNTIME", 10, "0", 1};
    const char *error = nullptr;
    EXPECT_EQ(VecSimIndex_ResolveParams(index.get(), &beam, 1, &runtime, QUERY_TYPE_KNN, &error),
              VecSimParamResolverErr_BadValue);
    ASSERT_NE(error, nullptr);
    const auto beam_value = std::to_string(count);
    beam.value = beam_value.c_str();
    beam.valLen = beam_value.size();
    ASSERT_EQ(VecSimIndex_ResolveParams(index.get(), &beam, 1, &runtime, QUERY_TYPE_KNN, &error),
              VecSim_OK);
    ASSERT_EQ(runtime.pipnnRuntimeParams.efRuntime, count);
    Point query{1.25f, -2.5f, 3.75f, 0.125f};
    auto expected = topK(exact.get(), query, 10);
    ASSERT_EQ(expected.size(), 10);
    runTopKSearchTest(
        index.get(), query.data(), 10,
        [&](size_t label, double score, size_t position) {
            EXPECT_EQ(label, expected[position].first);
            EXPECT_DOUBLE_EQ(score, expected[position].second);
        },
        &runtime);
}

TEST(PiPNNTest, InvalidTuningIsRejected) {
    const auto defaults = pipnnParams();
    for (size_t field = 0; field < 8; ++field) {
        auto params = defaults;
        auto &p = params.algoParams.pipnnParams;
        switch (field) {
        case 0:
            p.maxDegree = 1;
            break;
        case 1:
            p.leafSize = 1;
            break;
        case 2:
            p.fanout = 1;
            break;
        case 3:
            p.hashBits = 33;
            break;
        case 4:
            p.candidateLimit = p.maxDegree - 1;
            break;
        case 5:
            p.alpha = 0.5;
            break;
        case 6:
            p.alpha = std::numeric_limits<double>::quiet_NaN();
            break;
        case 7:
            p.fanout = 33;
            break;
        }
        EXPECT_EQ(newIndex(params), nullptr) << "Invalid parameter case " << field;
    }
}

TEST(PiPNNTest, LifecycleRejectsNullInputsAndOtherBackends) {
    auto index = newIndex(pipnnParams());
    auto exact = newExactIndex();
    ASSERT_NE(index, nullptr);
    ASSERT_NE(exact, nullptr);
    EXPECT_EQ(VecSimIndex_BuildGraph(nullptr), 0);
    EXPECT_FALSE(VecSimIndex_IsGraphReady(nullptr));
    EXPECT_EQ(VecSimIndex_SavePiPNN(nullptr, "/tmp/unused-pipnn-path"), 0);
    EXPECT_EQ(VecSimIndex_SavePiPNN(index.get(), nullptr), 0);
    EXPECT_EQ(VecSimIndex_SavePiPNN(index.get(), ""), 0);
    EXPECT_EQ(VecSimIndex_LoadPiPNN(nullptr, nullptr), nullptr);
    EXPECT_EQ(VecSimIndex_LoadPiPNN("", nullptr), nullptr);
    EXPECT_EQ(VecSimIndex_BuildGraph(exact.get()), 0);
    EXPECT_FALSE(VecSimIndex_IsGraphReady(exact.get()));
    EXPECT_EQ(VecSimIndex_SavePiPNN(exact.get(), "/tmp/unused-pipnn-path"), 0);
}

TEST(PiPNNTest, InvalidVectorsReturnCStatusAndPreserveStoredData) {
    auto index = newIndex(pipnnParams());
    ASSERT_NE(index, nullptr);
    const Point valid{1, 2, 3, 4};
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), valid.data(), 7), 1);
    for (bool build : {false, true}) {
        if (build) {
            ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
        }
        Point nonfinite = valid;
        for (float value :
             {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity()}) {
            nonfinite.back() = value;
            EXPECT_EQ(VecSimIndex_AddVector(index.get(), nonfinite.data(), 99), -1);
            EXPECT_EQ(VecSimIndex_AddVector(index.get(), nonfinite.data(), 7), -1);
            EXPECT_EQ(VecSimIndex_UpdateVectors(index.get(), 7, nonfinite.data(), 1),
                      VecSimUpdate_Failed);
        }
        EXPECT_EQ(VecSimIndex_AddVector(index.get(), nullptr, 99), -1);
        EXPECT_EQ(VecSimIndex_UpdateVectors(index.get(), 7, nullptr, 1), VecSimUpdate_Failed);
        EXPECT_EQ(VecSimIndex_IndexSize(index.get()), 1);
        expectNearest(index.get(), valid, 7);
    }
}

TEST(PiPNNTest, InvalidQueriesReturnNullBeforeAndAfterBuild) {
    auto index = newIndex(pipnnParams());
    ASSERT_NE(index, nullptr);
    const Point valid{1, 2, 3, 4};
    Point nan = valid, positive_infinity = valid, negative_infinity = valid;
    nan.back() = std::numeric_limits<float>::quiet_NaN();
    positive_infinity.back() = std::numeric_limits<float>::infinity();
    negative_infinity.back() = -std::numeric_limits<float>::infinity();
    const std::array<const void *, 4> invalid_queries{nullptr, nan.data(), positive_infinity.data(),
                                                      negative_infinity.data()};
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), valid.data(), 7), 1);
    for (bool build : {false, true}) {
        if (build) {
            ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
        }
        for (const void *query : invalid_queries) {
            ReplyPtr nearest(VecSimIndex_TopKQuery(index.get(), query, 1, nullptr, BY_SCORE),
                             VecSimQueryReply_Free);
            EXPECT_EQ(nearest, nullptr);
            ReplyPtr range(VecSimIndex_RangeQuery(index.get(), query, 10, nullptr, BY_SCORE),
                           VecSimQueryReply_Free);
            EXPECT_EQ(range, nullptr);
            BatchPtr batch(VecSimBatchIterator_New(index.get(), query, nullptr),
                           VecSimBatchIterator_Free);
            EXPECT_EQ(batch, nullptr);
            EXPECT_TRUE(std::isnan(VecSimIndex_GetDistanceFrom_Unsafe(index.get(), 7, query)));
        }
        for (double radius : {-1.0, std::numeric_limits<double>::quiet_NaN()}) {
            ReplyPtr range(
                VecSimIndex_RangeQuery(index.get(), valid.data(), radius, nullptr, BY_SCORE),
                VecSimQueryReply_Free);
            EXPECT_EQ(range, nullptr);
        }
        // BY_SCORE_THEN_ID is a public enum value, but is not a supported query order.
        ReplyPtr nearest(
            VecSimIndex_TopKQuery(index.get(), valid.data(), 1, nullptr, BY_SCORE_THEN_ID),
            VecSimQueryReply_Free);
        EXPECT_EQ(nearest, nullptr);
        ReplyPtr range(
            VecSimIndex_RangeQuery(index.get(), valid.data(), 10, nullptr, BY_SCORE_THEN_ID),
            VecSimQueryReply_Free);
        EXPECT_EQ(range, nullptr);
        expectNearest(index.get(), valid, 7);
    }
}

TEST(PiPNNTest, ExactBeforeBuildAndHighRecallAfterBuild) {
    constexpr size_t count = 160;
    constexpr size_t k = 10;
    auto index = newIndex(pipnnParams(count));
    auto exact = newExactIndex();
    ASSERT_NE(index, nullptr);
    ASSERT_NE(exact, nullptr);
    auto points = randomPoints(count);
    for (size_t i = 0; i < points.size(); ++i) {
        ASSERT_EQ(VecSimIndex_AddVector(index.get(), points[i].data(), i), 1);
        VecSimIndex_AddVector(exact.get(), points[i].data(), i);
    }
    ASSERT_FALSE(VecSimIndex_IsGraphReady(index.get()));
    Point query{1.25f, -2.5f, 3.75f, 0.125f};
    EXPECT_EQ(topK(index.get(), query, k), topK(exact.get(), query, k));
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    ASSERT_TRUE(VecSimIndex_IsGraphReady(index.get()));

    size_t hits = 0;
    constexpr size_t queries = 20;
    for (size_t q = 0; q < queries; ++q) {
        // Offset stored points so nearest-neighbor recall is not only self lookup.
        query = points[q * 7];
        query[0] += 0.17f;
        query[2] -= 0.31f;
        auto reference = topK(exact.get(), query, k);
        auto actual = topK(index.get(), query, k);
        ASSERT_EQ(actual.size(), k);
        for (const auto &[label, score] : actual) {
            const auto &point = points.at(label);
            double expected_score = 0;
            for (size_t d = 0; d < point.size(); ++d) {
                expected_score += double(point[d] - query[d]) * (point[d] - query[d]);
            }
            EXPECT_NEAR(score, expected_score, 1e-3);
            hits += std::any_of(reference.begin(), reference.end(),
                                [label](const auto &result) { return result.first == label; });
        }
    }
    EXPECT_GE(double(hits) / (queries * k), 0.95);
    EXPECT_TRUE(VecSimIndex_IsGraphReady(index.get()));
}

TEST(PiPNNTest, MutationsInvalidateAndRebuildAfterCompaction) {
    auto index = newIndex(pipnnParams());
    auto exact = newExactIndex();
    ASSERT_NE(index, nullptr);
    auto points = randomPoints(48);
    for (size_t label = 0; label < points.size(); ++label) {
        VecSimIndex_AddVector(index.get(), points[label].data(), label);
        VecSimIndex_AddVector(exact.get(), points[label].data(), label);
    }
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    Point replacement{100, 200, 300, 400};
    EXPECT_EQ(VecSimIndex_AddVector(index.get(), replacement.data(), 3), 0);
    VecSimIndex_AddVector(exact.get(), replacement.data(), 3);
    EXPECT_FALSE(VecSimIndex_IsGraphReady(index.get()));
    EXPECT_EQ(VecSimIndex_IndexSize(index.get()), points.size());
    expectNearest(index.get(), replacement, 3);
    EXPECT_EQ(topK(index.get(), points[3], 12), topK(exact.get(), points[3], 12));
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);

    // Remove a nonfinal slot, forcing the backend to keep moved vectors and labels aligned.
    ASSERT_EQ(VecSimIndex_DeleteVector(index.get(), 1), 1);
    VecSimIndex_DeleteVector(exact.get(), 1);
    EXPECT_FALSE(VecSimIndex_IsGraphReady(index.get()));
    expectNearest(index.get(), points.back(), points.size() - 1);
    EXPECT_EQ(topK(index.get(), points[1], 12), topK(exact.get(), points[1], 12));
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    EXPECT_TRUE(VecSimIndex_IsGraphReady(index.get()));
    expectNearest(index.get(), replacement, 3);
    expectNearest(index.get(), points.back(), points.size() - 1);

    Point added{-100, -200, -300, -400};
    EXPECT_EQ(VecSimIndex_AddVector(index.get(), added.data(), 90), 1);
    EXPECT_FALSE(VecSimIndex_IsGraphReady(index.get()));
    expectNearest(index.get(), added, 90);
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    expectNearest(index.get(), added, 90);
    EXPECT_EQ(VecSimIndex_IndexSize(index.get()), points.size());

    EXPECT_EQ(VecSimIndex_RelabelVector(index.get(), 90, 91), VecSimRelabel_OK);
    EXPECT_FALSE(VecSimIndex_IsGraphReady(index.get()));
    expectNearest(index.get(), added, 91);
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    EXPECT_EQ(VecSimIndex_UpdateVectors(index.get(), 91, replacement.data(), 1), VecSimUpdate_OK);
    EXPECT_FALSE(VecSimIndex_IsGraphReady(index.get()));
    EXPECT_EQ(VecSimIndex_DeleteVector(index.get(), 3), 1);
    expectNearest(index.get(), replacement, 91);
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    expectNearest(index.get(), replacement, 91);
}

TEST(PiPNNTest, RangeAndBatchRemainExactBeforeAndAfterBuild) {
    auto index = newIndex(pipnnParams());
    ASSERT_NE(index, nullptr);
    constexpr size_t count = 17;
    for (size_t i = 0; i < count; ++i) {
        GenerateAndAddVector<float>(index.get(), 4, i, i);
    }
    Point query{};
    for (bool build : {false, true}) {
        if (build) {
            ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
        }
        runRangeQueryTest(
            index.get(), query.data(), 4 * 4.5 * 4.5,
            [](size_t label, double score, size_t position) {
                EXPECT_EQ(label, position);
                EXPECT_DOUBLE_EQ(score, 4 * label * label);
            },
            5, BY_SCORE);
        BatchPtr batch(VecSimBatchIterator_New(index.get(), query.data(), nullptr),
                       VecSimBatchIterator_Free);
        ASSERT_NE(batch, nullptr);
        size_t seen = 0;
        while (VecSimBatchIterator_HasNext(batch.get()) && seen < count) {
            size_t start = seen;
            size_t expected = std::min(size_t(6), count - seen);
            runBatchIteratorSearchTest(
                batch.get(), 6,
                [&](size_t label, double score, size_t position) {
                    EXPECT_EQ(label, start + position);
                    EXPECT_DOUBLE_EQ(score, 4 * label * label);
                    ++seen;
                },
                BY_SCORE, expected);
        }
        EXPECT_EQ(seen, count);
        EXPECT_FALSE(VecSimBatchIterator_HasNext(batch.get()));
        VecSimBatchIterator_Reset(batch.get());
        EXPECT_TRUE(VecSimBatchIterator_HasNext(batch.get()));
        runBatchIteratorSearchTest(batch.get(), 1, [](size_t label, double score, size_t) {
            EXPECT_EQ(label, 0);
            EXPECT_DOUBLE_EQ(score, 0);
        });
    }
}

TEST(PiPNNTest, TimeoutPropagatesFromGraphFallbackRangeAndBatch) {
    auto index = newIndex(pipnnParams());
    ASSERT_NE(index, nullptr);
    for (size_t i = 0; i < 32; ++i) {
        GenerateAndAddVector<float>(index.get(), 4, i, i);
    }
    Point query{};
    for (bool build : {false, true}) {
        if (build) {
            ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
        }
        TimeoutGuard timeout;
        ReplyPtr nearest(VecSimIndex_TopKQuery(index.get(), query.data(), 5, nullptr, BY_SCORE),
                         VecSimQueryReply_Free);
        EXPECT_EQ(VecSimQueryReply_GetCode(nearest.get()), VecSim_QueryReply_TimedOut);
        ReplyPtr range(VecSimIndex_RangeQuery(index.get(), query.data(), 100, nullptr, BY_SCORE),
                       VecSimQueryReply_Free);
        EXPECT_EQ(VecSimQueryReply_GetCode(range.get()), VecSim_QueryReply_TimedOut);
        BatchPtr batch(VecSimBatchIterator_New(index.get(), query.data(), nullptr),
                       VecSimBatchIterator_Free);
        ReplyPtr first(VecSimBatchIterator_Next(batch.get(), 5, BY_SCORE), VecSimQueryReply_Free);
        EXPECT_EQ(VecSimQueryReply_GetCode(first.get()), VecSim_QueryReply_TimedOut);
    }
}

TEST(PiPNNTest, BatchRetriesPartialTimeoutAndZeroBatchDoesNotConsumeResults) {
    constexpr size_t count = 1101;
    auto index = newIndex(pipnnParams());
    ASSERT_NE(index, nullptr);
    auto points = randomPoints(count);
    for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(VecSimIndex_AddVector(index.get(), points[i].data(), i), 1);
    }
    const Point query = points.back();
    const auto expected = topK(index.get(), query, count);
    struct State {
        size_t calls = 0;
        bool enabled = true;
    } state;
    VecSimQueryParams params{};
    params.timeoutCtx = &state;
    TimeoutGuard restoreCallback;
    VecSim_SetTimeoutCallbackFunction([](void *context) {
        auto &state = *static_cast<State *>(context);
        return state.enabled && ++state.calls > 7 ? 1 : 0;
    });
    BatchPtr batch(VecSimBatchIterator_New(index.get(), query.data(), &params),
                   VecSimBatchIterator_Free);
    ASSERT_NE(batch, nullptr);
    ReplyPtr invalid(VecSimBatchIterator_Next(batch.get(), 1, BY_SCORE_THEN_ID),
                     VecSimQueryReply_Free);
    EXPECT_EQ(invalid, nullptr);
    ReplyPtr empty(VecSimBatchIterator_Next(batch.get(), 0, BY_SCORE), VecSimQueryReply_Free);
    ASSERT_NE(empty, nullptr);
    EXPECT_EQ(VecSimQueryReply_Len(empty.get()), 0);
    EXPECT_EQ(state.calls, 0);
    EXPECT_TRUE(VecSimBatchIterator_HasNext(batch.get()));

    ReplyPtr timedOut(VecSimBatchIterator_Next(batch.get(), 10, BY_SCORE), VecSimQueryReply_Free);
    ASSERT_NE(timedOut, nullptr);
    EXPECT_EQ(VecSimQueryReply_GetCode(timedOut.get()), VecSim_QueryReply_TimedOut);
    EXPECT_EQ(VecSimQueryReply_Len(timedOut.get()), 0);
    EXPECT_EQ(state.calls, 8);
    state.enabled = false;
    std::vector<std::pair<size_t, double>> actual;
    runBatchIteratorSearchTest(batch.get(), count, [&](size_t label, double score, size_t) {
        actual.emplace_back(label, score);
    });
    EXPECT_EQ(actual, expected);
    EXPECT_FALSE(VecSimBatchIterator_HasNext(batch.get()));
}

TEST(PiPNNTest, LabelsKeepAllBitsThroughBuildAndMutation) {
    if (sizeof(size_t) < sizeof(uint64_t)) {
        GTEST_SKIP() << "Requires 64-bit labels";
    }
    auto index = newIndex(pipnnParams());
    ASSERT_NE(index, nullptr);
    const size_t high_label = (uint64_t(1) << 40) + 7;
    Point near{1, 2, 3, 4}, far{10, 20, 30, 40};
    VecSimIndex_AddVector(index.get(), near.data(), high_label);
    VecSimIndex_AddVector(index.get(), far.data(), 7);
    for (bool build : {false, true}) {
        if (build) {
            ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
        }
        // The shared topK helper narrows labels to int, so inspect this reply directly.
        ReplyPtr reply(VecSimIndex_TopKQuery(index.get(), near.data(), 1, nullptr, BY_SCORE),
                       VecSimQueryReply_Free);
        ASSERT_EQ(VecSimQueryReply_Len(reply.get()), 1);
        auto iterator = VecSimQueryReply_GetIterator(reply.get());
        auto result = VecSimQueryReply_IteratorNext(iterator);
        EXPECT_EQ(VecSimQueryResult_GetId(result), high_label);
        EXPECT_DOUBLE_EQ(VecSimQueryResult_GetScore(result), 0);
        VecSimQueryReply_IteratorFree(iterator);
    }
    EXPECT_EQ(VecSimIndex_DeleteVector(index.get(), high_label), 1);
    EXPECT_EQ(VecSimIndex_IndexSize(index.get()), 1);
    expectNearest(index.get(), far, 7);
}

TEST(PiPNNTest, EmptyAndIdenticalVectorsBuildAndTerminate) {
    auto index = newIndex(pipnnParams());
    ASSERT_NE(index, nullptr);
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    Point query{};
    EXPECT_TRUE(topK(index.get(), query, 10).empty());
    for (size_t label = 0; label < 65; ++label) {
        VecSimIndex_AddVector(index.get(), query.data(), label);
    }
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    EXPECT_TRUE(VecSimIndex_IsGraphReady(index.get()));
    runTopKSearchTest(index.get(), query.data(), 20, [](size_t label, double score, size_t) {
        EXPECT_LT(label, 65);
        EXPECT_DOUBLE_EQ(score, 0);
    });
    for (size_t label = 0; label < 65; ++label) {
        ASSERT_EQ(VecSimIndex_DeleteVector(index.get(), label), 1);
    }
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    EXPECT_TRUE(topK(index.get(), query, 10).empty());
}

TEST(PiPNNTest, AllocatorReleasesGraphAndQueryStorage) {
    auto index = newIndex(pipnnParams());
    ASSERT_NE(index, nullptr);
    auto allocator = index->getAllocator();
    const size_t empty_size = allocator->getAllocationSize();
    auto points = randomPoints(100);
    for (size_t label = 0; label < points.size(); ++label) {
        VecSimIndex_AddVector(index.get(), points[label].data(), label);
    }
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    EXPECT_GT(allocator->getAllocationSize(), empty_size);
    const size_t built_size = allocator->getAllocationSize();
    expectNearest(index.get(), points[12], 12);
    EXPECT_EQ(allocator->getAllocationSize(), built_size);
    index.reset();
    EXPECT_EQ(allocator->getAllocationSize(), sizeof(VecSimAllocator));
}

TEST_F(PiPNNAllocationFailureTest, ConstructorFailuresReleaseAllAllocations) {
    bool completed = false;
    size_t injected_count = 0;
    for (size_t position = 1; position <= maxFailurePosition; ++position) {
        SCOPED_TRACE(position);
        bool created;
        bool injected;
        size_t live_allocations;
        {
            // Unlike operation tests, construction has no surviving allocator to inspect.
            FailNthAllocation failure(position, true);
            auto index = newIndex(pipnnParams());
            created = index != nullptr;
            index.reset();
            injected = failure.wasInjected();
            live_allocations = failure.liveAllocationCount();
        }
        EXPECT_EQ(created, !injected);
        EXPECT_EQ(live_allocations, 0);
        if (!injected) {
            completed = true;
            break;
        }
        ++injected_count;
    }
    EXPECT_TRUE(completed) << "Construction did not finish within the failure-position bound";
    EXPECT_GT(injected_count, 0);
}

TEST_F(PiPNNAllocationFailureTest, BuildFailuresPreserveFlatStorageAndCanBeRetried) {
    bool completed = false;
    size_t injected_count = 0;
    for (size_t position = 1; position <= maxFailurePosition; ++position) {
        SCOPED_TRACE(position);
        auto index = preparedIndex();
        ASSERT_NE(index, nullptr);
        int result;
        bool injected;
        {
            FailNthAllocation failure(position);
            result = VecSimIndex_BuildGraph(index.get());
            injected = failure.wasInjected();
        }
        EXPECT_EQ(result, injected ? 0 : 1);
        EXPECT_EQ(VecSimIndex_IndexSize(index.get()), storedCount);
        EXPECT_EQ(VecSimIndex_IsGraphReady(index.get()), !injected);
        verifyStoredVectors(index.get());
        if (!injected) {
            completed = true;
            break;
        }
        ++injected_count;
        ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
        verifyStoredVectors(index.get());
    }
    EXPECT_TRUE(completed) << "Build did not finish within the failure-position bound";
    EXPECT_GT(injected_count, 0);
}

TEST_F(PiPNNAllocationFailureTest, AddFailuresPreserveCountAndExistingVectors) {
    bool completed = false;
    size_t injected_count = 0;
    for (size_t position = 1; position <= maxFailurePosition; ++position) {
        SCOPED_TRACE(position);
        // The 17th vector crosses the configured block boundary.
        auto index = preparedIndex();
        ASSERT_NE(index, nullptr);
        int result;
        bool injected;
        {
            FailNthAllocation failure(position);
            result = VecSimIndex_AddVector(index.get(), points.back().data(), 99);
            injected = failure.wasInjected();
        }
        EXPECT_EQ(result, injected ? -1 : 1);
        EXPECT_EQ(VecSimIndex_IndexSize(index.get()), storedCount + (injected ? 0 : 1));
        verifyStoredVectors(index.get());
        if (!injected) {
            expectNearest(index.get(), points.back(), 99);
            completed = true;
            break;
        }
        ++injected_count;
        EXPECT_TRUE(
            std::isnan(VecSimIndex_GetDistanceFrom_Unsafe(index.get(), 99, points.back().data())));
        ASSERT_EQ(VecSimIndex_AddVector(index.get(), points.back().data(), 99), 1);
        expectNearest(index.get(), points.back(), 99);
    }
    EXPECT_TRUE(completed) << "Insertion did not finish within the failure-position bound";
    EXPECT_GT(injected_count, 0);
}

TEST_F(PiPNNAllocationFailureTest, RelabelFailuresKeepOriginalLabelAndData) {
    bool completed = false;
    size_t injected_count = 0;
    for (size_t position = 1; position <= maxFailurePosition; ++position) {
        SCOPED_TRACE(position);
        auto index = preparedIndex();
        ASSERT_NE(index, nullptr);
        VecSimRelabelCode result;
        bool injected;
        {
            FailNthAllocation failure(position);
            result = VecSimIndex_RelabelVector(index.get(), 3, 99);
            injected = failure.wasInjected();
        }
        EXPECT_EQ(result, injected ? VecSimRelabel_Failed : VecSimRelabel_OK);
        EXPECT_EQ(VecSimIndex_IndexSize(index.get()), storedCount);
        verifyStoredVectors(index.get(), !injected);
        const size_t absent_label = injected ? 99 : 3;
        EXPECT_TRUE(std::isnan(
            VecSimIndex_GetDistanceFrom_Unsafe(index.get(), absent_label, points[3].data())));
        if (!injected) {
            completed = true;
            break;
        }
        ++injected_count;
        ASSERT_EQ(VecSimIndex_RelabelVector(index.get(), 3, 99), VecSimRelabel_OK);
        verifyStoredVectors(index.get(), true);
    }
    EXPECT_TRUE(completed) << "Relabel did not finish within the failure-position bound";
    EXPECT_GT(injected_count, 0);
}

TEST_F(PiPNNAllocationFailureTest, QueryFailuresReleaseTemporaryStorage) {
    for (bool build : {false, true}) {
        for (bool range : {false, true}) {
            SCOPED_TRACE(build ? "built" : "fallback");
            SCOPED_TRACE(range ? "range" : "topK");
            bool completed = false;
            size_t injected_count = 0;
            for (size_t position = 1; position <= maxFailurePosition; ++position) {
                SCOPED_TRACE(position);
                auto index = preparedIndex(build);
                ASSERT_NE(index, nullptr);
                const auto before = index->getAllocationSize();
                ReplyPtr reply(nullptr, VecSimQueryReply_Free);
                bool injected;
                {
                    FailNthAllocation failure(position);
                    reply.reset(range ? VecSimIndex_RangeQuery(index.get(), points[3].data(), 1e6,
                                                               nullptr, BY_SCORE)
                                      : VecSimIndex_TopKQuery(index.get(), points[3].data(), 4,
                                                              nullptr, BY_SCORE));
                    injected = failure.wasInjected();
                }
                EXPECT_EQ(reply == nullptr, injected);
                if (reply) {
                    EXPECT_EQ(VecSimQueryReply_GetCode(reply.get()), VecSim_QueryReply_OK);
                    EXPECT_EQ(VecSimQueryReply_Len(reply.get()), range ? storedCount : 4);
                }
                reply.reset();
                EXPECT_EQ(index->getAllocationSize(), before);
                EXPECT_EQ(VecSimIndex_IndexSize(index.get()), storedCount);
                EXPECT_EQ(VecSimIndex_IsGraphReady(index.get()), build);
                verifyStoredVectors(index.get());
                if (!injected) {
                    completed = true;
                    break;
                }
                ++injected_count;
            }
            EXPECT_TRUE(completed) << "Query did not finish within the failure-position bound";
            EXPECT_GT(injected_count, 0);
        }
    }
}

TEST_F(PiPNNAllocationFailureTest, DebugIteratorFailuresReleaseTemporaryStorage) {
    auto index = preparedIndex();
    ASSERT_NE(index, nullptr);
    const auto before = index->getAllocationSize();
    bool completed = false;
    size_t injected_count = 0;
    for (size_t position = 1; position <= maxFailurePosition; ++position) {
        SCOPED_TRACE(position);
        std::unique_ptr<VecSimDebugInfoIterator, decltype(&VecSimDebugInfoIterator_Free)> iterator(
            nullptr, VecSimDebugInfoIterator_Free);
        bool injected;
        {
            FailNthAllocation failure(position);
            iterator.reset(VecSimIndex_DebugInfoIterator(index.get()));
            injected = failure.wasInjected();
        }
        EXPECT_EQ(iterator == nullptr, injected);
        if (iterator) {
            EXPECT_EQ(VecSimDebugInfoIterator_NumberOfFields(iterator.get()), 14);
        }
        iterator.reset();
        EXPECT_EQ(index->getAllocationSize(), before);
        EXPECT_TRUE(VecSimIndex_IsGraphReady(index.get()));
        if (!injected) {
            completed = true;
            break;
        }
        ++injected_count;
    }
    EXPECT_TRUE(completed);
    EXPECT_GT(injected_count, 0);
}

TEST_F(PiPNNAllocationFailureTest, BatchFailuresReleaseIteratorAndQueryStorage) {
    for (bool build : {false, true}) {
        SCOPED_TRACE(build ? "built" : "fallback");
        bool completed = false;
        size_t injected_count = 0;
        for (size_t position = 1; position <= maxFailurePosition; ++position) {
            SCOPED_TRACE(position);
            auto index = preparedIndex(build);
            ASSERT_NE(index, nullptr);
            const auto before = index->getAllocationSize();
            BatchPtr batch(nullptr, VecSimBatchIterator_Free);
            ReplyPtr reply(nullptr, VecSimQueryReply_Free);
            bool injected;
            {
                FailNthAllocation failure(position);
                batch.reset(VecSimBatchIterator_New(index.get(), points[3].data(), nullptr));
                if (batch) {
                    reply.reset(VecSimBatchIterator_Next(batch.get(), 4, BY_SCORE));
                }
                injected = failure.wasInjected();
            }
            EXPECT_EQ(batch == nullptr || reply == nullptr, injected);
            if (reply) {
                EXPECT_EQ(VecSimQueryReply_GetCode(reply.get()), VecSim_QueryReply_OK);
                EXPECT_EQ(VecSimQueryReply_Len(reply.get()), 4);
            }
            if (injected && batch && !reply) {
                auto expected = topK(index.get(), points[3], 4);
                ASSERT_EQ(expected.size(), 4);
                runBatchIteratorSearchTest(batch.get(), 4,
                                           [&](size_t label, double score, size_t position) {
                                               EXPECT_EQ(label, expected[position].first);
                                               EXPECT_DOUBLE_EQ(score, expected[position].second);
                                           });
            }
            reply.reset();
            batch.reset();
            EXPECT_EQ(index->getAllocationSize(), before);
            EXPECT_EQ(VecSimIndex_IndexSize(index.get()), storedCount);
            EXPECT_EQ(VecSimIndex_IsGraphReady(index.get()), build);
            verifyStoredVectors(index.get());
            if (!injected) {
                completed = true;
                break;
            }
            ++injected_count;
        }
        EXPECT_TRUE(completed) << "Batch did not finish within the failure-position bound";
        EXPECT_GT(injected_count, 0);
    }
}

TEST(PiPNNTest, PersistencePreservesReadyDirtyAndPostLoadMutations) {
    auto index = newIndex(pipnnParams());
    auto exact = newExactIndex();
    ASSERT_NE(index, nullptr);
    auto points = randomPoints(40);
    for (size_t label = 0; label < points.size(); ++label) {
        VecSimIndex_AddVector(index.get(), points[label].data(), label);
        VecSimIndex_AddVector(exact.get(), points[label].data(), label);
    }
    TemporaryFile file;
    ASSERT_FALSE(file.path.empty());
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    ASSERT_EQ(VecSimIndex_SavePiPNN(index.get(), file.path.c_str()), 1);
    IndexPtr loaded(VecSimIndex_LoadPiPNN(file.path.c_str(), nullptr), VecSimIndex_Free);
    ASSERT_NE(loaded, nullptr);
    EXPECT_TRUE(VecSimIndex_IsGraphReady(loaded.get()));
    EXPECT_EQ(VecSimIndex_IndexSize(loaded.get()), points.size());
    EXPECT_EQ(topK(loaded.get(), points[10], 10), topK(index.get(), points[10], 10));
    auto loaded_allocator = loaded->getAllocator();

    Point replacement{100, 200, 300, 400};
    EXPECT_EQ(VecSimIndex_AddVector(loaded.get(), replacement.data(), 4), 0);
    VecSimIndex_AddVector(exact.get(), replacement.data(), 4);
    EXPECT_EQ(VecSimIndex_DeleteVector(loaded.get(), 1), 1);
    VecSimIndex_DeleteVector(exact.get(), 1);
    EXPECT_FALSE(VecSimIndex_IsGraphReady(loaded.get()));
    ASSERT_EQ(VecSimIndex_SavePiPNN(loaded.get(), file.path.c_str()), 1);
    loaded.reset();
    EXPECT_EQ(loaded_allocator->getAllocationSize(), sizeof(VecSimAllocator));
    loaded.reset(VecSimIndex_LoadPiPNN(file.path.c_str(), nullptr));
    ASSERT_NE(loaded, nullptr);
    EXPECT_FALSE(VecSimIndex_IsGraphReady(loaded.get()));
    EXPECT_EQ(VecSimIndex_IndexSize(loaded.get()), points.size() - 1);
    EXPECT_EQ(topK(loaded.get(), points[1], 10), topK(exact.get(), points[1], 10));
    expectNearest(loaded.get(), replacement, 4);
    ASSERT_EQ(VecSimIndex_BuildGraph(loaded.get()), 1);
    expectNearest(loaded.get(), replacement, 4);
    Point added{-100, -200, -300, -400};
    EXPECT_EQ(VecSimIndex_AddVector(loaded.get(), added.data(), 99), 1);
    EXPECT_FALSE(VecSimIndex_IsGraphReady(loaded.get()));
    expectNearest(loaded.get(), added, 99);
}

TEST(PiPNNTest, PersistenceRejectsTruncationAndCorruptHeader) {
    auto index = newIndex(pipnnParams());
    ASSERT_NE(index, nullptr);
    auto points = randomPoints(20);
    for (size_t label = 0; label < points.size(); ++label) {
        VecSimIndex_AddVector(index.get(), points[label].data(), label);
    }
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    TemporaryFile valid, damaged;
    ASSERT_FALSE(valid.path.empty());
    ASSERT_FALSE(damaged.path.empty());
    ASSERT_EQ(VecSimIndex_SavePiPNN(index.get(), valid.path.c_str()), 1);
    std::ifstream input(valid.path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    ASSERT_GT(bytes.size(), 16);

    for (size_t length : {size_t(0), size_t(1), bytes.size() / 2, bytes.size() - 1}) {
        std::ofstream output(damaged.path, std::ios::binary | std::ios::trunc);
        output.write(bytes.data(), length);
        output.close();
        IndexPtr loaded(VecSimIndex_LoadPiPNN(damaged.path.c_str(), nullptr), VecSimIndex_Free);
        EXPECT_EQ(loaded, nullptr) << "Accepted a file truncated to " << length << " bytes";
    }
    bytes[0] ^= 0x7f;
    std::ofstream output(damaged.path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), bytes.size());
    output.close();
    IndexPtr loaded(VecSimIndex_LoadPiPNN(damaged.path.c_str(), nullptr), VecSimIndex_Free);
    EXPECT_EQ(loaded, nullptr);
}

TEST(PiPNNTest, PersistenceRejectsMissingSuccessorWithValidChecksum) {
    auto params = pipnnParams();
    auto index = newIndex(params);
    ASSERT_NE(index, nullptr);
    const Point first{1, 2, 3, 4}, second{5, 6, 7, 8};
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), first.data(), 7), 1);
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), second.data(), 8), 1);
    ASSERT_EQ(VecSimIndex_BuildGraph(index.get()), 1);
    TemporaryFile file;
    ASSERT_FALSE(file.path.empty());
    ASSERT_EQ(VecSimIndex_SavePiPNN(index.get(), file.path.c_str()), 1);
    IndexPtr pristine(VecSimIndex_LoadPiPNN(file.path.c_str(), nullptr), VecSimIndex_Free);
    ASSERT_NE(pristine, nullptr);
    ASSERT_TRUE(VecSimIndex_IsGraphReady(pristine.get()));

    std::ifstream input(file.path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    // V1: magic[8], version[4], ten tuning fields[8], count[8], ready[4],
    // then count records of label[8] + dim FP32 coordinates, followed by degrees[8].
    constexpr size_t count = 2;
    constexpr size_t header_size = 8 + 4 + 10 * 8 + 8 + 4;
    const size_t degree_offset = header_size + count * (8 + Point{}.size() * 4);
    ASSERT_GT(bytes.size(), degree_offset + count * 8 + 8);
    // A zero-degree row is otherwise structurally valid, but removes node 0's only successor.
    std::fill_n(bytes.begin() + degree_offset, 8, char{0});
    uint64_t checksum = 14695981039346656037ULL;
    for (size_t i = 0; i < bytes.size() - 8; ++i) {
        checksum = (checksum ^ static_cast<unsigned char>(bytes[i])) * 1099511628211ULL;
    }
    for (size_t i = 0; i < 8; ++i) {
        bytes[bytes.size() - 8 + i] = static_cast<char>(checksum >> (8 * i));
    }
    std::ofstream output(file.path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), bytes.size());
    output.close();
    ASSERT_TRUE(output.good());
    IndexPtr malformed(VecSimIndex_LoadPiPNN(file.path.c_str(), nullptr), VecSimIndex_Free);
    EXPECT_EQ(malformed, nullptr);
}

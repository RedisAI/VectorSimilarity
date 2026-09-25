/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "gtest/gtest.h"
#include "mock_thread_pool.h"
#include "unit_test_utils.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr size_t kM = 16;
constexpr size_t kEfConstruction = 200;
constexpr size_t kIndexEfRuntime = 800;
constexpr size_t kQueryEfRuntime = 10000;
constexpr size_t kDeferredThreshold = 100000;
constexpr size_t kLargeCount = 5000;
using LargeVector = std::array<float, 4>;
using SmallVector = std::array<float, 2>;

struct AsyncWriteModeGuard {
    VecSimWriteMode original = VecSimIndexInterface::asyncWriteMode;

    AsyncWriteModeGuard() { VecSimIndexInterface::setWriteMode(VecSim_WriteAsync); }
    ~AsyncWriteModeGuard() { VecSimIndexInterface::setWriteMode(original); }
};

struct TieredCleanup {
    tieredIndexMock &pool;

    ~TieredCleanup() {
        while (!pool.jobQ.empty()) {
            pool.jobQ.pop();
        }
        if (pool.ctx != nullptr) {
            auto allocator = pool.ctx->index_strong_ref ? pool.ctx->index_strong_ref->getAllocator()
                                                        : std::shared_ptr<VecSimAllocator>{};
            pool.reset_ctx();
            (void)allocator;
        }
    }
};

template <typename DataType, typename DistType>
struct RepairAccess : HNSWIndex<DataType, DistType> {
    using HNSWIndex<DataType, DistType>::isolateDeletedElement;
    using HNSWIndex<DataType, DistType>::mutuallyUpdateForRepairedNode;
    using HNSWIndex<DataType, DistType>::repairNodeConnections;
};

struct TieredBackendAccess : VecSimTieredIndex<float, float> {
    using VecSimTieredIndex<float, float>::backendIndex;
};

HNSWIndex<float, float> *getBackend(TieredHNSWIndex<float, float> *index) {
    auto backend = &TieredBackendAccess::backendIndex;
    return dynamic_cast<HNSWIndex<float, float> *>(index->*backend);
}

LargeVector makeLargeVector(size_t value) {
    return {static_cast<float>(value), static_cast<float>(value % 7), static_cast<float>(value % 5),
            static_cast<float>(value % 3)};
}

SmallVector makeSmallVector(size_t value) {
    return {static_cast<float>(value), static_cast<float>(value)};
}

TieredHNSWIndex<float, float> *createIndex(tieredIndexMock &pool, size_t dimension,
                                           bool enterprise_params = false) {
    HNSWParams hnsw_params{.type = VecSimType_FLOAT32,
                           .dim = dimension,
                           .metric = VecSimMetric_L2,
                           .M = kM,
                           .efConstruction = kEfConstruction,
                           .efRuntime = enterprise_params ? 100 : kIndexEfRuntime};
    VecSimParams primary_params = CreateParams(hnsw_params);
    TieredIndexParams tiered_params = {
        .jobQueue = &pool.jobQ,
        .jobQueueCtx = pool.ctx,
        .submitCb = tieredIndexMock::submit_callback,
        .flatBufferLimit = enterprise_params ? 1024 : kDeferredThreshold,
        .primaryIndexParams = &primary_params,
        .specificParams = {
            TieredHNSWParams{.swapJobThreshold = enterprise_params ? 1024 : kDeferredThreshold}}};
    auto *index =
        reinterpret_cast<TieredHNSWIndex<float, float> *>(TieredFactory::NewIndex(&tiered_params));
    pool.ctx->index_strong_ref.reset(index);
    return index;
}

template <size_t Dimension>
std::vector<labelType> queryLabels(TieredHNSWIndex<float, float> *index, size_t k,
                                   const std::array<float, Dimension> &query) {
    VecSimQueryParams query_params{};
    query_params.hnswRuntimeParams.efRuntime = kQueryEfRuntime;
    std::unique_ptr<VecSimQueryReply> reply(index->topKQuery(query.data(), k, &query_params));
    EXPECT_EQ(VecSim_QueryReply_OK, reply->code);
    std::vector<labelType> labels;
    labels.reserve(reply->results.size());
    for (const auto &result : reply->results) {
        labels.push_back(result.id);
    }
    return labels;
}

std::vector<labelType> expectedLabels(size_t count, size_t replacements) {
    std::vector<labelType> expected;
    expected.reserve(count);
    for (size_t value = replacements; value < count; ++value) {
        expected.push_back(value);
    }
    for (size_t value = 0; value < replacements; ++value) {
        expected.push_back(10000 + value);
    }
    std::sort(expected.begin(), expected.end());
    return expected;
}

std::vector<labelType> expectedAdjacentLabels() {
    auto expected = expectedLabels(40, 0);
    expected.erase(std::find(expected.begin(), expected.end(), 20));
    expected.erase(std::find(expected.begin(), expected.end(), 21));
    expected.push_back(10020);
    expected.push_back(10021);
    std::sort(expected.begin(), expected.end());
    return expected;
}

bool expectExactLabels(std::vector<labelType> actual, const std::vector<labelType> &expected) {
    std::sort(actual.begin(), actual.end());
    const bool exact = actual == expected;
    EXPECT_EQ(expected, actual);
    EXPECT_TRUE(std::adjacent_find(actual.begin(), actual.end()) == actual.end());
    return exact;
}

template <size_t Dimension, typename VectorFactory>
void loadBackend(TieredHNSWIndex<float, float> *index, size_t count, VectorFactory make_vector) {
    for (size_t value = 0; value < count; ++value) {
        const std::array<float, Dimension> vector = make_vector(value);
        ASSERT_EQ(1, getBackend(index)->addVector(vector.data(), value));
    }
}

template <size_t Dimension, typename VectorFactory>
void replace(TieredHNSWIndex<float, float> *index, size_t value, VectorFactory make_vector) {
    const std::array<float, Dimension> vector = make_vector(value);
    ASSERT_EQ(1, index->deleteVector(value));
    ASSERT_EQ(1, index->addVector(vector.data(), 10000 + value));
}

std::string jobIdentity(const AsyncJob *job) {
    if (job->jobType == HNSW_REPAIR_NODE_CONNECTIONS_JOB) {
        const auto *repair = static_cast<const HNSWRepairJob *>(job);
        return "repair:" + std::to_string(repair->node_id) + ":" + std::to_string(repair->level);
    }
    if (job->jobType == HNSW_INSERT_VECTOR_JOB) {
        return "insert:" + std::to_string(static_cast<const HNSWInsertJob *>(job)->label);
    }
    return "unexpected";
}

std::unordered_map<const AsyncJob *, std::string> captureJobIdentities(tieredIndexMock &pool) {
    std::unordered_map<const AsyncJob *, std::string> identities;
    const size_t count = pool.jobQ.size();
    for (size_t i = 0; i < count; ++i) {
        auto managed_job = std::move(pool.jobQ.front());
        pool.jobQ.pop();
        identities.emplace(managed_job.job, jobIdentity(managed_job.job));
        pool.jobQ.push(std::move(managed_job));
    }
    return identities;
}

bool executeScheduledJob(tieredIndexMock &pool,
                         const std::unordered_map<const AsyncJob *, std::string> &identities,
                         const std::string &wanted) {
    const size_t count = pool.jobQ.size();
    for (size_t i = 0; i < count; ++i) {
        if (identities.at(pool.jobQ.front().job) == wanted) {
            pool.thread_iteration();
            return true;
        }
        auto managed_job = std::move(pool.jobQ.front());
        pool.jobQ.pop();
        pool.jobQ.push(std::move(managed_job));
    }
    ADD_FAILURE() << "recorded job is absent from queue: " << wanted;
    return false;
}

std::vector<std::string> loadRecoverySchedule() {
    const auto path = std::filesystem::path(__FILE__).parent_path() / "fixtures" /
                      "mod17015_recovery_schedule.txt";
    std::ifstream input(path);
    EXPECT_TRUE(input.is_open()) << path;
    std::vector<std::string> schedule;
    for (std::string identity; std::getline(input, identity);) {
        if (!identity.empty()) {
            schedule.push_back(identity);
        }
    }
    return schedule;
}

void expectIntegrity(TieredHNSWIndex<float, float> *index) {
    EXPECT_TRUE(getBackend(index)->checkIntegrity().valid_state);
}

} // namespace

TEST(DISABLED_HNSWWorkerTrace, EnterpriseFifoReplacementsRemainFullyReachable) {
    AsyncWriteModeGuard write_mode;
    tieredIndexMock pool(1);
    TieredCleanup cleanup{pool};
    auto *index = createIndex(pool, 4, true);
    auto *backend = getBackend(index);
    ASSERT_NE(nullptr, backend);
    const LargeVector origin{};
    std::set<labelType> expected;
    for (size_t value = 0; value < kLargeCount; ++value) {
        auto vector = makeLargeVector(value);
        ASSERT_EQ(1, index->addVector(vector.data(), value + 1));
        while (!pool.jobQ.empty()) {
            pool.thread_iteration();
        }
        expected.insert(value + 1);
    }
    expectIntegrity(index);

    std::filesystem::create_directories("worker-trace-results");
    std::ofstream trace("worker-trace-results/operations.jsonl");
    ASSERT_TRUE(trace.is_open());
    std::vector<std::string> previous_nodes;
    size_t step = 0;
    size_t callbacks = 0;
    size_t first_missing_step = 0;

    // Record deltas after serial operations so every snapshot is stable, including tombstones.
    auto observe = [&](const std::string &operation) {
        VecSimQueryParams params{};
        params.hnswRuntimeParams.efRuntime = kLargeCount;
        std::unique_ptr<VecSimQueryReply> reply(
            index->topKQuery(origin.data(), kLargeCount, &params));
        EXPECT_EQ(VecSim_QueryReply_OK, reply->code);
        std::set<labelType> found;
        for (const auto &result : reply->results) {
            found.insert(result.id);
        }
        std::vector<labelType> missing;
        std::set_difference(expected.begin(), expected.end(), found.begin(), found.end(),
                            std::back_inserter(missing));
        std::vector<labelType> extra;
        std::set_difference(found.begin(), found.end(), expected.begin(), expected.end(),
                            std::back_inserter(extra));
        EXPECT_TRUE(extra.empty());
        EXPECT_EQ(found.size(), reply->results.size());
        if (!missing.empty() && first_missing_step == 0) {
            first_missing_step = step;
        }
        VecSimQueryReply_Code code = VecSim_QueryReply_OK;
        const idType landing = backend->searchBottomLayerEP(origin.data(), nullptr, &code);
        EXPECT_EQ(VecSim_QueryReply_OK, code);
        const auto [entry, max_level] = backend->safeGetEntryPointState();
        trace << "{\"step\":" << step++ << ",\"operation\":\"" << operation
              << "\",\"callbacks\":" << callbacks << ",\"pending_jobs\":" << pool.jobQ.size()
              << ",\"count\":" << reply->results.size() << ",\"expected_count\":" << expected.size()
              << ",\"nearest\":" << (reply->results.empty() ? 0 : reply->results.front().id)
              << ",\"entry\":" << entry << ",\"max_level\":" << max_level
              << ",\"landing\":" << landing << ",\"missing\":[";
        for (size_t i = 0; i < missing.size(); ++i) {
            trace << (i ? "," : "") << missing[i];
        }
        trace << "],\"node_count\":" << backend->indexSize() << ",\"changed_nodes\":[";
        std::vector<std::string> nodes;
        bool first_change = true;
        for (idType id = 0; id < backend->indexSize(); ++id) {
            std::ostringstream node;
            node << "{\"id\":" << id << ",\"label\":" << backend->getExternalLabel(id)
                 << ",\"value\":"
                 << reinterpret_cast<const float *>(backend->getDataByInternalId(id))[0]
                 << ",\"deleted\":" << backend->isMarkedDeleted(id)
                 << ",\"in_process\":" << backend->isInProcess(id) << ",\"levels\":[";
            auto *graph = backend->getGraphDataByInternalId(id);
            for (size_t level = 0; level <= graph->toplevel; ++level) {
                node << (level ? ",[" : "[");
                const auto &links = backend->getElementLevelData(id, level);
                for (size_t i = 0; i < links.getNumLinks(); ++i) {
                    node << (i ? "," : "") << links.getLinkAtPos(i);
                }
                node << ']';
            }
            node << "]}";
            nodes.push_back(node.str());
            if (id >= previous_nodes.size() || previous_nodes[id] != nodes.back()) {
                trace << (first_change ? "" : ",") << nodes.back();
                first_change = false;
            }
        }
        trace << "]}\n";
        trace.flush();
        EXPECT_TRUE(trace.good());
        previous_nodes = std::move(nodes);
        return found.size();
    };

    const size_t initial_count = observe("initial");
    ASSERT_EQ(kLargeCount, initial_count);
    ::testing::Test::RecordProperty("initial_count", initial_count);
    for (size_t value = 0; value < 128; ++value) {
        ASSERT_EQ(1, index->deleteVector(value + 1));
        expected.erase(value + 1);
        observe("delete:" + std::to_string(value + 1));
        auto vector = makeLargeVector(value);
        ASSERT_EQ(1, index->addVector(vector.data(), kLargeCount + value + 1));
        expected.insert(kLargeCount + value + 1);
        observe("enqueue:" + std::to_string(kLargeCount + value + 1));
    }
    const size_t queued = pool.jobQ.size();
    EXPECT_EQ(286U, queued);
    observe("queued");
    while (!pool.jobQ.empty()) {
        const std::string identity = jobIdentity(pool.jobQ.front().job);
        pool.thread_iteration();
        ++callbacks;
        observe(identity);
    }
    const size_t before_gc = observe("final-idle");
    expectIntegrity(index);
    index->runGC();
    const size_t after_gc = observe("after-gc");
    expectIntegrity(index);
    ::testing::Test::RecordProperty("callback_count", callbacks);
    ::testing::Test::RecordProperty("first_missing_step", first_missing_step);
    ::testing::Test::RecordProperty("final_count_before_gc", before_gc);
    ::testing::Test::RecordProperty("final_count_after_gc", after_gc);
    EXPECT_EQ(queued, callbacks);
    EXPECT_EQ(kLargeCount, before_gc);
    EXPECT_EQ(kLargeCount, after_gc);
}

TEST(HNSWRepairChains, AdjacentReplacementsRemainFullyReachable) {
    AsyncWriteModeGuard write_mode;
    tieredIndexMock pool(1);
    TieredCleanup cleanup{pool};
    auto *index = createIndex(pool, 2);
    loadBackend<2>(index, 40, makeSmallVector);

    const SmallVector origin{};
    const auto initial = queryLabels(index, 40, origin);
    ::testing::Test::RecordProperty("initial_count", initial.size());
    expectExactLabels(initial, expectedLabels(40, 0));
    expectIntegrity(index);

    replace<2>(index, 20, makeSmallVector);
    replace<2>(index, 21, makeSmallVector);
    while (!pool.jobQ.empty()) {
        pool.thread_iteration();
    }

    const auto expected = expectedAdjacentLabels();
    const auto before_gc = queryLabels(index, 40, origin);
    ::testing::Test::RecordProperty("final_count_before_gc", before_gc.size());
    const bool before_exact = expectExactLabels(before_gc, expected);
    index->runGC();
    const auto after_gc = queryLabels(index, 40, origin);
    ::testing::Test::RecordProperty("final_count_after_gc", after_gc.size());
    const bool after_exact = expectExactLabels(after_gc, expected);
    EXPECT_TRUE(before_exact);
    EXPECT_TRUE(after_exact);
    expectIntegrity(index);
}

TEST(HNSWRepairChains, BatchedFiveThousandVectorReplacementsRecoverAfterIdleAndGC) {
    AsyncWriteModeGuard write_mode;
    tieredIndexMock pool(1);
    TieredCleanup cleanup{pool};
    auto *index = createIndex(pool, 4);
    loadBackend<4>(index, kLargeCount, makeLargeVector);

    const LargeVector origin{};
    const auto initial = queryLabels(index, kLargeCount, origin);
    ::testing::Test::RecordProperty("initial_count", initial.size());
    expectExactLabels(initial, expectedLabels(kLargeCount, 0));
    expectIntegrity(index);

    size_t callbacks = 0;
    for (size_t first = 0; first < 16; first += 8) {
        for (size_t value = first; value < first + 8; ++value) {
            replace<4>(index, value, makeLargeVector);
        }
        while (!pool.jobQ.empty()) {
            pool.thread_iteration();
            ++callbacks;
        }
    }

    const auto expected = expectedLabels(kLargeCount, 16);
    const auto before_gc = queryLabels(index, kLargeCount, origin);
    ::testing::Test::RecordProperty("callback_count", callbacks);
    ::testing::Test::RecordProperty("final_count_before_gc", before_gc.size());
    const bool before_exact = expectExactLabels(before_gc, expected);
    index->runGC();
    const auto after_gc = queryLabels(index, kLargeCount, origin);
    ::testing::Test::RecordProperty("final_count_after_gc", after_gc.size());
    const bool after_exact = expectExactLabels(after_gc, expected);
    EXPECT_TRUE(before_exact);
    EXPECT_TRUE(after_exact);
    expectIntegrity(index);
}

TEST(HNSWRepairChains, PendingFiveThousandVectorReplacementJobsPreserveReachability) {
#ifndef __GLIBCXX__
    GTEST_SKIP() << "the recorded internal-node schedule depends on libstdc++'s "
                    "std::default_random_engine sequence";
#endif
    AsyncWriteModeGuard write_mode;
    tieredIndexMock pool(1);
    TieredCleanup cleanup{pool};
    auto *index = createIndex(pool, 4);
    loadBackend<4>(index, kLargeCount, makeLargeVector);

    const LargeVector origin{};
    const auto initial = queryLabels(index, kLargeCount, origin);
    ::testing::Test::RecordProperty("initial_count", initial.size());
    expectExactLabels(initial, expectedLabels(kLargeCount, 0));
    expectIntegrity(index);

    for (size_t value = 0; value < 128; ++value) {
        replace<4>(index, value, makeLargeVector);
    }
    const auto schedule = loadRecoverySchedule();
    const auto identities = captureJobIdentities(pool);
    EXPECT_EQ(286U, schedule.size());
    EXPECT_EQ(schedule.size(), pool.jobQ.size());
    EXPECT_EQ(schedule.size(), identities.size());

    size_t pending_nearest_misses = 0;
    size_t pending_incomplete_queries = 0;
    size_t pending_min_count = kLargeCount;
    size_t callbacks = 0;
    for (const auto &identity : schedule) {
        if (!executeScheduledJob(pool, identities, identity)) {
            break;
        }
        ++callbacks;
        const auto labels = queryLabels(index, kLargeCount, origin);
        pending_min_count = std::min(pending_min_count, labels.size());
        pending_nearest_misses += labels.empty() || labels.front() != 10000;
        pending_incomplete_queries += labels.size() != kLargeCount;
    }

    ::testing::Test::RecordProperty("pending_callback_count", callbacks);
    ::testing::Test::RecordProperty("pending_nearest_miss_count", pending_nearest_misses);
    ::testing::Test::RecordProperty("pending_incomplete_query_count", pending_incomplete_queries);
    ::testing::Test::RecordProperty("pending_min_result_count", pending_min_count);
    EXPECT_EQ(schedule.size(), callbacks);
    EXPECT_EQ(0U, pending_nearest_misses);
    EXPECT_EQ(0U, pending_incomplete_queries);
    EXPECT_TRUE(pool.jobQ.empty());

    const auto expected = expectedLabels(kLargeCount, 128);
    const auto before_gc = queryLabels(index, kLargeCount, origin);
    ::testing::Test::RecordProperty("final_count_before_gc", before_gc.size());
    const bool before_exact = expectExactLabels(before_gc, expected);
    index->runGC();
    const auto after_gc = queryLabels(index, kLargeCount, origin);
    ::testing::Test::RecordProperty("final_count_after_gc", after_gc.size());
    const bool after_exact = expectExactLabels(after_gc, expected);
    EXPECT_TRUE(before_exact);
    EXPECT_TRUE(after_exact);
    expectIntegrity(index);
}

TEST(HNSWRepairChains, PendingDeletedNodeRepairCannotReconnectAfterIsolation) {
    HNSWParams params{.type = VecSimType_FLOAT32,
                      .dim = 2,
                      .metric = VecSimMetric_L2,
                      .M = kM,
                      .efConstruction = kEfConstruction,
                      .efRuntime = kIndexEfRuntime};
    VecSimParams index_params = CreateParams(params);
    std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)> index(VecSimIndex_New(&index_params),
                                                                    VecSimIndex_Free);
    auto *hnsw = dynamic_cast<HNSWIndex_Single<float, float> *>(index.get());
    ASSERT_NE(nullptr, hnsw);
    for (size_t value = 0; value < 40; ++value) {
        const auto vector = makeSmallVector(value);
        ASSERT_EQ(1, hnsw->addVector(vector.data(), value));
    }

    const auto deleted_ids = hnsw->markDelete(20);
    ASSERT_EQ(1U, deleted_ids.size());
    const idType deleted_id = deleted_ids.front();
    auto repair = &RepairAccess<float, float>::repairNodeConnections;
    for (idType id = 0; id < 40; ++id) {
        if (id == deleted_id) {
            continue;
        }
        for (size_t level = 0; level <= hnsw->getGraphDataByInternalId(id)->toplevel; ++level) {
            (hnsw->*repair)(id, level);
        }
    }

    auto stale_links = hnsw->getElementLevelData(deleted_id, 0).copyLinks();
    ASSERT_FALSE(stale_links.empty());
    const idType chosen_id = stale_links.front();
    vecsim_stl::vector<idType> nodes_to_update(hnsw->getAllocator());
    nodes_to_update.insert(nodes_to_update.end(), stale_links.begin(), stale_links.end());
    vecsim_stl::vector<idType> chosen_neighbors(hnsw->getAllocator());
    chosen_neighbors.push_back(chosen_id);

    auto isolate = &RepairAccess<float, float>::isolateDeletedElement;
    (hnsw->*isolate)(deleted_id);
    auto commit = &RepairAccess<float, float>::mutuallyUpdateForRepairedNode;
    (hnsw->*commit)(deleted_id, 0, nodes_to_update, chosen_neighbors, 2 * hnsw->getM());

    const auto &deleted_level = hnsw->getElementLevelData(deleted_id, 0);
    const auto &chosen_level = hnsw->getElementLevelData(chosen_id, 0);
    EXPECT_EQ(0U, deleted_level.getNumLinks());
    EXPECT_TRUE(deleted_level.getIncomingEdges().empty());
    EXPECT_EQ(chosen_level.getIncomingEdges().end(),
              std::find(chosen_level.getIncomingEdges().begin(),
                        chosen_level.getIncomingEdges().end(), deleted_id));
    EXPECT_TRUE(hnsw->checkIntegrity().valid_state);
}

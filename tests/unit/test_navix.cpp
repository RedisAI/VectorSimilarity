/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/query_result_definitions.h"
#include <map>
#include <set>

namespace {
constexpr size_t labelBase = 10000;

struct Fixture {
    VecSimIndex *index;
    HNSWIndex<float, float> *graph;
    idType seed;
    std::set<size_t> allowed;
    std::map<size_t, size_t> filter_calls;
    VecSimNavixStats stats{};
    VecSimQueryParams params{};
    float query = 0;

    Fixture() {
        VecSimParams create{};
        create.algo = VecSimAlgo_HNSWLIB;
        create.algoParams.hnswParams = {.type = VecSimType_FLOAT32,
                                        .dim = 1,
                                        .metric = VecSimMetric_L2,
                                        .M = 4,
                                        .efConstruction = 40,
                                        .efRuntime = 1};
        index = VecSimIndex_New(&create);
        for (size_t i = 0; i < 24; ++i)
            VecSimIndex_AddVector(index, &query, labelBase + i);
        graph = static_cast<HNSWIndex<float, float> *>(index);
        seed = graph->getEntryPointLabel() - labelBase;
        for (idType i = 0; i < 24; ++i)
            graph->getElementLevelData(i, 0).numLinks = 0;
        params.hnswRuntimeParams.efRuntime = 1;
        params.navix = {[](void *ctx, size_t label) {
                            auto *fixture = static_cast<Fixture *>(ctx);
                            ++fixture->filter_calls[label];
                            return fixture->allowed.contains(label) ? 1 : 0;
                        },
                        this, &stats, false};
    }
    ~Fixture() { VecSimIndex_Free(index); }
    idType node(size_t offset) const { return (seed + offset) % 24; }
    void accept(size_t offset) { allowed.insert(labelBase + node(offset)); }
    void edges(size_t offset, std::initializer_list<size_t> targets) {
        auto &level = graph->getElementLevelData(node(offset), 0);
        level.numLinks = 0;
        for (auto target : targets)
            level.links[level.numLinks++] = node(target);
    }
    VecSimBatchIterator *iterator() { return VecSimBatchIterator_New(index, &query, &params); }
    std::set<size_t> drain(VecSimBatchIterator *it) {
        std::set<size_t> seen;
        size_t batches = 0;
        while (VecSimBatchIterator_HasNext(it) && batches++ < 30) {
            auto *reply = VecSimBatchIterator_Next(it, 1, BY_ID);
            EXPECT_EQ(reply->code, VecSim_QueryReply_OK);
            for (const auto &result : reply->results) {
                EXPECT_TRUE(allowed.contains(result.id));
                EXPECT_TRUE(seen.insert(result.id).second);
            }
            VecSimQueryReply_Free(reply);
        }
        EXPECT_LT(batches, 30u);
        EXPECT_FALSE(VecSimBatchIterator_HasNext(it));
        return seen;
    }
    void sparseGraph() {
        edges(0, {1, 2});
        edges(1, {3, 4, 0});
        edges(2, {4, 3});
        edges(3, {1, 4});
        edges(4, {2, 3});
        accept(3);
        accept(4);
    }
};
} // namespace

TEST(Navix, RejectedSeedBridgesCyclesBatchesAndReset) {
    Fixture f;
    f.sparseGraph();
    auto *it = f.iterator();
    EXPECT_EQ(f.drain(it), f.allowed);
    EXPECT_GT(f.stats.blindExpansions, 0u);
    EXPECT_GT(f.stats.secondHopVisits, 0u);
    VecSimBatchIterator_Reset(it);
    EXPECT_EQ(f.drain(it), f.allowed);
    VecSimBatchIterator_Free(it);
}

TEST(Navix, AdaptiveModesAndNoMatches) {
    for (size_t accepted : {0u, 2u, 4u}) {
        Fixture f;
        f.edges(0, {1, 2, 3, 4});
        for (size_t i = 1; i <= accepted; ++i)
            f.accept(i);
        auto *it = f.iterator();
        EXPECT_EQ(f.drain(it), f.allowed);
        if (accepted == 2)
            EXPECT_GT(f.stats.directedExpansions, 0u);
        if (accepted == 4)
            EXPECT_GT(f.stats.oneHopExpansions, 0u);
        VecSimBatchIterator_Free(it);
    }
}

TEST(Navix, CachesBothDecisionsAcrossBatchesAndInvalidatesOnReset) {
    for (bool blind : {false, true}) {
        Fixture f;
        f.sparseGraph();
        f.params.navix.forceBlind = blind;
        auto *it = f.iterator();
        EXPECT_EQ(f.drain(it), f.allowed);
        EXPECT_EQ(f.filter_calls.size(), 5u);
        for (const auto &[label, calls] : f.filter_calls)
            EXPECT_EQ(calls, 1u) << label;
        EXPECT_EQ(f.stats.filterChecks, 5u);
        EXPECT_EQ(f.stats.filterRejected, 3u);

        VecSimBatchIterator_Reset(it);
        EXPECT_EQ(f.drain(it), f.allowed);
        for (const auto &[label, calls] : f.filter_calls)
            EXPECT_EQ(calls, 2u) << label;
        EXPECT_EQ(f.stats.filterChecks, 10u);
        EXPECT_EQ(f.stats.filterRejected, 6u);
        VecSimBatchIterator_Free(it);
    }
}

TEST(Navix, FilterCacheIsLocalToEachIterator) {
    Fixture f;
    f.sparseGraph();
    auto *first = f.iterator();
    auto *second = f.iterator();
    EXPECT_EQ(f.drain(first), f.allowed);
    EXPECT_EQ(f.drain(second), f.allowed);
    EXPECT_EQ(f.filter_calls.size(), 5u);
    for (const auto &[label, calls] : f.filter_calls)
        EXPECT_EQ(calls, 2u) << label;
    VecSimBatchIterator_Free(first);
    VecSimBatchIterator_Free(second);
}

TEST(Navix, FilterCacheGrowsForNewBackendNodes) {
    Fixture f;
    f.sparseGraph();
    const auto new_id = f.graph->indexSize();
    f.allowed.insert(labelBase + new_id);
    auto *it = f.iterator();
    auto *reply = VecSimBatchIterator_Next(it, 1, BY_ID);
    ASSERT_EQ(reply->results.size(), 1u);
    const auto first_label = reply->results[0].id;
    VecSimQueryReply_Free(reply);

    const auto capacity = f.graph->indexCapacity();
    VecSimIndex_AddVector(f.index, &f.query, labelBase + new_id);
    EXPECT_EQ(f.graph->indexCapacity(), capacity);
    // Simulate an appended migration becoming reachable from the saved frontier.
    for (idType id = 0; id < f.graph->indexSize(); ++id)
        f.graph->getElementLevelData(id, 0).numLinks = 0;
    for (auto label : f.allowed) {
        if (label == first_label || label == labelBase + new_id)
            continue;
        auto &links = f.graph->getElementLevelData(label - labelBase, 0);
        links.links[0] = new_id;
        links.numLinks = 1;
    }
    auto expected = f.allowed;
    expected.erase(first_label);
    EXPECT_EQ(f.drain(it), expected);
    EXPECT_EQ(f.filter_calls[labelBase + new_id], 1u);
    VecSimBatchIterator_Free(it);
}

TEST(Navix, TimeoutInsideSecondHopThenReset) {
    Fixture f;
    f.sparseGraph();
    f.params.timeoutCtx = &f.stats;
    VecSim_SetTimeoutCallbackFunction([](void *ctx) {
        return static_cast<VecSimNavixStats *>(ctx)->secondHopVisits > 0 ? 1 : 0;
    });
    auto *it = f.iterator();
    auto *reply = VecSimBatchIterator_Next(it, 1, BY_ID);
    EXPECT_EQ(reply->code, VecSim_QueryReply_TimedOut);
    VecSimQueryReply_Free(reply);
    VecSim_SetTimeoutCallbackFunction([](void *) { return 0; });
    VecSimBatchIterator_Reset(it);
    EXPECT_EQ(f.drain(it), f.allowed);
    VecSimBatchIterator_Free(it);
}

TEST(Navix, FlatFrontendFiltersAndResets) {
    VecSimParams create{};
    create.algo = VecSimAlgo_BF;
    create.algoParams.bfParams = {.type = VecSimType_FLOAT32, .dim = 1, .metric = VecSimMetric_L2};
    auto *index = VecSimIndex_New(&create);
    for (size_t i = 0; i < 10; ++i) {
        float value = i;
        VecSimIndex_AddVector(index, &value, labelBase + i);
    }
    VecSimNavixStats stats{};
    VecSimQueryParams params{};
    params.navix = {[](void *, size_t label) { return label == labelBase + 7 ? 1 : 0; }, nullptr,
                    &stats, false};
    float query = 0;
    auto *it = VecSimBatchIterator_New(index, &query, &params);
    for (size_t repeat = 0; repeat < 2; ++repeat) {
        auto *reply = VecSimBatchIterator_Next(it, 3, BY_ID);
        EXPECT_EQ(reply->results.size(), 1u);
        if (reply->results.size() == 1)
            EXPECT_EQ(reply->results[0].id, labelBase + 7);
        VecSimQueryReply_Free(reply);
        EXPECT_FALSE(VecSimBatchIterator_HasNext(it));
        VecSimBatchIterator_Reset(it);
    }
    EXPECT_EQ(stats.frontendCandidates, 2u);
    VecSimBatchIterator_Free(it);
    params.navix.mayMatch = [](void *, size_t) { return 0; };
    it = VecSimBatchIterator_New(index, &query, &params);
    for (size_t repeat = 0; repeat < 2; ++repeat) {
        EXPECT_TRUE(VecSimBatchIterator_HasNext(it));
        auto *reply = VecSimBatchIterator_Next(it, 3, BY_ID);
        EXPECT_TRUE(reply->results.empty());
        VecSimQueryReply_Free(reply);
        EXPECT_FALSE(VecSimBatchIterator_HasNext(it));
        VecSimBatchIterator_Reset(it);
    }
    VecSimBatchIterator_Free(it);
    VecSimIndex_Free(index);
}

TEST(Navix, SkipsCompletedNeighborhoodsAcrossBatchesAndResets) {
    size_t baseline_visits = 0;
    for (bool skip : {false, true}) {
        Fixture f;
        f.sparseGraph();
        f.params.navix.skipScanned = skip;
        auto *it = f.iterator();
        EXPECT_EQ(f.drain(it), f.allowed);
        if (!skip) {
            baseline_visits = f.stats.secondHopVisits;
        } else {
            EXPECT_LT(f.stats.secondHopVisits, baseline_visits);
            EXPECT_GT(f.stats.secondHopSkipped, 0u);
            const auto visits = f.stats.secondHopVisits;
            VecSimBatchIterator_Reset(it);
            EXPECT_EQ(f.drain(it), f.allowed);
            EXPECT_EQ(f.stats.secondHopVisits, 2 * visits);
        }
        VecSimBatchIterator_Free(it);
    }
}

TEST(Navix, AdmissionLimitDoesNotCompletePartialNeighborhood) {
    Fixture f;
    f.params.navix.forceBlind = true;
    f.params.navix.skipScanned = true;
    f.edges(0, {1, 2, 3, 4, 5, 6, 7, 8});
    f.edges(1, {9, 10});
    f.edges(2, {1});
    for (size_t i = 2; i <= 10; ++i)
        f.accept(i);
    auto *it = f.iterator();
    EXPECT_EQ(f.drain(it), f.allowed);
    VecSimBatchIterator_Free(it);
}

TEST(Navix, SecondHopBudgetBoundsInspectionsAndLeavesPartialListsEligible) {
    Fixture f;
    f.params.navix.forceBlind = true;
    f.params.navix.skipScanned = true;
    f.params.navix.secondHopBudget = 2;
    f.edges(0, {1, 2});
    f.edges(1, {3, 4, 5});
    f.edges(2, {1});
    f.accept(2);
    f.accept(4);
    auto *it = f.iterator();
    EXPECT_EQ(f.drain(it), f.allowed);
    // Both expansions reach the still-partial list through rejected node 1.
    EXPECT_EQ(f.stats.secondHopBudgetExhausted, 2u);
    EXPECT_EQ(f.stats.secondHopVisits, 4u);
    EXPECT_LE(f.stats.secondHopVisits, f.stats.expandedNodes * f.params.navix.secondHopBudget);
    VecSimBatchIterator_Reset(it);
    EXPECT_EQ(f.drain(it), f.allowed);
    VecSimBatchIterator_Free(it);
}

TEST(Navix, OneHopBridgesRejectedChainsWithoutReturningThem) {
    for (bool matches : {false, true}) {
        Fixture f;
        f.params.navix.bridgeRejected = true;
        f.edges(0, {1});
        f.edges(1, {2, 0});
        f.edges(2, {3, 1});
        f.edges(3, {4, 2});
        if (matches) {
            f.accept(3);
            f.accept(4);
        }
        auto *it = f.iterator();
        EXPECT_EQ(f.drain(it), f.allowed);
        EXPECT_GT(f.stats.bridgeExpansions, 0u);
        EXPECT_EQ(f.stats.secondHopVisits, 0u);
        VecSimBatchIterator_Reset(it);
        EXPECT_EQ(f.drain(it), f.allowed);
        VecSimBatchIterator_Free(it);
    }
}

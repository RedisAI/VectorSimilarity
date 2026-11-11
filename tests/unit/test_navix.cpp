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
#include <set>

namespace {
constexpr size_t labelBase = 10000;

struct Fixture {
    VecSimIndex *index;
    HNSWIndex<float, float> *graph;
    idType seed;
    std::set<size_t> allowed;
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
                            return static_cast<Fixture *>(ctx)->allowed.contains(label) ? 1 : 0;
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
    VecSimIndex_Free(index);
}

/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/index_factories/components/components_factory.h"
#include "VecSim/index_factories/factory_utils.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/vec_sim.h"
#include "mock_thread_pool.h"
#include "unit_test_utils.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <thread>
#include <vector>

namespace {

constexpr size_t kDimension = 4;
constexpr size_t kBlockSize = 3;

class CapacityTestIndex final : public HNSWIndex_Single<float, float> {
    using Base = HNSWIndex_Single<float, float>;

public:
    CapacityTestIndex(const HNSWParams *params, const AbstractIndexInitParams &init,
                      const IndexComponents<float, float> &components)
        : Base(params, init, components, 100) {}

    using Base::growByBlock;
    using Base::storeVector;

    int insertStored(const float *vector, labelType label) {
        const auto state = this->storeVector(vector, label);
        this->indexVector(vector, label, state);
        if (state.currMaxLevel < state.elementMaxLevel) {
            this->unlockIndexDataGuard();
        }
        return 1;
    }

    void createVisitedHandlers(size_t count) {
        std::vector<VisitedNodesHandler *> handlers;
        handlers.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            handlers.push_back(this->getVisitedList());
        }
        for (auto *handler : handlers) {
            this->returnVisitedList(handler);
        }
    }

    void ensureVisitedCapacity(size_t capacity) {
        this->visitedNodesHandlerPool.ensureCapacity(capacity);
    }

    bool tryAcquireIndexGuard() {
        if (!this->indexDataGuard.try_lock()) {
            return false;
        }
        this->indexDataGuard.unlock();
        return true;
    }

    void tagHandlers(size_t old_id, size_t new_id) {
        std::vector<VisitedNodesHandler *> handlers;
        handlers.push_back(this->getVisitedList());
        handlers.push_back(this->getVisitedList());
        for (auto *handler : handlers) {
            const tag_t tag = handler->getFreshTag();
            handler->tagNode(static_cast<unsigned int>(old_id), tag);
            EXPECT_EQ(handler->getNodeTag(static_cast<unsigned int>(old_id)), tag);
            const tag_t new_tag = handler->getFreshTag();
            handler->tagNode(static_cast<unsigned int>(new_id), new_tag);
            EXPECT_EQ(handler->getNodeTag(static_cast<unsigned int>(new_id)), new_tag);
        }
        for (auto *handler : handlers) {
            this->returnVisitedList(handler);
        }
    }
};

struct IndexDeleter {
    void operator()(CapacityTestIndex *index) const {
        if (index != nullptr) {
            VecSimIndex_Free(index);
        }
    }
};

using IndexPtr = std::unique_ptr<CapacityTestIndex, IndexDeleter>;

IndexPtr makeIndex() {
    HNSWParams params = {.type = VecSimType_FLOAT32,
                         .dim = kDimension,
                         .metric = VecSimMetric_L2,
                         .blockSize = kBlockSize};
    auto init = VecSimFactory::NewAbstractInitParams(&params, nullptr, false);
    auto components =
        CreateIndexComponents<float, float>(init.allocator, params.metric, params.dim, false);
    return IndexPtr(new (init.allocator) CapacityTestIndex(&params, init, components));
}

void addVectors(CapacityTestIndex *index, size_t count) {
    const size_t start = index->indexSize();
    for (size_t i = start; i < start + count; ++i) {
        const float vector[kDimension] = {static_cast<float>(i), 0.0F, 0.0F, 0.0F};
        ASSERT_EQ(index->insertStored(vector, i), 1);
    }
}

void expectSearchAndLabels(CapacityTestIndex *index, size_t expected_size,
                           labelType expected_label = 0) {
    ASSERT_EQ(index->indexSize(), expected_size);
    ASSERT_TRUE(index->isLabelExists(expected_label));
    std::vector<std::vector<float>> stored;
    index->getDataByLabel(expected_label, stored);
    ASSERT_EQ(stored.size(), 1);

    const float query[kDimension] = {static_cast<float>(expected_label), 0.0F, 0.0F, 0.0F};
    auto *reply = VecSimIndex_TopKQuery(index, query, 1, nullptr, BY_SCORE);
    ASSERT_NE(reply, nullptr);
    EXPECT_EQ(VecSimQueryReply_GetCode(reply), VecSim_QueryReply_OK);
    EXPECT_EQ(VecSimQueryReply_Len(reply), 1);
    auto *iterator = VecSimQueryReply_GetIterator(reply);
    auto *result = VecSimQueryReply_IteratorNext(iterator);
    EXPECT_NE(result, nullptr);
    if (result) {
        EXPECT_EQ(VecSimQueryResult_GetId(result), expected_label);
        EXPECT_EQ(VecSimQueryResult_GetScore(result), 0.0);
    }
    VecSimQueryReply_IteratorFree(iterator);
    VecSimQueryReply_Free(reply);
    EXPECT_TRUE(index->checkIntegrity().valid_state);
}

struct MemoryHookState {
    size_t allocation_calls = 0;
    size_t matching_malloc_calls = 0;
    size_t failed_allocations = 0;
    size_t fail_from_call = SIZE_MAX;
    size_t fail_malloc_size = SIZE_MAX;
    size_t fail_matching_malloc_call = SIZE_MAX;
};

MemoryHookState *active_memory_hooks = nullptr;

bool failAllocation(MemoryHookState &state, size_t size) {
    const size_t call = state.allocation_calls++;
    if (call >= state.fail_from_call) {
        state.failed_allocations++;
        return true;
    }
    if (size == state.fail_malloc_size) {
        const size_t matching_call = state.matching_malloc_calls++;
        if (matching_call == state.fail_matching_malloc_call) {
            state.failed_allocations++;
            return true;
        }
    }
    return false;
}

void *hookMalloc(size_t size) {
    if (active_memory_hooks != nullptr && failAllocation(*active_memory_hooks, size)) {
        return nullptr;
    }
    return std::malloc(size);
}

void *hookCalloc(size_t count, size_t size) {
    if (active_memory_hooks != nullptr && failAllocation(*active_memory_hooks, count * size)) {
        return nullptr;
    }
    return std::calloc(count, size);
}

void *hookRealloc(void *pointer, size_t size) {
    if (active_memory_hooks != nullptr && failAllocation(*active_memory_hooks, size)) {
        return nullptr;
    }
    return std::realloc(pointer, size);
}

void hookFree(void *pointer) { std::free(pointer); }

class ScopedMemoryHooks {
public:
    explicit ScopedMemoryHooks(MemoryHookState &state) {
        active_memory_hooks = &state;
        VecSim_SetMemoryFunctions({hookMalloc, hookCalloc, hookRealloc, hookFree});
    }

    ~ScopedMemoryHooks() {
        VecSim_SetMemoryFunctions({std::malloc, std::calloc, std::realloc, std::free});
        active_memory_hooks = nullptr;
    }
};

size_t countGrowthAllocations() {
    auto index = makeIndex();
    addVectors(index.get(), 3);
    index->createVisitedHandlers(2);
    index->growByBlock();
    addVectors(index.get(), 3);

    MemoryHookState state;
    {
        ScopedMemoryHooks hooks(state);
        index->growByBlock();
    }
    return state.allocation_calls;
}

IndexPtr makePreparedIndex() {
    auto index = makeIndex();
    addVectors(index.get(), 3);
    index->createVisitedHandlers(2);
    index->growByBlock();
    addVectors(index.get(), 3);
    return index;
}

bool guardAvailableFromAnotherThread(CapacityTestIndex *index) {
    std::atomic_bool available = false;
    std::thread checker([&] { available = index->tryAcquireIndexGuard(); });
    checker.join();
    return available;
}

} // namespace

TEST(HNSWCapacityTest, PersistentFailureLeavesPreviousCapacityUsable) {
    const size_t growth_allocations = countGrowthAllocations();
    ASSERT_GT(growth_allocations, 0);

    for (size_t failure_call = 0; failure_call < growth_allocations; ++failure_call) {
        auto index = makePreparedIndex();

        MemoryHookState state{.fail_from_call = failure_call};
        {
            ScopedMemoryHooks hooks(state);
            const float vector[kDimension] = {99.0F, 0.0F, 0.0F, 0.0F};
            EXPECT_THROW(index->insertStored(vector, 99), std::bad_alloc);
        }

        EXPECT_EQ(index->indexCapacity(), kBlockSize * 2);
        EXPECT_GE(index->indexMetaDataCapacity(), index->indexCapacity());
        EXPECT_TRUE(guardAvailableFromAnotherThread(index.get()));
        index->tagHandlers(5, 5);
        expectSearchAndLabels(index.get(), kBlockSize * 2);

        const float retry_vector[kDimension] = {100.0F, 0.0F, 0.0F, 0.0F};
        EXPECT_EQ(index->insertStored(retry_vector, 100), 1);
        EXPECT_TRUE(index->isLabelExists(100));
        index->tagHandlers(5, 8);
        expectSearchAndLabels(index.get(), kBlockSize * 2 + 1);
    }
}

TEST(HNSWCapacityTest, SecondVisitedHandlerFailureFallsBackToRequiredCapacity) {
    auto index = makePreparedIndex();

    MemoryHookState state{.fail_malloc_size =
                              sizeof(tag_t) * 12 + VecSimAllocator::getAllocationOverheadSize(),
                          .fail_matching_malloc_call = 1};
    {
        ScopedMemoryHooks hooks(state);
        const float vector[kDimension] = {99.0F, 0.0F, 0.0F, 0.0F};
        EXPECT_EQ(index->insertStored(vector, 99), 1);
    }

    EXPECT_EQ(state.failed_allocations, 1);
    EXPECT_EQ(index->indexSize(), 7);
    EXPECT_EQ(index->indexCapacity(), 9);
    EXPECT_EQ(index->indexMetaDataCapacity(), 9);
    EXPECT_TRUE(guardAvailableFromAnotherThread(index.get()));
    index->tagHandlers(5, 8);
    expectSearchAndLabels(index.get(), 7);
}

TEST(HNSWCapacityTest, GrowDeleteRegrowAndFitMemoryWithSmallBlocks) {
    auto index = makeIndex();
    addVectors(index.get(), 7);
    EXPECT_EQ(index->indexCapacity(), 9);
    EXPECT_EQ(index->indexMetaDataCapacity(), 12);

    for (labelType label = 0; label < 4; ++label) {
        EXPECT_EQ(index->deleteVector(label), 1);
    }
    EXPECT_EQ(index->indexSize(), 3);
    EXPECT_EQ(index->indexCapacity(), 3);
    EXPECT_EQ(index->indexMetaDataCapacity(), 6);

    index->fitMemory();
    EXPECT_EQ(index->indexMetaDataCapacity(), 3);

    const float vector[kDimension] = {100.0F, 0.0F, 0.0F, 0.0F};
    EXPECT_EQ(index->insertStored(vector, 100), 1);
    EXPECT_EQ(index->indexCapacity(), 6);
    EXPECT_EQ(index->indexMetaDataCapacity(), 6);
    expectSearchAndLabels(index.get(), 4, 4);

    auto allocator = index->getAllocator();
    VecSimIndex_Free(index.release());
    EXPECT_EQ(allocator->getAllocationSize(), sizeof(VecSimAllocator));
}

TEST(HNSWCapacityTest, FailedShrinkCanBeFollowedByGrowth) {
    auto index = makeIndex();
    addVectors(index.get(), 15);
    EXPECT_EQ(index->indexCapacity(), 15);
    EXPECT_EQ(index->indexMetaDataCapacity(), 24);
    index->createVisitedHandlers(2);
    index->ensureVisitedCapacity(index->indexCapacity());

    for (labelType label = 0; label < 8; ++label) {
        EXPECT_EQ(index->deleteVector(label), 1);
    }
    EXPECT_EQ(index->indexSize(), 7);
    EXPECT_EQ(index->indexCapacity(), 9);

    MemoryHookState state{.fail_malloc_size =
                              sizeof(tag_t) * 12 + VecSimAllocator::getAllocationOverheadSize(),
                          .fail_matching_malloc_call = 0};
    {
        ScopedMemoryHooks hooks(state);
        EXPECT_EQ(index->deleteVector(8), 1);
    }

    EXPECT_EQ(state.failed_allocations, 1);
    EXPECT_EQ(index->indexSize(), 6);
    EXPECT_EQ(index->indexCapacity(), 6);
    EXPECT_GE(index->indexMetaDataCapacity(), index->indexCapacity());
    EXPECT_TRUE(guardAvailableFromAnotherThread(index.get()));
    expectSearchAndLabels(index.get(), 6, 9);

    const float vector[kDimension] = {100.0F, 0.0F, 0.0F, 0.0F};
    EXPECT_EQ(index->insertStored(vector, 100), 1);
    EXPECT_EQ(index->indexCapacity(), 9);
    EXPECT_GE(index->indexMetaDataCapacity(), index->indexCapacity());
    index->tagHandlers(5, 8);
    expectSearchAndLabels(index.get(), 7, 9);
}

TEST(HNSWCapacityTest, FailedVisitedResizePreservesTagsAndAllocation) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    {
        VisitedNodesHandler handler(6, allocator);
        const tag_t tag = handler.getFreshTag();
        handler.tagNode(5, tag);
        auto *original_tags = handler.getElementsTags();
        const auto original_bytes = allocator->getAllocationSize();
        MemoryHookState state{.fail_from_call = 0};
        {
            ScopedMemoryHooks hooks(state);
            EXPECT_THROW(handler.resize(12), std::bad_alloc);
        }
        EXPECT_EQ(handler.getElementsTags(), original_tags);
        EXPECT_EQ(handler.getNodeTag(5), tag);
        EXPECT_EQ(allocator->getAllocationSize(), original_bytes);
        handler.resize(12);
        EXPECT_EQ(handler.getNodeTag(5), 0);
        handler.tagNode(11, handler.getFreshTag());
        handler.resize(0);
        EXPECT_EQ(handler.getElementsTags(), nullptr);
        handler.reset();
        handler.resize(3);
        handler.tagNode(2, handler.getFreshTag());
    }
    EXPECT_EQ(allocator->getAllocationSize(), sizeof(VecSimAllocator));
}

TEST(HNSWCapacityTest, TieredGrowthFailureReleasesLocks) {
    for (const bool release_flat_guard : {false, true}) {
        SCOPED_TRACE(release_flat_guard ? "queued insertion" : "direct insertion");
        tieredIndexMock mock_thread_pool;
        HNSWParams hnsw_params = {.type = VecSimType_FLOAT32,
                                  .dim = kDimension,
                                  .metric = VecSimMetric_L2,
                                  .blockSize = kBlockSize};
        VecSimParams primary_params = CreateParams(hnsw_params);
        TieredIndexParams tiered_params = {
            .jobQueue = &mock_thread_pool.jobQ,
            .jobQueueCtx = mock_thread_pool.ctx,
            .submitCb = tieredIndexMock::submit_callback,
            .flatBufferLimit = release_flat_guard ? SIZE_MAX : 0,
            .primaryIndexParams = &primary_params,
            .specificParams = {TieredHNSWParams{.swapJobThreshold = 0}}};
        auto *tiered_index =
            static_cast<TieredHNSWIndex<float, float> *>(TieredFactory::NewIndex(&tiered_params));
        mock_thread_pool.ctx->index_strong_ref.reset(tiered_index);
        auto *hnsw_index = tiered_index->getHNSWIndex();

        for (labelType label = 0; label < kBlockSize; ++label) {
            const float vector[kDimension] = {static_cast<float>(label), 0.0F, 0.0F, 0.0F};
            ASSERT_EQ(tiered_index->addVector(vector, label), 1);
            if (release_flat_guard) {
                mock_thread_pool.thread_iteration();
            }
        }
        ASSERT_EQ(hnsw_index->indexSize(), kBlockSize);
        ASSERT_EQ(hnsw_index->indexCapacity(), kBlockSize);

        const float failing_vector[kDimension] = {99.0F, 0.0F, 0.0F, 0.0F};
        if (release_flat_guard) {
            ASSERT_EQ(tiered_index->addVector(failing_vector, 99), 1);
            ASSERT_EQ(mock_thread_pool.jobQ.size(), 1);
        }

        MemoryHookState state{.fail_malloc_size = sizeof(ElementMetaData) * 2 * kBlockSize +
                                                  VecSimAllocator::getAllocationOverheadSize(),
                              .fail_matching_malloc_call = 0};
        {
            ScopedMemoryHooks hooks(state);
            if (release_flat_guard) {
                EXPECT_THROW(mock_thread_pool.thread_iteration(), std::bad_alloc);
            } else {
                EXPECT_THROW(tiered_index->addVector(failing_vector, 99), std::bad_alloc);
            }
        }
        EXPECT_EQ(state.failed_allocations, 1);
        EXPECT_EQ(hnsw_index->indexSize(), kBlockSize);
        EXPECT_EQ(hnsw_index->indexCapacity(), kBlockSize);

        std::atomic_bool main_available = false;
        std::atomic_bool data_available = false;
        std::atomic_bool flat_available = false;
        std::thread checker([&] {
            main_available = tiered_index->mainIndexGuard.try_lock();
            if (main_available.load()) {
                tiered_index->mainIndexGuard.unlock();
            }
            data_available = hnsw_index->indexDataGuard.try_lock();
            if (data_available.load()) {
                hnsw_index->indexDataGuard.unlock();
            }
            flat_available = tiered_index->flatIndexGuard.try_lock();
            if (flat_available.load()) {
                tiered_index->flatIndexGuard.unlock();
            }
        });
        checker.join();
        EXPECT_TRUE(main_available.load());
        EXPECT_TRUE(data_available.load());
        EXPECT_TRUE(flat_available.load());

        if (release_flat_guard) {
            auto *failed_job = tiered_index->labelToInsertJobs.at(99).front();
            failed_job->Execute(failed_job);
        } else {
            EXPECT_EQ(tiered_index->addVector(failing_vector, 99), 1);
        }
        EXPECT_EQ(hnsw_index->indexSize(), kBlockSize + 1);
        EXPECT_TRUE(hnsw_index->checkIntegrity().valid_state);

        auto *reply = VecSimIndex_TopKQuery(hnsw_index, failing_vector, 1, nullptr, BY_SCORE);
        ASSERT_NE(reply, nullptr);
        EXPECT_EQ(VecSimQueryReply_GetCode(reply), VecSim_QueryReply_OK);
        EXPECT_EQ(VecSimQueryReply_Len(reply), 1);
        auto *iterator = VecSimQueryReply_GetIterator(reply);
        auto *result = VecSimQueryReply_IteratorNext(iterator);
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(VecSimQueryResult_GetId(result), 99);
        EXPECT_EQ(VecSimQueryResult_GetScore(result), 0.0);
        VecSimQueryReply_IteratorFree(iterator);
        VecSimQueryReply_Free(reply);
    }
}

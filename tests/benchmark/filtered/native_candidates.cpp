/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "native_candidates.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace filtered_bench {
namespace {

bool nearer(const Neighbor &left, const Neighbor &right) {
    return left.distance < right.distance ||
           (left.distance == right.distance && left.id < right.id);
}

using TopK = std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(&nearer)>;

void offer(TopK &heap, Neighbor value, size_t k) {
    if (!std::isfinite(value.distance)) {
        throw std::runtime_error("VecSim returned a nonfinite distance");
    }
    if (heap.size() < k) {
        heap.push(value);
    } else if (nearer(value, heap.top())) {
        heap.pop();
        heap.push(value);
    }
}

void finish(SearchResult &result, TopK &heap, const Parameters &parameters) {
    if (Clock::now() >= parameters.deadline) {
        result.status = "timeout";
    }
    if (result.status != "ok") {
        return;
    }
    result.neighbors.reserve(heap.size());
    while (!heap.empty()) {
        result.neighbors.push_back(heap.top());
        heap.pop();
    }
    std::reverse(result.neighbors.begin(), result.neighbors.end());
}

SearchResult scan(VecSimIndex *index, std::span<const float> query, const Eligibility &eligible,
                  size_t k, const Parameters &parameters) {
    SearchResult result;
    result.distance_evaluations = 0;
    if (eligible.ids.empty()) {
        return result;
    }
    std::vector<float> prepared(query.begin(), query.end());
    if (parameters.metric == VecSimMetric_Cosine) {
        VecSim_Normalize(prepared.data(), prepared.size(), VecSimType_FLOAT32);
    }
    TopK heap(&nearer);
    for (const uint64_t id : eligible.ids) {
        if (*result.distance_evaluations % 64 == 0 && Clock::now() >= parameters.deadline) {
            result.status = "timeout";
            break;
        }
        // The graph is immutable, single-value FP32, and these labels were validated.
        double distance = VecSimIndex_GetDistanceFrom_Unsafe(index, id, prepared.data());
        ++*result.distance_evaluations;
        offer(heap, {static_cast<size_t>(id), distance}, k);
    }
    finish(result, heap, parameters);
    return result;
}

SearchResult batches(VecSimIndex *index, std::span<const float> query, const Eligibility &eligible,
                     size_t k, const Parameters &parameters) {
    SearchResult result;
    result.candidates_returned = 0;
    result.batches = 0;
    const size_t target = std::min(k, eligible.ids.size());
    if (target == 0) {
        return result;
    }
    auto deadline = parameters.deadline;
    VecSimQueryParams runtime = {};
    runtime.hnswRuntimeParams.efRuntime = parameters.ef_runtime;
    runtime.timeoutCtx = &deadline;
    std::unique_ptr<VecSimBatchIterator, decltype(&VecSimBatchIterator_Free)> iterator(
        VecSimBatchIterator_New(index, query.data(), &runtime), &VecSimBatchIterator_Free);
    if (!iterator) {
        throw std::runtime_error("VecSim could not create a batch iterator");
    }
    TopK heap(&nearer);
    std::unordered_set<size_t> qualified;
    while (heap.size() < target && VecSimBatchIterator_HasNext(iterator.get())) {
        if (Clock::now() >= deadline) {
            result.status = "timeout";
            break;
        }
        std::unique_ptr<VecSimQueryReply, decltype(&VecSimQueryReply_Free)> reply(
            VecSimBatchIterator_Next(iterator.get(), parameters.batch_size, BY_SCORE),
            &VecSimQueryReply_Free);
        if (!reply) {
            throw std::runtime_error("VecSim returned a null batch");
        }
        ++*result.batches;
        *result.candidates_returned += VecSimQueryReply_Len(reply.get());
        if (VecSimQueryReply_GetCode(reply.get()) == VecSim_QueryReply_TimedOut) {
            result.status = "timeout";
            break;
        }
        if (VecSimQueryReply_GetCode(reply.get()) != VecSim_QueryReply_OK) {
            throw std::runtime_error("VecSim returned an unexpected batch status");
        }
        if (VecSimQueryReply_Len(reply.get()) == 0) {
            if (VecSimBatchIterator_HasNext(iterator.get())) {
                throw std::runtime_error("VecSim batch iterator made no progress");
            }
            break;
        }
        std::unique_ptr<VecSimQueryReply_Iterator, decltype(&VecSimQueryReply_IteratorFree)> items(
            VecSimQueryReply_GetIterator(reply.get()), &VecSimQueryReply_IteratorFree);
        while (VecSimQueryReply_IteratorHasNext(items.get())) {
            const VecSimQueryResult *item = VecSimQueryReply_IteratorNext(items.get());
            const int64_t id = VecSimQueryResult_GetId(item);
            if (id < 0 || static_cast<uint64_t>(id) >= eligible.membership.size()) {
                throw std::runtime_error("VecSim returned an out-of-range document label");
            }
            if (eligible.membership[id] && qualified.insert(id).second) {
                offer(heap, {static_cast<size_t>(id), VecSimQueryResult_GetScore(item)}, target);
            }
        }
    }
    finish(result, heap, parameters);
    return result;
}

} // namespace

int deadlineExpired(void *context) {
    return context && Clock::now() >= *static_cast<Clock::time_point *>(context);
}

std::vector<CandidateAdapter> candidateAdapters() {
    return {{"NATIVE_SCAN", scan}, {"NATIVE_BATCHES", batches}};
}

} // namespace filtered_bench

/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/batch_iterator.h"
#include "hnsw.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/utils/vec_utils.h"
#include <limits>

// Batch iterator that serves results from an immutable, point-in-time graph
// snapshot (captured once under the read lock by newBatchIterator), then iterates
// completely lock-free: every graph / vector / metadata access resolves through
// the captured `HNSWGraphSnapshot`, so concurrent writers proceed (copy-on-write)
// without perturbing this scan, and the iterator never touches a per-node mutex.
//
// It mirrors the resumable two-phase search of HNSW_BatchIterator (greedy descent
// to a level-0 entry point on the first batch, then ef-bounded best-first
// expansion that carries `candidates` / `top_candidates_extras` / `lower_bound`
// across batches), but reads exactly like HNSWIndex::topKFromSnapshot: links via
// snap.getLevelData, vectors via snap.getVectorData, and deleted/in-process flags
// + labels from the captured (frozen) metadata. The visited set is private to the
// iterator (a vector<bool> sized to the captured element count) rather than the
// shared visited-nodes pool, so it needs no synchronization with the live index.
//
// Internal ids stay valid for the iterator's whole life because SWAP slot reuse is
// deferred while the snapshot's liveToken is held (HNSWIndex::graphSnapshotActive).
template <typename DataType, typename DistType>
class HNSWSnapshot_BatchIterator : public VecSimBatchIterator {
protected:
    const HNSWIndex<DataType, DistType> *index;
    HNSWGraphSnapshot snap;
    // Frozen metadata captured with the snapshot (label + flags as of capture).
    // Null only for an empty-at-capture index, where no id is ever visible.
    const vecsim_stl::vector<ElementMetaData> *meta;
    vecsim_stl::vector<bool> visited; // private visited set, sized snap.curElementCount
    idType entry_point;               // level-0 entry point (resolved on first batch)
    bool depleted;
    bool entry_resolved;
    size_t ef;

    template <typename Identifier>
    using candidatesMinHeap = vecsim_stl::min_priority_queue<DistType, Identifier>;

    DistType lower_bound;
    candidatesMinHeap<labelType> top_candidates_extras;
    candidatesMinHeap<idType> candidates;

    // A link can only reference an id that existed at capture; guard defensively.
    inline bool visible(idType id) const { return id < (idType)snap.curElementCount; }
    inline bool snapDeleted(idType id) const { return ((*meta)[id].flags & DELETE_MARK) != 0; }
    inline bool snapInProcess(idType id) const { return ((*meta)[id].flags & IN_PROCESS) != 0; }
    inline labelType snapLabel(idType id) const { return (*meta)[id].label; }

    inline void visitNode(idType id) { visited[id] = true; }
    inline bool hasVisitedNode(idType id) const { return visited[id]; }

    // Greedy descent through the captured upper levels to the level-0 entry point
    // (analog of searchBottomLayerEP, reading the snapshot). Returns INVALID_ID if
    // the snapshot is empty / has no entry point.
    idType snapshotBottomLayerEP(VecSimQueryReply_Code *rc);
    VecSimQueryReply_Code scanSnapshotInternal(candidatesLabelsMaxHeap<DistType> *top_candidates);
    candidatesLabelsMaxHeap<DistType> *scanSnapshot(VecSimQueryReply_Code *rc);

    virtual inline void prepareResults(VecSimQueryReply *rep,
                                       candidatesLabelsMaxHeap<DistType> *top_candidates,
                                       size_t n_res) = 0;
    virtual inline void fillFromExtras(candidatesLabelsMaxHeap<DistType> *top_candidates) = 0;
    virtual inline void updateHeaps(candidatesLabelsMaxHeap<DistType> *top_candidates, DistType dist,
                                    idType id) = 0;

public:
    HNSWSnapshot_BatchIterator(void *query_vector, const HNSWIndex<DataType, DistType> *index,
                               HNSWGraphSnapshot &&snapshot, VecSimQueryParams *queryParams,
                               std::shared_ptr<VecSimAllocator> allocator);

    VecSimQueryReply *getNextResults(size_t n_res, VecSimQueryReply_Order order) override;
    bool isDepleted() override;
    void reset() override;
    ~HNSWSnapshot_BatchIterator() override = default;
};

/******************** Ctor **************/

template <typename DataType, typename DistType>
HNSWSnapshot_BatchIterator<DataType, DistType>::HNSWSnapshot_BatchIterator(
    void *query_vector, const HNSWIndex<DataType, DistType> *index, HNSWGraphSnapshot &&snapshot,
    VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimBatchIterator(query_vector, queryParams ? queryParams->timeoutCtx : nullptr,
                          std::move(allocator)),
      index(index), snap(std::move(snapshot)),
      meta(static_cast<const vecsim_stl::vector<ElementMetaData> *>(snap.metaData.get())),
      visited(snap.curElementCount, false, this->allocator), entry_point(INVALID_ID),
      depleted(false), entry_resolved(false), top_candidates_extras(this->allocator),
      candidates(this->allocator) {

    if (queryParams && queryParams->hnswRuntimeParams.efRuntime > 0) {
        this->ef = queryParams->hnswRuntimeParams.efRuntime;
    } else {
        this->ef = this->index->getEf();
    }
}

/******************** Implementation **************/

template <typename DataType, typename DistType>
idType HNSWSnapshot_BatchIterator<DataType, DistType>::snapshotBottomLayerEP(
    VecSimQueryReply_Code *rc) {
    *rc = VecSim_QueryReply_OK;
    if (!snap.valid() || snap.curElementCount == 0 || snap.entrypointNode == INVALID_ID) {
        return INVALID_ID;
    }
    const void *query = this->getQueryBlob();
    idType cur = snap.entrypointNode;
    DistType curDist = this->index->calcDistance(query, snap.getVectorData(cur));
    for (size_t level = snap.maxLevel; level > 0; level--) {
        bool changed = true;
        while (changed) {
            if (VECSIM_TIMEOUT(this->getTimeoutCtx())) {
                *rc = VecSim_QueryReply_TimedOut;
                return cur;
            }
            changed = false;
            ElementLevelData &node_level = snap.getLevelData(cur, level);
            for (linkListSize j = 0; j < node_level.getNumLinks(); j++) {
                idType cand = node_level.getLinkAtPos(j);
                if (!visible(cand) || snapInProcess(cand)) {
                    continue;
                }
                DistType d = this->index->calcDistance(query, snap.getVectorData(cand));
                if (d < curDist) {
                    curDist = d;
                    cur = cand;
                    changed = true;
                }
            }
        }
    }
    return cur;
}

template <typename DataType, typename DistType>
VecSimQueryReply_Code HNSWSnapshot_BatchIterator<DataType, DistType>::scanSnapshotInternal(
    candidatesLabelsMaxHeap<DistType> *top_candidates) {
    const void *query = this->getQueryBlob();
    while (!candidates.empty()) {
        DistType curr_node_dist = candidates.top().first;
        idType curr_node_id = candidates.top().second;

        // If the closest candidate is further than the furthest kept result and we
        // already have enough, the search frontier is exhausted for this batch.
        if (curr_node_dist > this->lower_bound && top_candidates->size() >= this->ef) {
            break;
        }
        if (VECSIM_TIMEOUT(this->getTimeoutCtx())) {
            return VecSim_QueryReply_TimedOut;
        }
        if (!snapDeleted(curr_node_id)) {
            updateHeaps(top_candidates, curr_node_dist, curr_node_id);
        }

        candidates.pop();
        ElementLevelData &node_level = snap.getLevelData(curr_node_id, 0);
        for (linkListSize j = 0; j < node_level.getNumLinks(); j++) {
            idType candidate_id = node_level.getLinkAtPos(j);
            if (!visible(candidate_id) || hasVisitedNode(candidate_id) ||
                snapInProcess(candidate_id)) {
                continue;
            }
            visitNode(candidate_id);
            DistType candidate_dist =
                this->index->calcDistance(query, snap.getVectorData(candidate_id));
            candidates.emplace(candidate_dist, candidate_id);
        }
    }
    return VecSim_QueryReply_OK;
}

template <typename DataType, typename DistType>
candidatesLabelsMaxHeap<DistType> *
HNSWSnapshot_BatchIterator<DataType, DistType>::scanSnapshot(VecSimQueryReply_Code *rc) {
    candidatesLabelsMaxHeap<DistType> *top_candidates = this->index->getNewMaxPriorityQueue();
    if (this->entry_point == INVALID_ID) {
        this->depleted = true;
        return top_candidates;
    }

    // First iteration: seed the candidate frontier with the entry point.
    if (this->getResultsCount() == 0 && this->top_candidates_extras.empty() &&
        this->candidates.empty()) {
        if (!snapDeleted(this->entry_point)) {
            this->lower_bound = this->index->calcDistance(this->getQueryBlob(),
                                                          snap.getVectorData(this->entry_point));
        } else {
            this->lower_bound = std::numeric_limits<DistType>::max();
        }
        this->visitNode(this->entry_point);
        candidates.emplace(this->lower_bound, this->entry_point);
    }
    if (VECSIM_TIMEOUT(this->getTimeoutCtx())) {
        *rc = VecSim_QueryReply_TimedOut;
        return top_candidates;
    }

    // Carry the spare results from the previous batch into this one.
    fillFromExtras(top_candidates);
    if (top_candidates->size() == this->ef) {
        return top_candidates;
    }
    *rc = this->scanSnapshotInternal(top_candidates);

    if (top_candidates->size() < this->ef) {
        this->depleted = true;
    }
    return top_candidates;
}

template <typename DataType, typename DistType>
VecSimQueryReply *
HNSWSnapshot_BatchIterator<DataType, DistType>::getNextResults(size_t n_res,
                                                               VecSimQueryReply_Order order) {
    auto batch = new VecSimQueryReply(this->allocator);
    size_t orig_ef = this->ef;
    if (orig_ef < n_res) {
        this->ef = n_res;
    }

    // On the first batch, descend the captured upper levels to the level-0 entry
    // point (lock-free, reading the snapshot).
    if (!this->entry_resolved) {
        this->entry_point = snapshotBottomLayerEP(&batch->code);
        this->entry_resolved = true;
        if (VecSim_OK != batch->code) {
            this->ef = orig_ef;
            return batch;
        }
    }

    auto *top_candidates = this->scanSnapshot(&batch->code);
    if (VecSim_OK != batch->code) {
        delete top_candidates;
        this->ef = orig_ef;
        return batch;
    }
    this->prepareResults(batch, top_candidates, n_res);
    delete top_candidates;

    this->updateResultsCount(VecSimQueryReply_Len(batch));
    if (order == BY_ID) {
        sort_results_by_id(batch);
    }
    this->ef = orig_ef;
    return batch;
}

template <typename DataType, typename DistType>
bool HNSWSnapshot_BatchIterator<DataType, DistType>::isDepleted() {
    return this->depleted && this->top_candidates_extras.empty();
}

// reset() restarts the scan over the SAME captured snapshot — it does NOT
// re-capture. The snapshot was taken once at construction and is owned for the
// iterator's whole life, so a reset cursor yields the identical point-in-time
// view it had before. (The tiered snapshot cursor differs: it re-captures a fresh
// snapshot on reset — see TieredHNSW_BatchIterator::reset.)
template <typename DataType, typename DistType>
void HNSWSnapshot_BatchIterator<DataType, DistType>::reset() {
    this->resetResultsCount();
    this->depleted = false;
    this->entry_resolved = false;
    this->entry_point = INVALID_ID;
    std::fill(this->visited.begin(), this->visited.end(), false);
    this->lower_bound = std::numeric_limits<DistType>::infinity();
    this->candidates = candidatesMinHeap<idType>(this->allocator);
    this->top_candidates_extras = candidatesMinHeap<labelType>(this->allocator);
}

/******************** Single-value variant **************/

template <typename DataType, typename DistType>
class HNSWSnapshotSingle_BatchIterator : public HNSWSnapshot_BatchIterator<DataType, DistType> {
private:
    inline void prepareResults(VecSimQueryReply *rep,
                               candidatesLabelsMaxHeap<DistType> *top_candidates,
                               size_t n_res) override {
        while (top_candidates->size() > n_res) {
            this->top_candidates_extras.emplace(top_candidates->top());
            top_candidates->pop();
        }
        rep->results.resize(top_candidates->size());
        for (auto result = rep->results.rbegin(); result != rep->results.rend(); ++result) {
            std::tie(result->score, result->id) = top_candidates->top();
            top_candidates->pop();
        }
    }
    inline void fillFromExtras(candidatesLabelsMaxHeap<DistType> *top_candidates) override {
        while (top_candidates->size() < this->ef && !this->top_candidates_extras.empty()) {
            top_candidates->emplace(this->top_candidates_extras.top().first,
                                    this->top_candidates_extras.top().second);
            this->top_candidates_extras.pop();
        }
    }
    inline void updateHeaps(candidatesLabelsMaxHeap<DistType> *top_candidates, DistType dist,
                            idType id) override {
        if (top_candidates->size() < this->ef) {
            top_candidates->emplace(dist, this->snapLabel(id));
            this->lower_bound = top_candidates->top().first;
        } else if (this->lower_bound > dist) {
            top_candidates->emplace(dist, this->snapLabel(id));
            this->top_candidates_extras.emplace(top_candidates->top().first,
                                                top_candidates->top().second);
            top_candidates->pop();
            this->lower_bound = top_candidates->top().first;
        }
    }

public:
    HNSWSnapshotSingle_BatchIterator(void *query_vector,
                                     const HNSWIndex<DataType, DistType> *index,
                                     HNSWGraphSnapshot &&snapshot, VecSimQueryParams *queryParams,
                                     std::shared_ptr<VecSimAllocator> allocator)
        : HNSWSnapshot_BatchIterator<DataType, DistType>(
              query_vector, index, std::move(snapshot), queryParams, std::move(allocator)) {}
    ~HNSWSnapshotSingle_BatchIterator() override = default;
};

/******************** Multi-value variant **************/

template <typename DataType, typename DistType>
class HNSWSnapshotMulti_BatchIterator : public HNSWSnapshot_BatchIterator<DataType, DistType> {
private:
    vecsim_stl::unordered_set<labelType> returned;

    inline void prepareResults(VecSimQueryReply *rep,
                               candidatesLabelsMaxHeap<DistType> *top_candidates,
                               size_t n_res) override {
        while (top_candidates->size() > n_res) {
            this->top_candidates_extras.emplace(top_candidates->top().first,
                                                top_candidates->top().second);
            top_candidates->pop();
        }
        rep->results.resize(top_candidates->size());
        for (auto result = rep->results.rbegin(); result != rep->results.rend(); ++result) {
            std::tie(result->score, result->id) = top_candidates->top();
            this->returned.insert(result->id);
            top_candidates->pop();
        }
    }
    inline void fillFromExtras(candidatesLabelsMaxHeap<DistType> *top_candidates) override {
        while (top_candidates->size() < this->ef && !this->top_candidates_extras.empty()) {
            if (returned.find(this->top_candidates_extras.top().second) == returned.end()) {
                top_candidates->emplace(this->top_candidates_extras.top().first,
                                        this->top_candidates_extras.top().second);
            }
            this->top_candidates_extras.pop();
        }
    }
    inline void updateHeaps(candidatesLabelsMaxHeap<DistType> *top_candidates, DistType dist,
                            idType id) override {
        if (this->lower_bound > dist || top_candidates->size() < this->ef) {
            labelType label = this->snapLabel(id);
            if (returned.find(label) == returned.end()) {
                top_candidates->emplace(dist, label);
                if (top_candidates->size() > this->ef) {
                    this->top_candidates_extras.emplace(top_candidates->top().first,
                                                        top_candidates->top().second);
                    top_candidates->pop();
                }
                this->lower_bound = top_candidates->top().first;
            }
        }
    }

public:
    HNSWSnapshotMulti_BatchIterator(void *query_vector, const HNSWIndex<DataType, DistType> *index,
                                    HNSWGraphSnapshot &&snapshot, VecSimQueryParams *queryParams,
                                    std::shared_ptr<VecSimAllocator> allocator)
        : HNSWSnapshot_BatchIterator<DataType, DistType>(query_vector, index, std::move(snapshot),
                                                         queryParams, std::move(allocator)),
          returned(index->indexSize(), this->allocator) {}
    ~HNSWSnapshotMulti_BatchIterator() override = default;

    void reset() override {
        this->returned.clear();
        HNSWSnapshot_BatchIterator<DataType, DistType>::reset();
    }
};

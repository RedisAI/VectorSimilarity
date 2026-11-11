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
#include "VecSim/spaces/spaces.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/algorithms/hnsw/visited_nodes_handler.h"
#include <limits>

using spaces::dist_func_t;

template <typename DataType, typename DistType>
class HNSW_BatchIterator : public VecSimBatchIterator {
protected:
    const HNSWIndex<DataType, DistType> *index;
    VisitedNodesHandler *visited_list; // Pointer to the hnsw visitedList structure.
    tag_t visited_tag;                 // Used to mark nodes that were scanned.
    idType entry_point;                // Internal id of the node to begin the scan from.
    bool depleted;
    size_t ef; // EF Runtime value for this query.

    // Data structure that holds the search state between iterations.
    template <typename Identifier>
    using candidatesMinHeap = vecsim_stl::min_priority_queue<DistType, Identifier>;

    VecSimNavixParams navix{};
    enum class NavixMatch : uint8_t { Unknown, No, Maybe };
    vecsim_stl::vector<NavixMatch> navix_filter_cache;
    vecsim_stl::vector<idType> navix_direct, navix_second;
    vecsim_stl::vector<uint8_t> navix_matches;
    vecsim_stl::vector<std::pair<DistType, idType>> navix_branches;
    bool navixMayMatch(idType id);
    DistType navixDistance(idType id);
    void navixLinks(idType id, vecsim_stl::vector<idType> &out);
    VecSimQueryReply_Code expandNavix(idType id);
    VecSimQueryReply_Code scanNavix(candidatesLabelsMaxHeap<DistType> *top_candidates);

    DistType lower_bound;
    candidatesMinHeap<labelType> top_candidates_extras;
    candidatesMinHeap<idType> candidates;

    VecSimQueryReply_Code scanGraphInternal(candidatesLabelsMaxHeap<DistType> *top_candidates);
    candidatesLabelsMaxHeap<DistType> *scanGraph(VecSimQueryReply_Code *rc);
    virtual inline void prepareResults(VecSimQueryReply *rep,
                                       candidatesLabelsMaxHeap<DistType> *top_candidates,
                                       size_t n_res) = 0;
    inline void visitNode(idType node_id) {
        this->visited_list->tagNode(node_id, this->visited_tag);
    }
    inline bool hasVisitedNode(idType node_id) const {
        return this->visited_list->getNodeTag(node_id) == this->visited_tag;
    }

    virtual inline void fillFromExtras(candidatesLabelsMaxHeap<DistType> *top_candidates) = 0;
    virtual inline void updateHeaps(candidatesLabelsMaxHeap<DistType> *top_candidates,
                                    DistType dist, idType id) = 0;

public:
    HNSW_BatchIterator(void *query_vector, const HNSWIndex<DataType, DistType> *index,
                       VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator);

    VecSimQueryReply *getNextResults(size_t n_res, VecSimQueryReply_Order order) override;

    bool isDepleted() override;

    void reset() override;

    virtual ~HNSW_BatchIterator() { index->returnVisitedList(this->visited_list); }
};

/******************** Ctor / Dtor **************/

template <typename DataType, typename DistType>
HNSW_BatchIterator<DataType, DistType>::HNSW_BatchIterator(
    void *query_vector, const HNSWIndex<DataType, DistType> *index, VecSimQueryParams *queryParams,
    std::shared_ptr<VecSimAllocator> allocator)
    : VecSimBatchIterator(query_vector, queryParams ? queryParams->timeoutCtx : nullptr,
                          std::move(allocator)),
      index(index), depleted(false), navix_filter_cache(this->allocator),
      navix_direct(this->allocator), navix_second(this->allocator), navix_matches(this->allocator),
      navix_branches(this->allocator), top_candidates_extras(this->allocator),
      candidates(this->allocator) {

    if (queryParams)
        navix = queryParams->navix;
    this->entry_point = INVALID_ID; // temporary until we store the entry point to level 0.
    // Use "fresh" tag to mark nodes that were visited along the search in some iteration.
    this->visited_list = index->getVisitedList();
    this->visited_tag = this->visited_list->getFreshTag();

    if (queryParams && queryParams->hnswRuntimeParams.efRuntime > 0) {
        this->ef = queryParams->hnswRuntimeParams.efRuntime;
    } else {
        this->ef = this->index->getEf();
    }
}

/******************** Implementation **************/

template <typename DataType, typename DistType>
bool HNSW_BatchIterator<DataType, DistType>::navixMayMatch(idType id) {
    if (index->isMarkedDeleted(id))
        return false;
    // The NaviX contract keeps internal IDs and the filter stable until reset.
    if (navix_filter_cache.empty())
        navix_filter_cache.resize(index->indexSize(), NavixMatch::Unknown);
    auto &cached = navix_filter_cache[id];
    if (cached != NavixMatch::Unknown)
        return cached == NavixMatch::Maybe;
    if (navix.stats)
        ++navix.stats->filterChecks;
    bool pass = navix.mayMatch(navix.context, index->getExternalLabel(id)) != 0;
    cached = pass ? NavixMatch::Maybe : NavixMatch::No;
    if (!pass && navix.stats)
        ++navix.stats->filterRejected;
    return pass;
}

template <typename DataType, typename DistType>
DistType HNSW_BatchIterator<DataType, DistType>::navixDistance(idType id) {
    if (navix.stats)
        ++navix.stats->distanceComputations;
    return index->calcDistanceForQuery(index->getDataByInternalId(id), this->getQueryBlob());
}

template <typename DataType, typename DistType>
void HNSW_BatchIterator<DataType, DistType>::navixLinks(idType id,
                                                        vecsim_stl::vector<idType> &out) {
    // RAM HNSW's base-layer degree is bounded by 2*M. Allocate before locking;
    // neither a callback nor a second adjacency lock runs under this node lock.
    out.resize(2 * index->getM());
    index->lockNodeLinks(id);
    const auto &links = index->getElementLevelData(id, 0);
    assert(links.numLinks <= out.size());
    const auto count = links.numLinks;
    std::copy_n(links.links, count, out.begin());
    index->unlockNodeLinks(id);
    out.resize(count);
}

template <typename DataType, typename DistType>
VecSimQueryReply_Code HNSW_BatchIterator<DataType, DistType>::expandNavix(idType id) {
    navixLinks(id, navix_direct);
    navix_matches.resize(navix_direct.size());
    size_t passing = 0;
    for (size_t i = 0; i < navix_direct.size(); ++i) {
        if (VECSIM_TIMEOUT(this->getTimeoutCtx()))
            return VecSim_QueryReply_TimedOut;
        passing += (navix_matches[i] = navixMayMatch(navix_direct[i]));
    }
    const size_t degree = 2 * index->getM();
    const double fraction = navix_direct.empty() ? 0.0 : double(passing) / navix_direct.size();
    const bool onehop = !navix.forceBlind && fraction > 0.5;
    const bool directed =
        !onehop && !navix.forceBlind && fraction * double(degree) * (degree + 1) >= 3.0 * degree;
    if (navix.stats) {
        ++navix.stats->expandedNodes;
        if (onehop)
            ++navix.stats->oneHopExpansions;
        else if (directed)
            ++navix.stats->directedExpansions;
        else
            ++navix.stats->blindExpansions;
    }

    size_t admitted = 0;
    navix_branches.clear();
    for (size_t i = 0; i < navix_direct.size(); ++i) {
        if (VECSIM_TIMEOUT(this->getTimeoutCtx()))
            return VecSim_QueryReply_TimedOut;
        idType neighbor = navix_direct[i];
        const bool enqueue = navix_matches[i] && !hasVisitedNode(neighbor);
        // Directed expansion orders even rejected branches; reuse this score on admission.
        DistType distance = (directed || enqueue) ? navixDistance(neighbor) : 0;
        if (directed)
            navix_branches.emplace_back(distance, neighbor);
        if (enqueue) {
            visitNode(neighbor);
            candidates.emplace(distance, neighbor);
            ++admitted;
        }
    }
    if (onehop || admitted >= degree)
        return VecSim_QueryReply_OK;
    if (directed)
        std::sort(navix_branches.begin(), navix_branches.end());

    for (size_t i = 0; i < navix_direct.size() && admitted < degree; ++i) {
        if (VECSIM_TIMEOUT(this->getTimeoutCtx()))
            return VecSim_QueryReply_TimedOut;
        const idType branch = directed ? navix_branches[i].second : navix_direct[i];
        navixLinks(branch, navix_second);
        for (idType neighbor : navix_second) {
            if (VECSIM_TIMEOUT(this->getTimeoutCtx()))
                return VecSim_QueryReply_TimedOut;
            if (navix.stats)
                ++navix.stats->secondHopVisits;
            if (hasVisitedNode(neighbor) || !navixMayMatch(neighbor))
                continue;
            visitNode(neighbor);
            candidates.emplace(navixDistance(neighbor), neighbor);
            if (++admitted == degree)
                break;
        }
    }
    return VecSim_QueryReply_OK;
}

template <typename DataType, typename DistType>
VecSimQueryReply_Code HNSW_BatchIterator<DataType, DistType>::scanNavix(
    candidatesLabelsMaxHeap<DistType> *top_candidates) {
    while (!candidates.empty()) {
        auto [distance, id] = candidates.top();
        // Leave bound ties on the frontier for the next batch.
        if (distance >= this->lower_bound && top_candidates->size() >= this->ef)
            break;
        if (VECSIM_TIMEOUT(this->getTimeoutCtx()))
            return VecSim_QueryReply_TimedOut;
        candidates.pop();
        // A rejected entry point may navigate, but cannot tighten the result bound.
        if (navixMayMatch(id))
            updateHeaps(top_candidates, distance, id);
        auto code = expandNavix(id);
        if (code != VecSim_QueryReply_OK)
            return code;
    }
    return VecSim_QueryReply_OK;
}

template <typename DataType, typename DistType>
VecSimQueryReply_Code HNSW_BatchIterator<DataType, DistType>::scanGraphInternal(
    candidatesLabelsMaxHeap<DistType> *top_candidates) {
    if (navix.mayMatch)
        return scanNavix(top_candidates);
    while (!candidates.empty()) {
        DistType curr_node_dist = candidates.top().first;
        idType curr_node_id = candidates.top().second;

        __builtin_prefetch(this->index->getGraphDataByInternalId(curr_node_id));
        __builtin_prefetch(this->index->getMetaDataAddress(curr_node_id));
        // If the closest element in the candidates set is further than the furthest element in the
        // top candidates set, and we have enough results, we finish the search.
        if (curr_node_dist > this->lower_bound && top_candidates->size() >= this->ef) {
            break;
        }
        if (VECSIM_TIMEOUT(this->getTimeoutCtx())) {
            return VecSim_QueryReply_TimedOut;
        }
        // Checks if we need to add the current id to the top_candidates heap,
        // and updates the extras heap accordingly.
        if (!index->isMarkedDeleted(curr_node_id))
            updateHeaps(top_candidates, curr_node_dist, curr_node_id);

        // Take the current node out of the candidates queue and go over his neighbours.
        candidates.pop();
        auto *node_graph_data = this->index->getGraphDataByInternalId(curr_node_id);
        this->index->lockNodeLinks(curr_node_id);
        ElementLevelData &node_level_data = this->index->getElementLevelData(node_graph_data, 0);
        if (node_level_data.numLinks > 0) {

            // Pre-fetch first candidate tag address.
            __builtin_prefetch(visited_list->getElementsTags() + node_level_data.links[0]);
            // // Pre-fetch first candidate data block address.
            __builtin_prefetch(index->getDataByInternalId(node_level_data.links[0]));

            for (linkListSize j = 0; j < node_level_data.numLinks - 1; j++) {
                idType candidate_id = node_level_data.links[j];

                // Pre-fetch next candidate tag address.
                __builtin_prefetch(visited_list->getElementsTags() + node_level_data.links[j + 1]);
                // Pre-fetch next candidate data block address.
                __builtin_prefetch(index->getDataByInternalId(node_level_data.links[j + 1]));

                if (this->hasVisitedNode(candidate_id)) {
                    continue;
                }
                this->visitNode(candidate_id);

                const char *candidate_data = this->index->getDataByInternalId(candidate_id);
                DistType candidate_dist = this->index->calcDistanceForQuery(
                    (const void *)candidate_data, this->getQueryBlob());

                candidates.emplace(candidate_dist, candidate_id);
            }
            // Running the last candidate outside the loop to avoid prefetching invalid candidate
            idType candidate_id = node_level_data.links[node_level_data.numLinks - 1];

            if (!this->hasVisitedNode(candidate_id)) {
                this->visitNode(candidate_id);

                const char *candidate_data = this->index->getDataByInternalId(candidate_id);
                DistType candidate_dist = this->index->calcDistanceForQuery(
                    (const void *)candidate_data, this->getQueryBlob());

                candidates.emplace(candidate_dist, candidate_id);
            }
        }
        this->index->unlockNodeLinks(curr_node_id);
    }
    return VecSim_QueryReply_OK;
}

template <typename DataType, typename DistType>
candidatesLabelsMaxHeap<DistType> *
HNSW_BatchIterator<DataType, DistType>::scanGraph(VecSimQueryReply_Code *rc) {

    candidatesLabelsMaxHeap<DistType> *top_candidates = this->index->getNewMaxPriorityQueue();
    if (this->entry_point == INVALID_ID) {
        this->depleted = true;
        return top_candidates;
    }

    // In the first iteration, add the entry point to the empty candidates set.
    if (this->getResultsCount() == 0 && this->top_candidates_extras.empty() &&
        this->candidates.empty()) {
        if (!index->isMarkedDeleted(this->entry_point)) {
            this->lower_bound = this->index->calcDistanceForQuery(
                this->index->getDataByInternalId(this->entry_point), this->getQueryBlob());
        } else {
            this->lower_bound = std::numeric_limits<DistType>::max();
        }
        this->visitNode(this->entry_point);
        candidates.emplace(this->lower_bound, this->entry_point);
    }
    // Checks that we didn't got timeout between iterations.
    if (VECSIM_TIMEOUT(this->getTimeoutCtx())) {
        *rc = VecSim_QueryReply_TimedOut;
        return top_candidates;
    }

    // Move extras from previous iteration to the top candidates.
    fillFromExtras(top_candidates);
    if (top_candidates->size() == this->ef) {
        return top_candidates;
    }
    *rc = this->scanGraphInternal(top_candidates);

    // If we found fewer results than wanted, mark the search as depleted.
    if (top_candidates->size() < this->ef) {
        this->depleted = true;
    }
    return top_candidates;
}

template <typename DataType, typename DistType>
VecSimQueryReply *
HNSW_BatchIterator<DataType, DistType>::getNextResults(size_t n_res, VecSimQueryReply_Order order) {

    auto batch = new VecSimQueryReply(this->allocator);
    // If ef_runtime lower than the number of results to return, increase it. Therefore, we assume
    // that the number of results that return from the graph scan is at least n_res (if exist).
    size_t orig_ef = this->ef;
    if (orig_ef < n_res) {
        this->ef = n_res;
    }

    // In the first iteration, we search the graph from top bottom to find the initial entry point,
    // and then we scan the graph to get results (layer 0).
    if (this->getResultsCount() == 0) {
        idType bottom_layer_ep = this->index->searchBottomLayerEP(
            this->getQueryBlob(), this->getTimeoutCtx(), &batch->code);
        if (VecSim_OK != batch->code) {
            this->ef = orig_ef;
            return batch;
        }
        this->entry_point = bottom_layer_ep;
    }
    // We ask for at least n_res candidate from the scan. In fact, at most ef results will return,
    // and it could be that ef > n_res.
    auto *top_candidates = this->scanGraph(&batch->code);
    if (VecSim_OK != batch->code) {
        this->ef = orig_ef;
        delete top_candidates;
        return batch;
    }
    // Move the spare results to the "extras" queue if needed, and create the batch results array.
    this->prepareResults(batch, top_candidates, n_res);
    delete top_candidates;

    this->updateResultsCount(VecSimQueryReply_Len(batch));
    if (this->getResultsCount() == this->index->indexLabelCount()) {
        this->depleted = true;
    }
    // By default, results are ordered by score.
    if (order == BY_ID) {
        sort_results_by_id(batch);
    }
    this->ef = orig_ef;
    return batch;
}

template <typename DataType, typename DistType>
bool HNSW_BatchIterator<DataType, DistType>::isDepleted() {
    return this->depleted && this->top_candidates_extras.empty();
}

template <typename DataType, typename DistType>
void HNSW_BatchIterator<DataType, DistType>::reset() {
    this->resetResultsCount();
    this->depleted = false;
    navix_filter_cache.clear();

    // Reset the visited nodes handler.
    this->visited_tag = this->visited_list->getFreshTag();
    this->lower_bound = std::numeric_limits<DistType>::infinity();
    // Clear the queues.
    this->candidates = candidatesMinHeap<idType>(this->allocator);
    this->top_candidates_extras = candidatesMinHeap<labelType>(this->allocator);
}

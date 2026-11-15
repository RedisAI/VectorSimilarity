/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "pipnn.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
BFParams storageParams(const PiPNNParams &params) {
    return {.type = params.type,
            .dim = params.dim,
            .metric = params.metric,
            .multi = false,
            .initialCapacity = params.initialCapacity,
            .blockSize = params.blockSize};
}

size_t checkedProduct(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b) {
        throw std::length_error("PiPNN allocation size overflow");
    }
    return a * b;
}

struct Candidate {
    float distance;
    idType id;
    uint32_t hash;
};

bool nearer(const Candidate &a, const Candidate &b) {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

// The flat iterator's base receives null. A failed constructor therefore cannot
// free a copied query that is still owned by its caller.
class PiPNNBatchIterator final : public BFS_BatchIterator<float, float> {
    MemoryUtils::unique_blob query_;

    VecSimQueryReply_Code calculateScores() override {
        this->index_label_count = this->index->indexLabelCount();
        vecsim_stl::vector<std::pair<float, labelType>> staged(this->allocator);
        staged.reserve(this->index_label_count);
        for (size_t i = 0; i < this->index_label_count; ++i) {
            if (VECSIM_TIMEOUT(this->getTimeoutCtx())) {
                return VecSim_QueryReply_TimedOut;
            }
            const idType id = static_cast<idType>(i);
            const float score = this->index->calcDistanceForQuery(
                this->index->getDataByInternalId(id), this->getQueryBlob());
            staged.emplace_back(score, this->index->getVectorLabel(id));
        }
        // Publish only a complete scan. A timeout or allocation failure leaves scores empty,
        // so a subsequent batch retries the full scan rather than using a partial prefix.
        this->scores = std::move(staged);
        return VecSim_QueryReply_OK;
    }

public:
    PiPNNBatchIterator(MemoryUtils::unique_blob &query, const PiPNNIndex *index,
                       VecSimQueryParams *params, const std::shared_ptr<VecSimAllocator> &allocator)
        : BFS_BatchIterator<float, float>(nullptr, index, params, allocator),
          query_(std::move(query)) {}

    const void *getQueryBlob() const override { return query_.get(); }

    VecSimQueryReply *getNextResults(size_t n, VecSimQueryReply_Order order) override {
        try {
            if (order != BY_ID && order != BY_SCORE) {
                return nullptr;
            }
            if (n == 0) {
                return new VecSimQueryReply(this->allocator);
            }
            return BFS_BatchIterator<float, float>::getNextResults(n, order);
        } catch (...) {
            // A failure must not propagate through the batch C API. The flat search stages
            // its allocations before committing the next batch, so the iterator can be retried.
            return nullptr;
        }
    }
};

struct Bucket {
    vecsim_stl::vector<idType> ids;
    size_t depth;
    Bucket(const std::shared_ptr<VecSimAllocator> &allocator, size_t depth)
        : ids(allocator), depth(depth) {}
};
} // namespace

PiPNNIndex::PiPNNIndex(const PiPNNParams *params, const AbstractIndexInitParams &abstractInitParams,
                       const IndexComponents<float, float> &components)
    : PiPNNIndex(storageParams(*params), params, abstractInitParams, components) {}

PiPNNIndex::PiPNNIndex(const BFParams &storageParams, const PiPNNParams *params,
                       const AbstractIndexInitParams &abstractInitParams,
                       const IndexComponents<float, float> &components)
    : Base(&storageParams, abstractInitParams, components), params_(*params),
      graph_(this->allocator), degrees_(this->allocator) {}

void PiPNNIndex::validateVector(const void *blob) const {
    if (!blob) {
        throw std::invalid_argument("PiPNN requires a non-null FP32 vector");
    }
    // Test the representation so finite validation also works in builds using -ffast-math.
    auto bytes = static_cast<const unsigned char *>(blob);
    for (size_t i = 0; i < this->dim; ++i) {
        uint32_t bits;
        std::memcpy(&bits, bytes + i * sizeof(float), sizeof(bits));
        if ((bits & 0x7f800000U) == 0x7f800000U) {
            throw std::invalid_argument("PiPNN requires finite FP32 components");
        }
    }
}

void PiPNNIndex::invalidateGraph() {
    graphReady_ = false;
    vecsim_stl::vector<idType> emptyGraph(this->allocator);
    vecsim_stl::vector<size_t> emptyDegrees(this->allocator);
    graph_.swap(emptyGraph);
    degrees_.swap(emptyDegrees);
}

int PiPNNIndex::addVector(const void *vector_data, labelType label) {
    validateVector(vector_data);
    auto existing = this->labelToIdLookup.find(label);
    if (this->count == std::numeric_limits<idType>::max() &&
        existing == this->labelToIdLookup.end()) {
        throw std::length_error("PiPNN internal ID capacity exhausted");
    }
    auto prepared = this->preprocessForStorage(vector_data);
    invalidateGraph();
    if (existing != this->labelToIdLookup.end()) {
        this->vectors->updateElement(existing->second, prepared.get());
        return 0;
    }

    // Stage metadata growth so a failure cannot leave size and capacity inconsistent.
    vecsim_stl::vector<labelType> expanded(this->allocator);
    if (this->indexSize() == this->indexCapacity()) {
        if (this->indexCapacity() > std::numeric_limits<size_t>::max() - this->blockSize) {
            throw std::length_error("PiPNN metadata capacity overflow");
        }
        expanded.resize(this->indexCapacity() + this->blockSize);
        std::copy_n(this->idToLabelMapping.begin(), this->count, expanded.begin());
        expanded[this->count] = label;
    }
    const idType id = this->count;
    this->labelToIdLookup.emplace(label, id);
    try {
        this->vectors->addElement(prepared.get(), id);
    } catch (...) {
        this->labelToIdLookup.erase(label);
        throw;
    }
    if (!expanded.empty()) {
        this->idToLabelMapping.swap(expanded);
    } else {
        this->setVectorLabel(id, label);
    }
    ++this->count;
    return 1;
}

int PiPNNIndex::deleteVector(labelType label) {
    const auto found = this->labelToIdLookup.find(label);
    if (found == this->labelToIdLookup.end()) {
        return 0;
    }
    invalidateGraph();
    const idType removed = found->second;
    const idType last = this->count - 1;
    if (removed != last) {
        const labelType movedLabel = this->getVectorLabel(last);
        this->vectors->updateElement(removed, this->getDataByInternalId(last));
        this->setVectorLabel(removed, movedLabel);
        this->labelToIdLookup.find(movedLabel)->second = removed;
    }
    this->vectors->removeElement(last);
    this->labelToIdLookup.erase(found);
    --this->count;
    // Retain metadata capacity: deletion must not allocate to shrink it. Empty vector-data
    // blocks are still released by the storage container.
    return 1;
}

int PiPNNIndex::deleteVectorById(labelType label, idType id) {
    // The single-value flat contract resolves the vector by label, including this entry point.
    return deleteVector(label);
}

std::unordered_map<idType, std::pair<idType, labelType>>
PiPNNIndex::deleteVectorAndGetUpdatedIds(labelType label) {
    std::unordered_map<idType, std::pair<idType, labelType>> updated;
    const auto found = this->labelToIdLookup.find(label);
    if (found == this->labelToIdLookup.end()) {
        return updated;
    }
    const idType removed = found->second;
    const idType last = this->count - 1;
    if (removed != last) {
        updated.emplace(removed, std::make_pair(last, this->getVectorLabel(last)));
    }
    deleteVector(label);
    return updated;
}

VecSimUpdateCode PiPNNIndex::updateVectors(labelType label, const void *new_blobs, size_t n) {
    if (n > 1) {
        return VecSimUpdate_MultiNotSupported;
    }
    if (n == 1) {
        addVector(new_blobs, label);
    } else {
        deleteVector(label);
    }
    return VecSimUpdate_OK;
}

VecSimRelabelCode PiPNNIndex::relabelVector(labelType old_label, labelType new_label) {
    if (old_label == new_label) {
        return VecSimRelabel_SameLabel;
    }
    const auto old = this->labelToIdLookup.find(old_label);
    if (old == this->labelToIdLookup.end()) {
        return VecSimRelabel_OldLabelMissing;
    }
    if (this->labelToIdLookup.find(new_label) != this->labelToIdLookup.end()) {
        return VecSimRelabel_NewLabelTaken;
    }
    const idType id = old->second;
    invalidateGraph();
    // Insert before removing the old mapping; an allocation failure preserves the label.
    // Erase by key because insertion may rehash and invalidate the old iterator.
    this->labelToIdLookup.emplace(new_label, id);
    this->setVectorLabel(id, new_label);
    this->labelToIdLookup.erase(old_label);
    return VecSimRelabel_OK;
}

double PiPNNIndex::getDistanceFrom_Unsafe(labelType label, const void *vector_data) const {
    validateVector(vector_data);
    return Base::getDistanceFrom_Unsafe(label, vector_data);
}

VecSimQueryReply *PiPNNIndex::rangeQuery(const void *queryBlob, double radius,
                                         VecSimQueryParams *queryParams) const {
    validateVector(queryBlob);
    auto query = this->preprocessQuery(queryBlob);
    auto reply = std::make_unique<VecSimQueryReply>(this->allocator);
    this->lastMode = RANGE_QUERY;
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
    const float bound = static_cast<float>(radius);
    for (idType id = 0; id < this->count; ++id) {
        if (VECSIM_TIMEOUT(timeoutCtx)) {
            reply->code = VecSim_QueryReply_TimedOut;
            break;
        }
        const float score = this->calcDistanceForQuery(this->getDataByInternalId(id), query.get());
        if (score <= bound) {
            reply->results.push_back({this->getVectorLabel(id), score});
        }
    }
    return reply.release();
}

VecSimBatchIterator *PiPNNIndex::newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) const {
    validateVector(queryBlob);
    auto query = this->preprocessQuery(queryBlob, true);
    return new (this->allocator) PiPNNBatchIterator(query, this, queryParams, this->allocator);
}

void PiPNNIndex::buildGraph() {
    // Building is transactional: a failed build leaves valid flat storage and exact queries.
    invalidateGraph();
    const size_t n = this->count;
    vecsim_stl::vector<idType> graph(checkedProduct(n, params_.maxDegree), idType{0},
                                     this->allocator);
    vecsim_stl::vector<size_t> degrees(n, size_t{0}, this->allocator);
    if (n < 2) {
        graph_.swap(graph);
        degrees_.swap(degrees);
        graphReady_ = true;
        return;
    }

    const size_t cap = std::min(params_.candidateLimit, n - 1);
    vecsim_stl::vector<Candidate> candidates(checkedProduct(n, cap), this->allocator);
    vecsim_stl::vector<size_t> candidateCounts(n, size_t{0}, this->allocator);
    std::mt19937_64 random(params_.seed);

    // Project each point once. Subtracting the source projection from the target projection
    // hashes the residual direction, rather than the point's position relative to the origin.
    // Double precision avoids overflow for finite FP32 input during these dot products.
    const size_t bits = params_.hashBits;
    vecsim_stl::vector<double> planes(checkedProduct(bits, this->dim), this->allocator);
    std::normal_distribution<double> gaussian(0.0, 1.0);
    for (double &component : planes) {
        component = gaussian(random);
    }
    vecsim_stl::vector<double> sketches(checkedProduct(n, bits), 0.0, this->allocator);
    for (size_t i = 0; i < n; ++i) {
        const float *point = this->getDataByInternalId(static_cast<idType>(i));
        for (size_t bit = 0; bit < bits; ++bit) {
            double projection = 0;
            for (size_t d = 0; d < this->dim; ++d) {
                projection += double(point[d]) * planes[bit * this->dim + d];
            }
            sketches[i * bits + bit] = projection;
        }
    }
    auto merge = [&](idType source, idType target, float distance) {
        uint32_t hash = 0;
        for (size_t bit = 0; bit < bits; ++bit) {
            if (sketches[size_t(target) * bits + bit] > sketches[size_t(source) * bits + bit]) {
                hash |= uint32_t{1} << bit;
            }
        }
        Candidate incoming{distance, target, hash};
        Candidate *row = candidates.data() + size_t(source) * cap;
        size_t &size = candidateCounts[source];
        size_t worst = 0;
        for (size_t j = 0; j < size; ++j) {
            if (row[j].hash == hash) {
                if (nearer(incoming, row[j])) {
                    row[j] = incoming;
                }
                return;
            }
            if (nearer(row[worst], row[j])) {
                worst = j;
            }
        }
        if (size < cap) {
            row[size++] = incoming;
        } else if (nearer(incoming, row[worst])) {
            row[worst] = incoming;
        }
    };

    auto processLeaf = [&](const vecsim_stl::vector<idType> &ids, size_t begin, size_t end) {
        // PiPNN collects only a small local neighborhood from each overlapping leaf. Both
        // directions are considered, while the global reservoir remains candidateLimit wide.
        for (size_t i = begin; i < end; ++i) {
            Candidate first{std::numeric_limits<float>::infinity(),
                            std::numeric_limits<idType>::max(), 0};
            Candidate second = first;
            for (size_t j = begin; j < end; ++j) {
                if (i == j) {
                    continue;
                }
                Candidate value{this->calcDistance(this->getDataByInternalId(ids[i]),
                                                   this->getDataByInternalId(ids[j])),
                                ids[j], 0};
                if (nearer(value, first)) {
                    second = first;
                    first = value;
                } else if (nearer(value, second)) {
                    second = value;
                }
            }
            for (const Candidate &value : {first, second}) {
                if (value.id != std::numeric_limits<idType>::max()) {
                    merge(ids[i], value.id, value.distance);
                    merge(value.id, ids[i], value.distance);
                }
            }
        }
    };

    vecsim_stl::vector<Bucket> pending(this->allocator);
    pending.emplace_back(this->allocator, 0);
    pending.back().ids.resize(n);
    std::iota(pending.back().ids.begin(), pending.back().ids.end(), idType{0});
    while (!pending.empty()) {
        Bucket bucket = std::move(pending.back());
        pending.pop_back();
        if (bucket.ids.size() <= params_.leafSize) {
            processLeaf(bucket.ids, 0, bucket.ids.size());
            continue;
        }
        // Sampling a bounded number of leaders keeps assignment linear in the bucket size.
        // Overlap is limited to the first two levels, preventing replication at every level.
        const size_t leaderCount =
            std::min(bucket.ids.size(),
                     std::min(size_t{512},
                              std::max(params_.fanout + 1,
                                       std::max(size_t{3}, bucket.ids.size() / params_.leafSize))));
        const size_t overlap = bucket.depth < 2 ? std::min(params_.fanout, leaderCount) : 1;
        vecsim_stl::vector<idType> leaders(this->allocator);
        leaders = bucket.ids;
        // Partial Fisher-Yates samples distinct leaders without shuffling the whole bucket.
        for (size_t j = 0; j < leaderCount; ++j) {
            const size_t chosen = j + random() % (leaders.size() - j);
            std::swap(leaders[j], leaders[chosen]);
        }
        leaders.resize(leaderCount);
        vecsim_stl::vector<Bucket> children(this->allocator);
        children.reserve(leaderCount);
        for (size_t j = 0; j < leaderCount; ++j) {
            children.emplace_back(this->allocator, bucket.depth + 1);
        }
        vecsim_stl::vector<std::pair<float, size_t>> distances(leaderCount, this->allocator);
        for (idType id : bucket.ids) {
            for (size_t j = 0; j < leaderCount; ++j) {
                distances[j] = {this->calcDistance(this->getDataByInternalId(id),
                                                   this->getDataByInternalId(leaders[j])),
                                j};
            }
            std::partial_sort(distances.begin(), distances.begin() + overlap, distances.end());
            for (size_t j = 0; j < overlap; ++j) {
                children[distances[j].second].ids.push_back(id);
            }
        }
        for (Bucket &child : children) {
            if (child.ids.empty()) {
                continue;
            }
            // Coincident points can choose the same leaders forever. A depth limit and a
            // strict shrink check terminate that case using bounded deterministic leaves.
            if (bucket.depth >= 4 ||
                child.ids.size() >= bucket.ids.size() - bucket.ids.size() / 5) {
                for (size_t begin = 0; begin < child.ids.size(); begin += params_.leafSize) {
                    processLeaf(child.ids, begin,
                                std::min(begin + params_.leafSize, child.ids.size()));
                }
            } else {
                pending.push_back(std::move(child));
            }
        }
    }

    for (size_t source = 0; source < n; ++source) {
        Candidate *row = candidates.data() + source * cap;
        std::sort(row, row + candidateCounts[source], nearer);
        size_t &degree = degrees[source];
        // Reserve one edge for a directed spanning cycle. This is a connectivity safeguard
        // within the configured degree bound, including fully duplicated datasets.
        const idType successor = static_cast<idType>((source + 1) % n);
        for (size_t j = 0; j < candidateCounts[source] && degree < params_.maxDegree - 1; ++j) {
            const Candidate &candidate = row[j];
            if (candidate.id == successor) {
                continue;
            }
            bool occluded = false;
            for (size_t edge = 0; edge < degree; ++edge) {
                const idType selected = graph[source * params_.maxDegree + edge];
                const float between = this->calcDistance(this->getDataByInternalId(selected),
                                                         this->getDataByInternalId(candidate.id));
                if (params_.alpha * double(between) <= double(candidate.distance)) {
                    occluded = true;
                    break;
                }
            }
            if (!occluded) {
                graph[source * params_.maxDegree + degree++] = candidate.id;
            }
        }
        graph[source * params_.maxDegree + degree++] = successor;
    }
    graph_.swap(graph);
    degrees_.swap(degrees);
    graphReady_ = true;
}

VecSimQueryReply *PiPNNIndex::topKQuery(const void *queryBlob, size_t k,
                                        VecSimQueryParams *queryParams) const {
    k = std::min(k, size_t(this->count));
    if (k == 0) {
        this->lastMode = STANDARD_KNN;
        return new VecSimQueryReply(this->allocator);
    }
    validateVector(queryBlob);
    if (!graphReady_) {
        auto reply = std::make_unique<VecSimQueryReply>(this->allocator);
        this->lastMode = STANDARD_KNN;
        auto query = this->preprocessQuery(queryBlob);
        void *timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
        using ScoredLabel = std::pair<float, labelType>;
        vecsim_stl::vector<ScoredLabel> storage(this->allocator);
        std::priority_queue<ScoredLabel, vecsim_stl::vector<ScoredLabel>> nearest(
            std::less<ScoredLabel>{}, std::move(storage));
        for (idType id = 0; id < this->count; ++id) {
            if (VECSIM_TIMEOUT(timeoutCtx)) {
                reply->code = VecSim_QueryReply_TimedOut;
                return reply.release();
            }
            const ScoredLabel candidate{
                this->calcDistanceForQuery(this->getDataByInternalId(id), query.get()),
                this->getVectorLabel(id)};
            if (nearest.size() < k || candidate < nearest.top()) {
                nearest.push(candidate);
                if (nearest.size() > k) {
                    nearest.pop();
                }
            }
        }
        reply->results.resize(nearest.size());
        for (auto it = reply->results.rbegin(); it != reply->results.rend(); ++it) {
            std::tie(it->score, it->id) = nearest.top();
            nearest.pop();
        }
        return reply.release();
    }
    auto reply = std::make_unique<VecSimQueryReply>(this->allocator);
    this->lastMode = STANDARD_KNN;
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
    auto query = this->preprocessQuery(queryBlob);
    const size_t requestedEf = queryParams && queryParams->pipnnRuntimeParams.efRuntime
                                   ? queryParams->pipnnRuntimeParams.efRuntime
                                   : params_.efRuntime;
    const size_t beam = std::min(size_t(this->count), std::max(k, requestedEf));
    using ScoredId = std::pair<float, idType>;
    vecsim_stl::vector<ScoredId> frontierStorage(this->allocator);
    vecsim_stl::vector<ScoredId> bestStorage(this->allocator);
    std::priority_queue<ScoredId, vecsim_stl::vector<ScoredId>, std::greater<ScoredId>> frontier(
        std::greater<ScoredId>{}, std::move(frontierStorage));
    std::priority_queue<ScoredId, vecsim_stl::vector<ScoredId>> best(std::less<ScoredId>{},
                                                                     std::move(bestStorage));
    vecsim_stl::unordered_set<idType> visited(this->allocator);
    auto visit = [&](idType id) {
        if (!visited.insert(id).second) {
            return;
        }
        ScoredId candidate{this->calcDistanceForQuery(this->getDataByInternalId(id), query.get()),
                           id};
        if (best.size() < beam || candidate < best.top()) {
            best.push(candidate);
            frontier.push(candidate);
            if (best.size() > beam) {
                best.pop();
            }
        }
    };
    // Deterministic starting points avoid dependence on the insertion order of a single entry.
    const size_t startCount = std::min(size_t{4}, size_t(this->count));
    for (size_t i = 0; i < startCount; ++i) {
        if (VECSIM_TIMEOUT(timeoutCtx)) {
            reply->code = VecSim_QueryReply_TimedOut;
            return reply.release();
        }
        visit(static_cast<idType>(i * (size_t(this->count) / startCount)));
    }
    while (!frontier.empty()) {
        if (VECSIM_TIMEOUT(timeoutCtx)) {
            reply->code = VecSim_QueryReply_TimedOut;
            return reply.release();
        }
        const ScoredId next = frontier.top();
        frontier.pop();
        if (best.size() == beam && next > best.top()) {
            break;
        }
        for (size_t j = 0; j < degrees_[next.second]; ++j) {
            if (VECSIM_TIMEOUT(timeoutCtx)) {
                reply->code = VecSim_QueryReply_TimedOut;
                return reply.release();
            }
            visit(graph_[size_t(next.second) * params_.maxDegree + j]);
        }
    }
    vecsim_stl::vector<VecSimQueryResult> ordered(this->allocator);
    ordered.reserve(best.size());
    while (!best.empty()) {
        const auto [score, id] = best.top();
        best.pop();
        ordered.push_back({this->getVectorLabel(id), score});
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
        return a.score < b.score || (a.score == b.score && a.id < b.id);
    });
    ordered.resize(std::min(k, ordered.size()));
    reply->results = std::move(ordered);
    return reply.release();
}

VecSimIndexBasicInfo PiPNNIndex::basicInfo() const {
    auto info = Base::basicInfo();
    info.algo = VecSimAlgo_PIPNN;
    return info;
}

VecSimIndexDebugInfo PiPNNIndex::debugInfo() const {
    VecSimIndexDebugInfo info{};
    info.commonInfo = this->getCommonInfo();
    info.commonInfo.basicInfo = basicInfo();
    return info;
}

VecSimDebugInfoIterator *PiPNNIndex::debugInfoIterator() const {
    const auto info = this->debugInfo();
    // Reserve common, PiPNN and the C API's shared-memory field before publishing the iterator.
    auto iterator = std::make_unique<VecSimDebugInfoIterator>(14, this->allocator);
    iterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::ALGORITHM_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimCommonStrings::PIPNN_STRING}}});
    this->addCommonInfoToIterator(iterator.get(), info.commonInfo);
    iterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = this->getBlockSize()}}});
    iterator->addInfoField(
        VecSim_InfoField{.fieldName = "GRAPH_READY",
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = graphReady_}}});
    iterator->addInfoField(
        VecSim_InfoField{.fieldName = "MAX_DEGREE",
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = params_.maxDegree}}});
    return iterator.release();
}

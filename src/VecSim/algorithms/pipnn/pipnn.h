/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include <string>

// An explicitly built, single-value FP32/L2 PiPNN graph. Storage, exact range queries and
// batch iteration use the flat index contracts. Callers must serialize all operations, including
// queries and batch iteration, because the inherited query statistics are mutable.
class PiPNNIndex final : public BruteForceIndex_Single<float, float> {
    using Base = BruteForceIndex_Single<float, float>;

    PiPNNParams params_;
    // Each row occupies maxDegree slots, only degrees_[id] of which are initialized edges.
    vecsim_stl::vector<idType> graph_;
    vecsim_stl::vector<size_t> degrees_;
    bool graphReady_ = false;

    PiPNNIndex(const BFParams &storageParams, const PiPNNParams *params,
               const AbstractIndexInitParams &abstractInitParams,
               const IndexComponents<float, float> &components);
    void invalidateGraph();
    void validateVector(const void *blob) const;

public:
    PiPNNIndex(const PiPNNParams *params, const AbstractIndexInitParams &abstractInitParams,
               const IndexComponents<float, float> &components);

    int addVector(const void *vector_data, labelType label) override;
    int deleteVector(labelType label) override;
    int deleteVectorById(labelType label, idType id) override;
    std::unordered_map<idType, std::pair<idType, labelType>>
    deleteVectorAndGetUpdatedIds(labelType label) override;
    VecSimUpdateCode updateVectors(labelType label, const void *new_blobs, size_t n) override;
    VecSimRelabelCode relabelVector(labelType old_label, labelType new_label) override;
    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override;

    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override;
    using Base::rangeQuery;
    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams) const override;
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override;
    VecSimIndexBasicInfo basicInfo() const override;
    VecSimIndexDebugInfo debugInfo() const override;
    VecSimDebugInfoIterator *debugInfoIterator() const override;

    // No query builds a graph implicitly. A successful mutation discards the graph and queries
    // fall back to exact scanning until the caller explicitly builds again.
    void buildGraph();
    bool isGraphReady() const { return graphReady_; }
    void saveIndex(const std::string &path) const;
    static PiPNNIndex *loadIndex(const std::string &path, void *logCtx = nullptr);
};

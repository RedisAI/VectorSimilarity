/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"

namespace TQHNSWDetails {

template <typename DataType, typename DistType>
class TQHNSWIndex_Single : public HNSWIndex_Single<DataType, DistType> {
public:
    TQHNSWIndex_Single(const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
                       const IndexComponents<DataType, DistType> &components,
                       size_t random_seed = 100)
        : HNSWIndex_Single<DataType, DistType>(params, abstractInitParams, components,
                                               random_seed) {}

    VecSimIndexDebugInfo debugInfo() const override {
        VecSimIndexDebugInfo info = HNSWIndex_Single<DataType, DistType>::debugInfo();
        info.commonInfo.basicInfo.algo = VecSimAlgo_TQ_HNSW;
        return info;
    }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_TQ_HNSW;
        info.isTiered = false;
        return info;
    }

#ifdef BUILD_TESTS
    void saveIndex(const std::string &location) override {
        UNUSED(location);
        throw std::runtime_error("TQ-HNSW serialization is not supported yet");
    }
#endif
};

template <typename DataType, typename DistType>
class TQHNSWIndex_Multi : public HNSWIndex_Multi<DataType, DistType> {
public:
    TQHNSWIndex_Multi(const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
                      const IndexComponents<DataType, DistType> &components,
                      size_t random_seed = 100)
        : HNSWIndex_Multi<DataType, DistType>(params, abstractInitParams, components, random_seed) {
    }

    VecSimIndexDebugInfo debugInfo() const override {
        VecSimIndexDebugInfo info = HNSWIndex_Multi<DataType, DistType>::debugInfo();
        info.commonInfo.basicInfo.algo = VecSimAlgo_TQ_HNSW;
        return info;
    }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_TQ_HNSW;
        info.isTiered = false;
        return info;
    }

#ifdef BUILD_TESTS
    void saveIndex(const std::string &location) override {
        UNUSED(location);
        throw std::runtime_error("TQ-HNSW serialization is not supported yet");
    }
#endif
};

} // namespace TQHNSWDetails

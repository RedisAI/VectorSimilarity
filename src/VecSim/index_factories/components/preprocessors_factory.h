/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/vec_sim_common.h"

struct PreprocessorsContainerParams {
    VecSimMetric metric;
    size_t dim;
    unsigned char alignment;
};

template <typename DataType>
PreprocessorsContainerAbstract *
CreatePreprocessorsContainer(std::shared_ptr<VecSimAllocator> allocator,
                             PreprocessorsContainerParams params) {

    if (params.metric == VecSimMetric_Cosine) {
        auto multiPPContainer =
            new (allocator) MultiPreprocessorsContainer<DataType, 1>(allocator, params.alignment);
        auto cosine_preprocessor =
            new (allocator) CosinePreprocessor<DataType>(allocator, params.dim);
        int next_valid_pp_index = multiPPContainer->addPreprocessor(cosine_preprocessor);
        UNUSED(next_valid_pp_index);
        assert(next_valid_pp_index != -1 && "Cosine preprocessor was not added correctly");
        return multiPPContainer;
    }

    return new (allocator) PreprocessorsContainerAbstract(allocator, params.alignment);
}

template <typename DataType>
size_t EstimatePreprocessorsContainerMemory(VecSimMetric metric, bool is_normalized = false) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    VecSimMetric pp_metric;
    if (is_normalized && metric == VecSimMetric_Cosine) {
        pp_metric = VecSimMetric_IP;
    } else {
        pp_metric = metric;
    }

    if (pp_metric == VecSimMetric_Cosine) {
        constexpr size_t n_preprocessors = 1;
        // One entry in preprocessors array
        size_t est =
            allocations_overhead + sizeof(MultiPreprocessorsContainer<DataType, n_preprocessors>);
        est += allocations_overhead + sizeof(CosinePreprocessor<DataType>);
        return est;
    }

    return allocations_overhead + sizeof(PreprocessorsContainerAbstract);
}

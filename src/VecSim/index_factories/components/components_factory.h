/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/spaces/spaces.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/index_factories/components/preprocessors_factory.h"
#include "VecSim/spaces/computer/calculator.h"

template <typename DataType, typename DistType>
IndexComponents<DataType, DistType>
CreateIndexComponents(std::shared_ptr<VecSimAllocator> allocator, VecSimMetric metric, size_t dim,
                      bool is_normalized) {
    unsigned char alignment = 0;
    spaces::dist_func_t<DistType> distFunc =
        spaces::GetDistFunc<DataType, DistType>(metric, dim, &alignment);
    // Currently we have only one distance calculator implementation
    auto indexCalculator = new (allocator) DistanceCalculatorCommon<DistType>(allocator, distFunc);

    auto preprocessors =
        CreatePreprocessorsContainer<DataType>(allocator, metric, dim, is_normalized, alignment);

    return {indexCalculator, preprocessors};
}

template <typename DataType, typename DistType>
size_t EstimateComponentsMemory(VecSimMetric metric, bool is_normalized) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();

    // Currently we have only one distance calculator implementation
    size_t est = allocations_overhead + sizeof(DistanceCalculatorCommon<DistType>);

    est += EstimatePreprocessorsContainerMemory<DataType>(metric, is_normalized);

    return est;
}

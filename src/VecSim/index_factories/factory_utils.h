/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <cstddef>
#include "VecSim/spaces/computer/computer.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/vec_sim_common.h"

// TODO: Implement here the abstract index components creation and utils.

template <typename DataType, typename DistType = DataType>
IndexComputerInterface<DistType> *CreateIndexComputer(std::shared_ptr<VecSimAllocator> allocator,
                                                      VecSimMetric metric, size_t dim) {
    unsigned char alignment = 0;
    spaces::dist_func_t<DistType> distFunc =
        spaces::GetDistFunc<DataType, DistType>(metric, dim, &alignment);
    auto distance_calculator =
        new (allocator) DistanceCalculatorCommon<DistType>(allocator, distFunc);

    if (metric == VecSimMetric_Cosine) {
        auto indexComputer = new (allocator)
            IndexComputerExtended<DistType, spaces::dist_func_t<DistType>, 1>(allocator, alignment,
                                                                              distance_calculator);
        auto cosine_preprocessor = new (allocator) CosinePreprocessor<DataType>(allocator, dim);
        int next_valid_pp_index = indexComputer->addPreprocessor(cosine_preprocessor);
        UNUSED(next_valid_pp_index);
        assert(next_valid_pp_index == 0 && "Cosine preprocessor was not added correctly");

        return indexComputer;
    }

    auto indexComputer = new (allocator)
        IndexComputerBasic<DistType, spaces::dist_func_t<DistType>>(allocator, alignment,
                                                                    distance_calculator);

    return indexComputer;
}

// TODO: move here the size estimation of data and graph containers.
template <typename DataType, typename DistType = DataType>
size_t EstimateComponentsMemory(VecSimMetric metric) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    size_t est = allocations_overhead + sizeof(DistanceCalculatorCommon<DistType>);

    if (metric == VecSimMetric_Cosine) {
        constexpr size_t n_preprocessors = 1;
        est +=
            allocations_overhead +
            sizeof(IndexComputerExtended<DistType, spaces::dist_func_t<DistType>, n_preprocessors>);
        // One entry in preprocessors array
        est += allocations_overhead + sizeof(CosinePreprocessor<DataType>);
        return est;
    }
    est +=
        allocations_overhead + sizeof(IndexComputerBasic<DistType, spaces::dist_func_t<DistType>>);

    return est;
}

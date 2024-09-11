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

template <typename DataType, typename DistType = DataType>
IndexComputerAbstract<DistType> *CreateIndexComputer(std::shared_ptr<VecSimAllocator> allocator,
                                                     VecSimMetric metric, size_t dim) {
    unsigned char alignment = 0;
    spaces::dist_func_t<DistType> distFunc =
        spaces::GetDistFunc<DataType, DistType>(metric, dim, &alignment);
    DistanceCalculatorCommon<DistType> *distance_calculator =
        new (allocator) DistanceCalculatorCommon<DistType>(allocator, distFunc);

    if (metric == VecSimMetric_Cosine) {
        IndexComputerExtended<DistType, spaces::dist_func_t<DistType>> *indexComputer =
            new (allocator) IndexComputerExtended<DistType, spaces::dist_func_t<DistType>>(
                allocator, alignment, 1, distance_calculator);
        PreprocessorAbstract *cosine_preprocessor =
            new (allocator) CosinePreprocessor<DataType>(allocator, dim);
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

template <typename DataType, typename DistType = DataType>
size_t EstimateContainersMemory(VecSimMetric metric) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    size_t est = allocations_overhead + sizeof(DistanceCalculatorCommon<DistType>);

    if (metric == VecSimMetric_Cosine) {
        est += allocations_overhead +
               sizeof(IndexComputerExtended<DistType, spaces::dist_func_t<DistType>>);
        // One entry in preprocessors array
        est += allocations_overhead + 1 * sizeof(PreprocessorAbstract *);
        est += allocations_overhead + sizeof(CosinePreprocessor<DataType>);
        return est;
    }
    est +=
        allocations_overhead + sizeof(IndexComputerBasic<DistType, spaces::dist_func_t<DistType>>);

    return est;
}

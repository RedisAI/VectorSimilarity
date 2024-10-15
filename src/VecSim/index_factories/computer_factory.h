/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/spaces/computer/computer.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/vec_sim_common.h"

template <typename DataType, typename DistType = DataType>
IndexComputerAbstract<DistType> *CreateIndexComputer(std::shared_ptr<VecSimAllocator> allocator,
                                                     VecSimMetric metric, size_t dim) {
    unsigned char alignment = 0;
    spaces::dist_func_t<DistType> distFunc =
        spaces::GetDistFunc<DataType, DistType>(metric, dim, &alignment);
    auto distance_calculator =
        new (allocator) DistanceCalculatorCommon<DistType>(allocator, distFunc);

    PreprocessorsContainerAbstract *preprocessors = nullptr;
    if (metric == VecSimMetric_Cosine) {
        auto multiPPContainer =
            new (allocator) MultiPreprocessorsContainer<DistType, 1>(allocator, alignment);
        auto cosine_preprocessor = new (allocator) CosinePreprocessor<DataType>(allocator, dim);
        int next_valid_pp_index = multiPPContainer->addPreprocessor(cosine_preprocessor);
        UNUSED(next_valid_pp_index);
        assert(next_valid_pp_index == 0 && "Cosine preprocessor was not added correctly");
        preprocessors = multiPPContainer;

        return new (allocator) IndexComputerBasic<DistType, spaces::dist_func_t<DistType>>(
            allocator, preprocessors, distance_calculator);
    }

    // The index computer will be created with the default preprocessors container.
    return new (allocator) IndexComputerBasic<DistType, spaces::dist_func_t<DistType>>(
        allocator, alignment, distance_calculator);
}

template <typename DataType, typename DistType = DataType>
size_t EstimateComputerMemory(VecSimMetric metric) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    size_t est = allocations_overhead + sizeof(DistanceCalculatorCommon<DistType>);

    if (metric == VecSimMetric_Cosine) {
        constexpr size_t n_preprocessors = 1;
        // One entry in preprocessors array
        est +=
            allocations_overhead + sizeof(MultiPreprocessorsContainer<DistType, n_preprocessors>);
        est += allocations_overhead + sizeof(CosinePreprocessor<DistType>);
    } else {
        est += allocations_overhead + sizeof(PreprocessorsContainerAbstract);
    }
    est +=
        allocations_overhead + sizeof(IndexComputerBasic<DistType, spaces::dist_func_t<DistType>>);

    return est;
}

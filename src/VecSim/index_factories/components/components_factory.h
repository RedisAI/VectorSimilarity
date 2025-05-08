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

/**
 * @brief Creates parameters for a preprocessors container based on the given metric, dimension,
 *        normalization flag, and alignment.
 *
 * @tparam DataType The data type of the vector elements (e.g., float, int).
 * @param metric The similarity metric to be used (e.g., Cosine, Inner Product).
 * @param dim The dimensionality of the vectors.
 * @param is_normalized A flag indicating whether the vectors are already normalized.
 * @param alignment The alignment requirement for the data.
 * @return A PreprocessorsContainerParams object containing the processed parameters:
 *         - metric: The adjusted metric based on the input and normalization flag.
 *         - dim: The dimensionality of the vectors.
 *         - alignment: The alignment requirement for the data.
 *         - processed_bytes_count: The size of the processed data blob in bytes.
 *
 * @details
 * If the metric is Cosine and the data type is integral, the processed bytes count may include
 * additional space for normalization (currently commented out). If the vectors are already
 * normalized (is_normalized == true), the metric is adjusted to Inner Product (IP) to skip
 * redundant normalization during preprocessing.
 */
template <typename DataType>
PreprocessorsContainerParams CreatePreprocessorsContainerParams(VecSimMetric metric, size_t dim,
                                                                bool is_normalized,
                                                                unsigned char alignment) {
    // By default the processed blob size is the same as the original blob size.
    size_t processed_bytes_count = dim * sizeof(DataType);

    // If the index metric is Cosine, and
    VecSimMetric pp_metric = metric;
    if (metric == VecSimMetric_Cosine) {
        // if metric is cosine and DataType is integral, the processed_bytes_count includes the
        // norm appended to the vector.
        if (std::is_integral<DataType>::value) {
            processed_bytes_count += sizeof(float);
        }
        // if is_normalized == true, we will enforce skipping normalizing vector and query blobs by
        // setting the metric to IP.
        if (is_normalized) {
            pp_metric = VecSimMetric_IP;
        }
    }
    return {.metric = pp_metric,
            .dim = dim,
            .alignment = alignment,
            .processed_bytes_count = processed_bytes_count};
}

template <typename DataType, typename DistType>
IndexComponents<DataType, DistType>
CreateIndexComponents(std::shared_ptr<VecSimAllocator> allocator, VecSimMetric metric, size_t dim,
                      bool is_normalized) {
    unsigned char alignment = 0;
    spaces::dist_func_t<DistType> distFunc =
        spaces::GetDistFunc<DataType, DistType>(metric, dim, &alignment);
    // Currently we have only one distance calculator implementation
    auto indexCalculator = new (allocator) DistanceCalculatorCommon<DistType>(allocator, distFunc);

    PreprocessorsContainerParams ppParams =
        CreatePreprocessorsContainerParams<DataType>(metric, dim, is_normalized, alignment);
    auto preprocessors = CreatePreprocessorsContainer<DataType>(allocator, ppParams);

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

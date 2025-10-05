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

// Create index components with scalar quantization enabled
// This creates INT8 distance calculator + quantization preprocessor
// For Cosine metric, also adds normalization preprocessor before quantization
template <typename DataType, typename DistType>
IndexComponents<DataType, DistType>
CreateQuantizedIndexComponents(std::shared_ptr<VecSimAllocator> allocator, VecSimMetric metric,
                                size_t dim, bool is_normalized) {
    unsigned char alignment = 0;

    // For Cosine metric with quantization, use Inner Product distance
    // (cosine_similarity = inner_product for normalized vectors)
    // For other metrics, use the metric as-is
    VecSimMetric distance_metric = (metric == VecSimMetric_Cosine) ? VecSimMetric_IP : metric;

    // Use INT8 distance function for quantized vectors
    spaces::dist_func_t<DistType> distFunc =
        spaces::GetDistFunc<int8_t, DistType>(distance_metric, dim, &alignment);

    auto indexCalculator = new (allocator) DistanceCalculatorCommon<DistType>(allocator, distFunc);

    // Create preprocessor container with space for 2 preprocessors (normalization + quantization)
    constexpr size_t n_preprocessors = 2;
    auto preprocessors =
        new (allocator) MultiPreprocessorsContainer<DataType, n_preprocessors>(allocator, alignment);

    // For Cosine metric, add normalization preprocessor first
    // This ensures vectors are in [-1, 1] range before quantization
    if (metric == VecSimMetric_Cosine && !is_normalized) {
        auto cosine_preprocessor =
            new (allocator) CosinePreprocessor<DataType>(allocator, dim, dim * sizeof(DataType));
        int next_idx = preprocessors->addPreprocessor(cosine_preprocessor);
        UNUSED(next_idx);
        assert(next_idx != -1 && "Failed to add Cosine preprocessor");
    }

    // Add scalar quantization preprocessor
    // This quantizes float vectors to int8 assuming [-1, 1] range
    auto quant_preprocessor =
        new (allocator) ScalarQuantizationPreprocessor<DataType>(allocator, dim);
    int next_idx2 = preprocessors->addPreprocessor(quant_preprocessor);
    UNUSED(next_idx2);
    assert(next_idx2 != -1 && "Failed to add quantization preprocessor");

    return {indexCalculator, preprocessors};
}

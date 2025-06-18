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
    size_t processed_bytes_count;
};

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
 * additional space for normalization. If the vectors are already
 * normalized (is_normalized == true), the metric is adjusted to Inner Product (IP) to skip
 * redundant normalization during preprocessing.
 */
template <typename DataType>
PreprocessorsContainerParams CreatePreprocessorsContainerParams(VecSimMetric metric, size_t dim,
                                                                bool is_normalized,
                                                                unsigned char alignment) {
    // By default the processed blob size is the same as the original blob size.
    size_t processed_bytes_count = dim * sizeof(DataType);

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

template <typename DataType>
PreprocessorsContainerAbstract *
CreatePreprocessorsContainer(std::shared_ptr<VecSimAllocator> allocator,
                             PreprocessorsContainerParams params) {

    if (params.metric == VecSimMetric_Cosine) {
        auto multiPPContainer =
            new (allocator) MultiPreprocessorsContainer<DataType, 1>(allocator, params.alignment);
        auto cosine_preprocessor = new (allocator)
            CosinePreprocessor<DataType>(allocator, params.dim, params.processed_bytes_count);
        int next_valid_pp_index = multiPPContainer->addPreprocessor(cosine_preprocessor);
        UNUSED(next_valid_pp_index);
        assert(next_valid_pp_index != -1 && "Cosine preprocessor was not added correctly");
        return multiPPContainer;
    }

    return new (allocator) PreprocessorsContainerAbstract(allocator, params.alignment);
}

template <typename DataType>
PreprocessorsContainerAbstract *
CreatePreprocessorsContainer(std::shared_ptr<VecSimAllocator> allocator, VecSimMetric metric,
                             size_t dim, bool is_normalized, unsigned char alignment) {

    PreprocessorsContainerParams ppParams =
        CreatePreprocessorsContainerParams<DataType>(metric, dim, is_normalized, alignment);
    return CreatePreprocessorsContainer<DataType>(allocator, ppParams);
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

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
#include "VecSim/spaces/IP_space.h"
#include "VecSim/types/sq8_a8.h"
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

    // TODO: take into account quantization
    auto preprocessors =
        CreatePreprocessorsContainer<DataType>(allocator, metric, dim, is_normalized, alignment);

    return {indexCalculator, preprocessors};
}

// Asymmetric dispatch reports alignment for the stored operand only. Ask the query type's
// dispatcher for the query allocation alignment.
template <typename DataType>
[[nodiscard]] unsigned char GetQueryAlignment(VecSimMetric metric, size_t dim) {
    unsigned char alignment = 0;
    spaces::GetDistFunc<DataType, float>(metric, dim, &alignment);
    return alignment;
}

// Builds the quantization preprocessor, its container and the distance calculator of a quantized
// index. query_func compares a stored blob with a query blob, sym_func two stored blobs.
template <typename DataType, bool WithMean, typename Preprocessor, typename Calculator>
IndexComponents<DataType, float>
AssembleQuantizedIndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                                 const float *mean_ptr, spaces::dist_func_t<float> query_func,
                                 spaces::dist_func_t<float> sym_func, unsigned char query_alignment,
                                 unsigned char storage_alignment) {
    PreprocessorInterface *pp = nullptr;
    IndexCalculatorInterface<float> *calc = nullptr;

    if constexpr (WithMean) {
        vecsim_stl::vector<float> mean_vec(allocator);
        mean_vec.assign(mean_ptr, mean_ptr + dim);

        float mean_sum_squares = 0.0f;
        for (float v : mean_vec) {
            mean_sum_squares += v * v;
        }

        pp = new (allocator) Preprocessor(allocator, dim, mean_vec);
        calc = new (allocator) Calculator(allocator, query_func, sym_func, mean_sum_squares);
    } else {
        pp = new (allocator) Preprocessor(allocator, dim);
        calc = new (allocator) DistanceCalculatorCommon<float>(allocator, sym_func, query_func);
    }

    auto *container = new (allocator)
        MultiPreprocessorsContainer<DataType, 1>(allocator, query_alignment, storage_alignment);
    [[maybe_unused]] const int ret = container->addPreprocessor(pp);
    assert(ret != -1 && "Quantization preprocessor was not added correctly");

    return {calc, container};
}

// SQ8 IP queries take the A8 path where IP_SQ8_A8_GetDistFunc has a kernel for this dim and CPU:
// FP32 and FP16 queries are quantized once to int8 codes (vecsim_types::sq8_a8_query). Everywhere
// else, and for every other storage type and metric, queries stay asymmetric. arch_opt overrides
// the CPU features for the A8 choice (tests only).
template <typename DataType, VecSimMetric Metric, bool WithMean, typename StorageType>
IndexComponents<DataType, float>
CreateQuantizedIndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                               const float *mean_ptr, const void *arch_opt = nullptr) {
    if constexpr (std::is_same_v<StorageType, vecsim_types::sq8> && Metric == VecSimMetric_IP) {
        unsigned char a8_storage_alignment = 0;
        if (auto a8_func = spaces::IP_SQ8_A8_GetDistFunc(dim, &a8_storage_alignment, arch_opt)) {
            unsigned char storage_alignment = 0;
            auto sym_func =
                spaces::GetDistFunc<StorageType, float>(Metric, dim, &storage_alignment);
            return AssembleQuantizedIndexComponents<
                DataType, WithMean, SQ8A8QueryPreprocessor<DataType, Metric, WithMean>,
                SQ8A8QueryDistanceCalculatorWithNorm<DataType, float, Metric>>(
                allocator, dim, mean_ptr, a8_func, sym_func,
                vecsim_types::sq8_a8_query::blob_alignment,
                spaces::combineAlignments(storage_alignment, a8_storage_alignment));
        }
    }

    using Preprocessor = QuantPreprocessor<DataType, Metric, WithMean, StorageType>;
    using QueryType = typename Preprocessor::QueryType;
    unsigned char storage_alignment = 0, asym_storage_alignment = 0;

    // Graph construction compares two stored quantized blobs; search compares a stored blob with a
    // QueryType query. Both dispatchers report alignment for the stored operand.
    auto sym_func = spaces::GetDistFunc<StorageType, float>(Metric, dim, &storage_alignment);
    auto asym_func =
        spaces::GetDistFunc<StorageType, float, QueryType>(Metric, dim, &asym_storage_alignment);
    storage_alignment = spaces::combineAlignments(storage_alignment, asym_storage_alignment);
    const unsigned char query_alignment = GetQueryAlignment<QueryType>(Metric, dim);

    return AssembleQuantizedIndexComponents<
        DataType, WithMean, Preprocessor,
        DistanceCalculatorWithNorm<DataType, float, Metric, StorageType>>(
        allocator, dim, mean_ptr, asym_func, sym_func, query_alignment, storage_alignment);
}

// Run-time overload: selects WithMean from whether mean_ptr is set.
template <typename DataType, VecSimMetric Metric, typename StorageType>
IndexComponents<DataType, float>
CreateQuantizedIndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                               const float *mean_ptr, const void *arch_opt = nullptr) {
    if (mean_ptr) {
        return CreateQuantizedIndexComponents<DataType, Metric, true, StorageType>(
            allocator, dim, mean_ptr, arch_opt);
    }
    return CreateQuantizedIndexComponents<DataType, Metric, false, StorageType>(allocator, dim,
                                                                                mean_ptr, arch_opt);
}

template <typename DataType, VecSimMetric Metric, bool WithMean>
IndexComponents<DataType, float>
CreateSQ8IndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                         const float *mean_ptr, const void *arch_opt = nullptr) {
    return CreateQuantizedIndexComponents<DataType, Metric, WithMean, vecsim_types::sq8>(
        allocator, dim, mean_ptr, arch_opt);
}

template <typename DataType, VecSimMetric Metric>
IndexComponents<DataType, float>
CreateSQ8IndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                         const float *mean_ptr, const void *arch_opt = nullptr) {
    return CreateQuantizedIndexComponents<DataType, Metric, vecsim_types::sq8>(allocator, dim,
                                                                               mean_ptr, arch_opt);
}

template <typename DataType, typename DistType>
size_t EstimateComponentsMemory(VecSimMetric metric, bool is_normalized) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();

    // Currently we have only one distance calculator implementation
    size_t est = allocations_overhead + sizeof(DistanceCalculatorCommon<DistType>);

    est += EstimatePreprocessorsContainerMemory<DataType>(metric, is_normalized);

    return est;
}

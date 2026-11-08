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

// A prebuilt context allows callers with fitted typed state to use the common wrappers.
template <typename DataType, VecSimMetric Metric, bool WithMean, typename StorageType>
IndexComponents<DataType, float> CreateQuantizedIndexComponents(
    const std::shared_ptr<VecSimAllocator> &allocator, size_t,
    std::shared_ptr<StorageContextFor<StorageType, DataType, Metric, WithMean>> context) {
    using Preprocessor = QuantPreprocessor<DataType, Metric, WithMean, StorageType>;
    using Context = typename Preprocessor::Context;
    auto calculator = std::unique_ptr<IndexCalculatorInterface<float>>(
        new (allocator) QuantizedCalculator<Context>(allocator, context));
    auto preprocessor =
        std::unique_ptr<PreprocessorInterface>(new (allocator) Preprocessor(allocator, context));
    auto container = std::unique_ptr<MultiPreprocessorsContainer<DataType, 1>>(
        new (allocator) MultiPreprocessorsContainer<DataType, 1>(
            allocator, context->queryAlignment(), context->storageAlignment()));
    [[maybe_unused]] const int ret = container->addPreprocessor(preprocessor.get());
    assert(ret != -1 && "Quantization preprocessor was not added correctly");
    preprocessor.release();
    return {calculator.release(), container.release(), context.get()};
}

template <typename DataType, VecSimMetric Metric, bool WithMean, typename StorageType>
IndexComponents<DataType, float>
CreateQuantizedIndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                               const QuantizationInit &init) {
    using Context = StorageContextFor<StorageType, DataType, Metric, WithMean>;
    auto context =
        std::allocate_shared<Context>(VecsimSTLAllocator<Context>(allocator), allocator, dim, init);
    return CreateQuantizedIndexComponents<DataType, Metric, WithMean, StorageType>(
        allocator, dim, std::move(context));
}

template <typename DataType, VecSimMetric Metric, bool WithMean, typename StorageType>
IndexComponents<DataType, float>
CreateQuantizedIndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                               const float *mean_ptr) {
    return CreateQuantizedIndexComponents<DataType, Metric, WithMean, StorageType>(
        allocator, dim, QuantizationInit{.parameters = mean_ptr});
}

// The context interprets parameters; this boundary only selects the mean-aware type.
template <typename DataType, VecSimMetric Metric, typename StorageType>
IndexComponents<DataType, float>
CreateQuantizedIndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                               const QuantizationInit &init) {
    if (init.parameters || init.prepareForTraining) {
        return CreateQuantizedIndexComponents<DataType, Metric, true, StorageType>(allocator, dim,
                                                                                   init);
    }
    return CreateQuantizedIndexComponents<DataType, Metric, false, StorageType>(allocator, dim,
                                                                                init);
}

template <typename DataType, VecSimMetric Metric, typename StorageType>
IndexComponents<DataType, float>
CreateQuantizedIndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                               const float *mean_ptr) {
    return CreateQuantizedIndexComponents<DataType, Metric, StorageType>(
        allocator, dim, QuantizationInit{.parameters = mean_ptr});
}

namespace quantization_detail {

struct AllocationMeasured {};
inline size_t &sharedAllocationBytes() {
    static thread_local size_t bytes = 0;
    return bytes;
}

// The control block layout is implementation-defined. Use the same allocator state size and
// alignment as VecsimSTLAllocator so its rebound allocate_shared block has the same layout.
// Stop at allocation: estimating never constructs a context or fits quantization state.
template <typename T>
struct SharedAllocationProbe {
    using value_type = T;
    std::shared_ptr<VecSimAllocator> allocator;
    SharedAllocationProbe() = default;
    template <typename U>
    SharedAllocationProbe(const SharedAllocationProbe<U> &other) : allocator(other.allocator) {}
    T *allocate(size_t count) {
        sharedAllocationBytes() = count * sizeof(T);
        throw AllocationMeasured{};
    }
    void deallocate(T *, size_t) {}
};

template <typename Context>
size_t contextControlBytes() {
    static_assert(sizeof(SharedAllocationProbe<Context>) == sizeof(VecsimSTLAllocator<Context>));
    static_assert(alignof(SharedAllocationProbe<Context>) == alignof(VecsimSTLAllocator<Context>));
    static const size_t bytes = [] {
        try {
            std::allocate_shared<Context>(SharedAllocationProbe<Context>{},
                                          std::shared_ptr<VecSimAllocator>{}, size_t{1},
                                          QuantizationInit{});
        } catch (const AllocationMeasured &) {
            return sharedAllocationBytes() - sizeof(Context);
        }
        std::terminate();
    }();
    return bytes;
}

} // namespace quantization_detail

template <typename DataType, VecSimMetric Metric, bool WithMean, typename StorageType>
size_t EstimateQuantizedComponentsMemory(size_t dim, const QuantizationInit &init) {
    using Context = StorageContextFor<StorageType, DataType, Metric, WithMean>;
    const size_t overhead = VecSimAllocator::getAllocationOverheadSize();
    return Context::estimateMemory(dim, init) +
           quantization_detail::contextControlBytes<Context>() + overhead +
           sizeof(QuantPreprocessor<DataType, Metric, WithMean, StorageType>) + overhead +
           sizeof(QuantizedCalculator<Context>) + overhead +
           sizeof(MultiPreprocessorsContainer<DataType, 1>) + overhead;
}

template <typename DataType, VecSimMetric Metric, typename StorageType>
size_t EstimateQuantizedComponentsMemory(size_t dim, const QuantizationInit &init) {
    if (init.parameters || init.prepareForTraining) {
        return EstimateQuantizedComponentsMemory<DataType, Metric, true, StorageType>(dim, init);
    }
    return EstimateQuantizedComponentsMemory<DataType, Metric, false, StorageType>(dim, init);
}

template <typename DataType, VecSimMetric Metric, bool WithMean>
IndexComponents<DataType, float>
CreateSQ8IndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                         const float *mean_ptr) {
    return CreateQuantizedIndexComponents<DataType, Metric, WithMean, vecsim_types::sq8>(
        allocator, dim, mean_ptr);
}

template <typename DataType, VecSimMetric Metric>
IndexComponents<DataType, float>
CreateSQ8IndexComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                         const float *mean_ptr) {
    return CreateQuantizedIndexComponents<DataType, Metric, vecsim_types::sq8>(allocator, dim,
                                                                               mean_ptr);
}

template <typename DataType, typename DistType>
size_t EstimateComponentsMemory(VecSimMetric metric, bool is_normalized) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();

    // Currently we have only one distance calculator implementation
    size_t est = allocations_overhead + sizeof(DistanceCalculatorCommon<DistType>);

    est += EstimatePreprocessorsContainerMemory<DataType>(metric, is_normalized);

    return est;
}

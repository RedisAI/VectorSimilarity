/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "VecSim/index_factories/svs_factory.h"

#if HAVE_SVS
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/algorithms/svs/svs.h"
#include "VecSim/index_factories/components/components_factory.h"

namespace SVSFactory {

namespace {
AbstractIndexInitParams NewAbstractInitParams(const VecSimParams *params) {
    auto &svsParams = params->algoParams.svsParams;
    size_t dataSize = VecSimParams_GetDataSize(svsParams.type, svsParams.dim, svsParams.metric);
    return {.allocator = VecSimAllocator::newVecsimAllocator(),
            .dim = svsParams.dim,
            .vecType = svsParams.type,
            .dataSize = dataSize,
            .metric = svsParams.metric,
            .blockSize = svsParams.blockSize,
            .multi = false,
            .logCtx = params->logCtx};
}

// NewVectorsImpl() is the chain of a template helper functions to create a new SVS index.
template <typename MetricType, typename DataType, size_t QuantBits, size_t ResidualBits,
          bool IsLeanVec>
VecSimIndex *NewIndexImpl(const VecSimParams *params, bool is_normalized) {
    auto abstractInitParams = NewAbstractInitParams(params);
    auto &svsParams = params->algoParams.svsParams;
    auto components = CreateIndexComponents<svs_details::vecsim_dt<DataType>, float>(
        abstractInitParams.allocator, svsParams.metric, svsParams.dim, is_normalized);
    bool forcePreprocessing = !is_normalized && svsParams.metric == VecSimMetric_Cosine;
    return new (abstractInitParams.allocator)
        SVSIndex<MetricType, DataType, QuantBits, ResidualBits, IsLeanVec>(
            svsParams, abstractInitParams, components, forcePreprocessing);
}

template <typename MetricType, typename DataType>
VecSimIndex *NewIndexImpl(const VecSimParams *params, bool is_normalized) {
    // Ignore the 'supported' flag because we always fallback at least to the non-quantized mode
    // elsewhere we got code coverage failure for the `supported==false` case
    auto quantBits =
        std::get<0>(svs_details::isSVSQuantBitsSupported(params->algoParams.svsParams.quantBits));

    switch (quantBits) {
    case VecSimSvsQuant_NONE:
        return NewIndexImpl<MetricType, DataType, 0, 0, false>(params, is_normalized);
    case VecSimSvsQuant_8:
        return NewIndexImpl<MetricType, DataType, 8, 0, false>(params, is_normalized);
    case VecSimSvsQuant_4:
        return NewIndexImpl<MetricType, DataType, 4, 0, false>(params, is_normalized);
    case VecSimSvsQuant_4x4:
        return NewIndexImpl<MetricType, DataType, 4, 4, false>(params, is_normalized);
    case VecSimSvsQuant_4x8:
        return NewIndexImpl<MetricType, DataType, 4, 8, false>(params, is_normalized);
    case VecSimSvsQuant_4x8_LeanVec:
        return NewIndexImpl<MetricType, DataType, 4, 8, true>(params, is_normalized);
    case VecSimSvsQuant_8x8_LeanVec:
        return NewIndexImpl<MetricType, DataType, 8, 8, true>(params, is_normalized);
    default:
        // If we got here something is wrong.
        assert(false && "Unsupported quantization mode");
        return NULL;
    }
}

template <typename MetricType>
VecSimIndex *NewIndexImpl(const VecSimParams *params, bool is_normalized) {
    assert(params && params->algo == VecSimAlgo_SVS);
    switch (params->algoParams.svsParams.type) {
    case VecSimType_FLOAT32:
        return NewIndexImpl<MetricType, float>(params, is_normalized);
    case VecSimType_FLOAT16:
        return NewIndexImpl<MetricType, svs::Float16>(params, is_normalized);
    default:
        // If we got here something is wrong.
        assert(false && "Unsupported data type");
        return NULL;
    }
}

VecSimIndex *NewIndexImpl(const VecSimParams *params, bool is_normalized) {
    assert(params && params->algo == VecSimAlgo_SVS);
    switch (params->algoParams.svsParams.metric) {
    case VecSimMetric_L2:
        return NewIndexImpl<svs::distance::DistanceL2>(params, is_normalized);
    case VecSimMetric_IP:
    case VecSimMetric_Cosine:
        return NewIndexImpl<svs::distance::DistanceIP>(params, is_normalized);
    default:
        // If we got here something is wrong.
        assert(false && "Unknown distance metric type");
        return NULL;
    }
}

// QuantizedVectorSize() is the chain of template functions to estimate vector DataSize.
template <typename DataType, size_t QuantBits, size_t ResidualBits, bool IsLeanVec>
constexpr size_t QuantizedVectorSize(size_t dims, size_t alignment = 0) {
    return SVSStorageTraits<DataType, QuantBits, ResidualBits, IsLeanVec>::element_size(dims,
                                                                                        alignment);
}

template <typename DataType>
size_t QuantizedVectorSize(VecSimSvsQuantBits quant_bits, size_t dims, size_t alignment = 0) {
    // Ignore the 'supported' flag because we always fallback at least to the non-quantized mode
    // elsewhere we got code coverage failure for the `supported==false` case
    auto quantBits = std::get<0>(svs_details::isSVSQuantBitsSupported(quant_bits));

    switch (quantBits) {
    case VecSimSvsQuant_NONE:
        return QuantizedVectorSize<DataType, 0, 0, false>(dims, alignment);
    case VecSimSvsQuant_8:
        return QuantizedVectorSize<DataType, 8, 0, false>(dims, alignment);
    case VecSimSvsQuant_4:
        return QuantizedVectorSize<DataType, 4, 0, false>(dims, alignment);
    case VecSimSvsQuant_4x4:
        return QuantizedVectorSize<DataType, 4, 4, false>(dims, alignment);
    case VecSimSvsQuant_4x8:
        return QuantizedVectorSize<DataType, 4, 8, false>(dims, alignment);
    case VecSimSvsQuant_4x8_LeanVec:
        return QuantizedVectorSize<DataType, 4, 8, true>(dims, alignment);
    case VecSimSvsQuant_8x8_LeanVec:
        return QuantizedVectorSize<DataType, 8, 8, true>(dims, alignment);
    default:
        // If we got here something is wrong.
        assert(false && "Unsupported quantization mode");
        return 0;
    }
}

size_t QuantizedVectorSize(VecSimType data_type, VecSimSvsQuantBits quant_bits, size_t dims,
                           size_t alignment = 0) {
    switch (data_type) {
    case VecSimType_FLOAT32:
        return QuantizedVectorSize<float>(quant_bits, dims, alignment);
    case VecSimType_FLOAT16:
        return QuantizedVectorSize<svs::Float16>(quant_bits, dims, alignment);
    default:
        // If we got here something is wrong.
        assert(false && "Unsupported data type");
        return 0;
    }
}

template <typename DataType>
size_t EstimateComponentsMemorySVS(VecSimMetric metric, bool is_normalized) {
    return EstimateComponentsMemory<svs_details::vecsim_dt<DataType>, float>(metric, is_normalized);
}

size_t EstimateComponentsMemorySVS(VecSimType type, VecSimMetric metric, bool is_normalized) {
    switch (type) {
    case VecSimType_FLOAT32:
        return EstimateComponentsMemorySVS<float>(metric, is_normalized);
    case VecSimType_FLOAT16:
        return EstimateComponentsMemorySVS<svs::Float16>(metric, is_normalized);
    default:
        // If we got here something is wrong.
        assert(false && "Unsupported data type");
        return 0;
    }
}
} // namespace

VecSimIndex *NewIndex(const VecSimParams *params, bool is_normalized) {
    return NewIndexImpl(params, is_normalized);
}

size_t EstimateElementSize(const SVSParams *params) {
    using graph_idx_type = uint32_t;
    const auto graph_node_size =
        SVSGraphBuilder<graph_idx_type>::element_size(params->graph_max_degree);
    const auto vector_size = QuantizedVectorSize(params->type, params->quantBits, params->dim);

    return vector_size + graph_node_size;
}

size_t EstimateInitialSize(const SVSParams *params, bool is_normalized) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    size_t est = sizeof(VecSimAllocator) + allocations_overhead;

    // Assume all non-quant floats have same initial size
    // Assume quantBits>0 cases have same sizes
    est += (params->quantBits == 0)
               ? sizeof(SVSIndex<svs::distance::DistanceL2, float, 0, 0, false>)
               : sizeof(SVSIndex<svs::distance::DistanceL2, float, 8, 0, false>);
    est += EstimateComponentsMemorySVS(params->type, params->metric, is_normalized);
    est += sizeof(DataBlocksContainer) + allocations_overhead;
    return est;
}

} // namespace SVSFactory

// This is a temporary solution to avoid breaking the build when SVS is not available
// and to allow the code to compile without SVS support.
// TODO: remove HAVE_SVS when SVS will support all Redis platforms and compilers
#else  // HAVE_SVS
namespace SVSFactory {
VecSimIndex *NewIndex(const VecSimParams *params, bool is_normalized) { return NULL; }
size_t EstimateInitialSize(const SVSParams *params, bool is_normalized) { return -1; }
size_t EstimateElementSize(const SVSParams *params) { return -1; }
}; // namespace SVSFactory
#endif // HAVE_SVS

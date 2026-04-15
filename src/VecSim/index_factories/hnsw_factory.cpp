/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "VecSim/index_factories/components/components_factory.h"
#include "VecSim/index_factories/factory_utils.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/algorithms/tq/tq_flat.h"
#include "VecSim/algorithms/tq/tq_hnsw.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

namespace HNSWFactory {

template <typename DataType, typename DistType = DataType>
inline HNSWIndex<DataType, DistType> *
NewIndex_ChooseMultiOrSingle(const HNSWParams *params,
                             const AbstractIndexInitParams &abstractInitParams,
                             IndexComponents<DataType, DistType> &components) {
    // check if single and return new hnsw_index
    if (params->multi)
        return new (abstractInitParams.allocator)
            HNSWIndex_Multi<DataType, DistType>(params, abstractInitParams, components);
    else
        return new (abstractInitParams.allocator)
            HNSWIndex_Single<DataType, DistType>(params, abstractInitParams, components);
}

template <typename DataType, typename DistType = DataType>
inline HNSWIndex<DataType, DistType> *
NewTQIndex_ChooseMultiOrSingle(const HNSWParams *params,
                               const AbstractIndexInitParams &abstractInitParams,
                               IndexComponents<DataType, DistType> &components) {
    if (params->multi)
        return new (abstractInitParams.allocator)
            TQHNSWDetails::TQHNSWIndex_Multi<DataType, DistType>(params, abstractInitParams,
                                                                 components);
    else
        return new (abstractInitParams.allocator)
            TQHNSWDetails::TQHNSWIndex_Single<DataType, DistType>(params, abstractInitParams,
                                                                  components);
}

inline HNSWParams AsHNSWParams(const TQHNSWParams &params) {
    return {
        .type = params.type,
        .dim = params.dim,
        .metric = params.metric,
        .multi = params.multi,
        .initialCapacity = params.initialCapacity,
        .blockSize = params.blockSize,
        .M = params.M,
        .efConstruction = params.efConstruction,
        .efRuntime = params.efRuntime,
        .epsilon = params.epsilon,
    };
}

inline AbstractIndexInitParams NewTQAbstractInitParams(const TQHNSWParams *params, void *logCtx,
                                                       std::shared_ptr<VecSimAllocator> allocator,
                                                       size_t stored_data_size) {
    return {.allocator = allocator,
            .dim = params->dim,
            .vecType = params->type,
            .storedDataSize = stored_data_size,
            .metric = params->metric,
            .blockSize = params->blockSize,
            .multi = params->multi,
            .isDisk = false,
            .logCtx = logCtx,
            .inputBlobSize = params->dim * VecSimType_sizeof(params->type)};
}

template <VecSimMetric Metric>
VecSimIndex *NewTQIndexImpl(const VecSimParams *params) {
    const auto &tq_params = params->algoParams.tqHnswParams;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    const auto *tq_core_params = reinterpret_cast<const TQFlatParams *>(&tq_params);
    auto components = TQFlatDetails::CreateTQHNSWComponents<Metric>(allocator, tq_core_params);
    auto stored_data_size = TQFlatDetails::GetStorageDataSize<Metric>(tq_core_params);
    auto abstract_init_params =
        NewTQAbstractInitParams(&tq_params, params->logCtx, allocator, stored_data_size);
    HNSWParams hnsw_params = AsHNSWParams(tq_params);
    return NewTQIndex_ChooseMultiOrSingle<float, float>(&hnsw_params, abstract_init_params,
                                                        components);
}

VecSimIndex *NewIndex(const VecSimParams *params, bool is_normalized) {
    if (params->algo == VecSimAlgo_TQ_HNSW) {
        const auto &tq_params = params->algoParams.tqHnswParams;
        if (tq_params.type != VecSimType_FLOAT32) {
            throw std::invalid_argument("TQ-HNSW currently supports FLOAT32 input only");
        }
        switch (tq_params.metric) {
        case VecSimMetric_L2:
            return NewTQIndexImpl<VecSimMetric_L2>(params);
        case VecSimMetric_IP:
            return NewTQIndexImpl<VecSimMetric_IP>(params);
        case VecSimMetric_Cosine:
            return NewTQIndexImpl<VecSimMetric_Cosine>(params);
        }
        return nullptr;
    }

    const HNSWParams *hnswParams = &params->algoParams.hnswParams;
    AbstractIndexInitParams abstractInitParams =
        VecSimFactory::NewAbstractInitParams(hnswParams, params->logCtx, is_normalized);

    if (hnswParams->type == VecSimType_FLOAT32) {
        IndexComponents<float, float> indexComponents = CreateIndexComponents<float, float>(
            abstractInitParams.allocator, hnswParams->metric, hnswParams->dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<float>(hnswParams, abstractInitParams, indexComponents);

    } else if (hnswParams->type == VecSimType_FLOAT64) {
        IndexComponents<double, double> indexComponents = CreateIndexComponents<double, double>(
            abstractInitParams.allocator, hnswParams->metric, hnswParams->dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<double>(hnswParams, abstractInitParams,
                                                    indexComponents);

    } else if (hnswParams->type == VecSimType_BFLOAT16) {
        IndexComponents<bfloat16, float> indexComponents = CreateIndexComponents<bfloat16, float>(
            abstractInitParams.allocator, hnswParams->metric, hnswParams->dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<bfloat16, float>(hnswParams, abstractInitParams,
                                                             indexComponents);
    } else if (hnswParams->type == VecSimType_FLOAT16) {
        IndexComponents<float16, float> indexComponents = CreateIndexComponents<float16, float>(
            abstractInitParams.allocator, hnswParams->metric, hnswParams->dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<float16, float>(hnswParams, abstractInitParams,
                                                            indexComponents);
    } else if (hnswParams->type == VecSimType_INT8) {
        IndexComponents<int8_t, float> indexComponents = CreateIndexComponents<int8_t, float>(
            abstractInitParams.allocator, hnswParams->metric, hnswParams->dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<int8_t, float>(hnswParams, abstractInitParams,
                                                           indexComponents);
    } else if (hnswParams->type == VecSimType_UINT8) {
        IndexComponents<uint8_t, float> indexComponents = CreateIndexComponents<uint8_t, float>(
            abstractInitParams.allocator, hnswParams->metric, hnswParams->dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<uint8_t, float>(hnswParams, abstractInitParams,
                                                            indexComponents);
    }

    // If we got here something is wrong.
    return NULL;
}

VecSimIndex *NewIndex(const HNSWParams *params, bool is_normalized) {
    VecSimParams vecSimParams = {.algoParams = {.hnswParams = HNSWParams{*params}}};
    return NewIndex(&vecSimParams);
}

template <typename DataType, typename DistType = DataType>
inline size_t EstimateInitialSize_ChooseMultiOrSingle(bool is_multi) {
    // check if single or multi and return the size of the matching class struct.
    if (is_multi)
        return sizeof(HNSWIndex_Multi<DataType, DistType>);
    else
        return sizeof(HNSWIndex_Single<DataType, DistType>);
}

size_t EstimateInitialSize(const HNSWParams *params, bool is_normalized) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();

    size_t est = sizeof(VecSimAllocator) + allocations_overhead;
    if (params->type == VecSimType_FLOAT32) {
        est += EstimateComponentsMemory<float, float>(params->metric, is_normalized);
        est += EstimateInitialSize_ChooseMultiOrSingle<float>(params->multi);
    } else if (params->type == VecSimType_FLOAT64) {
        est += EstimateComponentsMemory<double, double>(params->metric, is_normalized);
        est += EstimateInitialSize_ChooseMultiOrSingle<double>(params->multi);
    } else if (params->type == VecSimType_BFLOAT16) {
        est += EstimateComponentsMemory<bfloat16, float>(params->metric, is_normalized);
        est += EstimateInitialSize_ChooseMultiOrSingle<bfloat16, float>(params->multi);
    } else if (params->type == VecSimType_FLOAT16) {
        est += EstimateComponentsMemory<float16, float>(params->metric, is_normalized);
        est += EstimateInitialSize_ChooseMultiOrSingle<float16, float>(params->multi);
    } else if (params->type == VecSimType_INT8) {
        est += EstimateComponentsMemory<int8_t, float>(params->metric, is_normalized);
        est += EstimateInitialSize_ChooseMultiOrSingle<int8_t, float>(params->multi);
    } else if (params->type == VecSimType_UINT8) {
        est += EstimateComponentsMemory<uint8_t, float>(params->metric, is_normalized);
        est += EstimateInitialSize_ChooseMultiOrSingle<uint8_t, float>(params->multi);
    } else {
        throw std::invalid_argument("Invalid params->type");
    }
    est += sizeof(DataBlocksContainer) + allocations_overhead;

    return est;
}

size_t EstimateElementSize(const HNSWParams *params) {

    size_t M = (params->M) ? params->M : HNSW_DEFAULT_M;
    size_t elementGraphDataSize = sizeof(ElementGraphData) + sizeof(idType) * M * 2;

    size_t size_total_data_per_element =
        elementGraphDataSize +
        VecSimParams_GetStoredDataSize(params->type, params->dim, params->metric);

    // when reserving space for new labels in the lookup hash table, each entry is a pointer to a
    // label node (bucket).
    size_t size_label_lookup_entry = sizeof(void *);

    // 1 entry in visited nodes + 1 entry in element metadata map + 1 node lock + (approximately)
    // 1 bucket in labels lookup hash map.
    size_t size_meta_data = sizeof(tag_t) + sizeof(ElementMetaData) +
                            sizeof(vecsim_stl::one_byte_mutex) + size_label_lookup_entry;

    /* Disclaimer: we are neglecting two additional factors that consume memory:
     * 1. The overall bucket size in labels_lookup hash table is usually higher than the number of
     * requested buckets (which is the index capacity), and it is auto selected according to the
     * hashing policy and the max load factor.
     * 2. The incoming edges that aren't bidirectional are stored in a dynamic array
     * (vecsim_stl::vector) Those edges' memory *is omitted completely* from this estimation.
     */
    return size_meta_data + size_total_data_per_element;
}

template <VecSimMetric Metric>
size_t EstimateTQInitialSizeImpl(const TQHNSWParams *params) {
    HNSWParams hnsw_params = AsHNSWParams(*params);
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    size_t est = sizeof(VecSimAllocator) + allocations_overhead;
    est += sizeof(TQHNSWDetails::TQHNSWIndex_Single<float, float>);
    est += allocations_overhead + sizeof(TQFlatDetails::TQDistanceCalculator<Metric>);
    est += allocations_overhead + sizeof(MultiPreprocessorsContainer<float, 1>);
    est += allocations_overhead + sizeof(TQFlatDetails::TQPreprocessor<Metric>);
    est += params->dim * params->dim * sizeof(float);
    est += params->projections * params->dim * sizeof(float);
    est += (size_t{1} << (params->bits - 1)) * 2 * sizeof(float);
    est += sizeof(DataBlocksContainer) + allocations_overhead;
    est += sizeof(tag_t) * RoundUpInitialCapacity(params->initialCapacity, params->blockSize);
    est += EstimateElementSize(&hnsw_params) * RoundUpInitialCapacity(params->initialCapacity,
                                                                      params->blockSize);
    return est;
}

size_t EstimateInitialSize(const TQHNSWParams *params) {
    switch (params->metric) {
    case VecSimMetric_L2:
        return EstimateTQInitialSizeImpl<VecSimMetric_L2>(params);
    case VecSimMetric_IP:
        return EstimateTQInitialSizeImpl<VecSimMetric_IP>(params);
    case VecSimMetric_Cosine:
        return EstimateTQInitialSizeImpl<VecSimMetric_Cosine>(params);
    }
    return 0;
}

size_t EstimateElementSize(const TQHNSWParams *params) {
    HNSWParams hnsw_params = AsHNSWParams(*params);
    size_t M = (hnsw_params.M) ? hnsw_params.M : HNSW_DEFAULT_M;
    size_t elementGraphDataSize = sizeof(ElementGraphData) + sizeof(idType) * M * 2;
    size_t size_total_data_per_element =
        elementGraphDataSize +
        TQFlatDetails::GetStorageDataSize<VecSimMetric_Cosine>(
            reinterpret_cast<const TQFlatParams *>(params));
    size_t size_label_lookup_entry = sizeof(void *);
    size_t size_meta_data = sizeof(tag_t) + sizeof(ElementMetaData) + size_label_lookup_entry;
    return size_meta_data + size_total_data_per_element;
}

#ifdef BUILD_TESTS

template <typename DataType, typename DistType = DataType>
inline VecSimIndex *NewIndex_ChooseMultiOrSingle(std::ifstream &input, const HNSWParams *params,
                                                 const AbstractIndexInitParams &abstractInitParams,
                                                 IndexComponents<DataType, DistType> &components,
                                                 HNSWSerializer::EncodingVersion version) {
    HNSWIndex<DataType, DistType> *index = nullptr;
    // check if single and call the ctor that loads index information from file.
    if (params->multi)
        index = new (abstractInitParams.allocator) HNSWIndex_Multi<DataType, DistType>(
            input, params, abstractInitParams, components, version);
    else
        index = new (abstractInitParams.allocator) HNSWIndex_Single<DataType, DistType>(
            input, params, abstractInitParams, components, version);

    index->restoreGraph(input, version);

    return index;
}

// Initialize @params from file for V3
static void InitializeParams(std::ifstream &source_params, HNSWParams &params) {
    Serializer::readBinaryPOD(source_params, params.dim);
    Serializer::readBinaryPOD(source_params, params.type);
    Serializer::readBinaryPOD(source_params, params.metric);
    Serializer::readBinaryPOD(source_params, params.blockSize);
    Serializer::readBinaryPOD(source_params, params.multi);
    Serializer::readBinaryPOD(source_params, params.initialCapacity);
}

VecSimIndex *NewIndex(const std::string &location, bool is_normalized) {

    std::ifstream input(location, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    HNSWSerializer::EncodingVersion version = HNSWSerializer::ReadVersion(input);

    VecSimAlgo algo = VecSimAlgo_BF;
    Serializer::readBinaryPOD(input, algo);
    if (algo != VecSimAlgo_HNSWLIB) {
        input.close();
        auto bad_name = VecSimAlgo_ToString(algo);
        if (bad_name == nullptr) {
            bad_name = "Unknown (corrupted file?)";
        }
        throw std::runtime_error(
            std::string("Cannot load index: Expected HNSW file but got algorithm type: ") +
            bad_name);
    }

    HNSWParams params;
    InitializeParams(input, params);

    VecSimParams vecsimParams = {.algo = VecSimAlgo_HNSWLIB,
                                 .algoParams = {.hnswParams = HNSWParams{params}}};

    AbstractIndexInitParams abstractInitParams =
        VecSimFactory::NewAbstractInitParams(&params, vecsimParams.logCtx, is_normalized);
    if (params.type == VecSimType_FLOAT32) {
        IndexComponents<float, float> indexComponents = CreateIndexComponents<float, float>(
            abstractInitParams.allocator, params.metric, abstractInitParams.dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<float>(input, &params, abstractInitParams,
                                                   indexComponents, version);
    } else if (params.type == VecSimType_FLOAT64) {
        IndexComponents<double, double> indexComponents = CreateIndexComponents<double, double>(
            abstractInitParams.allocator, params.metric, abstractInitParams.dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<double>(input, &params, abstractInitParams,
                                                    indexComponents, version);
    } else if (params.type == VecSimType_BFLOAT16) {
        IndexComponents<bfloat16, float> indexComponents = CreateIndexComponents<bfloat16, float>(
            abstractInitParams.allocator, params.metric, abstractInitParams.dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<bfloat16, float>(input, &params, abstractInitParams,
                                                             indexComponents, version);
    } else if (params.type == VecSimType_FLOAT16) {
        IndexComponents<float16, float> indexComponents = CreateIndexComponents<float16, float>(
            abstractInitParams.allocator, params.metric, abstractInitParams.dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<float16, float>(input, &params, abstractInitParams,
                                                            indexComponents, version);
    } else if (params.type == VecSimType_INT8) {
        IndexComponents<int8_t, float> indexComponents = CreateIndexComponents<int8_t, float>(
            abstractInitParams.allocator, params.metric, abstractInitParams.dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<int8_t, float>(input, &params, abstractInitParams,
                                                           indexComponents, version);
    } else if (params.type == VecSimType_UINT8) {
        IndexComponents<uint8_t, float> indexComponents = CreateIndexComponents<uint8_t, float>(
            abstractInitParams.allocator, params.metric, abstractInitParams.dim, is_normalized);
        return NewIndex_ChooseMultiOrSingle<uint8_t, float>(input, &params, abstractInitParams,
                                                            indexComponents, version);
    } else {
        auto bad_name = VecSimType_ToString(params.type);
        if (bad_name == nullptr) {
            bad_name = "Unknown (corrupted file?)";
        }
        throw std::runtime_error(std::string("Cannot load index: bad index data type: ") +
                                 bad_name);
    }
}
#endif

}; // namespace HNSWFactory

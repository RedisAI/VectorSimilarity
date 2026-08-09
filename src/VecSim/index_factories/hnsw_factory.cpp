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
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;
using sq8 = vecsim_types::sq8;

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

template <VecSimMetric Metric>
[[nodiscard]] constexpr size_t GetSQ8StoredDataSize(size_t dim, bool with_norm) {
    static_assert(Metric == VecSimMetric_L2 || Metric == VecSimMetric_IP);

    // WithNorm is a template parameter, so dispatch the runtime flag to the two instantiations.
    return with_norm ? sq8::storage_bytes_count<Metric, true>(dim)
                     : sq8::storage_bytes_count<Metric, false>(dim);
}

// Alignment required by a query blob of type DataType. Per the asymmetric-types contract in
// spaces.h, the hint returned alongside an asymmetric distance function describes its first
// (storage) operand, so the query side must be obtained from the symmetric dispatcher for the
// query's own type. Only that hint is wanted here, never the function it returns, so the call is
// contained in this adapter instead of leaving a discarded value at the call site.
template <typename DataType>
[[nodiscard]] unsigned char GetQueryAlignment(VecSimMetric metric, size_t dim) {
    unsigned char alignment = 0;
    spaces::GetDistFunc<DataType, float>(metric, dim, &alignment);
    return alignment;
}

// Helper to build an SQ8-quantized HNSW index given compile-time DataType and Metric.
template <typename DataType, VecSimMetric Metric>
VecSimIndex *NewIndex_SQ8(const HNSWParams *hnswParams, AbstractIndexInitParams abstractInitParams,
                          const float *mean_ptr) {
    auto &allocator = abstractInitParams.allocator;
    const size_t dim = abstractInitParams.dim;
    const bool with_norm = mean_ptr != nullptr;
    unsigned char storage_alignment = 0, asym_storage_alignment = 0;

    // Override blob size for the SQ8 storage layout.
    abstractInitParams.storedDataSize = GetSQ8StoredDataSize<Metric>(dim, with_norm);
    abstractInitParams.isQuantized = true;

    // Symmetric: both stored vectors are SQ8 blobs.
    auto sym_func = spaces::GetDistFunc<sq8, float>(Metric, dim, &storage_alignment);
    // Asymmetric: stored vector is SQ8 blob, query is DataType.
    auto asym_func =
        spaces::GetDistFunc<sq8, float, DataType>(Metric, dim, &asym_storage_alignment);
    // Both hints describe the same stored blob, so they must be combined rather than overwritten.
    storage_alignment = spaces::combineAlignments(storage_alignment, asym_storage_alignment);
    // Queries stay in DataType and are compared against stored blobs by asym_func.
    const unsigned char query_alignment = GetQueryAlignment<DataType>(Metric, dim);

    PreprocessorInterface *pp = nullptr;
    IndexCalculatorInterface<float> *calc = nullptr;

    if (with_norm) {
        // Mean-centered SQ8 quantization with norm correction.
        vecsim_stl::vector<float> mean_vec(allocator);
        mean_vec.assign(mean_ptr, mean_ptr + dim);

        float mean_sum_squares = 0.0f;
        for (float v : mean_vec) {
            mean_sum_squares += v * v;
        }

        pp = new (allocator) QuantPreprocessor<DataType, Metric, true>(allocator, dim, mean_vec);
        calc = new (allocator) DistanceCalculatorWithNorm<DataType, float, Metric>(
            allocator, asym_func, sym_func, mean_sum_squares);
    } else {
        // Plain SQ8 quantization without mean centering.
        pp = new (allocator) QuantPreprocessor<DataType, Metric>(allocator, dim);
        // sym_func for storage-storage; asym_func for query-storage.
        calc = new (allocator) DistanceCalculatorCommon<float>(allocator, sym_func, asym_func);
    }

    auto *container = new (allocator)
        MultiPreprocessorsContainer<DataType, 1>(allocator, query_alignment, storage_alignment);
    [[maybe_unused]] const int ret = container->addPreprocessor(pp);
    assert(ret != -1 && "SQ8 preprocessor was not added correctly");

    IndexComponents<DataType, float> components{calc, container};
    return NewIndex_ChooseMultiOrSingle<DataType, float>(hnswParams, abstractInitParams,
                                                         components);
}

VecSimIndex *NewIndex(const VecSimParams *params, bool is_normalized) {
    const HNSWParams *hnswParams = &params->algoParams.hnswParams;
    AbstractIndexInitParams abstractInitParams =
        VecSimFactory::NewAbstractInitParams(hnswParams, params->logCtx, is_normalized);

    if (hnswParams->quantType == VecSimQuant_SQ8) {
        if (hnswParams->type != VecSimType_FLOAT32 && hnswParams->type != VecSimType_FLOAT16) {
            return NULL; // SQ8 supports FP32 and FP16 only.
        }

        VecSimMetric metric = hnswParams->metric;
        if (is_normalized && metric == VecSimMetric_Cosine) {
            metric = VecSimMetric_IP;
        }

        if (metric == VecSimMetric_Cosine) {
            return NULL; // SQ8 does not support cosine metric.
        }

        const float *mean_ptr = static_cast<const float *>(hnswParams->quantParams);

        // Mean-centred FP16 L2 is not supported: QuantPreprocessor centres the query and narrows
        // the result back into the FP16 query body, while storage keeps its centred min/delta in
        // FP32. The two then disagree, so an identical vector and query pair yields a non-zero
        // distance (mean 10000 gives a per-component error of 1.0), and a large enough mean
        // overflows FP16 to infinity. Enabling this needs an asymmetric kernel that takes an FP32
        // centred query.
        if (hnswParams->type == VecSimType_FLOAT16 && mean_ptr != nullptr &&
            metric == VecSimMetric_L2) {
            return NULL;
        }

        if (hnswParams->type == VecSimType_FLOAT32) {
            if (metric == VecSimMetric_L2) {
                return NewIndex_SQ8<float, VecSimMetric_L2>(hnswParams, abstractInitParams,
                                                            mean_ptr);
            } else if (metric == VecSimMetric_IP) {
                return NewIndex_SQ8<float, VecSimMetric_IP>(hnswParams, abstractInitParams,
                                                            mean_ptr);
            }
        } else if (hnswParams->type == VecSimType_FLOAT16) {
            if (metric == VecSimMetric_L2) {
                return NewIndex_SQ8<float16, VecSimMetric_L2>(hnswParams, abstractInitParams,
                                                              mean_ptr);
            } else if (metric == VecSimMetric_IP) {
                return NewIndex_SQ8<float16, VecSimMetric_IP>(hnswParams, abstractInitParams,
                                                              mean_ptr);
            }
        }

        // Unreachable today: the checks above leave only FP32/FP16 x L2/IP. The assert makes a
        // debug build shout if a new type or metric ever reaches here, and the return keeps a
        // release build failing closed rather than falling through and silently building an
        // unquantized index instead.
        assert(false && "unhandled SQ8 data type and metric combination");
        return NULL;
    }

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

    if (params->quantType == VecSimQuant_SQ8) {
        if (params->type != VecSimType_FLOAT32 && params->type != VecSimType_FLOAT16) {
            throw std::invalid_argument("Invalid params->type for VecSimQuant_SQ8");
        }
        // Calculator + preprocessor container + preprocessor.
        // Use representative types; sizeof is independent of the template parameters.
        if (params->quantParams) { // mean provided, WithNorm = true
            est += allocations_overhead +
                   sizeof(DistanceCalculatorWithNorm<float, float, VecSimMetric_L2>);
            est += allocations_overhead + sizeof(MultiPreprocessorsContainer<float, 1>);
            est += allocations_overhead + sizeof(QuantPreprocessor<float, VecSimMetric_L2, true>);
            est += allocations_overhead +
                   params->dim * sizeof(float); // mean vector in QuantPreprocessor
        } else {
            est += allocations_overhead + sizeof(DistanceCalculatorCommon<float>);
            est += allocations_overhead + sizeof(MultiPreprocessorsContainer<float, 1>);
            est += allocations_overhead + sizeof(QuantPreprocessor<float, VecSimMetric_L2>);
        }
        est += EstimateInitialSize_ChooseMultiOrSingle<float>(params->multi);
    } else if (params->type == VecSimType_FLOAT32) {
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

    size_t stored_data_size;
    if (params->quantType == VecSimQuant_SQ8) {
        bool with_norm = params->quantParams != nullptr;
        if (params->metric == VecSimMetric_L2) {
            stored_data_size = GetSQ8StoredDataSize<VecSimMetric_L2>(params->dim, with_norm);
        } else {
            stored_data_size = GetSQ8StoredDataSize<VecSimMetric_IP>(params->dim, with_norm);
        }
    } else {
        stored_data_size =
            VecSimParams_GetStoredDataSize(params->type, params->dim, params->metric);
    }

    size_t size_total_data_per_element = elementGraphDataSize + stored_data_size;

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

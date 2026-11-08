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
#include "VecSim/types/turboquant.h"

#include <type_traits>

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

template <VecSimMetric Metric, typename StorageType = sq8>
[[nodiscard]] constexpr size_t GetQuantizedStoredDataSize(size_t dim, bool with_norm) {
    static_assert(Metric == VecSimMetric_L2 || Metric == VecSimMetric_IP);

    return with_norm ? StorageType::template storage_bytes_count<Metric, true>(dim)
                     : StorageType::template storage_bytes_count<Metric, false>(dim);
}

// Cosine over pre-normalized vectors is computed as inner product.
[[nodiscard]] constexpr VecSimMetric ResolveQuantizedMetric(VecSimMetric metric,
                                                            bool is_normalized) {
    return (is_normalized && metric == VecSimMetric_Cosine) ? VecSimMetric_IP : metric;
}

// Keep construction and initial-size validation in sync.
bool QuantizationParamsSupported(VecSimQuantType quant_type, VecSimType type,
                                 VecSimMetric resolved_metric) {
    if (quant_type == VecSimQuant_TQ2 || quant_type == VecSimQuant_TQ3)
        return type == VecSimType_FLOAT32 && resolved_metric == VecSimMetric_IP;
    // Reject unknown quantizers instead of silently creating an unquantized index.
    if (quant_type != VecSimQuant_SQ8) {
        return false;
    }
    // SQ8 kernels accept only FLOAT32 and FLOAT16 input.
    if (type != VecSimType_FLOAT32 && type != VecSimType_FLOAT16) {
        return false;
    }
    // Only L2 and inner product have scalar quantization kernels.
    if (resolved_metric != VecSimMetric_L2 && resolved_metric != VecSimMetric_IP) {
        return false;
    }
    return true;
}

template <typename DataType, VecSimMetric Metric, typename StorageType = sq8>
VecSimIndex *NewIndex_Quantized(const HNSWParams *hnswParams,
                                AbstractIndexInitParams abstractInitParams,
                                const QuantizationInit &init) {
    abstractInitParams.isQuantized = true;
    auto components = CreateQuantizedIndexComponents<DataType, Metric, StorageType>(
        abstractInitParams.allocator, abstractInitParams.dim, init);
    abstractInitParams.storedDataSize = components.quantization->storageBytes();
    auto index =
        NewIndex_ChooseMultiOrSingle<DataType, float>(hnswParams, abstractInitParams, components);
#ifdef BUILD_TESTS
    index->quantType = hnswParams->quantType;
    if constexpr (std::is_same_v<StorageType, sq8>) {
        const auto mean = components.quantization->mean();
        index->serializedMeanVector.assign(mean.begin(), mean.end());
    }
#endif
    return index;
}

VecSimIndex *NewIndex(const VecSimParams *params, bool is_normalized, bool prepareForTraining) {
    const HNSWParams *hnswParams = &params->algoParams.hnswParams;
    AbstractIndexInitParams abstractInitParams =
        VecSimFactory::NewAbstractInitParams(hnswParams, params->logCtx, is_normalized);

    if (hnswParams->quantType != VecSimQuant_NONE) {
        const VecSimMetric metric = ResolveQuantizedMetric(hnswParams->metric, is_normalized);
        const QuantizationInit init{.parameters = hnswParams->quantParams,
                                    .prepareForTraining = prepareForTraining};

        if (!QuantizationParamsSupported(hnswParams->quantType, hnswParams->type, metric)) {
            return NULL;
        }

        if (hnswParams->quantType == VecSimQuant_TQ2)
            return NewIndex_Quantized<float, VecSimMetric_IP, vecsim_types::tq2>(
                hnswParams, abstractInitParams, init);
        if (hnswParams->quantType == VecSimQuant_TQ3)
            return NewIndex_Quantized<float, VecSimMetric_IP, vecsim_types::tq3>(
                hnswParams, abstractInitParams, init);
        if (hnswParams->type == VecSimType_FLOAT32) {
            if (metric == VecSimMetric_L2) {
                return NewIndex_Quantized<float, VecSimMetric_L2>(hnswParams, abstractInitParams,
                                                                  init);
            } else if (metric == VecSimMetric_IP) {
                return NewIndex_Quantized<float, VecSimMetric_IP>(hnswParams, abstractInitParams,
                                                                  init);
            }
        } else if (hnswParams->type == VecSimType_FLOAT16) {
            if (metric == VecSimMetric_L2) {
                return NewIndex_Quantized<float16, VecSimMetric_L2>(hnswParams, abstractInitParams,
                                                                    init);
            } else if (metric == VecSimMetric_IP) {
                return NewIndex_Quantized<float16, VecSimMetric_IP>(hnswParams, abstractInitParams,
                                                                    init);
            }
        }

        // Keep the return for release builds so future dispatch changes cannot fall through.
        assert(false && "unhandled quantized data type and metric combination");
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

VecSimIndex *NewIndex(const HNSWParams *params, bool is_normalized, bool prepareForTraining) {
    VecSimParams vecSimParams = {.algoParams = {.hnswParams = HNSWParams{*params}}};
    return NewIndex(&vecSimParams, is_normalized, prepareForTraining);
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
    return EstimateInitialSize(params, is_normalized,
                               /* with_mean = */ params->quantParams != nullptr);
}

size_t EstimateInitialSize(const HNSWParams *params, bool is_normalized, bool with_mean) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();

    size_t est = sizeof(VecSimAllocator) + allocations_overhead;

    if (params->quantType != VecSimQuant_NONE) {
        const auto metric = ResolveQuantizedMetric(params->metric, is_normalized);
        if (!QuantizationParamsSupported(params->quantType, params->type, metric))
            throw std::invalid_argument("Unsupported quantization params for HNSW index");
        const QuantizationInit init{.parameters = params->quantParams};
        if (params->quantType == VecSimQuant_TQ2)
            est += EstimateQuantizedComponentsMemory<float, VecSimMetric_IP, vecsim_types::tq2>(
                params->dim, init);
        else if (params->quantType == VecSimQuant_TQ3)
            est += EstimateQuantizedComponentsMemory<float, VecSimMetric_IP, vecsim_types::tq3>(
                params->dim, init);
        else if (with_mean)
            est += EstimateQuantizedComponentsMemory<float, VecSimMetric_L2, true, sq8>(params->dim,
                                                                                        init);
        else
            est += EstimateQuantizedComponentsMemory<float, VecSimMetric_L2, false, sq8>(
                params->dim, init);
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
    return EstimateElementSize(params, /* with_mean = */ params->quantParams != nullptr);
}

size_t EstimateElementSize(const HNSWParams *params, bool with_mean) {
    size_t M = (params->M) ? params->M : HNSW_DEFAULT_M;
    size_t elementGraphDataSize = sizeof(ElementGraphData) + sizeof(idType) * M * 2;

    size_t stored_data_size;
    // Preserve the element estimator's existing contract: return a size and validate in NewIndex.
    if (vecsim::quantization::isTurboQuant(params->quantType)) {
        if (params->type != VecSimType_FLOAT32 ||
            (params->metric != VecSimMetric_IP && params->metric != VecSimMetric_Cosine))
            return SIZE_MAX;
        try {
            stored_data_size = vecsim::quantization::turboQuantStorageBytes(
                params->dim, vecsim::quantization::turboQuantBits(params->quantType));
        } catch (const std::invalid_argument &) {
            return SIZE_MAX;
        }
    } else if (params->quantType == VecSimQuant_SQ8) {
        if (params->metric == VecSimMetric_L2) {
            stored_data_size = GetQuantizedStoredDataSize<VecSimMetric_L2>(params->dim, with_mean);
        } else {
            stored_data_size = GetQuantizedStoredDataSize<VecSimMetric_IP>(params->dim, with_mean);
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

    index->quantType = params->quantType;
    if (components.quantization && components.quantization->encodingVersion() == 5) {
        const auto mean = components.quantization->mean();
        index->serializedMeanVector.assign(mean.begin(), mean.end());
    }
    return index;
}

static void ValidateQuantizedGraph(std::ifstream &input, const HNSWParams &params,
                                   const std::shared_ptr<VecSimAllocator> &allocator,
                                   const QuantizationRuntime &runtime);

template <typename DataType, VecSimMetric Metric, typename StorageType = sq8>
VecSimIndex *LoadIndex_Quantized(std::ifstream &input, const HNSWParams &params,
                                 AbstractIndexInitParams abstractInitParams,
                                 HNSWSerializer::EncodingVersion version,
                                 const QuantizationInit &init) {
    abstractInitParams.isQuantized = true;
    auto components = CreateQuantizedIndexComponents<DataType, Metric, StorageType>(
        abstractInitParams.allocator, params.dim, init);
    abstractInitParams.storedDataSize = components.quantization->storageBytes();
    if (version >= HNSWSerializer::EncodingVersion::V6) {
        try {
            ValidateQuantizedGraph(input, params, abstractInitParams.allocator,
                                   *components.quantization);
        } catch (...) {
            delete components.preprocessors;
            delete components.indexCalculator;
            throw;
        }
    }
    return NewIndex_ChooseMultiOrSingle<DataType, float>(input, &params, abstractInitParams,
                                                         components, version);
}

// Initialize @params from file for V3+. For V5+ also reads quantType and mean vector.
// Returns false if the file holds a quantizer value this version does not know.
static bool InitializeParams(std::ifstream &source_params, HNSWParams &params,
                             HNSWSerializer::EncodingVersion version,
                             std::vector<float> &meanVector) {
    Serializer::readBinaryPOD(source_params, params.dim);
    Serializer::readBinaryPOD(source_params, params.type);
    Serializer::readBinaryPOD(source_params, params.metric);
    Serializer::readBinaryPOD(source_params, params.blockSize);
    Serializer::readBinaryPOD(source_params, params.multi);
    Serializer::readBinaryPOD(source_params, params.initialCapacity);

    // V5: read quantization fields
    if (version >= HNSWSerializer::EncodingVersion::V5) {
        // Validate the raw value before converting it: an out-of-range value is not a valid
        // VecSimQuantType, so it must never be loaded into the enum.
        std::underlying_type_t<VecSimQuantType> raw_quant_type;
        Serializer::readBinaryPOD(source_params, raw_quant_type);
        if (version == HNSWSerializer::EncodingVersion::V6) {
            if (!source_params || (raw_quant_type != VecSimQuant_TQ2 &&
                                   raw_quant_type != VecSimQuant_TQ3))
                throw std::runtime_error("Invalid V6 quantization type");
            params.quantType = static_cast<VecSimQuantType>(raw_quant_type);
            return true;
        }
        if (raw_quant_type != VecSimQuant_NONE && raw_quant_type != VecSimQuant_SQ8) {
            return false;
        }
        params.quantType = static_cast<VecSimQuantType>(raw_quant_type);
        if (params.quantType != VecSimQuant_NONE) {
            bool hasMean = false;
            Serializer::readBinaryPOD(source_params, hasMean);
            if (hasMean) {
                meanVector.resize(params.dim);
                source_params.read(reinterpret_cast<char *>(meanVector.data()),
                                   params.dim * sizeof(float));
            }
        }
    }
    return true;
}

// Validate a V6 graph before the legacy loader allocates nodes. That loader assumes a
// complete trusted stream; checking first also prevents partial graph destruction on failure.
static void ValidateQuantizedGraph(std::ifstream &input, const HNSWParams &params,
                                   const std::shared_ptr<VecSimAllocator> &allocator,
                                   const QuantizationRuntime &runtime) {
    const size_t record_bytes = runtime.storageBytes();
    const auto start = input.tellg();
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    input.seekg(start);
    auto read = [&](auto &value) {
        input.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!input) throw std::runtime_error("Truncated V6 HNSW graph");
    };
    size_t m, m0, efc, ef, count, deleted, max_level;
    double epsilon, mult;
    idType entrypoint;
    read(m); read(m0); read(efc); read(ef); read(epsilon); read(mult);
    read(count); read(deleted); read(max_level); read(entrypoint);
    if (m < 2 || m > UINT16_MAX / 2 || m0 != 2 * m || !efc || !ef ||
        !std::isfinite(epsilon) || epsilon <= 0 || !std::isfinite(mult) || mult <= 0 ||
        count >= INVALID_ID || deleted > count || !params.blockSize ||
        params.blockSize > UINT_MAX ||
        (count && (entrypoint >= count || max_level > 65535)) ||
        (!count && (entrypoint != INVALID_ID || max_level != HNSW_INVALID_LEVEL)))
        throw std::runtime_error("Invalid V6 HNSW graph parameters");
    const uint64_t row_bytes = sizeof(labelType) + sizeof(elementFlags) + record_bytes;
    const auto remaining = end - input.tellg();
    if (remaining < 0 || uint64_t(count) > uint64_t(remaining) / row_bytes)
        throw std::runtime_error("Invalid V6 HNSW vector count");
    size_t marked = 0;
    for (size_t i = 0; i < count; ++i) {
        labelType label;
        elementFlags flags;
        read(label); read(flags);
        if (flags & ~DELETE_MARK)
            throw std::runtime_error("Invalid V6 HNSW element flags");
        marked += (flags & DELETE_MARK) != 0;
    }
    if (marked != deleted) throw std::runtime_error("Inconsistent V6 HNSW deletion count");
    auto row = allocator->allocate_unique(record_bytes);
    for (size_t i = 0; i < count; ++i) {
        input.read(static_cast<char *>(row.get()), record_bytes);
        if (!input) throw std::runtime_error("Truncated V6 HNSW records");
        runtime.validateStoredRecord(row.get());
    }
    vecsim_stl::vector<size_t> node_levels(count, 0, allocator);
    vecsim_stl::vector<size_t> required_levels(count, 0, allocator);
    for (size_t block_start = 0; block_start < count; block_start += params.blockSize) {
        unsigned block_length = 0;
        read(block_length);
        if (block_length != std::min(params.blockSize, count - block_start))
            throw std::runtime_error("Invalid V6 HNSW graph block");
        for (size_t i = 0; i < block_length; ++i) {
            size_t level = 0;
            read(level);
            if (level > max_level) throw std::runtime_error("Invalid V6 HNSW node level");
            node_levels[block_start + i] = level;
            for (size_t l = 0; l <= level; ++l) {
                linkListSize links = 0;
                read(links);
                if (links > (l == 0 ? m0 : m))
                    throw std::runtime_error("Invalid V6 HNSW link count");
                for (size_t j = 0; j < links; ++j) {
                    idType neighbor; read(neighbor);
                    if (neighbor >= count) throw std::runtime_error("Invalid V6 HNSW neighbor");
                    required_levels[neighbor] = std::max(required_levels[neighbor], l);
                }
                unsigned incoming = 0;
                read(incoming);
                if (incoming > count) throw std::runtime_error("Invalid V6 HNSW incoming count");
                for (size_t j = 0; j < incoming; ++j) {
                    idType neighbor; read(neighbor);
                    if (neighbor >= count) throw std::runtime_error("Invalid V6 HNSW incoming edge");
                    required_levels[neighbor] = std::max(required_levels[neighbor], l);
                }
            }
        }
    }
    if (count && node_levels[entrypoint] != max_level)
        throw std::runtime_error("Inconsistent V6 HNSW entrypoint level");
    for (size_t i = 0; i < count; ++i)
        if (node_levels[i] < required_levels[i])
            throw std::runtime_error("V6 HNSW edge targets a missing level");
    if (input.tellg() != end) throw std::runtime_error("Trailing V6 HNSW graph bytes");
    input.seekg(start);
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

    HNSWParams params = {};
    std::vector<float> meanVector;
    if (!InitializeParams(input, params, version, meanVector)) {
        return NULL;
    }

    AbstractIndexInitParams abstractInitParams =
        VecSimFactory::NewAbstractInitParams(&params, nullptr, is_normalized);

    if (params.quantType != VecSimQuant_NONE) {
        const QuantizationInit init{
            .parameters = meanVector.empty() ? nullptr : meanVector.data(),
            .fittedState = version >= HNSWSerializer::EncodingVersion::V6 ? &input : nullptr};
        const VecSimMetric metric = ResolveQuantizedMetric(params.metric, is_normalized);

        if (!QuantizationParamsSupported(params.quantType, params.type, metric)) {
            return NULL;
        }

        if (params.quantType == VecSimQuant_TQ2)
            return LoadIndex_Quantized<float, VecSimMetric_IP, vecsim_types::tq2>(
                input, params, abstractInitParams, version, init);
        if (params.quantType == VecSimQuant_TQ3)
            return LoadIndex_Quantized<float, VecSimMetric_IP, vecsim_types::tq3>(
                input, params, abstractInitParams, version, init);
        if (params.type == VecSimType_FLOAT32) {
            if (metric == VecSimMetric_L2) {
                return LoadIndex_Quantized<float, VecSimMetric_L2>(
                    input, params, abstractInitParams, version, init);
            } else {
                return LoadIndex_Quantized<float, VecSimMetric_IP>(
                    input, params, abstractInitParams, version, init);
            }
        } else if (params.type == VecSimType_FLOAT16) {
            if (metric == VecSimMetric_L2) {
                return LoadIndex_Quantized<float16, VecSimMetric_L2>(
                    input, params, abstractInitParams, version, init);
            } else {
                return LoadIndex_Quantized<float16, VecSimMetric_IP>(
                    input, params, abstractInitParams, version, init);
            }
        }
    }

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

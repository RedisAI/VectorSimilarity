/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "VecSim/algorithms/hnsw/hnsw.h"

namespace HNSWFactory {

template <typename DataType, typename DistType = DataType>
inline HNSWIndex<DataType, DistType> *
NewIndex_ChooseMultiOrSingle(const HNSWParams *params,
                             const AbstractIndexInitParams &abstractInitParams) {
    // check if single and return new hnsw_index
    if (params->multi)
        return new (abstractInitParams.allocator)
            HNSWIndex_Multi<DataType, DistType>(params, abstractInitParams);
    else
        return new (abstractInitParams.allocator)
            HNSWIndex_Single<DataType, DistType>(params, abstractInitParams);
}

static AbstractIndexInitParams NewAbstractInitParams(const VecSimParams *params) {
    const HNSWParams *hnswParams = &params->hnswParams;
    AbstractIndexInitParams abstractInitParams = {.allocator =
                                                      VecSimAllocator::newVecsimAllocator(),
                                                  .dim = hnswParams->dim,
                                                  .vecType = hnswParams->type,
                                                  .metric = hnswParams->metric,
                                                  .blockSize = hnswParams->blockSize,
                                                  .multi = hnswParams->multi,
                                                  .logCtx = params->logCtx};
    return abstractInitParams;
}

VecSimIndex *NewIndex(const VecSimParams *params) {
    const HNSWParams *hnswParams = &params->hnswParams;
    AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(params);
    if (hnswParams->type == VecSimType_FLOAT32) {
        return NewIndex_ChooseMultiOrSingle<float>(hnswParams, abstractInitParams);
    } else if (hnswParams->type == VecSimType_FLOAT64) {
        return NewIndex_ChooseMultiOrSingle<double>(hnswParams, abstractInitParams);
    }

    // If we got here something is wrong.
    return NULL;
}

VecSimIndex *NewIndex(const HNSWParams *params) {
    VecSimParams vecSimParams = {.hnswParams = *params};
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

size_t EstimateInitialSize(const HNSWParams *params) {
    size_t blockSize = params->blockSize ? params->blockSize : DEFAULT_BLOCK_SIZE;
    size_t initial_cap =
        params->initialCapacity % blockSize == 0
            ? params->initialCapacity
            : params->initialCapacity + blockSize - params->initialCapacity % blockSize;

    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();

    size_t est = sizeof(VecSimAllocator) + allocations_overhead;
    if (params->type == VecSimType_FLOAT32) {
        est += EstimateInitialSize_ChooseMultiOrSingle<float>(params->multi);
    } else if (params->type == VecSimType_FLOAT64) {
        est += EstimateInitialSize_ChooseMultiOrSingle<double>(params->multi);
    }

    // Account for the visited nodes pool (assume that it holds one pointer to a handler).
    est += sizeof(VisitedNodesHandler) + allocations_overhead;
    // The visited nodes pool inner vector buffer (contains one pointer).
    est += sizeof(void *) + allocations_overhead;
    est += sizeof(tag_t) * initial_cap + allocations_overhead; // visited nodes array

    // Implicit allocation calls - allocates memory + a header only with positive capacity.
    if (initial_cap) {
        size_t num_blocks = initial_cap / blockSize; // should be divisible by block size
        est += sizeof(DataBlock) * num_blocks + allocations_overhead;          // data blocks
        est += sizeof(DataBlock) * num_blocks + allocations_overhead;          // meta blocks
        est += sizeof(element_meta_data) * initial_cap + allocations_overhead; // idToMetaData
        // Labels lookup hash table buckets.
        est += sizeof(size_t) * initial_cap + allocations_overhead;
        // lock per vector
        est += sizeof(std::mutex) * initial_cap + allocations_overhead;
    }

    return est;
}

size_t EstimateElementSize(const HNSWParams *params) {

    size_t M = (params->M) ? params->M : HNSW_DEFAULT_M;
    size_t element_graph_data_size_ = sizeof(element_graph_data) + sizeof(idType) * M * 2;

    size_t size_total_data_per_element =
        element_graph_data_size_ + params->dim * VecSimType_sizeof(params->type);

    size_t size_label_lookup_node;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    if (params->multi) {
        // For each new insertion (of a new label), we add a new node to the label_lookup_ map,
        // and a new element to the vector in the map. These two allocations both results in a new
        // allocation and therefore another VecSimAllocator::allocation_header_size.
        size_label_lookup_node =
            sizeof(vecsim_stl::unordered_map<labelType, vecsim_stl::vector<idType>>::value_type) +
            sizeof(size_t);
    } else {
        // For each new insertion (of a new label), we add a new node to the label_lookup_ map. This
        // results in a new allocation and therefore another VecSimAllocator::allocation_header_size
        size_label_lookup_node =
            sizeof(vecsim_stl::unordered_map<labelType, idType>::value_type) + sizeof(size_t);
    }

    // 1 entry in visited nodes + 1 entry in element metadata map + (approximately) 1 bucket in
    // labels lookup hash map.
    size_t size_meta_data =
        sizeof(tag_t) + sizeof(element_meta_data) + size_label_lookup_node;
    size_t size_lock = sizeof(std::mutex);

    /* Disclaimer: we are neglecting two additional factors that consume memory:
     * 1. The overall bucket size in labels_lookup hash table is usually higher than the number of
     * requested buckets (which is the index capacity), and it is auto selected according to the
     * hashing policy and the max load factor.
     * 2. The incoming edges that aren't bidirectional are stored in a dynamic array
     * (vecsim_stl::vector) Those edges' memory *is omitted completely* from this estimation.
     */
    return size_meta_data + size_total_data_per_element + size_lock;
}

#ifdef BUILD_TESTS

template <typename DataType, typename DistType = DataType>
inline VecSimIndex *NewIndex_ChooseMultiOrSingle(std::ifstream &input, const HNSWParams *params,
                                                 const AbstractIndexInitParams &abstractInitParams,
                                                 Serializer::EncodingVersion version) {
    HNSWIndex<DataType, DistType> *index = nullptr;
    // check if single and call the ctor that loads index information from file.
    if (params->multi)
        index = new (abstractInitParams.allocator)
            HNSWIndex_Multi<DataType, DistType>(input, params, abstractInitParams, version);
    else
        index = new (abstractInitParams.allocator)
            HNSWIndex_Single<DataType, DistType>(input, params, abstractInitParams, version);

    index->restoreGraph(input);

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

VecSimIndex *NewIndex(const std::string &location) {

    std::ifstream input(location, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    Serializer::EncodingVersion version = Serializer::ReadVersion(input);

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

    VecSimParams vecsimParams = {.algo = VecSimAlgo_HNSWLIB, .hnswParams = params};
    AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(&vecsimParams);
    if (params.type == VecSimType_FLOAT32) {
        return NewIndex_ChooseMultiOrSingle<float>(input, &params, abstractInitParams, version);
    } else if (params.type == VecSimType_FLOAT64) {
        return NewIndex_ChooseMultiOrSingle<double>(input, &params, abstractInitParams, version);
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

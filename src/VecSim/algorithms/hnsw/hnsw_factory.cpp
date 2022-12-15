/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "VecSim/algorithms/hnsw/hnsw.h"

namespace HNSWFactory {

template <typename DataType, typename DistType = DataType>
inline VecSimIndex *NewIndex_ChooseMultiOrSingle(const HNSWParams *params,
                                                 std::shared_ptr<VecSimAllocator> allocator) {
    // check if single and return new hnsw_index
    if (params->multi)
        return new (allocator) HNSWIndex_Multi<DataType, DistType>(params, allocator);
    else
        return new (allocator) HNSWIndex_Single<DataType, DistType>(params, allocator);
}

VecSimIndex *NewIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator) {
    if (params->type == VecSimType_FLOAT32) {
        return NewIndex_ChooseMultiOrSingle<float>(params, allocator);
    } else if (params->type == VecSimType_FLOAT64) {
        return NewIndex_ChooseMultiOrSingle<double>(params, allocator);
    }

    // If we got here something is wrong.
    return NULL;
}

template <typename DataType, typename DistType = DataType>
inline size_t EstimateInitialSize_ChooseMultiOrSingle(bool is_multi) {
    // check if single and return new bf_index
    if (is_multi)
        return sizeof(HNSWIndex_Multi<DataType, DistType>);
    else
        return sizeof(HNSWIndex_Single<DataType, DistType>);
}

size_t EstimateInitialSize(const HNSWParams *params) {
    size_t blockSize = (params->blockSize) ? params->blockSize : DEFAULT_BLOCK_SIZE;

    size_t est = sizeof(VecSimAllocator) + sizeof(size_t);
    if (params->type == VecSimType_FLOAT32) {
        est += EstimateInitialSize_ChooseMultiOrSingle<float>(params->multi);
    } else if (params->type == VecSimType_FLOAT64) {
        est += EstimateInitialSize_ChooseMultiOrSingle<double>(params->multi);
    }

#ifdef ENABLE_PARALLELIZATION
    // Used for synchronization only when parallel indexing / searching is enabled.
    est += sizeof(VisitedNodesHandlerPool) + sizeof(size_t);
#else
    est += sizeof(VisitedNodesHandler) + sizeof(size_t);
    est += sizeof(tag_t) * params->initialCapacity + sizeof(size_t); // visited nodes
#endif

    // Implicit allocation calls - allocates memory + a header only with positive capacity.
    if (params->initialCapacity) {
        size_t num_blocks = ceil((float)params->initialCapacity / (float)blockSize);
        est += sizeof(DataBlock) * num_blocks + sizeof(size_t); // data blocks
        est += sizeof(DataBlock) * num_blocks + sizeof(size_t); // meta blocks
        est += (sizeof(labelType) + sizeof(elementFlags)) * params->initialCapacity +
               sizeof(size_t); // idToMetaData
        est += sizeof(size_t) * params->initialCapacity +
               sizeof(size_t); // Labels lookup hash table buckets.
    }

    return est;
}

size_t EstimateElementSize(const HNSWParams *params) {
    size_t M = (params->M) ? params->M : HNSW_DEFAULT_M;
    size_t element_graph_data_size_ = sizeof(element_graph_data) + sizeof(idType) * M * 2;

    size_t size_total_data_per_element =
        element_graph_data_size_ + params->dim * VecSimType_sizeof(params->type);

    size_t size_label_lookup_node;
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
        sizeof(tag_t) + (sizeof(labelType) + sizeof(elementFlags)) + size_label_lookup_node;

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
                                                 std::shared_ptr<VecSimAllocator> allocator,
                                                 Serializer::EncodingVersion version) {
    HNSWIndex<DataType, DistType> *index = nullptr;
    // check if single and call the ctor that loads index information from file.
    if (params->multi)
        index =
            new (allocator) HNSWIndex_Multi<DataType, DistType>(input, params, allocator, version);
    else
        index =
            new (allocator) HNSWIndex_Single<DataType, DistType>(input, params, allocator, version);

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

    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    if (params.type == VecSimType_FLOAT32) {
        return NewIndex_ChooseMultiOrSingle<float>(input, &params, allocator, version);
    } else if (params.type == VecSimType_FLOAT64) {
        return NewIndex_ChooseMultiOrSingle<double>(input, &params, allocator, version);
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

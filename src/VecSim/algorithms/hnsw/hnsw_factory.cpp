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
    size_t M = (params->M) ? params->M : HNSW_DEFAULT_M;

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
        est += sizeof(size_t) * params->initialCapacity + sizeof(size_t); // element level
        est += sizeof(size_t) * params->initialCapacity +
               sizeof(size_t); // Labels lookup hash table buckets.
    }

    // Explicit allocation calls - always allocate a header.
    est += sizeof(void *) * params->initialCapacity + sizeof(size_t); // link lists (for levels > 0)

    size_t size_links_level0 = sizeof(linklistsizeint) + M * 2 * sizeof(idType) + sizeof(void *);
    size_t size_total_data_per_element =
        size_links_level0 + params->dim * VecSimType_sizeof(params->type) + sizeof(labelType);
    est += params->initialCapacity * size_total_data_per_element + sizeof(size_t);

    return est;
}

size_t EstimateElementSize(const HNSWParams *params) {
    size_t M = (params->M) ? params->M : HNSW_DEFAULT_M;
    size_t size_links_level0 = sizeof(linklistsizeint) + M * 2 * sizeof(idType) + sizeof(void *) +
                               sizeof(vecsim_stl::vector<idType>);
    size_t size_links_higher_level = sizeof(linklistsizeint) + M * sizeof(idType) + sizeof(void *) +
                                     sizeof(vecsim_stl::vector<idType>);
    // The Expectancy for the random variable which is the number of levels per element equals
    // 1/ln(M). Since the max_level is rounded to the "floor" integer, the actual average number
    // of levels is lower (intuitively, we "loose" a level every time the random generated number
    // should have been rounded up to the larger integer). So, we "fix" the expectancy and take
    // 1/2*ln(M) instead as an approximation.
    size_t expected_size_links_higher_levels =
        ceil((1 / (2 * log(M))) * (float)size_links_higher_level);

    size_t size_total_data_per_element = size_links_level0 + expected_size_links_higher_levels +
                                         params->dim * VecSimType_sizeof(params->type) +
                                         sizeof(labelType);

    size_t size_label_lookup_node;
    if (params->multi) {
        // For each new insertion (of a new label), we add a new node to the label_lookup_ map,
        // and a new element to the vector in the map. These two allocations both results in a new
        // allocation and therefore another VecSimAllocator::allocation_header_size.
        size_label_lookup_node =
            sizeof(vecsim_stl::unordered_map<labelType, vecsim_stl::vector<idType>>::value_type) +
            sizeof(size_t) + sizeof(vecsim_stl::vector<idType>::value_type) + sizeof(size_t);
    } else {
        // For each new insertion (of a new label), we add a new node to the label_lookup_ map. This
        // results in a new allocation and therefore another VecSimAllocator::allocation_header_size
        // plus an internal pointer
        size_label_lookup_node = sizeof(vecsim_stl::unordered_map<labelType, idType>::value_type) +
                                 sizeof(size_t) + sizeof(size_t);
    }

    // 1 entry in visited nodes + 1 entry in element levels + (approximately) 1 bucket in labels
    // lookup hash map.
    size_t size_meta_data =
        sizeof(tag_t) + sizeof(size_t) + sizeof(size_t) + size_label_lookup_node;

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
                                                 std::shared_ptr<VecSimAllocator> allocator) {
    VecSimIndex *index = nullptr;
    // check if single and call the ctor that loads index information from file.
    if (params->multi)
        index = new (allocator) HNSWIndex_Multi<DataType, DistType>(input, params, allocator);
    else
        index = new (allocator) HNSWIndex_Single<DataType, DistType>(input, params, allocator);

    reinterpret_cast<HNSWIndex<DataType, DistType> *>(index)->restoreGraph(input);

    return index;
}

void InitializeParams(std::ifstream &input, HNSWParams &params) {
    Serializer::readBinaryPOD(input, params.dim);
    Serializer::readBinaryPOD(input, params.type);
    Serializer::readBinaryPOD(input, params.metric);
    Serializer::readBinaryPOD(input, params.blockSize);
    Serializer::readBinaryPOD(input, params.multi);
    Serializer::readBinaryPOD(input, params.epsilon);
}

VecSimIndex *NewIndex(const std::string &location, VecSimType type, bool is_multi) {

    std::ifstream input(location, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    Serializer::EncodingVersion version = Serializer::ReadVersion(input);

    HNSWParams params;
    switch (version) {
    case Serializer::EncodingVersion_V2: {
        // Algorithm type is only serialized from V2 up.
        VecSimAlgo algo = VecSimAlgo_INVALID;
        Serializer::readBinaryPOD(input, algo);
        if (algo != VecSimAlgo_HNSWLIB) {
            input.close();
            throw std::runtime_error("Cannot load index: bad algorithm type");
        }
        // this information is serialized from V2 and up
        InitializeParams(input, params);
        break;
    }
    case Serializer::EncodingVersion_V1: {
        params.type = type;
        params.dim = 4;
        params.metric = VecSimMetric_L2;
        params.multi = is_multi;
        params.blockSize = 1;
        params.epsilon = 0.004;
        break;
    }
    // Something is wrong
    default:
        throw std::runtime_error("Cannot load index: bad encoding version");
    }
    Serializer::readBinaryPOD(input, params.initialCapacity);

    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    if (params.type == VecSimType_FLOAT32) {
        return NewIndex_ChooseMultiOrSingle<float>(input, &params, allocator);
    } else if (params.type == VecSimType_FLOAT64) {
        return NewIndex_ChooseMultiOrSingle<double>(input, &params, allocator);
    } else {
        throw std::runtime_error("Cannot load index: bad index type");
    }
}
#endif

}; // namespace HNSWFactory

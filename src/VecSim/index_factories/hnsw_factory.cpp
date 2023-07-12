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
    const HNSWParams *hnswParams = &params->algoParams.hnswParams;
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
    const HNSWParams *hnswParams = &params->algoParams.hnswParams;
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

size_t EstimateInitialSize(const HNSWParams *params) {
    size_t M = (params->M) ? params->M : HNSW_DEFAULT_M;

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
    est += sizeof(tag_t) * params->initialCapacity + allocations_overhead; // visited nodes array

    // Implicit allocation calls - allocates memory + a header only with positive capacity.
    if (params->initialCapacity) {
        est += sizeof(size_t) * params->initialCapacity + allocations_overhead; // element level
        est += sizeof(size_t) * params->initialCapacity +
               allocations_overhead; // Labels lookup hash table buckets.
        est +=
            sizeof(std::mutex) * params->initialCapacity + allocations_overhead; // lock per vector
    }

    // Explicit allocation calls - always allocate a header.
    est += sizeof(void *) * params->initialCapacity +
           allocations_overhead; // link lists (for levels > 0)

    size_t size_links_level0 =
        sizeof(elementFlags) + sizeof(linkListSize) + M * 2 * sizeof(idType) + sizeof(void *);
    size_t size_total_data_per_element =
        size_links_level0 + params->dim * VecSimType_sizeof(params->type) + sizeof(labelType);
    est += params->initialCapacity * size_total_data_per_element + allocations_overhead;

    return est;
}

size_t EstimateElementSize(const HNSWParams *params) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();

    size_t M = (params->M) ? params->M : HNSW_DEFAULT_M;
    size_t size_links_level0 = sizeof(linkListSize) + M * 2 * sizeof(idType) + sizeof(void *) +
                               sizeof(vecsim_stl::vector<idType>) + allocations_overhead;
    size_t size_links_higher_level = sizeof(linkListSize) + M * sizeof(idType) + sizeof(void *) +
                                     sizeof(vecsim_stl::vector<idType>) + allocations_overhead;
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
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    if (params->multi) {
        auto dummy_lookup =
            vecsim_stl::unordered_map<labelType, vecsim_stl::vector<idType>>(1, allocator);
        size_t memory_before = allocator->getAllocationSize();
        // For each new insertion (of a new label), we add a new node to the label_lookup_ map.
        dummy_lookup.emplace(0, vecsim_stl::vector<idType>{allocator});
        // In addition, a new element to the vector in the map.
        dummy_lookup.at(0).push_back(0);
        size_t memory_after = allocator->getAllocationSize();
        // size_t memory_before = allocator->getAllocationSize();

        size_label_lookup_node = memory_after - memory_before;
    } else {
        auto dummy_lookup = vecsim_stl::unordered_map<size_t, unsigned int>(1, allocator);
        size_t memory_before = allocator->getAllocationSize();
        // For each new insertion (of a new label), we add a new node to the label_lookup_ map.
        dummy_lookup.insert({1, 1}); // Insert a dummy {key, value} element pair.
        size_t memory_after = allocator->getAllocationSize();
        size_label_lookup_node = memory_after - memory_before;
    }

    // 1 entry in visited nodes + 1 entry in element levels + (approximately) 1 bucket in labels
    // lookup hash map.
    size_t size_meta_data =
        sizeof(tag_t) + sizeof(size_t) + sizeof(size_t) + size_label_lookup_node;
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

// Intialize @params from file for V2
static void InitializeParams(std::ifstream &source_params, HNSWParams &params) {
    Serializer::readBinaryPOD(source_params, params.dim);
    Serializer::readBinaryPOD(source_params, params.type);
    Serializer::readBinaryPOD(source_params, params.metric);
    Serializer::readBinaryPOD(source_params, params.blockSize);
    Serializer::readBinaryPOD(source_params, params.multi);
    Serializer::readBinaryPOD(source_params, params.epsilon);
}

// Intialize @params for V1
static void InitializeParams(const HNSWParams *source_params, HNSWParams &params) {
    params.type = source_params->type;
    params.dim = source_params->dim;
    params.metric = source_params->metric;
    params.multi = source_params->multi;
    params.blockSize = source_params->blockSize ? source_params->blockSize : DEFAULT_BLOCK_SIZE;
    params.epsilon = source_params->epsilon ? source_params->epsilon : HNSW_DEFAULT_EPSILON;
}

VecSimIndex *NewIndex(const std::string &location, const HNSWParams *v1_params) {

    std::ifstream input(location, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    Serializer::EncodingVersion version = Serializer::ReadVersion(input);
    HNSWParams params;
    switch (version) {
    case Serializer::EncodingVersion_V2: {
        // Algorithm type is only serialized from V2 up.
        VecSimAlgo algo = VecSimAlgo_BF;
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
        assert(v1_params);
        InitializeParams(v1_params, params);
        break;
    }
    // Something is wrong
    default:
        throw std::runtime_error("Cannot load index: bad encoding version");
    }
    Serializer::readBinaryPOD(input, params.initialCapacity);

    VecSimParams vecsimParams = {.algo = VecSimAlgo_HNSWLIB,
                                 .algoParams = {.hnswParams = HNSWParams{params}}};
    AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(&vecsimParams);
    if (params.type == VecSimType_FLOAT32) {
        return NewIndex_ChooseMultiOrSingle<float>(input, &params, abstractInitParams, version);
    } else if (params.type == VecSimType_FLOAT64) {
        return NewIndex_ChooseMultiOrSingle<double>(input, &params, abstractInitParams, version);
    } else {
        throw std::runtime_error("Cannot load index: bad index data type");
    }
}
#endif

}; // namespace HNSWFactory

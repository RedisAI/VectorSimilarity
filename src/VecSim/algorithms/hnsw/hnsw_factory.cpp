#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "VecSim/algorithms/hnsw/hnsw.h" //linklistsizeint
#include "VecSim/vec_sim_common.h"       // labelType
#include "VecSim/algorithms/hnsw/hnsw_batch_iterator.h"

namespace HNSWFactory {

VecSimIndex *NewIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator) {
    // check if single and return new bf_index
    assert(!params->multi);
    return new (allocator) HNSWIndex<float, float>(params, allocator);
}

size_t EstimateInitialSize(const HNSWParams *params) {

    size_t est = sizeof(VecSimAllocator) + sizeof(size_t);
    est += sizeof(HNSWIndex<float, float>);
    // Used for synchronization only when parallel indexing / searching is enabled.
#ifdef ENABLE_PARALLELIZATION
    est += sizeof(VisitedNodesHandlerPool) + sizeof(size_t);
#else
    est += sizeof(VisitedNodesHandler) + sizeof(size_t);
#endif
    est += sizeof(tag_t) * params->initialCapacity + sizeof(size_t); // visited nodes

    est += sizeof(void *) * params->initialCapacity + sizeof(size_t); // link lists (for levels > 0)
    est += sizeof(size_t) * params->initialCapacity + sizeof(size_t); // element level
    est += sizeof(size_t) * params->initialCapacity +
           sizeof(size_t); // Labels lookup hash table buckets.

    size_t size_links_level0 =
        sizeof(linklistsizeint) + params->M * 2 * sizeof(idType) + sizeof(void *);
    size_t size_total_data_per_element =
        size_links_level0 + params->dim * VecSimType_sizeof(params->type) + sizeof(labelType);
    est += params->initialCapacity * size_total_data_per_element + sizeof(size_t);

    return est;
}

size_t EstimateElementSize(const HNSWParams *params) {
    size_t size_links_level0 = sizeof(linklistsizeint) + params->M * 2 * sizeof(idType) +
                               sizeof(void *) + sizeof(vecsim_stl::vector<idType>);
    size_t size_links_higher_level = sizeof(linklistsizeint) + params->M * sizeof(idType) +
                                     sizeof(void *) + sizeof(vecsim_stl::vector<idType>);
    // The Expectancy for the random variable which is the number of levels per element equals
    // 1/ln(M). Since the max_level is rounded to the "floor" integer, the actual average number
    // of levels is lower (intuitively, we "loose" a level every time the random generated number
    // should have been rounded up to the larger integer). So, we "fix" the expectancy and take
    // 1/2*ln(M) instead as an approximation.
    size_t expected_size_links_higher_levels =
        ceil((1 / (2 * log(params->M))) * (float)size_links_higher_level);

    size_t size_total_data_per_element = size_links_level0 + expected_size_links_higher_levels +
                                         params->dim * VecSimType_sizeof(params->type) +
                                         sizeof(labelType);

    // For every new vector, a new node of size 24 is allocated in a bucket of the hash table.
    size_t size_label_lookup_node =
        24 + sizeof(size_t); // 24 + VecSimAllocator::allocation_header_size
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

// TODO overload for doubles
VecSimBatchIterator *newBatchIterator(void *queryBlob, VecSimQueryParams *queryParams,
                                      std::shared_ptr<VecSimAllocator> allocator,
                                      HNSWIndex<float, float> *index) {

    return new (allocator)
        HNSW_BatchIterator<float, float>(queryBlob, index, queryParams, allocator);
}
}; // namespace HNSWFactory

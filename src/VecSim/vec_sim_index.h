#pragma once
#include "vec_sim_common.h"
#include "query_results.h"
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "info_iterator_struct.h"

/**
 * @brief Abstract C++ class for vector index, delete and lookup
 *
 */
struct VecSimIndex : public VecsimBaseObject {
public:
    /**
     * @brief Construct a new Vec Sim Index object
     *
     */
    VecSimIndex(std::shared_ptr<VecSimAllocator> allocator) : VecsimBaseObject(allocator) {}

    /**
     * @brief Destroy the Vec Sim Index object
     *
     */
    virtual ~VecSimIndex() {}

    /**
     * @brief Add a vector blob and its id to the index.
     *
     * @param blob binary representation of the vector. Blob size should match the index data type
     * and dimension.
     * @param id the id of the added vector
     * @return always returns true
     */
    virtual int addVector(const void *blob, size_t id) = 0;

    /**
     * @brief Remove a vector from an index.
     *
     * @param id the id of the removed vector
     * @return always returns true
     */
    virtual int deleteVector(size_t id) = 0;

    /**
     * @brief Return the number of vectors in the index using irs SizeFn.
     *
     * @return index size.
     */
    virtual size_t indexSize() const = 0;

    /**
     * @brief Search for the k closest vectors to a given vector in the index.
     *
     * @param queryBlob binary representation of the query vector. Blob size should match the index
     * data type and dimension.
     * @param k the number of "nearest neighbours" to return (upper bound).
     * @param queryParams run time params for the search, which are algorithm-specific.
     * @return An opaque object the represents a list of results. User can access the id and score
     * (which is the distance according to the index metric) of every result through
     * VecSimQueryResult_Iterator.
     */
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) = 0;

    /**
     * @brief Return index information.
     *
     * @return Index general and specific meta-data.
     */
    virtual VecSimIndexInfo info() = 0;

    /**
     * @brief Returns an index information in an iterable structure.
     *
     * @return VecSimInfoIterator Index general and specific meta-data.
     */
    virtual VecSimInfoIterator *infoIterator() = 0;

    /**
     * @brief Create a new batch iterator for a specific index, for a specific query vector,
     * using the Index_BatchIteratorNew method of the index. Should be released with
     * VecSimBatchIterator_Free call.
     *
     * @param queryBlob binary representation of the vector. Blob size should match the index data
     * type and dimension.
     * @return Fresh batch iterator
     */
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob) = 0;
};

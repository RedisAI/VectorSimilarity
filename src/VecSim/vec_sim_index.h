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
     * @brief Calculate the distance of a vector from an index to a vector.
     * @param index the index from which the first vector is located, and that defines the distance
     * metric.
     * @param id the id of the vector in the index.
     * @param blob binary representation of the second vector. Blob size should match the index data
     * type and dimension, and pre-normalized if needed.
     * @return The distance (according to the index's distance metric) between `blob` and the vector
     * with id `id`.
     */
    virtual double getDistanceFrom(size_t id, const void *blob) = 0;

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
    virtual VecSimIndexInfo info() const = 0;

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
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) = 0;

    /**
     * @brief Return True if heuristics says that it is better to use ad-hoc brute-force
     * search over the index instead of using batch iterator.
     *
     * @param subsetSize the estimated number of vectors in the index that pass the filter
     * (that is, query results can be only from a subset of vector of this size).
     *
     * @param k the number of required results to return from the query.
     *
     * @param initial_check flag to indicate if this check is performed for the first time (upon
     * creating the hybrid iterator), or after running batches.
     */

    virtual bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) = 0;

    /**
     * @brief Set the latest search mode in the index data (for info/debugging).
     * @param mode The search mode.
     */
    virtual inline void setLastSearchMode(VecSearchMode mode) = 0;
};

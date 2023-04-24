/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "vec_sim_common.h"
#include "vec_sim.h"
#include "query_results.h"
#include "VecSim/memory/vecsim_base.h"
#include "info_iterator_struct.h"

#include <stddef.h>
#include <stdexcept>
#include <cstdarg>
/**
 * @brief Abstract C++ class for vector index, delete and lookup
 *
 */
struct VecSimIndexInterface : public VecsimBaseObject {

public:
    /**
     * @brief Construct a new Vec Sim Index object
     *
     */
    VecSimIndexInterface(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}

    /**
     * @brief Destroy the Vec Sim Index object
     *
     */
    virtual ~VecSimIndexInterface() {}

    /**
     * @brief Add a vector blob and its id to the index.
     *
     * @param blob binary representation of the vector. Blob size should match the index data type
     * and dimension.
     * @param label the label of the added vector.
     * @param auxiliaryCtx if this is not the main index (but a layer in a tiered index for example)
     * we pass a state of the index to be used internally. Otherwise, if auxiliaryCtx just perform
     * a "vanilla" insertion of a new vector.
     * In addition, if id is not given, and this label already exists overwrite it. Otherwise,
     * it's the caller main index responsibility to validate that the new label and id are
     * appropriate.
     * @return the number of new vectors inserted (1 for new insertion, 0 for override).
     */
    virtual int addVector(const void *blob, labelType label, void *auxiliaryCtx = nullptr);

    /**
     * @brief A function to be implemented by the inheriting index and called by addVector.
     * @param blob is a processed vector (for example, if the distance metric is cosine,
     * blob is already *normalized*)
     * It is assumed the the index saves its own copy of the blob.
     */
    virtual int addVectorImp(const void *blob, labelType label, void *auxiliaryCtx = nullptr) = 0;

    /**
     * @brief Remove a vector from an index.
     *
     * @param label the label of the vector to remove
     * @return the number of vectors deleted
     */
    virtual int deleteVector(labelType label) = 0;

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
    virtual double getDistanceFrom(labelType id, const void *blob) const = 0;

    /**
     * @brief Return the number of vectors in the index (including ones that are marked as deleted).
     *
     * @return index size.
     */
    virtual size_t indexSize() const = 0;

    /**
     * @brief Return the index capacity, so we know if resize is required for adding new vectors.
     *
     * @return index capacity.
     */
    virtual size_t indexCapacity() const = 0;

    /**
     * @brief Change the index capacity (without changing its data), by adding another block.
     */
    virtual void increaseCapacity() = 0;

    /**
     * @brief Return the number of unique labels in the index (which are not deleted).
     *
     * @return index label count.
     */
    virtual size_t indexLabelCount() const = 0;

    /**
     * @brief Search for the k closest vectors to a given vector in the index.
     *
     * @param queryBlob binary representation of the query vector. Blob size should match the index
     * data type and dimension.
     * @param k the number of "nearest neighbors" to return (upper bound).
     * @param queryParams run time params for the search, which are algorithm-specific.
     * @return An opaque object the represents a list of results. User can access the id and score
     * (which is the distance according to the index metric) of every result through
     * VecSimQueryResult_Iterator.
     */
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams);

    /**
     * @brief A function to be implemented by the inheriting index and called by topKQuery.
     * @param blob is a processed vector (for example, if the distance metric is cosine,
     * blob is already *normalized* )
     */
    virtual VecSimQueryResult_List topKQueryImp(const void *queryBlob, size_t k,
                                                VecSimQueryParams *queryParams) = 0;
    /**
     * @brief Search for the vectors that are in a given range in the index with respect to a given
     * vector. The results can be ordered by their score or id.
     * @param queryBlob binary representation of the query vector. Blob size should match the index
     * data type and dimension.
     * @param radius the radius around the query vector to search vectors within it.
     * @param queryParams run time params for the search, which are algorithm-specific.
     * @param order the criterion to sort the results list by it. Options are by score, or by id.
     * @return An opaque object the represents a list of results. User can access the id and score
     * (which is the distance according to the index metric) of every result through
     * VecSimQueryResult_Iterator.
     */
    virtual VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius,
                                              VecSimQueryParams *queryParams);
    /**
     * @brief A function to be implemented by the inheriting index and called by rangeQuery.
     * @param blob is a processed vector (for example, if the distance metric is cosine,
     * blob is already *normalized* )
     */
    virtual VecSimQueryResult_List rangeQueryImp(const void *queryBlob, double radius,
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
    virtual VecSimInfoIterator *infoIterator() const = 0;

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
                                                  VecSimQueryParams *queryParams) const = 0;

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

    /**
     * @brief Allow 3rd party timeout callback to be used for limiting runtime of a query.
     *
     * @param callback timeoutCallbackFunction function. should get void* and return int.
     */
    static timeoutCallbackFunction timeoutCallback;
    inline static void setTimeoutCallbackFunction(timeoutCallbackFunction callback) {
        VecSimIndexInterface::timeoutCallback = callback;
    }

    static logCallbackFunction logCallback;
    inline static void setLogCallbackFunction(logCallbackFunction callback) {
        VecSimIndexInterface::logCallback = callback;
    }

    /**
     * @brief This functions gets a blob, and if required, *allocates its own copy* of it,
     * and performs operations before sending it to the index
     * Returns the blob to be used by the index.
     * NOTE: It is the caller responsibility to call returnProcessedBlob() after the the processed
     * vector is no longer needed to free resources.
     */
    virtual const void *processBlob(const void *blob) = 0;

    /**
     * @brief A function to free resources used to generate the processed blob.
     */
    virtual void returnProcessedBlob(const void *processed_blob) = 0;
};

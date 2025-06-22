/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include "vec_sim_common.h"
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
    virtual ~VecSimIndexInterface() = default;

    /**
     * @brief Add a vector blob and its id to the index.
     *
     * @param blob binary representation of the vector. Blob size should match the index data type
     * and dimension. The blob will be copied and processed by the index.
     * @param label the label of the added vector.
     * @return the number of new vectors inserted (1 for new insertion, 0 for override).
     */
    virtual int addVector(const void *blob, labelType label) = 0;

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
    virtual double getDistanceFrom_Unsafe(labelType id, const void *blob) const = 0;

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
     * @brief Return the number of unique labels in the index (which are not deleted).
     * !!! Note: this should only be called in debug mode, as in tiered indexes it may require
     * locking the indexes and going over the labels sets, which is time-consuming. !!!
     *
     * @return index label count.
     */
    virtual size_t indexLabelCount() const = 0;

    /**
     * @brief Search for the k closest vectors to a given vector in the index.
     * @param queryBlob binary representation of the query vector. Blob size should match the index
     * data type and dimension. The index is responsible to process the query vector.
     * @param k the number of "nearest neighbors" to return (upper bound).
     * @param queryParams run time params for the search, which are algorithm-specific.
     * @return An opaque object the represents a list of results. User can access the id and score
     * (which is the distance according to the index metric) of every result through
     * VecSimQueryReply_Iterator.
     */
    virtual VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                        VecSimQueryParams *queryParams) const = 0;

    /**
     * @brief Search for the vectors that are in a given range in the index with respect to a given
     * vector. The results can be ordered by their score or id.
     * @param queryBlob binary representation of the query vector. Blob size should match the index
     * data type and dimension. The index is responsible to process the query vector.
     * @param radius the radius around the query vector to search vectors within it.
     * @param queryParams run time params for the search, which are algorithm-specific.
     * @param order the criterion to sort the results list by it. Options are by score, or by id.
     * @return An opaque object the represents a list of results. User can access the id and score
     * (which is the distance according to the index metric) of every result through
     * VecSimQueryReply_Iterator.
     */
    virtual VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                         VecSimQueryParams *queryParams,
                                         VecSimQueryReply_Order order) const = 0;

    /**
     * @brief Return index information.
     *
     * @return Index general and specific meta-data. Note that this operation might
     * be time consuming (specially for tiered index where computing label count required
     * locking and going over the labels sets). So this should be used carefully.
     */
    virtual VecSimIndexDebugInfo debugInfo() const = 0;

    /**
     * @brief Return index static information.
     *
     * @return Index general and specific meta-data (for quick and lock-less data retrieval)
     */
    virtual VecSimIndexBasicInfo basicInfo() const = 0;

    /**
     * @brief Return index statistic information.
     *
     * @return Index general and specific statistic data (for quick and lock-less retrieval)
     */
    virtual VecSimIndexStatsInfo statisticInfo() const = 0;

    /**
     * @brief Returns an index information in an iterable structure.
     *
     * @return VecSimDebugInfoIterator Index general and specific meta-data.
     */
    virtual VecSimDebugInfoIterator *debugInfoIterator() const = 0;

    /**
     * @brief A function to be implemented by the inheriting index and called by rangeQuery.
     * @param queryBlob binary representation of the query vector. Blob size should match the index
     * data type and dimension. The index is responsible to process the query vector.
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

    virtual bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const = 0;

    /**
     * @brief Set the latest search mode in the index data (for info/debugging).
     * @param mode The search mode.
     */
    virtual inline void setLastSearchMode(VecSearchMode mode) = 0;

    /**
     * @brief Run async garbage collection for tiered async index.
     */
    virtual void runGC() = 0;

    /**
     * @brief Acquire the locks for shared ownership in tiered async index.
     */
    virtual void acquireSharedLocks() = 0;

    /**
     * @brief Release the locks for shared ownership in tiered async index.
     */
    virtual void releaseSharedLocks() = 0;

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
     * @brief Allow 3rd party to set the write mode for tiered index - async insert/delete using
     * background jobs, or insert/delete inplace.
     *
     * @param mode VecSimWriteMode the mode in which we add/remove vectors (async or in-place).
     */
    static VecSimWriteMode asyncWriteMode;
    inline static void setWriteMode(VecSimWriteMode mode) {
        VecSimIndexInterface::asyncWriteMode = mode;
    }
#ifdef BUILD_TESTS
    virtual void fitMemory() = 0;
#endif
};

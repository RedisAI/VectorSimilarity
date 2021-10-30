#pragma once
#include "vec_sim_common.h"
#include "query_results.h"
#include <stddef.h>

/**
 * @brief Abstract C++ class for vector index, delete and lookup
 *
 */
class VecSimIndex {
protected:
    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;

public:
    /**
     * @brief Construct a new Vec Sim Index object
     *
     * @param params VecSimParams struct, the base object takes the vector dimensions, type and
     * distance metric.
     */
    VecSimIndex(const VecSimParams *params)
        : dim(params->size), vecType(params->type), metric(params->metric) {}

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
    virtual size_t indexSize() = 0;

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
     * @brief Create a new batch iterator for a specific index, for a specific query vector,
     * using the Index_BatchIteratorNew method of the index. Should be released with
     * VecSimBatchIterator_Free call.
     *
     * @param queryBlob binary representation of the vector. Blob size should match the index data
     * type and dimension.
     * @return Fresh batch iterator
     */
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob) = 0;

    /**
     * @brief Get the vector dimension
     *
     * @return vector dimension
     */
    inline size_t getVectorDim() { return this->dim; }

    /**
     * @brief Get the vector metric
     *
     * @return Index metric
     */
    inline VecSimMetric getMetric() { return this->metric; }

    /**
     * @brief Get the vector type
     *
     * @return vector type
     */
    inline VecSimType getVectorType() { return this->vecType; }
};

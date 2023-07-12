/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "vec_sim_common.h"
#include "vec_sim_interface.h"
#include "query_results.h"
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/utils/alignment.h"
#include "VecSim/spaces/spaces.h"
#include "info_iterator_struct.h"
#include <cassert>

using spaces::dist_func_t;

/**
 * @brief Struct for initializing an abstract index class.
 *
 * @param allocator The allocator to use for the index.
 * @param dim The dimension of the vectors in the index.
 * @param vecType The type of the vectors in the index.
 * @param metric The metric to use in the index.
 * @param blockSize The block size to use in the index.
 * @param multi Determines if the index should multi-index or not.
 * @param logCtx The context to use for logging.
 */
struct AbstractIndexInitParams {
    std::shared_ptr<VecSimAllocator> allocator;
    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;
    size_t blockSize;
    bool multi;
    void *logCtx;
};

/**
 * @brief Abstract C++ class for vector index, delete and lookup
 *
 */
template <typename DistType>
struct VecSimIndexAbstract : public VecSimIndexInterface {
protected:
    size_t dim;          // Vector's dimension.
    VecSimType vecType;  // Datatype to index.
    size_t dataSize;     // Vector size in bytes
    VecSimMetric metric; // Distance metric to use in the index.
    size_t blockSize;    // Index's vector block size (determines by how many vectors to resize when
                         // resizing)
    dist_func_t<DistType>
        distFunc;            // Index's distance function. Chosen by the type, metric and dimension.
    unsigned char alignment; // Alignment hint to allocate vectors with.
    mutable VecSearchMode lastMode; // The last search mode in RediSearch (used for debug/testing).
    bool isMulti;                   // Determines if the index should multi-index or not.
    void *logCallbackCtx;           // Context for the log callback.

    /**
     * @brief Get the common info object
     *
     * @return CommonInfo
     */
    CommonInfo getCommonInfo() const {
        CommonInfo info;
        info.basicInfo = this->getBasicInfo();
        info.lastMode = this->lastMode;
        info.memory = this->getAllocationSize();
        info.indexSize = this->indexSize();
        info.indexLabelCount = this->indexLabelCount();
        return info;
    }

    normalizeVector_f normalize_func; // A pointer to a normalization function of specific type.

public:
    /**
     * @brief Construct a new Vec Sim Index object
     *
     */
    VecSimIndexAbstract(const AbstractIndexInitParams &params)
        : VecSimIndexInterface(params.allocator), dim(params.dim), vecType(params.vecType),
          dataSize(dim * VecSimType_sizeof(vecType)), metric(params.metric),
          blockSize(params.blockSize ? params.blockSize : DEFAULT_BLOCK_SIZE), alignment(0),
          lastMode(EMPTY_MODE), isMulti(params.multi), logCallbackCtx(params.logCtx) {
        assert(VecSimType_sizeof(vecType));
        spaces::SetDistFunc(metric, dim, &distFunc, &alignment);
        normalize_func =
            vecType == VecSimType_FLOAT32 ? normalizeVectorFloat : normalizeVectorDouble;
    }

    /**
     * @brief Destroy the Vec Sim Index object
     *
     */
    virtual ~VecSimIndexAbstract() {}

    inline dist_func_t<DistType> getDistFunc() const { return distFunc; }
    inline size_t getDim() const { return dim; }
    inline void setLastSearchMode(VecSearchMode mode) override { this->lastMode = mode; }
    inline bool isMultiValue() const { return isMulti; }
    inline VecSimType getType() const { return vecType; }
    inline VecSimMetric getMetric() const { return metric; }
    inline size_t getDataSize() const { return dataSize; }
    inline size_t getBlockSize() const { return blockSize; }

    virtual VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius,
                                              VecSimQueryParams *queryParams) const = 0;
    VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius,
                                      VecSimQueryParams *queryParams,
                                      VecSimQueryResult_Order order) const override {
        auto results = rangeQuery(queryBlob, radius, queryParams);
        sort_results(results, order);
        return results;
    }

    void log(const char *fmt, ...) const {
        if (VecSimIndexInterface::logCallback) {
            // Format the message and call the callback
            va_list args;
            va_start(args, fmt);
            int len = vsnprintf(NULL, 0, fmt, args);
            va_end(args);
            char *buf = new char[len + 1];
            va_start(args, fmt);
            vsnprintf(buf, len + 1, fmt, args);
            va_end(args);
            logCallback(this->logCallbackCtx, buf);
            delete[] buf;
        }
    }

    // Adds all common info to the info iterator, besides the block size (currently 8 fields).
    void addCommonInfoToIterator(VecSimInfoIterator *infoIterator, const CommonInfo &info) const {
        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::TYPE_STRING,
            .fieldType = INFOFIELD_STRING,
            .fieldValue = {FieldValue{.stringValue = VecSimType_ToString(info.basicInfo.type)}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::DIMENSION_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.basicInfo.dim}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::METRIC_STRING,
                             .fieldType = INFOFIELD_STRING,
                             .fieldValue = {FieldValue{
                                 .stringValue = VecSimMetric_ToString(info.basicInfo.metric)}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::IS_MULTI_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.basicInfo.isMulti}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.indexSize}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_LABEL_COUNT_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.indexLabelCount}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::MEMORY_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.memory}}});
        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::SEARCH_MODE_STRING,
            .fieldType = INFOFIELD_STRING,
            .fieldValue = {FieldValue{.stringValue = VecSimSearchMode_ToString(info.lastMode)}}});
    }
    const void *processBlob(const void *original_blob, void *aligned_mem) const {
        void *processed_blob;
        // if the blob is not aligned, or we need to normalize, we copy it
        if ((this->alignment && (uintptr_t)original_blob % this->alignment) ||
            this->metric == VecSimMetric_Cosine) {
            memcpy(aligned_mem, original_blob, this->dataSize);
            processed_blob = aligned_mem;
        } else {
            processed_blob = (void *)original_blob;
        }

        // if the metric is cosine, we need to normalize
        if (this->metric == VecSimMetric_Cosine) {
            // normalize the copy in place
            normalize_func(processed_blob, this->dim);
        }

        return processed_blob;
    }

    /**
     * @brief Get the basic static info object
     *
     * @return basicInfo
     */
    VecSimIndexBasicInfo getBasicInfo() const {
        VecSimIndexBasicInfo info{.blockSize = this->blockSize,
                                  .metric = this->metric,
                                  .type = this->vecType,
                                  .isMulti = this->isMulti,
                                  .dim = this->dim};
        return info;
    }

protected:
    virtual int addVectorWrapper(const void *blob, labelType label, void *auxiliaryCtx) override {
        char PORTABLE_ALIGN aligned_mem[this->dataSize];
        const void *processed_blob = processBlob(blob, aligned_mem);

        return this->addVector(processed_blob, label, auxiliaryCtx);
    }

    virtual VecSimQueryResult_List topKQueryWrapper(const void *queryBlob, size_t k,
                                                    VecSimQueryParams *queryParams) const override {
        char PORTABLE_ALIGN aligned_mem[this->dataSize];
        const void *processed_blob = processBlob(queryBlob, aligned_mem);

        return this->topKQuery(processed_blob, k, queryParams);
    }

    virtual VecSimQueryResult_List rangeQueryWrapper(const void *queryBlob, double radius,
                                                     VecSimQueryParams *queryParams,
                                                     VecSimQueryResult_Order order) const override {
        char PORTABLE_ALIGN aligned_mem[this->dataSize];
        const void *processed_blob = processBlob(queryBlob, aligned_mem);

        return this->rangeQuery(processed_blob, radius, queryParams, order);
    }

    virtual VecSimBatchIterator *
    newBatchIteratorWrapper(const void *queryBlob, VecSimQueryParams *queryParams) const override {
        char PORTABLE_ALIGN aligned_mem[this->dataSize];
        const void *processed_blob = processBlob(queryBlob, aligned_mem);

        return this->newBatchIterator(processed_blob, queryParams);
    }

    void runGC() override {} // Do nothing, relevant for tiered index only.
};

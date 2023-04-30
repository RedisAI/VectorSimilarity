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
    VecSimMetric metric; // Distance metric to use in the index.
    size_t blockSize;    // Index's vector block size (determines by how many vectors to resize when
                         // resizing)
    dist_func_t<DistType>
        dist_func;           // Index's distance function. Chosen by the type, metric and dimension.
    VecSearchMode last_mode; // The last search mode in RediSearch (used for debug/testing).
    bool isMulti;            // Determines if the index should multi-index or not.
    void *logCallbackCtx;    // Context for the log callback.

    /**
     * @brief Get the common info object
     *
     * @return CommonInfo
     */
    CommonInfo getCommonInfo() const {
        CommonInfo info;
        info.dim = this->dim;
        info.type = this->vecType;
        info.metric = this->metric;
        info.blockSize = this->blockSize;
        info.last_mode = this->last_mode;
        info.isMulti = this->isMulti;
        info.memory = this->allocator->getAllocationSize();
        info.indexSize = this->indexSize();
        info.indexLabelCount = this->indexLabelCount();
        return info;
    }

    void addCommonInfoToIterator(VecSimInfoIterator *infoIterator, const CommonInfo &info) const {
        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::TYPE_STRING,
            .fieldType = INFOFIELD_STRING,
            .fieldValue = {FieldValue{.stringValue = VecSimType_ToString(info.type)}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::DIMENSION_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.dim}}});
        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::METRIC_STRING,
            .fieldType = INFOFIELD_STRING,
            .fieldValue = {FieldValue{.stringValue = VecSimMetric_ToString(info.metric)}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::IS_MULTI_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.isMulti}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.indexSize}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_LABEL_COUNT_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.indexLabelCount}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.blockSize}}});
        infoIterator->addInfoField(
            VecSim_InfoField{.fieldName = VecSimCommonStrings::MEMORY_STRING,
                             .fieldType = INFOFIELD_UINT64,
                             .fieldValue = {FieldValue{.uintegerValue = info.memory}}});
        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::SEARCH_MODE_STRING,
            .fieldType = INFOFIELD_STRING,
            .fieldValue = {FieldValue{.stringValue = VecSimSearchMode_ToString(info.last_mode)}}});
    }

public:
    /**
     * @brief Construct a new Vec Sim Index object
     *
     */
    VecSimIndexAbstract(const AbstractIndexInitParams &params)
        : VecSimIndexInterface(params.allocator), dim(params.dim), vecType(params.vecType),
          metric(params.metric),
          blockSize(params.blockSize ? params.blockSize : DEFAULT_BLOCK_SIZE),
          last_mode(EMPTY_MODE), isMulti(params.multi), logCallbackCtx(params.logCtx) {
        assert(VecSimType_sizeof(vecType));
        spaces::SetDistFunc(metric, dim, &dist_func);
    }

    /**
     * @brief Destroy the Vec Sim Index object
     *
     */
    virtual ~VecSimIndexAbstract() {}

    inline dist_func_t<DistType> getDistFunc() const { return dist_func; }
    inline size_t getDim() const { return dim; }
    inline void setLastSearchMode(VecSearchMode mode) override { this->last_mode = mode; }
    inline bool isMultiValue() const { return isMulti; }
    inline VecSimType getType() const { return vecType; }
    inline VecSimMetric getMetric() const { return metric; }

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
};

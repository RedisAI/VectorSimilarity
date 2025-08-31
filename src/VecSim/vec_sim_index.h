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
#include "vec_sim_interface.h"
#include "query_results.h"
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "info_iterator_struct.h"
#include "containers/data_blocks_container.h"
#include "containers/raw_data_container_interface.h"

#include <cassert>
#include <functional>

/**
 * @brief Struct for initializing an abstract index class.
 *
 * @param allocator The allocator to use for the index.
 * @param dim The dimension of the vectors in the index.
 * @param vecType The type of the vectors in the index.
 * @param dataSize The size of stored vectors in bytes.
 * @param metric The metric to use in the index.
 * @param blockSize The block size to use in the index.
 * @param multi Determines if the index should multi-index or not.
 * @param logCtx The context to use for logging.
 */
struct AbstractIndexInitParams {
    std::shared_ptr<VecSimAllocator> allocator;
    size_t dim;
    VecSimType vecType;
    size_t dataSize;
    VecSimMetric metric;
    size_t blockSize;
    bool multi;
    void *logCtx;
};

/**
 * @brief Struct for initializing the components of the abstract index.
 * The index takes ownership of the components allocations' and is responsible for freeing
 * them when the index is destroyed.
 *
 * @param indexCalculator The distance calculator for the index.
 * @param preprocessors The preprocessing pipeline for ingesting user data before storage and
 * querying.
 */
template <typename DataType, typename DistType>
struct IndexComponents {
    IndexCalculatorInterface<DistType> *indexCalculator;
    PreprocessorsContainerAbstract *preprocessors;
};

/**
 * @brief Abstract C++ class for vector index, delete and lookup
 *
 */
template <typename DataType, typename DistType>
struct VecSimIndexAbstract : public VecSimIndexInterface {
protected:
    size_t dim;          // Vector's dimension.
    VecSimType vecType;  // Datatype to index.
    size_t dataSize;     // Vector size in bytes
    VecSimMetric metric; // Distance metric to use in the index.
    size_t blockSize;    // Index's vector block size (determines by how many vectors to resize when
                         // resizing)
    IndexCalculatorInterface<DistType> *indexCalculator; // Distance calculator.
    PreprocessorsContainerAbstract *preprocessors;       // Stroage and query preprocessors.
    mutable VecSearchMode lastMode; // The last search mode in RediSearch (used for debug/testing).
    bool isMulti;                   // Determines if the index should multi-index or not.
    void *logCallbackCtx;           // Context for the log callback.

    RawDataContainer *vectors; // The raw vectors data container.

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

public:
    /**
     * @brief Construct a new Vec Sim Index object
     *
     */
    VecSimIndexAbstract(const AbstractIndexInitParams &params,
                        const IndexComponents<DataType, DistType> &components)
        : VecSimIndexInterface(params.allocator), dim(params.dim), vecType(params.vecType),
          dataSize(params.dataSize), metric(params.metric),
          blockSize(params.blockSize ? params.blockSize : DEFAULT_BLOCK_SIZE),
          indexCalculator(components.indexCalculator), preprocessors(components.preprocessors),
          lastMode(EMPTY_MODE), isMulti(params.multi), logCallbackCtx(params.logCtx) {
        assert(VecSimType_sizeof(vecType));
        assert(dataSize);
        this->vectors = new (this->allocator) DataBlocksContainer(
            this->blockSize, this->dataSize, this->allocator, this->getAlignment());
    }

    /**
     * @brief Destroy the Vec Sim Index object
     *
     */
    virtual ~VecSimIndexAbstract() noexcept {
        delete this->vectors;
        delete indexCalculator;
        delete preprocessors;
    }

    /**
     * @brief Calculate the distance between two vectors based on index parameters.
     *
     * @return the distance between the vectors.
     */
    DistType calcDistance(const void *vector_data1, const void *vector_data2) const {
        return indexCalculator->calcDistance(vector_data1, vector_data2, this->dim);
    }

    /**
     * @brief Preprocess a blob for both storage and query.
     *
     * @param original_blob will be copied.
     * @return two unique_ptr of the processed blobs.
     */
    ProcessedBlobs preprocess(const void *original_blob) const;

    /**
     * @brief Preprocess a blob for query.
     *
     * @param queryBlob will be copied if preprocessing is required, or if force_copy is set to
     * true.
     * @return unique_ptr of the processed blob.
     */
    MemoryUtils::unique_blob preprocessQuery(const void *queryBlob, bool force_copy = false) const;

    /**
     * @brief Preprocess a blob for storage.
     *
     * @param original_blob will be copied.
     * @return unique_ptr of the processed blob.
     */
    MemoryUtils::unique_blob preprocessForStorage(const void *original_blob) const;

    /**
     * @brief Preprocess a blob for storage in place.
     *
     * @param blob will be directly modified, not copied.
     */
    void preprocessStorageInPlace(void *blob) const;

    inline size_t getDim() const { return dim; }
    inline void setLastSearchMode(VecSearchMode mode) override { this->lastMode = mode; }
    inline bool isMultiValue() const { return isMulti; }
    inline VecSimType getType() const { return vecType; }
    inline VecSimMetric getMetric() const { return metric; }
    inline size_t getDataSize() const { return dataSize; }
    inline size_t getBlockSize() const { return blockSize; }
    inline auto getAlignment() const { return this->preprocessors->getAlignment(); }

    virtual inline VecSimIndexStatsInfo statisticInfo() const override {
        return VecSimIndexStatsInfo{
            .memory = this->getAllocationSize(),
            .numberOfMarkedDeleted = 0,
        };
    }

    virtual VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                         VecSimQueryParams *queryParams) const = 0;
    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams,
                                 VecSimQueryReply_Order order) const override {
        auto results = rangeQuery(queryBlob, radius, queryParams);
        sort_results(results, order);
        return results;
    }

    void log(const char *level, const char *fmt, ...) const {
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
            logCallback(this->logCallbackCtx, level, buf);
            delete[] buf;
        }
    }

    // Adds all common info to the info iterator, besides the block size (currently 8 fields).
    void addCommonInfoToIterator(VecSimDebugInfoIterator *infoIterator,
                                 const CommonInfo &info) const {
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

    /**
     * @brief Get the basic static info object
     *
     * @return basicInfo
     */
    VecSimIndexBasicInfo getBasicInfo() const {
        VecSimIndexBasicInfo info{
            .metric = this->metric,
            .type = this->vecType,
            .isMulti = this->isMulti,
            .blockSize = this->blockSize,
            .dim = this->dim,
        };
        return info;
    }
#ifdef BUILD_TESTS
    void replacePPContainer(PreprocessorsContainerAbstract *newPPContainer) {
        delete this->preprocessors;
        this->preprocessors = newPPContainer;
    }

    IndexComponents<DataType, DistType> get_components() const {
        return {.indexCalculator = this->indexCalculator, .preprocessors = this->preprocessors};
    }

    /**
     * @brief Used for testing - get only the vector elements associated with a given label.
     * This function copies only the vector(s) elements into the output vector,
     * without any additional metadata that might be stored with the vector.
     *
     * Important: This method returns ONLY the vector elements, even if the stored vector contains
     * additional metadata. For example, with int8_t/uint8_t vectors using cosine similarity,
     * this method will NOT return the norm that is stored with the vector(s).
     *
     * If you need the complete data including any metadata, use getStoredVectorDataByLabel()
     * instead.
     *
     * @param label The label to retrieve vector(s) elements for
     * @param vectors_output Empty vector to be filled with vector(s)
     */
    virtual void getDataByLabel(labelType label,
                                std::vector<std::vector<DataType>> &vectors_output) const = 0;

    /**
     * @brief Used for testing - get the complete raw data associated with a given label.
     * This function returns the ENTIRE vector(s) data as stored in the index, including any
     * additional metadata that might be stored alongside the vector elements.
     *
     * For example:
     * - For int8_t/uint8_t vectors with cosine similarity, this includes the norm stored at the end
     * - For other vector types or future implementations, this will include any additional data
     *   that might be stored with the vector
     *
     * Use this method when you need access to the complete vector data as it is stored internally.
     *
     * @param label The label to retrieve data for
     * @return A vector containing the complete vector data (elements + metadata) for the given
     * label
     */
    virtual std::vector<std::vector<char>> getStoredVectorDataByLabel(labelType label) const = 0;
#endif

    /**
     * Virtual functions that access the label lookup which is implemented in the derived classes
     * Return all the labels in the index - this should be used for computing the number of distinct
     * labels in a tiered index, and caller should hold the appropriate guards.
     */
    virtual vecsim_stl::set<labelType> getLabelsSet() const = 0;

protected:
    void runGC() override {}              // Do nothing, relevant for tiered index only.
    void acquireSharedLocks() override {} // Do nothing, relevant for tiered index only.
    void releaseSharedLocks() override {} // Do nothing, relevant for tiered index only.
};

template <typename DataType, typename DistType>
ProcessedBlobs VecSimIndexAbstract<DataType, DistType>::preprocess(const void *blob) const {
    return this->preprocessors->preprocess(blob, this->dataSize);
}

template <typename DataType, typename DistType>
MemoryUtils::unique_blob
VecSimIndexAbstract<DataType, DistType>::preprocessQuery(const void *queryBlob,
                                                         bool force_copy) const {
    return this->preprocessors->preprocessQuery(queryBlob, this->dataSize, force_copy);
}

template <typename DataType, typename DistType>
MemoryUtils::unique_blob
VecSimIndexAbstract<DataType, DistType>::preprocessForStorage(const void *original_blob) const {
    return this->preprocessors->preprocessForStorage(original_blob, this->dataSize);
}

template <typename DataType, typename DistType>
void VecSimIndexAbstract<DataType, DistType>::preprocessStorageInPlace(void *blob) const {
    this->preprocessors->preprocessStorageInPlace(blob, this->dataSize);
}

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
#include "VecSim/spaces/computer/computer.h"
#include "info_iterator_struct.h"
#include <cassert>
#include <functional>

using spaces::dist_func_t; // TODO calculator : remove!!!

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
 * @brief AddVectorCtx facilitates the management and transfer of vector processing contexts.
 *
 */

struct AddVectorCtx {
    AddVectorCtx() = default;
    explicit AddVectorCtx(ProcessedBlobs processedBlobs)
        : processedBlobs(std::move(processedBlobs)) {}

    AddVectorCtx(AddVectorCtx &&other) noexcept = default;
    AddVectorCtx &operator=(AddVectorCtx &&other) noexcept = default;

    const void setBlobs(ProcessedBlobs processedBlobs) {
        this->processedBlobs = std::move(processedBlobs);
    }

    const void *getStorageBlob() const { return this->processedBlobs.getStorageBlob(); }
    const void *getQueryBlob() const { return this->processedBlobs.getQueryBlob(); }

private:
    ProcessedBlobs processedBlobs;
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
    IndexComputerAbstract<DistType> *indexComputer; // Index's computer.
    // TODO: remove alignment once datablock is implemented in HNSW
    unsigned char alignment;        // Alignment hint to allocate vectors with.
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

    spaces::normalizeVector_f<DataType>
        normalize_func; // A pointer to a normalization function of specific type.

public:
    /**
     * @brief Construct a new Vec Sim Index object
     *
     */
    VecSimIndexAbstract(const AbstractIndexInitParams &params,
                        IndexComputerAbstract<DistType> *indexComputer)
        : VecSimIndexInterface(params.allocator), dim(params.dim), vecType(params.vecType),
          dataSize(dim * VecSimType_sizeof(vecType)), metric(params.metric),
          blockSize(params.blockSize ? params.blockSize : DEFAULT_BLOCK_SIZE),
          indexComputer(indexComputer),
          alignment(
              indexComputer
                  ->getAlignment()), // computer TODO: remove alignmen also from the index members
          lastMode(EMPTY_MODE), isMulti(params.multi), logCallbackCtx(params.logCtx),
          normalize_func(spaces::GetNormalizeFunc<DataType>()) {
        assert(VecSimType_sizeof(vecType));
    }

    /**
     * @brief Destroy the Vec Sim Index object
     *
     */
    virtual ~VecSimIndexAbstract() { delete indexComputer; }

    /**
     * @brief Add a vector blob and its id to the index.
     *
     * @param blob binary representation of the vector. Blob size should match the index data type
     * and dimension. The blob will be copied and processed as needed.
     * @param label the label of the added vector.
     * @return the number of new vectors inserted (1 for new insertion, 0 for override).
     */
    virtual int addVector(const void *blob, labelType label) = 0;

    /**
     * @brief Add a preprocessed vector and its id to the index.
     *
     * @param add_vector_ctx contains the vector blob processed for storage purposes, and the vector
     * blob processed for query (for example, to find its nearest neighbors in a HNSW graph).
     * The processed blobs may be identical, for example in case of a dense vectors cosine index,
     * where both storage and query are normalized, or different for example in SQ index, in which
     * we need to quantize the storage blob and keep the query blob as is. It is the index
     * computer's responsibility to handle complex cases like cosine-SQ index.
     * @param label the label of the added vector.
     * @return the number of new vectors inserted (1 for new insertion, 0 for override).
     */
    virtual int addVector(const AddVectorCtx *add_vector_ctx, labelType label) = 0;

    DistType calcDistance(const void *vector_data1, const void *vector_data2) const {
        return indexComputer->calcDistance(vector_data1, vector_data2, this->dim);
    }

    /**
     * @brief Preprocess a blob for both storage and query.
     *
     * @param blob will be copied.
     * @return unique_ptr of the processed blobs.
     */
    ProcessedBlobs preprocess(const void *blob) const;

    /**
     * @brief Preprocess a blob for query.
     *
     * @param blob will be copied.
     * @return unique_ptr of the processed blob.
     */
    std::unique_ptr<void, alloc_deleter_t> processQuery(const void *queryBlob) const;

    inline size_t getDim() const { return dim; }
    inline void setLastSearchMode(VecSearchMode mode) override { this->lastMode = mode; }
    inline bool isMultiValue() const { return isMulti; }
    inline VecSimType getType() const { return vecType; }
    inline VecSimMetric getMetric() const { return metric; }
    inline size_t getDataSize() const { return dataSize; }
    inline size_t getBlockSize() const { return blockSize; }
    inline auto getAlignment() const { return alignment; }

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
    virtual VecSimQueryReply *topKQueryWrapper(const void *queryBlob, size_t k,
                                               VecSimQueryParams *queryParams) const override {
        auto aligned_mem = this->indexComputer->preprocessQuery(queryBlob, this->dataSize);
        return this->topKQuery(aligned_mem.get(), k, queryParams);
    }

    virtual VecSimQueryReply *rangeQueryWrapper(const void *queryBlob, double radius,
                                                VecSimQueryParams *queryParams,
                                                VecSimQueryReply_Order order) const override {
        auto aligned_mem = this->indexComputer->preprocessQuery(queryBlob, this->dataSize);
        return this->rangeQuery(aligned_mem.get(), radius, queryParams, order);
    }

    virtual VecSimBatchIterator *
    newBatchIteratorWrapper(const void *queryBlob, VecSimQueryParams *queryParams) const override {
        auto aligned_mem = this->indexComputer->preprocessQuery(queryBlob, this->dataSize);
        return this->newBatchIterator(aligned_mem.get(), queryParams);
    }

    void runGC() override {}              // Do nothing, relevant for tiered index only.
    void acquireSharedLocks() override {} // Do nothing, relevant for tiered index only.
    void releaseSharedLocks() override {} // Do nothing, relevant for tiered index only.
};

template <typename DataType, typename DistType>
ProcessedBlobs VecSimIndexAbstract<DataType, DistType>::preprocess(const void *blob) const {
    return this->indexComputer->preprocess(blob, this->dataSize);
}

template <typename DataType, typename DistType>
std::unique_ptr<void, alloc_deleter_t>
VecSimIndexAbstract<DataType, DistType>::processQuery(const void *queryBlob) const {
    return this->indexComputer->preprocessQuery(queryBlob, this->dataSize);
}

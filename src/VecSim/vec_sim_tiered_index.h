#pragma once

#include "vec_sim_index.h"
#include "algorithms/brute_force/brute_force.h"
#include "VecSim/batch_iterator.h"
#include "VecSim/tombstone_interface.h"
#include "VecSim/utils/query_result_utils.h"
#include "VecSim/utils/alignment.h"

#include <shared_mutex>

#define TIERED_LOG this->backendIndex->log

/**
 * Definition of generic job structure for asynchronous tiered index.
 */
struct AsyncJob : public VecsimBaseObject {
    JobType jobType;
    JobCallback Execute; // A callback that receives a job as its input and executes the job.
    VecSimIndex *index;
    bool isValid;

    AsyncJob(std::shared_ptr<VecSimAllocator> allocator, JobType type, JobCallback callback,
             VecSimIndex *index_ref)
        : VecsimBaseObject(allocator), jobType(type), Execute(callback), index(index_ref),
          isValid(true) {}
};

// All read operations (including KNN, range, batch iterators and get-distance-from) are guaranteed
// to consider all vectors that were added to the index before the query was submitted. The results
// may include vectors that were added after the query was submitted, with no guarantees.
template <typename DataType, typename DistType>
class VecSimTieredIndex : public VecSimIndexInterface {
protected:
    VecSimIndexAbstract<DataType, DistType> *backendIndex;
    BruteForceIndex<DataType, DistType> *frontendIndex;

    void *jobQueue;
    void *jobQueueCtx; // External context to be sent to the submit callback.
    SubmitCB SubmitJobsToQueue;

    mutable std::shared_mutex flatIndexGuard;
    mutable std::shared_mutex mainIndexGuard;

    size_t flatBufferLimit;

    void submitSingleJob(AsyncJob *job) {
        this->SubmitJobsToQueue(this->jobQueue, this->jobQueueCtx, &job, &job->Execute, 1);
    }

    void submitJobs(vecsim_stl::vector<AsyncJob *> &jobs) {
        vecsim_stl::vector<JobCallback> callbacks(jobs.size(), this->allocator);
        for (size_t i = 0; i < jobs.size(); i++) {
            callbacks[i] = jobs[i]->Execute;
        }
        this->SubmitJobsToQueue(this->jobQueue, this->jobQueueCtx, jobs.data(), callbacks.data(),
                                jobs.size());
    }

public:
    VecSimTieredIndex(VecSimIndexAbstract<DataType, DistType> *backendIndex_,
                      BruteForceIndex<DataType, DistType> *frontendIndex_,
                      TieredIndexParams tieredParams, std::shared_ptr<VecSimAllocator> allocator)
        : VecSimIndexInterface(allocator), backendIndex(backendIndex_),
          frontendIndex(frontendIndex_), jobQueue(tieredParams.jobQueue),
          jobQueueCtx(tieredParams.jobQueueCtx), SubmitJobsToQueue(tieredParams.submitCb),
          flatBufferLimit(tieredParams.flatBufferLimit) {}

    virtual ~VecSimTieredIndex() {
        VecSimIndex_Free(backendIndex);
        VecSimIndex_Free(frontendIndex);
    }

    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override;

    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams,
                                 VecSimQueryReply_Order order) const override;

    virtual inline uint64_t getAllocationSize() const override {
        return this->allocator->getAllocationSize() + this->backendIndex->getAllocationSize() +
               this->frontendIndex->getAllocationSize();
    }

    VecSimIndexStatsInfo statisticInfo() const override;
    virtual VecSimIndexDebugInfo debugInfo() const override;
    virtual VecSimDebugInfoIterator *debugInfoIterator() const override;

    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override {
        // For now, decide according to the bigger index.
        return this->backendIndex->indexSize() > this->frontendIndex->indexSize()
                   ? this->backendIndex->preferAdHocSearch(subsetSize, k, initial_check)
                   : this->frontendIndex->preferAdHocSearch(subsetSize, k, initial_check);
    }

    // Return the current state of the global write mode (async/in-place).
    static VecSimWriteMode getWriteMode() { return VecSimIndexInterface::asyncWriteMode; }

#ifdef BUILD_TESTS
    inline VecSimIndexAbstract<DataType, DistType> *getFlatBufferIndex() {
        return this->frontendIndex;
    }
    inline size_t getFlatBufferLimit() { return this->flatBufferLimit; }

    virtual void fitMemory() override {
        this->backendIndex->fitMemory();
        this->frontendIndex->fitMemory();
    }
#endif
};
template <typename DataType, typename DistType>
VecSimQueryReply *
VecSimTieredIndex<DataType, DistType>::topKQuery(const void *queryBlob, size_t k,
                                                 VecSimQueryParams *queryParams) const {
    this->flatIndexGuard.lock_shared();

    // If the flat buffer is empty, we can simply query the main index.
    if (this->frontendIndex->indexSize() == 0) {
        // Release the flat lock and acquire the main lock.
        this->flatIndexGuard.unlock_shared();

        // Simply query the main index and return the results while holding the lock.
        auto processed_query_ptr = this->frontendIndex->preprocessQuery(queryBlob);
        const void *processed_query = processed_query_ptr.get();
        this->mainIndexGuard.lock_shared();
        auto res = this->backendIndex->topKQuery(processed_query, k, queryParams);
        this->mainIndexGuard.unlock_shared();

        return res;
    } else {
        // No luck... first query the flat buffer and release the lock.
        // The query blob is already processed according to the frontend index.
        auto flat_results = this->frontendIndex->topKQuery(queryBlob, k, queryParams);
        this->flatIndexGuard.unlock_shared();

        // If the query failed (currently only on timeout), return the error code.
        if (flat_results->code != VecSim_QueryReply_OK) {
            assert(flat_results->results.empty());
            return flat_results;
        }

        auto processed_query_ptr = this->frontendIndex->preprocessQuery(queryBlob);
        const void *processed_query = processed_query_ptr.get();
        // Lock the main index and query it.
        this->mainIndexGuard.lock_shared();
        auto main_results = this->backendIndex->topKQuery(processed_query, k, queryParams);
        this->mainIndexGuard.unlock_shared();

        // If the query failed (currently only on timeout), return the error code.
        if (main_results->code != VecSim_QueryReply_OK) {
            // Free the flat results.
            VecSimQueryReply_Free(flat_results);

            assert(main_results->results.empty());
            return main_results;
        }

        // Merge the results and return, avoiding duplicates.
        if (this->backendIndex->isMultiValue()) {
            return merge_result_lists<true>(main_results, flat_results, k);
        } else {
            return merge_result_lists<false>(main_results, flat_results, k);
        }
    }
}

template <typename DataType, typename DistType>
VecSimQueryReply *
VecSimTieredIndex<DataType, DistType>::rangeQuery(const void *queryBlob, double radius,
                                                  VecSimQueryParams *queryParams,
                                                  VecSimQueryReply_Order order) const {
    this->flatIndexGuard.lock_shared();

    // If the flat buffer is empty, we can simply query the main index.
    if (this->frontendIndex->indexSize() == 0) {
        // Release the flat lock and acquire the main lock.
        this->flatIndexGuard.unlock_shared();

        auto processed_query_ptr = this->frontendIndex->preprocessQuery(queryBlob);
        const void *processed_query = processed_query_ptr.get();
        // Simply query the main index and return the results while holding the lock.
        this->mainIndexGuard.lock_shared();
        auto res = this->backendIndex->rangeQuery(processed_query, radius, queryParams);
        this->mainIndexGuard.unlock_shared();

        // We could have passed the order to the main index, but we can sort them here after
        // unlocking it instead.
        sort_results(res, order);
        return res;
    } else {
        // No luck... first query the flat buffer and release the lock.
        // The query blob is already processed according to the frontend index.
        auto flat_results = this->frontendIndex->rangeQuery(queryBlob, radius, queryParams);
        this->flatIndexGuard.unlock_shared();

        // If the query failed (currently only on timeout), return the error code and the partial
        // results.
        if (flat_results->code != VecSim_QueryReply_OK) {
            return flat_results;
        }

        auto processed_query_ptr = this->frontendIndex->preprocessQuery(queryBlob);
        const void *processed_query = processed_query_ptr.get();
        // Lock the main index and query it.
        this->mainIndexGuard.lock_shared();
        auto main_results = this->backendIndex->rangeQuery(processed_query, radius, queryParams);
        this->mainIndexGuard.unlock_shared();

        // Merge the results and return, avoiding duplicates.
        // At this point, the return code of the FLAT index is OK, and the return code of the MAIN
        // index is either OK or TIMEOUT. Make sure to return the return code of the MAIN index.
        if (BY_SCORE == order) {
            sort_results_by_score_then_id(main_results);
            sort_results_by_score_then_id(flat_results);

            // Keep the return code of the main index.
            auto code = main_results->code;

            // Merge the sorted results with no limit (all the results are valid).
            VecSimQueryReply *ret;
            if (this->backendIndex->isMultiValue()) {
                ret = merge_result_lists<true>(main_results, flat_results, -1);
            } else {
                ret = merge_result_lists<false>(main_results, flat_results, -1);
            }
            // Restore the return code and return.
            ret->code = code;
            return ret;

        } else { // BY_ID
            // Notice that we don't modify the return code of the main index in any step.
            concat_results(main_results, flat_results);
            if (this->backendIndex->isMultiValue()) {
                filter_results_by_id<true>(main_results);
            } else {
                filter_results_by_id<false>(main_results);
            }
            return main_results;
        }
    }
}

template <typename DataType, typename DistType>
VecSimIndexStatsInfo VecSimTieredIndex<DataType, DistType>::statisticInfo() const {
    auto stats = VecSimIndexStatsInfo{
        .memory = this->getAllocationSize(),
        .numberOfMarkedDeleted = 0, // Default value if cast fails
    };

    // If backend implements VecSimIndexTombstone, get number of marked deleted
    if (auto tombstone = dynamic_cast<VecSimIndexTombstone *>(this->backendIndex)) {
        stats.numberOfMarkedDeleted = tombstone->getNumMarkedDeleted();
    }

    return stats;
}

template <typename DataType, typename DistType>
VecSimIndexDebugInfo VecSimTieredIndex<DataType, DistType>::debugInfo() const {
    VecSimIndexDebugInfo info;
    this->flatIndexGuard.lock_shared();
    this->mainIndexGuard.lock_shared();
    VecSimIndexDebugInfo frontendInfo = this->frontendIndex->debugInfo();
    VecSimIndexDebugInfo backendInfo = this->backendIndex->debugInfo();
    this->flatIndexGuard.unlock_shared();
    this->mainIndexGuard.unlock_shared();

    info.commonInfo.indexLabelCount = this->indexLabelCount();
    info.commonInfo.indexSize =
        frontendInfo.commonInfo.indexSize + backendInfo.commonInfo.indexSize;
    info.commonInfo.memory = this->getAllocationSize();
    info.commonInfo.lastMode = backendInfo.commonInfo.lastMode;

    VecSimIndexBasicInfo basic_info{
        .algo = backendInfo.commonInfo.basicInfo.algo,
        .metric = backendInfo.commonInfo.basicInfo.metric,
        .type = backendInfo.commonInfo.basicInfo.type,
        .isMulti = this->backendIndex->isMultiValue(),
        .isTiered = true,
        .blockSize = backendInfo.commonInfo.basicInfo.blockSize,
        .dim = backendInfo.commonInfo.basicInfo.dim,
    };
    info.commonInfo.basicInfo = basic_info;

    switch (backendInfo.commonInfo.basicInfo.algo) {
    case VecSimAlgo_HNSWLIB:
        info.tieredInfo.backendInfo.hnswInfo = backendInfo.hnswInfo;
        break;
    case VecSimAlgo_SVS:
        break;
    case VecSimAlgo_BF:
    case VecSimAlgo_TIERED:
        assert(false && "Invalid backend algorithm");
    }

    info.tieredInfo.backendCommonInfo = backendInfo.commonInfo;
    // For now, this is hard coded to FLAT
    info.tieredInfo.frontendCommonInfo = frontendInfo.commonInfo;
    info.tieredInfo.bfInfo = frontendInfo.bfInfo;

    info.tieredInfo.backgroundIndexing = frontendInfo.commonInfo.indexSize > 0;
    info.tieredInfo.management_layer_memory = this->allocator->getAllocationSize();
    info.tieredInfo.bufferLimit = this->flatBufferLimit;
    return info;
}

template <typename DataType, typename DistType>
VecSimDebugInfoIterator *VecSimTieredIndex<DataType, DistType>::debugInfoIterator() const {
    VecSimIndexDebugInfo info = this->debugInfo();
    // For readability. Update this number when needed.
    size_t numberOfInfoFields = 14;
    auto *infoIterator = new VecSimDebugInfoIterator(numberOfInfoFields, this->allocator);

    // Set tiered explicitly as algo name for root iterator.
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::ALGORITHM_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimCommonStrings::TIERED_STRING}}});

    this->backendIndex->addCommonInfoToIterator(infoIterator, info.commonInfo);

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::TIERED_MANAGEMENT_MEMORY_STRING,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.tieredInfo.management_layer_memory}}});

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::TIERED_BACKGROUND_INDEXING_STRING,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.tieredInfo.backgroundIndexing}}});

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::TIERED_BUFFER_LIMIT_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.tieredInfo.bufferLimit}}});

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::FRONTEND_INDEX_STRING,
        .fieldType = INFOFIELD_ITERATOR,
        .fieldValue = {FieldValue{.iteratorValue = this->frontendIndex->debugInfoIterator()}}});

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::BACKEND_INDEX_STRING,
        .fieldType = INFOFIELD_ITERATOR,
        .fieldValue = {FieldValue{.iteratorValue = this->backendIndex->debugInfoIterator()}}});
    return infoIterator;
};

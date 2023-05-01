#pragma once

#include "vec_sim_index.h"
#include "algorithms/brute_force/brute_force.h"
#include "VecSim/batch_iterator.h"
#include "VecSim/utils/query_result_utils.h"

#include <shared_mutex>

/**
 * Definition of generic job structure for asynchronous tiered index.
 */
struct AsyncJob : public VecsimBaseObject {
    JobType jobType;
    JobCallback Execute; // A callback that receives a job as its input and executes the job.
    VecSimIndex *index;

    AsyncJob(std::shared_ptr<VecSimAllocator> allocator, JobType type, JobCallback callback,
             VecSimIndex *index_ref)
        : VecsimBaseObject(allocator), jobType(type), Execute(callback), index(index_ref) {}
};

template <typename DataType, typename DistType>
class VecSimTieredIndex : public VecSimIndexInterface {
protected:
    VecSimIndexAbstract<DistType> *backendIndex;
    BruteForceIndex<DataType, DistType> *frontendIndex;

    void *jobQueue;
    void *jobQueueCtx; // External context to be sent to the submit callback.
    SubmitCB SubmitJobsToQueue;

    void *memoryCtx;
    UpdateMemoryCB UpdateIndexMemory;

    mutable std::shared_mutex flatIndexGuard;
    mutable std::shared_mutex mainIndexGuard;

    size_t flatBufferLimit;

    void submitSingleJob(AsyncJob *job) {
        auto **jobs = array_new<AsyncJob *>(1);
        jobs = array_append(jobs, job);
        this->SubmitJobsToQueue(this->jobQueue, (AsyncJob **)jobs, 1, this->jobQueueCtx);
        array_free(jobs);
    }

public:
    VecSimTieredIndex(VecSimIndexAbstract<DistType> *backendIndex_,
                      BruteForceIndex<DataType, DistType> *frontendIndex_,
                      TieredIndexParams tieredParams)
        : VecSimIndexInterface(backendIndex_->getAllocator()), backendIndex(backendIndex_),
          frontendIndex(frontendIndex_), jobQueue(tieredParams.jobQueue),
          jobQueueCtx(tieredParams.jobQueueCtx), SubmitJobsToQueue(tieredParams.submitCb),
          memoryCtx(tieredParams.memoryCtx), UpdateIndexMemory(tieredParams.UpdateMemCb),
          flatBufferLimit(tieredParams.flatBufferLimit) {}

    virtual ~VecSimTieredIndex() {
        VecSimIndex_Free(backendIndex);
        VecSimIndex_Free(frontendIndex);
    }

    VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                     VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius,
                                      VecSimQueryParams *queryParams,
                                      VecSimQueryResult_Order order) override;

    // Return the current state of the global write mode (async/in-place).
    static VecSimWriteMode getWriteMode() { return VecSimIndexInterface::asyncWriteMode; }

private:
    virtual int addVectorWrapper(const void *blob, labelType label, void *auxiliaryCtx) override {
        // Will be used only if a processing stage is needed
        char processed_blob[this->backendIndex->getDataSize()];
        const void *vector_to_add = this->backendIndex->processBlob(blob, processed_blob);
        return this->addVector(vector_to_add, label, auxiliaryCtx);
    }

    virtual VecSimQueryResult_List topKQueryWrapper(const void *queryBlob, size_t k,
                                                    VecSimQueryParams *queryParams) override {
        // Will be used only if a processing stage is needed
        char processed_blob[this->backendIndex->getDataSize()];
        const void *query_to_send = this->backendIndex->processBlob(queryBlob, processed_blob);
        return this->topKQuery(query_to_send, k, queryParams);
    }

    virtual VecSimQueryResult_List rangeQueryWrapper(const void *queryBlob, double radius,
                                                     VecSimQueryParams *queryParams,
                                                     VecSimQueryResult_Order order) override {
        // Will be used only if a processing stage is needed
        char processed_blob[this->backendIndex->getDataSize()];
        const void *query_to_send = this->backendIndex->processBlob(queryBlob, processed_blob);

        return this->rangeQuery(query_to_send, radius, queryParams, order);
    }

    virtual VecSimBatchIterator *
    newBatchIteratorWrapper(const void *queryBlob, VecSimQueryParams *queryParams) const override {
        // Will be used only if a processing stage is needed
        char processed_blob[this->backendIndex->getDataSize()];
        const void *query_to_send = this->backendIndex->processBlob(queryBlob, processed_blob);

        return this->newBatchIterator(query_to_send, queryParams);
    }
};

template <typename DataType, typename DistType>
VecSimQueryResult_List
VecSimTieredIndex<DataType, DistType>::topKQuery(const void *queryBlob, size_t k,
                                                 VecSimQueryParams *queryParams) {
    this->flatIndexGuard.lock_shared();

    // If the flat buffer is empty, we can simply query the main index.
    if (this->frontendIndex->indexSize() == 0) {
        // Release the flat lock and acquire the main lock.
        this->flatIndexGuard.unlock_shared();

        // Simply query the main index and return the results while holding the lock.
        this->mainIndexGuard.lock_shared();
        auto res = this->backendIndex->topKQuery(queryBlob, k, queryParams);
        this->mainIndexGuard.unlock_shared();

        return res;
    } else {
        // No luck... first query the flat buffer and release the lock.
        auto flat_results = this->frontendIndex->topKQuery(queryBlob, k, queryParams);
        this->flatIndexGuard.unlock_shared();

        // If the query failed (currently only on timeout), return the error code.
        if (flat_results.code != VecSim_QueryResult_OK) {
            assert(flat_results.results == nullptr);
            return flat_results;
        }

        // Lock the main index and query it.
        this->mainIndexGuard.lock_shared();
        auto main_results = this->backendIndex->topKQuery(queryBlob, k, queryParams);
        this->mainIndexGuard.unlock_shared();

        // If the query failed (currently only on timeout), return the error code.
        if (main_results.code != VecSim_QueryResult_OK) {
            // Free the flat results.
            VecSimQueryResult_Free(flat_results);

            assert(main_results.results == nullptr);
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
VecSimQueryResult_List
VecSimTieredIndex<DataType, DistType>::rangeQuery(const void *queryBlob, double radius,
                                                  VecSimQueryParams *queryParams,
                                                  VecSimQueryResult_Order order) {
    this->flatIndexGuard.lock_shared();

    // If the flat buffer is empty, we can simply query the main index.
    if (this->frontendIndex->indexSize() == 0) {
        // Release the flat lock and acquire the main lock.
        this->flatIndexGuard.unlock_shared();

        // Simply query the main index and return the results while holding the lock.
        this->mainIndexGuard.lock_shared();
        auto res = this->backendIndex->rangeQuery(queryBlob, radius, queryParams);
        this->mainIndexGuard.unlock_shared();

        // We could have passed the order to the main index, but we can sort them here after
        // unlocking it instead.
        sort_results(res, order);
        return res;
    } else {
        // No luck... first query the flat buffer and release the lock.
        auto flat_results = this->frontendIndex->rangeQuery(queryBlob, radius, queryParams);
        this->flatIndexGuard.unlock_shared();

        // If the query failed (currently only on timeout), return the error code and the partial
        // results.
        if (flat_results.code != VecSim_QueryResult_OK) {
            return flat_results;
        }

        // Lock the main index and query it.
        this->mainIndexGuard.lock_shared();
        auto main_results = this->backendIndex->rangeQuery(queryBlob, radius, queryParams);
        this->mainIndexGuard.unlock_shared();

        // Merge the results and return, avoiding duplicates.
        // At this point, the return code of the FLAT index is OK, and the return code of the MAIN
        // index is either OK or TIMEOUT. Make sure to return the return code of the MAIN index.
        if (BY_SCORE == order) {
            sort_results_by_score_then_id(main_results);
            sort_results_by_score_then_id(flat_results);

            // Keep the return code of the main index.
            auto code = main_results.code;

            // Merge the sorted results with no limit (all the results are valid).
            VecSimQueryResult_List ret;
            if (this->backendIndex->isMultiValue()) {
                ret = merge_result_lists<true>(main_results, flat_results, -1);
            } else {
                ret = merge_result_lists<false>(main_results, flat_results, -1);
            }
            // Restore the return code and return.
            ret.code = code;
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

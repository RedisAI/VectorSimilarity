#pragma once

#include "vec_sim_index.h"
#include "algorithms/brute_force/brute_force.h"
#include "VecSim/batch_iterator.h"
#include "VecSim/utils/merge_results.h"

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

    void submitSingleJob(AsyncJob *job) {
        auto **jobs = array_new<AsyncJob *>(1);
        jobs = array_append(jobs, job);
        this->SubmitJobsToQueue(this->jobQueue, (AsyncJob **)jobs, 1, this->jobQueueCtx);
        array_free(jobs);
    }

public:
    VecSimTieredIndex(VecSimIndexAbstract<DistType> *backendIndex_,
                      BruteForceIndex<DataType, DistType> *frontendIndex_,
                      TieredIndexParams tieredParams, std::shared_ptr<VecSimAllocator> allocator)
        : VecSimIndexInterface(allocator), backendIndex(backendIndex_),
          frontendIndex(frontendIndex_), jobQueue(tieredParams.jobQueue),
          jobQueueCtx(tieredParams.jobQueueCtx), SubmitJobsToQueue(tieredParams.submitCb),
          memoryCtx(tieredParams.memoryCtx), UpdateIndexMemory(tieredParams.UpdateMemCb) {}

    virtual ~VecSimTieredIndex() {
        VecSimIndex_Free(backendIndex);
        VecSimIndex_Free(frontendIndex);
    }

    VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                     VecSimQueryParams *queryParams) override;

    virtual inline int64_t getAllocationSize() const override {
        return this->allocator->getAllocationSize() + this->backendIndex->getAllocationSize() +
               this->frontendIndex->getAllocationSize();
    }

    virtual VecSimIndexInfo info() const override;
    virtual VecSimInfoIterator *infoIterator() const override;
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
VecSimIndexInfo VecSimTieredIndex<DataType, DistType>::info() const {
    VecSimIndexInfo info;
    info.algo = VecSimAlgo_TIERED;
    info.commonInfo.indexLabelCount = this->indexLabelCount();
    info.commonInfo.indexSize = this->indexSize();
    info.commonInfo.memory = this->getAllocationSize();
    info.commonInfo.isMulti = this->backendIndex->isMultiValue();
    VecSimIndexInfo backendInfo = this->backendIndex->info();

    info.tieredInfo.backendAlgo = backendInfo.algo;
    switch (backendInfo.algo) {
        case VecSimAlgo_HNSWLIB:
            info.tieredInfo.backendInfo.hnswInfo = backendInfo.hnswInfo;
            break;
        default:
            assert(false && "Unsupported backend algorithm");
    }

    info.tieredInfo.backendCommonInfo = backendInfo.commonInfo;
    VecSimIndexInfo frontendInfo = this->frontendIndex->info();
    // For now, this is hard coded to FLAT
    info.tieredInfo.frontendCommonInfo = frontendInfo.commonInfo;
    info.tieredInfo.bfInfo = frontendInfo.bfInfo;

    info.tieredInfo.backgroundIndexing = this->frontendIndex->indexSize() > 0;
    info.tieredInfo.management_layer_memory = this->allocator->getAllocationSize();
    return info;
}
template <typename DataType, typename DistType>
 VecSimInfoIterator *VecSimTieredIndex<DataType, DistType>::infoIterator() const {
    return NULL;
 };

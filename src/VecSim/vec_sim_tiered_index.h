#pragma once

#include "vec_sim_index.h"
#include "algorithms/brute_force/brute_force.h"

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
    BruteForceIndex<DataType, DistType> *flatBuffer;
    VecSimIndexAbstract<DistType> *index;

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
    VecSimTieredIndex(VecSimIndexAbstract<DistType> *index_, TieredIndexParams tieredParams)
        : VecSimIndexInterface(index_->getAllocator()), index(index_),
          jobQueue(tieredParams.jobQueue), jobQueueCtx(tieredParams.jobQueueCtx),
          SubmitJobsToQueue(tieredParams.submitCb), memoryCtx(tieredParams.memoryCtx),
          UpdateIndexMemory(tieredParams.UpdateMemCb) {
        BFParams bf_params = {.type = index_->getType(),
                              .dim = index_->getDim(),
                              .metric = index_->getMetric(),
                              .multi = index_->isMultiValue()};
        flatBuffer = static_cast<BruteForceIndex<DataType, DistType> *>(
            BruteForceFactory::NewIndex(&bf_params, index->getAllocator()));
    }
    ~VecSimTieredIndex() {
        VecSimIndex_Free(index);
        VecSimIndex_Free(flatBuffer);
    }

    VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                     VecSimQueryParams *queryParams) override;
};

template <typename DataType, typename DistType>
VecSimQueryResult_List
VecSimTieredIndex<DataType, DistType>::topKQuery(const void *queryBlob, size_t k,
                                                 VecSimQueryParams *queryParams) {
    this->flatIndexGuard.lock_shared();

    // If the flat buffer is empty, we can simply query the main index.
    if (this->flatBuffer->indexSize() == 0) {
        // Release the flat lock and acquire the main lock.
        this->flatIndexGuard.unlock_shared();

        // Simply query the main index and return the results while holding the lock.
        this->mainIndexGuard.lock_shared();
        auto res = this->index->topKQuery(queryBlob, k, queryParams);
        this->mainIndexGuard.unlock_shared();

        return res;
    } else {
        // No luck... first query the flat buffer and release the lock.
        auto flat_results = this->flatBuffer->topKQuery(queryBlob, k, queryParams);
        this->flatIndexGuard.unlock_shared();

        // If the query failed (currently only on timeout), return the error code.
        if (flat_results.code != VecSim_QueryResult_OK) {
            assert(flat_results.results == nullptr);
            return flat_results;
        }

        // Lock the main index and query it.
        this->mainIndexGuard.lock_shared();
        auto main_results = this->index->topKQuery(queryBlob, k, queryParams);
        this->mainIndexGuard.unlock_shared();

        // If the query failed (currently only on timeout), return the error code.
        if (main_results.code != VecSim_QueryResult_OK) {
            // Free the flat results.
            VecSimQueryResult_Free(flat_results);

            assert(main_results.results == nullptr);
            return main_results;
        }

        // Merge the results and return, avoiding duplicates.
        if (this->index->isMultiValue()) {
            return merge_results<true>(main_results, flat_results, k);
        } else {
            return merge_results<false>(main_results, flat_results, k);
        }
    }
}

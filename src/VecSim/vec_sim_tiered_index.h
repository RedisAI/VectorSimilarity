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
    double getDistanceFrom(labelType label, const void *blob) const override;
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

// `getDistanceFrom` returns the minimum distance between the given blob and the vector with the
// given label. If the label doesn't exist, the distance will be NaN.
// Therefore, it's better to just call `getDistanceFrom` on both indexes and return the minimum
// instead of checking if the label exists in each index. We first try to get the distance from the
// flat buffer, as vectors in the buffer might move to the Main while we're "between" the locks.
// Behavior for single (regular) index:
// 1. label doesn't exist in both indexes - return NaN
// 2. label exists in one of the indexes only - return the distance from that index (which is valid)
// 3. label exists in both indexes - return the value from the flat buffer (which is valid and equal
//    to the value from the Main index), saving us from locking the Main index.
// Behavior for multi index:
// 1. label doesn't exist in both indexes - return NaN
// 2. label exists in one of the indexes only - return the distance from that index (which is valid)
// 3. label exists in both indexes - we may have some of the vectors with the same label in the flat
//    buffer only and some in the Main index only (and maybe temporal duplications).
//    So, we get the distance from both indexes and return the minimum.
template <typename DataType, typename DistType>
double VecSimTieredIndex<DataType, DistType>::getDistanceFrom(labelType label,
                                                              const void *blob) const {
    // Try to get the distance from the flat buffer.
    // If the label doesn't exist, the distance will be NaN.
    this->flatIndexGuard.lock_shared();
    auto flat_dist = this->flatBuffer->getDistanceFrom(label, blob);
    this->flatIndexGuard.unlock_shared();

    // Optimization. TODO: consider having different implementations for single and multi indexes,
    // to avoid checking the index type on every query.
    if (!this->index->isMultiValue() && !std::isnan(flat_dist)) {
        // If the index is single value, and we got a valid distance from the flat buffer,
        // we can return the distance without querying the Main index.
        return flat_dist;
    }

    // Try to get the distance from the Main index.
    this->mainIndexGuard.lock_shared();
    auto main_dist = this->index->getDistanceFrom(label, blob);
    this->mainIndexGuard.unlock_shared();

    // Return the minimum distance that is not NaN.
    return std::fmin(flat_dist, main_dist);
}

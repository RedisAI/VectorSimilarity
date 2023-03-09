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
};

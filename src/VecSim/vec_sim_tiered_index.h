#pragma once

#include "vec_sim_index.h"
#include "algorithms/brute_force/brute_force.h"

#include <shared_mutex>

template <typename DataType, typename DistType>
class VecSimTieredIndex : public VecSimIndexInterface {
protected:
    BruteForceIndex<DataType, DistType> *tempFlat;
    VecSimIndexAbstract<DistType> *index;

    void *jobQueue;
    SubmitCB SubmitJobsToQueue;

    void *memoryCtx;
    UpdateMemoryCB UpdateIndexMemory;

    // Consider putting these in the derived class instead. Also - see if we should use
    // std::shared_mutex
    std::shared_mutex flatIndexGuard;
    std::shared_mutex mainIndexGuard;

public:
    VecSimTieredIndex(VecSimIndexAbstract<DistType> *index_, void *job_queue_, SubmitCB submitCb,
                      void *memory_ctx, UpdateMemoryCB UpdateMemCb)
        : VecSimIndexInterface(index_->getAllocator()), index(index_), jobQueue(job_queue_),
          SubmitJobsToQueue(submitCb), memoryCtx(memory_ctx), UpdateIndexMemory(UpdateMemCb) {
        BFParams bf_params = {.type = index_->getType(),
                              .dim = index_->getDim(),
                              .metric = index_->getMetric(),
                              .multi = index_->isMultiValue()};
        tempFlat = static_cast<BruteForceIndex<DataType, DistType> *>(
            BruteForceFactory::NewIndex(&bf_params, index->getAllocator()));
    }
    ~VecSimTieredIndex() {
        VecSimIndex_Free(index);
        VecSimIndex_Free(tempFlat);
    }
};

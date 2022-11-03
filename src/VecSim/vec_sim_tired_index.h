#pragma once

#include "vec_sim_index.h"
#include "algorithms/brute_force/brute_force.h"

#include <shared_mutex>

using SubmitCB = int (*) (void *job_queue, void **jobs);
using UpdateMemoryCB = int (*) (void *memory_ctx, size_t memory);

template <typename DataType, typename DistType>
class VecSimTieredIndex {
protected:
	BruteForceIndex<DataType, DistType> tempFlat;
	VecSimIndexAbstract<DistType> *index;
	void *jobQueue;
	SubmitCB SubmitJobsToQueue;

	void *memoryCtx;
	UpdateMemoryCB UpdateIndexMemory;

	// Consider putting these in the derived class instead. Also - see if we should use std::shared_mutex
	std::shared_timed_mutex flatIndexGuard;
	std::shared_timed_mutex mainIndexGuard;

public:
	VecSimTieredIndex(VecSimParams bf_params, VecSimIndexAbstract<DistType> *index_, void *job_queue_,
	                  SubmitCB submitCb, void *memory_ctx, UpdateMemoryCB UpdateMemCb) :
					  index(index_), jobQueue(job_queue_), SubmitJobsToQueue(submitCb), memoryCtx(memory_ctx),
					  UpdateIndexMemory(UpdateMemCb) {
		tempFlat = BruteForceFactory::NewIndex(&bf_params, index->getAllocator());
	}
	~VecSimTieredIndex() {
		VecSimIndex_Free(index);
	}
};
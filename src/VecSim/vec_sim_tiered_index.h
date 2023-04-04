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

// Forward declaration of the batch iterator.
template <typename DataType, typename DistType>
class TieredIndex_BatchIterator;

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

    friend class TieredIndex_BatchIterator<DataType, DistType>;

public:
    VecSimTieredIndex(VecSimIndexAbstract<DistType> *backendIndex_,
                      BruteForceIndex<DataType, DistType> *frontendIndex_,
                      TieredIndexParams tieredParams)
        : VecSimIndexInterface(backendIndex_->getAllocator()), backendIndex(backendIndex_),
          frontendIndex(frontendIndex_), jobQueue(tieredParams.jobQueue),
          jobQueueCtx(tieredParams.jobQueueCtx), SubmitJobsToQueue(tieredParams.submitCb),
          memoryCtx(tieredParams.memoryCtx), UpdateIndexMemory(tieredParams.UpdateMemCb) {}

    virtual ~VecSimTieredIndex() {
        VecSimIndex_Free(backendIndex);
        VecSimIndex_Free(frontendIndex);
    }

    VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                     VecSimQueryParams *queryParams) override;
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        return new (this->allocator) TieredIndex_BatchIterator<DataType, DistType>(
            queryBlob, this, queryParams, this->allocator);
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType, typename DistType>
class TieredIndex_BatchIterator : public VecSimBatchIterator {
private:
    const VecSimTieredIndex<DataType, DistType> *index;
    bool holding_main_lock;

    VecSimQueryResult_List flat_results;
    VecSimQueryResult_List main_results;

    VecSimBatchIterator *flat_iterator;
    VecSimBatchIterator *main_iterator;

    vecsim_stl::unordered_set<labelType> flat_results_set;

private:
    inline VecSimQueryResult_List get_current_batch(size_t n_res);
    inline void filter_irrelevant_results();

public:
    TieredIndex_BatchIterator(const void *query_vector,
                              const VecSimTieredIndex<DataType, DistType> *index,
                              VecSimQueryParams *queryParams,
                              std::shared_ptr<VecSimAllocator> allocator);

    ~TieredIndex_BatchIterator();

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;
};

/******************** Ctor / Dtor *****************/

template <typename DataType, typename DistType>
TieredIndex_BatchIterator<DataType, DistType>::TieredIndex_BatchIterator(
    const void *query_vector, const VecSimTieredIndex<DataType, DistType> *index,
    VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimBatchIterator(nullptr, queryParams ? queryParams->timeoutCtx : nullptr,
                          std::move(allocator)),
      index(index), holding_main_lock(false),
      flat_iterator(this->index->flatBuffer->newBatchIterator(query_vector, queryParams)),
      main_iterator(this->index->index->newBatchIterator(query_vector, queryParams)),
      flat_results_set(this->allocator) {}

template <typename DataType, typename DistType>
TieredIndex_BatchIterator<DataType, DistType>::~TieredIndex_BatchIterator() {
    delete this->flat_iterator;
    delete this->main_iterator;

    VecSimQueryResult_Free(this->flat_results);
    VecSimQueryResult_Free(this->main_results);

    if (this->holding_main_lock) {
        this->index->mainIndexGuard.unlock_shared();
    }
}

/******************** Implementation **************/

template <typename DataType, typename DistType>
VecSimQueryResult_List
TieredIndex_BatchIterator<DataType, DistType>::getNextResults(size_t n_res,
                                                              VecSimQueryResult_Order order) {

    if (this->getResultsCount() == 0) {
        // First call to getNextResults. The call to the BF iterator will include calculating all
        // the distances and access the BF index. We take the lock on this call.
        this->index->flatIndexGuard.lock_shared();
        this->flat_results = this->flat_iterator->getNextResults(n_res, BY_SCORE);
        this->index->flatIndexGuard.unlock_shared();
        // This is also the only time `getNextResults` on the BF iterator can fail.
        if (VecSim_OK != flat_results.code) {
            return flat_results;
        }
        // We also take the lock on the main index on the first call to getNextResults, and we hold
        // it until the iterator is depleted or freed.
        this->index->mainIndexGuard.lock_shared();
        this->holding_main_lock = true;
        this->main_results = this->main_iterator->getNextResults(n_res, BY_SCORE);
    } else {
        if (VecSimQueryResult_Len(this->flat_results) < n_res &&
            !this->flat_iterator->isDepleted()) {
            auto tail = this->flat_iterator->getNextResults(
                n_res - VecSimQueryResult_Len(this->flat_results), BY_SCORE);
            concat_results(this->flat_results, tail);
            VecSimQueryResult_Free(tail);
        }
        auto code = VecSim_QueryResult_OK;
        while (VecSimQueryResult_Len(this->main_results) < n_res &&
               !this->main_iterator->isDepleted() && code == VecSim_OK) {
            auto tail = this->main_iterator->getNextResults(
                n_res - VecSimQueryResult_Len(this->main_results), BY_SCORE);
            code = tail.code; // Set the main_results code to the last `getNextResults` code.
            // New batch may contain better results than the previous batch, so we need to merge
            this->main_results = merge_result_lists<false>(this->main_results, tail, n_res);
            this->main_results.code = code;
            filter_irrelevant_results();
        }
    }

    if (VecSim_OK != main_results.code) {
        return {NULL, main_results.code};
    }

    auto batch = get_current_batch(n_res);
    if (order == BY_ID) {
        sort_results_by_id(batch);
    }
    size_t batch_len = VecSimQueryResult_Len(batch);
    this->updateResultsCount(batch_len);
    if (batch_len < n_res) {
        this->index->mainIndexGuard.unlock_shared();
        this->holding_main_lock = false;
    }
    return batch;
}

template <typename DataType, typename DistType>
bool TieredIndex_BatchIterator<DataType, DistType>::isDepleted() {
    return this->flat_iterator->isDepleted() && VecSimQueryResult_Len(this->flat_results) == 0 &&
           this->main_iterator->isDepleted() && VecSimQueryResult_Len(this->main_results) == 0;
}

template <typename DataType, typename DistType>
void TieredIndex_BatchIterator<DataType, DistType>::reset() {
    if (this->holding_main_lock) {
        this->index->mainIndexGuard.unlock_shared();
    }
    this->resetResultsCount();
    this->flat_iterator->reset();
    this->main_iterator->reset();
    VecSimQueryResult_Free(this->flat_results);
    VecSimQueryResult_Free(this->main_results);
    flat_results_set.clear();
}

/****************** Helper Functions **************/

template <typename DataType, typename DistType>
VecSimQueryResult_List
TieredIndex_BatchIterator<DataType, DistType>::get_current_batch(size_t n_res) {
    // Set pointers
    auto bf_res = this->flat_results.results;
    auto main_res = this->main_results.results;
    const auto bf_end = bf_res + VecSimQueryResult_Len(this->flat_results);
    const auto main_end = main_res + VecSimQueryResult_Len(this->main_results);

    // Merge results
    VecSimQueryResult *batch_res;
    if (this->index->index->isMultiValue()) {
        batch_res = merge_results<true>(main_res, main_end, bf_res, bf_end, n_res);
    } else {
        batch_res = merge_results<false>(main_res, main_end, bf_res, bf_end, n_res);
    }

    // Update set of results returned from FLAT index
    for (auto it = this->flat_results.results; it != bf_res; ++it) {
        this->flat_results_set.insert(it->id);
    }

    // Update results
    array_pop_front_n(this->flat_results.results, bf_end - bf_res);
    array_pop_front_n(this->main_results.results, main_end - main_res);

    // Return current batch
    return {batch_res, VecSim_QueryResult_OK};
}

template <typename DataType, typename DistType>
void TieredIndex_BatchIterator<DataType, DistType>::filter_irrelevant_results() {
    // Filter out results that were already returned from the FLAT index
    auto it = this->main_results.results;
    auto end = it + VecSimQueryResult_Len(this->main_results);
    // Skip results that not returned from the FLAT index
    while (it != end && this->flat_results_set.count(it->id) == 0) {
        ++it;
    }
    // If none of the results were returned from the FLAT index, return
    if (it == end) {
        return;
    }
    // Mark the current result as the first result to be filtered
    auto cur_end = it;
    ++it;
    // "Append" all results that were not returned from the FLAT index
    while (it != end) {
        if (this->flat_results_set.count(it->id) == 0) {
            *cur_end = *it;
            ++cur_end;
        }
        ++it;
    }
    // Update number of results
    array_hdr(this->main_results.results)->len = cur_end - this->main_results.results;
}

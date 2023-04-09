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
    class TieredIndex_BatchIterator : public VecSimBatchIterator {
    private:
        const VecSimTieredIndex<DataType, DistType> *index;
        bool holding_main_lock;

        VecSimQueryResult_List flat_results;
        VecSimQueryResult_List main_results;

        VecSimBatchIterator *flat_iterator;
        VecSimBatchIterator *main_iterator;

        // On single value indices, this set holds the IDs of the results that were returned from
        // the flat buffer.
        // On multi value indices, this set holds the IDs of all the results that were returned.
        vecsim_stl::unordered_set<labelType> returned_results_set;

    private:
        template <bool isMultiValue>
        inline VecSimQueryResult_List get_current_batch(size_t n_res);
        inline void filter_irrelevant_results(VecSimQueryResult_List &);

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
        return new (this->allocator)
            TieredIndex_BatchIterator(queryBlob, this, queryParams, this->allocator);
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

////////////////////////////////////////////////////////////////////////////////////////////////////
//  TieredIndex_BatchIterator                                                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////

/******************** Ctor / Dtor *****************/

template <typename DataType, typename DistType>
VecSimTieredIndex<DataType, DistType>::TieredIndex_BatchIterator::TieredIndex_BatchIterator(
    const void *query_vector, const VecSimTieredIndex<DataType, DistType> *index,
    VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimBatchIterator(nullptr, queryParams ? queryParams->timeoutCtx : nullptr,
                          std::move(allocator)),
      index(index), holding_main_lock(false), flat_results({0}), main_results({0}),
      flat_iterator(this->index->flatBuffer->newBatchIterator(query_vector, queryParams)),
      main_iterator(this->index->index->newBatchIterator(query_vector, queryParams)),
      returned_results_set(this->allocator) {}

template <typename DataType, typename DistType>
VecSimTieredIndex<DataType, DistType>::TieredIndex_BatchIterator::~TieredIndex_BatchIterator() {
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
VecSimTieredIndex<DataType, DistType>::TieredIndex_BatchIterator::getNextResults(
    size_t n_res, VecSimQueryResult_Order order) {

    const bool isMulti = this->index->index->isMultiValue();

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
        while (VecSimQueryResult_Len(this->flat_results) < n_res &&
               !this->flat_iterator->isDepleted()) {
            auto tail = this->flat_iterator->getNextResults(
                n_res - VecSimQueryResult_Len(this->flat_results), BY_SCORE);
            concat_results(this->flat_results, tail);
            VecSimQueryResult_Free(tail);

            if (!isMulti) {
                // On single-value indexes, duplicates will never appear in the main results before
                // they appear in the flat results (at the same time or later if the approximation
                // misses) so we don't need to try and filter the flat results (and recheck
                // conditions).
                break;
            } else {
                // On multi-value indexes, the flat results may contain results that are already
                // returned from the main results. We need to filter them out.
                filter_irrelevant_results(this->flat_results);
            }
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
            filter_irrelevant_results(this->main_results);
        }
    }

    if (VecSim_OK != main_results.code) {
        return {NULL, main_results.code};
    }

    VecSimQueryResult_List batch;
    if (isMulti)
        batch = get_current_batch<true>(n_res);
    else
        batch = get_current_batch<false>(n_res);

    if (order == BY_ID) {
        sort_results_by_id(batch);
    }
    size_t batch_len = VecSimQueryResult_Len(batch);
    this->updateResultsCount(batch_len);
    if (batch_len < n_res /* && this->holding_main_lock */) {
        assert(this->holding_main_lock); // TODO: verify if this is always true
        this->index->mainIndexGuard.unlock_shared();
        this->holding_main_lock = false;
    }
    return batch;
}

template <typename DataType, typename DistType>
bool VecSimTieredIndex<DataType, DistType>::TieredIndex_BatchIterator::isDepleted() {
    return this->flat_iterator->isDepleted() && VecSimQueryResult_Len(this->flat_results) == 0 &&
           this->main_iterator->isDepleted() && VecSimQueryResult_Len(this->main_results) == 0;
}

template <typename DataType, typename DistType>
void VecSimTieredIndex<DataType, DistType>::TieredIndex_BatchIterator::reset() {
    if (this->holding_main_lock) {
        this->index->mainIndexGuard.unlock_shared();
    }
    this->resetResultsCount();
    this->flat_iterator->reset();
    this->main_iterator->reset();
    VecSimQueryResult_Free(this->flat_results);
    VecSimQueryResult_Free(this->main_results);
    returned_results_set.clear();
}

/****************** Helper Functions **************/

template <typename DataType, typename DistType>
template <bool isMultiValue>
VecSimQueryResult_List
VecSimTieredIndex<DataType, DistType>::TieredIndex_BatchIterator::get_current_batch(size_t n_res) {
    // Set pointers
    auto bf_res = this->flat_results.results;
    auto main_res = this->main_results.results;
    const auto bf_end = bf_res + VecSimQueryResult_Len(this->flat_results);
    const auto main_end = main_res + VecSimQueryResult_Len(this->main_results);

    // Merge results
    VecSimQueryResult *batch_res;
    if (isMultiValue) {
        batch_res = merge_results<true>(main_res, main_end, bf_res, bf_end, n_res);
    } else {
        batch_res = merge_results<false>(main_res, main_end, bf_res, bf_end, n_res);
    }

    // If we're on a single-value index, update the set of results returned from the FLAT index
    // before popping them.
    if (!isMultiValue) {
        for (auto it = this->flat_results.results; it != bf_res; ++it) {
            this->returned_results_set.insert(it->id);
        }
    }

    // Update results
    array_pop_front_n(this->flat_results.results, bf_res - this->flat_results.results);
    array_pop_front_n(this->main_results.results, main_res - this->main_results.results);

    // If we're on a multi-value index, update the set of results returned (from `batch_res`), and
    // clean up the results
    if (isMultiValue) {
        // Update set of results returned
        for (size_t i = 0; i < array_len(batch_res); ++i) {
            this->returned_results_set.insert(batch_res[i].id);
        }
        // On multi-value indexes, one (or both) results lists may contain results that are already
        // returned form the other list (with a different score). We need to filter them out.
        filter_irrelevant_results(this->flat_results);
        filter_irrelevant_results(this->main_results);
    }

    // Return current batch
    return {batch_res, VecSim_QueryResult_OK};
}

template <typename DataType, typename DistType>
void VecSimTieredIndex<DataType, DistType>::TieredIndex_BatchIterator::filter_irrelevant_results(
    VecSimQueryResult_List &rl) {
    // Filter out results that were already returned from the FLAT index
    auto it = rl.results;
    auto end = it + VecSimQueryResult_Len(rl);
    // Skip results that not returned yet
    while (it != end && this->returned_results_set.count(it->id) == 0) {
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
        if (this->returned_results_set.count(it->id) == 0) {
            *cur_end = *it;
            ++cur_end;
        }
        ++it;
    }
    // Update number of results
    array_hdr(rl.results)->len = cur_end - rl.results;
}

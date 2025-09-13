/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/vec_sim_tiered_index.h"
#include "hnsw.h"
#include "VecSim/index_factories/hnsw_factory.h"

/**
 * Definition of a job that inserts a new vector from flat into HNSW Index.
 */
struct HNSWInsertJob : public AsyncJob {
    labelType label;
    idType id;

    HNSWInsertJob(std::shared_ptr<VecSimAllocator> allocator, labelType label_, idType id_,
                  JobCallback insertCb, VecSimIndex *index_)
        : AsyncJob(allocator, HNSW_INSERT_VECTOR_JOB, insertCb, index_), label(label_), id(id_) {}
};

/**
 * Definition of a job that swaps last id with a deleted id in HNSW Index after delete operation.
 */
struct HNSWSwapJob : public VecsimBaseObject {
    idType deleted_id;
    std::atomic_int
        pending_repair_jobs_counter; // number of repair jobs left to complete before this job
                                     // is ready to be executed (atomic counter).
    HNSWSwapJob(std::shared_ptr<VecSimAllocator> allocator, idType deletedId)
        : VecsimBaseObject(allocator), deleted_id(deletedId), pending_repair_jobs_counter(0) {}
    void setRepairJobsNum(long num_repair_jobs) { pending_repair_jobs_counter = num_repair_jobs; }
    int atomicDecreasePendingJobsNum() {
        int ret = --pending_repair_jobs_counter;
        assert(pending_repair_jobs_counter >= 0);
        return ret;
    }
};

static const size_t DEFAULT_PENDING_SWAP_JOBS_THRESHOLD = DEFAULT_BLOCK_SIZE;
static const size_t MAX_PENDING_SWAP_JOBS_THRESHOLD = 100000;

/**
 * Definition of a job that repairs a certain node's connection in HNSW Index after delete
 * operation.
 */
struct HNSWRepairJob : public AsyncJob {
    idType node_id;
    unsigned short level;
    vecsim_stl::vector<HNSWSwapJob *> associatedSwapJobs;

    HNSWRepairJob(std::shared_ptr<VecSimAllocator> allocator, idType id_, unsigned short level_,
                  JobCallback repairCb, VecSimIndex *index_, HNSWSwapJob *swapJob)
        : AsyncJob(allocator, HNSW_REPAIR_NODE_CONNECTIONS_JOB, repairCb, index_), node_id(id_),
          level(level_),
          // Insert the first swap job from which this repair job was created.
          associatedSwapJobs(1, swapJob, this->allocator) {}
    // In case that a repair job is required for deleting another neighbor of the node, save a
    // reference to additional swap job.
    void appendAnotherAssociatedSwapJob(HNSWSwapJob *swapJob) {
        associatedSwapJobs.push_back(swapJob);
    }
};

template <typename DataType, typename DistType>
class TieredHNSWIndex : public VecSimTieredIndex<DataType, DistType> {
private:
    /// Mappings from id/label to associated jobs, for invalidating and update ids if necessary.
    // In MULTI, we can have more than one insert job pending per label.
    // **This map is protected with the flat buffer lock**
    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<HNSWInsertJob *>> labelToInsertJobs;
    vecsim_stl::unordered_map<idType, vecsim_stl::vector<HNSWRepairJob *>> idToRepairJobs;
    vecsim_stl::unordered_map<idType, HNSWSwapJob *> idToSwapJob;

    // A mapping to hold invalid jobs, so we can dispose them upon index deletion.
    vecsim_stl::unordered_map<idType, AsyncJob *> invalidJobs;
    idType currInvalidJobId; // A unique arbitrary identifier for accessing invalid jobs
    std::mutex invalidJobsLookupGuard;

    // This threshold is tested upon deleting a label from HNSW, and once the number of deleted
    // vectors reached this limit, we apply swap jobs *only for vectors that has no more pending
    // repair jobs*, and are ready to be removed from the graph.
    size_t pendingSwapJobsThreshold;
    size_t readySwapJobs;

    // Protect the both idToRepairJobs lookup and the pending_repair_jobs_counter for the
    // associated swap jobs.
    std::mutex idToRepairJobsGuard;

    void executeInsertJob(HNSWInsertJob *job);
    void executeRepairJob(HNSWRepairJob *job);

    // To be executed synchronously upon deleting a vector, doesn't require a wrapper. Main HNSW
    // lock is assumed to be held exclusive here.
    void executeSwapJob(idType deleted_id, vecsim_stl::vector<idType> &idsToRemove);

    // Execute the ready swap jobs, run no more than 'maxSwapsToRun' jobs (run all of them for -1).
    void executeReadySwapJobs(size_t maxSwapsToRun = -1);

    // Wrappers static functions to be sent as callbacks upon creating the jobs (since members
    // functions cannot serve as callback, this serve as the "gateway" to the appropriate index).
    static void executeInsertJobWrapper(AsyncJob *job);
    static void executeRepairJobWrapper(AsyncJob *job);

    inline HNSWIndex<DataType, DistType> *getHNSWIndex() const;

    // Helper function for deleting a vector from the flat buffer (after it has already been
    // ingested into HNSW or deleted). This includes removing the corresponding insert job from the
    // label-to-insert-jobs lookup. Also, since deletion a vector triggers swapping of the
    // internal last id with the deleted vector id, here we update the pending insert job(s) for the
    // last id (if needed). This should be called while *flat lock is held* (exclusive lock).
    void updateInsertJobInternalId(idType prev_id, idType new_id, labelType label);

    // Helper function for performing in place mark delete of vector(s) associated with a label
    // and creating the appropriate repair jobs for the effected connections. This should be called
    // while *HNSW shared lock is held* (shared locked).
    int deleteLabelFromHNSW(labelType label);

    // Insert a single vector to HNSW. This can be called in both write modes - insert async and
    // in-place. For the async mode, we have to release the flat index guard that is held for shared
    // ownership (we do it right after we update the HNSW global data and receive the new state).
    template <bool releaseFlatGuard>
    void insertVectorToHNSW(HNSWIndex<DataType, DistType> *hnsw_index, labelType label,
                            const void *blob);

    // Set an insert/repair job as invalid, put the job pointer in the invalid jobs lookup under
    // the current available id, increase it and return it (while holding invalidJobsLookupGuard).
    // Returns the id that the job was stored under (to be set in the job id field).
    idType setAndSaveInvalidJob(AsyncJob *job);

    // Handle deletion of vector inplace considering that async deletion might occurred beforehand.
    int deleteLabelFromHNSWInplace(labelType label);

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_tiered_tests_friends.h"
#endif

public:
    class TieredHNSW_BatchIterator : public VecSimBatchIterator {
    private:
        const TieredHNSWIndex<DataType, DistType> *index;
        VecSimQueryParams *queryParams;

        VecSimQueryResultContainer flat_results;
        VecSimQueryResultContainer hnsw_results;

        VecSimBatchIterator *flat_iterator;
        VecSimBatchIterator *hnsw_iterator;

        // On single value indices, this set holds the IDs of the results that were returned from
        // the flat buffer.
        // On multi value indices, this set holds the IDs of all the results that were returned.
        // The difference between the two cases is that on multi value indices, the same ID can
        // appear in both indexes and results with different scores, and therefore we can't tell in
        // advance when we expect a possibility of a duplicate.
        // On single value indices, a duplicate may appear at the same batch (and we will handle it
        // when merging the results) Or it may appear in a different batches, first from the flat
        // buffer and then from the HNSW, in the cases where a better result if found later in HNSW
        // because of the approximate nature of the algorithm.
        vecsim_stl::unordered_set<labelType> returned_results_set;

    private:
        template <bool isMultiValue>
        inline VecSimQueryReply *compute_current_batch(size_t n_res);
        inline void filter_irrelevant_results(VecSimQueryResultContainer &);

    public:
        TieredHNSW_BatchIterator(const void *query_vector,
                                 const TieredHNSWIndex<DataType, DistType> *index,
                                 VecSimQueryParams *queryParams,
                                 std::shared_ptr<VecSimAllocator> allocator);

        ~TieredHNSW_BatchIterator();

        VecSimQueryReply *getNextResults(size_t n_res, VecSimQueryReply_Order order) override;

        bool isDepleted() override;

        void reset() override;
    };

public:
    TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index,
                    BruteForceIndex<DataType, DistType> *bf_index,
                    const TieredIndexParams &tieredParams,
                    std::shared_ptr<VecSimAllocator> allocator);
    virtual ~TieredHNSWIndex();

    int addVector(const void *blob, labelType label) override;
    int deleteVector(labelType label) override;
    size_t indexSize() const override;
    size_t indexCapacity() const override;
    double getDistanceFrom_Unsafe(labelType label, const void *blob) const override;
    // Do nothing here, each tier (flat buffer and HNSW) should increase capacity for itself when
    // needed.
    VecSimIndexDebugInfo debugInfo() const override;
    VecSimIndexBasicInfo basicInfo() const override;
    VecSimDebugInfoIterator *debugInfoIterator() const override;
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        // The query blob will be processed and copied by the internal indexes's batch iterator.
        return new (this->allocator)
            TieredHNSW_BatchIterator(queryBlob, this, queryParams, this->allocator);
    }
    inline void setLastSearchMode(VecSearchMode mode) override {
        return this->backendIndex->setLastSearchMode(mode);
    }
    void runGC() override {
        // Run no more than pendingSwapJobsThreshold value jobs.
        TIERED_LOG(VecSimCommonStrings::LOG_VERBOSE_STRING,
                   "running asynchronous GC for tiered HNSW index");
        this->executeReadySwapJobs(this->pendingSwapJobsThreshold);
    }
    void acquireSharedLocks() override {
        this->flatIndexGuard.lock_shared();
        this->mainIndexGuard.lock_shared();
        this->getHNSWIndex()->lockSharedIndexDataGuard();
    }

    void releaseSharedLocks() override {
        this->flatIndexGuard.unlock_shared();
        this->mainIndexGuard.unlock_shared();
        this->getHNSWIndex()->unlockSharedIndexDataGuard();
    }

    VecSimDebugCommandCode getHNSWElementNeighbors(size_t label, int ***neighborsData) {
        this->mainIndexGuard.lock_shared();
        auto res = this->getHNSWIndex()->getHNSWElementNeighbors(label, neighborsData);
        this->mainIndexGuard.unlock_shared();
        return res;
    }

#ifdef BUILD_TESTS
    void getDataByLabel(labelType label, std::vector<std::vector<DataType>> &vectors_output) const;
#endif
};

/**
 ******************************* Implementation **************************
 */

/* Helper methods */
template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeInsertJobWrapper(AsyncJob *job) {
    auto *insert_job = reinterpret_cast<HNSWInsertJob *>(job);
    auto *job_index = reinterpret_cast<TieredHNSWIndex<DataType, DistType> *>(insert_job->index);
    job_index->executeInsertJob(insert_job);
    delete job;
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeRepairJobWrapper(AsyncJob *job) {
    auto *repair_job = reinterpret_cast<HNSWRepairJob *>(job);
    auto *job_index = reinterpret_cast<TieredHNSWIndex<DataType, DistType> *>(repair_job->index);
    job_index->executeRepairJob(repair_job);
    delete job;
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeSwapJob(idType deleted_id,
                                                         vecsim_stl::vector<idType> &idsToRemove) {
    // Get the id that was last and was had been swapped with the job's deleted id.
    idType prev_last_id = this->getHNSWIndex()->indexSize();

    // Invalidate repair jobs for the disposed id (if exist), and update the associated swap jobs.
    if (idToRepairJobs.find(deleted_id) != idToRepairJobs.end()) {
        for (auto &job_it : idToRepairJobs.at(deleted_id)) {
            job_it->node_id = this->setAndSaveInvalidJob(job_it);
            for (auto &swap_job_it : job_it->associatedSwapJobs) {
                if (swap_job_it->atomicDecreasePendingJobsNum() == 0) {
                    readySwapJobs++;
                }
            }
        }
        idToRepairJobs.erase(deleted_id);
    }
    // Swap the ids in the pending jobs for the current last id (if exist).
    if (idToRepairJobs.find(prev_last_id) != idToRepairJobs.end()) {
        for (auto &job_it : idToRepairJobs.at(prev_last_id)) {
            job_it->node_id = deleted_id;
        }
        idToRepairJobs.insert({deleted_id, idToRepairJobs.at(prev_last_id)});
        idToRepairJobs.erase(prev_last_id);
    }
    // Update the swap jobs if the last id also needs a swap, otherwise just collect to deleted id
    // to be removed from the swap jobs.
    if (prev_last_id != deleted_id && idToSwapJob.find(prev_last_id) != idToSwapJob.end() &&
        std::find(idsToRemove.begin(), idsToRemove.end(), prev_last_id) == idsToRemove.end()) {
        // Update the curr_last_id pending swap job id after the removal that renamed curr_last_id
        // with the deleted id.
        idsToRemove.push_back(prev_last_id);
        idToSwapJob.at(prev_last_id)->deleted_id = deleted_id;
        // If id was deleted in-place and there is no swap job for it, this will create a new entry
        // in idToSwapJob for the swapped id, otherwise it will update the existing entry.
        idToSwapJob[deleted_id] = idToSwapJob.at(prev_last_id);
    } else {
        idsToRemove.push_back(deleted_id);
    }
}

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType> *TieredHNSWIndex<DataType, DistType>::getHNSWIndex() const {
    return dynamic_cast<HNSWIndex<DataType, DistType> *>(this->backendIndex);
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeReadySwapJobs(size_t maxJobsToRun) {

    // Execute swap jobs - acquire hnsw write lock.
    this->lockMainIndexGuard();
    TIERED_LOG(VecSimCommonStrings::LOG_VERBOSE_STRING,
               "Tiered HNSW index GC: there are %zu ready swap jobs. Start executing %zu swap jobs",
               readySwapJobs, std::min(readySwapJobs, maxJobsToRun));

    vecsim_stl::vector<idType> idsToRemove(this->allocator);
    idsToRemove.reserve(idToSwapJob.size());
    for (auto &it : idToSwapJob) {
        auto *swap_job = it.second;
        if (swap_job->pending_repair_jobs_counter.load() == 0) {
            // Swap job is ready for execution - execute and delete it.
            this->getHNSWIndex()->removeAndSwapMarkDeletedElement(swap_job->deleted_id);
            this->executeSwapJob(swap_job->deleted_id, idsToRemove);
            delete swap_job;
        }
        if (maxJobsToRun > 0 && idsToRemove.size() >= maxJobsToRun) {
            break;
        }
    }
    for (idType id : idsToRemove) {
        idToSwapJob.erase(id);
    }
    readySwapJobs -= idsToRemove.size();
    TIERED_LOG(VecSimCommonStrings::LOG_VERBOSE_STRING,
               "Tiered HNSW index GC: done executing %zu swap jobs", idsToRemove.size());
    this->unlockMainIndexGuard();
}

template <typename DataType, typename DistType>
int TieredHNSWIndex<DataType, DistType>::deleteLabelFromHNSW(labelType label) {
    auto *hnsw_index = getHNSWIndex();
    this->mainIndexGuard.lock_shared();

    // Get the required data about the relevant ids to delete.
    // Internally, this will hold the index data lock.
    auto internal_ids = hnsw_index->markDelete(label);

    for (size_t i = 0; i < internal_ids.size(); i++) {
        idType id = internal_ids[i];
        vecsim_stl::vector<AsyncJob *> repair_jobs(this->allocator);
        auto *swap_job = new (this->allocator) HNSWSwapJob(this->allocator, id);

        // Go over all the deleted element links in every level and create repair jobs.
        auto incomingEdges = hnsw_index->safeCollectAllNodeIncomingNeighbors(id);

        // Protect the id->repair_jobs lookup while we update it with the new jobs.
        this->idToRepairJobsGuard.lock();
        for (pair<idType, unsigned short> &node : incomingEdges) {
            bool repair_job_exists = false;
            HNSWRepairJob *repair_job = nullptr;
            if (idToRepairJobs.find(node.first) != idToRepairJobs.end()) {
                for (auto it : idToRepairJobs.at(node.first)) {
                    if (it->level == node.second) {
                        // There is already an existing pending repair job for this node due to
                        // the deletion of another node - avoid creating another job.
                        repair_job_exists = true;
                        repair_job = it;
                        break;
                    }
                }
            } else {
                // There is no repair jobs at all for this element, create a new array for it.
                idToRepairJobs.insert(
                    {node.first, vecsim_stl::vector<HNSWRepairJob *>(this->allocator)});
            }
            if (repair_job_exists) {
                repair_job->appendAnotherAssociatedSwapJob(swap_job);
            } else {
                repair_job =
                    new (this->allocator) HNSWRepairJob(this->allocator, node.first, node.second,
                                                        executeRepairJobWrapper, this, swap_job);
                repair_jobs.emplace_back(repair_job);
                idToRepairJobs.at(node.first).push_back(repair_job);
            }
        }
        swap_job->setRepairJobsNum(incomingEdges.size());
        if (incomingEdges.size() == 0) {
            // No pending repair jobs, so swap jobs is ready from the beginning.
            readySwapJobs++;
        }
        this->idToRepairJobsGuard.unlock();

        this->submitJobs(repair_jobs);
        // Insert the swap job into the swap jobs lookup (for fast update in case that the
        // node id is changed due to swap job).
        assert(idToSwapJob.find(id) == idToSwapJob.end());
        idToSwapJob[id] = swap_job;
    }
    this->mainIndexGuard.unlock_shared();
    return internal_ids.size();
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::updateInsertJobInternalId(idType prev_id, idType new_id,
                                                                    labelType label) {
    // Update the pending job id, due to a swap that was caused after the removal of new_id.
    assert(new_id != INVALID_ID && prev_id != INVALID_ID);
    auto it = this->labelToInsertJobs.find(label);
    if (it != this->labelToInsertJobs.end()) {
        // There is a pending job for the label of the swapped last id - update its id.
        for (HNSWInsertJob *job_it : it->second) {
            if (job_it->id == prev_id) {
                job_it->id = new_id;
            }
        }
    }
}

template <typename DataType, typename DistType>
template <bool releaseFlatGuard>
void TieredHNSWIndex<DataType, DistType>::insertVectorToHNSW(
    HNSWIndex<DataType, DistType> *hnsw_index, labelType label, const void *blob) {

    // Preprocess for storage and indexing in the hnsw index
    ProcessedBlobs processed_blobs = hnsw_index->preprocess(blob);
    const void *processed_storage_blob = processed_blobs.getStorageBlob();
    const void *processed_for_index = processed_blobs.getQueryBlob();

    // Acquire the index data lock, so we know what is the exact index size at this time. Acquire
    // the main r/w lock before to avoid deadlocks.
    this->mainIndexGuard.lock_shared();
    hnsw_index->lockIndexDataGuard();
    // Check if resizing is needed for HNSW index (requires write lock).
    if (hnsw_index->isCapacityFull()) {
        // Release the inner HNSW data lock before we re-acquire the global HNSW lock.
        this->mainIndexGuard.unlock_shared();
        hnsw_index->unlockIndexDataGuard();
        this->lockMainIndexGuard();
        hnsw_index->lockIndexDataGuard();

        // Hold the index data lock while we store the new element. If the new node's max level is
        // higher than the current one, hold the lock through the entire insertion to ensure that
        // graph scans will not occur, as they will try access the entry point's neighbors.
        // If an index resize is still needed, `storeNewElement` will perform it. This is OK since
        // we hold the main index lock for exclusive access.
        auto state = hnsw_index->storeNewElement(label, processed_storage_blob);
        if constexpr (releaseFlatGuard) {
            this->flatIndexGuard.unlock_shared();
        }

        // If we're still holding the index data guard, we cannot take the main index lock for
        // shared ownership as it may cause deadlocks, and we also cannot release the main index
        // lock between, since we cannot allow swap jobs to happen, as they will make the
        // saved state invalid. Hence, we insert the vector with the current exclusive lock held.
        if (state.elementMaxLevel <= state.currMaxLevel) {
            hnsw_index->unlockIndexDataGuard();
        }
        // Take the vector from the flat buffer and insert it to HNSW (overwrite should not occur).
        hnsw_index->indexVector(processed_for_index, label, state);
        if (state.elementMaxLevel > state.currMaxLevel) {
            hnsw_index->unlockIndexDataGuard();
        }
        this->unlockMainIndexGuard();
    } else {
        // Do the same as above except for changing the capacity, but with *shared* lock held:
        // Hold the index data lock while we store the new element. If the new node's max level is
        // higher than the current one, hold the lock through the entire insertion to ensure that
        // graph scans will not occur, as they will try access the entry point's neighbors.
        // At this point we are certain that the index has enough capacity for the new element, and
        // this call will not resize the index.
        auto state = hnsw_index->storeNewElement(label, processed_storage_blob);
        if constexpr (releaseFlatGuard) {
            this->flatIndexGuard.unlock_shared();
        }

        if (state.elementMaxLevel <= state.currMaxLevel) {
            hnsw_index->unlockIndexDataGuard();
        }
        // Take the vector from the flat buffer and insert it to HNSW (overwrite should not occur).
        hnsw_index->indexVector(processed_for_index, label, state);
        if (state.elementMaxLevel > state.currMaxLevel) {
            hnsw_index->unlockIndexDataGuard();
        }
        this->mainIndexGuard.unlock_shared();
    }
}

template <typename DataType, typename DistType>
idType TieredHNSWIndex<DataType, DistType>::setAndSaveInvalidJob(AsyncJob *job) {
    this->invalidJobsLookupGuard.lock();
    job->isValid = false;
    idType curInvalidId = currInvalidJobId++;
    this->invalidJobs.insert({curInvalidId, job});
    this->invalidJobsLookupGuard.unlock();
    return curInvalidId;
}

template <typename DataType, typename DistType>
int TieredHNSWIndex<DataType, DistType>::deleteLabelFromHNSWInplace(labelType label) {
    auto *hnsw_index = this->getHNSWIndex();

    auto ids = hnsw_index->getElementIds(label);
    // Dispose pending repair and swap jobs for the removed ids.
    vecsim_stl::vector<idType> idsToRemove(this->allocator);
    idsToRemove.reserve(ids.size());
    readySwapJobs += ids.size(); // account for the current ids that are going to be removed.
    for (size_t id_ind = 0; id_ind < ids.size(); id_ind++) {
        // Get the id in every iteration, since the ids can be swapped in every iteration.
        idType id = hnsw_index->getElementIds(label).at(id_ind);
        hnsw_index->removeVectorInPlace(id);
        this->executeSwapJob(id, idsToRemove);
    }
    hnsw_index->removeLabel(label);
    for (idType id : idsToRemove) {
        idToSwapJob.erase(id);
    }
    readySwapJobs -= idsToRemove.size();
    return ids.size();
}

/******************** Job's callbacks **********************************/
template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeInsertJob(HNSWInsertJob *job) {
    // Note that accessing the job fields should occur with flat index guard held (here and later).
    this->flatIndexGuard.lock_shared();
    if (!job->isValid) {
        this->flatIndexGuard.unlock_shared();
        // Job has been invalidated in the meantime - nothing to execute, and remove it from the
        // lookup.
        this->invalidJobsLookupGuard.lock();
        this->invalidJobs.erase(job->id);
        this->invalidJobsLookupGuard.unlock();
        return;
    }

    HNSWIndex<DataType, DistType> *hnsw_index = this->getHNSWIndex();
    // Copy the vector blob from the flat buffer, so we can release the flat lock while we are
    // indexing the vector into HNSW index.
    size_t data_size = this->frontendIndex->getDataSize();
    auto blob_copy = this->getAllocator()->allocate_unique(data_size);
    // Assuming the size of the blob stored in the frontend index matches the size of the blob
    // stored in the HNSW index.
    memcpy(blob_copy.get(), this->frontendIndex->getDataByInternalId(job->id), data_size);

    this->insertVectorToHNSW<true>(hnsw_index, job->label, blob_copy.get());

    // Remove the vector and the insert job from the flat buffer.
    this->flatIndexGuard.lock();
    // The job might have been invalidated due to overwrite in the meantime. In this case,
    // it was already deleted and the job has been evicted. Otherwise, we need to do it now.
    if (job->isValid) {
        // Remove the job pointer from the labelToInsertJobs mapping.
        auto &jobs = labelToInsertJobs.at(job->label);
        for (size_t i = 0; i < jobs.size(); i++) {
            if (jobs[i]->id == job->id) {
                jobs.erase(jobs.begin() + (long)i);
                break;
            }
        }
        if (labelToInsertJobs.at(job->label).empty()) {
            labelToInsertJobs.erase(job->label);
        }
        // Remove the vector from the flat buffer. This may cause the last vector id to swap with
        // the deleted id. Hold the label for the last id, so we can later on update its
        // corresponding job id. Note that after calling deleteVectorById, the last id's label
        // shouldn't be available, since it is removed from the lookup.
        labelType last_vec_label =
            this->frontendIndex->getVectorLabel(this->frontendIndex->indexSize() - 1);
        int deleted = this->frontendIndex->deleteVectorById(job->label, job->id);
        if (deleted && job->id != this->frontendIndex->indexSize()) {
            // If the vector removal caused a swap with the last id, update the relevant insert job.
            this->updateInsertJobInternalId(this->frontendIndex->indexSize(), job->id,
                                            last_vec_label);
        }
    } else {
        // Remove the current job from the invalid jobs' lookup, as we are about to delete it now.
        this->invalidJobsLookupGuard.lock();
        this->invalidJobs.erase(job->id);
        this->invalidJobsLookupGuard.unlock();
    }
    this->flatIndexGuard.unlock();
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeRepairJob(HNSWRepairJob *job) {
    // Lock the HNSW shared lock before accessing its internals.
    this->mainIndexGuard.lock_shared();
    if (!job->isValid) {
        this->mainIndexGuard.unlock_shared();
        // The current node has already been removed and disposed.
        this->invalidJobsLookupGuard.lock();
        this->invalidJobs.erase(job->node_id);
        this->invalidJobsLookupGuard.unlock();
        return;
    }
    HNSWIndex<DataType, DistType> *hnsw_index = this->getHNSWIndex();

    // Remove this job pointer from the repair jobs lookup BEFORE it has been executed. Had we done
    // it after executing the repair job, we might have see that there is a pending repair job for
    // this node id upon deleting another neighbor of this node, and we may avoid creating another
    // repair job even though *it has already been executed*.
    this->idToRepairJobsGuard.lock();
    auto &repair_jobs = this->idToRepairJobs.at(job->node_id);
    assert(repair_jobs.size() > 0);
    if (repair_jobs.size() == 1) {
        // This was the only pending repair job for this id.
        this->idToRepairJobs.erase(job->node_id);
    } else {
        // There are more pending jobs for the current id, remove just this job from the pending
        // repair jobs list for this element id by replacing it with the last one (and trim the
        // last job in the list).
        auto it = std::find(repair_jobs.begin(), repair_jobs.end(), job);
        assert(it != repair_jobs.end());
        *it = repair_jobs.back();
        repair_jobs.pop_back();
    }
    for (auto &it : job->associatedSwapJobs) {
        if (it->atomicDecreasePendingJobsNum() == 0) {
            readySwapJobs++;
        }
    }
    this->idToRepairJobsGuard.unlock();

    hnsw_index->repairNodeConnections(job->node_id, job->level);

    this->mainIndexGuard.unlock_shared();
}

/******************** Index API ****************************************/

template <typename DataType, typename DistType>
TieredHNSWIndex<DataType, DistType>::TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index,
                                                     BruteForceIndex<DataType, DistType> *bf_index,
                                                     const TieredIndexParams &tiered_index_params,
                                                     std::shared_ptr<VecSimAllocator> allocator)
    : VecSimTieredIndex<DataType, DistType>(hnsw_index, bf_index, tiered_index_params, allocator),
      labelToInsertJobs(this->allocator), idToRepairJobs(this->allocator),
      idToSwapJob(this->allocator), invalidJobs(this->allocator), currInvalidJobId(0),
      readySwapJobs(0) {
    // If the param for swapJobThreshold is 0 use the default value, if it exceeds the maximum
    // allowed, use the maximum value.
    this->pendingSwapJobsThreshold =
        tiered_index_params.specificParams.tieredHnswParams.swapJobThreshold == 0
            ? DEFAULT_PENDING_SWAP_JOBS_THRESHOLD
            : std::min(tiered_index_params.specificParams.tieredHnswParams.swapJobThreshold,
                       MAX_PENDING_SWAP_JOBS_THRESHOLD);
}

template <typename DataType, typename DistType>
TieredHNSWIndex<DataType, DistType>::~TieredHNSWIndex() {
    // Delete all the pending insert jobs.
    for (auto &jobs : this->labelToInsertJobs) {
        for (auto *job : jobs.second) {
            delete job;
        }
    }
    // Delete all the pending repair jobs.
    for (auto &jobs : this->idToRepairJobs) {
        for (auto *job : jobs.second) {
            delete job;
        }
    }
    // Delete all the pending swap jobs.
    for (auto &it : this->idToSwapJob) {
        delete it.second;
    }
    // Delete all the pending invalid jobs.
    for (auto &it : this->invalidJobs) {
        delete it.second;
    }
}

template <typename DataType, typename DistType>
size_t TieredHNSWIndex<DataType, DistType>::indexSize() const {
    this->flatIndexGuard.lock_shared();
    this->getHNSWIndex()->lockSharedIndexDataGuard();
    size_t res = this->backendIndex->indexSize() + this->frontendIndex->indexSize();
    this->getHNSWIndex()->unlockSharedIndexDataGuard();
    this->flatIndexGuard.unlock_shared();
    return res;
}

template <typename DataType, typename DistType>
size_t TieredHNSWIndex<DataType, DistType>::indexCapacity() const {
    return this->backendIndex->indexCapacity() + this->frontendIndex->indexCapacity();
}

// In the tiered index, we assume that the blobs are processed by the flat buffer
// before being transferred to the HNSW index.
// When inserting vectors directly into the HNSW index—such as in VecSim_WriteInPlace mode— or when
// the flat buffer is full, we must manually preprocess the blob according to the **frontend** index
// parameters.
template <typename DataType, typename DistType>
int TieredHNSWIndex<DataType, DistType>::addVector(const void *blob, labelType label) {
    int ret = 1;
    auto hnsw_index = this->getHNSWIndex();
    // writeMode is not protected since it is assumed to be called only from the "main thread"
    // (that is the thread that is exclusively calling add/delete vector).
    if (this->getWriteMode() == VecSim_WriteInPlace) {
        // First, check if we need to overwrite the vector in-place for single (from both indexes).
        if (!this->backendIndex->isMultiValue()) {
            ret -= this->deleteVector(label);
        }

        // Use the frontend parameters to manually prepare the blob for its transfer to the HNSW
        // index.
        auto storage_blob = this->frontendIndex->preprocessForStorage(blob);
        // Insert the vector to the HNSW index. Internally, we will never have to overwrite the
        // label since we already checked it outside.
        this->lockMainIndexGuard();
        hnsw_index->addVector(storage_blob.get(), label);
        this->unlockMainIndexGuard();
        return ret;
    }
    if (this->frontendIndex->indexSize() >= this->flatBufferLimit) {
        // Handle overwrite situation.
        if (!this->backendIndex->isMultiValue()) {
            // This will do nothing (and return 0) if this label doesn't exist. Otherwise, it may
            // remove vector from the flat buffer and/or the HNSW index.
            ret -= this->deleteVector(label);
        }
        if (this->frontendIndex->indexSize() >= this->flatBufferLimit) {
            // We didn't remove a vector from flat buffer due to overwrite, insert the new vector
            // directly to HNSW. Since flat buffer guard was not held, no need to release it
            // internally.
            // Use the frontend parameters to manually prepare the blob for its transfer to the HNSW
            // index.
            auto storage_blob = this->frontendIndex->preprocessForStorage(blob);
            this->insertVectorToHNSW<false>(hnsw_index, label, storage_blob.get());
            return ret;
        }
        // Otherwise, we fall back to the "regular" insertion into the flat buffer
        // (since it is not full anymore after removing the previous vector stored under the label).
    }
    this->flatIndexGuard.lock();
    idType new_flat_id = this->frontendIndex->indexSize();
    if (this->frontendIndex->isLabelExists(label) && !this->frontendIndex->isMultiValue()) {
        // Overwrite the vector and invalidate its only pending job (since we are not in MULTI).
        auto *old_job = this->labelToInsertJobs.at(label).at(0);
        old_job->id = this->setAndSaveInvalidJob(old_job);
        this->labelToInsertJobs.erase(label);
        ret = 0;
        // We are going to update the internal id that currently holds the vector associated with
        // the given label.
        new_flat_id =
            dynamic_cast<BruteForceIndex_Single<DataType, DistType> *>(this->frontendIndex)
                ->getIdOfLabel(label);
        // If we are adding a new element (rather than updating an exiting one) we may need to
        // increase index capacity.
    }
    // If this label already exists, this will do overwrite.
    this->frontendIndex->addVector(blob, label);

    AsyncJob *new_insert_job = new (this->allocator)
        HNSWInsertJob(this->allocator, label, new_flat_id, executeInsertJobWrapper, this);
    // Save a pointer to the job, so that if the vector is overwritten, we'll have an indication.
    if (this->labelToInsertJobs.find(label) != this->labelToInsertJobs.end()) {
        // There's already a pending insert job for this label, add another one (without overwrite,
        // only possible in multi index)
        assert(this->backendIndex->isMultiValue());
        this->labelToInsertJobs.at(label).push_back((HNSWInsertJob *)new_insert_job);
    } else {
        vecsim_stl::vector<HNSWInsertJob *> new_jobs_vec(1, (HNSWInsertJob *)new_insert_job,
                                                         this->allocator);
        this->labelToInsertJobs.insert({label, new_jobs_vec});
    }
    this->flatIndexGuard.unlock();

    // Here, a worker might ingest the previous vector that was stored under "label"
    // (in case of override in non-MULTI index) - so if it's there, we remove it (and create the
    // required repair jobs), *before* we submit the insert job.
    if (!this->backendIndex->isMultiValue()) {
        // If we removed the previous vector from both HNSW and flat in the overwrite process,
        // we still return 0 (not -1).
        ret = std::max(ret - this->deleteLabelFromHNSW(label), 0);
    }
    // Apply ready swap jobs if number of deleted vectors reached the threshold (under exclusive
    // lock of the main index guard).
    // If swapJobs size is equal or larger than a threshold, go over the swap jobs and execute a
    // batch of jobs for which all of its pending repair jobs were executed (otherwise finish and
    // return).
    if (readySwapJobs >= this->pendingSwapJobsThreshold) {
        this->executeReadySwapJobs(this->pendingSwapJobsThreshold);
    }

    // Insert job to the queue and signal the workers' updater.
    this->submitSingleJob(new_insert_job);
    return ret;
}

template <typename DataType, typename DistType>
int TieredHNSWIndex<DataType, DistType>::deleteVector(labelType label) {
    int num_deleted_vectors = 0;
    this->flatIndexGuard.lock_shared();
    if (this->frontendIndex->isLabelExists(label)) {
        this->flatIndexGuard.unlock_shared();
        this->flatIndexGuard.lock();
        // Check again if the label exists, as it may have been removed while we released the lock.
        if (this->frontendIndex->isLabelExists(label)) {
            // Invalidate the pending insert job(s) into HNSW associated with this label
            auto &insert_jobs = this->labelToInsertJobs.at(label);
            for (auto *job : insert_jobs) {
                job->id = this->setAndSaveInvalidJob(job);
            }
            num_deleted_vectors += insert_jobs.size();
            // Remove the pending insert job(s) from the labelToInsertJobs mapping.
            this->labelToInsertJobs.erase(label);
            // Go over the every id that corresponds the label and remove it from the flat buffer.
            // Every delete may cause a swap of the deleted id with the last id, and we return a
            // mapping from id to the original id that resides in this id after the deletion(s) (see
            // an example in this function implementation in MULTI index).
            auto updated_ids = this->frontendIndex->deleteVectorAndGetUpdatedIds(label);
            for (auto &it : updated_ids) {
                idType prev_id = it.second.first;
                labelType updated_vec_label = it.second.second;
                this->updateInsertJobInternalId(prev_id, it.first, updated_vec_label);
            }
        }
        this->flatIndexGuard.unlock();
    } else {
        this->flatIndexGuard.unlock_shared();
    }

    // Next, check if there vector(s) stored under the given label in HNSW and delete them as well.
    // Note that we may remove the same vector that has been removed from the flat index, if it was
    // being ingested at that time.
    // writeMode is not protected since it is assumed to be called only from the "main thread"
    // (that is the thread that is exclusively calling add/delete vector).
    if (this->getWriteMode() == VecSim_WriteAsync) {
        num_deleted_vectors += this->deleteLabelFromHNSW(label);
        // Apply ready swap jobs if number of deleted vectors reached the threshold
        // (under exclusive lock of the main index guard).
        if (readySwapJobs >= this->pendingSwapJobsThreshold) {
            this->executeReadySwapJobs(this->pendingSwapJobsThreshold);
        }
    } else {
        // delete in place.
        this->lockMainIndexGuard();
        num_deleted_vectors += this->deleteLabelFromHNSWInplace(label);
        this->unlockMainIndexGuard();
    }

    return num_deleted_vectors;
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

// IMPORTANT: this should be called when the *tiered index locks are locked for shared ownership*,
// along with HNSW index data guard lock. That is since the internal getDistanceFrom calls access
// the indexes' data, and it is not safe to run insert/delete operation in parallel. Also, we avoid
// acquiring the locks internally, since this is usually called for every vector individually, and
// the overhead of acquiring and releasing the locks is significant in that case.
template <typename DataType, typename DistType>
double TieredHNSWIndex<DataType, DistType>::getDistanceFrom_Unsafe(labelType label,
                                                                   const void *blob) const {
    // Try to get the distance from the flat buffer.
    // If the label doesn't exist, the distance will be NaN.
    auto flat_dist = this->frontendIndex->getDistanceFrom_Unsafe(label, blob);

    // Optimization. TODO: consider having different implementations for single and multi indexes,
    // to avoid checking the index type on every query.
    if (!this->backendIndex->isMultiValue() && !std::isnan(flat_dist)) {
        // If the index is single value, and we got a valid distance from the flat buffer,
        // we can return the distance without querying the Main index.
        return flat_dist;
    }

    // Try to get the distance from the Main index.
    auto hnsw_dist = getHNSWIndex()->getDistanceFrom_Unsafe(label, blob);

    // Return the minimum distance that is not NaN.
    return std::fmin(flat_dist, hnsw_dist);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//  TieredHNSW_BatchIterator                                                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////

/******************** Ctor / Dtor *****************/

// Defining spacial values for the hnsw_iterator field, to indicate if the iterator is uninitialized
// or depleted when we don't have a valid iterator.
#define UNINITIALIZED ((VecSimBatchIterator *)0)
#define DEPLETED      ((VecSimBatchIterator *)1)

template <typename DataType, typename DistType>
TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::TieredHNSW_BatchIterator(
    const void *query_vector, const TieredHNSWIndex<DataType, DistType> *index,
    VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator)
    // Tiered batch iterator doesn't hold its own copy of the query vector.
    // Instead, each internal batch iterators (flat_iterator and hnsw_iterator) create their own
    // copies: flat_iterator copy is created during TieredHNSW_BatchIterator construction When
    // TieredHNSW_BatchIterator::getNextResults() is called and hnsw_iterator is not initialized, it
    // retrieves the blob from flat_iterator
    : VecSimBatchIterator(nullptr, queryParams ? queryParams->timeoutCtx : nullptr,
                          std::move(allocator)),
      index(index), flat_results(this->allocator), hnsw_results(this->allocator),
      flat_iterator(this->index->frontendIndex->newBatchIterator(query_vector, queryParams)),
      hnsw_iterator(UNINITIALIZED), returned_results_set(this->allocator) {
    // Save a copy of the query params to initialize the HNSW iterator with (on first batch and
    // first batch after reset).
    if (queryParams) {
        this->queryParams =
            (VecSimQueryParams *)this->allocator->allocate(sizeof(VecSimQueryParams));
        *this->queryParams = *queryParams;
    } else {
        this->queryParams = nullptr;
    }
}

template <typename DataType, typename DistType>
TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::~TieredHNSW_BatchIterator() {
    delete this->flat_iterator;

    if (this->hnsw_iterator != UNINITIALIZED && this->hnsw_iterator != DEPLETED) {
        delete this->hnsw_iterator;
        this->index->mainIndexGuard.unlock_shared();
    }

    this->allocator->free_allocation(this->queryParams);
}

/******************** Implementation **************/

template <typename DataType, typename DistType>
VecSimQueryReply *TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::getNextResults(
    size_t n_res, VecSimQueryReply_Order order) {

    const bool isMulti = this->index->backendIndex->isMultiValue();
    auto hnsw_code = VecSim_QueryReply_OK;

    if (this->hnsw_iterator == UNINITIALIZED) {
        // First call to getNextResults. The call to the BF iterator will include calculating all
        // the distances and access the BF index. We take the lock on this call.
        this->index->flatIndexGuard.lock_shared();
        auto cur_flat_results = this->flat_iterator->getNextResults(n_res, BY_SCORE_THEN_ID);
        this->index->flatIndexGuard.unlock_shared();
        // This is also the only time `getNextResults` on the BF iterator can fail.
        if (VecSim_OK != cur_flat_results->code) {
            return cur_flat_results;
        }
        this->flat_results.swap(cur_flat_results->results);
        VecSimQueryReply_Free(cur_flat_results);
        // We also take the lock on the main index on the first call to getNextResults, and we hold
        // it until the iterator is depleted or freed.
        this->index->mainIndexGuard.lock_shared();
        this->hnsw_iterator = this->index->backendIndex->newBatchIterator(
            this->flat_iterator->getQueryBlob(), queryParams);
        auto cur_hnsw_results = this->hnsw_iterator->getNextResults(n_res, BY_SCORE_THEN_ID);
        hnsw_code = cur_hnsw_results->code;
        this->hnsw_results.swap(cur_hnsw_results->results);
        VecSimQueryReply_Free(cur_hnsw_results);
        if (this->hnsw_iterator->isDepleted()) {
            delete this->hnsw_iterator;
            this->hnsw_iterator = DEPLETED;
            this->index->mainIndexGuard.unlock_shared();
        }
    } else {
        while (this->flat_results.size() < n_res && !this->flat_iterator->isDepleted()) {
            auto tail = this->flat_iterator->getNextResults(n_res - this->flat_results.size(),
                                                            BY_SCORE_THEN_ID);
            this->flat_results.insert(this->flat_results.end(), tail->results.begin(),
                                      tail->results.end());
            VecSimQueryReply_Free(tail);

            if (!isMulti) {
                // On single-value indexes, duplicates will never appear in the hnsw results before
                // they appear in the flat results (at the same time or later if the approximation
                // misses) so we don't need to try and filter the flat results (and recheck
                // conditions).
                break;
            } else {
                // On multi-value indexes, the flat results may contain results that are already
                // returned from the hnsw index. We need to filter them out.
                filter_irrelevant_results(this->flat_results);
            }
        }

        while (this->hnsw_results.size() < n_res && this->hnsw_iterator != DEPLETED &&
               hnsw_code == VecSim_OK) {
            auto tail = this->hnsw_iterator->getNextResults(n_res - this->hnsw_results.size(),
                                                            BY_SCORE_THEN_ID);
            hnsw_code = tail->code; // Set the hnsw_results code to the last `getNextResults` code.
            // New batch may contain better results than the previous batch, so we need to merge.
            // We don't expect duplications (hence the <false>), as the iterator guarantees that
            // no result is returned twice.
            VecSimQueryResultContainer cur_hnsw_results(this->allocator);
            merge_results<false>(cur_hnsw_results, this->hnsw_results, tail->results, n_res);
            VecSimQueryReply_Free(tail);
            this->hnsw_results.swap(cur_hnsw_results);
            filter_irrelevant_results(this->hnsw_results);
            if (this->hnsw_iterator->isDepleted()) {
                delete this->hnsw_iterator;
                this->hnsw_iterator = DEPLETED;
                this->index->mainIndexGuard.unlock_shared();
            }
        }
    }

    if (VecSim_OK != hnsw_code) {
        return new VecSimQueryReply(this->allocator, hnsw_code);
    }

    VecSimQueryReply *batch;
    if (isMulti)
        batch = compute_current_batch<true>(n_res);
    else
        batch = compute_current_batch<false>(n_res);

    if (order == BY_ID) {
        sort_results_by_id(batch);
    }
    size_t batch_len = VecSimQueryReply_Len(batch);
    this->updateResultsCount(batch_len);

    return batch;
}

// DISCLAIMER: After the last batch, one of the iterators may report that it is not depleted,
// while all of its remaining results were already returned from the other iterator.
// (On single-value indexes, this can happen to the hnsw iterator only, on multi-value
//  indexes, this can happen to both iterators).
// The next call to `getNextResults` will return an empty batch, and then the iterators will
// correctly report that they are depleted.
template <typename DataType, typename DistType>
bool TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::isDepleted() {
    return this->flat_results.empty() && this->flat_iterator->isDepleted() &&
           this->hnsw_results.empty() && this->hnsw_iterator == DEPLETED;
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::reset() {
    if (this->hnsw_iterator != UNINITIALIZED && this->hnsw_iterator != DEPLETED) {
        delete this->hnsw_iterator;
        this->index->mainIndexGuard.unlock_shared();
    }
    this->resetResultsCount();
    this->flat_iterator->reset();
    this->hnsw_iterator = UNINITIALIZED;
    this->flat_results.clear();
    this->hnsw_results.clear();
    returned_results_set.clear();
}

/****************** Helper Functions **************/

template <typename DataType, typename DistType>
template <bool isMultiValue>
VecSimQueryReply *
TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::compute_current_batch(size_t n_res) {
    // Merge results
    // This call will update `hnsw_res` and `bf_res` to point to the end of the merged results.
    auto batch_res = new VecSimQueryReply(this->allocator);
    std::pair<size_t, size_t> p;
    if (isMultiValue) {
        p = merge_results<true>(batch_res->results, this->hnsw_results, this->flat_results, n_res);
    } else {
        p = merge_results<false>(batch_res->results, this->hnsw_results, this->flat_results, n_res);
    }
    auto [from_hnsw, from_flat] = p;

    if (!isMultiValue) {
        // If we're on a single-value index, update the set of results returned from the FLAT index
        // before popping them, to prevent them to be returned from the HNSW index in later batches.
        for (size_t i = 0; i < from_flat; ++i) {
            this->returned_results_set.insert(this->flat_results[i].id);
        }
    } else {
        // If we're on a multi-value index, update the set of results returned (from `batch_res`)
        for (size_t i = 0; i < batch_res->results.size(); ++i) {
            this->returned_results_set.insert(batch_res->results[i].id);
        }
    }

    // Update results
    this->flat_results.erase(this->flat_results.begin(), this->flat_results.begin() + from_flat);
    this->hnsw_results.erase(this->hnsw_results.begin(), this->hnsw_results.begin() + from_hnsw);

    // clean up the results
    // On multi-value indexes, one (or both) results lists may contain results that are already
    // returned form the other list (with a different score). We need to filter them out.
    if (isMultiValue) {
        filter_irrelevant_results(this->flat_results);
        filter_irrelevant_results(this->hnsw_results);
    }

    // Return current batch
    return batch_res;
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::filter_irrelevant_results(
    VecSimQueryResultContainer &results) {
    // Filter out results that were already returned.
    auto it = results.begin();
    const auto end = results.end();
    // Skip results that not returned yet
    while (it != end && this->returned_results_set.count(it->id) == 0) {
        ++it;
    }
    // If none of the results were returned, return
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
    // Update number of results (pop the tail)
    results.resize(cur_end - results.begin());
}

template <typename DataType, typename DistType>
VecSimIndexDebugInfo TieredHNSWIndex<DataType, DistType>::debugInfo() const {
    auto info = VecSimTieredIndex<DataType, DistType>::debugInfo();

    HnswTieredInfo hnswTieredInfo = {.pendingSwapJobsThreshold = this->pendingSwapJobsThreshold};
    info.tieredInfo.specificTieredBackendInfo.hnswTieredInfo = hnswTieredInfo;

    info.tieredInfo.backgroundIndexing =
        info.tieredInfo.frontendCommonInfo.indexSize > 0 ? VecSimBool_TRUE : VecSimBool_FALSE;

    return info;
}

template <typename DataType, typename DistType>
VecSimDebugInfoIterator *TieredHNSWIndex<DataType, DistType>::debugInfoIterator() const {
    VecSimIndexDebugInfo info = this->debugInfo();
    // Get the base tiered fields.
    auto *infoIterator = VecSimTieredIndex<DataType, DistType>::debugInfoIterator();

    // Tiered HNSW specific param.
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::TIERED_HNSW_SWAP_JOBS_THRESHOLD_STRING,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.tieredInfo.specificTieredBackendInfo
                                                       .hnswTieredInfo.pendingSwapJobsThreshold}}});

    return infoIterator;
}

template <typename DataType, typename DistType>
VecSimIndexBasicInfo TieredHNSWIndex<DataType, DistType>::basicInfo() const {
    VecSimIndexBasicInfo info = this->backendIndex->getBasicInfo();
    info.blockSize = info.blockSize;
    info.isTiered = true;
    info.algo = VecSimAlgo_HNSWLIB;
    return info;
}

#ifdef BUILD_TESTS
template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::getDataByLabel(
    labelType label, std::vector<std::vector<DataType>> &vectors_output) const {
    this->getHNSWIndex()->getDataByLabel(label, vectors_output);
}

#endif

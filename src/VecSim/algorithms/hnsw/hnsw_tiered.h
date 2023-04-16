#pragma once

#include "VecSim/vec_sim_tiered_index.h"
#include "hnsw.h"
#include "VecSim/index_factories/hnsw_factory.h"

#include <unordered_map>
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
    void atomicDecreasePendingJobsNum() {
        pending_repair_jobs_counter--;
        assert(pending_repair_jobs_counter >= 0);
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

    // This threshold is tested upon deleting a label from HNSW, and once the number of deleted
    // vectors reached this limit, we apply swap jobs *only for vectors that has no more pending
    // repair jobs*, and are ready to be removed from the graph.
    size_t pendingSwapJobsThreshold;

    // Protect the both idToRepairJobs lookup and the pending_repair_jobs_counter for the
    // associated swap jobs.
    std::mutex idToRepairJobsGuard;

    void executeInsertJob(HNSWInsertJob *job);
    void executeRepairJob(HNSWRepairJob *job);

    // To be executed synchronously upon deleting a vector, doesn't require a wrapper. Main HNSW
    // lock is assumed to be held exclusive here.
    void executeSwapJob(HNSWSwapJob *job, vecsim_stl::vector<idType> &idsToRemove);

    void executeReadySwapJobs();

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
    void updateInsertJobInternalId(idType prev_id, idType new_id);

    // Helper function for performing in place mark delete of vector(s) associated with a label
    // and creating the appropriate repair jobs for the effected connections. This should be called
    // while *HNSW shared lock is held* (shared locked).
    int deleteLabelFromHNSW(labelType label);

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_tiered_tests_friends.h"
#endif

public:
    class TieredHNSW_BatchIterator : public VecSimBatchIterator {
    private:
        const TieredHNSWIndex<DataType, DistType> *index;
        VecSimQueryParams *queryParams;

        VecSimQueryResult_List flat_results;
        VecSimQueryResult_List hnsw_results;

        VecSimBatchIterator *flat_iterator;
        VecSimBatchIterator *hnsw_iterator;

        // On single value indices, this set holds the IDs of the results that were returned from
        // the flat buffer.
        // On multi value indices, this set holds the IDs of all the results that were returned.
        vecsim_stl::unordered_set<labelType> returned_results_set;

    private:
        template <bool isMultiValue>
        inline VecSimQueryResult_List get_current_batch(size_t n_res);
        inline void filter_irrelevant_results(VecSimQueryResult_List &);

    public:
        TieredHNSW_BatchIterator(void *query_vector,
                                 const TieredHNSWIndex<DataType, DistType> *index,
                                 VecSimQueryParams *queryParams,
                                 std::shared_ptr<VecSimAllocator> allocator);

        ~TieredHNSW_BatchIterator();

        VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

        bool isDepleted() override;

        void reset() override;
    };

public:
    TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index,
                    BruteForceIndex<DataType, DistType> *bf_index,
                    const TieredIndexParams &tieredParams);
    virtual ~TieredHNSWIndex();

    int addVector(const void *blob, labelType label, void *auxiliaryCtx = nullptr) override;
    int deleteVector(labelType label) override;
    size_t indexSize() const override;
    size_t indexLabelCount() const override;
    size_t indexCapacity() const override;
    double getDistanceFrom(labelType label, const void *blob) const override;
    // Do nothing here, each tier (flat buffer and HNSW) should increase capacity for itself when
    // needed.
    void increaseCapacity() override {}

    // TODO: Implement the actual methods instead of these temporary ones.
    VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius,
                                      VecSimQueryParams *queryParams) override {
        return this->backendIndex->rangeQuery(queryBlob, radius, queryParams);
    }
    VecSimIndexInfo info() const override { return this->backendIndex->info(); }
    VecSimInfoIterator *infoIterator() const override { return this->backendIndex->infoIterator(); }
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        size_t blobSize = this->backendIndex->getDim() * sizeof(DataType);
        void *queryBlobCopy = this->allocator->allocate(blobSize);
        memcpy(queryBlobCopy, queryBlob, blobSize);
        return new (this->allocator)
            TieredHNSW_BatchIterator(queryBlobCopy, this, queryParams, this->allocator);
    }
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override {
        return this->backendIndex->preferAdHocSearch(subsetSize, k, initial_check);
    }
    inline void setLastSearchMode(VecSearchMode mode) override {
        return this->backendIndex->setLastSearchMode(mode);
    }
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
    delete insert_job;
    job_index->UpdateIndexMemory(job_index->memoryCtx, job_index->getAllocationSize());
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeRepairJobWrapper(AsyncJob *job) {
    auto *repair_job = reinterpret_cast<HNSWRepairJob *>(job);
    auto *job_index = reinterpret_cast<TieredHNSWIndex<DataType, DistType> *>(repair_job->index);
    job_index->executeRepairJob(repair_job);
    delete repair_job;
    job_index->UpdateIndexMemory(job_index->memoryCtx, job_index->getAllocationSize());
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeSwapJob(HNSWSwapJob *job,
                                                         vecsim_stl::vector<idType> &idsToRemove) {
    auto hnsw_index = this->getHNSWIndex();
    hnsw_index->removeAndSwapDeletedElement(job->deleted_id);
    // Get the id that was last and was had been swapped with the job's deleted id.
    idType prev_last_id = this->getHNSWIndex()->indexSize();

    // Invalidate repair jobs for the disposed id (if exist), and update the associated swap jobs.
    if (idToRepairJobs.find(job->deleted_id) != idToRepairJobs.end()) {
        for (auto &job_it : idToRepairJobs.at(job->deleted_id)) {
            job_it->node_id = INVALID_JOB_ID;
            for (auto &swap_job_it : job_it->associatedSwapJobs) {
                swap_job_it->atomicDecreasePendingJobsNum();
            }
        }
        idToRepairJobs.erase(job->deleted_id);
    }
    // Swap the ids in the pending jobs for the current last id (if exist).
    if (idToRepairJobs.find(prev_last_id) != idToRepairJobs.end()) {
        for (auto &job_it : idToRepairJobs.at(prev_last_id)) {
            job_it->node_id = job->deleted_id;
        }
        idToRepairJobs.insert({job->deleted_id, idToRepairJobs.at(prev_last_id)});
        idToRepairJobs.erase(prev_last_id);
    }
    // Update the swap jobs if the last id also needs a swap, otherwise just collect to deleted id
    // to be removed from the swap jobs.
    if (prev_last_id != job->deleted_id && idToSwapJob.find(prev_last_id) != idToSwapJob.end() &&
        std::find(idsToRemove.begin(), idsToRemove.end(), prev_last_id) == idsToRemove.end()) {
        // Update the curr_last_id pending swap job id after the removal that renamed curr_last_id
        // with the deleted id.
        idsToRemove.push_back(prev_last_id);
        idToSwapJob.at(prev_last_id)->deleted_id = job->deleted_id;
        idToSwapJob.at(job->deleted_id) = idToSwapJob.at(prev_last_id);
    } else {
        idsToRemove.push_back(job->deleted_id);
    }
    delete job;
}

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType> *TieredHNSWIndex<DataType, DistType>::getHNSWIndex() const {
    return dynamic_cast<HNSWIndex<DataType, DistType> *>(this->backendIndex);
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeReadySwapJobs() {
    // If swapJobs size is equal or larger than a threshold, go over the swap jobs and execute every
    // job for which all of its pending repair jobs were executed (otherwise finish and return).
    if (idToSwapJob.size() < this->pendingSwapJobsThreshold) {
        return;
    }
    // Execute swap jobs - acquire hnsw write lock.
    this->mainIndexGuard.lock();

    vecsim_stl::vector<idType> idsToRemove(this->allocator);
    idsToRemove.reserve(idToSwapJob.size());
    for (auto &it : idToSwapJob) {
        if (it.second->pending_repair_jobs_counter.load() == 0) {
            // Swap job is ready for execution - execute and delete it.
            this->executeSwapJob(it.second, idsToRemove);
        }
    }
    for (idType id : idsToRemove) {
        idToSwapJob.erase(id);
    }
    this->mainIndexGuard.unlock();
}

template <typename DataType, typename DistType>
int TieredHNSWIndex<DataType, DistType>::deleteLabelFromHNSW(labelType label) {
    auto *hnsw_index = getHNSWIndex();

    // Get the required data about the relevant ids to delete.
    // Internally, this will hold the index data lock.
    auto internal_ids = hnsw_index->markDelete(label);

    for (size_t i = 0; i < internal_ids.size(); i++) {
        idType id = internal_ids[i];
        vecsim_stl::vector<HNSWRepairJob *> repair_jobs(this->allocator);
        auto *swap_job = new (this->allocator) HNSWSwapJob(this->allocator, id);

        // Go over all the deleted element links in every level and create repair jobs.
        auto incoming_edges =
            hnsw_index->safeCollectAllNodeIncomingNeighbors(id, hnsw_index->getElementTopLevel(id));

        // Protect the id->repair_jobs lookup while we update it with the new jobs.
        this->idToRepairJobsGuard.lock();
        for (pair<idType, ushort> &node : incoming_edges) {
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
        swap_job->setRepairJobsNum(incoming_edges.size());
        this->idToRepairJobsGuard.unlock();

        this->SubmitJobsToQueue(this->jobQueue, (AsyncJob **)repair_jobs.data(), repair_jobs.size(),
                                this->jobQueueCtx);
        // Insert the swap job into the swap jobs lookup (for fast update in case that the
        // node id is changed due to swap job).
        assert(idToSwapJob.find(id) == idToSwapJob.end());
        idToSwapJob[id] = swap_job;
    }
    return internal_ids.size();
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::updateInsertJobInternalId(idType prev_id, idType new_id) {
    // Update the pending job id, due to a swap that was caused after the removal of new_id.
    assert(new_id != INVALID_ID && prev_id != INVALID_ID);
    labelType last_idx_label = this->frontendIndex->getLabelByInternalId(prev_id);
    auto it = this->labelToInsertJobs.find(last_idx_label);
    if (it != this->labelToInsertJobs.end()) {
        // There is a pending job for the label of the swapped last id - update its id.
        for (HNSWInsertJob *job_it : it->second) {
            if (job_it->id == prev_id) {
                job_it->id = new_id;
            }
        }
    }
}

/******************** Job's callbacks **********************************/
template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeInsertJob(HNSWInsertJob *job) {
    // Note: this method had not been tested with yet overwriting scenarios, where job may
    // have been invalidate before it is executed (TODO in the future).
    HNSWIndex<DataType, DistType> *hnsw_index = this->getHNSWIndex();
    // Note that accessing the job fields should occur with flat index guard held (here and later).
    this->flatIndexGuard.lock_shared();
    if (job->id == INVALID_JOB_ID) {
        // Job has been invalidated in the meantime.
        this->flatIndexGuard.unlock_shared();
        return;
    }

    // Copy the vector blob from the flat buffer, so we can release the flat lock while we are
    // indexing the vector into HNSW index.
    DataType blob_copy[this->frontendIndex->getDim()];
    memcpy(blob_copy, this->frontendIndex->getDataByInternalId(job->id),
           this->frontendIndex->getDim() * sizeof(DataType));

    // Acquire the index data lock, so we know what is the exact index size at this time. Acquire
    // the main r/w lock before to avoid deadlocks.
    AddVectorCtx state = {0};
    this->mainIndexGuard.lock_shared();
    hnsw_index->lockIndexDataGuard();
    // Check if resizing is needed for HNSW index (requires write lock).
    if (hnsw_index->indexCapacity() == hnsw_index->indexSize()) {
        // Release the inner HNSW data lock before we re-acquire the global HNSW lock.
        this->mainIndexGuard.unlock_shared();
        hnsw_index->unlockIndexDataGuard();
        this->mainIndexGuard.lock();
        hnsw_index->lockIndexDataGuard();
        // Check if resizing is still required (another thread might have done it in the meantime
        // while we release the shared lock).
        if (hnsw_index->indexCapacity() == hnsw_index->indexSize()) {
            hnsw_index->increaseCapacity();
        }
        // Hold the index data lock while we store the new element. If the new node's max level is
        // higher than the current one, hold the lock through the entire insertion to ensure that
        // graph scans will not occur, as they will try access the entry point's neighbors.
        state = hnsw_index->storeNewElement(job->label);
        this->flatIndexGuard.unlock_shared();

        // If we're still holding the index data guard, we cannot take the main index lock for
        // shared ownership as it may cause deadlocks, and we also cannot release the main index
        // lock between, since we cannot allow swap jobs to happen, as they will make the
        // saved state invalid. Hence, we insert the vector with the current exclusive lock held.
        if (state.elementMaxLevel <= state.currMaxLevel) {
            hnsw_index->unlockIndexDataGuard();
        }
        // Take the vector from the flat buffer and insert it to HNSW (overwrite should not occur).
        hnsw_index->addVector(blob_copy, job->label, &state);
        if (state.elementMaxLevel > state.currMaxLevel) {
            hnsw_index->unlockIndexDataGuard();
        }
        this->mainIndexGuard.unlock();
    } else {
        // Do the same as above except for changing the capacity, but with *shared* lock held:
        // Hold the index data lock while we store the new element. If the new node's max level is
        // higher than the current one, hold the lock through the entire insertion to ensure that
        // graph scans will not occur, as they will try access the entry point's neighbors.
        state = hnsw_index->storeNewElement(job->label);
        this->flatIndexGuard.unlock_shared();

        if (state.elementMaxLevel <= state.currMaxLevel) {
            hnsw_index->unlockIndexDataGuard();
        }
        // Take the vector from the flat buffer and insert it to HNSW (overwrite should not occur).
        hnsw_index->addVector(blob_copy, job->label, &state);
        if (state.elementMaxLevel > state.currMaxLevel) {
            hnsw_index->unlockIndexDataGuard();
        }
        this->mainIndexGuard.unlock_shared();
    }

    // Remove the vector and the insert job from the flat buffer.
    this->flatIndexGuard.lock();
    // The job might have been invalidated due to overwrite in the meantime. In this case,
    // it was already deleted and the job has been evicted. Otherwise, we need to do it now.
    if (job->id != INVALID_JOB_ID) {
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
        // Remove the vector from the flat buffer.
        int deleted = this->frontendIndex->deleteVectorById(job->label, job->id);
        if (deleted && job->id != this->frontendIndex->indexSize()) {
            // If the vector removal caused a swap with the last id, update the relevant insert job.
            this->updateInsertJobInternalId(this->frontendIndex->indexSize(), job->id);
        }
    }
    this->flatIndexGuard.unlock();
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeRepairJob(HNSWRepairJob *job) {
    // Lock the HNSW shared lock before accessing its internals.
    this->mainIndexGuard.lock_shared();
    if (job->node_id == INVALID_JOB_ID) {
        // The current node has already been removed and disposed.
        this->mainIndexGuard.unlock_shared();
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
        it->atomicDecreasePendingJobsNum();
    }
    this->idToRepairJobsGuard.unlock();

    hnsw_index->repairNodeConnections(job->node_id, job->level);

    this->mainIndexGuard.unlock_shared();
    this->UpdateIndexMemory(this->memoryCtx, this->getAllocationSize());
}

/******************** Index API ****************************************/

template <typename DataType, typename DistType>
TieredHNSWIndex<DataType, DistType>::TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index,
                                                     BruteForceIndex<DataType, DistType> *bf_index,
                                                     const TieredIndexParams &tiered_index_params)
    : VecSimTieredIndex<DataType, DistType>(hnsw_index, bf_index, tiered_index_params),
      labelToInsertJobs(this->allocator), idToRepairJobs(this->allocator),
      idToSwapJob(this->allocator) {
    // If the param for swapJobThreshold is 0 use the default value, if it exceeds the maximum
    // allowed, use the maximum value.
    this->pendingSwapJobsThreshold =
        tiered_index_params.specificParams.tieredHnswParams.swapJobThreshold == 0
            ? DEFAULT_PENDING_SWAP_JOBS_THRESHOLD
            : std::min(tiered_index_params.specificParams.tieredHnswParams.swapJobThreshold,
                       MAX_PENDING_SWAP_JOBS_THRESHOLD);
    this->UpdateIndexMemory(this->memoryCtx, this->getAllocationSize());
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
}

template <typename DataType, typename DistType>
size_t TieredHNSWIndex<DataType, DistType>::indexSize() const {
    this->flatIndexGuard.lock_shared();
    this->getHNSWIndex()->lockIndexDataGuard();
    size_t res = this->backendIndex->indexSize() + this->frontendIndex->indexSize();
    this->getHNSWIndex()->unlockIndexDataGuard();
    this->flatIndexGuard.unlock_shared();
    return res;
}

template <typename DataType, typename DistType>
size_t TieredHNSWIndex<DataType, DistType>::indexCapacity() const {
    return this->backendIndex->indexCapacity() + this->frontendIndex->indexCapacity();
}

template <typename DataType, typename DistType>
size_t TieredHNSWIndex<DataType, DistType>::indexLabelCount() const {
    // Compute the union of both labels set in both tiers of the index.
    this->flatIndexGuard.lock();
    this->mainIndexGuard.lock();
    auto flat_labels = this->frontendIndex->getLabelsSet();
    auto hnsw_labels = this->getHNSWIndex()->getLabelsSet();
    std::vector<labelType> output;
    std::set_union(flat_labels.begin(), flat_labels.end(), hnsw_labels.begin(), hnsw_labels.end(),
                   std::back_inserter(output));
    this->flatIndexGuard.unlock();
    this->mainIndexGuard.unlock();
    return output.size();
}

template <typename DataType, typename DistType>
int TieredHNSWIndex<DataType, DistType>::addVector(const void *blob, labelType label,
                                                   void *auxiliaryCtx) {
    /* Note: this currently doesn't support overriding (assuming that the label doesn't exist)! */
    this->flatIndexGuard.lock();
    if (this->frontendIndex->indexCapacity() == this->frontendIndex->indexSize()) {
        this->frontendIndex->increaseCapacity();
    }
    idType new_flat_id = this->frontendIndex->indexSize();
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

    // Insert job to the queue and signal the workers updater
    this->submitSingleJob(new_insert_job);
    this->UpdateIndexMemory(this->memoryCtx, this->getAllocationSize());
    return 1;
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
                reinterpret_cast<HNSWInsertJob *>(job)->id = INVALID_JOB_ID;
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
                this->updateInsertJobInternalId(it.second, it.first);
            }
        }
        this->flatIndexGuard.unlock();
    } else {
        this->flatIndexGuard.unlock_shared();
    }

    // Next, check if there vector(s) stored under the given label in HNSW and delete them as well.
    // Note that we may remove the same vector that has been removed from the flat index, if it was
    // being ingested at that time.
    this->mainIndexGuard.lock_shared();
    num_deleted_vectors += this->deleteLabelFromHNSW(label);
    this->mainIndexGuard.unlock_shared();

    // Apply ready swap jobs if number of deleted vectors reached the threshold
    // (under exclusive lock of the main index guard).
    this->executeReadySwapJobs();

    this->UpdateIndexMemory(this->memoryCtx, this->getAllocationSize());
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
template <typename DataType, typename DistType>
double TieredHNSWIndex<DataType, DistType>::getDistanceFrom(labelType label,
                                                            const void *blob) const {
    // Try to get the distance from the flat buffer.
    // If the label doesn't exist, the distance will be NaN.
    this->flatIndexGuard.lock_shared();
    auto flat_dist = this->frontendIndex->getDistanceFrom(label, blob);
    this->flatIndexGuard.unlock_shared();

    // Optimization. TODO: consider having different implementations for single and multi indexes,
    // to avoid checking the index type on every query.
    if (!this->backendIndex->isMultiValue() && !std::isnan(flat_dist)) {
        // If the index is single value, and we got a valid distance from the flat buffer,
        // we can return the distance without querying the Main index.
        return flat_dist;
    }

    // Try to get the distance from the Main index.
    this->mainIndexGuard.lock_shared();
    auto hnsw = getHNSWIndex();
    auto hnsw_dist = hnsw->safeGetDistanceFrom(label, blob);
    this->mainIndexGuard.unlock_shared();

    // Return the minimum distance that is not NaN.
    return std::fmin(flat_dist, hnsw_dist);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//  TieredHNSW_BatchIterator                                                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////

/******************** Ctor / Dtor *****************/

#define DEPLETED ((VecSimBatchIterator *)1)

template <typename DataType, typename DistType>
TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::TieredHNSW_BatchIterator(
    void *query_vector, const TieredHNSWIndex<DataType, DistType> *index,
    VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimBatchIterator(query_vector, queryParams ? queryParams->timeoutCtx : nullptr,
                          std::move(allocator)),
      index(index), flat_results({0}), hnsw_results({0}),
      flat_iterator(this->index->frontendIndex->newBatchIterator(query_vector, queryParams)),
      hnsw_iterator(nullptr), returned_results_set(this->allocator) {
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

    if (this->hnsw_iterator && this->hnsw_iterator != DEPLETED) {
        delete this->hnsw_iterator;
        this->index->mainIndexGuard.unlock_shared();
    }

    delete this->queryParams;

    VecSimQueryResult_Free(this->flat_results);
    VecSimQueryResult_Free(this->hnsw_results);
}

/******************** Implementation **************/

template <typename DataType, typename DistType>
VecSimQueryResult_List
TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::getNextResults(
    size_t n_res, VecSimQueryResult_Order order) {

    const bool isMulti = this->index->backendIndex->isMultiValue();

    if (this->getResultsCount() == 0) {
        // First call to getNextResults. The call to the BF iterator will include calculating all
        // the distances and access the BF index. We take the lock on this call.
        this->index->flatIndexGuard.lock_shared();
        this->flat_results = this->flat_iterator->getNextResults(n_res, BY_SCORE_THEN_ID);
        this->index->flatIndexGuard.unlock_shared();
        // This is also the only time `getNextResults` on the BF iterator can fail.
        if (VecSim_OK != flat_results.code) {
            return flat_results;
        }
        // We also take the lock on the main index on the first call to getNextResults, and we hold
        // it until the iterator is depleted or freed.
        this->index->mainIndexGuard.lock_shared();
        this->hnsw_iterator = this->index->backendIndex->newBatchIterator(getQueryBlob(), queryParams);
        this->hnsw_results = this->hnsw_iterator->getNextResults(n_res, BY_SCORE_THEN_ID);
        if (this->hnsw_iterator->isDepleted()) {
            delete this->hnsw_iterator;
            this->hnsw_iterator = DEPLETED;
            this->index->mainIndexGuard.unlock_shared();
        }
    } else {
        while (VecSimQueryResult_Len(this->flat_results) < n_res &&
               !this->flat_iterator->isDepleted()) {
            auto tail = this->flat_iterator->getNextResults(
                n_res - VecSimQueryResult_Len(this->flat_results), BY_SCORE_THEN_ID);
            concat_results(this->flat_results, tail);
            VecSimQueryResult_Free(tail);

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

        auto code = VecSim_QueryResult_OK;
        while (VecSimQueryResult_Len(this->hnsw_results) < n_res &&
               this->hnsw_iterator != DEPLETED && code == VecSim_OK) {
            auto tail = this->hnsw_iterator->getNextResults(
                n_res - VecSimQueryResult_Len(this->hnsw_results), BY_SCORE_THEN_ID);
            code = tail.code; // Set the hnsw_results code to the last `getNextResults` code.
            // New batch may contain better results than the previous batch, so we need to merge
            this->hnsw_results = merge_result_lists<false>(this->hnsw_results, tail, n_res);
            this->hnsw_results.code = code;
            filter_irrelevant_results(this->hnsw_results);
            if (this->hnsw_iterator->isDepleted()) {
                delete this->hnsw_iterator;
                this->hnsw_iterator = DEPLETED;
                this->index->mainIndexGuard.unlock_shared();
            }
        }
    }

    if (VecSim_OK != hnsw_results.code) {
        return {NULL, hnsw_results.code};
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
    return VecSimQueryResult_Len(this->flat_results) == 0 && this->flat_iterator->isDepleted() &&
           VecSimQueryResult_Len(this->hnsw_results) == 0 && this->hnsw_iterator == DEPLETED;
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::reset() {
    if (this->hnsw_iterator && this->hnsw_iterator != DEPLETED) {
        delete this->hnsw_iterator;
        this->index->mainIndexGuard.unlock_shared();
    }
    this->resetResultsCount();
    this->flat_iterator->reset();
    this->hnsw_iterator = nullptr;
    VecSimQueryResult_Free(this->flat_results);
    VecSimQueryResult_Free(this->hnsw_results);
    this->flat_results = {0};
    this->hnsw_results = {0};
    returned_results_set.clear();
}

/****************** Helper Functions **************/

template <typename DataType, typename DistType>
template <bool isMultiValue>
VecSimQueryResult_List
TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::get_current_batch(size_t n_res) {
    // Set pointers
    auto bf_res = this->flat_results.results;
    auto hnsw_res = this->hnsw_results.results;
    const auto bf_end = bf_res + VecSimQueryResult_Len(this->flat_results);
    const auto hnsw_end = hnsw_res + VecSimQueryResult_Len(this->hnsw_results);

    // Merge results
    VecSimQueryResult *batch_res;
    if (isMultiValue) {
        batch_res = merge_results<true>(hnsw_res, hnsw_end, bf_res, bf_end, n_res);
    } else {
        batch_res = merge_results<false>(hnsw_res, hnsw_end, bf_res, bf_end, n_res);
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
    array_pop_front_n(this->hnsw_results.results, hnsw_res - this->hnsw_results.results);

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
        filter_irrelevant_results(this->hnsw_results);
    }

    // Return current batch
    return {batch_res, VecSim_QueryResult_OK};
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::TieredHNSW_BatchIterator::filter_irrelevant_results(
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

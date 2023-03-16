#pragma once

#include "VecSim/vec_sim_tiered_index.h"
#include "hnsw.h"
#include "hnsw_factory.h"
#include "VecSim/utils/merge_results.h"

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
    std::atomic_uint
        pending_repair_jobs_counter; // number of repair jobs left to complete before this job
                                     // is ready to be executed (atomic counter).
    HNSWSwapJob(std::shared_ptr<VecSimAllocator> allocator, idType deletedId)
        : VecsimBaseObject(allocator), deleted_id(deletedId), pending_repair_jobs_counter(0) {}
    void setRepairJobsNum(long num_repair_jobs) { pending_repair_jobs_counter = num_repair_jobs; }
    void atomicDecreasePendingJobsNum() { pending_repair_jobs_counter--; }
};

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
    vecsim_stl::vector<HNSWSwapJob *> swapJobs;

    /// Mappings from id/label to associated jobs, for invalidating and update ids if necessary.
    // In MULTI, we can have more than one insert job pending per label
    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<HNSWInsertJob *>> labelToInsertJobs;
    vecsim_stl::unordered_map<idType, vecsim_stl::vector<HNSWRepairJob *>> idToRepairJobs;
    vecsim_stl::unordered_map<idType, HNSWSwapJob *> idToSwapJob;

    std::mutex idToRepairJobsGuard;

    void executeInsertJob(HNSWInsertJob *job);
    void executeRepairJob(HNSWRepairJob *job) {}

    // To be executed synchronously upon deleting a vector, doesn't require a wrapper.
    void executeSwapJob(HNSWSwapJob *job) {}

    // Wrappers static functions to be sent as callbacks upon creating the jobs (since members
    // functions cannot serve as callback, this serve as the "gateway" to the appropriate index).
    static void executeInsertJobWrapper(AsyncJob *job);
    static void executeRepairJobWrapper(AsyncJob *job) {}

    inline HNSWIndex<DataType, DistType> *getHNSWIndex() const;

    // Helper function for performing in place mark delete of vector(s) associated with a label
    // and creating the appropriate repair jobs for the effected connections. This should be called
    // while *HNSW shared lock is held* (shared locked).
    int deleteLabelFromHNSW(labelType label);

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_tiered_tests_friends.h"
#endif

public:
    TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index, TieredIndexParams tieredParams);
    virtual ~TieredHNSWIndex();

    int addVector(const void *blob, labelType label, idType new_vec_id = INVALID_ID) override;
    size_t indexSize() const override;
    size_t indexLabelCount() const override;
    size_t indexCapacity() const override;
    double getDistanceFrom(labelType label, const void *blob) const override;
    // Do nothing here, each tier (flat buffer and HNSW) should increase capacity for itself when
    // needed.
    void increaseCapacity() override {}

    // TODO: Implement the actual methods instead of these temporary ones.
    int deleteVector(labelType id) override { return this->index->deleteVector(id); }
    VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius,
                                      VecSimQueryParams *queryParams) override {
        return this->index->rangeQuery(queryBlob, radius, queryParams);
    }
    VecSimIndexInfo info() const override { return this->index->info(); }
    VecSimInfoIterator *infoIterator() const override { return this->index->infoIterator(); }
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        return this->index->newBatchIterator(queryBlob, queryParams);
    }
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override {
        return this->index->preferAdHocSearch(subsetSize, k, initial_check);
    }
    inline void setLastSearchMode(VecSearchMode mode) override {
        return this->index->setLastSearchMode(mode);
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
    job_index->UpdateIndexMemory(job_index->memoryCtx, job_index->getAllocationSize());
    delete insert_job;
}

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType> *TieredHNSWIndex<DataType, DistType>::getHNSWIndex() const {
    return reinterpret_cast<HNSWIndex<DataType, DistType> *>(this->index);
}

template <typename DataType, typename DistType>
int TieredHNSWIndex<DataType, DistType>::deleteLabelFromHNSW(labelType label) {
    auto *hnsw_index = getHNSWIndex();

    // Get the required data about the relevant ids to delete.
    // Internally, this will hold the index data lock.
    auto internal_ids = hnsw_index->markDelete(label);

    if (internal_ids.empty()) {
        // Label doesn't exist, or it has already marked as deleted - we can finish now.
        return 0;
    }
    vecsim_stl::vector<size_t> ids_top_level(this->allocator);
    for (idType id : internal_ids) {
        ids_top_level.push_back(hnsw_index->getElementTopLevel(id));
    }

    for (size_t i = 0; i < internal_ids.size(); i++) {
        idType id = internal_ids[i];
        vecsim_stl::vector<HNSWRepairJob *> repair_jobs(this->allocator);
        auto *swap_job = new (this->allocator) HNSWSwapJob(this->allocator, id);

        // Go over all the deleted element links in every level and create repair jobs.
        auto incoming_edges = hnsw_index->safeCollectAllNodeIncomingNeighbors(id, ids_top_level[i]);
        {
            // Protect the id->repair_jobs lookup while we update it with the new jobs.
            std::unique_lock<std::mutex> lock(this->idToRepairJobsGuard);
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
                    repair_job = new (this->allocator)
                        HNSWRepairJob(this->allocator, node.first, node.second,
                                      executeRepairJobWrapper, this, swap_job);
                    repair_jobs.emplace_back(repair_job);
                    idToRepairJobs.at(node.first).push_back(repair_job);
                }
            }
        }
        swap_job->setRepairJobsNum(incoming_edges.size());
        this->SubmitJobsToQueue(this->jobQueue, (AsyncJob **)repair_jobs.data(), repair_jobs.size(),
                                this->jobQueueCtx);
        swapJobs.push_back(swap_job);
        // Insert the swap job into the swap jobs lookup (for fast update in case that the
        // node id is changed due to swap job).
        idToSwapJob[id] = swap_job;
    }

    // Todo: if swapJobs size is larger than a threshold, go over the swap jobs and execute it,
    //  if all its pending repair jobs are executed (to be implemented later on).
    return internal_ids.size();
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
    // Acquire the index data lock, so we know what is the exact index size at this time.
    hnsw_index->lockIndexDataGuard();

    // Check if resizing is needed for HNSW index (requires write lock).
    if (hnsw_index->indexCapacity() == hnsw_index->indexSize()) {
        // Release the inner HNSW data lock before we re-acquire the global HNSW lock.
        hnsw_index->unlockIndexDataGuard();

        this->mainIndexGuard.lock();
        hnsw_index->lockIndexDataGuard();
        // Check if resizing is still required (another thread might have done it in the meantime
        // while we release the shared lock).
        if (hnsw_index->indexCapacity() == hnsw_index->indexSize()) {
            hnsw_index->increaseCapacity();
        }
        this->mainIndexGuard.unlock();
    }

    if (job->id == INVALID_JOB_ID) {
        // Job has been invalidated in the meantime (by overwriting this label) while we released
        // the flat index guard.
        this->flatIndexGuard.unlock_shared();
        return;
    }

    idType new_vec_id = hnsw_index->indexSize();
    hnsw_index->incrementIndexSize();
    hnsw_index->unlockIndexDataGuard();

    // Take the vector from the flat buffer and insert it to HNSW (overwrite should not occur).
    this->mainIndexGuard.lock_shared();
    hnsw_index->addVector(this->flatBuffer->getDataByInternalId(job->id), job->label, new_vec_id);
    this->mainIndexGuard.unlock_shared();

    // Remove the vector and the insert job from the flat buffer.
    this->flatIndexGuard.unlock_shared();
    {
        std::unique_lock<std::shared_mutex> flat_lock(this->flatIndexGuard);
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
            int deleted = this->flatBuffer->deleteVectorById(job->label, job->id);
            // This will cause the last id to swap with the deleted id - update the job with the
            // pending job with the last id, unless the deleted id is the last id.
            if (deleted && job->id != this->flatBuffer->indexSize()) {
                labelType last_idx_label = this->flatBuffer->getLabelByInternalId(job->id);
                if (this->labelToInsertJobs.find(last_idx_label) != this->labelToInsertJobs.end()) {
                    // There is a pending job for the label of the swapped last id - update its id.
                    for (HNSWInsertJob *job_it : this->labelToInsertJobs.at(last_idx_label)) {
                        if (job_it->id == this->flatBuffer->indexSize()) {
                            job_it->id = job->id;
                        }
                    }
                }
            }
        }
    }
}

/******************** Index API ****************************************/

template <typename DataType, typename DistType>
TieredHNSWIndex<DataType, DistType>::TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index,
                                                     TieredIndexParams tieredParams)
    : VecSimTieredIndex<DataType, DistType>(hnsw_index, tieredParams), swapJobs(this->allocator),
      labelToInsertJobs(this->allocator), idToRepairJobs(this->allocator),
      idToSwapJob(this->allocator) {
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
    for (auto *job : this->swapJobs) {
        delete job;
    }
}

template <typename DataType, typename DistType>
size_t TieredHNSWIndex<DataType, DistType>::indexSize() const {
    return this->index->indexSize() + this->flatBuffer->indexSize();
}

template <typename DataType, typename DistType>
size_t TieredHNSWIndex<DataType, DistType>::indexCapacity() const {
    return this->index->indexCapacity() + this->flatBuffer->indexCapacity();
}

template <typename DataType, typename DistType>
size_t TieredHNSWIndex<DataType, DistType>::indexLabelCount() const {
    return this->flatBuffer->indexLabelCount() + this->index->indexLabelCount();
}

template <typename DataType, typename DistType>
int TieredHNSWIndex<DataType, DistType>::addVector(const void *blob, labelType label,
                                                   idType new_vec_id) {
    /* Note: this currently doesn't support overriding (assuming that the label doesn't exist)! */
    this->flatIndexGuard.lock();
    if (this->flatBuffer->indexCapacity() == this->flatBuffer->indexSize()) {
        this->flatBuffer->increaseCapacity();
    }
    idType new_flat_id = this->flatBuffer->indexSize();
    this->flatBuffer->addVector(blob, label, false);
    AsyncJob *new_insert_job = new (this->allocator)
        HNSWInsertJob(this->allocator, label, new_flat_id, executeInsertJobWrapper, this);
    // Save a pointer to the job, so that if the vector is overwritten, we'll have an indication.
    if (this->labelToInsertJobs.find(label) != this->labelToInsertJobs.end()) {
        // There's already a pending insert job for this label, add another one (without overwrite,
        // only possible in multi index)
        assert(this->index->isMultiValue());
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
    auto hnsw = getHNSWIndex();
    auto hnsw_dist = hnsw->safeGetDistanceFrom(label, blob);
    this->mainIndexGuard.unlock_shared();

    // Return the minimum distance that is not NaN.
    return std::fmin(flat_dist, hnsw_dist);
}

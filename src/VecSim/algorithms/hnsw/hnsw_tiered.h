#pragma once

#include "VecSim/vec_sim_tiered_index.h"
#include "hnsw.h"
#include "hnsw_factory.h"

#include <unordered_map>
/**
 * Definition of a job that inserts a new vector from flat into HNSW Index.
 */
struct HNSWInsertJob : public AsyncJob {
    VecSimIndex *index;
    labelType label;
    idType id;

    HNSWInsertJob(std::shared_ptr<VecSimAllocator> allocator, labelType label_, idType id_,
                  JobCallback insertCb, VecSimIndex *index_)
        : AsyncJob(allocator, HNSW_INSERT_VECTOR_JOB, insertCb), index(index_), label(label_),
          id(id_) {}
};

/**
 * Definition of a job that swaps last id with a deleted id in HNSW Index after delete operation.
 */
struct HNSWSwapJob : public AsyncJob {
    VecSimIndex *index;
    idType deleted_id;
    long pending_repair_jobs_counter; // number of repair jobs left to complete before this job
                                      // is ready to be executed (atomic counter).
    // TODO: implement contractor
};

/**
 * Definition of a job that repairs a certain node's connection in HNSW Index after delete
 * operation.
 */
struct HNSWRepairJob : public AsyncJob {
    VecSimIndex *index;
    idType node_id;
    unsigned short level;
    HNSWSwapJob *associated_swap_job;

    // TODO: implement contractor
};

template <typename DataType, typename DistType>
class TieredHNSWIndex : public VecSimTieredIndex<DataType, DistType> {
private:
    /// Mappings from id/label to associated jobs, for invalidating and update ids if necessary.
    // In MULTI, we can have more than one insert job pending per label
    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<HNSWInsertJob *>> labelToInsertJobs;
    vecsim_stl::unordered_map<idType, vecsim_stl::vector<HNSWRepairJob *>> idToRepairJobs;
    vecsim_stl::unordered_map<idType, HNSWSwapJob *> idToSwapJob;

    // Todo: implement these methods later on
    void executeInsertJob(HNSWInsertJob *job);
    void executeRepairJob(HNSWRepairJob *job) {}

    // To be executed synchronously upon deleting a vector, doesn't require a wrapper.
    void executeSwapJob(HNSWSwapJob *job) {}

    // Wrappers static functions to be sent as callbacks upon creating the jobs (since members
    // functions cannot serve as callback, this serve as the "gateway" to the appropriate index).
    static void executeInsertJobWrapper(void *job);
    static void executeRepairJobWrapper(void *job) {}

    void submitSingleJob(AsyncJob *job);
    inline HNSWIndex<DataType, DistType> *getHNSWIndex();

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_tiered_tests_friends.h"
#endif

public:
    TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index, TieredIndexParams tieredParams);
    virtual ~TieredHNSWIndex();

    int addVector(const void *blob, labelType label, bool overwrite_allowed) override;
    size_t indexSize() const override;
    size_t indexLabelCount() const override;
    size_t indexCapacity() const override;
    // Do nothing here, each tier (flat buffer and HNSW) should increase capacity for itself when
    // needed.
    void increaseCapacity() override {}

    // TODO: Implement the actual methods instead of these temporary ones.
    int deleteVector(labelType id) override { return this->index->deleteVector(id); }
    double getDistanceFrom(labelType id, const void *blob) const override {
        return this->index->getDistanceFrom(id, blob);
    }
    VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                     VecSimQueryParams *queryParams) override {
        return this->index->topKQuery(queryBlob, k, queryParams);
    }
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
void TieredHNSWIndex<DataType, DistType>::executeInsertJobWrapper(void *job) {
    auto *insert_job = reinterpret_cast<HNSWInsertJob *>(job);
    reinterpret_cast<TieredHNSWIndex<DataType, DistType> *>(insert_job->index)
        ->executeInsertJob(insert_job);
}

template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::submitSingleJob(AsyncJob *job) {
    auto **jobs = array_new<AsyncJob *>(1);
    jobs = array_append(jobs, job);
    this->SubmitJobsToQueue(this->jobQueue, (void **)jobs, 1);
    array_free(jobs);
}

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType> * TieredHNSWIndex<DataType, DistType>::getHNSWIndex() {
    return reinterpret_cast<HNSWIndex<DataType, DistType>*>(this->index);
}


/******************** Job's callbacks **********************************/
template <typename DataType, typename DistType>
void TieredHNSWIndex<DataType, DistType>::executeInsertJob(HNSWInsertJob *job) {
    HNSWIndex<DataType, DistType> *hnsw_index = this->getHNSWIndex();
    this->flatIndexGuard.lock_shared();
    if (job->label == HNSW_INVALID_LABEL) {
        // Job has been invalidated in the meantime.
        this->flatIndexGuard.unlock_shared();
        goto finish;
    }
    // Acquire the index data lock, so we can immediately insert the vector to HNSW labels lookup.
    // If it becomes invalid afterwards, it will require another delete job.
    this->mainIndexGuard.lock_shared();
    hnsw_index->index_data_guard_.lock();

    // Check if resizing is needed for HNSW index (requires write lock).
    if (hnsw_index->indexCapacity() == hnsw_index->indexSize()) {
        hnsw_index->index_data_guard_.unlock();
        this->flatIndexGuard.unlock_shared();
        this->mainIndexGuard.unlock_shared();
        this->mainIndexGuard.lock();
        hnsw_index->increaseCapacity();
        this->mainIndexGuard.unlock();
        // reacquire the read locks
        this->flatIndexGuard.lock_shared();
        this->mainIndexGuard.lock_shared();
        hnsw_index->index_data_guard_.lock();
    }

    if (job->label == HNSW_INVALID_LABEL) {
        // Job has been invalidated in the meantime.
        this->flatIndexGuard.unlock_shared();
        this->mainIndexGuard.unlock_shared();
        goto finish;
    }
    // Set the label with a temporary invalid ID to indicate that the label exists (in case a
    // different thread will try to delete/overwrite this label).
    hnsw_index->setVectorId(job->label, HNSW_INVALID_ID);
    hnsw_index->index_data_guard_.unlock();

    // Take the vector from the flat buffer and insert it to HNSW (overwrite should not occur).
    hnsw_index->addVector(this->flatBuffer->getDataByInternalId(job->id), job->label, false);
    this->mainIndexGuard.unlock_shared();

    // Remove the vector and the insert job from the flat buffer.
    this->flatIndexGuard.unlock_shared();
    this->flatIndexGuard.lock();

    if (job->label != HNSW_INVALID_LABEL) {
        // Job has been invalidated in the meantime (vector was overwritten).
        this->flatBuffer->deleteVectorById(job->label, job->id);
    }
    finish:
    // Delete the job
    auto &jobs = labelToInsertJobs.at(job->label);
    for (size_t i = 0; i < jobs.size(); i++) {
        if (jobs[i]->id == job->id) {
            delete job;
            jobs.erase(jobs.begin() + (long)i);
            break;
        }
    }
    this->flatIndexGuard.unlock();
}


/******************** Index API ****************************************/

template <typename DataType, typename DistType>
TieredHNSWIndex<DataType, DistType>::TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index,
                                                     TieredIndexParams tieredParams)
    : VecSimTieredIndex<DataType, DistType>(hnsw_index, tieredParams),
      labelToInsertJobs(this->allocator), idToRepairJobs(this->allocator),
      idToSwapJob(this->allocator) {
    this->UpdateIndexMemory(this->memoryCtx, this->getAllocationSize());
}

template <typename DataType, typename DistType>
TieredHNSWIndex<DataType, DistType>::~TieredHNSWIndex() {
    // Delete all the pending insert jobs.
    for (auto jobs : this->labelToInsertJobs) {
        for (auto *job : jobs.second) {
            delete job;
        }
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
                                                   bool overwrite_allowed) {
    /* Note: this currently doesn't support overriding (assuming that the label doesn't exist)! */
    this->flatIndexGuard.lock();
    if (this->flatBuffer->indexCapacity() == this->flatBuffer->indexSize()) {
        this->flatBuffer->increaseCapacity();
    }
    idType new_id = this->flatBuffer->indexSize();
    this->flatBuffer->addVector(blob, label, false);
    AsyncJob *new_insert_job = new (this->allocator)
        HNSWInsertJob(this->allocator, label, new_id, executeInsertJobWrapper, this);
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

#pragma once

#include "VecSim/vec_sim_tiered_index.h"
#include "hnsw.h"
#include "hnsw_factory.h"

#include <unordered_map>

template <typename DataType, typename DistType>
class TieredHNSWIndex : public VecSimTieredIndex<DataType, DistType> {
private:
    /// Mappings from id/label to associated jobs, for invalidating and update ids if necessary.
    // In MULTI, we can have more than one insert job pending per label
    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<HNSWInsertJob *>> labelToInsertJobs;
    vecsim_stl::unordered_map<idType, vecsim_stl::vector<HNSWRepairJob *>> idToRepairJobs;
    vecsim_stl::unordered_map<idType, HNSWSwapJob *> idToSwapJob;

    // Todo: implement these methods later on
    void executeInsertJob(HNSWInsertJob *job) {}
    void executeRepairJob(HNSWRepairJob *job) {}

    // To be executed synchronously upon deleting a vector, doesn't require a wrapper.
    void executeSwapJob(HNSWSwapJob *job) {}

    // Wrappers static functions to be sent as callbacks upon creating the jobs (since members
    // functions cannot serve as callback, this serve as the "gateway" to the appropriate index).
    static void executeInsertJobWrapper(void *job);
    static void executeRepairJobWrapper(void *job) {}

    AsyncJob *createHNSWIngestJob(labelType label, idType internal_id);
    void submitSingleJob(AsyncJob *job);

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_tiered_tests_friends.h"
#endif

public:
    TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index, TieredIndexParams tieredParams);
    virtual ~TieredHNSWIndex() = default;

    int addVector(const void *blob, labelType label, bool overwrite_allowed) override;
    size_t indexSize() const override;
    size_t indexCapacity() const override;
    // Do nothing here, each tier (flat buffer and HNSW) should increase capacity for itself when
    // needed.
    void increaseCapacity() override {}

    // TODO: Implement the actual methods instead of these temporary ones.
    int deleteVector(labelType id) override { return this->index->deleteVector(id); }
    double getDistanceFrom(labelType id, const void *blob) const override {
        return this->index->getDistanceFrom(id, blob);
    }
    size_t indexLabelCount() const override { return this->index->indexLabelCount(); }
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
AsyncJob *TieredHNSWIndex<DataType, DistType>::createHNSWIngestJob(labelType label,
                                                                   idType internal_id) {
    return (AsyncJob *)new HNSWInsertJob{
        .base = AsyncJob{.jobType = HNSW_INSERT_VECTOR_JOB, .Execute = executeInsertJobWrapper},
        .index = this,
        .label = label,
        .id = internal_id};
}

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
size_t TieredHNSWIndex<DataType, DistType>::indexSize() const {
    return this->index->indexSize() + this->flatBuffer->indexSize();
}

template <typename DataType, typename DistType>
size_t TieredHNSWIndex<DataType, DistType>::indexCapacity() const {
    return this->index->indexCapacity() + this->flatBuffer->indexCapacity();
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
    AsyncJob *new_insert_job = this->createHNSWIngestJob(label, new_id);
    // Save a pointer to the job, so that if the vector is overwritten, we'll have an indication.
    auto jobs_vec =
        vecsim_stl::vector<HNSWInsertJob *>(1, (HNSWInsertJob *)new_insert_job, this->allocator);
    this->labelToInsertJobs.insert({label, jobs_vec});
    this->flatIndexGuard.unlock();

    // Insert job to the queue and signal the workers updater
    this->submitSingleJob(new_insert_job);
    this->UpdateIndexMemory(this->memoryCtx, this->getAllocationSize());
    return 1;
}

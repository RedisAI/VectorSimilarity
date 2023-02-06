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
    vecsim_stl::unordered_map<labelType, std::vector<HNSWInsertJob *>> labelToInsertJobs;
    vecsim_stl::unordered_map<idType, std::vector<HNSWRepairJob *>> idToRepairJobs;
    vecsim_stl::unordered_map<idType, HNSWSwapJob *> idToSwapJob;

    // Todo: implement these methods later on
    void executeInsertJob(HNSWInsertJob *job) {}
    void executeRepairJob(HNSWRepairJob *job) {}

    // To be executed synchronously upon deleting a vector, doesn't require a wrapper.
    void executeSwapJob(HNSWSwapJob *job) {}

    // Wrappers static functions to be sent as callbacks upon creating the jobs (since members
    // functions cannot serve as callback, this serve as the "gateway" to the appropriate index).
    static void executeInsertJobWrapper(void *job) {
        auto *insert_job = reinterpret_cast<HNSWInsertJob *>(job);
        reinterpret_cast<TieredHNSWIndex<DataType, DistType> *>(insert_job->index)
            ->executeInsertJob(insert_job);
    }
    static void executeRepairJobWrapper(void *job) {
        reinterpret_cast<TieredHNSWIndex<DataType, DistType> *>(
            reinterpret_cast<HNSWRepairJob *>(job)->index)->executeRepairJob(job);
    }

    AsyncJob *createHNSWIngestJob(labelType label, idType internal_id) {
        return (AsyncJob *) new HNSWInsertJob {.base = AsyncJob {.jobType = HNSW_INSERT_VECTOR_JOB,
                                                              .Execute = executeInsertJobWrapper},
                                             .index = this,
                                             .label = label,
                                             .id = internal_id};
    }

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_tiered_tests_friends.h"
#endif

public:
    TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index, TieredIndexParams tieredParams)
        : VecSimTieredIndex<DataType, DistType>(hnsw_index, tieredParams),
            labelToInsertJobs(hnsw_index->getIndexAllocator()),
          idToRepairJobs(hnsw_index->getIndexAllocator()),
          idToSwapJob(hnsw_index->getIndexAllocator()) {}
    virtual ~TieredHNSWIndex() = default;

    // TODO: Implement the actual methods instead of these temporary ones.
    int addVector(const void *blob, labelType label, bool overwrite_allowed) override;
    int deleteVector(labelType id) override { return this->index->deleteVector(id); }
    double getDistanceFrom(labelType id, const void *blob) const override {
        return this->index->getDistanceFrom(id, blob);
    }
    size_t indexSize() const override {
        return this->index->indexSize() + this->flatBuffer->indexSize();
    }
    size_t indexCapacity() const override { return this->index->indexCapacity(); }
    void increaseCapacity() override { this->index->increaseCapacity(); }
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
    this->labelToInsertJobs[label].push_back((HNSWInsertJob *)new_insert_job);
    this->flatIndexGuard.unlock();

    // Insert job to the queue and signal the workers updater
    auto **jobs = array_new<AsyncJob *>(1);
    jobs = array_append(jobs, new_insert_job);
    this->SubmitJobsToQueue(this->jobQueue, (void **)jobs, 1);
    this->UpdateIndexMemory(this->memoryCtx, this->getAllocationSize());
    return 1;
}

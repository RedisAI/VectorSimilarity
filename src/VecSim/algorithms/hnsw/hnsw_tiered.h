#pragma once

#include "VecSim/vec_sim_tiered_index.h"
#include "hnsw.h"
#include "hnsw_factory.h"

template <typename DataType, typename DistType>
class TieredHNSWIndex : public VecSimTieredIndex<DataType, DistType> {

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_tiered_tests_friends.h"
#endif

public:
    TieredHNSWIndex(HNSWIndex<DataType, DistType> *hnsw_index, void *job_queue_, SubmitCB submitCb,
                    void *memory_ctx, UpdateMemoryCB UpdateMemCb)
        : VecSimTieredIndex<DataType, DistType>(hnsw_index, job_queue_, submitCb, memory_ctx,
                                                UpdateMemCb) {}
    virtual ~TieredHNSWIndex() = default;

    // TODO: Implement the actual methods instead of these temporary ones.
    int addVector(const void *blob, labelType id) override {
        return this->index->addVector(blob, id);
    }
    int deleteVector(labelType id) override { return this->index->deleteVector(id); }
    double getDistanceFrom(labelType id, const void *blob) const override {
        return this->index->getDistanceFrom(id, blob);
    }
    size_t indexSize() const override { return this->index->indexSize(); }
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

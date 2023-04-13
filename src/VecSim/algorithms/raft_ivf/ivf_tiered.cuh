#pragma once

#include "VecSim/vec_sim_tiered_index.h"
#include "ivf_index_interface.h"
#include "ivf_factory.h"

#include <unordered_map>

class TieredRaftIvfIndex : public VecSimTieredIndex<float, float> {
    using DataType = float;
    using DistType = float;
public:
    TieredRaftIvfIndex(RaftIvfIndexInterface *ivf_index, TieredIndexParams tieredParams)
        : VecSimTieredIndex<DataType, DistType>(ivf_index, tieredParams) {}
    virtual ~TieredRaftIvfIndex() = default;

    // TODO: Implement the actual methods instead of these temporary ones.
    int addVector(const void *blob, labelType label, bool overwrite_allowed) override {
        return this->flatBuffer->addVector(blob, label, overwrite_allowed);
    }
    int deleteVector(labelType id) override { return this->flatBuffer->deleteVector(id); }
    double getDistanceFrom(labelType id, const void *blob) const override {
        return this->index->getDistanceFrom(id, blob);
    }
    size_t indexSize() const override { return this->index->indexSize(); }
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

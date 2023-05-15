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
        : VecSimTieredIndex<DataType, DistType>(ivf_index, tieredParams), ivf_index_(ivf_index) {}
    virtual ~TieredRaftIvfIndex() = default;

    // TODO: Implement the actual methods instead of these temporary ones.
    int addVector(const void *blob, labelType label, bool overwrite_allowed) override {
        updateIvfIndex = true;
        return this->flatBuffer->addVector(blob, label, overwrite_allowed);
    }
    int deleteVector(labelType id) override {
        updateIvfIndex = true;
        return this->flatBuffer->deleteVector(id);
    }
    double getDistanceFrom(labelType id, const void *blob) const override {
        return this->flatBuffer->getDistanceFrom(id, blob);
    }
    size_t indexSize() const override { return this->index->indexSize(); }
    size_t indexCapacity() const override { return this->index->indexCapacity(); }
    void increaseCapacity() override { this->index->increaseCapacity(); }
    size_t indexLabelCount() const override { return this->index->indexLabelCount(); }
    VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                     VecSimQueryParams *queryParams) override {
        if (updateIvfIndex)
            transferToIvf();
        return this->index->topKQuery(queryBlob, k, queryParams);
    }
    VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius,
                                      VecSimQueryParams *queryParams) override {
        return this->flatBuffer->rangeQuery(queryBlob, radius, queryParams);
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
    void transferToIvf() {
        auto dim = this->index->getDim();
        const auto &vectorBlocks = this->flatBuffer->getVectorBlocks();
        auto vectorDataGpuBuffer = raft::make_device_matrix<DataType, std::int64_t>(
            res_, this->flatBuffer->indexSize(), dim);
        auto labels_gpu = raft::make_device_matrix<std::int64_t, std::int64_t>(
            res_, this->flatBuffer->indexSize(), 1);
        std::int64_t offset = 0;
        for (int block_id = 0; block_id < vectorBlocks.size(); block_id++) {
            RAFT_CUDA_TRY(
                cudaMemcpyAsync(vectorDataGpuBuffer.data_handle() + offset * dim * sizeof(float),
                                vectorBlocks[block_id]->getVector(0),
                                vectorBlocks[block_id]->getLength() * dim * sizeof(float),
                                cudaMemcpyDefault, res_.get_stream()));
            auto label_original = this->flatBuffer->getLabels();
            auto label_converted =
                std::vector<std::int64_t>(label_original.begin(), label_original.end());
            RAFT_CUDA_TRY(cudaMemcpyAsync(labels_gpu.data_handle() + offset, label_converted.data(),
                                          label_original.size() * sizeof(std::int64_t),
                                          cudaMemcpyDefault, res_.get_stream()));

            offset += vectorBlocks[block_id]->getLength();
        }
        this->ivf_index_->addVectorBatchGpuBuffer(vectorDataGpuBuffer.data_handle(),
                                                  labels_gpu.data_handle(),
                                                  this->flatBuffer->indexSize());
        updateIvfIndex = false;
    }

private:
    raft::device_resources res_;
    RaftIvfIndexInterface *ivf_index_;
    bool updateIvfIndex;
};

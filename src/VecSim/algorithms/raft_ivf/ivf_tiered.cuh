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
    size_t indexSize() const override { return this->flatBuffer->indexSize(); }
    size_t indexCapacity() const override { return this->flatBuffer->indexCapacity(); }
    void increaseCapacity() override { this->flatBuffer->increaseCapacity(); }
    size_t indexLabelCount() const override { return this->flatBuffer->indexLabelCount(); }
    VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                     VecSimQueryParams *queryParams) override {
        // Use flatbuffer if the dataset is small
        if (this->flatBuffer->indexSize() < this->ivf_index_->nLists())
            return this->flatBuffer->topKQuery(queryBlob, k, queryParams);

        if (updateIvfIndex)
            transferToIvf();
        return this->index->topKQuery(queryBlob, k, queryParams);
    }
    VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius,
                                      VecSimQueryParams *queryParams) override {
        return this->flatBuffer->rangeQuery(queryBlob, radius, queryParams);
    }
    VecSimIndexInfo info() const override { return this->index->info(); }
    VecSimInfoIterator *infoIterator() const override { return this->flatBuffer->infoIterator(); }
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        return this->flatBuffer->newBatchIterator(queryBlob, queryParams);
    }
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override {
        return this->flatBuffer->preferAdHocSearch(subsetSize, k, initial_check);
    }
    inline void setLastSearchMode(VecSearchMode mode) override {
        return this->flatBuffer->setLastSearchMode(mode);
    }
    void transferToIvf() {
        auto dim = this->index->getDim();
        auto nVectors = this->flatBuffer->indexSize();
        const auto &vectorBlocks = this->flatBuffer->getVectorBlocks();
        auto vectorDataGpuBuffer = raft::make_device_matrix<DataType, std::int64_t>(
            res_, nVectors, dim);
        auto labels_gpu = raft::make_device_vector<std::int64_t, std::int64_t>(
            res_, nVectors);
        auto label_original = this->flatBuffer->getLabels();
        auto label_converted =
            std::vector<std::int64_t>(label_original.begin(), label_original.begin() + nVectors);
        RAFT_CUDA_TRY(cudaMemcpyAsync(labels_gpu.data_handle(), label_converted.data(),
                                      label_converted.size() * sizeof(std::int64_t),
                                      cudaMemcpyDefault, res_.get_stream()));
        std::int64_t offset = 0;
        for (int block_id = 0; block_id < vectorBlocks.size(); block_id++) {
            if (vectorBlocks[block_id]->getLength() == 0)
                continue;
            RAFT_CUDA_TRY(
                cudaMemcpyAsync(vectorDataGpuBuffer.data_handle() + offset * dim,
                                vectorBlocks[block_id]->getVector(0),
                                vectorBlocks[block_id]->getLength() * dim * sizeof(float),
                                cudaMemcpyDefault, res_.get_stream()));

            offset += vectorBlocks[block_id]->getLength();
        }
        this->ivf_index_->addVectorBatchGpuBuffer(vectorDataGpuBuffer.data_handle(),
                                                  labels_gpu.data_handle(),
                                                  nVectors);
        updateIvfIndex = false;
    }

private:
    raft::device_resources res_;
    RaftIvfIndexInterface *ivf_index_;
    bool updateIvfIndex;
};

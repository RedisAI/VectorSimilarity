#pragma once

#include <mutex>
#include "VecSim/algorithms/raft_ivf/ivf.cuh"
#include "VecSim/vec_sim_tiered_index.h"

struct RAFTTransferJob : public AsyncJob {
    RAFTTransferJob(std::shared_ptr<VecSimAllocator> allocator,
                    JobCallback insertCb, VecSimIndex *index_)
        : AsyncJob{allocator, RAFT_TRANSFER_JOB, insertCb, index_}
    {
    }
};

template <typename DataType, typename DistType>
struct TieredRaftIVFIndex : public VecSimTieredIndex<DataType, DistType> {
    int addVector(const void *blob, labelType label, void *auxiliaryCtx) override {
        auto frontend_lock = std::scoped_lock(this->flatIndexGuard);
        auto result = this->frontendIndex->addVector(blob, label);
        if (this->frontendIndex->indexSize() >= this->flatBufferLimit) {
            transferToBackend();
        }
        return result;
    }

    int deleteVector(labelType label) override {
        // TODO(wphicks)
        // If in flatIndex, delete
        // If being transferred to backend, wait for transfer
        // If in backendIndex, delete
        return 0;
    }

    size_t indexSize() {
        auto frontend_lock = std::scoped_lock(this->flatIndexGuard);
        auto backend_lock = std::scoped_lock(this->mainIndexGuard);
        return (getBackendIndex().indexSize() + this->frontendIndex.indexSize());
    }

    size_t indexLabelCount() const override {
        // TODO(wphicks) Count unique labels between both indexes
    }

    size_t indexCapacity() const override {
        return (getBackendIndex().indexCapacity() + this->flatBufferLimit);
    }

    void increaseCapacity() override { getBackendIndex().increaseCapacity(); }

    double getDistanceFrom_Unsafe(labelType label, const void *blob) const override {
        auto frontend_lock = std::unique_lock(this->flatIndexGuard);
        auto flat_dist = this->frontendIndex->getDistanceFrom_Unsafe(label, blob);
        frontend_lock.unlock();
        auto backend_lock = std::scoped_lock(this->mainIndexGuard);
        auto raft_dist = getBackendIndex().getDistanceFrom_Unsafe(label, blob);
        return std::fmin(flat_dist, raft_dist);
    }

    void executeTransferJob(RAFTTransferJob *job) { transferToBackend(); }

private:
    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<RAFTTransferJob *>> labelToTransferJobs;
    auto &getBackendIndex() {
        return *dynamic_cast<IVFIndex<DataType, DistType> *>(this->backendIndex);
    }

    void transferToBackend() {
        auto frontend_lock = std::unique_lock(this->flatIndexGuard);
        auto nVectors = this->flatBuffer->indexSize();
        if (nVectors == 0) {
            frontend_lock.unlock();
            return;
        }
        auto dim = this->index->getDim();
        const auto &vectorBlocks = this->flatBuffer->getVectorBlocks();
        auto vectorData = raft::make_host_matrix(getBackendIndex().get_resources(), nVectors, dim);

        auto *out = vectorData.data_handle();
        for (auto block_id = 0; block_id < vectorBlocks.size(); ++block_id) {
            auto *in_begin = reinterpret_cast<DataType *>(vectorBlocks[block_id].getElement(0));
            auto length = vectorBlocks[block_id].getLength();
            std::copy(in_begin, in_begin + length, out);
            out += length;
        }

        auto backend_lock = std::scoped_lock(this->mainIndexGuard);
        this->flatBuffer->clear();
        frontend_lock.unlock();
        getBackendIndex().addVectorBatch(vectorData.data_handle(), this->flatBuffer->getLabels(),
                                         nVectors);
    }
};

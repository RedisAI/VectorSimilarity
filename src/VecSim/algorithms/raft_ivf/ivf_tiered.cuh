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
struct TieredRaftIvfIndex : public VecSimTieredIndex<DataType, DistType> {
    TieredRaftIvfIndex(RaftIvfIndex<DataType, DistType>* raftIvfIndex,
                       BruteForceIndex<DataType, DistType> *bf_index,
                       const TieredIndexParams &tieredParams,
                       std::shared_ptr<VecSimAllocator> allocator)
        : VecSimTieredIndex<DataType, DistType>(raftIvfIndex, bf_index, tieredParams, allocator)
    {
        assert(raftIvfIndex->nLists() < this->flatBufferLimit &&
               "The flat buffer limit must be greater than the number of lists in the backend index");
    }
    ~TieredRaftIvfIndex() {
        // Delete all the pending jobs
    }

    int addVector(const void *blob, labelType label, void *auxiliaryCtx) override {
        int ret = 1;
        // If the flat index is full, write to the backend index
        if (this->frontendIndex->indexSize() >= this->flatBufferLimit) {
            // If the backend index is empty, build it with all the vectors
            // Otherwise, just add the vector to the backend index
            if (this->backendIndex->indexSize() == 0) {
                executeTransferJob();
            } else {
                this->mainIndexGuard.lock();
                ret = this->backendIndex->addVector(blob, label);
                this->mainIndexGuard.unlock();
                return ret;
            }
        }

        // Add the vector to the flat index
        this->flatIndexGuard.lock();
        ret = this->frontendIndex->addVector(blob, label);
        this->flatIndexGuard.unlock();

        // Submit a transfer job
        AsyncJob *new_insert_job = new (this->allocator)
            RAFTTransferJob(this->allocator, executeTransferJobWrapper, this);
        this->submitSingleJob(new_insert_job);
        return ret;
    }

    int deleteVector(labelType label) override {
        this->flatIndexGuard.lock();
        auto result = this->frontendIndex->deleteVector(label);
        this->mainIndexGuard.lock();
        this->flatIndexGuard.unlock();
        result += this->backendIndex->deleteVector(label);
        this->mainIndexGuard.unlock();
        return result;
    }

    size_t indexSize() const override {
        this->flatIndexGuard.lock_shared();
        this->mainIndexGuard.lock_shared();
        size_t result = (this->backendIndex->indexSize() + this->frontendIndex->indexSize());
        this->flatIndexGuard.unlock_shared();
        this->mainIndexGuard.unlock_shared();
        return result;
    }

    size_t indexLabelCount() const override {
        // TODO(wphicks) Count unique labels between both indexes
    }

    size_t indexCapacity() const override {
        return (this->backendIndex->indexCapacity() + this->frontendIndex->indexCapacity());
    }

    double getDistanceFrom_Unsafe(labelType label, const void *blob) const override {
        auto flat_dist = this->frontendIndex->getDistanceFrom_Unsafe(label, blob);
        auto raft_dist = getBackendIndex().getDistanceFrom_Unsafe(label, blob);
        return std::fmin(flat_dist, raft_dist);
    }

    static void executeTransferJobWrapper(AsyncJob *job) {
        if (job->isValid) {
            auto *transfer_job = reinterpret_cast<RAFTTransferJob *>(job);
            auto *job_index = reinterpret_cast<TieredRaftIvfIndex<DataType, DistType> *>(transfer_job->index);
            job_index->executeTransferJob();
        }
        delete job;
    }

    VecSimIndexBasicInfo basicInfo() const override{}

    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
        VecSimQueryParams *queryParams) const override {}

    inline void setLastSearchMode(VecSearchMode mode) override {}

    void runGC() override {}

    void acquireSharedLocks() override {
        this->flatIndexGuard.lock_shared();
        this->mainIndexGuard.lock_shared();
    }

    void releaseSharedLocks() override {
        this->flatIndexGuard.unlock_shared();
        this->mainIndexGuard.unlock_shared();
    }

private:

    inline auto &getBackendIndex() const {
        return *dynamic_cast<RaftIvfIndex<DataType, DistType> *>(this->backendIndex);
    }

    void executeTransferJob() {
        auto frontend_lock = std::unique_lock(this->flatIndexGuard);
        auto nVectors = this->frontendIndex->indexSize();
        // No vectors to transfer
        if (nVectors == 0) {
            frontend_lock.unlock();
            return;
        }

        // If the backend index is empty, don't transfer less than nLists vectors
        this->mainIndexGuard.lock_shared();
        auto main_nVectors = this->backendIndex->indexSize();
        this->mainIndexGuard.unlock_shared();
        if (main_nVectors == 0) {
            if (nVectors < getBackendIndex().nLists()) {
                frontend_lock.unlock();
                return;
            }
        }
        auto dim = this->backendIndex->getDim();
        const auto &vectorBlocks = this->frontendIndex->getVectorBlocks();
        auto* vectorData = (DataType *)this->allocator->allocate(nVectors * dim * sizeof (DataType));

        // Transfer vectors to a contiguous buffer
        auto *curr_ptr = vectorData;
        for (auto block_id = 0; block_id < vectorBlocks.size(); ++block_id) {
            const auto *in_begin = reinterpret_cast<const DataType *>(vectorBlocks[block_id].getElement(0));
            auto length = vectorBlocks[block_id].getLength();
            std::copy(in_begin, in_begin + (length * dim), curr_ptr);
            curr_ptr += length * dim;
        }

        // Add the vectors to the backend index
        auto backend_lock = std::scoped_lock(this->mainIndexGuard);
        getBackendIndex().addVectorBatch(vectorData, this->frontendIndex->getLabels().data(),
                                         nVectors);
        this->frontendIndex->clear();
        frontend_lock.unlock();
        this->allocator->free_allocation(vectorData);
    }
};

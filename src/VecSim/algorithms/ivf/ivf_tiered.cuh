#include <mutex>
#include "VecSim/algorithms/ivf/ivf.cuh"
#include "VecSim/vec_sim_tiered_index.h"

struct RAFTTransferJob : public AsyncJob {
  bool overwrite_allowed{true};
  RAFTTransferJob(
    std::shared_ptr<VecSimAllocator> allocator, bool overwrite_allowed_,
    JobCallback insertCb,
    VecSimIndex *index_
  ) : AsyncJob{allocator, RAFT_TRANSFER_JOB, insertCb, index_},
    overwrite_allowed{overwrite_allowed_} {}
};

template <typename DataType, typename DistType>
struct TieredIVFIndex : public VecSimTieredIndex<DataType, DistType> {
  auto addVector(
    const void* blob, labelType label, bool overwrite_allowed
  ) {
    auto frontend_lock = std::scoped_lock(this->flatIndexGuard);
    auto result = this->frontendIndex->addVector(
      blob, label, overwrite_allowed
    );
    if (this->frontendIndex->indexSize() >= this->flatBufferLimit) {
      transferToBackend(overwrite_allowed);
    }
    return result;
  }

  auto deleteVector(labelType label) {
    // TODO(wphicks)
    // If in flatIndex, delete
    // If being transferred to backend, wait for transfer
    // If in backendIndex, delete
  }

  auto indexSize() {
    auto frontend_lock = std::scoped_lock(this->flatIndexGuard);
    auto backend_lock = std::scoped_lock(this->mainIndexGuard);
    return (
      getBackendIndex().indexSize() + this->frontendIndex.indexSize()
    );
  }

  auto indexLabelCount() const override {
    // TODO(wphicks) Count unique labels between both indexes
  }

  auto indexCapacity() const override {
    return (
      getBackendIndex().indexCapacity() +
      this->flatBufferLimit
    );
  }

  void increaseCapacity() override {
    getBackendIndex().increaseCapacity();
  }

  auto getDistanceFrom(labelType label, const void* blob) {
    auto frontend_lock = std::unique_lock(this->flatIndexGuard);
    auto flat_dist = this->frontendIndex->getDistanceFrom(label, blob);
    frontend_lock.unlock();
    auto backend_lock = std::scoped_lock(this->mainIndexGuard);
    auto raft_dist = getBackendIndex().getDistanceFrom(label, blob);
    return std::fmin(flat_dist, raft_dist);
  }

  void executeTransferJob(RAFTTransferJob* job) {
    transferToBackend(job->overwrite_allowed);
  }

 private:
  vecsim_stl::unordered_map<labelType, vecsim_stl::vector<RAFTTransferJob *>> labelToTransferJobs;
  auto& getBackendIndex() {
    return *dynamic_cast<IVFIndex<DataType, DistType>*>(
      this->backendIndex
    );
  }

  void transferToBackend(overwrite_allowed=true) {
	auto dim = this->index->getDim();
    auto frontend_lock = std::unique_lock(this->flatIndexGuard);
	auto nVectors = this->flatBuffer->indexSize();
	const auto &vectorBlocks = this->flatBuffer->getVectorBlocks();
    auto vectorData = raft::make_host_matrix(
      getBackendIndex().get_resources(),
      nVectors,
      dim
    );

    auto* out = vectorData.data_handle();
    for (auto block_id = 0; block_id < vectorBlocks.size(); ++block_id) {
      auto* in_begin = reinterpret_cast<DataType*>(
        vectorBlocks[block_id].getElement(0)
      );
      auto length = vectorBlocks[block_id].getLength();
      std::copy(
        in_begin,
        in_begin + length,
        out
      );
      out += length;
    }

    auto backend_lock = std::scoped_lock(this->mainIndexGuard);
    this->flatBuffer->clear();
    frontend_lock.unlock();
    getBackendIndex().addVectorBatch(
      vectorData.data_handle(),
      this->flatBuffer->getLabels(),
      nVectors,
      overwrite_allowed
    );
  }
};

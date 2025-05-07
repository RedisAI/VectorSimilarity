#pragma once
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/vec_sim_tiered_index.h"
#include "VecSim/algorithms/svs/svs.h"
#include "VecSim/index_factories/svs_factory.h"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

/**
 * @class SVSMultiThreadJob
 * @brief Represents a multi-threaded asynchronous job for the SVS algorithm.
 *
 * This class is responsible for managing multi-threaded jobs, including thread reservation,
 * synchronization, and execution of tasks. It uses a control block to coordinate threads
 * and ensure proper execution of the job.
 *
 * @details
 * The SVSMultiThreadJob class supports creating multiple threads for a task and ensures
 * synchronization between them. It uses a nested ControlBlock class to manage thread
 * reservations and job completion. Additionally, it includes a nested ReserveThreadJob
 * class to handle individual thread reservations.
 *
 * The main job executes a user-defined task with the number of reserved threads, while
 * additional threads wait for the main job to complete.
 *
 * @note This class is designed to work with the AsyncJob framework.
 */
class SVSMultiThreadJob : public AsyncJob {
public:
    class JobsRegistry {
        vecsim_stl::unordered_set<AsyncJob *> jobs;
        std::mutex m_jobs;

    public:
        JobsRegistry(const std::shared_ptr<VecSimAllocator> &allocator) : jobs(allocator) {}

        ~JobsRegistry() {
            std::lock_guard lock{m_jobs};
            for (auto job : jobs) {
                delete job;
            }
            jobs.clear();
        }

        void register_jobs(const vecsim_stl::vector<AsyncJob *> &jobs) {
            std::lock_guard lock{m_jobs};
            this->jobs.insert(jobs.begin(), jobs.end());
        }

        void delete_job(AsyncJob *job) {
            {
                std::lock_guard lock{m_jobs};
                jobs.erase(job);
            }
            delete job;
        }
    };

private:
    // Thread reservation control block shared between all threads
    // to reserve threads and wait for the job to be done
    // actual reserved threads can be less than requested if timeout is reached
    class ControlBlock {
        const size_t requestedThreads;           // number of threads requested to reserve
        const std::chrono::microseconds timeout; // timeout for threads reservation
        size_t reservedThreads;                  // number of threads reserved
        bool jobDone;
        std::mutex m_reserve;
        std::condition_variable cv_reserve;
        std::mutex m_done;
        std::condition_variable cv_done;

    public:
        template <typename Rep, typename Period>
        ControlBlock(size_t requested_threads,
                     std::chrono::duration<Rep, Period> threads_wait_timeout)
            : requestedThreads{requested_threads}, timeout{threads_wait_timeout},
              reservedThreads{0}, jobDone{false} {}

        // reserve a thread and wait for the job to be done
        void reserveThreadAndWait() {
            // count current thread
            {
                std::unique_lock lock{m_reserve};
                ++reservedThreads;
            }
            cv_reserve.notify_one();
            std::unique_lock lock{m_done};
            // Wait until the job is marked as done, handling potential spurious wakeups.
            cv_done.wait(lock, [&] { return jobDone; });
        }

        // wait for threads to be reserved
        // return actual number of reserved threads
        size_t waitForThreads() {
            std::unique_lock lock{m_reserve};
            ++reservedThreads; // count current thread
            cv_reserve.wait_for(lock, timeout, [&] { return reservedThreads >= requestedThreads; });
            return reservedThreads;
        }

        // mark the whole job as done
        void markJobDone() {
            {
                std::lock_guard lock{m_done};
                jobDone = true;
            }
            cv_done.notify_all();
        }
    };

    // Job to reserve a thread and wait for the job to be done
    class ReserveThreadJob : public AsyncJob {
        std::weak_ptr<ControlBlock> controlBlock; // control block is owned by the main job and can
                                                  // be destroyed before this job is started
        JobsRegistry *jobsRegistry;

        static void Execute_impl(AsyncJob *job) {
            auto *jobPtr = static_cast<ReserveThreadJob *>(job);
            // if control block is already destroyed by the update job, just delete the job
            auto controlBlock = jobPtr->controlBlock.lock();
            if (controlBlock) {
                controlBlock->reserveThreadAndWait();
            }
            jobPtr->jobsRegistry->delete_job(job);
        }

    public:
        ReserveThreadJob(std::shared_ptr<VecSimAllocator> allocator, JobType jobType,
                         VecSimIndex *index, std::weak_ptr<ControlBlock> controlBlock,
                         JobsRegistry *registry)
            : AsyncJob(std::move(allocator), jobType, Execute_impl, index),
              controlBlock(std::move(controlBlock)), jobsRegistry(registry) {}
    };

    using task_type = std::function<void(VecSimIndex *, size_t)>;
    task_type task;
    std::shared_ptr<ControlBlock> controlBlock;
    JobsRegistry *jobsRegistry;

    static void Execute_impl(AsyncJob *job) {
        auto *jobPtr = static_cast<SVSMultiThreadJob *>(job);
        auto controlBlock = jobPtr->controlBlock;
        size_t num_threads = 1;
        if (controlBlock) {
            num_threads = controlBlock->waitForThreads();
        }
        assert(num_threads > 0);
        jobPtr->task(jobPtr->index, num_threads);
        if (controlBlock) {
            jobPtr->controlBlock->markJobDone();
        }
        jobPtr->jobsRegistry->delete_job(job);
    }

    SVSMultiThreadJob(std::shared_ptr<VecSimAllocator> allocator, JobType jobType,
                      task_type callback, VecSimIndex *index,
                      std::shared_ptr<ControlBlock> controlBlock, JobsRegistry *registry)
        : AsyncJob(std::move(allocator), jobType, Execute_impl, index), task(std::move(callback)),
          controlBlock(std::move(controlBlock)), jobsRegistry(registry) {}

public:
    template <typename Rep, typename Period>
    static vecsim_stl::vector<AsyncJob *>
    createJobs(const std::shared_ptr<VecSimAllocator> &allocator, JobType jobType,
               std::function<void(VecSimIndex *, size_t)> callback, VecSimIndex *index,
               size_t num_threads, std::chrono::duration<Rep, Period> threads_wait_timeout,
               JobsRegistry *registry) {
        assert(num_threads > 0);
        std::shared_ptr<ControlBlock> controlBlock =
            num_threads == 1 ? nullptr
                             : std::make_shared<ControlBlock>(num_threads, threads_wait_timeout);

        vecsim_stl::vector<AsyncJob *> jobs(num_threads, allocator);
        jobs[0] = new (allocator)
            SVSMultiThreadJob(allocator, jobType, callback, index, controlBlock, registry);
        for (size_t i = 1; i < num_threads; ++i) {
            jobs[i] =
                new (allocator) ReserveThreadJob(allocator, jobType, index, controlBlock, registry);
        }
        registry->register_jobs(jobs);
        return jobs;
    }

#ifdef BUILD_TESTS
public:
    static constexpr size_t estimateSize(size_t num_threads) {
        return sizeof(SVSMultiThreadJob) + (num_threads - 1) * sizeof(ReserveThreadJob);
    }
#endif
};

template <typename DataType>
class TieredSVSIndex : public VecSimTieredIndex<DataType, float> {
    using Self = TieredSVSIndex<DataType>;
    using Base = VecSimTieredIndex<DataType, float>;
    using flat_index_t = BruteForceIndex_Single<DataType, float>;
    using backend_index_t = VecSimIndexAbstract<DataType, float>;
    using svs_index_t = SVSIndexBase;

    // Add: true, Delete: false
    using journal_record = std::pair<labelType, bool>;
    size_t updateJobThreshold;
    size_t updateJobWaitTime;
    std::vector<journal_record> journal;
    std::shared_mutex journal_mutex;
    std::atomic_flag indexUpdateScheduled = ATOMIC_FLAG_INIT;
    std::mutex updateJobMutex;

    // The reason of following container just to properly destroy jobs which not executed yet
    SVSMultiThreadJob::JobsRegistry uncompletedJobs;

    /// <batch_iterator>
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //  TieredSVS_BatchIterator //
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    class TieredSVS_BatchIterator : public VecSimBatchIterator {
        // Defining spacial values for the svs_iterator field, to indicate if the iterator is
        // uninitialized or depleted when we don't have a valid iterator.
        static constexpr VecSimBatchIterator *depleted() {
            constexpr VecSimBatchIterator *p = nullptr;
            return p + 1;
        }

    private:
        using Index = TieredSVSIndex<DataType>;
        const Index *index;
        VecSimQueryParams *queryParams;

        VecSimQueryResultContainer flat_results;
        VecSimQueryResultContainer svs_results;

        VecSimBatchIterator *flat_iterator;
        VecSimBatchIterator *svs_iterator;
        std::shared_lock<std::shared_mutex> svs_lock;

        // On single value indices, this set holds the IDs of the results that were returned from
        // the flat buffer.
        // On multi value indices, this set holds the IDs of all the results that were returned.
        // The difference between the two cases is that on multi value indices, the same ID can
        // appear in both indexes and results with different scores, and therefore we can't tell in
        // advance when we expect a possibility of a duplicate.
        // On single value indices, a duplicate may appear at the same batch (and we will handle it
        // when merging the results) Or it may appear in a different batches, first from the flat
        // buffer and then from the SVS, in the cases where a better result if found later in SVS
        // because of the approximate nature of the algorithm.
        vecsim_stl::unordered_set<labelType> returned_results_set;

    private:
        VecSimQueryReply *compute_current_batch(size_t n_res) {
            // Merge results
            // This call will update `svs_res` and `bf_res` to point to the end of the merged
            // results. results.
            auto batch_res = new VecSimQueryReply(allocator);
            auto [from_svs, from_flat] =
                merge_results<false>(batch_res->results, svs_results, flat_results, n_res);

            // We're on a single-value index, update the set of results returned from the FLAT index
            // before popping them, to prevent them to be returned from the SVS index in later
            // batches. batches.
            for (size_t i = 0; i < from_flat; ++i) {
                returned_results_set.insert(flat_results[i].id);
            }

            // Update results
            flat_results.erase(flat_results.begin(), flat_results.begin() + from_flat);
            svs_results.erase(svs_results.begin(), svs_results.begin() + from_svs);

            // Return current batch
            return batch_res;
        }

        void filter_irrelevant_results(VecSimQueryResultContainer &results) {
            // Filter out results that were already returned.
            const auto it = std::remove_if(results.begin(), results.end(), [this](const auto &r) {
                return returned_results_set.count(r.id) != 0;
            });
            results.erase(it, results.end());
        }

        void acquire_svs_iterator() {
            if (svs_iterator == nullptr) {
                this->index->mainIndexGuard.lock_shared();
                svs_iterator = index->backendIndex->newBatchIterator(
                    this->flat_iterator->getQueryBlob(), queryParams);
            }
        }

        void release_svs_iterator() {
            if (svs_iterator != nullptr && svs_iterator != depleted()) {
                delete svs_iterator;
                svs_iterator = nullptr;
                this->index->mainIndexGuard.unlock_shared();
            }
        }

        void handle_svs_depletion() {
            assert(svs_iterator != depleted());
            if (svs_iterator->isDepleted()) {
                release_svs_iterator();
                svs_iterator = depleted();
            }
        }

    public:
        TieredSVS_BatchIterator(void *query_vector, const Index *index,
                                VecSimQueryParams *queryParams,
                                std::shared_ptr<VecSimAllocator> allocator)
            : VecSimBatchIterator(query_vector, queryParams ? queryParams->timeoutCtx : nullptr,
                                  std::move(allocator)),
              index(index), flat_results(this->allocator), svs_results(this->allocator),
              flat_iterator(index->frontendIndex->newBatchIterator(query_vector, queryParams)),
              svs_iterator(nullptr), svs_lock(index->mainIndexGuard, std::defer_lock),
              returned_results_set(this->allocator) {
            if (queryParams) {
                this->queryParams =
                    (VecSimQueryParams *)this->allocator->allocate(sizeof(VecSimQueryParams));
                *this->queryParams = *queryParams;
            } else {
                this->queryParams = nullptr;
            }
        }

        ~TieredSVS_BatchIterator() {
            release_svs_iterator();
            if (queryParams) {
                this->allocator->free_allocation(queryParams);
            }
            delete flat_iterator;
        }

        VecSimQueryReply *getNextResults(size_t n_res, VecSimQueryReply_Order order) override {
            auto svs_code = VecSim_QueryReply_OK;

            if (svs_iterator == nullptr) { // first call
                // First call to getNextResults. The call to the BF iterator will include
                // calculating all the distances and access the BF index. We take the lock on this
                // call.
                auto cur_flat_results = [this, n_res]() {
                    std::shared_lock<std::shared_mutex> flat_lock{index->flatIndexGuard};
                    return flat_iterator->getNextResults(n_res, BY_SCORE_THEN_ID);
                }();
                // This is also the only time `getNextResults` on the BF iterator can fail.
                if (VecSim_OK != cur_flat_results->code) {
                    return cur_flat_results;
                }
                flat_results.swap(cur_flat_results->results);
                VecSimQueryReply_Free(cur_flat_results);
                // We also take the lock on the main index on the first call to getNextResults, and
                // we hold it until the iterator is depleted or freed.
                acquire_svs_iterator();
                auto cur_svs_results = svs_iterator->getNextResults(n_res, BY_SCORE_THEN_ID);
                svs_code = cur_svs_results->code;
                svs_results.swap(cur_svs_results->results);
                VecSimQueryReply_Free(cur_svs_results);
                handle_svs_depletion();
            } else {
                if (flat_results.size() < n_res && !flat_iterator->isDepleted()) {
                    auto tail = flat_iterator->getNextResults(n_res - flat_results.size(),
                                                              BY_SCORE_THEN_ID);
                    flat_results.insert(flat_results.end(), tail->results.begin(),
                                        tail->results.end());
                    VecSimQueryReply_Free(tail);
                }

                while (svs_results.size() < n_res && svs_iterator != depleted() &&
                       svs_code == VecSim_OK) {
                    auto tail =
                        svs_iterator->getNextResults(n_res - svs_results.size(), BY_SCORE_THEN_ID);
                    svs_code =
                        tail->code; // Set the svs_results code to the last `getNextResults` code.
                    // New batch may contain better results than the previous batch, so we need to
                    // merge. We don't expect duplications (hence the <false>), as the iterator
                    // guarantees that no result is returned twice.
                    VecSimQueryResultContainer cur_svs_results(this->allocator);
                    merge_results<false>(cur_svs_results, svs_results, tail->results, n_res);
                    VecSimQueryReply_Free(tail);
                    svs_results.swap(cur_svs_results);
                    filter_irrelevant_results(svs_results);
                    handle_svs_depletion();
                }
            }

            if (VecSim_OK != svs_code) {
                return new VecSimQueryReply(this->allocator, svs_code);
            }

            VecSimQueryReply *batch;
            batch = compute_current_batch(n_res);

            if (order == BY_ID) {
                sort_results_by_id(batch);
            }
            size_t batch_len = VecSimQueryReply_Len(batch);
            this->updateResultsCount(batch_len);

            return batch;
        }

        // DISCLAIMER: After the last batch, one of the iterators may report that it is not
        // depleted, while all of its remaining results were already returned from the other
        // iterator. (On single-value indexes, this can happen to the svs iterator only, on
        // multi-value
        //  indexes, this can happen to both iterators).
        // The next call to `getNextResults` will return an empty batch, and then the iterators will
        // correctly report that they are depleted.
        bool isDepleted() override {
            return flat_results.empty() && flat_iterator->isDepleted() && svs_results.empty() &&
                   svs_iterator == depleted();
        }

        void reset() override {
            release_svs_iterator();
            resetResultsCount();
            flat_iterator->reset();
            svs_iterator = nullptr;
            flat_results.clear();
            svs_results.clear();
            returned_results_set.clear();
        }
    };

    /// <batch_iterator>

#ifdef BUILD_TESTS
public:
#endif
    flat_index_t *GetFlatIndex() {
        auto result = dynamic_cast<flat_index_t *>(this->frontendIndex);
        assert(result);
        return result;
    }

    svs_index_t *GetSVSIndex() {
        auto result = dynamic_cast<svs_index_t *>(this->backendIndex);
        assert(result);
        return result;
    }

#ifdef BUILD_TESTS
public:
    backend_index_t *GetBackendIndex() { return this->backendIndex; }
    void submitSingleJob(AsyncJob *job) { Base::submitSingleJob(job); }
    void submitJobs(vecsim_stl::vector<AsyncJob *> &jobs) { Base::submitJobs(jobs); }
#endif

private:
    void TakeSnapshot(std::set<labelType> *to_delete, std::set<labelType> *to_add) {
        std::vector<journal_record> journal_snapshot;
        journal_snapshot.reserve(this->updateJobThreshold * 2);

        { // Get current journal and replace with empty
            std::scoped_lock journal_lock{journal_mutex};
            std::swap(this->journal, journal_snapshot);
        }

        for (auto &p : journal_snapshot) {
            // `id` is the label and `add` is a boolean indicating addition (true) or deletion
            // (false)
            const auto [id, add] = p;
            if (add) {
                to_add->insert(id);
            } else {
                to_delete->insert(id);
                to_add->erase(id); // add non-deleted only
            }
        }
    }

    static void updateSVSIndexWrapper(VecSimIndex *idx, size_t availableThreads) {
        auto index = static_cast<TieredSVSIndex<DataType> *>(idx);
        assert(index);
        // prevent parallel updates
        std::lock_guard<std::mutex> lock(index->updateJobMutex);
        // Release the scheduled flag to allow scheduling again
        index->indexUpdateScheduled.clear();
        // Update the SVS index
        index->GetSVSIndex()->setNumThreads(availableThreads);
        index->updateSVSIndex();
    }

#ifdef BUILD_TESTS
public:
#endif
    void scheduleSVSIndexUpdate() {
        // do not schedule if scheduled already
        if (indexUpdateScheduled.test_and_set()) {
            return;
        }

        auto total_threads = this->GetSVSIndex()->getThreadPoolCapacity();
        auto jobs = SVSMultiThreadJob::createJobs(
            this->allocator, HNSW_INSERT_VECTOR_JOB, updateSVSIndexWrapper, this, total_threads,
            std::chrono::microseconds(updateJobWaitTime), &uncompletedJobs);
        this->submitJobs(jobs);
    }

private:
    void updateSVSIndex() {
        std::set<labelType> to_delete;
        std::set<labelType> to_add;
        TakeSnapshot(&to_delete, &to_add);

        std::vector<labelType> labels_to_delete;
        std::vector<labelType> labels_to_add;
        std::vector<DataType> vectors_to_add;

        { // lock frontendIndex from modifications
            std::shared_lock<std::shared_mutex> frontend_lock{this->flatIndexGuard};

            auto flat_index = this->GetFlatIndex();
            const size_t dim = flat_index->getDim();

            // Update snapshot to sync with current frontend index status
            TakeSnapshot(&to_delete, &to_add);

            labels_to_delete.reserve(to_delete.size());
            labels_to_add.reserve(to_add.size());
            vectors_to_add.reserve(to_add.size() * dim);

            labels_to_delete.insert(labels_to_delete.end(), to_delete.begin(), to_delete.end());
            labels_to_add.insert(labels_to_add.end(), to_add.begin(), to_add.end());
            for (auto label : labels_to_add) {
                if (this->frontendIndex->isLabelExists(label)) {
                    const auto id = flat_index->getIdOfLabel(label);
                    if (id != INVALID_ID) {
                        auto data = flat_index->getDataByInternalId(id);
                        vectors_to_add.insert(vectors_to_add.end(), data, data + dim);
                    }
                }
            }
        } // release frontend index

        { // lock both indicies for writing - these changes to be synchronized
            std::scoped_lock lock(this->flatIndexGuard, this->mainIndexGuard);
            auto svs_index = GetSVSIndex();
            auto deleted_num =
                svs_index->deleteVectors(labels_to_delete.data(), labels_to_delete.size());
            assert(deleted_num == 0);
            assert(labels_to_add.size() == vectors_to_add.size() / this->frontendIndex->getDim());
            svs_index->addVectors(vectors_to_add.data(), labels_to_add.data(),
                                  labels_to_add.size());

            // clean-up frontend index
            { // avoid deleting modified vectors
                std::shared_lock<std::shared_mutex> journal_lock{this->journal_mutex};
                for (auto &p : this->journal) {
                    to_add.erase(p.first);
                }
            }

            // erase moved vectors
            size_t deleted = 0;
            for (auto &label : to_add) {
                deleted += this->frontendIndex->deleteVector(label);
            }
            assert(deleted == to_add.size());
        }
    }

public:
    TieredSVSIndex(VecSimIndexAbstract<DataType, float> *svs_index, flat_index_t *bf_index,
                   const TieredIndexParams &tiered_index_params,
                   std::shared_ptr<VecSimAllocator> allocator)
        : Base(svs_index, bf_index, tiered_index_params, allocator),
          updateJobThreshold(
              tiered_index_params.specificParams.tieredSVSParams.updateJobThreshold == 0
                  ? DEFAULT_PENDING_SWAP_JOBS_THRESHOLD
                  : std::min(tiered_index_params.specificParams.tieredSVSParams.updateJobThreshold,
                             MAX_PENDING_SWAP_JOBS_THRESHOLD)),
          updateJobWaitTime(
              tiered_index_params.specificParams.tieredSVSParams.updateJobWaitTime == 0
                  ? 1000 // default wait time: 1ms
                  : tiered_index_params.specificParams.tieredSVSParams.updateJobWaitTime),
          uncompletedJobs(this->allocator) {
        this->journal.reserve(this->updateJobThreshold * 2);
    }

    int addVector(const void *blob, labelType label) override {
        int ret = 0;
        auto svs_index = GetSVSIndex();
        if (this->getWriteMode() == VecSim_WriteInPlace) {
            // Use the frontend parameters to manually prepare the blob for its transfer to the SVS
            // index.
            auto storage_blob = this->frontendIndex->preprocessForStorage(blob);
            std::scoped_lock lock(this->updateJobMutex, this->mainIndexGuard);
            return svs_index->addVectors(storage_blob.get(), &label, 1);
        }
        bool index_update_needed = false;
        {
            std::scoped_lock lock(this->flatIndexGuard, this->mainIndexGuard, this->journal_mutex);
            ret = this->frontendIndex->addVector(blob, label);
            ret = std::max(ret - svs_index->deleteVectors(&label, 1), 0);
            journal.emplace_back(label, true);
            index_update_needed = this->backendIndex->indexSize() > 0 ||
                                  this->journal.size() >= this->updateJobThreshold;
        }

        if (index_update_needed) {
            scheduleSVSIndexUpdate();
        }

        return ret;
    }

    int deleteVector(labelType label) override {
        int ret = 0;
        auto svs_index = GetSVSIndex();
        if (this->getWriteMode() == VecSim_WriteInPlace) {
            assert([&] {
                std::shared_lock<std::shared_mutex> flat_lock(this->flatIndexGuard);
                return !this->frontendIndex->isLabelExists(label);
            }());

            std::scoped_lock lock(this->updateJobMutex, this->mainIndexGuard);
            return svs_index->deleteVectors(&label, 1);
        }

        bool label_exists = [&] {
            std::shared_lock<std::shared_mutex> flat_lock(this->flatIndexGuard);
            return this->frontendIndex->isLabelExists(label);
        }();

        bool index_update_needed = false;
        if (label_exists) {
            std::scoped_lock lock(this->flatIndexGuard, this->mainIndexGuard, this->journal_mutex);
            if (this->frontendIndex->isLabelExists(label)) {
                ret = this->frontendIndex->deleteVector(label);
                assert(ret == 1 && "unexpected deleteVector result");
            }
            ret += svs_index->deleteVectors(&label, 1);
            assert(ret < 2 && "deleteVector: vector duplication in both indices");
            journal.emplace_back(label, false);
            index_update_needed = this->journal.size() >= this->updateJobThreshold;
        } else {
            std::scoped_lock lock(this->mainIndexGuard);
            ret += svs_index->deleteVectors(&label, 1);
        }

        if (index_update_needed) {
            scheduleSVSIndexUpdate();
        }
        return ret;
    }

    size_t indexSize() const override {
        std::shared_lock<std::shared_mutex> flat_lock(this->flatIndexGuard);
        std::shared_lock<std::shared_mutex> main_lock(this->mainIndexGuard);
        return this->frontendIndex->indexSize() + this->backendIndex->indexSize();
    }

    size_t indexLabelCount() const override {
        std::shared_lock<std::shared_mutex> flat_lock(this->flatIndexGuard);
        std::shared_lock<std::shared_mutex> main_lock(this->mainIndexGuard);
        return this->frontendIndex->indexLabelCount() + this->backendIndex->indexLabelCount();
    }
    size_t indexCapacity() const override {
        std::shared_lock<std::shared_mutex> flat_lock(this->flatIndexGuard);
        std::shared_lock<std::shared_mutex> main_lock(this->mainIndexGuard);
        return this->frontendIndex->indexCapacity() + this->backendIndex->indexCapacity();
    }

    double getDistanceFrom_Unsafe(labelType label, const void *blob) const override {
        double flat_dist = std::numeric_limits<double>::quiet_NaN();
        {
            std::shared_lock<decltype(this->flatIndexGuard)> lock(this->flatIndexGuard);
            flat_dist = this->frontendIndex->getDistanceFrom_Unsafe(label, blob);
        }
        if (!std::isnan(flat_dist)) {
            return flat_dist;
        } else {
            std::shared_lock<decltype(this->mainIndexGuard)> lock(this->mainIndexGuard);
            return this->backendIndex->getDistanceFrom_Unsafe(label, blob);
        }
    }

    VecSimIndexDebugInfo debugInfo() const override {
        auto info = Base::debugInfo();
        return info;
    }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->backendIndex->getBasicInfo();
        info.blockSize = info.blockSize;
        info.isTiered = true;
        info.algo = VecSimAlgo_SVS;
        return info;
    }

    VecSimDebugInfoIterator *debugInfoIterator() const override {
        //  Get the base tiered fields.
        auto *infoIterator = Base::debugInfoIterator();
        return infoIterator;
    }

    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        size_t blobSize = this->backendIndex->getDim() * sizeof(DataType);
        void *queryBlobCopy = this->allocator->allocate(blobSize);
        memcpy(queryBlobCopy, queryBlob, blobSize);
        return new (this->allocator)
            TieredSVS_BatchIterator(queryBlobCopy, this, queryParams, this->allocator);
    }

    void setLastSearchMode(VecSearchMode mode) override {
        return this->backendIndex->setLastSearchMode(mode);
    }

    void runGC() override {
        // Run no more than pendingSwapJobsThreshold value jobs.
        TIERED_LOG(VecSimCommonStrings::LOG_VERBOSE_STRING,
                   "running asynchronous GC for tiered SVS index");
        if (!indexUpdateScheduled.test_and_set()) {
            updateSVSIndexWrapper(this, 1);
        }
        std::unique_lock<std::shared_mutex> backend_lock{this->mainIndexGuard};
        // VecSimIndexAbstract::runGC() is protected
        static_cast<VecSimIndexInterface *>(this->backendIndex)->runGC();
    }

    void acquireSharedLocks() override {
        this->flatIndexGuard.lock_shared();
        this->mainIndexGuard.lock_shared();
    }

    void releaseSharedLocks() override {
        this->flatIndexGuard.unlock_shared();
        this->mainIndexGuard.unlock_shared();
    }
};

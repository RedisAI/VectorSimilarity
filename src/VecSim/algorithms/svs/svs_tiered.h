#pragma once
#include "VecSim/vec_sim_common.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/vec_sim_tiered_index.h"
#include "VecSim/algorithms/svs/svs.h"
#include "VecSim/index_factories/svs_factory.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>

/**
 * Definition of a job that inserts a new vector from flat into SVS Index.
 */
struct SVSInsertJob : public TieredInsertJob {
    // Pending -> Executing -> Done. Delayed means a relabel is remapping the label in the backend
    // right now, and publishing would put the vector under a label that is not settled yet.
    // Written under flatIndexGuard, except Executing/Done which waitForJob() reads unlocked.
    enum class Status : uint8_t { Pending, Delayed, Executing, Done };
    std::atomic<Status> status{Status::Pending};

    SVSInsertJob(std::shared_ptr<VecSimAllocator> allocator, labelType label_, idType id_,
                 JobCallback insertCb, VecSimIndex *index_)
        : TieredInsertJob(allocator, SVS_INSERT_VECTOR_JOB, label_, id_, insertCb, index_) {}
};

/**
 * Definition of a job that launches partial consolidation on SVS Index.
 */
struct SVSConsolidateJob : public AsyncJob {
    std::vector<labelType> labels;
    std::atomic<bool> executing{false};

    SVSConsolidateJob(std::shared_ptr<VecSimAllocator> allocator,
                      const std::vector<labelType> &labels_, JobCallback insertCb,
                      VecSimIndex *index_)
        : AsyncJob(allocator, SVS_CONSOLIDATE_JOB, insertCb, index_), labels(labels_) {}
};

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

        static void ExecuteReserveThreadImpl(AsyncJob *job) {
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
            : AsyncJob(std::move(allocator), jobType, ExecuteReserveThreadImpl, index),
              controlBlock(std::move(controlBlock)), jobsRegistry(registry) {}
    };

    using task_type = std::function<void(VecSimIndex *, size_t)>;
    task_type task;
    std::shared_ptr<ControlBlock> controlBlock;
    JobsRegistry *jobsRegistry;
    bool isScheduled; // true if this job holds a pending-job reservation on the thread pool

    static void ExecuteMultiThreadJobImpl(AsyncJob *job) {
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
                      std::shared_ptr<ControlBlock> controlBlock, JobsRegistry *registry,
                      bool scheduled = false)
        : AsyncJob(std::move(allocator), jobType, ExecuteMultiThreadJobImpl, index),
          task(std::move(callback)), controlBlock(std::move(controlBlock)), jobsRegistry(registry),
          isScheduled(scheduled) {}

    ~SVSMultiThreadJob() {
        if (isScheduled) {
            VecSimSVSThreadPoolImpl::instance()->endScheduledJob();
        }
    }

public:
    template <typename Rep, typename Period>
    static vecsim_stl::vector<AsyncJob *>
    createScheduledJobs(const std::shared_ptr<VecSimAllocator> &allocator, JobType jobType,
                        std::function<void(VecSimIndex *, size_t)> callback, VecSimIndex *index,
                        std::chrono::duration<Rep, Period> threads_wait_timeout,
                        JobsRegistry *registry) {
        size_t num_threads = VecSimSVSThreadPoolImpl::instance()->beginScheduledJob();
        return createJobs(allocator, jobType, callback, index, num_threads, threads_wait_timeout,
                          registry, /*scheduled=*/true);
    }

    template <typename Rep, typename Period>
    static vecsim_stl::vector<AsyncJob *>
    createJobs(const std::shared_ptr<VecSimAllocator> &allocator, JobType jobType,
               std::function<void(VecSimIndex *, size_t)> callback, VecSimIndex *index,
               size_t num_threads, std::chrono::duration<Rep, Period> threads_wait_timeout,
               JobsRegistry *registry, bool scheduled = false) {
        assert(num_threads > 0);
        std::shared_ptr<ControlBlock> controlBlock =
            num_threads == 1 ? nullptr
                             : std::make_shared<ControlBlock>(num_threads, threads_wait_timeout);

        vecsim_stl::vector<AsyncJob *> jobs(num_threads, allocator);
        jobs[0] = new (allocator) SVSMultiThreadJob(allocator, jobType, callback, index,
                                                    controlBlock, registry, scheduled);
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
    using DistType = float;
    using Self = TieredSVSIndex<DataType>;
    using Base = VecSimTieredIndex<DataType, DistType>;
    using flat_index_t = BruteForceIndex<DataType, DistType>;
    using backend_index_t = VecSimIndexAbstract<DataType, DistType>;
    using svs_index_t = SVSIndexBase;

    size_t trainingTriggerThreshold;
    size_t updateTriggerThreshold;
    size_t updateJobWaitTime;
    // Used to prevent scheduling multiple index update jobs at the same time.
    // As far as the update job does a batch update, job queue should have just 1 job at the moment.
    std::atomic_flag indexUpdateScheduled = ATOMIC_FLAG_INIT;
    // Used to prevent scheduling multiple index GC jobs at the same time.
    std::atomic_flag indexGCScheduled = ATOMIC_FLAG_INIT;
    // Used to prevent running multiple index update jobs in parallel.
    // Even if update jobs scheduled sequentially, they can be started in parallel.
    mutable std::shared_mutex updateJobMutex;

    // The reason of following container just to properly destroy jobs which not executed yet
    SVSMultiThreadJob::JobsRegistry uncompletedJobs;

    // frontend ids holding training data for the backend (re)initialization
    std::unordered_set<idType> ids_to_init_;

    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<SVSConsolidateJob *>>
        labelToConsolidateJobs;
    mutable std::shared_mutex consolidateJobsGuard;

    size_t flat_buffer_bound;

    /// <batch_iterator>
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //  TieredSVS_BatchIterator //
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    class TieredSVS_BatchIterator : public TieredIndex_BatchIterator {
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

        VecSimBatchIterator *flat_iterator;
        VecSimBatchIterator *svs_iterator;

        void acquire_svs_iterator() {
            assert(svs_iterator == nullptr);
            svs_iterator = index->backendIndex->newBatchIterator(
                this->flat_iterator->getQueryBlob(), queryParams);
        }

        void release_svs_iterator() {
            if (svs_iterator != nullptr && svs_iterator != depleted()) {
                delete svs_iterator;
                svs_iterator = nullptr;
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
        TieredSVS_BatchIterator(const void *query_vector, const Index *index,
                                VecSimQueryParams *queryParams,
                                std::shared_ptr<VecSimAllocator> allocator)
            // Tiered batch iterator doesn't hold its own copy of the query vector.
            // Instead, each internal batch iterators (flat_iterator and svs_iterator) create their
            // own copies: flat_iterator copy is created during TieredSVS_BatchIterator
            // construction When TieredSVS_BatchIterator::getNextResults() is called and
            // svs_iterator is not initialized, it retrieves the blob from flat_iterator
            : TieredIndex_BatchIterator(nullptr, queryParams ? queryParams->timeoutCtx : nullptr,
                                        std::move(allocator)),
              index(index),
              flat_iterator(index->frontendIndex->newBatchIterator(query_vector, queryParams)),
              svs_iterator(nullptr) {
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
                    std::shared_lock flat_lock{index->flatIndexGuard};
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
                backend_results.swap(cur_svs_results->results);
                VecSimQueryReply_Free(cur_svs_results);
                handle_svs_depletion();
            } else {
                while (flat_results.size() < n_res && !flat_iterator->isDepleted()) {
                    auto tail = flat_iterator->getNextResults(n_res - flat_results.size(),
                                                              BY_SCORE_THEN_ID);
                    flat_results.insert(flat_results.end(), tail->results.begin(),
                                        tail->results.end());
                    VecSimQueryReply_Free(tail);

                    // The flat results may contain labels already returned from the SVS index.
                    filter_irrelevant_results(this->flat_results);
                }

                while (backend_results.size() < n_res && svs_iterator != depleted() &&
                       svs_code == VecSim_OK) {
                    auto tail = svs_iterator->getNextResults(n_res - backend_results.size(),
                                                             BY_SCORE_THEN_ID);
                    svs_code =
                        tail->code; // Set the svs_results code to the last `getNextResults` code.
                    // New batch may contain better results than the previous batch, so we need to
                    // merge. We don't expect duplications (hence the <false>), as the iterator
                    // guarantees that no result is returned twice.
                    VecSimQueryResultContainer cur_svs_results(this->allocator);
                    merge_results<false>(cur_svs_results, backend_results, tail->results, n_res);
                    VecSimQueryReply_Free(tail);
                    backend_results.swap(cur_svs_results);
                    filter_irrelevant_results(backend_results);
                    handle_svs_depletion();
                }
            }

            if (VecSim_OK != svs_code) {
                return new VecSimQueryReply(this->allocator, svs_code);
            }

            VecSimQueryReply *batch;
            // In concurent execution we can observe a vector duplication in backend and frontend
            batch = compute_current_batch<true>(n_res);

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
        // multi-value indexes, this can happen to both iterators).
        // The next call to `getNextResults` will return an empty batch, and then the iterators will
        // correctly report that they are depleted.
        bool isDepleted() override {
            return flat_results.empty() && flat_iterator->isDepleted() && backend_results.empty() &&
                   svs_iterator == depleted();
        }

        void reset() override {
            release_svs_iterator();
            resetResultsCount();
            flat_iterator->reset();
            svs_iterator = nullptr;
            flat_results.clear();
            backend_results.clear();
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

    svs_index_t *GetSVSIndex() const {
        auto result = dynamic_cast<svs_index_t *>(this->backendIndex);
        assert(result);
        return result;
    }

#ifdef BUILD_TESTS
public:
    backend_index_t *GetBackendIndex() { return this->backendIndex; }
    void submitSingleJob(AsyncJob *job) { Base::submitSingleJob(job); }
    void submitJobs(vecsim_stl::vector<AsyncJob *> &jobs) { Base::submitJobs(jobs); }

    // Tracing helpers can be used to trace/inject code in the index update process.
    std::map<std::string, std::function<void()>> tracingCallbacks;
    void registerTracingCallback(const std::string &name, std::function<void()> callback) {
        tracingCallbacks[name] = std::move(callback);
    }
    void executeTracingCallback(const std::string &name) const {
        auto it = tracingCallbacks.find(name);
        if (it != tracingCallbacks.end()) {
            it->second();
        }
    }
    size_t indexMetaDataCapacity() const override {
        std::shared_lock<std::shared_mutex> flat_lock(this->flatIndexGuard);
        std::shared_lock<std::shared_mutex> main_lock(this->mainIndexGuard);
        return this->frontendIndex->indexMetaDataCapacity() +
               this->backendIndex->indexMetaDataCapacity();
    }
#else
    void executeTracingCallback(const std::string &) const {
        // In production, we do nothing.
    }
#endif

protected:
    ScopedLocks lockMainIndexForQuery() const override {
        // No-op: SVS does its query locking internally.
        return ScopedLocks();
    }

    ScopedLocks lockIndexForSize() const override { return ScopedLocks(this->flatIndexLockable); }

    ScopedLocks lockIndexForCapacity() const override {
        return ScopedLocks(this->flatIndexLockable);
    }

private:
    /**
     * @brief Init the SVS index in a thread-safe manner.
     *
     * This static wrapper function performs the following actions:
     * - Acquires a lock on the index's updateJobMutex to prevent concurrent updates.
     * - Configures the number of threads for the underlying SVS index update operation.
     * - Calls the initSVSIndex method to perform the actual index update.
     * - Clears the indexUpdateScheduled flag to allow future scheduling.
     *
     * @param idx Pointer to the VecSimIndex to be updated.
     * @param availableThreads The number of threads available for the update operation. Current
     * thread us used as well, so the minimal value is 1.
     */
    static void initSVSIndexWrapper(VecSimIndex *idx, size_t availableThreads) {
        assert(availableThreads > 0);
        auto index = static_cast<TieredSVSIndex<DataType> *>(idx);
        assert(index);
        // prevent parallel updates
        std::lock_guard<std::shared_mutex> lock(index->updateJobMutex);
        // flag stays set while init runs: !ready() + !flag means "no one will init the backend"
        struct ClearOnExit {
            std::atomic_flag &flag;
            ~ClearOnExit() { flag.clear(std::memory_order_release); }
        } clear_on_exit{index->indexUpdateScheduled};
        index->initSVSIndex(availableThreads);
    }

    static void executeInsertJobWrapper(AsyncJob *job) {
        auto *insert_job = static_cast<SVSInsertJob *>(job);
        auto *job_index = static_cast<TieredSVSIndex<DataType> *>(insert_job->index);
        InsertJobOutcome outcome;
        {
            // prevent parallel execution with index initilizing job
            std::shared_lock<std::shared_mutex> lock(job_index->updateJobMutex);
            outcome = job_index->executeInsertJob(insert_job);
        }

        if (outcome == InsertJobOutcome::Deferred) {
            job_index->submitSingleJob(job);
            return;
        }
        delete job;
    }

    /**
     * @brief Run SVS index GC in a thread-safe manner.
     *
     * This static wrapper function performs the following actions:
     * - Acquires a lock on the index's mainIndexGuard to ensure thread safety during the GC
     * - Configures the number of threads for the underlying SVS index update operation.
     * - Calls the SVSIndex::runGC() method to perform the actual index update.
     * - Clears the indexGCScheduled flag to allow future scheduling.
     *
     * @param idx Pointer to the VecSimIndex to be updated.
     * @param availableThreads The number of threads available for the update operation. Current
     * thread us used as well, so the minimal value is 1.
     * @note no need to implement extra non-static method, as GC logic is simple enough to be done
     * here.
     */
    static void SVSIndexGCWrapper(VecSimIndex *idx, size_t availableThreads) {
        assert(availableThreads > 0);
        auto index = static_cast<TieredSVSIndex<DataType> *>(idx);
        assert(index);

        // Do SVS index GC
        index->backendIndex->log(VecSimCommonStrings::LOG_VERBOSE_STRING,
                                 "running asynchronous GC for tiered SVS index");
        auto svs_index = index->GetSVSIndex();
        if (index->backendIndex->indexSize() == 0) {
            // No need to run GC on an empty index.
            index->indexGCScheduled.clear();
            return;
        }
        index->executeTracingCallback("GCJob::before_run_gc");
        std::lock_guard<std::shared_mutex> lock(index->updateJobMutex);

        // Release the scheduled flag to allow scheduling again
        index->indexGCScheduled.clear();

        svs_index->setParallelism(std::min(availableThreads, index->backendIndex->indexSize()));
        // VecSimIndexAbstract::runGC() is protected
        static_cast<VecSimIndexInterface *>(index->backendIndex)->runGC();
        svs_index->setParallelism(1);
    }

    static void SVSIndexConsolidateWrapper(AsyncJob *job) {
        auto consolidate_job = static_cast<SVSConsolidateJob *>(job);
        auto index = static_cast<TieredSVSIndex<DataType> *>(consolidate_job->index);

        std::shared_lock<std::shared_mutex> lock(index->updateJobMutex);
        auto svs_index = index->GetSVSIndex();
        svs_index->setParallelism(1);

        bool valid = false;
        {
            std::shared_lock<std::shared_mutex> flat_lock(index->flatIndexGuard);
            valid = consolidate_job->isValid;
            if (valid) {
                consolidate_job->executing.store(true, std::memory_order_release);
            }
        }
        if (valid) {
            svs_index->consolidate(consolidate_job->labels);
            // Cleared before the registry lock is needed again, so a waiter cannot deadlock us.
            consolidate_job->executing.store(false, std::memory_order_release);
        }
        index->forgetConsolidateJob(consolidate_job);
        delete job;
    }

#ifdef BUILD_TESTS
public:
#endif

    void scheduleSVSIndexInit() {
        // do not schedule if scheduled already
        if (indexUpdateScheduled.test_and_set()) {
            return;
        }

        auto jobs = SVSMultiThreadJob::createScheduledJobs(
            this->allocator, SVS_BATCH_UPDATE_JOB, initSVSIndexWrapper, this,
            std::chrono::microseconds(updateJobWaitTime), &uncompletedJobs);
        this->submitJobs(jobs);
    }

    void scheduleSVSIndexGC() {
        // do not schedule if scheduled already
        if (indexGCScheduled.test_and_set()) {
            return;
        }

        auto jobs = SVSMultiThreadJob::createScheduledJobs(
            this->allocator, SVS_GC_JOB, SVSIndexGCWrapper, this,
            std::chrono::microseconds(updateJobWaitTime), &uncompletedJobs);
        this->submitJobs(jobs);
    }

    void scheduleSVSIndexConsolidate(labelType label) {
        auto *new_consolidate_job = new (this->allocator)
            SVSConsolidateJob(this->allocator, {label}, SVSIndexConsolidateWrapper, this);

        {
            std::lock_guard<std::shared_mutex> lock(this->consolidateJobsGuard);
            auto it = this->labelToConsolidateJobs.find(label);
            if (it != this->labelToConsolidateJobs.end()) {
                it->second.push_back(new_consolidate_job);
            } else {
                vecsim_stl::vector<SVSConsolidateJob *> jobs(1, new_consolidate_job,
                                                             this->allocator);
                this->labelToConsolidateJobs.insert({label, std::move(jobs)});
            }
        }
        // Insert job to the queue.
        this->submitSingleJob(new_consolidate_job);
    }

private:
    void forgetConsolidateJob(SVSConsolidateJob *job) {
        std::lock_guard<std::shared_mutex> lock(this->consolidateJobsGuard);
        for (auto label : job->labels) {
            auto it = this->labelToConsolidateJobs.find(label);
            if (it == this->labelToConsolidateJobs.end()) {
                continue;
            }
            auto &jobs = it->second;
            jobs.erase(std::remove(jobs.begin(), jobs.end(), job), jobs.end());
            if (jobs.empty()) {
                this->labelToConsolidateJobs.erase(it);
            }
        }
    }
    // Caller must hold flatIndexGuard exclusive
    std::vector<labelType> takeOverConsolidateOf(labelType label) {
        std::vector<labelType> taken_over;
        vecsim_stl::vector<SVSConsolidateJob *> running(this->allocator);
        {
            std::shared_lock<std::shared_mutex> lock(this->consolidateJobsGuard);
            auto it = this->labelToConsolidateJobs.find(label);
            if (it == this->labelToConsolidateJobs.end()) {
                return taken_over;
            }
            for (auto *job : it->second) {
                if (!job->isValid) {
                    continue; // already taken over by an earlier call
                }
                if (job->executing.load(std::memory_order_acquire)) {
                    running.push_back(job);
                } else {
                    job->isValid = false;
                    taken_over.insert(taken_over.end(), job->labels.begin(), job->labels.end());
                }
            }
        }
        for (auto *job : running) {
            waitForJob(job);
        }
        return taken_over;
    }

    // Wait until the job leaves its publish window
    // Safe to call while holding flatIndexGuard exclusive
    static void waitForJob(SVSInsertJob *job) {
        while (job->status.load(std::memory_order_acquire) == SVSInsertJob::Status::Executing) {
            std::this_thread::yield();
        }
    }

    static void waitForJob(SVSConsolidateJob *job) {
        while (job->executing.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    idType setAndSaveInvalidJob(AsyncJob *job) override {
        waitForJob(static_cast<SVSInsertJob *>(job));
        return Base::setAndSaveInvalidJob(job);
    }

    enum class InsertJobOutcome { Completed, Deferred };

    InsertJobOutcome executeInsertJob(SVSInsertJob *job) {
        auto svs_index = GetSVSIndex();

        // Note that accessing the job fields should occur with flat index guard held (here and
        // later).
        this->flatIndexGuard.lock_shared();
        if (!job->isValid) {
            this->flatIndexGuard.unlock_shared();
            // Job has been invalidated in the meantime - nothing to execute, and remove it from the
            // lookup.
            this->invalidJobsLookupGuard.lock();
            this->invalidJobs.erase(job->id);
            this->invalidJobsLookupGuard.unlock();
            return InsertJobOutcome::Completed;
        }

        if (job->status.load(std::memory_order_relaxed) == SVSInsertJob::Status::Delayed) {
            this->flatIndexGuard.unlock_shared();
            // relabelVector() clears it once the backend remap settled the label
            return InsertJobOutcome::Deferred;
        }

        // a job never inits the backend: one point cannot train the compression
        if (!svs_index->ready()) {
            this->flatIndexGuard.unlock_shared();
            // init pending -> wait for it. no init pending -> back to the training buffer
            return this->indexUpdateScheduled.test(std::memory_order_acquire)
                       ? InsertJobOutcome::Deferred
                       : adoptJobIntoInitBuffer(job);
        }

        // Copy the vector blob out of the flat buffer while holding flatIndexGuard, so we
        // can release the flat lock before indexing into the SVS backend
        size_t data_size = this->frontendIndex->getStoredDataSize();
        auto blob_copy = this->getAllocator()->allocate_unique(data_size);
        memcpy(blob_copy.get(), this->frontendIndex->getDataByInternalId(job->id), data_size);

        job->status.store(SVSInsertJob::Status::Executing, std::memory_order_release);
        this->flatIndexGuard.unlock_shared();

        // if a concurrent deletion drops the instance
        // initializing it from this single point would refit the compression
        const int added = svs_index->addVectorsIfInitialized(blob_copy.get(), &job->label, 1);
        const bool published = added != SVSIndexBase::kNotInitialized;

        job->status.store(published ? SVSInsertJob::Status::Done : SVSInsertJob::Status::Pending,
                          std::memory_order_release);
        if (!published) {
            return adoptJobIntoInitBuffer(job);
        }
        // Remove the vector and the insert job from the flat buffer.
        this->removeIngestedVectorFromFlat(job);
        return InsertJobOutcome::Completed;
    }

    void initSVSIndex(size_t availableThreads) {
        std::vector<idType> ids_to_move;
        std::vector<labelType> labels_to_move;
        std::vector<DataType> vectors_to_move;

        executeTracingCallback("UpdateJob::before_add_to_svs");
        { // lock frontendIndex from modifications
            // The whole initialization is done under flatIndexGuard
            std::lock_guard flat_lock{this->flatIndexGuard};

            auto flat_index = this->GetFlatIndex();
            const auto init_batch_size = ids_to_init_.size();
            const size_t dim = flat_index->getDim();
            ids_to_move.reserve(init_batch_size);
            labels_to_move.reserve(init_batch_size);
            vectors_to_move.reserve(init_batch_size * dim);

            for (idType id : ids_to_init_) {
                ids_to_move.push_back(id);
                labels_to_move.push_back(flat_index->getVectorLabel(id));
                auto data = flat_index->getDataByInternalId(id);
                vectors_to_move.insert(vectors_to_move.end(), data, data + dim);
            }
            ids_to_init_.clear();

            // Nothing to initialize from an empty batch, and setParallelism(0) is not a
            // valid request against the shared pool.
            if (!labels_to_move.empty()) {
                auto svs_index = GetSVSIndex();
                svs_index->setParallelism(std::min(availableThreads, labels_to_move.size()));
                assert(labels_to_move.size() ==
                       vectors_to_move.size() / this->frontendIndex->getDim());
                // The backend may already have been initialized while this job was queued
                if (svs_index->ready()) {
                    svs_index->addVectors(vectors_to_move.data(), labels_to_move.data(),
                                          labels_to_move.size());
                    svs_index->setParallelism(1);
                } else {
                    auto impl = svs_index->createImpl(vectors_to_move.data(), labels_to_move.data(),
                                                      labels_to_move.size());
                    svs_index->setParallelism(1);
                    svs_index->setImpl(std::move(impl));
                }
            }

            std::sort(ids_to_move.begin(), ids_to_move.end());
            [[maybe_unused]] size_t total_deleted = 0;
            for (auto it = ids_to_move.rbegin(); it != ids_to_move.rend(); ++it) {
                idType id = *it;
                auto label = this->frontendIndex->getVectorLabel(id);
                // Delete the vector from the frontend index if not in-place updated.
                labelType last_vec_label =
                    this->frontendIndex->getVectorLabel(this->frontendIndex->indexSize() - 1);
                int deleted = this->frontendIndex->deleteVectorById(label, id);
                if (deleted && id != this->frontendIndex->indexSize()) {
                    // If the vector removal caused a swap with the last id, update the relevant
                    // insert job.
                    this->updateInsertJobInternalId(this->frontendIndex->indexSize(), id,
                                                    last_vec_label);
                }
                total_deleted += deleted;
            }

            assert(total_deleted == labels_to_move.size() &&
                   "Deleted vectors count does not match the number of labels to delete");
        } // release frontend index
        executeTracingCallback("UpdateJob::after_add_to_svs");
    }

public:
    TieredSVSIndex(VecSimIndexAbstract<DataType, float> *svs_index, flat_index_t *bf_index,
                   const TieredIndexParams &tiered_index_params,
                   std::shared_ptr<VecSimAllocator> allocator)
        : Base(svs_index, bf_index, tiered_index_params, allocator),
          uncompletedJobs(this->allocator), labelToConsolidateJobs(this->allocator) {
        const auto &tiered_svs_params = tiered_index_params.specificParams.tieredSVSParams;

        // If flatBufferLimit is not initialized (0), use the default update threshold.
        flat_buffer_bound = tiered_index_params.flatBufferLimit == 0
                                ? SVS_VAMANA_DEFAULT_UPDATE_THRESHOLD
                                : tiered_index_params.flatBufferLimit;

        this->updateTriggerThreshold =
            tiered_svs_params.updateTriggerThreshold == 0
                ? SVS_VAMANA_DEFAULT_UPDATE_THRESHOLD
                : std::min({tiered_svs_params.updateTriggerThreshold, flat_buffer_bound,
                            static_cast<size_t>(SVS_VAMANA_DEFAULT_UPDATE_THRESHOLD)});

        const size_t default_training_threshold = this->GetSVSIndex()->isCompressed()
                                                      ? SVS_VAMANA_DEFAULT_TRAINING_THRESHOLD
                                                      : this->updateTriggerThreshold;

        this->trainingTriggerThreshold =
            tiered_svs_params.trainingTriggerThreshold == 0
                ? default_training_threshold
                : std::min(tiered_svs_params.trainingTriggerThreshold, SVS_MAX_TRAINING_THRESHOLD);

        this->updateJobWaitTime = tiered_svs_params.updateJobWaitTime == 0
                                      ? SVS_DEFAULT_UPDATE_JOB_WAIT_TIME
                                      : tiered_svs_params.updateJobWaitTime;
    }

    ~TieredSVSIndex() {
        // Delete all the pending consolidate jobs
        for (auto &jobs : this->labelToConsolidateJobs) {
            for (auto *job : jobs.second) {
                delete job;
            }
        }
    }

    int addVector(const void *blob, labelType label) override {
        int ret = 0;
        auto svs_index = GetSVSIndex();

        // In-Place mode - add vector syncronously to the backend index.
        if (this->getWriteMode() == VecSim_WriteInPlace) {
            // Backend index initialization data have to be buffered for proper
            // compression/training.
            if (!svs_index->ready()) {
                if (auto buffered = tryBufferForTraining(blob, label, /*in_place=*/true)) {
                    return *buffered;
                }
            }
            // backend index is initialized - we can add the vector directly
            auto storage_blob = this->frontendIndex->preprocessForStorage(blob);
            int deleted = 0;
            {
                // prevent update job from running in parallel and lock any access to the backend
                // index
                // Only updateJobMutex is needed here, not mainIndexGuard: the concurrent
                // backend index serializes writes against concurrent readers itself.
                std::lock_guard<std::shared_mutex> lock(this->updateJobMutex);
                // Defensive: ensure single-threaded operation for write-in-place mode.
                // parallelism_ defaults to 1, so this is a no-op in the normal case.
                svs_index->setParallelism(1);
                if (!this->backendIndex->isMultiValue()) {
                    deleted = svs_index->deleteVector(label);
                    if (deleted > 0)
                        svs_index->consolidate({label});
                }
                if (svs_index->ready()) {
                    return this->backendIndex->addVector(storage_blob.get(), label) - deleted;
                }
            }
            // the delete above dropped the last backend vector: adding now would refit the
            // compression to this single point
            if (auto buffered = tryBufferForTraining(blob, label, /*in_place=*/true)) {
                return std::max(*buffered - deleted, 0);
            }
            // a queued init job rebuilt the backend meanwhile
            std::lock_guard<std::shared_mutex> lock(this->updateJobMutex);
            return std::max(this->backendIndex->addVector(storage_blob.get(), label) - deleted, 0);
        }
        assert(this->getWriteMode() != VecSim_WriteInPlace && "InPlace mode returns early");

        // Async mode - buffer training data until the backend can be inited from a batch.
        if (!svs_index->ready() && !this->indexUpdateScheduled.test(std::memory_order_acquire)) {
            if (auto buffered = tryBufferForTraining(blob, label, /*in_place=*/false)) {
                return *buffered;
            }
        }

        this->flatIndexGuard.lock_shared();
        if (svs_index->ready() && (this->frontendIndex->indexSize() >= flat_buffer_bound)) {
            this->flatIndexGuard.unlock_shared();
            auto storage_blob = this->frontendIndex->preprocessForStorage(blob);

            const int overwritten =
                this->backendIndex->isMultiValue() ? 0 : this->deleteVector(label);

            // the delete above may have dropped the last backend vector
            if (!svs_index->ready()) {
                if (auto buffered = tryBufferForTraining(blob, label, /*in_place=*/false)) {
                    return std::max(*buffered - overwritten, 0);
                }
            }

            std::shared_lock<std::shared_mutex> lock(updateJobMutex);
            ret = svs_index->addVector(storage_blob.get(), label);
            return std::max(ret - overwritten, 0);
        } else {
            this->flatIndexGuard.unlock_shared();
            this->flatIndexGuard.lock();
            idType new_flat_id = this->frontendIndex->indexSize();
            if (this->frontendIndex->isLabelExists(label) && !this->frontendIndex->isMultiValue()) {
                if (this->labelToInsertJobs.count(label) == 0) {
                    // No pending insert job.
                    // Just replace vector in the frontend
                    // If this label already exists, this will do overwrite.
                    ret = this->frontendIndex->addVector(blob, label);
                    this->flatIndexGuard.unlock();
                    return ret;
                }

                // Overwrite the vector and invalidate its only pending job (since we are not in
                // MULTI). Label exists, but job doesn't. It means the label is used for
                // initialization. Just create new job, this job will remove duplicate from the
                // backend.
                auto *old_job = this->labelToInsertJobs.at(label).at(0);
                old_job->id = this->setAndSaveInvalidJob(old_job);
                this->labelToInsertJobs.erase(label);
                ret = 0;
                // We are going to update the internal id that currently holds the vector associated
                // with the given label.
                new_flat_id =
                    dynamic_cast<BruteForceIndex_Single<DataType, DistType> *>(this->frontendIndex)
                        ->getIdOfLabel(label);
                // If we are adding a new element (rather than updating an exiting one) we may need
                // to increase index capacity.
            }
            // If this label already exists, this will do overwrite.
            ret += this->frontendIndex->addVector(blob, label);

            TieredInsertJob *new_insert_job = createInsertJob(label, new_flat_id);
            this->flatIndexGuard.unlock();

            // Here, a worker might ingest the previous vector that was stored under "label"
            // (in case of override in non-MULTI index) - so if it's there, we remove it
            //  we submit the insert job.
            if (!this->backendIndex->isMultiValue()) {
                if (svs_index->ready()) {
                    // If we removed the previous vector from both svs and flat in the overwrite
                    // process, we still return 0 (not -1).
                    auto deleted = svs_index->deleteVector(label);
                    if (deleted > 0)
                        scheduleSVSIndexConsolidate(label);
                    ret = std::max(ret - deleted, 0);
                }
            }

            // Insert job to the queue and signal the workers' updater.
            this->submitSingleJob(new_insert_job);
            return ret;
        }
    }

    // training data is tracked by frontend id, so it has to follow the same swaps as insert jobs
    void updateInsertJobInternalId(idType prev_id, idType new_id, labelType label) override {
        if (ids_to_init_.erase(prev_id) > 0) {
            ids_to_init_.insert(new_id);
        }
        Base::updateInsertJobInternalId(prev_id, new_id, label);
    }

    // Returns the number of vectors removed from the frontend index.
    int deleteAndUpdateInitIds(labelType label) {
        auto deleting_ids = this->frontendIndex->getElementIds(label);

        std::sort(deleting_ids.begin(), deleting_ids.end());

        // Delete vector from the frontend index.
        auto updated_ids = this->frontendIndex->deleteVectorAndGetUpdatedIds(label);

        assert(std::all_of(updated_ids.begin(), updated_ids.end(),
                           [&deleting_ids](const auto &pair) {
                               return std::binary_search(deleting_ids.begin(), deleting_ids.end(),
                                                         pair.first);
                           }) &&
               "updated_ids should be a subset of deleting_ids");

        std::vector<idType> new_ids_to_init;
        for (auto &it : updated_ids) {
            idType prev_id = it.second.first;
            idType new_id = it.first;

            if (ids_to_init_.count(prev_id) > 0) {
                ids_to_init_.erase(prev_id);
                new_ids_to_init.push_back(new_id);
            }
            labelType updated_vec_label = it.second.second;
            this->updateInsertJobInternalId(prev_id, new_id, updated_vec_label);
        }

        for (idType new_id : new_ids_to_init) {
            ids_to_init_.insert(new_id);
        }

        return static_cast<int>(deleting_ids.size());
    }

    int removeLabelFromFlat(labelType label) {
        if (!this->frontendIndex->isLabelExists(label)) {
            return 0;
        }
        auto deleting_ids = this->frontendIndex->getElementIds(label);
        if (deleting_ids.size() == 0) {
            return 0;
        }

        // assert if all elements of deleting_ids are unique
        assert(std::set(deleting_ids.begin(), deleting_ids.end()).size() == deleting_ids.size() &&
               "deleting_ids should contain unique ids");

        // If id is deleted, don't use it for initialization
        if (!ids_to_init_.empty()) {
            for (idType id : deleting_ids) {
                ids_to_init_.erase(id);
            }
        }

        if (this->labelToInsertJobs.count(label) > 0) {
            // Invalidate the pending insert job(s) into SVS associated with this label
            auto &insert_jobs = this->labelToInsertJobs.at(label);
            for (auto *job : insert_jobs) {
                job->id = this->setAndSaveInvalidJob(job);
            }
            // Remove the pending insert job(s) from the labelToInsertJobs mapping.
            this->labelToInsertJobs.erase(label);
        }

        deleteAndUpdateInitIds(label);
        return static_cast<int>(deleting_ids.size());
    }

    // Create a pending insert job for a vector already buffered in the frontend index
    // Caller must hold flatIndexGuard exclusively.
    TieredInsertJob *createInsertJob(labelType label, idType flat_id) {
        TieredInsertJob *job = new (this->allocator)
            SVSInsertJob(this->allocator, label, flat_id, executeInsertJobWrapper, this);
        auto it = this->labelToInsertJobs.find(label);
        if (it != this->labelToInsertJobs.end()) {
            // There's already a pending insert job for this label, add another one (without
            // overwrite, only possible in multi index)
            assert(this->backendIndex->isMultiValue());
            it->second.push_back(job);
        } else {
            vecsim_stl::vector<TieredInsertJob *> new_jobs_vec(1, job, this->allocator);
            this->labelToInsertJobs.insert({label, new_jobs_vec});
        }
        return job;
    }

    // Buffer a vector in the frontend index as training data for the backend (re)initialization.
    // Returns 1 for a new label, 0 for an overwrite, or nullopt if the backend turned ready while
    // the guard was taken
    std::optional<int> tryBufferForTraining(const void *blob, labelType label, bool in_place) {
        int ret = 0;
        bool run_init = false;
        {
            std::lock_guard flat_lock{this->flatIndexGuard};
            // never buffer into a ready backend: ids_to_init_ would have no consumer
            if (GetSVSIndex()->ready()) {
                return std::nullopt;
            }
            int deleted = 0;
            if (!this->frontendIndex->isMultiValue() && this->frontendIndex->isLabelExists(label)) {
                // once the backend has been dropped the buffer may also hold job-backed vectors,
                // so the overwrite has to invalidate pending jobs
                deleted = removeLabelFromFlat(label);
            }
            ids_to_init_.insert(this->frontendIndex->indexSize());
            ret = std::max(this->frontendIndex->addVector(blob, label) - deleted, 0);
            run_init = ids_to_init_.size() >= this->trainingTriggerThreshold;
            if (run_init && !in_place) {
                scheduleSVSIndexInit();
                run_init = false;
            }
        }
        if (run_init) {
            initSVSIndexWrapper(this, 1);
        }
        return ret;
    }

    // Move a pending job's vector from the insert-job path to the training batch. The vector stays
    // in the frontend index, only its ownership changes.
    InsertJobOutcome adoptJobIntoInitBuffer(SVSInsertJob *job) {
        std::lock_guard flat_lock{this->flatIndexGuard};
        if (!job->isValid) {
            std::lock_guard invalid_lock{this->invalidJobsLookupGuard};
            this->invalidJobs.erase(job->id);
            return InsertJobOutcome::Completed;
        }
        if (GetSVSIndex()->ready() || this->indexUpdateScheduled.test(std::memory_order_acquire)) {
            return InsertJobOutcome::Deferred;
        }
        ids_to_init_.insert(job->id);
        this->detachInsertJob(job);
        if (ids_to_init_.size() >= this->trainingTriggerThreshold) {
            scheduleSVSIndexInit();
        }
        return InsertJobOutcome::Completed;
    }

    int deleteVector(labelType label) override {
        int ret = 0;

        this->flatIndexGuard.lock_shared();
        if (this->frontendIndex->isLabelExists(label)) {
            this->flatIndexGuard.unlock_shared();
            std::lock_guard flat_lock{this->flatIndexGuard};
            ret += removeLabelFromFlat(label);
        } else {
            this->flatIndexGuard.unlock_shared();
        }

        auto deleted = this->backendIndex->deleteVector(label);
        if (deleted > 0) {
            if (this->getWriteMode() == VecSim_WriteInPlace) {
                GetSVSIndex()->consolidate({label});
            } else {
                scheduleSVSIndexConsolidate(label);
            }
        }
        ret += deleted;
        return ret;
    }

#if HAVE_SVS_REPLACE_EXTERNAL_ID
    // Only declared when the SVS this was built against offers `replace_external_id`, mirroring
    // `SVSIndex::relabelVector`. Left out otherwise, so the interface default reports
    // `VecSimRelabel_Unsupported` for the whole tier rather than this moving a buffered label
    // and refusing an ingested one -- a caller cannot act on a capability that depends on which
    // tier happens to hold the label. It also keeps the runtime probe honest: an override that
    // answered `SameLabel` before consulting the backend would look capable on a build that
    // is not.
    /**
     * Move `old_label` onto `new_label`, leaving the vector where it is in whichever tier holds
     * it. `new_label` must be unused in both tiers, not just the one holding `old_label`: a
     * multi-value label routinely has copies in each, and a target taken in either would collide
     * once the buffer drains.
     *
     * Reports `Unsupported` when the backend holds the label and cannot move it, which is the
     * case when built against an SVS without `replace_external_id`. All-or-nothing: on any code
     * other than `VecSimRelabel_OK` both tiers are untouched.
     */
    VecSimRelabelCode relabelVector(labelType old_label, labelType new_label) override {
        if (old_label == new_label) {
            return VecSimRelabel_SameLabel;
        }
        auto *svs_index = GetSVSIndex();

        std::shared_lock<std::shared_mutex> lock(this->updateJobMutex);

        bool flat_holds_old = false;
        bool delayed_any = false;
        std::vector<labelType> taken_over;
        {
            std::lock_guard flat_lock{this->flatIndexGuard};

            flat_holds_old = this->frontendIndex->isLabelExists(old_label);
            if (!flat_holds_old && !svs_index->isLabelExists(old_label)) {
                return VecSimRelabel_OldLabelMissing;
            }
            if (this->frontendIndex->isLabelExists(new_label) ||
                svs_index->isLabelExists(new_label)) {
                return VecSimRelabel_NewLabelTaken;
            }

            auto pending = this->labelToInsertJobs.find(old_label);
            if (pending != this->labelToInsertJobs.end()) {
                auto jobs = std::move(pending->second);
                this->labelToInsertJobs.erase(pending);
                for (auto *job : jobs) {
                    auto *insert_job = static_cast<SVSInsertJob *>(job);
                    // wait it out: a job that published before the remap is remapped with the
                    // backend, and still drops its own buffer copy under the new label
                    waitForJob(insert_job);
                    job->label = new_label;
                    if (insert_job->status.load(std::memory_order_relaxed) ==
                        SVSInsertJob::Status::Pending) {
                        // did not publish yet - hold it until the remap settles the label
                        insert_job->status.store(SVSInsertJob::Status::Delayed,
                                                 std::memory_order_relaxed);
                        delayed_any = true;
                    }
                }
                this->labelToInsertJobs.emplace(new_label, std::move(jobs));
            }

            // isLabelExists() said new_label is free, but SVS keeps the translator entry of a
            // soft-deleted label until its consolidate job runs, and would refuse the remap.
            taken_over = takeOverConsolidateOf(new_label);

            if (flat_holds_old) {
                const VecSimRelabelCode flat_ret =
                    this->frontendIndex->relabelVector(old_label, new_label);
                assert(flat_ret == VecSimRelabel_OK &&
                       "the buffer just reported holding this label");
                UNUSED(flat_ret);
            }
        }

        // Do the taken-over job's,
        // the remap below isn't refused by a leftover translator entry
        if (!taken_over.empty()) {
            svs_index->consolidate(taken_over);
        }

        const bool rollback =
            this->backendIndex->relabelVector(old_label, new_label) == VecSimRelabel_NewLabelTaken;
        const labelType final_label = rollback ? old_label : new_label;

        if (rollback || delayed_any) {
            std::lock_guard flat_lock{this->flatIndexGuard};
            if (rollback) {
                auto pending = this->labelToInsertJobs.find(new_label);
                if (pending != this->labelToInsertJobs.end()) {
                    auto jobs = std::move(pending->second);
                    this->labelToInsertJobs.erase(pending);
                    for (auto *job : jobs) {
                        job->label = old_label;
                    }
                    this->labelToInsertJobs.emplace(old_label, std::move(jobs));
                }
                if (flat_holds_old) {
                    this->frontendIndex->relabelVector(new_label, old_label);
                }
            }
            if (delayed_any) {
                auto it = this->labelToInsertJobs.find(final_label);
                if (it != this->labelToInsertJobs.end()) {
                    for (auto *job : it->second) {
                        auto *insert_job = static_cast<SVSInsertJob *>(job);
                        if (insert_job->status.load(std::memory_order_relaxed) ==
                            SVSInsertJob::Status::Delayed) {
                            insert_job->status.store(SVSInsertJob::Status::Pending,
                                                     std::memory_order_relaxed);
                        }
                    }
                }
            }
        }
        return rollback ? VecSimRelabel_NewLabelTaken : VecSimRelabel_OK;
    }

#endif // HAVE_SVS_REPLACE_EXTERNAL_ID

    size_t getNumMarkedDeleted() const override {
        return this->GetSVSIndex()->getNumMarkedDeleted();
    }

    VecSimIndexDebugInfo debugInfo() const override {
        auto info = Base::debugInfo();

        SvsTieredInfo svsTieredInfo = {
            .trainingTriggerThreshold = this->trainingTriggerThreshold,
            .updateTriggerThreshold = this->updateTriggerThreshold,
            .updateJobWaitTime = this->updateJobWaitTime,
        };

        svsTieredInfo.indexUpdateScheduled =
            (info.tieredInfo.frontendCommonInfo.indexSize > 0) &&
            this->indexUpdateScheduled.test(std::memory_order_acquire);
        info.tieredInfo.specificTieredBackendInfo.svsTieredInfo = svsTieredInfo;

        // Background indexing is in progress whenever the flat buffer is non-empty
        info.tieredInfo.backgroundIndexing =
            info.tieredInfo.frontendCommonInfo.indexSize > 0 ? VecSimBool_TRUE : VecSimBool_FALSE;
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
        VecSimIndexDebugInfo info = this->debugInfo();

        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::TIERED_SVS_TRAINING_THRESHOLD_STRING,
            .fieldType = INFOFIELD_UINT64,
            .fieldValue = {
                FieldValue{.uintegerValue = info.tieredInfo.specificTieredBackendInfo.svsTieredInfo
                                                .trainingTriggerThreshold}}});

        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::TIERED_SVS_UPDATE_THRESHOLD_STRING,
            .fieldType = INFOFIELD_UINT64,
            .fieldValue = {
                FieldValue{.uintegerValue = info.tieredInfo.specificTieredBackendInfo.svsTieredInfo
                                                .updateTriggerThreshold}}});

        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::TIERED_SVS_THREADS_RESERVE_TIMEOUT_STRING,
            .fieldType = INFOFIELD_UINT64,
            .fieldValue = {FieldValue{
                .uintegerValue =
                    info.tieredInfo.specificTieredBackendInfo.svsTieredInfo.updateJobWaitTime}}});
        return infoIterator;
    }

    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        // The query blob will be processed and copied by the internal indexes's batch iterator.
        return new (this->allocator)
            TieredSVS_BatchIterator(queryBlob, this, queryParams, this->allocator);
    }

    void setLastSearchMode(VecSearchMode mode) override {
        return this->backendIndex->setLastSearchMode(mode);
    }

    void runGC() override {
        if (this->getWriteMode() == VecSim_WriteInPlace) {
            TIERED_LOG(VecSimCommonStrings::LOG_VERBOSE_STRING,
                       "running synchronous GC for tiered SVS index in write-in-place mode");
            // In write-in-place mode, we run GC synchronously.
            if (this->backendIndex->indexSize() == 0) {
                // No need to run GC on an empty index.
                return;
            }
            // Force single thread for write-in-place mode.
            this->GetSVSIndex()->setParallelism(1);
            // VecSimIndexAbstract::runGC() is protected
            static_cast<VecSimIndexInterface *>(this->backendIndex)->runGC();
            return;
        }
        TIERED_LOG(VecSimCommonStrings::LOG_VERBOSE_STRING,
                   "scheduling asynchronous GC for tiered SVS index");
        scheduleSVSIndexGC();
    }

    void acquireSharedLocks() override { this->flatIndexGuard.lock_shared(); }

    void releaseSharedLocks() override { this->flatIndexGuard.unlock_shared(); }
};

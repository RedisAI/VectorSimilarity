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
#include <tuple>

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
                      std::shared_ptr<ControlBlock> controlBlock, JobsRegistry *registry)
        : AsyncJob(std::move(allocator), jobType, ExecuteMultiThreadJobImpl, index),
          task(std::move(callback)), controlBlock(std::move(controlBlock)), jobsRegistry(registry) {
    }

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
    using flat_index_t = BruteForceIndex<DataType, float>;
    using backend_index_t = VecSimIndexAbstract<DataType, float>;
    using svs_index_t = SVSIndexBase;

    // swaps_journal is used by updateSVSIndex() to track vectors swap operations that were done in
    // the Flat index during SVS index updating.
    // The journal contains tuples of (label, oldId, newId).
    // oldId is the index of the label in flat index before the swap.
    // newId is the index of the label in flat index after the swap.
    // if oldId == newId, it means that the vector was not moved in the Flat index, but was removed
    // from the end of the Flat index (no id swaps occurred internally).
    // if label == SKIP_LABEL, it means that the vector was not moved in the Flat index, but
    // updated in-place (hence was already removed  and no need to remove it again)
    using swap_record = std::tuple<labelType, idType, idType>;
    constexpr static size_t SKIP_LABEL = std::numeric_limits<labelType>::max();
    std::vector<swap_record> swaps_journal;

    size_t trainingTriggerThreshold;
    size_t updateTriggerThreshold;
    size_t updateJobWaitTime;
    // Used to prevent scheduling multiple index update jobs at the same time.
    // As far as the update job does a batch update, job queue should have just 1 job at the moment.
    std::atomic_flag indexUpdateScheduled = ATOMIC_FLAG_INIT;
    // Used to prevent running multiple index update jobs in parallel.
    // Even if update jobs scheduled sequentially, they can be started in parallel.
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

        VecSimQueryReply *compute_current_batch(size_t n_res, bool isMultiValue) {
            // Merge results
            // This call will update `svs_res` and `bf_res` to point to the end of the merged
            // results.
            auto batch_res = new VecSimQueryReply(allocator);
            // VecSim and SVS distance computation is implemented differently, so we always have to
            // merge results with set.
            auto [from_svs, from_flat] =
                merge_results<true>(batch_res->results, svs_results, flat_results, n_res);

            if (!isMultiValue) {
                // If we're on a single-value index, update the set of results returned from the
                // FLAT index before popping them, to prevent them to be returned from the SVS index
                // in later batches.
                for (size_t i = 0; i < from_flat; ++i) {
                    this->returned_results_set.insert(this->flat_results[i].id);
                }
            } else {
                // If we're on a multi-value index, update the set of results returned (from
                // `batch_res`)
                for (size_t i = 0; i < batch_res->results.size(); ++i) {
                    this->returned_results_set.insert(batch_res->results[i].id);
                }
            }

            // Update results
            flat_results.erase(flat_results.begin(), flat_results.begin() + from_flat);
            svs_results.erase(svs_results.begin(), svs_results.begin() + from_svs);

            // clean up the results
            // On multi-value indexes, one (or both) results lists may contain results that are
            // already returned form the other list (with a different score). We need to filter them
            // out.
            if (isMultiValue) {
                filter_irrelevant_results(this->flat_results);
                filter_irrelevant_results(this->svs_results);
            }

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
            assert(svs_iterator == nullptr);
            this->index->mainIndexGuard.lock_shared();
            svs_iterator = index->backendIndex->newBatchIterator(
                this->flat_iterator->getQueryBlob(), queryParams);
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
        TieredSVS_BatchIterator(const void *query_vector, const Index *index,
                                VecSimQueryParams *queryParams,
                                std::shared_ptr<VecSimAllocator> allocator)
            // Tiered batch iterator doesn't hold its own copy of the query vector.
            // Instead, each internal batch iterators (flat_iterator and svs_iterator) create their
            // own copies: flat_iterator copy is created during TieredSVS_BatchIterator
            // construction When TieredSVS_BatchIterator::getNextResults() is called and
            // svs_iterator is not initialized, it retrieves the blob from flat_iterator
            : VecSimBatchIterator(nullptr, queryParams ? queryParams->timeoutCtx : nullptr,
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

            const bool isMulti = this->index->backendIndex->isMultiValue();
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
                svs_results.swap(cur_svs_results->results);
                VecSimQueryReply_Free(cur_svs_results);
                handle_svs_depletion();
            } else {
                while (flat_results.size() < n_res && !flat_iterator->isDepleted()) {
                    auto tail = flat_iterator->getNextResults(n_res - flat_results.size(),
                                                              BY_SCORE_THEN_ID);
                    flat_results.insert(flat_results.end(), tail->results.begin(),
                                        tail->results.end());
                    VecSimQueryReply_Free(tail);

                    if (!isMulti) {
                        // On single-value indexes, duplicates will never appear in the hnsw results
                        // before they appear in the flat results (at the same time or later if the
                        // approximation misses) so we don't need to try and filter the flat results
                        // (and recheck conditions).
                        break;
                    } else {
                        // On multi-value indexes, the flat results may contain results that are
                        // already returned from the hnsw index. We need to filter them out.
                        filter_irrelevant_results(this->flat_results);
                    }
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
            batch = compute_current_batch(n_res, isMulti);

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
#else
    void executeTracingCallback(const std::string &) const {
        // In production, we do nothing.
    }
#endif

private:
    /**
     * @brief Updates the SVS index in a thread-safe manner.
     *
     * This static wrapper function performs the following actions:
     * - Acquires a lock on the index's updateJobMutex to prevent concurrent updates.
     * - Clears the indexUpdateScheduled flag to allow future scheduling.
     * - Configures the number of threads for the underlying SVS index update operation.
     * - Calls the updateSVSIndex method to perform the actual index update.
     *
     * @param idx Pointer to the VecSimIndex to be updated.
     * @param availableThreads The number of threads available for the update operation. Current
     * thread us used as well, so the minimal value is 1.
     */
    static void updateSVSIndexWrapper(VecSimIndex *idx, size_t availableThreads) {
        assert(availableThreads > 0);
        auto index = static_cast<TieredSVSIndex<DataType> *>(idx);
        assert(index);
        // prevent parallel updates
        std::lock_guard<std::mutex> lock(index->updateJobMutex);
        // Release the scheduled flag to allow scheduling again
        index->indexUpdateScheduled.clear();
        // Update the SVS index
        index->updateSVSIndex(availableThreads);
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
            this->allocator, SVS_BATCH_UPDATE_JOB, updateSVSIndexWrapper, this, total_threads,
            std::chrono::microseconds(updateJobWaitTime), &uncompletedJobs);
        this->submitJobs(jobs);
    }

private:
    static void applySwapsToLabelsArray(std::vector<size_t> &labels,
                                        const std::vector<swap_record> &swaps) {
        // Enumerate journal and reflect swaps in the labels.
        // The journal contains tuples of (label, oldId, newId).
        // oldId is the index of the label in flat index before the swap.
        // newId is the index of the label in flat index after the swap.
        for (const auto &p : swaps) {
            auto oldId = std::get<1>(p);
            auto newId = std::get<2>(p);

            if (oldId == newId || oldId >= labels.size()) {
                // If oldId == newId, it means that the vector was not moved in the Flat index,
                // but was removed or updated in-place.
                // If oldId is out of bounds - new vector was added and swapped meanwhile.
                // In both cases, we should not touch flat index at the new position.
                if (newId < labels.size()) {
                    labels[newId] = SKIP_LABEL;
                }
                continue; // Next swap record.
            }

            // Real swap case.
            // If oldId != newId and oldId is in bounds, it means that the tracked vector was moved
            // So, move the label to new position and skip the old position from deletions.
            // NOTE: labels[oldId] can be SKIP_LABEL, but we still need to move it to newId.
            labels[newId] = labels[oldId];
            labels[oldId] = SKIP_LABEL;
        }
    }

    void updateSVSIndex(size_t availableThreads) {
        std::vector<labelType> labels_to_move;
        std::vector<DataType> vectors_to_move;

        { // lock frontendIndex from modifications
            std::shared_lock flat_lock{this->flatIndexGuard};

            auto flat_index = this->GetFlatIndex();
            const auto frontend_index_size = this->frontendIndex->indexSize();
            const size_t dim = flat_index->getDim();
            labels_to_move.reserve(frontend_index_size);
            vectors_to_move.reserve(frontend_index_size * dim);

            for (idType i = 0; i < frontend_index_size; i++) {
                labels_to_move.push_back(flat_index->getVectorLabel(i));
                auto data = flat_index->getDataByInternalId(i);
                vectors_to_move.insert(vectors_to_move.end(), data, data + dim);
            }
            // reset journal to the current frontend index state
            swaps_journal.clear();
        } // release frontend index

        executeTracingCallback("UpdateJob::before_add_to_svs");
        { // lock backend index for writing and add vectors there
            std::lock_guard lock(this->mainIndexGuard);
            auto svs_index = GetSVSIndex();
            assert(labels_to_move.size() == vectors_to_move.size() / this->frontendIndex->getDim());
            svs_index->setNumThreads(std::min(availableThreads, labels_to_move.size()));
            svs_index->addVectors(vectors_to_move.data(), labels_to_move.data(),
                                  labels_to_move.size());
        }
        executeTracingCallback("UpdateJob::after_add_to_svs");
        // clean-up frontend index
        { // lock frontend index for writing and delete moved vectors
            std::lock_guard lock(this->flatIndexGuard);

            // Apply swaps from journal to labels_to_move to reflect changes made in meanwhile.
            applySwapsToLabelsArray(labels_to_move, this->swaps_journal);

            // delete vectors from the frontend index in reverse order
            // it increases the chance of avoiding swaps in the frontend index and performance
            // improvement
            int deleted = 0;
            idType id = labels_to_move.size();
            while (id-- > 0) {
                auto label = labels_to_move[id];
                // Delete the vector from the frontend index if not in-place updated.
                if (label != SKIP_LABEL) {
                    deleted += this->frontendIndex->deleteVectorById(label, id);
                }
            }
            assert(deleted == std::count_if(labels_to_move.begin(), labels_to_move.end(),
                                            [](labelType label) { return label != SKIP_LABEL; }) &&
                   "Deleted vectors count does not match the number of labels to delete");
        }
    }

public:
    TieredSVSIndex(VecSimIndexAbstract<DataType, float> *svs_index, flat_index_t *bf_index,
                   const TieredIndexParams &tiered_index_params,
                   std::shared_ptr<VecSimAllocator> allocator)
        : Base(svs_index, bf_index, tiered_index_params, allocator),
          uncompletedJobs(this->allocator) {
        const auto &tiered_svs_params = tiered_index_params.specificParams.tieredSVSParams;

        this->trainingTriggerThreshold =
            tiered_svs_params.trainingTriggerThreshold == 0
                ? SVS_VAMANA_DEFAULT_TRAINING_THRESHOLD
                : std::min(tiered_svs_params.trainingTriggerThreshold, SVS_MAX_TRAINING_THRESHOLD);

        // If flatBufferLimit is not initialized (0), use the default update threshold.
        const size_t flat_buffer_bound = tiered_index_params.flatBufferLimit == 0
                                             ? SVS_DEFAULT_UPDATE_THRESHOLD
                                             : tiered_index_params.flatBufferLimit;

        this->updateTriggerThreshold =
            tiered_svs_params.updateTriggerThreshold == 0
                ? SVS_DEFAULT_UPDATE_THRESHOLD
                : std::min({tiered_svs_params.updateTriggerThreshold, flat_buffer_bound,
                            SVS_DEFAULT_UPDATE_THRESHOLD});

        this->updateJobWaitTime = tiered_svs_params.updateJobWaitTime == 0
                                      ? SVS_DEFAULT_UPDATE_JOB_WAIT_TIME
                                      : tiered_svs_params.updateJobWaitTime;

        // Reserve space for the journal to avoid reallocation.
        this->swaps_journal.reserve(this->trainingTriggerThreshold);
    }

    int addVector(const void *blob, labelType label) override {
        int ret = 0;
        auto svs_index = GetSVSIndex();
        size_t update_threshold = 0;
        size_t frontend_index_size = 0;

        // In-Place mode - add vector syncronously to the backend index.
        if (this->getWriteMode() == VecSim_WriteInPlace) {
            // It is ok to lock everything at once for in-place mode,
            // but we will have to unlock averything before calling updateSVSIndexWrapper()
            // so make the minimal needed lock here.
            std::shared_lock backend_shared_lock(this->mainIndexGuard);
            // Backend index initialization data have to be buffered for proper
            // compression/training.
            if (this->backendIndex->indexSize() == 0) {
                // If backend index size is 0, first collect vectors in frontend index
                // lock in scope to ensure that these will be released before
                // updateSVSIndexWrapper() is called.
                {
                    std::lock_guard lock(this->flatIndexGuard);
                    ret = this->frontendIndex->addVector(blob, label);
                    // If frontend size exceeds the update job threshold, ...
                    frontend_index_size = this->frontendIndex->indexSize();
                }
                // ... move vectors to the backend index.
                if (frontend_index_size >= this->trainingTriggerThreshold) {
                    // updateSVSIndexWrapper() accures it's own locks
                    backend_shared_lock.unlock();
                    // initialize the SVS index synchonously using current thread only
                    updateSVSIndexWrapper(this, 1);
                }
                return ret;
            } else {
                // backend index is initialized - we can add the vector directly
                backend_shared_lock.unlock();
                auto storage_blob = this->frontendIndex->preprocessForStorage(blob);
                // prevent update job from running in parallel and lock any access to the backend
                // index
                std::scoped_lock lock(this->updateJobMutex, this->mainIndexGuard);
                return svs_index->addVectors(storage_blob.get(), &label, 1);
            }
        }
        assert(this->getWriteMode() != VecSim_WriteInPlace && "InPlace mode returns early");

        // Async mode - add vector to the frontend index and schedule an update job if needed.
        if (!this->backendIndex->isMultiValue()) {
            {
                std::shared_lock lock(this->flatIndexGuard);
                // If the label already exists in the frontend index, we should count it
                // to prevent the case when existing vector is moved meanwhile by the update job.
                if (this->frontendIndex->isLabelExists(label)) {
                    ret = -1;
                }
            }
            // Remove vector from the backend index if it exists in case of non-MULTI.
            std::lock_guard lock(this->mainIndexGuard);
            ret -= svs_index->deleteVectors(&label, 1);
        }
        { // Add vector to the frontend index.
            std::lock_guard lock(this->flatIndexGuard);
            const auto ft_ret = this->frontendIndex->addVector(blob, label);

            if (ft_ret == 0) { // Vector was overriden - add 'skiping' swap to the journal.
                assert(!this->backendIndex->isMultiValue() &&
                       "addVector() may return 0 for single value indices only");
                for (auto id : this->frontendIndex->getElementIds(label)) {
                    this->swaps_journal.emplace_back(SKIP_LABEL, id, id);
                }
            }
            ret = std::max(ret + ft_ret, 0);
            // Check frontend index size to determine if an update job schedule is needed.
            frontend_index_size = this->frontendIndex->indexSize();
        }
        {
            // If main index is empty then update_threshold is trainingTriggerThreshold,
            // overwise it is updateTriggerThreshold.
            std::shared_lock lock(this->mainIndexGuard);
            update_threshold = this->backendIndex->indexSize() == 0 ? this->trainingTriggerThreshold
                                                                    : this->updateTriggerThreshold;
        }
        if (frontend_index_size >= update_threshold) {
            scheduleSVSIndexUpdate();
        }

        return ret;
    }

    int deleteAndRecordSwaps_Unsafe(labelType label) {
        auto deleting_ids = this->frontendIndex->getElementIds(label);

        // assert if all elements of deleting_ids are unique
        assert(std::set(deleting_ids.begin(), deleting_ids.end()).size() == deleting_ids.size() &&
               "deleting_ids should contain unique ids");

        // Sort deleting_ids by id descending order
        std::sort(deleting_ids.begin(), deleting_ids.end(),
                  [](const auto &a, const auto &b) { return a > b; });

        // Delete vector from the frontend index.
        auto updated_ids = this->frontendIndex->deleteVectorAndGetUpdatedIds(label);

        assert(std::all_of(updated_ids.begin(), updated_ids.end(),
                           [&deleting_ids](const auto &pair) {
                               return std::find(deleting_ids.begin(), deleting_ids.end(),
                                                pair.first) != deleting_ids.end();
                           }) &&
               "updated_ids should be a subset of deleting_ids");

        // Record swaps in the journal.
        for (auto id : deleting_ids) {
            auto it = updated_ids.find(id);
            if (it != updated_ids.end()) {
                assert(id == it->first && "id in updated_ids should match the id in deleting_ids");
                auto newId = id;
                auto oldId = it->second.first;
                auto oldLabel = it->second.second;
                this->swaps_journal.emplace_back(oldLabel, oldId, newId);
            } else {
                // No swap, just delete is marked by oldId == newId == deleted id
                this->swaps_journal.emplace_back(SKIP_LABEL, id, id);
            }
        }

        return deleting_ids.size();
    }

    int deleteVector(labelType label) override {
        int ret = 0;
        auto svs_index = GetSVSIndex();
        // Backend index deletions to be synchronized with the frontend index,
        // elsewhere there is the risk of labels duplication in both indices which can lead to wrong
        // results of topK queries. In such case we should behave as if InPlace mode is always set.
        bool label_exists = [&]() {
            std::shared_lock lock(this->flatIndexGuard);
            return this->frontendIndex->isLabelExists(label);
        }();

        if (label_exists) {
            std::lock_guard lock(this->flatIndexGuard);
            ret = this->deleteAndRecordSwaps_Unsafe(label);
        }
        {
            std::lock_guard lock(this->mainIndexGuard);
            ret += svs_index->deleteVectors(&label, 1);
        }
        return ret;
    }

    size_t indexSize() const override {
        std::shared_lock<std::shared_mutex> flat_lock(this->flatIndexGuard);
        std::shared_lock<std::shared_mutex> main_lock(this->mainIndexGuard);
        return this->frontendIndex->indexSize() + this->backendIndex->indexSize();
    }

    size_t indexCapacity() const override {
        std::shared_lock<std::shared_mutex> flat_lock(this->flatIndexGuard);
        std::shared_lock<std::shared_mutex> main_lock(this->mainIndexGuard);
        return this->frontendIndex->indexCapacity() + this->backendIndex->indexCapacity();
    }

    double getDistanceFrom_Unsafe(labelType label, const void *blob) const override {
        // Try to get the distance from the flat buffer.
        // If the label doesn't exist, the distance will be NaN.
        auto flat_dist = this->frontendIndex->getDistanceFrom_Unsafe(label, blob);

        // Optimization. TODO: consider having different implementations for single and multi
        // indexes, to avoid checking the index type on every query.
        if (!this->backendIndex->isMultiValue() && !std::isnan(flat_dist)) {
            // If the index is single value, and we got a valid distance from the flat buffer,
            // we can return the distance without querying the Main index.
            return flat_dist;
        }

        // Try to get the distance from the Main index.
        auto svs_dist = this->backendIndex->getDistanceFrom_Unsafe(label, blob);

        // Return the minimum distance that is not NaN.
        return std::fmin(flat_dist, svs_dist);
    }

    VecSimIndexDebugInfo debugInfo() const override {
        auto info = Base::debugInfo();
        SvsTieredInfo svsTieredInfo = {.trainingTriggerThreshold = this->trainingTriggerThreshold,
                                       .updateTriggerThreshold = this->updateTriggerThreshold,
                                       .updateJobWaitTime = this->updateJobWaitTime,
                                       .indexUpdateScheduled =
                                           static_cast<bool>(this->indexUpdateScheduled.test())};
        info.tieredInfo.specificTieredBackendInfo.svsTieredInfo = svsTieredInfo;
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
        // TODO: Add SVS specific info.
        return infoIterator;
    }

    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override {
        if (this->GetSVSIndex()->isCompressed() || this->backendIndex->isMultiValue()) {
            // SVS compressed distance computation precision is lower, so we always have to
            // merge results with set.
            return this->template topKQueryImp<true>(queryBlob, k, queryParams);
        } else {
            // Calling with withSet=false for optimized performance, assuming that shared IDs across
            // lists also have identical scores — in which case duplicates are implicitly avoided by
            // the merge logic.
            return this->template topKQueryImp<false>(queryBlob, k, queryParams);
        }
    }

    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams,
                                 VecSimQueryReply_Order order) const override {
        if (this->GetSVSIndex()->isCompressed() || this->backendIndex->isMultiValue()) {
            // SVS compressed distance computation precision is lower, so we always have to
            // merge results with set.
            return this->template rangeQueryImp<true>(queryBlob, radius, queryParams, order);
        } else {
            // Calling with withSet=false for optimized performance, assuming that shared IDs across
            // lists also have identical scores — in which case duplicates are implicitly avoided by
            // the merge logic.
            return this->template rangeQueryImp<false>(queryBlob, radius, queryParams, order);
        }
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
        TIERED_LOG(VecSimCommonStrings::LOG_VERBOSE_STRING,
                   "running asynchronous GC for tiered SVS index");
        std::lock_guard lock{this->mainIndexGuard};
        // VecSimIndexAbstract::runGC() is protected
        static_cast<VecSimIndexInterface *>(this->backendIndex)->runGC();
    }

    void acquireSharedLocks() override {
        this->flatIndexGuard.lock_shared();
        this->mainIndexGuard.lock_shared();
    }

    void releaseSharedLocks() override {
        this->mainIndexGuard.unlock_shared();
        this->flatIndexGuard.unlock_shared();
    }
};

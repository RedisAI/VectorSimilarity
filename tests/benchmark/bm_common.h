/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include "bm_vecsim_index.h"

size_t BM_VecSimGeneral::block_size = 1024;

// Helper function to calculate percentile from a sorted vector
static double calculate_percentile(std::vector<double> &values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(values.size() * percentile);
    if (index >= values.size()) {
        index = values.size() - 1;
    }
    return values[index];
}

// Class for common bm for basic index and updated index.
template <typename index_type_t>
class BM_VecSimCommon : public BM_VecSimIndex<index_type_t> {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    BM_VecSimCommon() = default;
    ~BM_VecSimCommon() = default;

    // index_offset: Offset added to base index types to access variants (0=original, 1=updated)

    static void RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                             std::atomic_int &correct, unsigned short index_offset = 0);

    // Search for the K closest vectors to the query in the index. K is defined in the
    // test registration (initialization file).
    static void TopK_BF(benchmark::State &st, unsigned short index_offset = 0);
    // Run TopK using both HNSW and flat index and calculate the recall of the HNSW algorithm
    // with respect to the results returned by the flat index.
    static void TopK_HNSW(benchmark::State &st, unsigned short index_offset = 0);
    static void TopK_HNSW_DISK(benchmark::State &st);
    static void TopK_HNSW_DISK_Parallel(benchmark::State &st);
    static void TopK_HNSW_DISK_Parallel_WithInserts(benchmark::State &st);
    static void TopK_HNSW_DISK_MarkDeleted(benchmark::State &st);
    static void TopK_HNSW_DISK_DeleteLabel(benchmark::State &st);
    // Same as DeleteLabel but excludes ground truth vectors from deletion to keep recall stable
    static void TopK_HNSW_DISK_DeleteLabel_ProtectGT(benchmark::State &st);
    // Test deletion performance with different batch sizes
    static void TopK_HNSW_DISK_DeleteLabel_BatchSize(benchmark::State &st);
    // Stress test with high deletion ratios (50%, 75%, 90%+)
    static void TopK_HNSW_DISK_DeleteLabel_Stress(benchmark::State &st);
    static void TopK_Tiered(benchmark::State &st, unsigned short index_offset = 0);

    // Does nothing but returning the index memory.
    static void Memory(benchmark::State &st, IndexTypeIndex index_type);
    static void Disk(benchmark::State &st, IndexTypeIndex index_type);
};

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter,
                                                 size_t k, std::atomic_int &correct,
                                                 unsigned short index_offset) {
    HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
    auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);

    auto hnsw_index = GET_INDEX(INDEX_HNSW + index_offset);
    auto bf_index = GET_INDEX(INDEX_BF + index_offset);
    auto &q = QUERIES[iter % N_QUERIES];

    auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);
    st.PauseTiming();

    // Measure recall:
    auto bf_results = VecSimIndex_TopKQuery(bf_index, q.data(), k, nullptr, BY_SCORE);

    BM_VecSimGeneral::MeasureRecall(hnsw_results, bf_results, correct);

    VecSimQueryReply_Free(bf_results);
    VecSimQueryReply_Free(hnsw_results);
    st.ResumeTiming();
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::Disk(benchmark::State &st, IndexTypeIndex index_type) {
    auto index = static_cast<HNSWDiskIndex<data_t, dist_t> *>(GET_INDEX(index_type));

    for (auto _ : st) {
        // Do nothing...
    }
    // Output detailed RocksDB memory usage breakdown
    auto mem_breakdown = index->getDBMemoryBreakdown();
    st.counters["rocksdb_memory_total"] = static_cast<double>(mem_breakdown.total);
    st.counters["rocksdb_memtables"] = static_cast<double>(mem_breakdown.memtables);
    st.counters["rocksdb_table_readers"] = static_cast<double>(mem_breakdown.table_readers);
    st.counters["rocksdb_block_cache"] = static_cast<double>(mem_breakdown.block_cache);
    st.counters["rocksdb_pinned_blocks"] = static_cast<double>(mem_breakdown.pinned_blocks);
    st.counters["db_disk"] = (double)VecSimIndex_StatsInfo(index).db_disk;
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::Memory(benchmark::State &st, IndexTypeIndex index_type) {
    auto index = GET_INDEX(index_type);
    index->fitMemory();

    for (auto _ : st) {
        // Do nothing...
    }

    st.counters["memory"] = (double)VecSimIndex_StatsInfo(index).memory;
    st.counters["vectors_memory"] = (double)VecSimIndex_StatsInfo(index).vectors_memory;
}

// TopK search BM

// Run TopK using disk-based HNSW index vs BF to measure recall
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW_DISK(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    std::atomic_int correct = 0;
    size_t iter = 0;
    auto hnsw_index = GET_INDEX(INDEX_HNSW_DISK);

    // Get DB statistics if available
    auto hnsw_disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(hnsw_index);
    auto db_stats = hnsw_disk_index->getDBStatistics();
    size_t cache_misses = 0;
    if (db_stats) {
        cache_misses = db_stats->getTickerCount(rocksdb::Tickers::BLOCK_CACHE_MISS);
    }

    // Collect individual query times for percentile calculation
    std::vector<double> query_times;

    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];
        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);

        // Collect execution time from the query reply
        query_times.push_back(VecSimQueryReply_GetExecutionTime(hnsw_results));

        st.PauseTiming();
        auto bf_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(iter % N_QUERIES, k);
        BM_VecSimGeneral::MeasureRecall(hnsw_results, bf_results, correct);
        VecSimQueryReply_Free(bf_results);
        VecSimQueryReply_Free(hnsw_results);
        st.ResumeTiming();
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);

    // Calculate and report percentiles
    if (!query_times.empty()) {
        double sum = std::accumulate(query_times.begin(), query_times.end(), 0.0);
        st.counters["ms_per_query"] = sum / query_times.size();
        st.counters["p50_ms"] = calculate_percentile(query_times, 0.50);
        st.counters["p99_ms"] = calculate_percentile(query_times, 0.99);
    }

    if (db_stats) {
        cache_misses = db_stats->getTickerCount(rocksdb::Tickers::BLOCK_CACHE_MISS) - cache_misses;
        st.counters["cache_misses_per_query"] = static_cast<double>(cache_misses) / iter;
    }

    // Output detailed RocksDB memory usage breakdown
    auto mem_breakdown = hnsw_disk_index->getDBMemoryBreakdown();
    st.counters["rocksdb_memory_total"] = static_cast<double>(mem_breakdown.total);
    st.counters["rocksdb_memtables"] = static_cast<double>(mem_breakdown.memtables);
    st.counters["rocksdb_table_readers"] = static_cast<double>(mem_breakdown.table_readers);
    st.counters["rocksdb_block_cache"] = static_cast<double>(mem_breakdown.block_cache);
    st.counters["rocksdb_pinned_blocks"] = static_cast<double>(mem_breakdown.pinned_blocks);
}

// Run TopK using disk-based HNSW index vs BF to measure recall (parallel).
// This benchmark uses the global mock_thread_pool to drive concurrent queries.
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW_DISK_Parallel(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    size_t concurrency = st.range(2);

    auto hnsw_index = GET_INDEX(INDEX_HNSW_DISK);
    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(hnsw_index);

    auto *pool = BM_VecSimGeneral::mock_thread_pool;

    if (!disk_index || !pool || concurrency == 0) {
        // Fallback: single-threaded behavior if the pool is not available or misconfigured.
        TopK_HNSW_DISK(st);
        st.counters["concurrency"] = static_cast<double>(concurrency ? concurrency : 1);
        return;
    }

    pool->reconfigure_threads(concurrency);

    auto db_stats = disk_index->getDBStatistics();
    size_t block_cache_miss = 0;
    if (db_stats) {
        block_cache_miss = db_stats->getTickerCount(rocksdb::Tickers::BLOCK_CACHE_MISS);
    }

    // Limit by the number of distinct queries we actually have.
    const size_t max_queries = static_cast<size_t>(N_QUERIES);
    const size_t num_queries = max_queries;

    // We perform the full parallel run once (Iterations=1), but keep the
    // benchmark loop structure so Google Benchmark measures the body time
    // correctly.
    for (auto _ : st) {
        (void)_;

        std::atomic_int correct{0};

        size_t bytes_read_before = 0;
        if (db_stats) {
            bytes_read_before = db_stats->getTickerCount(rocksdb::Tickers::BYTES_READ);
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // Submit all queries once (0..num_queries-1) and let the thread pool, whose
        // size equals `concurrency`, drain the work queue. We never reuse a query
        // index, so there is no hot-cache bias from repeated queries.
        std::vector<VecSimQueryReply *> all_results(num_queries, nullptr);
        std::vector<AsyncJob *> jobs(num_queries);
        std::vector<JobCallback> cbs(num_queries);

        for (size_t q_idx = 0; q_idx < num_queries; ++q_idx) {
            auto allocator = disk_index->getAllocator();
            auto search_job = new (allocator)
                tieredIndexMock::SearchJobMock(allocator, [](AsyncJob *job) {
                    auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
                    HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = search_job->ef};
                    auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
                    size_t q_idx_inner = search_job->iter;

                    auto hnsw_results = VecSimIndex_TopKQuery(
                        GET_INDEX(INDEX_HNSW_DISK),
                        QUERIES[q_idx_inner].data(),
                        search_job->k,
                        &query_params,
                        BY_SCORE);

                    // all_results points to a single slot for this job.
                    search_job->all_results[0] = hnsw_results;
                    delete job;
                },
                                               hnsw_index, k, ef, q_idx, &all_results[q_idx]);

            jobs[q_idx] = search_job;
            cbs[q_idx] = search_job->Execute;
        }

        pool->submit_callback_internal(jobs.data(), cbs.data(), num_queries);
        pool->thread_pool_wait();

        // Collect individual query times for percentile calculation
        std::vector<double> query_times;
        query_times.reserve(num_queries);

        // Measure recall using fresh GT queries; each query index is used once.
        for (size_t q_idx = 0; q_idx < num_queries; ++q_idx) {
            auto bf_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(q_idx, k);
            BM_VecSimGeneral::MeasureRecall(all_results[q_idx], bf_results, correct);

            // Collect execution time from the query reply
            query_times.push_back(VecSimQueryReply_GetExecutionTime(all_results[q_idx]));

            VecSimQueryReply_Free(bf_results);
            VecSimQueryReply_Free(all_results[q_idx]);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        size_t executed_queries = num_queries;

        // Calculate and report percentiles from individual query times
        double ms_per_query = 0.0;
        if (!query_times.empty()) {
            double sum = std::accumulate(query_times.begin(), query_times.end(), 0.0);
            ms_per_query = sum / query_times.size();
            st.counters["p50_ms"] = calculate_percentile(query_times, 0.50);
            st.counters["p99_ms"] = calculate_percentile(query_times, 0.99);
        }

        double qps = total_ms > 0.0 ?
            static_cast<double>(executed_queries) / (total_ms / 1000.0) : 0.0;

        // st.counters["concurrency"] = static_cast<double>(concurrency);
        // st.counters["num_queries"] = static_cast<double>(executed_queries);
        st.counters["avg_ms"] = ms_per_query;
        st.counters["total_time_ms"] = total_ms;
        st.counters["qps"] = qps;
        st.counters["Recall"] = executed_queries ?
            static_cast<double>(correct.load()) / static_cast<double>(k * executed_queries) : 0.0;

        if (db_stats && executed_queries > 0) {
            size_t total_cache_miss = db_stats->getTickerCount(rocksdb::Tickers::BLOCK_CACHE_MISS) - block_cache_miss;
            st.counters["cache_misses_per_query"] = static_cast<double>(total_cache_miss) / executed_queries;
        }
    }
}

// Run TopK using disk-based HNSW index with concurrent inserts from a background writer thread.
// Measures search QPS and insert throughput while both run concurrently.
// Test ends when all searches complete; insert count at that time is recorded.
// st.range(0) = ef_runtime
// st.range(1) = k
// st.range(2) = concurrency (number of search threads)
// st.range(3) = batch_threshold (vectors accumulated before graph update)
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW_DISK_Parallel_WithInserts(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    size_t concurrency = st.range(2);
    size_t batch_threshold = st.range(3);

    // Note: We don't reload the index here to avoid cleanup issues.
    // Each run will add vectors to the existing index.
    auto hnsw_index = GET_INDEX(INDEX_HNSW_DISK);
    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(hnsw_index);

    auto *pool = BM_VecSimGeneral::mock_thread_pool;

    if (!disk_index || !pool || concurrency == 0) {
        st.SkipWithError("Disk index or thread pool not available");
        return;
    }

    pool->reconfigure_threads(concurrency);
    disk_index->setBatchThreshold(batch_threshold);

    // Load insert vectors from file
    const char *inserts_file = "tests/benchmark/data/deep1B-inserts-100K.fbin";
    std::string inserts_path = BM_VecSimGeneral::AttachRootPath(inserts_file);
    std::ifstream file(inserts_path, std::ios::binary);
    if (!file.is_open()) {
        st.SkipWithError("Failed to open inserts file");
        return;
    }

    uint32_t num_insert_vectors = 0;
    uint32_t insert_dim = 0;
    file.read(reinterpret_cast<char *>(&num_insert_vectors), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&insert_dim), sizeof(uint32_t));

    size_t index_dim = disk_index->getDim();
    if (insert_dim != index_dim) {
        st.SkipWithError("Insert vectors dimension mismatch");
        return;
    }

    // Read all insert vectors into memory
    std::vector<std::vector<float>> insert_vectors(num_insert_vectors);
    for (size_t i = 0; i < num_insert_vectors; i++) {
        insert_vectors[i].resize(insert_dim);
        file.read(reinterpret_cast<char *>(insert_vectors[i].data()), insert_dim * sizeof(float));
    }
    file.close();

    // Get starting label for inserts (after existing vectors)
    labelType next_label = disk_index->indexSize();

    const size_t num_queries = static_cast<size_t>(N_QUERIES);

    for (auto _ : st) {
        (void)_;

        std::atomic_int correct{0};
        std::atomic<size_t> vectors_inserted{0};
        std::atomic<bool> stop_writer{false};

        auto t_start = std::chrono::high_resolution_clock::now();

        // Start background writer thread - inserts continuously until stopped
        std::thread writer_thread([&]() {
            size_t insert_idx = 0;
            while (!stop_writer.load(std::memory_order_relaxed) && insert_idx < num_insert_vectors) {
                disk_index->addVector(insert_vectors[insert_idx].data(), next_label + insert_idx);
                vectors_inserted.fetch_add(1, std::memory_order_relaxed);
                insert_idx++;
            }
        });

        // Submit all search queries
        std::vector<VecSimQueryReply *> all_results(num_queries, nullptr);
        std::vector<AsyncJob *> jobs(num_queries);
        std::vector<JobCallback> cbs(num_queries);

        for (size_t q_idx = 0; q_idx < num_queries; ++q_idx) {
            auto allocator = disk_index->getAllocator();
            auto search_job = new (allocator)
                tieredIndexMock::SearchJobMock(allocator, [](AsyncJob *job) {
                    auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
                    HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = search_job->ef};
                    auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
                    size_t q_idx_inner = search_job->iter;

                    auto hnsw_results = VecSimIndex_TopKQuery(
                        GET_INDEX(INDEX_HNSW_DISK),
                        QUERIES[q_idx_inner].data(),
                        search_job->k,
                        &query_params,
                        BY_SCORE);

                    search_job->all_results[0] = hnsw_results;
                    delete job;
                },
                                               hnsw_index, k, ef, q_idx, &all_results[q_idx]);

            jobs[q_idx] = search_job;
            cbs[q_idx] = search_job->Execute;
        }

        pool->submit_callback_internal(jobs.data(), cbs.data(), num_queries);
        pool->thread_pool_wait();

        // Stop writer and record time when searches complete
        auto t_end = std::chrono::high_resolution_clock::now();
        stop_writer.store(true, std::memory_order_relaxed);
        writer_thread.join();

        double total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        size_t final_vectors_inserted = vectors_inserted.load();

        // Measure recall
        for (size_t q_idx = 0; q_idx < num_queries; ++q_idx) {
            auto bf_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(q_idx, k);
            BM_VecSimGeneral::MeasureRecall(all_results[q_idx], bf_results, correct);
            VecSimQueryReply_Free(bf_results);
            VecSimQueryReply_Free(all_results[q_idx]);
        }

        double qps = total_ms > 0.0 ?
            static_cast<double>(num_queries) / (total_ms / 1000.0) : 0.0;
        double inserts_per_sec = total_ms > 0.0 ?
            static_cast<double>(final_vectors_inserted) / (total_ms / 1000.0) : 0.0;

        st.counters["total_time_ms"] = total_ms;
        st.counters["search_qps"] = qps;
        st.counters["insert_qps"] = inserts_per_sec;
        st.counters["Recall"] = num_queries ?
            static_cast<double>(correct.load()) / static_cast<double>(k * num_queries) : 0.0;
        st.counters["vectors_inserted"] = static_cast<double>(final_vectors_inserted);
        st.counters["index_size_after"] = static_cast<double>(disk_index->indexSize());
    }
}

// Benchmark TopK performance with marked deleted vectors
// st.range(0) = ef_runtime
// st.range(1) = k
// st.range(2) = number of vectors to mark as deleted
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW_DISK_MarkDeleted(benchmark::State &st) {
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    size_t iter = 0;

    // Reload the index to get a fresh copy without any marked deleted vectors
    std::string folder_path = BM_VecSimGeneral::AttachRootPath(BM_VecSimGeneral::hnsw_index_file);
    INDICES[INDEX_HNSW_DISK] = IndexPtr(HNSWDiskFactory::NewIndex(folder_path));
    auto hnsw_index = GET_INDEX(INDEX_HNSW_DISK);
    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(hnsw_index);

    // Mark vectors as deleted before starting the benchmark
    // We mark vectors starting from label 0, ensuring they exist in the index
    std::vector<labelType> deleted_labels;
    const size_t num_to_delete = st.range(2);

    // get psuedo random unique labels, but the same ones for all runs of the benchmark across different runs
    // Divide N_VECTORS into num_to_delete equal strata and pick one from each
    std::mt19937 rng(42); // Fixed seed for determinism
    for (size_t i = 0; i < num_to_delete && i < N_VECTORS; i++) {
        size_t stratum_start = (i * N_VECTORS) / num_to_delete;
        size_t stratum_end = ((i + 1) * N_VECTORS) / num_to_delete;
        size_t stratum_size = stratum_end - stratum_start;

        std::uniform_int_distribution<size_t> dist(0, stratum_size - 1);
        labelType label = stratum_start + dist(rng);
        deleted_labels.push_back(label);
        disk_index->markDelete(label);
    }

    // Create hash set for O(1) lookup during ground truth filtering
    // With up to 50K deleted labels, this avoids O(n) linear search overhead
    std::unordered_set<labelType> deleted_labels_set(deleted_labels.begin(), deleted_labels.end());

    size_t total_marked = disk_index->getNumMarkedDeleted();
    st.counters["num_marked_deleted"] = total_marked;

    // Get DB statistics before benchmark
    auto stats = disk_index->getDBStatistics();
    size_t io_bytes_before = 0;
    if (stats) {
        io_bytes_before = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
    }

    std::atomic_int correct = 0;
    size_t ef = st.range(0);
    size_t k = st.range(1);

    // Collect individual query times for percentile calculation
    std::vector<double> query_times;

    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];

        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);

        // Collect execution time from the query reply
        query_times.push_back(VecSimQueryReply_GetExecutionTime(hnsw_results));

        st.PauseTiming();

        // get all (100) ground truth results
        auto gt_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(iter % N_QUERIES, 100);

        auto filtered_res = new VecSimQueryReply(VecSimAllocator::newVecsimAllocator());
        for (const auto &res : gt_results->results) {
            // Use hash set for O(1) lookup instead of O(n) linear search
            if (deleted_labels_set.find(res.id) == deleted_labels_set.end()) {
                filtered_res->results.emplace_back(res.id, res.score);
                // Stop once we have k non-deleted results
                if (filtered_res->results.size() >= k) {
                    break;
                }
            }
        }
        if (filtered_res->results.size() < k) {
            std::cout << "Not enough non-deleted ground truth results to compare against (only " << filtered_res->results.size() << " out of " << k << " requested)" << std::endl;
        }

        BM_VecSimGeneral::MeasureRecall(hnsw_results, filtered_res, correct);

        VecSimQueryReply_Free(hnsw_results);
        VecSimQueryReply_Free(filtered_res);
        VecSimQueryReply_Free(gt_results);
        st.ResumeTiming();
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);

    // Calculate and report percentiles
    if (!query_times.empty()) {
        double sum = std::accumulate(query_times.begin(), query_times.end(), 0.0);
        st.counters["ms_per_query"] = sum / query_times.size();
        st.counters["p50_ms"] = calculate_percentile(query_times, 0.50);
        st.counters["p99_ms"] = calculate_percentile(query_times, 0.99);
    }

    if (stats) {
        size_t io_bytes_after = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
        st.counters["io_bytes_per_query"] = static_cast<double>(io_bytes_after - io_bytes_before) / iter;
    }

    // Output detailed RocksDB memory usage breakdown
    auto mem_breakdown = disk_index->getDBMemoryBreakdown();
    st.counters["rocksdb_memory_total"] = static_cast<double>(mem_breakdown.total);
    st.counters["rocksdb_memtables"] = static_cast<double>(mem_breakdown.memtables);
    st.counters["rocksdb_table_readers"] = static_cast<double>(mem_breakdown.table_readers);
    st.counters["rocksdb_block_cache"] = static_cast<double>(mem_breakdown.block_cache);
    st.counters["rocksdb_pinned_blocks"] = static_cast<double>(mem_breakdown.pinned_blocks);
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW_DISK_DeleteLabel(benchmark::State &st) {
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    size_t iter = 0;

    // Reload the index to get a fresh copy without any deleted vectors
    std::string folder_path = BM_VecSimGeneral::AttachRootPath(BM_VecSimGeneral::hnsw_index_file);
    INDICES[INDEX_HNSW_DISK] = IndexPtr(HNSWDiskFactory::NewIndex(folder_path));
    auto hnsw_index = GET_INDEX(INDEX_HNSW_DISK);
    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(hnsw_index);

    // Delete vectors using deleteVector (which processes batch and repairs graph)
    std::vector<labelType> deleted_labels;
    const size_t num_to_delete = st.range(2);

    // Get pseudo-random unique labels, but the same ones for all runs of the benchmark
    // Divide N_VECTORS into num_to_delete equal strata and pick one from each
    std::mt19937 rng(42); // Fixed seed for determinism
    for (size_t i = 0; i < num_to_delete && i < N_VECTORS; i++) {
        size_t stratum_start = (i * N_VECTORS) / num_to_delete;
        size_t stratum_end = ((i + 1) * N_VECTORS) / num_to_delete;
        size_t stratum_size = stratum_end - stratum_start;

        std::uniform_int_distribution<size_t> dist(0, stratum_size - 1);
        labelType label = stratum_start + dist(rng);
        deleted_labels.push_back(label);
    }

    // Measure the time spent on deleteVector calls (includes automatic batch processing)
    auto delete_start = std::chrono::high_resolution_clock::now();
    for (const auto &label : deleted_labels) {
        disk_index->deleteVector(label);
    }
    // Force flush any pending deletes to ensure graph is fully repaired
    disk_index->flushDeleteBatch();
    auto delete_end = std::chrono::high_resolution_clock::now();
    double delete_time_ms = std::chrono::duration<double, std::milli>(delete_end - delete_start).count();

    // Create hash set for O(1) lookup during ground truth filtering
    // With up to 50K deleted labels, this avoids O(n) linear search overhead
    std::unordered_set<labelType> deleted_labels_set(deleted_labels.begin(), deleted_labels.end());

    size_t total_deleted = deleted_labels.size();
    st.counters["num_deleted"] = total_deleted;
    st.counters["delete_time_ms"] = delete_time_ms;
    if (total_deleted > 0) {
        st.counters["delete_time_per_vector_ms"] = delete_time_ms / total_deleted;
    } else {
        st.counters["delete_time_per_vector_ms"] = 0.0;
    }

    // Get DB statistics before benchmark
    auto stats = disk_index->getDBStatistics();
    size_t io_bytes_before = 0;
    if (stats) {
        io_bytes_before = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
    }

    std::atomic_int correct = 0;
    size_t ef = st.range(0);
    size_t k = st.range(1);

    // Collect individual query times for percentile calculation
    std::vector<double> query_times;

    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];

        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);

        // Collect execution time from the query reply
        query_times.push_back(VecSimQueryReply_GetExecutionTime(hnsw_results));

        st.PauseTiming();

        // Get all (100) ground truth results
        auto gt_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(iter % N_QUERIES, 100);

        auto filtered_res = new VecSimQueryReply(VecSimAllocator::newVecsimAllocator());
        for (const auto &res : gt_results->results) {
            // Use hash set for O(1) lookup instead of O(n) linear search
            if (deleted_labels_set.find(res.id) == deleted_labels_set.end()) {
                filtered_res->results.emplace_back(res.id, res.score);
                // Stop once we have k non-deleted results
                if (filtered_res->results.size() >= k) {
                    break;
                }
            }
        }
        if (filtered_res->results.size() < k) {
            std::cout << "Not enough non-deleted ground truth results to compare against (only "
                      << filtered_res->results.size() << " out of " << k << " requested)" << std::endl;
        }

        BM_VecSimGeneral::MeasureRecall(hnsw_results, filtered_res, correct);

        VecSimQueryReply_Free(hnsw_results);
        VecSimQueryReply_Free(filtered_res);
        VecSimQueryReply_Free(gt_results);
        st.ResumeTiming();
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);

    // Calculate and report percentiles
    if (!query_times.empty()) {
        double sum = std::accumulate(query_times.begin(), query_times.end(), 0.0);
        st.counters["ms_per_query"] = sum / query_times.size();
        st.counters["p50_ms"] = calculate_percentile(query_times, 0.50);
        st.counters["p99_ms"] = calculate_percentile(query_times, 0.99);
    }

    if (stats) {
        size_t io_bytes_after = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
        st.counters["io_bytes_per_query"] = static_cast<double>(io_bytes_after - io_bytes_before) / iter;
    }
}

// Same as TopK_HNSW_DISK_DeleteLabel but excludes ground truth vectors from deletion.
// This keeps the ground truth stable across different deletion counts for fair recall comparison.
// st.range(0) = ef_runtime
// st.range(1) = k
// st.range(2) = number of vectors to delete
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW_DISK_DeleteLabel_ProtectGT(benchmark::State &st) {
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    // Build a set of all ground truth vector IDs (to protect from deletion)
    // Note: We collect GT labels for ALL queries, not just st.iterations() which returns 0 before the loop
    std::unordered_set<labelType> gt_labels_set;
    for (size_t q = 0; q < N_QUERIES; q++) {
        auto gt_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(q, 100);
        for (const auto &res : gt_results->results) {
            gt_labels_set.insert(res.id);
        }
        VecSimQueryReply_Free(gt_results);
    }

    // Reload the index to get a fresh copy without any deleted vectors
    std::string folder_path = BM_VecSimGeneral::AttachRootPath(BM_VecSimGeneral::hnsw_index_file);
    INDICES[INDEX_HNSW_DISK] = IndexPtr(HNSWDiskFactory::NewIndex(folder_path));
    auto hnsw_index = GET_INDEX(INDEX_HNSW_DISK);
    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(hnsw_index);

    // Delete vectors using deleteVector, but skip ground truth vectors
    std::vector<labelType> deleted_labels;
    const size_t num_to_delete = st.range(2);

    // Get pseudo-random unique labels, but the same ones for all runs of the benchmark
    // Divide N_VECTORS into num_to_delete equal strata and pick one from each
    // Skip any labels that are in ground truth
    std::mt19937 rng(42); // Fixed seed for determinism
    size_t skipped_gt = 0;
    for (size_t i = 0; i < num_to_delete && i < N_VECTORS; i++) {
        size_t stratum_start = (i * N_VECTORS) / num_to_delete;
        size_t stratum_end = ((i + 1) * N_VECTORS) / num_to_delete;
        size_t stratum_size = stratum_end - stratum_start;

        std::uniform_int_distribution<size_t> dist(0, stratum_size - 1);
        labelType label = stratum_start + dist(rng);

        // Skip if this label is in ground truth
        if (gt_labels_set.find(label) != gt_labels_set.end()) {
            skipped_gt++;
            continue;
        }
        deleted_labels.push_back(label);
    }

    // Measure the time spent on deleteVector calls (includes automatic batch processing)
    auto delete_start = std::chrono::high_resolution_clock::now();
    for (const auto &label : deleted_labels) {
        disk_index->deleteVector(label);
    }
    // Force flush any pending deletes to ensure graph is fully repaired
    disk_index->flushDeleteBatch();
    auto delete_end = std::chrono::high_resolution_clock::now();
    double delete_time_ms = std::chrono::duration<double, std::milli>(delete_end - delete_start).count();

    size_t total_deleted = deleted_labels.size();
    st.counters["num_deleted"] = total_deleted;
    st.counters["num_gt_protected"] = skipped_gt;
    st.counters["delete_time_ms"] = delete_time_ms;
    if (total_deleted > 0) {
        st.counters["delete_time_per_vector_ms"] = delete_time_ms / total_deleted;
    }

    // Get DB statistics before benchmark
    auto stats = disk_index->getDBStatistics();
    size_t io_bytes_before = 0;
    if (stats) {
        io_bytes_before = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
    }

    size_t iter = 0;
    std::atomic_int correct = 0;
    size_t ef = st.range(0);
    size_t k = st.range(1);

    // Collect individual query times for percentile calculation
    std::vector<double> query_times;

    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];

        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);

        // Collect execution time from the query reply
        query_times.push_back(VecSimQueryReply_GetExecutionTime(hnsw_results));

        st.PauseTiming();

        // Ground truth is unchanged since we protected all GT vectors from deletion
        auto gt_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(iter % N_QUERIES, k);

        BM_VecSimGeneral::MeasureRecall(hnsw_results, gt_results, correct);

        VecSimQueryReply_Free(hnsw_results);
        VecSimQueryReply_Free(gt_results);
        st.ResumeTiming();
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);

    // Calculate and report percentiles
    if (!query_times.empty()) {
        double sum = std::accumulate(query_times.begin(), query_times.end(), 0.0);
        st.counters["ms_per_query"] = sum / query_times.size();
        st.counters["p50_ms"] = calculate_percentile(query_times, 0.50);
        st.counters["p99_ms"] = calculate_percentile(query_times, 0.99);
    }

    if (stats) {
        size_t io_bytes_after = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
        st.counters["io_bytes_per_query"] = static_cast<double>(io_bytes_after - io_bytes_before) / iter;
    }
}

// Benchmark deletion performance with different batch sizes
// st.range(0) = ef_runtime
// st.range(1) = num_deleted
// st.range(2) = batch_size
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW_DISK_DeleteLabel_BatchSize(benchmark::State &st) {
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    size_t iter = 0;
    const size_t ef = st.range(0);
    const size_t num_to_delete = st.range(1);
    const size_t batch_size = st.range(2);

    // Reload the index to get a fresh copy without any deleted vectors
    std::string folder_path = BM_VecSimGeneral::AttachRootPath(BM_VecSimGeneral::hnsw_index_file);
    INDICES[INDEX_HNSW_DISK] = IndexPtr(HNSWDiskFactory::NewIndex(folder_path));
    auto hnsw_index = GET_INDEX(INDEX_HNSW_DISK);
    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(hnsw_index);

    // Set the batch threshold to the specified batch size
    disk_index->setDeleteBatchThreshold(batch_size);

    // Get pseudo-random unique labels using stratified sampling
    std::vector<labelType> deleted_labels;
    std::mt19937 rng(42); // Fixed seed for determinism
    for (size_t i = 0; i < num_to_delete && i < N_VECTORS; i++) {
        size_t stratum_start = (i * N_VECTORS) / num_to_delete;
        size_t stratum_end = ((i + 1) * N_VECTORS) / num_to_delete;
        size_t stratum_size = stratum_end - stratum_start;

        std::uniform_int_distribution<size_t> dist(0, stratum_size - 1);
        labelType label = stratum_start + dist(rng);
        deleted_labels.push_back(label);
    }

    // Create hash set for O(1) lookup during ground truth filtering
    std::unordered_set<labelType> deleted_labels_set(deleted_labels.begin(), deleted_labels.end());

    // Measure deletion time (includes batch processing triggered automatically)
    auto delete_start = std::chrono::high_resolution_clock::now();
    size_t batch_flushes = 0;
    for (const auto &label : deleted_labels) {
        size_t pending_before = disk_index->getPendingDeleteCount();
        disk_index->deleteVector(label);
        // Count when a batch flush was triggered
        if (disk_index->getPendingDeleteCount() < pending_before) {
            batch_flushes++;
        }
    }
    // Force flush any remaining pending deletes
    if (disk_index->getPendingDeleteCount() > 0) {
        disk_index->flushDeleteBatch();
        batch_flushes++;
    }
    auto delete_end = std::chrono::high_resolution_clock::now();
    double delete_time_ms = std::chrono::duration<double, std::milli>(delete_end - delete_start).count();

    // Report metrics
    st.counters["batch_size"] = batch_size;
    st.counters["num_deleted"] = deleted_labels.size();
    st.counters["delete_time_ms"] = delete_time_ms;
    st.counters["batch_flushes"] = batch_flushes;
    if (deleted_labels.size() > 0) {
        st.counters["delete_time_per_vector_ms"] = delete_time_ms / deleted_labels.size();
    }

    // Get DB statistics before search benchmark
    auto stats = disk_index->getDBStatistics();
    size_t io_bytes_before = 0;
    if (stats) {
        io_bytes_before = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
    }

    std::atomic_int correct = 0;
    size_t k = 10; // Fixed k for batch size testing

    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];

        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);
        st.PauseTiming();

        // Get ground truth and filter deleted vectors
        auto gt_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(iter % N_QUERIES, 100);

        auto filtered_res = new VecSimQueryReply(VecSimAllocator::newVecsimAllocator());
        for (const auto &res : gt_results->results) {
            if (deleted_labels_set.find(res.id) == deleted_labels_set.end()) {
                filtered_res->results.emplace_back(res.id, res.score);
                if (filtered_res->results.size() >= k) {
                    break;
                }
            }
        }

        BM_VecSimGeneral::MeasureRecall(hnsw_results, filtered_res, correct);

        VecSimQueryReply_Free(hnsw_results);
        VecSimQueryReply_Free(filtered_res);
        VecSimQueryReply_Free(gt_results);
        st.ResumeTiming();
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
    if (stats) {
        size_t io_bytes_after = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
        st.counters["io_bytes_per_query"] = static_cast<double>(io_bytes_after - io_bytes_before) / iter;
    }
}

// Stress test with high deletion ratios (50%, 75%, 90%+)
// st.range(0) = ef_runtime
// st.range(1) = num_deleted (expected to be high percentage of N_VECTORS)
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW_DISK_DeleteLabel_Stress(benchmark::State &st) {
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    size_t iter = 0;
    const size_t ef = st.range(0);
    const size_t num_to_delete = st.range(1);

    // Reload the index to get a fresh copy
    std::string folder_path = BM_VecSimGeneral::AttachRootPath(BM_VecSimGeneral::hnsw_index_file);
    INDICES[INDEX_HNSW_DISK] = IndexPtr(HNSWDiskFactory::NewIndex(folder_path));
    auto hnsw_index = GET_INDEX(INDEX_HNSW_DISK);
    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(hnsw_index);

    // Get pseudo-random unique labels using stratified sampling
    std::vector<labelType> deleted_labels;
    std::mt19937 rng(42); // Fixed seed for determinism
    for (size_t i = 0; i < num_to_delete && i < N_VECTORS; i++) {
        size_t stratum_start = (i * N_VECTORS) / num_to_delete;
        size_t stratum_end = ((i + 1) * N_VECTORS) / num_to_delete;
        size_t stratum_size = stratum_end - stratum_start;

        std::uniform_int_distribution<size_t> dist(0, stratum_size - 1);
        labelType label = stratum_start + dist(rng);
        deleted_labels.push_back(label);
    }

    // Create hash set for O(1) lookup
    std::unordered_set<labelType> deleted_labels_set(deleted_labels.begin(), deleted_labels.end());

    // Measure deletion time
    auto delete_start = std::chrono::high_resolution_clock::now();
    for (const auto &label : deleted_labels) {
        disk_index->deleteVector(label);
    }
    disk_index->flushDeleteBatch();
    auto delete_end = std::chrono::high_resolution_clock::now();
    double delete_time_ms = std::chrono::duration<double, std::milli>(delete_end - delete_start).count();

    // Calculate deletion ratio
    double deletion_ratio = (double)deleted_labels.size() / (double)N_VECTORS * 100.0;

    // Report metrics
    st.counters["num_deleted"] = deleted_labels.size();
    st.counters["deletion_ratio_pct"] = deletion_ratio;
    st.counters["delete_time_ms"] = delete_time_ms;
    st.counters["remaining_vectors"] = N_VECTORS - deleted_labels.size();
    if (deleted_labels.size() > 0) {
        st.counters["delete_time_per_vector_ms"] = delete_time_ms / deleted_labels.size();
    }

    // Get DB statistics before search benchmark
    auto stats = disk_index->getDBStatistics();
    size_t io_bytes_before = 0;
    if (stats) {
        io_bytes_before = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
    }

    std::atomic_int correct = 0;
    size_t k = 10; // Fixed k for stress testing

    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];

        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);
        st.PauseTiming();

        // Get ground truth and filter deleted vectors
        auto gt_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(iter % N_QUERIES, 100);

        auto filtered_res = new VecSimQueryReply(VecSimAllocator::newVecsimAllocator());
        for (const auto &res : gt_results->results) {
            if (deleted_labels_set.find(res.id) == deleted_labels_set.end()) {
                filtered_res->results.emplace_back(res.id, res.score);
                if (filtered_res->results.size() >= k) {
                    break;
                }
            }
        }

        // For stress tests, it's OK if we can't find k results
        if (filtered_res->results.size() > 0) {
            BM_VecSimGeneral::MeasureRecall(hnsw_results, filtered_res, correct);
        }

        VecSimQueryReply_Free(hnsw_results);
        VecSimQueryReply_Free(filtered_res);
        VecSimQueryReply_Free(gt_results);
        st.ResumeTiming();
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
    if (stats) {
        size_t io_bytes_after = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
        st.counters["io_bytes_per_query"] = static_cast<double>(io_bytes_after - io_bytes_before) / iter;
    }
}

template <typename index_type_t>

void BM_VecSimCommon<index_type_t>::TopK_BF(benchmark::State &st, unsigned short index_offset) {
    size_t k = st.range(0);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimIndex_TopKQuery(GET_INDEX(INDEX_BF + index_offset), QUERIES[iter % N_QUERIES].data(),
                              k, nullptr, BY_SCORE);
        iter++;
    }
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW(benchmark::State &st, unsigned short index_offset) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    std::atomic_int correct = 0;
    size_t iter = 0;
    for (auto _ : st) {
        RunTopK_HNSW(st, ef, iter, k, correct, index_offset);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_Tiered(benchmark::State &st, unsigned short index_offset) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    std::atomic_int correct = 0;
    std::atomic_int iter = 0;
    auto tiered_index =
        dynamic_cast<TieredHNSWIndex<data_t, dist_t> *>(GET_INDEX(INDEX_TIERED_HNSW));
    size_t total_iters = 50;
    VecSimQueryReply *all_results[total_iters];

    auto parallel_knn_search = [](AsyncJob *job) {
        auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = search_job->ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        size_t cur_iter = search_job->iter;
        auto hnsw_results = VecSimIndex_TopKQuery(GET_INDEX(INDEX_TIERED_HNSW),
                                                  QUERIES[cur_iter % N_QUERIES].data(),
                                                  search_job->k, &query_params, BY_SCORE);
        search_job->all_results[cur_iter] = hnsw_results;
        delete job;
    };

    for (auto _ : st) {
        auto search_job = new (tiered_index->getAllocator())
            tieredIndexMock::SearchJobMock(tiered_index->getAllocator(), parallel_knn_search,
                                           tiered_index, k, ef, iter++, all_results);
        tiered_index->submitSingleJob(search_job);
        if (iter == total_iters) {
            BM_VecSimGeneral::mock_thread_pool->thread_pool_wait();
        }
    }

    // Measure recall
    for (iter = 0; iter < total_iters; iter++) {
        auto bf_results =
            VecSimIndex_TopKQuery(GET_INDEX(INDEX_BF + index_offset),
                                  QUERIES[iter % N_QUERIES].data(), k, nullptr, BY_SCORE);
        BM_VecSimGeneral::MeasureRecall(all_results[iter], bf_results, correct);

        VecSimQueryReply_Free(bf_results);
        VecSimQueryReply_Free(all_results[iter]);
    }

    st.counters["Recall"] = (float)correct / (float)(k * iter);
    st.counters["num_threads"] = (double)BM_VecSimGeneral::mock_thread_pool->thread_pool_size;
}

#define REGISTER_TopK_BF(BM_CLASS, BM_FUNC)                                                        \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Arg(10)                                                                                  \
        ->Arg(100)                                                                                 \
        ->Arg(500)                                                                                 \
        ->ArgName("k")                                                                             \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

// {ef_runtime, k} (recall that always ef_runtime >= k)
#define REGISTER_TopK_HNSW(BM_CLASS, BM_FUNC)                                                      \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({10, 10})                                                                           \
        ->Args({200, 10})                                                                          \
        ->Args({100, 100})                                                                         \
        ->Args({200, 100})                                                                         \
        ->Args({500, 500})                                                                         \
        ->ArgNames({"ef_runtime", "k"})                                                            \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

// {ef_runtime, k} (recall that always ef_runtime >= k)
#define REGISTER_TopK_HNSW_DISK(BM_CLASS, BM_FUNC)                                                 \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({10, 10})                                                                           \
        ->Args({100, 10})                                                                          \
        ->Args({100, 50})                                                                         \
        ->Args({200, 50})                                                                         \
        ->Args({100, 100})                                                                         \
        ->Args({200, 100})                                                                         \
        ->Args({500, 100})                                                                         \
        ->ArgNames({"ef_runtime", "k"})                                                            \
        ->Iterations(1000)                                                                           \
        ->Unit(benchmark::kMillisecond)

#define REGISTER_TopK_HNSW_DISK_PARALLEL(BM_CLASS, BM_FUNC)                                        \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({10, 10, 10})                                                                       \
        ->Args({10, 10, 20})                                                                       \
        ->Args({10, 10, 30})                                                                       \
        ->Args({200, 10, 10})                                                                      \
        ->Args({200, 10, 20})                                                                      \
        ->Args({200, 10, 30})                                                                      \
        ->Args({100, 100, 10})                                                                     \
        ->Args({100, 100, 20})                                                                     \
        ->Args({100, 100, 30})                                                                     \
        ->Args({200, 100, 10})                                                                     \
        ->Args({200, 100, 20})                                                                     \
        ->Args({200, 100, 30})                                                                     \
        ->ArgNames({"ef_runtime", "k", "concurrency"})                                             \
        ->Iterations(1)                                                                            \
        ->Unit(benchmark::kMillisecond)

// {ef_runtime, k, concurrency, batch_threshold}
// Test parallel search with concurrent background inserts (single writer thread)
#define REGISTER_TopK_HNSW_DISK_PARALLEL_WITH_INSERTS(BM_CLASS, BM_FUNC)                           \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({200, 10, 10, 10})                                                                  \
        ->Args({200, 10, 20, 10})                                                                  \
        ->Args({200, 10, 30, 10})                                                                  \
        ->Args({200, 10, 10, 100})                                                                 \
        ->Args({200, 10, 20, 100})                                                                 \
        ->Args({200, 10, 30, 100})                                                                 \
        ->Args({200, 10, 10, 1000})                                                                \
        ->Args({200, 10, 20, 1000})                                                                \
        ->Args({200, 10, 30, 1000})                                                                \
        ->ArgNames({"ef_runtime", "k", "concurrency", "batch_threshold"})                          \
        ->Iterations(1)                                                                            \
        ->Unit(benchmark::kMillisecond)

// {ef_runtime, k, num_marked_deleted}
// Test the performance impact of marked deleted vectors
#define REGISTER_TopK_HNSW_DISK_MarkDeleted(BM_CLASS, BM_FUNC)                                    \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({10, 10, 1000})                                                                      \
        ->Args({10, 10, 10000})                                                                   \
        ->Args({10, 10, 25000})                                                                   \
        ->Args({200, 50, 1000})                                                                   \
        ->Args({200, 50, 10000})                                                                   \
        ->Args({200, 50, 25000})                                                                   \
        ->ArgNames({"ef_runtime", "k", "num_marked_deleted"})                                      \
        ->Iterations(100)                                                                           \
        ->Unit(benchmark::kMillisecond)

// {ef_runtime, k, num_deleted}
// Test the performance after fully deleting vectors (with graph repair)
#define REGISTER_TopK_HNSW_DISK_DeleteLabel(BM_CLASS, BM_FUNC)                                    \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({10, 10, 1000})                                                                      \
        ->Args({10, 10, 10000})                                                                   \
        ->Args({10, 10, 25000})                                                                   \
        ->Args({200, 50, 1000})                                                                   \
        ->Args({200, 50, 10000})                                                                   \
        ->Args({200, 50, 25000})                                                                   \
        ->ArgNames({"ef_runtime", "k", "num_deleted"})                                             \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

// {ef_runtime, k, num_deleted}
// Same as DeleteLabel but protects ground truth vectors from deletion for stable recall comparison
#define REGISTER_TopK_HNSW_DISK_DeleteLabel_ProtectGT(BM_CLASS, BM_FUNC)                          \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({10, 10, 1000})                                                                      \
        ->Args({10, 10, 10000})                                                                   \
        ->Args({10, 10, 25000})                                                                   \
        ->Args({200, 50, 1000})                                                                   \
        ->Args({200, 50, 10000})                                                                   \
        ->Args({200, 50, 25000})                                                                   \
        ->ArgNames({"ef_runtime", "k", "num_deleted"})                                             \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

// {ef_runtime, k} (recall that always ef_runtime >= k)
#define REGISTER_TopK_Tiered(BM_CLASS, BM_FUNC)                                                    \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({10, 10})                                                                           \
        ->Args({200, 10})                                                                          \
        ->Args({100, 100})                                                                         \
        ->Args({200, 100})                                                                         \
        ->Args({500, 500})                                                                         \
        ->ArgNames({"ef_runtime", "k"})                                                            \
        ->Iterations(50)                                                                           \
        ->Unit(benchmark::kMillisecond)

// Registration macro for batch size testing
#define REGISTER_TopK_HNSW_DISK_DeleteLabel_BatchSize(BM_CLASS, BM_FUNC)                          \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({100, 5000, 1})     /* ef=100, 5K deletions, batch_size=1 */                      \
        ->Args({100, 5000, 5})     /* batch_size=5 */                                             \
        ->Args({100, 5000, 50})    /* batch_size=50 */                                            \
        ->Args({100, 5000, 500})   /* batch_size=500 */                                           \
        ->Args({100, 5000, 1000})  /* batch_size=1000 */                                          \
        ->Args({50, 2000, 1})      /* Lower ef, fewer deletions */                                \
        ->Args({50, 2000, 10})                                                                     \
        ->Args({50, 2000, 100})                                                                    \
        ->Args({200, 10000, 1})    /* Higher ef, more deletions */                                \
        ->Args({200, 10000, 50})                                                                   \
        ->Args({200, 10000, 1000}) /* Large batch */                                              \
        ->ArgNames({"ef_runtime", "num_deleted", "batch_size"})                                    \
        ->Unit(benchmark::kMicrosecond);

// Registration macro for stress testing
#define REGISTER_TopK_HNSW_DISK_DeleteLabel_Stress(BM_CLASS, BM_FUNC)                             \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({50, 12500})   /* 50% deletion ratio */                                            \
        ->Args({100, 12500})  /* 50% deletion, higher ef */                                       \
        ->Args({200, 12500})  /* 50% deletion, max ef */                                          \
        ->Args({50, 18750})   /* 75% deletion ratio */                                            \
        ->Args({100, 18750})  /* 75% deletion, higher ef */                                       \
        ->Args({50, 22500})   /* 90% deletion ratio (extreme stress) */                           \
        ->Args({100, 22500})  /* 90% deletion, higher ef */                                       \
        ->Args({25, 24000})   /* 96% deletion (maximum stress) */                                 \
        ->ArgNames({"ef_runtime", "num_deleted"})                                                  \
        ->Unit(benchmark::kMicrosecond)                                                            \
        ->Iterations(3);      /* Fewer iterations for stress tests */

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
    static void TopK_Tiered(benchmark::State &st, unsigned short index_offset = 0);

    // Benchmark TopK performance with marked deleted vectors
    static void TopK_HNSW_DISK_MarkDeleted(benchmark::State &st);

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
    auto index = GET_INDEX(index_type);

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["db_disk"] = (double)VecSimIndex_StatsInfo(index).db_disk;
    st.counters["db_memory"] = (double)VecSimIndex_StatsInfo(index).db_memory;
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::Memory(benchmark::State &st, IndexTypeIndex index_type) {
    auto index = GET_INDEX(index_type);
    index->fitMemory();

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] = (double)VecSimIndex_StatsInfo(index).memory;
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
    auto db_stats = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(hnsw_index)->getDBStatistics();
    size_t byte_reads = 0;
    if (db_stats) {
        byte_reads = db_stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
    }

    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];
        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);
        st.PauseTiming();
        auto bf_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(iter % N_QUERIES, k);
        BM_VecSimGeneral::MeasureRecall(hnsw_results, bf_results, correct);
        VecSimQueryReply_Free(bf_results);
        VecSimQueryReply_Free(hnsw_results);
        st.ResumeTiming();
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);

    if (db_stats) {
        byte_reads = db_stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO) - byte_reads;
        st.counters["byte_reads"] = static_cast<double>(byte_reads) / iter;
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

// Benchmark TopK performance with marked deleted vectors
// st.range(0) = ef_runtime
// st.range(1) = k
// st.range(2) = number of vectors to mark as deleted
template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW_DISK_MarkDeleted(benchmark::State &st) {
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    size_t ef = st.range(0);
    size_t k = st.range(1);
    size_t num_to_delete = st.range(2);
    std::atomic_int correct = 0;
    size_t iter = 0;
    auto hnsw_index = GET_INDEX(INDEX_HNSW_DISK);
    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(hnsw_index);

    // Mark vectors as deleted before starting the benchmark
    // We mark vectors starting from label 0, ensuring they exist in the index
    // The vectors to delete come from the test_queries_file that were added to the index
    std::vector<labelType> deleted_labels;
    std::vector<labelType> newly_marked_labels;
    size_t initial_marked_deleted = disk_index->getNumMarkedDeleted();

    for (size_t i = 0; i < num_to_delete && i < N_VECTORS; i++) {
        bool was_already_deleted = disk_index->isMarkedDeleted(i);
        deleted_labels.push_back(i);
        disk_index->markDelete(i);
        if (!was_already_deleted) {
            newly_marked_labels.push_back(i);
        }
    }

    size_t actual_marked = disk_index->getNumMarkedDeleted() - initial_marked_deleted;
    std::cout << "Marked " << actual_marked << " NEW vectors as deleted (total marked: "
              << disk_index->getNumMarkedDeleted() << ")" << std::endl;
    st.counters["num_marked_deleted"] = actual_marked;

    // Get DB statistics before benchmark
    auto stats = disk_index->getDBStatistics();
    size_t io_bytes_before = 0;
    if (stats) {
        io_bytes_before = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
    }

    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];
        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);
        st.PauseTiming();
        auto bf_results = BM_VecSimIndex<fp32_index_t>::TopKGroundTruth(iter % N_QUERIES, k);
        BM_VecSimGeneral::MeasureRecall(hnsw_results, bf_results, correct);
        VecSimQueryReply_Free(bf_results);
        VecSimQueryReply_Free(hnsw_results);
        st.ResumeTiming();
        iter++;
    }

    st.counters["Recall"] = (float)correct / (float)(k * iter);
    if (stats) {
        size_t io_bytes_after = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
        st.counters["io_bytes_per_query"] = static_cast<double>(io_bytes_after - io_bytes_before) / iter;
    }

    // Cleanup: Unmark all deleted vectors to restore index state for next benchmark run
    // This ensures each benchmark configuration starts with a clean slate
    // for (const auto &label : deleted_labels) {
    //     // Get internal ID for this label
    //     auto it = disk_index->labelToIdMap.find(label);
    //     if (it != disk_index->labelToIdMap.end()) {
    //         idType internalId = it->second;
    //         // Unmark the DELETE_MARK flag
    //         disk_index->unmarkAs<DELETE_MARK>(internalId);
    //         disk_index->numMarkedDeleted--;
    //     }
    // }
    // std::cout << "Cleaned up: unmarked " << deleted_labels.size() << " vectors" << std::endl;
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
        ->Args({200, 10})                                                                          \
        ->Args({100, 100})                                                                         \
        ->Args({200, 100})                                                                         \
        ->ArgNames({"ef_runtime", "k"})                                                            \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

// {ef_runtime, k, num_marked_deleted}
// Test the performance impact of marked deleted vectors
#define REGISTER_TopK_HNSW_DISK_MarkDeleted(BM_CLASS, BM_FUNC)                                    \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({200, 10, 1})                                                                       \
        ->Args({200, 10, 1000})                                                                    \
        ->Args({200, 10, 50000})                                                                   \
        ->ArgNames({"ef_runtime", "k", "num_marked_deleted"})                                      \
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

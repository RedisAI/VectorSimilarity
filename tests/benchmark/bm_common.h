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
        cache_misses = db_stats->getTickerCount(rocksdb::Tickers::BLOCK_CACHE_MISS) - cache_misses;
        st.counters["cache_misses_per_query"] = static_cast<double>(cache_misses) / iter;
    }

    // Output RocksDB memory usage
    uint64_t db_memory = hnsw_disk_index->getDBMemorySize();
    st.counters["rocksdb_memory"] = static_cast<double>(db_memory);
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
    
    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];

        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);
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
    if (stats) {
        size_t io_bytes_after = stats->getTickerCount(rocksdb::Tickers::BYTES_COMPRESSED_TO);
        st.counters["io_bytes_per_query"] = static_cast<double>(io_bytes_after - io_bytes_before) / iter;
    }

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

    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];

        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);
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
    size_t num_iterations = st.iterations();
    std::unordered_set<labelType> gt_labels_set;
    for (size_t q = 0; q < std::min(num_iterations, N_QUERIES); q++) {
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

    for (auto _ : st) {
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        auto &q = QUERIES[iter % N_QUERIES];

        auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index, q.data(), k, &query_params, BY_SCORE);
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

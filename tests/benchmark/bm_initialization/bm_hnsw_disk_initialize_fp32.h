/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once
/**************************************
  Define and register tests for disk-based HNSW index
  NOTE: benchmarks' tests order can affect their results. Please add new benchmarks at the end of
the file.
***************************************/

// Memory benchmarks
// BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, FLAT), fp32_index_t)
// (benchmark::State &st) { Memory(st, INDEX_BF); }
// BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, FLAT))->Iterations(1);


BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, HNSWDisk), fp32_index_t)
(benchmark::State &st) { Memory(st, INDEX_HNSW_DISK); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Memory, HNSWDisk))->Iterations(1);

// Disk benchmarks
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(Disk, HNSWDisk), fp32_index_t)
(benchmark::State &st) { Disk(st, INDEX_HNSW_DISK); }
BENCHMARK_REGISTER_F(BM_VecSimCommon, BM_FUNC_NAME(Disk, HNSWDisk))->Iterations(1);

// AddLabel benchmarks
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_ADD_LABEL, fp32_index_t)
(benchmark::State &st) { AddLabel(st); }
REGISTER_AddLabel(BM_ADD_LABEL, INDEX_HNSW_DISK);

// TopK benchmark (single-threaded)
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(TopK, HNSWDisk), fp32_index_t)
(benchmark::State &st) { TopK_HNSW_DISK(st); }
REGISTER_TopK_HNSW_DISK(BM_VecSimCommon, BM_FUNC_NAME(TopK, HNSWDisk));

// TopK benchmark (parallel)
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(TopKParallel, HNSWDisk), fp32_index_t)
(benchmark::State &st) { TopK_HNSW_DISK_Parallel(st); }
REGISTER_TopK_HNSW_DISK_PARALLEL(BM_VecSimCommon, BM_FUNC_NAME(TopKParallel, HNSWDisk));

// TopK benchmark with marked deleted vectors
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(TopK_MarkDeleted, HNSWDisk), fp32_index_t)
(benchmark::State &st) { TopK_HNSW_DISK_MarkDeleted(st); }
REGISTER_TopK_HNSW_DISK_MarkDeleted(BM_VecSimCommon, BM_FUNC_NAME(TopK_MarkDeleted, HNSWDisk));

// TopK benchmark after deleting vectors (with graph repair)
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(TopK_DeleteLabel, HNSWDisk), fp32_index_t)
(benchmark::State &st) { TopK_HNSW_DISK_DeleteLabel(st); }
REGISTER_TopK_HNSW_DISK_DeleteLabel(BM_VecSimCommon, BM_FUNC_NAME(TopK_DeleteLabel, HNSWDisk));

// TopK benchmark after deleting vectors (with graph repair), protecting GT vectors for stable recall
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimCommon, BM_FUNC_NAME(TopK_DeleteLabel_ProtectGT, HNSWDisk), fp32_index_t)
(benchmark::State &st) { TopK_HNSW_DISK_DeleteLabel_ProtectGT(st); }
REGISTER_TopK_HNSW_DISK_DeleteLabel_ProtectGT(BM_VecSimCommon, BM_FUNC_NAME(TopK_DeleteLabel_ProtectGT, HNSWDisk));
// Special disk-based HNSW benchmarks for batch processing with multi-threaded async ingest
// Args: {INDEX_HNSW_DISK, thread_count}
// This benchmark reloads the disk index for each run since async operations modify the index
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_ADD_LABEL_ASYNC_DISK, fp32_index_t)
(benchmark::State &st) {
    // Reload the disk index fresh for each benchmark run
    std::string folder_path = AttachRootPath(hnsw_index_file);

    // Clean up existing thread pool and index
    if (BM_VecSimGeneral::mock_thread_pool) {
        BM_VecSimGeneral::mock_thread_pool->thread_pool_join();
        delete BM_VecSimGeneral::mock_thread_pool;
        BM_VecSimGeneral::mock_thread_pool = nullptr;
    }
    indices[INDEX_HNSW_DISK] = IndexPtr(nullptr);

    // Reload the index from the checkpoint
    indices[INDEX_HNSW_DISK] = IndexPtr(HNSWDiskFactory::NewIndex(folder_path));

    // Create new mock thread pool
    BM_VecSimGeneral::mock_thread_pool = new tieredIndexMock();
    auto &mock_thread_pool = *BM_VecSimGeneral::mock_thread_pool;
    mock_thread_pool.ctx->index_strong_ref = indices[INDEX_HNSW_DISK].get_shared();

    // Set up job queue for async operations on the disk index
    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(indices[INDEX_HNSW_DISK].get());
    if (disk_index) {
        disk_index->setJobQueue(&mock_thread_pool.jobQ, mock_thread_pool.ctx,
                                tieredIndexMock::submit_callback);
    }

    // Configure thread pool size from benchmark argument and start threads
    size_t thread_count = st.range(1);
    mock_thread_pool.thread_pool_size = thread_count;
    mock_thread_pool.init_threads();

    // Get initial state
    auto *index = indices[INDEX_HNSW_DISK].get();
    size_t initial_index_size = VecSimIndex_IndexSize(index);

    // Measure the AddLabel_AsyncIngest benchmark
    auto start_time = std::chrono::high_resolution_clock::now();
    AddLabel_AsyncIngest(st);
    // Wait for all jobs to complete before measuring end time
    mock_thread_pool.thread_pool_wait();
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate stats
    size_t final_index_size = VecSimIndex_IndexSize(index);
    size_t vectors_added = final_index_size - initial_index_size;
    double total_time_ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
    double avg_time_per_label_ns = vectors_added > 0 ? total_time_ns / vectors_added : 0;

    // Add custom counters
    st.counters["vectors_added"] = vectors_added;
    st.counters["total_time_ns"] = total_time_ns;
    st.counters["avg_ns_per_label"] = avg_time_per_label_ns;
}
BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_ADD_LABEL_ASYNC_DISK)
    ->Unit(benchmark::kNanosecond)
    ->Iterations(10000)
    ->Args({INDEX_HNSW_DISK, 1})
    ->Args({INDEX_HNSW_DISK, 4})
    ->Args({INDEX_HNSW_DISK, 8})
    ->ArgNames({"IndexType", "Threads"});

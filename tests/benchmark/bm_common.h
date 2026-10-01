/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include "bm_vecsim_index.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

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
                             std::atomic_int &correct, unsigned short index_offset = 0,
                             IndexTypeIndex index_type = INDEX_HNSW);

    // Search for the K closest vectors to the query in the index. K is defined in the
    // test registration (initialization file).
    static void TopK_BF(benchmark::State &st, unsigned short index_offset = 0);
    // Run TopK using both HNSW and flat index and calculate the recall of the HNSW algorithm
    // with respect to the results returned by the flat index.
    static void TopK_HNSW(benchmark::State &st, unsigned short index_offset = 0,
                          IndexTypeIndex index_type = INDEX_HNSW);
    struct FP32GroundTruth {
        std::vector<std::uint64_t> ids;
        std::uint64_t boundary_tied_queries;
    };
    static void TopK_HNSW_SQ8_Recall1000(benchmark::State &st);
    static FP32GroundTruth LoadOrBuildFP32GroundTruth(const std::string &cache_path);
    static void TopK_Tiered(benchmark::State &st, unsigned short index_offset = 0,
                            IndexTypeIndex index_type = INDEX_TIERED_HNSW);

    // Does nothing but returning the index memory.
    static void Memory(benchmark::State &st, IndexTypeIndex index_type);
};

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter,
                                                 size_t k, std::atomic_int &correct,
                                                 unsigned short index_offset,
                                                 IndexTypeIndex index_type) {
    HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = ef};
    auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
    auto hnsw_results =
        VecSimIndex_TopKQuery(GET_INDEX(index_type + index_offset),
                              QUERIES[iter % N_QUERIES].data(), k, &query_params, BY_SCORE);
    st.PauseTiming();

    // Measure recall:
    auto bf_results = VecSimIndex_TopKQuery(GET_INDEX(INDEX_BF + index_offset),
                                            QUERIES[iter % N_QUERIES].data(), k, nullptr, BY_SCORE);

    BM_VecSimGeneral::MeasureRecall(hnsw_results, bf_results, correct);

    VecSimQueryReply_Free(bf_results);
    VecSimQueryReply_Free(hnsw_results);
    st.ResumeTiming();
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::Memory(benchmark::State &st, IndexTypeIndex index_type) {
    auto index = GET_INDEX(index_type);
    index->fitMemory();

    for (auto _ : st) {
        // Do nothing...
    }
    st.counters["memory"] =
        benchmark::Counter((double)VecSimIndex_StatsInfo(index).memory,
                           benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1024);
}

// TopK search BM

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
void BM_VecSimCommon<index_type_t>::TopK_HNSW(benchmark::State &st, unsigned short index_offset,
                                              IndexTypeIndex index_type) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    std::atomic_int correct = 0;
    size_t iter = 0;
    for (auto _ : st) {
        RunTopK_HNSW(st, ef, iter, k, correct, index_offset, index_type);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(k * iter);
}

template <typename index_type_t>
typename BM_VecSimCommon<index_type_t>::FP32GroundTruth
BM_VecSimCommon<index_type_t>::LoadOrBuildFP32GroundTruth(const std::string &cache_path) {
    constexpr std::uint64_t magic = 0x4d31393136394632ULL;
    constexpr size_t query_count = 1000;
    constexpr size_t k = 100;
    constexpr size_t header_size = 6;
    if (N_QUERIES < query_count || QUERIES.size() < query_count) {
        throw std::runtime_error("FP32 recall requires at least 1000 loaded queries");
    }
    auto *hnsw = BM_VecSimIndex<index_type_t>::template get_typed_index<
        HNSWIndex<data_t, dist_t>>(INDEX_HNSW);
    if (!hnsw || hnsw->indexSize() != BM_VecSimGeneral::n_vectors ||
        hnsw->indexLabelCount() < k + 1) {
        throw std::runtime_error("FP32 recall requires the complete loaded FP16 HNSW source");
    }

    const auto checksum = [](const std::vector<std::uint64_t> &ids,
                             std::uint64_t boundary_tied_queries) {
        std::uint64_t hash = 14695981039346656037ULL;
        hash ^= boundary_tied_queries;
        hash *= 1099511628211ULL;
        for (std::uint64_t id : ids) {
            hash ^= id;
            hash *= 1099511628211ULL;
        }
        return hash;
    };
    const auto validate_rows = [=](const std::vector<std::uint64_t> &ids) {
        for (size_t q = 0; q < query_count; ++q) {
            const auto first = ids.begin() + q * k;
            if (std::unordered_set<std::uint64_t>(first, first + k).size() != k) {
                throw std::runtime_error("FP32 ground-truth cache has fewer than 100 unique labels "
                                         "for query " +
                                         std::to_string(q));
            }
        }
    };
    std::array<std::uint64_t, header_size> header{magic, DIM, query_count, k, 0, 0};
    std::vector<std::uint64_t> ids(query_count * k);
    std::uint64_t boundary_tied_queries = 0;
    std::error_code file_error;
    const bool exists = std::filesystem::exists(cache_path, file_error);
    if (file_error) {
        throw std::runtime_error("Cannot inspect FP32 ground-truth cache: " + file_error.message());
    }
    if (exists) {
        std::ifstream input(cache_path, std::ios::binary);
        std::array<std::uint64_t, header_size> loaded_header{};
        if (!input.read(reinterpret_cast<char *>(loaded_header.data()), sizeof(loaded_header)) ||
            !std::equal(header.begin(), header.begin() + 4, loaded_header.begin()) ||
            !input.read(reinterpret_cast<char *>(ids.data()), ids.size() * sizeof(ids[0])) ||
            input.peek() != std::char_traits<char>::eof() || input.bad() ||
            loaded_header[4] > query_count || loaded_header[5] != checksum(ids, loaded_header[4])) {
            throw std::runtime_error("FP32 ground-truth cache is corrupt or mismatched: " +
                                     cache_path);
        }
        validate_rows(ids);
        return {std::move(ids), loaded_header[4]};
    }

    BFParams params = {.type = VecSimType_FLOAT32,
                       .dim = DIM,
                       .metric = VecSimMetric_IP,
                       .multi = BM_VecSimGeneral::is_multi,
                       .blockSize = BM_VecSimGeneral::block_size};
    IndexPtr bf(BM_VecSimGeneral::CreateNewIndex(params));
    if (!bf.get()) {
        throw std::runtime_error("Cannot create FP32 BF ground-truth index");
    }
    std::vector<float> widened(DIM);
    for (size_t i = 0; i < BM_VecSimGeneral::n_vectors; ++i) {
        const auto *stored = reinterpret_cast<const data_t *>(hnsw->getDataByInternalId(i));
        for (size_t j = 0; j < DIM; ++j) {
            widened[j] = vecsim_types::FP16_to_FP32(stored[j]);
            if (!std::isfinite(widened[j])) {
                throw std::runtime_error("FP16 source contains a nonfinite value");
            }
        }
        VecSimIndex_AddVector(bf, widened.data(), hnsw->getExternalLabel(i));
    }
    if (VecSimIndex_IndexSize(bf) != hnsw->indexSize() ||
        bf.get()->indexLabelCount() != hnsw->indexLabelCount()) {
        throw std::runtime_error("FP32 BF source differs in vector or label count");
    }

    std::vector<float> fp32_query(DIM);
    for (size_t q = 0; q < query_count; ++q) {
        for (size_t j = 0; j < DIM; ++j) {
            fp32_query[j] = vecsim_types::FP16_to_FP32(QUERIES[q][j]);
            if (!std::isfinite(fp32_query[j])) {
                throw std::runtime_error("FP16 query contains a nonfinite value");
            }
        }
        auto *reply = VecSimIndex_TopKQuery(bf, fp32_query.data(), k + 1, nullptr, BY_SCORE);
        if (!reply || VecSimQueryReply_GetCode(reply) != VecSim_QueryReply_OK ||
            VecSimQueryReply_Len(reply) != k + 1) {
            if (reply)
                VecSimQueryReply_Free(reply);
            throw std::runtime_error(
                "Exhaustive FP32 BF returned fewer than 101 labels for query " + std::to_string(q));
        }
        auto *iterator = VecSimQueryReply_GetIterator(reply);
        size_t rank = 0;
        double score_at_k = 0.0, score_after_k = 0.0;
        bool nonfinite_score = false;
        while (VecSimQueryReply_IteratorHasNext(iterator)) {
            const auto *item = VecSimQueryReply_IteratorNext(iterator);
            const double score = VecSimQueryResult_GetScore(item);
            if (!std::isfinite(score)) {
                nonfinite_score = true;
                break;
            }
            if (rank < k) {
                ids[q * k + rank] = static_cast<std::uint64_t>(VecSimQueryResult_GetId(item));
            }
            if (rank == k - 1)
                score_at_k = score;
            if (rank == k)
                score_after_k = score;
            ++rank;
        }
        VecSimQueryReply_IteratorFree(iterator);
        VecSimQueryReply_Free(reply);
        if (nonfinite_score || rank != k + 1) {
            throw std::runtime_error("Exhaustive FP32 BF returned invalid scores or width for "
                                     "query " +
                                     std::to_string(q));
        }
        boundary_tied_queries += score_at_k == score_after_k;
    }
    validate_rows(ids);
    header[4] = boundary_tied_queries;
    header[5] = checksum(ids, boundary_tied_queries);
    std::ofstream output(cache_path, std::ios::binary | std::ios::trunc);
    if (!output || !output.write(reinterpret_cast<const char *>(header.data()), sizeof(header)) ||
        !output.write(reinterpret_cast<const char *>(ids.data()), ids.size() * sizeof(ids[0])) ||
        !output.flush()) {
        throw std::runtime_error("Cannot write FP32 ground-truth cache: " + cache_path);
    }
    return {std::move(ids), boundary_tied_queries};
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_HNSW_SQ8_Recall1000(benchmark::State &st) {
    constexpr size_t query_count = 1000;
    constexpr size_t k = 100;
    const char *cache_path = std::getenv("MOD19169_GT_CACHE");
    if (!cache_path || !*cache_path) {
        st.SkipWithError("MOD19169_GT_CACHE must name the FP32 recall cache file");
        return;
    }
    FP32GroundTruth ground_truth;
    try {
        ground_truth = LoadOrBuildFP32GroundTruth(cache_path);
    } catch (const std::exception &error) {
        st.SkipWithError(error.what());
        return;
    }
    auto *index = GET_INDEX(INDEX_HNSW_SQ8);
    if (!index) {
        st.SkipWithError("SQ8 HNSW index is unavailable");
        return;
    }
    HNSWRuntimeParams runtime_params = {.efRuntime = static_cast<size_t>(st.range(0))};
    auto query_params = BM_VecSimGeneral::CreateQueryParams(runtime_params);
    size_t correct = 0;
    size_t iter = 0;
    size_t q = 0;
    const void *query = QUERIES[q].data();
    for (auto _ : st) {
        auto *reply = VecSimIndex_TopKQuery(index, query, k, &query_params, BY_SCORE);
        st.PauseTiming();
        if (!reply || VecSimQueryReply_GetCode(reply) != VecSim_QueryReply_OK ||
            VecSimQueryReply_Len(reply) != k) {
            if (reply)
                VecSimQueryReply_Free(reply);
            st.ResumeTiming();
            st.SkipWithError("SQ8 HNSW did not return 100 successful labels");
            return;
        }
        const auto first = ground_truth.ids.begin() + q * k;
        std::unordered_set<std::uint64_t> exact(first, first + k);
        auto *iterator = VecSimQueryReply_GetIterator(reply);
        bool nonfinite_score = false;
        while (VecSimQueryReply_IteratorHasNext(iterator)) {
            const auto *item = VecSimQueryReply_IteratorNext(iterator);
            if (!std::isfinite(VecSimQueryResult_GetScore(item))) {
                nonfinite_score = true;
                break;
            }
            correct += exact.erase(static_cast<std::uint64_t>(VecSimQueryResult_GetId(item)));
        }
        VecSimQueryReply_IteratorFree(iterator);
        VecSimQueryReply_Free(reply);
        if (nonfinite_score) {
            st.ResumeTiming();
            st.SkipWithError("SQ8 HNSW returned a nonfinite score");
            return;
        }
        ++iter;
        q = iter % query_count;
        query = QUERIES[q].data();
        st.ResumeTiming();
    }
    st.counters["Recall_vs_FP32_BF_over_FP16_values"] = static_cast<double>(correct) / (k * iter);
    st.counters["FP32_BF_boundary_tied_queries"] = ground_truth.boundary_tied_queries;
    st.SetLabel("reference=FP32_BF_over_FP16_values");
}

template <typename index_type_t>
void BM_VecSimCommon<index_type_t>::TopK_Tiered(benchmark::State &st, unsigned short index_offset,
                                                IndexTypeIndex index_type) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    std::atomic_int correct = 0;
    std::atomic_int iter = 0;
    auto tiered_index = dynamic_cast<TieredHNSWIndex<data_t, dist_t> *>(GET_INDEX(index_type));
    constexpr size_t total_iters = BM_VecSimGeneral::tiered_topk_iterations;
    VecSimQueryReply *all_results[total_iters];

    auto parallel_knn_search = [](AsyncJob *job) {
        auto *search_job = reinterpret_cast<tieredIndexMock::SearchJobMock *>(job);
        HNSWRuntimeParams hnswRuntimeParams = {.efRuntime = search_job->ef};
        auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);
        size_t cur_iter = search_job->iter;
        auto hnsw_results =
            VecSimIndex_TopKQuery(search_job->index, QUERIES[cur_iter % N_QUERIES].data(),
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
#define REGISTER_TopK_Tiered(BM_CLASS, BM_FUNC)                                                    \
    BENCHMARK_REGISTER_F(BM_CLASS, BM_FUNC)                                                        \
        ->Args({10, 10})                                                                           \
        ->Args({200, 10})                                                                          \
        ->Args({100, 100})                                                                         \
        ->Args({200, 100})                                                                         \
        ->Args({500, 500})                                                                         \
        ->ArgNames({"ef_runtime", "k"})                                                            \
        ->Iterations(BM_VecSimGeneral::tiered_topk_iterations)                                     \
        ->Unit(benchmark::kMillisecond)

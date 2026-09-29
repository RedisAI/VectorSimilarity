/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "bm_vecsim_general.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/index_factories/tiered_factory.h"

#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename index_type_t>
class BM_VecSimSQ8Training : public BM_VecSimGeneral {
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    static std::vector<std::vector<data_t>> test_vectors;
    VecSimWriteMode original_mode = VecSim_WriteInPlace;

    TieredHNSWIndex<data_t, dist_t> *CreateIndex(tieredIndexMock &pool, size_t threshold) {
        HNSWParams hnsw_params = {.type = index_type_t::get_index_type(),
                                  .dim = dim,
                                  .metric = VecSimMetric_Cosine,
                                  .multi = false,
                                  .M = M,
                                  .efConstruction = EF_C,
                                  .quantType = VecSimQuant_SQ8};
        VecSimParams params = CreateParams(hnsw_params);
        TieredIndexParams tiered_params = {
            .jobQueue = &pool.jobQ,
            .jobQueueCtx = pool.ctx,
            .submitCb = tieredIndexMock::submit_callback,
            .flatBufferLimit = block_size,
            .primaryIndexParams = &params,
            .specificParams = {
                TieredHNSWParams{.swapJobThreshold = 0, .QuantNormalizationSetSize = threshold}}};
        auto *index = reinterpret_cast<TieredHNSWIndex<data_t, dist_t> *>(
            TieredFactory::NewIndex(&tiered_params));
        if (index) {
            pool.ctx->index_strong_ref.reset(index);
        } else {
            // An empty mock context cannot be destroyed by tieredIndexMock.
            pool.reset_ctx();
        }
        return index;
    }

    static bool CheckSizes(benchmark::State &st, VecSimIndex *index, size_t total, size_t frontend,
                           size_t backend) {
        const auto info = VecSimIndex_DebugInfo(index);
        if (info.commonInfo.indexSize != total ||
            info.tieredInfo.frontendCommonInfo.indexSize != frontend ||
            info.tieredInfo.backendCommonInfo.indexSize != backend) {
            st.SkipWithError("Unexpected SQ8 tiered index sizes");
            return false;
        }
        return true;
    }

    static bool AddRange(benchmark::State &st, VecSimIndex *index, size_t first, size_t last) {
        for (size_t i = first; i < last; ++i) {
            if (VecSimIndex_AddVector(index, test_vectors[i].data(), i) != 1) {
                st.SkipWithError("SQ8 vector insertion failed");
                return false;
            }
        }
        return true;
    }

    template <bool async>
    void RunTrain(benchmark::State &st) {
        const size_t threshold = st.range(0);
        const unsigned int threads = async ? static_cast<unsigned int>(st.range(1)) : 0;
        if (threshold == 0 || threshold > test_vectors.size()) {
            st.SkipWithError("Insufficient vectors for SQ8 training threshold");
            return;
        }
        if constexpr (async) {
            if (threads > std::thread::hardware_concurrency()) {
                st.SkipWithError("Requested SQ8 worker threads are unavailable");
                return;
            }
        }
        VecSim_SetWriteMode(async ? VecSim_WriteAsync : VecSim_WriteInPlace);
        for (auto _ : st) {
            st.PauseTiming();
            {
                tieredIndexMock pool(threads);
                auto *index = CreateIndex(pool, threshold);
                if (!index) {
                    st.SkipWithError("Failed to create SQ8 tiered index");
                    return;
                }
                if (!AddRange(st, index, 0, threshold - 1) ||
                    !CheckSizes(st, index, threshold - 1, threshold - 1, 0)) {
                    return;
                }
                if constexpr (async) {
                    pool.init_threads();
                }
                st.ResumeTiming();
                const int added =
                    VecSimIndex_AddVector(index, test_vectors[threshold - 1].data(), threshold - 1);
                if constexpr (async) {
                    pool.thread_pool_wait();
                }
                st.PauseTiming();
                if (added != 1 || !CheckSizes(st, index, threshold, 0, threshold)) {
                    if (added != 1) {
                        st.SkipWithError("SQ8 threshold insertion failed");
                    }
                    return;
                }
            }
            st.ResumeTiming();
        }
    }

public:
    BM_VecSimSQ8Training() {
        if (!test_vectors.empty()) {
            return;
        }
        std::ifstream input(AttachRootPath(test_queries_file), std::ios::binary);
        if (!input) {
            throw std::runtime_error("SQ8 training vector file was not found");
        }
        test_vectors.reserve(n_queries);
        for (size_t i = 0; i < n_queries; ++i) {
            std::vector<data_t> vector(dim);
            if (!input.read(reinterpret_cast<char *>(vector.data()), dim * sizeof(data_t))) {
                throw std::runtime_error("SQ8 training vector file is too short");
            }
            test_vectors.push_back(std::move(vector));
        }
    }

    void SetUp(const benchmark::State &st) override {
        BM_VecSimGeneral::SetUp(st);
        original_mode = VecSimIndexInterface::asyncWriteMode;
    }

    void TearDown(const benchmark::State &st) override {
        VecSim_SetWriteMode(original_mode);
        BM_VecSimGeneral::TearDown(st);
    }

    void Train(benchmark::State &st) { RunTrain<false>(st); }
    void TrainAsync(benchmark::State &st) { RunTrain<true>(st); }

    void AddVectorsDuringInitialIngest(benchmark::State &st) {
        const size_t threshold = st.range(0);
        const unsigned int threads = static_cast<unsigned int>(st.range(1));
        constexpr size_t batch_size = 1000;
        if (threshold == 0 || threshold + batch_size > test_vectors.size()) {
            st.SkipWithError("Insufficient vectors for SQ8 initial ingest benchmark");
            return;
        }
        if (threads > std::thread::hardware_concurrency()) {
            st.SkipWithError("Requested SQ8 worker threads are unavailable");
            return;
        }
        VecSim_SetWriteMode(VecSim_WriteAsync);
        for (auto _ : st) {
            st.PauseTiming();
            {
                tieredIndexMock pool(threads);
                auto *index = CreateIndex(pool, threshold);
                if (!index) {
                    st.SkipWithError("Failed to create SQ8 tiered index");
                    return;
                }
                if (!AddRange(st, index, 0, threshold) ||
                    !CheckSizes(st, index, threshold, threshold, 0)) {
                    return;
                }
                pool.init_threads();
                // Workers start on initial insertion jobs; their progress races with timed writes.
                st.ResumeTiming();
                const bool added = AddRange(st, index, threshold, threshold + batch_size);
                st.PauseTiming();
                pool.thread_pool_wait();
                if (!added ||
                    !CheckSizes(st, index, threshold + batch_size, 0, threshold + batch_size)) {
                    return;
                }
            }
            st.ResumeTiming();
        }
    }
};

template <typename index_type_t>
std::vector<std::vector<typename index_type_t::data_t>>
    BM_VecSimSQ8Training<index_type_t>::test_vectors{};

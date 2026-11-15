/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/vec_sim.h"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <type_traits>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using IndexPtr = std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)>;
using ReplyPtr = std::unique_ptr<VecSimQueryReply, decltype(&VecSimQueryReply_Free)>;
using IteratorPtr =
    std::unique_ptr<VecSimQueryReply_Iterator, decltype(&VecSimQueryReply_IteratorFree)>;
constexpr size_t k = 10;

// fbin/ibin have two little-endian uint32 dimensions followed by a dense row-major payload.
static_assert(std::endian::native == std::endian::little);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

template <typename T>
struct Matrix {
    uint32_t rows = 0;
    uint32_t dim = 0;
    std::vector<T> values;
    const T *row(size_t i) const { return values.data() + i * dim; }
};

template <typename T>
Matrix<T> readMatrix(const char *path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(std::string("Cannot open matrix: ") + path);
    }
    const std::streamoff file_size = input.tellg();
    input.seekg(0);
    Matrix<T> matrix;
    input.read(reinterpret_cast<char *>(&matrix.rows), sizeof(matrix.rows));
    input.read(reinterpret_cast<char *>(&matrix.dim), sizeof(matrix.dim));
    const uint64_t count = uint64_t(matrix.rows) * matrix.dim;
    if (!input || !matrix.rows || !matrix.dim || file_size < 8 ||
        count > (std::numeric_limits<uint64_t>::max() - 8) / sizeof(T) ||
        count > matrix.values.max_size() ||
        count > uint64_t(std::numeric_limits<std::streamsize>::max()) / sizeof(T) ||
        uint64_t(file_size) != 8 + count * sizeof(T)) {
        throw std::runtime_error(std::string("Invalid matrix header or file size: ") + path);
    }
    matrix.values.resize(static_cast<size_t>(count));
    input.read(reinterpret_cast<char *>(matrix.values.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) {
        throw std::runtime_error(std::string("Short matrix read: ") + path);
    }
    if constexpr (std::is_same_v<T, float>) {
        // Inspect bits so validation also works when the library is built with fast-math.
        for (float value : matrix.values) {
            if ((std::bit_cast<uint32_t>(value) & 0x7f800000U) == 0x7f800000U) {
                throw std::runtime_error(std::string("Non-finite FP32 matrix component: ") + path);
            }
        }
    }
    return matrix;
}

size_t positiveInteger(std::string_view value) {
    size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        !result) {
        throw std::runtime_error("Expected a positive integer: " + std::string(value));
    }
    return result;
}

std::vector<size_t> parseEfList(std::string_view value) {
    std::vector<size_t> result;
    for (;;) {
        const size_t comma = value.find(',');
        const size_t ef = positiveInteger(value.substr(0, comma));
        if (ef < k || ef > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Each ef must be between 10 and UINT32_MAX");
        }
        result.push_back(ef);
        if (comma == std::string_view::npos) {
            return result;
        }
        value.remove_prefix(comma + 1);
    }
}

uint64_t peakRssBytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
        throw std::runtime_error("getrusage failed");
    }
#ifdef __APPLE__
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024;
#endif
}

double seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

ReplyPtr query(VecSimIndex *index, const float *blob, VecSimQueryParams &params) {
    ReplyPtr reply(VecSimIndex_TopKQuery(index, blob, k, &params, BY_SCORE), VecSimQueryReply_Free);
    if (!reply || VecSimQueryReply_GetCode(reply.get()) != VecSim_QueryReply_OK ||
        VecSimQueryReply_Len(reply.get()) != k) {
        throw std::runtime_error("Query failed or returned fewer than 10 results");
    }
    return reply;
}

size_t countHits(VecSimQueryReply *reply, const uint32_t *truth, size_t base_count) {
    IteratorPtr iterator(VecSimQueryReply_GetIterator(reply), VecSimQueryReply_IteratorFree);
    if (!iterator) {
        throw std::runtime_error("Cannot allocate result iterator");
    }
    std::array<uint32_t, k> seen{};
    size_t results = 0, hits = 0;
    while (VecSimQueryReply_IteratorHasNext(iterator.get())) {
        const int64_t id = VecSimQueryResult_GetId(VecSimQueryReply_IteratorNext(iterator.get()));
        if (id < 0 || uint64_t(id) >= base_count || results == k ||
            std::find(seen.begin(), seen.begin() + results, uint32_t(id)) !=
                seen.begin() + results) {
            throw std::runtime_error("Out-of-range or duplicate result ID");
        }
        seen[results++] = static_cast<uint32_t>(id);
        hits += std::find(truth, truth + k, uint32_t(id)) != truth + k;
    }
    if (results != k) {
        throw std::runtime_error("Result iterator length mismatch");
    }
    return hits;
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 6 && argc != 7) {
            throw std::runtime_error(
                "Usage: pipnn_comparison base.fbin query.fbin truth.ibin pipnn|hnsw "
                "ef[,ef...] [query_limit]");
        }
        const std::string backend = argv[4];
        if (backend != "pipnn" && backend != "hnsw") {
            throw std::runtime_error("Backend must be pipnn or hnsw");
        }
        const auto ef_values = parseEfList(argv[5]);
        const size_t query_limit = argc == 7 ? positiveInteger(argv[6]) : SIZE_MAX;
        auto base = readMatrix<float>(argv[1]);
        if (base.rows < k) {
            throw std::runtime_error("Base must contain at least 10 vectors");
        }
        VecSimParams params{};
        if (backend == "pipnn") {
            params.algo = VecSimAlgo_PIPNN;
            params.algoParams.pipnnParams = {.type = VecSimType_FLOAT32,
                                             .dim = base.dim,
                                             .metric = VecSimMetric_L2,
                                             .maxDegree = 32,
                                             .seed = 100};
        } else {
            params.algo = VecSimAlgo_HNSWLIB;
            params.algoParams.hnswParams = {.type = VecSimType_FLOAT32,
                                            .dim = base.dim,
                                            .metric = VecSimMetric_L2,
                                            .M = 16,
                                            .efConstruction = 100};
        }
        const auto build_start = Clock::now();
        IndexPtr index(VecSimIndex_New(&params), VecSimIndex_Free);
        if (!index) {
            throw std::runtime_error("Index creation failed");
        }
        for (size_t i = 0; i < base.rows; ++i) {
            if (VecSimIndex_AddVector(index.get(), base.row(i), i) != 1) {
                throw std::runtime_error("Vector insertion failed at row " + std::to_string(i));
            }
        }
        const auto insert_end = Clock::now();
        if (backend == "pipnn" &&
            (!VecSimIndex_BuildGraph(index.get()) || !VecSimIndex_IsGraphReady(index.get()))) {
            throw std::runtime_error("PiPNN graph build failed");
        }
        const auto build_end = Clock::now();
        const auto index_bytes = VecSimIndex_StatsInfo(index.get()).memory;
        const auto build_peak_rss = peakRssBytes();
        const size_t base_count = base.rows;
        const size_t dimension = base.dim;
        std::vector<float>().swap(base.values);

        auto queries = readMatrix<float>(argv[2]);
        const auto truth = readMatrix<uint32_t>(argv[3]);
        if (queries.dim != dimension || truth.rows != queries.rows || truth.dim < k) {
            throw std::runtime_error("Query/truth dimensions do not match the base or each other");
        }
        const size_t query_count = std::min<size_t>(queries.rows, query_limit);
        for (size_t i = 0; i < query_count; ++i) {
            const auto *row = truth.row(i);
            for (size_t j = 0; j < k; ++j) {
                if (row[j] >= base_count || std::find(row, row + j, row[j]) != row + j) {
                    throw std::runtime_error("Out-of-range or duplicate ground-truth ID");
                }
            }
        }
        std::cout << std::setprecision(9) << "backend=" << backend << " n=" << base_count
                  << " dim=" << dimension << " queries=" << query_count
                  << " metric=L2 type=FLOAT32 seed=100";
        if (backend == "pipnn") {
            std::cout << " max_degree=32 leaf_size=128 fanout=2 hash_bits=12 candidate_limit=80"
                         " alpha=1.1";
        } else {
            std::cout << " M=16 ef_construction=100";
        }
        // Dataset I/O is excluded. Build RSS includes the base input matrix and process overhead.
        std::cout << " build_seconds=" << seconds(build_start, build_end)
                  << " creation_insert_seconds=" << seconds(build_start, insert_end)
                  << " graph_build_seconds=" << seconds(insert_end, build_end)
                  << " index_bytes=" << index_bytes << " build_peak_rss_bytes=" << build_peak_rss
                  << '\n';
        for (size_t ef : ef_values) {
            VecSimQueryParams query_params{};
            if (backend == "pipnn") {
                query_params.pipnnRuntimeParams.efRuntime = ef;
            } else {
                query_params.hnswRuntimeParams.efRuntime = ef;
            }
            for (size_t i = 0; i < std::min<size_t>(query_count, 100); ++i) {
                query(index.get(), queries.row(i), query_params);
            }
            std::vector<double> latency_us;
            latency_us.reserve(query_count);
            size_t hits = 0;
            for (size_t i = 0; i < query_count; ++i) {
                const auto start = Clock::now();
                auto reply = query(index.get(), queries.row(i), query_params);
                const auto end = Clock::now();
                latency_us.push_back(
                    std::chrono::duration<double, std::micro>(end - start).count());
                hits += countHits(reply.get(), truth.row(i), base_count);
            }
            const double query_us = std::accumulate(latency_us.begin(), latency_us.end(), 0.0);
            std::sort(latency_us.begin(), latency_us.end());
            // Nearest-rank percentiles: ceil(p * n) - 1, without floating-point rounding.
            const size_t p50 = query_count / 2 + query_count % 2 - 1;
            const size_t p95 = query_count - query_count / 20 - 1;
            std::cout << "backend=" << backend << " ef=" << ef
                      << " recall10=" << double(hits) / (double(k) * query_count)
                      << " p50_us=" << latency_us[p50] << " p95_us=" << latency_us[p95]
                      << " serial_query_qps=" << 1e6 * query_count / query_us
                      << " peak_rss_bytes=" << peakRssBytes() << '\n';
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "pipnn_comparison: " << error.what() << '\n';
        return 1;
    }
}

/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 *
 * Simple HNSW benchmark that creates an in-memory index for comparison with Rust.
 * This benchmark is self-contained and doesn't require external data files.
 *
 * Usage: ./simple_hnsw_bench
 */
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/index_factories/hnsw_factory.h"

constexpr size_t DIM = 128;
constexpr size_t N_VECTORS = 10000;
constexpr size_t M = 16;
constexpr size_t EF_CONSTRUCTION = 100;
constexpr size_t EF_RUNTIME = 100;
constexpr size_t N_QUERIES = 1000;

std::vector<float> generate_random_vector(size_t dim, std::mt19937 &gen) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}

int main() {
    std::mt19937 gen(42); // Fixed seed for reproducibility

    std::cout << "=== C++ HNSW Benchmark ===" << std::endl;
    std::cout << "Config: " << N_VECTORS << " vectors, " << DIM << " dimensions, M=" << M
              << ", ef_construction=" << EF_CONSTRUCTION << ", ef_runtime=" << EF_RUNTIME
              << std::endl;
    std::cout << std::endl;

    // Create HNSW parameters
    HNSWParams params = {.dim = DIM,
                         .metric = VecSimMetric_L2,
                         .type = VecSimType_FLOAT32,
                         .M = M,
                         .efConstruction = EF_CONSTRUCTION,
                         .efRuntime = EF_RUNTIME};

    // Create index
    VecSimIndex *index = HNSWFactory::NewIndex(&params);

    // Generate and insert vectors
    std::cout << "Inserting " << N_VECTORS << " vectors..." << std::endl;
    auto insert_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N_VECTORS; ++i) {
        auto vec = generate_random_vector(DIM, gen);
        VecSimIndex_AddVector(index, vec.data(), i);
    }

    auto insert_end = std::chrono::high_resolution_clock::now();
    auto insert_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(insert_end - insert_start);
    double insert_throughput = N_VECTORS * 1000.0 / insert_duration.count();

    std::cout << "Insertion time: " << insert_duration.count() << " ms (" << insert_throughput
              << " vec/s)" << std::endl;
    std::cout << std::endl;

    // Generate query vectors
    std::vector<std::vector<float>> queries;
    for (size_t i = 0; i < N_QUERIES; ++i) {
        queries.push_back(generate_random_vector(DIM, gen));
    }

    // KNN Search benchmark
    std::cout << "Running " << N_QUERIES << " KNN queries (k=10)..." << std::endl;

    auto knn_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N_QUERIES; ++i) {
        VecSimQueryReply *results =
            VecSimIndex_TopKQuery(index, queries[i].data(), 10, nullptr, BY_SCORE);
        VecSimQueryReply_Free(results);
    }

    auto knn_end = std::chrono::high_resolution_clock::now();
    auto knn_duration = std::chrono::duration_cast<std::chrono::microseconds>(knn_end - knn_start);
    double avg_knn_time = static_cast<double>(knn_duration.count()) / N_QUERIES;
    double knn_throughput = N_QUERIES * 1000000.0 / knn_duration.count();

    std::cout << "KNN k=10 (ef=" << EF_RUNTIME << "): avg " << avg_knn_time << " µs ("
              << knn_throughput << " queries/s)" << std::endl;
    std::cout << std::endl;

    // Range Search benchmark
    std::cout << "Running " << N_QUERIES << " Range queries (radius=10.0)..." << std::endl;

    auto range_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N_QUERIES; ++i) {
        VecSimQueryReply *results =
            VecSimIndex_RangeQuery(index, queries[i].data(), 10.0, nullptr, BY_SCORE);
        VecSimQueryReply_Free(results);
    }

    auto range_end = std::chrono::high_resolution_clock::now();
    auto range_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(range_end - range_start);
    double avg_range_time = static_cast<double>(range_duration.count()) / N_QUERIES;
    double range_throughput = N_QUERIES * 1000000.0 / range_duration.count();

    std::cout << "Range (r=10): avg " << avg_range_time << " µs (" << range_throughput
              << " queries/s)" << std::endl;
    std::cout << std::endl;

    // Cleanup
    VecSimIndex_Free(index);

    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Insertion:    " << insert_throughput << " vec/s" << std::endl;
    std::cout << "KNN (k=10):   " << avg_knn_time << " µs (" << knn_throughput << " q/s)"
              << std::endl;
    std::cout << "Range (r=10): " << avg_range_time << " µs (" << range_throughput << " q/s)"
              << std::endl;

    return 0;
}

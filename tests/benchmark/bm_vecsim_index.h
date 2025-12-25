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
#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/index_factories/hnsw_disk_factory.h"
#include "VecSim/index_factories/components/components_factory.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "VecSim/algorithms/hnsw/hnsw_disk.h"
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/statistics.h>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>

#include <iomanip>

template <typename index_type_t>
class BM_VecSimIndex : public BM_VecSimGeneral {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    // Tracks initialization state to ensure one-time initialization
    static bool is_initialized;
    static std::vector<std::vector<data_t>> queries;
    static std::array<IndexPtr, NUMBER_OF_INDEX_TYPES> indices;

    BM_VecSimIndex() {
        if (!is_initialized) {
            Initialize();
            is_initialized = true;
        }
    }

    virtual ~BM_VecSimIndex() {
        // Cleanup is handled by the factory's managed resources
    }

    // The implicit conversion operator in IndexPtr allows automatic conversion to VecSimIndex*.
    // This lets indices[algo] be used directly as a VecSimIndex* without explicitly calling .get().
    static VecSimIndex *get_index(IndexTypeIndex algo) { return indices[algo]; }

    static VecSimIndex *get_index(int64_t google_bm_arg) {
        return get_index(static_cast<IndexTypeIndex>(google_bm_arg));
    }

    // Helper method for dynamic casting
    template <typename T>
    static T *get_typed_index(IndexTypeIndex algo) {
        return dynamic_cast<T *>(get_index(algo));
    }

protected:
    static inline HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        return dynamic_cast<HNSWIndex<data_t, dist_t> *>(index);
    }
    static inline const char *GetHNSWDataByInternalId(size_t id, unsigned short index_offset = 0) {
        return CastToHNSW(indices[INDEX_HNSW + index_offset])->getDataByInternalId(id);
    }

    static VecSimQueryReply *TopKGroundTruth(size_t query_id, size_t k);

private:
    // Flag to control whether to build HNSW_DISK from scratch or load from serialized index
    static constexpr bool BUILD_HNSW_DISK_FROM_SCRATCH = false;

    static void Initialize();
    static void InsertToQueries(std::ifstream &input);
    static void loadTestVectors(const std::string &test_file, VecSimType type);
    static void buildHNSWDiskFromScratch(VecSimType type);
    static void loadHNSWDiskFromFolder();
};

template <typename index_type_t>
bool BM_VecSimIndex<index_type_t>::is_initialized = false;

// Needs to be explicitly initalized
template <>
std::vector<std::vector<float>> BM_VecSimIndex<fp32_index_t>::queries{};

template <>
std::vector<std::vector<double>> BM_VecSimIndex<fp64_index_t>::queries{};

template <>
std::vector<std::vector<vecsim_types::bfloat16>> BM_VecSimIndex<bf16_index_t>::queries{};

template <>
std::vector<std::vector<vecsim_types::float16>> BM_VecSimIndex<fp16_index_t>::queries{};

template <>
std::vector<std::vector<int8_t>> BM_VecSimIndex<int8_index_t>::queries{};

template <>
std::vector<std::vector<uint8_t>> BM_VecSimIndex<uint8_index_t>::queries{};

// Indices array specializations - use IndexPtr default constructor
template <>
std::array<IndexPtr, NUMBER_OF_INDEX_TYPES> BM_VecSimIndex<fp32_index_t>::indices{};

template <>
std::array<IndexPtr, NUMBER_OF_INDEX_TYPES> BM_VecSimIndex<fp64_index_t>::indices{};

template <>
std::array<IndexPtr, NUMBER_OF_INDEX_TYPES> BM_VecSimIndex<bf16_index_t>::indices{};

template <>
std::array<IndexPtr, NUMBER_OF_INDEX_TYPES> BM_VecSimIndex<fp16_index_t>::indices{};

template <>
std::array<IndexPtr, NUMBER_OF_INDEX_TYPES> BM_VecSimIndex<int8_index_t>::indices{};

template <>
std::array<IndexPtr, NUMBER_OF_INDEX_TYPES> BM_VecSimIndex<uint8_index_t>::indices{};

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::Initialize() {
    using clock = std::chrono::high_resolution_clock;
    VecSimType type = index_type_t::get_index_type();

    if (enabled_index_types & IndexTypeFlags::INDEX_MASK_HNSW) {
        auto t0 = clock::now();
        indices[INDEX_HNSW] = IndexPtr(HNSWFactory::NewIndex(AttachRootPath(hnsw_index_file)));

        auto *hnsw_index = CastToHNSW(indices[INDEX_HNSW]);
        auto took =
            std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
        std::cerr << "[Init] Loaded HNSW. indexSize=" << hnsw_index->indexSize() << ", took "
                  << took << " ms" << std::endl;

        size_t ef_r = 10;
        hnsw_index->setEf(ef_r);
        // Create tiered index from the loaded HNSW index.
        if (enabled_index_types & IndexTypeFlags::INDEX_MASK_TIERED_HNSW) {
            BM_VecSimGeneral::mock_thread_pool = new tieredIndexMock();
            auto &mock_thread_pool = *BM_VecSimGeneral::mock_thread_pool;
            TieredIndexParams tiered_params = {
                .jobQueue = &mock_thread_pool.jobQ,
                .jobQueueCtx = mock_thread_pool.ctx,
                .submitCb = tieredIndexMock::submit_callback,
                .flatBufferLimit = block_size,
                .primaryIndexParams = nullptr,
                .specificParams = {TieredHNSWParams{.swapJobThreshold = 0}}};

            auto *tiered_index = TieredFactory::TieredHNSWFactory::NewIndex<data_t, dist_t>(
                &tiered_params, hnsw_index);

            // Ownership model:
            // 1. Mock thread pool holds a strong reference (shared_ptr) to tiered index
            // 2. Tiered index owns and will free the HNSW index in its destructor
            indices[INDEX_TIERED_HNSW] = IndexPtr(tiered_index);
            mock_thread_pool.ctx->index_strong_ref = indices[INDEX_TIERED_HNSW].get_shared();
            // Release HNSW ownership since tiered will free it (sets owns_ptr=false)
            indices[INDEX_HNSW].release_ownership();

            // Launch the BG threads loop that takes jobs from the queue and executes them.
            mock_thread_pool.init_threads();
        }
    }
    

    if (enabled_index_types & IndexTypeFlags::INDEX_MASK_HNSW_DISK) {
        if (BUILD_HNSW_DISK_FROM_SCRATCH) {
            buildHNSWDiskFromScratch(type);
        } else {
            loadHNSWDiskFromFolder();
        }

        // If no tiered index is enabled, but we want to benchmark parallel HNSW_DISK queries,
        // create a mock thread pool that owns a strong reference to the disk index and
        // launches background worker threads. This lets disk-only benchmarks drive parallel
        // searches without involving TieredHNSW.
        if (!BM_VecSimGeneral::mock_thread_pool) {
            BM_VecSimGeneral::mock_thread_pool = new tieredIndexMock();
            auto &mock_thread_pool = *BM_VecSimGeneral::mock_thread_pool;
            mock_thread_pool.ctx->index_strong_ref = indices[INDEX_HNSW_DISK].get_shared();
            // Threads will be started on-demand by the benchmark via reconfigure_threads().
            // NOTE: Job queue is NOT set here - individual benchmarks that need async
            // processing should call setJobQueue() with appropriate thread configuration.
            // The regular AddLabel benchmark uses single-threaded mode (no job queue).
        }
    }

    if (enabled_index_types & IndexTypeFlags::INDEX_MASK_BF) {
        constexpr const char *FBIN_PATH = "tests/benchmark/data/deep.base.10K.fbin";
        constexpr size_t BATCH_SIZE = 40;
        BFParams bf_params = {.type = type,
                              .dim = dim,
                              .metric = VecSimMetric_Cosine,
                              .multi = is_multi,
                              .blockSize = block_size};
        indices[INDEX_BF] = IndexPtr(CreateNewIndex(bf_params));

        std::ifstream file(AttachRootPath(FBIN_PATH), std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open vectors file for BF index");
        }
        uint32_t file_num_vectors = 0, file_dim = 0;
        file.read(reinterpret_cast<char *>(&file_num_vectors), sizeof(uint32_t));
        file.read(reinterpret_cast<char *>(&file_dim), sizeof(uint32_t));
        size_t load_count = std::min<size_t>(file_num_vectors, n_vectors);
        size_t dim_from_file = file_dim;

        std::vector<float> batch_buffer;
        batch_buffer.resize(BATCH_SIZE * dim_from_file);

        size_t processed = 0;
        while (processed < load_count) {
            size_t in_batch = std::min(BATCH_SIZE, load_count - processed);
            size_t floats_to_read = in_batch * dim_from_file;
            file.read(reinterpret_cast<char *>(batch_buffer.data()),
                      floats_to_read * sizeof(float));
            if (!file) {
                throw std::runtime_error("Failed to read vectors from fbin file for BF index");
            }
            for (size_t j = 0; j < in_batch; ++j) {
                size_t i = processed + j;
                const float *vec_ptr = batch_buffer.data() + j * dim_from_file;
                VecSimIndex_AddVector(indices[INDEX_BF], vec_ptr, i);
            }
            processed += in_batch;
        }
    }

    // Load the test query vectors from file. Index file path is relative to repository root dir.
    loadTestVectors(AttachRootPath(test_queries_file), type);
    VecSim_SetLogCallbackFunction(nullptr);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::loadTestVectors(const std::string &test_file, VecSimType type) {

    if (enabled_index_types & IndexTypeFlags::INDEX_MASK_HNSW_DISK) {
        // Open the .fbin file for reading: [num_vectors (uint32), vector_dim (uint32), data
        // (float32)]
        std::ifstream file(test_file, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[Init] Error: Could not open file " << test_file << std::endl;
            throw std::runtime_error("Failed to open vectors file");
        }

        // Read header (without extra validation)
        uint32_t file_num_vectors = 0;
        uint32_t file_dim = 0;
        file.read(reinterpret_cast<char *>(&file_num_vectors), sizeof(uint32_t));
        file.read(reinterpret_cast<char *>(&file_dim), sizeof(uint32_t));
        assert(file_dim == dim && "Query file dimension mismatch");
        n_queries = std::min((size_t)file_num_vectors, n_queries);
        std::cerr << "[Init] Loaded " << n_queries << " queries from " << test_file << std::endl;
        InsertToQueries(file);
    } else {
        std::ifstream input(test_file, std::ios::binary);

        if (!input.is_open()) {
            throw std::runtime_error("Test vectors file was not found in path. Exiting...");
        }
        input.seekg(0, std::ifstream::beg);

        InsertToQueries(input);
    }
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::InsertToQueries(std::ifstream &input) {
    for (size_t i = 0; i < n_queries; i++) {
        std::vector<data_t> query(dim);
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}

template <typename index_type_t>
VecSimQueryReply *BM_VecSimIndex<index_type_t>::TopKGroundTruth(size_t query_id, size_t k) {
    std::ifstream file(AttachRootPath(ground_truth_file), std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open ground truth file");
    }
    uint32_t file_num_vectors = 0, gt_k = 0;
    file.read(reinterpret_cast<char *>(&file_num_vectors), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&gt_k), sizeof(uint32_t));
    std::vector<int32_t> query(gt_k);
    file.seekg((2 + query_id * gt_k) * sizeof(int32_t), std::ifstream::beg);
    file.read((char *)query.data(), gt_k * sizeof(int32_t));
    auto res = new VecSimQueryReply(VecSimAllocator::newVecsimAllocator());
    for (size_t i = 0; i < k; i++) {
        res->results.push_back(VecSimQueryResult{.id = (size_t)query[i], .score = 0.0});
    }
    return res;
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::loadHNSWDiskFromFolder() {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    // Load from pre-built disk index folder (hnsw_index_file points to folder for disk index)
    std::string folder_path = AttachRootPath(hnsw_index_file);

    std::cerr << "[Init] Loading HNSW_DISK from folder: " << folder_path << std::endl;

    // Load the index from folder (opens checkpoint and redirects writes to temp)
    indices[INDEX_HNSW_DISK] = IndexPtr(HNSWDiskFactory::NewIndex(folder_path));

    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(indices[INDEX_HNSW_DISK].get());
    auto took = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
    std::cerr << "[Init] Loaded HNSW_DISK from folder. indexSize="
                  << VecSimIndex_IndexSize(indices[INDEX_HNSW_DISK])
                  << ", maxLevel=" << disk_index->getMaxLevel()
                  << ", took " << took << " ms" << std::endl;
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::buildHNSWDiskFromScratch(VecSimType type) {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    // Build index from scratch using 100K vectors file
    constexpr const char *FBIN_PATH = "tests/benchmark/data/deep.base.1M.fbin";
    constexpr size_t BATCH_SIZE = 1000;

    // Create temporary RocksDB directory
    std::string rocksdb_path = "/tmp/hnsw_disk_benchmark_" + std::to_string(getpid());
    std::filesystem::create_directories(rocksdb_path);

    std::cerr << "[Init] Building HNSW_DISK from scratch using: " << FBIN_PATH << std::endl;

    // Create HNSW disk index parameters
    HNSWDiskParams disk_params;
    disk_params.type = type;
    disk_params.dim = dim;
    disk_params.metric = VecSimMetric_L2;
    disk_params.multi = false;
    disk_params.initialCapacity = 0;
    disk_params.blockSize = block_size;
    disk_params.M = 32;
    disk_params.efConstruction = 200;
    disk_params.efRuntime = 10;
    disk_params.epsilon = HNSW_DEFAULT_EPSILON;
    disk_params.dbPath = rocksdb_path.c_str();

    VecSimParams vecsimParams;
    vecsimParams.algo = VecSimAlgo_HNSWLIB_DISK;
    vecsimParams.algoParams.hnswDiskParams = disk_params;

    indices[INDEX_HNSW_DISK] = IndexPtr(VecSimIndex_New(&vecsimParams));

    // Load vectors from fbin file
    std::ifstream file(AttachRootPath(FBIN_PATH), std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open vectors file for HNSW_DISK index");
    }
    uint32_t file_num_vectors = 0, file_dim = 0;
    file.read(reinterpret_cast<char *>(&file_num_vectors), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(&file_dim), sizeof(uint32_t));

    std::vector<float> batch_buffer;
    batch_buffer.resize(BATCH_SIZE * file_dim);

    size_t processed = 0;
    while (processed < file_num_vectors) {
        size_t in_batch = std::min(BATCH_SIZE, static_cast<size_t>(file_num_vectors) - processed);
        size_t floats_to_read = in_batch * file_dim;
        file.read(reinterpret_cast<char *>(batch_buffer.data()),
                  floats_to_read * sizeof(float));
        if (!file) {
            throw std::runtime_error("Failed to read vectors from fbin file for HNSW_DISK index");
        }
        for (size_t j = 0; j < in_batch; ++j) {
            size_t i = processed + j;
            const float *vec_ptr = batch_buffer.data() + j * file_dim;
            VecSimIndex_AddVector(indices[INDEX_HNSW_DISK], vec_ptr, i);
        }
        processed += in_batch;
    }

    auto *disk_index = dynamic_cast<HNSWDiskIndex<data_t, dist_t> *>(indices[INDEX_HNSW_DISK].get());
    auto took = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
    std::cerr << "[Init] Built HNSW_DISK from scratch. indexSize="
                  << VecSimIndex_IndexSize(indices[INDEX_HNSW_DISK])
                  << ", maxLevel=" << disk_index->getMaxLevel()
                  << ", took " << took << " ms" << std::endl;
}

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

    // RocksDB management for disk-based HNSW indices (fp32 only)
    static std::unique_ptr<rocksdb::DB> benchmark_db;
    static rocksdb::ColumnFamilyHandle* benchmark_cf;
    static std::string rocksdb_temp_dir;

    BM_VecSimIndex() {
        if (!is_initialized) {
            Initialize();
            is_initialized = true;
        }
    }

    virtual ~BM_VecSimIndex() {
        // Cleanup RocksDB resources for fp32 type only
        CleanupRocksDB();
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
    static void InitializeRocksDB();
    static void CleanupRocksDB();
    static void Initialize();
    static void InsertToQueries(std::ifstream &input);
    static void loadTestVectors(const std::string &test_file, VecSimType type);

    // FBIN loading configuration
    static constexpr size_t BATCH_SIZE = 40;
    static constexpr const char *FBIN_PATH = "tests/benchmark/data/deep.base.10K.fbin";

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

// RocksDB static member specializations
template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<fp32_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<fp32_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<fp32_index_t>::rocksdb_temp_dir{};

// // fp64
template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<fp64_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<fp64_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<fp64_index_t>::rocksdb_temp_dir{};

// bf16
template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<bf16_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<bf16_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<bf16_index_t>::rocksdb_temp_dir{};

// fp16
template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<fp16_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<fp16_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<fp16_index_t>::rocksdb_temp_dir{};

// int8
template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<int8_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<int8_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<int8_index_t>::rocksdb_temp_dir{};

// uint8
template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<uint8_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<uint8_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<uint8_index_t>::rocksdb_temp_dir{};

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::Initialize() {
    using clock = std::chrono::high_resolution_clock;
    VecSimType type = index_type_t::get_index_type();

    if (enabled_index_types & IndexTypeFlags::INDEX_MASK_HNSW) {
        auto t0 = clock::now();
        indices[INDEX_HNSW] = IndexPtr(HNSWFactory::NewIndex(AttachRootPath(hnsw_index_file)));

        auto *hnsw_index = CastToHNSW(indices[INDEX_HNSW]);
        auto took =std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
        std::cerr << "[Init] Loaded HNSW. indexSize=" << hnsw_index->indexSize()
                  << ", took " << took << " ms" << std::endl;

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
        // Initialize RocksDB for disk-based HNSW index
        InitializeRocksDB();
        // Open the .fbin file for reading: [num_vectors (uint32), vector_dim (uint32), data (float32)]
        std::ifstream file(AttachRootPath(FBIN_PATH), std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[Init] Error: Could not open file " << FBIN_PATH << std::endl;
            throw std::runtime_error("Failed to open vectors file");
        }

        // Read header (without extra validation)
        uint32_t file_num_vectors = 0;
        uint32_t file_dim = 0;
        file.read(reinterpret_cast<char*>(&file_num_vectors), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&file_dim), sizeof(uint32_t));
        
        n_vectors = file_num_vectors;
        dim = file_dim;
        std::cerr << "[Init] Starting initialization: dim=" << dim << ", M=" << M << ", efC=" << EF_C
              << ", n_vectors=" << n_vectors << std::endl;
        // Create disk-based HNSW index with RocksDB
        HNSWParams hnsw_disk_params = {.type = type,
                                       .dim = dim,
                                       .metric = VecSimMetric_L2,
                                       .multi = is_multi,
                                       .initialCapacity = 0, // Deprecated
                                       .blockSize = block_size,
                                       .M = M,
                                       .efConstruction = EF_C,
                                       .efRuntime = 10,
                                       .epsilon = 0.01};
        
        // Create AbstractIndexInitParams manually
        AbstractIndexInitParams abstractInitParams;
        abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
        abstractInitParams.dim = dim;
        abstractInitParams.vecType = type;
        abstractInitParams.dataSize = dim * sizeof(int8_t);  // Quantized storage (int8)
        abstractInitParams.metric = VecSimMetric_L2;
        abstractInitParams.blockSize = block_size;
        abstractInitParams.multi = is_multi;
        abstractInitParams.logCtx = nullptr;

        // Create quantized index components (scalar quantization enabled by default)
        IndexComponents<float, float> indexComponents = CreateQuantizedIndexComponents<float, float>(
            abstractInitParams.allocator, VecSimMetric_L2, dim, false);

        indices[INDEX_HNSW_DISK] = IndexPtr(new (abstractInitParams.allocator) HNSWDiskIndex<float, float>(
            &hnsw_disk_params, abstractInitParams, indexComponents, benchmark_db.get(), benchmark_cf));
            
        // Populate the disk index by loading vectors from file
        if (enabled_index_types & IndexTypeFlags::INDEX_MASK_HNSW) {
            
            auto t0 = clock::now();
            // Allocate reusable buffer for batch reading (float32-aligned)
            std::vector<float> batch_buffer;
            batch_buffer.resize(BATCH_SIZE * dim);

            size_t vectors_processed = 0;
            size_t print_every = std::max<size_t>(size_t(1000), n_vectors / 20); // ~5% or 1000
            while (vectors_processed < n_vectors) {
                size_t vectors_in_batch = std::min(BATCH_SIZE, n_vectors - vectors_processed);
                size_t floats_to_read = vectors_in_batch * dim;

                file.read(reinterpret_cast<char*>(batch_buffer.data()), floats_to_read * sizeof(float));
                if (!file) {
                    std::cerr << "[Init] Error: Failed reading fbin data at vector " << vectors_processed << std::endl;
                    throw std::runtime_error("Failed to read vectors from fbin file");
                }

                for (size_t j = 0; j < vectors_in_batch; ++j) {
                    size_t i = vectors_processed + j; // global id
                    const void* vector_data = batch_buffer.data() + j * dim;
                    // if (vectors_processed == 0) {
                    //     std::cerr << "[Init] Adding vector " << i << " : " << std::endl;
                    //     for (size_t k = 0; k < dim; ++k) {
                    //         std::cerr << " " << ((const float*)vector_data)[k];
                    //     }
                    //     std::cerr << std::endl;
                    // }
                    VecSimIndex_AddVector(indices[INDEX_HNSW_DISK], vector_data, i);

                    if (i % print_every == 0 || i + 1 == n_vectors) {
                        auto now = clock::now();
                        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
                        double elapsed_s = std::max<double>(1e-6, elapsed_ms / 1000.0);
                        double rate = (double)(i + 1) / elapsed_s; // vectors/sec
                        size_t pct = ((i + 1) * 100) / n_vectors;
                        std::cerr << "[Init " << elapsed_s << "s] HNSWDiskIndex fbin load " << pct << "% ("
                                  << (i + 1) << "/" << n_vectors << ") "
                                  << (size_t)rate << " vec/s" << std::endl;
                    }
                }

                vectors_processed += vectors_in_batch;
            }

            file.close();

            auto t1 = clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();
            int hours = duration / 3600;
            int minutes = (duration % 3600) / 60;
            int seconds = duration % 60;
            std::cerr << "[Init] HNSWDiskIndex population done in "
                      << std::setfill('0') << std::setw(2) << hours << ":"
                      << std::setfill('0') << std::setw(2) << minutes << ":"
                      << std::setfill('0') << std::setw(2) << seconds << std::endl;
        }
    }

    if (enabled_index_types & IndexTypeFlags::INDEX_MASK_BF) {
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
        file.read(reinterpret_cast<char*>(&file_num_vectors), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&file_dim), sizeof(uint32_t));
        size_t load_count = std::min<size_t>(file_num_vectors, n_vectors);
        size_t dim_from_file = file_dim;

        std::vector<float> batch_buffer;
        batch_buffer.resize(BATCH_SIZE * dim_from_file);

        size_t processed = 0;
        while (processed < load_count) {
            size_t in_batch = std::min(BATCH_SIZE, load_count - processed);
            size_t floats_to_read = in_batch * dim_from_file;
            file.read(reinterpret_cast<char*>(batch_buffer.data()), floats_to_read * sizeof(float));
            if (!file) {
                throw std::runtime_error("Failed to read vectors from fbin file for BF index");
            }
            for (size_t j = 0; j < in_batch; ++j) {
                size_t i = processed + j;
                const float* vec_ptr = batch_buffer.data() + j * dim_from_file;
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
        // Open the .fbin file for reading: [num_vectors (uint32), vector_dim (uint32), data (float32)]
        std::ifstream file(test_file, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[Init] Error: Could not open file " << FBIN_PATH << std::endl;
            throw std::runtime_error("Failed to open vectors file");
        }

        // Read header (without extra validation)
        uint32_t file_num_vectors = 0;
        uint32_t file_dim = 0;
        file.read(reinterpret_cast<char*>(&file_num_vectors), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&file_dim), sizeof(uint32_t));
        assert(file_dim == dim && "Query file dimension mismatch");
        n_queries = std::min((size_t)file_num_vectors, n_queries);
        std::cerr << "[Init] Loaded " << n_queries << " queries from " << test_file << std::endl;
        InsertToQueries(file);
    }
    else {
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
    std::map <std::string, std::string> gt_files = {
        {"deep.base.10K.fbin", "deep.groundtruth.10K.10K.ibin"},
        {"deep.base.100K.fbin", "deep.groundtruth.100K.10K.ibin"},
        {"deep.base.1M.fbin", "deep.groundtruth.1M.10K.ibin"},
        {"deep.base.10M.fbin", "deep.groundtruth.10M.10K.ibin"}
    };
    std::filesystem::path fbin_path(FBIN_PATH);
    std::string filename = fbin_path.filename().string();
    std::string directory = fbin_path.parent_path().string();
    std::string gt_file_name = gt_files[filename];
    std::string ground_truth_file = (directory + "/" + gt_file_name);
    std::ifstream file(AttachRootPath(ground_truth_file), std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open ground truth file");
    }
    uint32_t file_num_vectors = 0, gt_k = 0;
    file.read(reinterpret_cast<char*>(&file_num_vectors), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&gt_k), sizeof(uint32_t));
    std::vector<int32_t> query(gt_k);
    file.seekg((2 + query_id * gt_k) * sizeof(int32_t), std::ifstream::beg);
    file.read((char *)query.data(), gt_k * sizeof(int32_t));
    auto res = new VecSimQueryReply(VecSimAllocator::newVecsimAllocator());
    for (size_t i = 0; i < k; i++) {
        res->results.push_back(VecSimQueryResult{.id = (size_t)query[i], .score = 0.0});
    }
    return res;
}

// RocksDB initialization and cleanup methods
template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::InitializeRocksDB() {
    if (benchmark_db) {
        return; // Already initialized
    }

    // Create a temporary directory for RocksDB
    rocksdb_temp_dir = "/tmp/hnsw_disk_benchmark_" + std::to_string(getpid());
    std::cerr << "RocksDB temp dir: " << rocksdb_temp_dir << std::endl;
    // Ensure the directory exists
    std::filesystem::create_directories(rocksdb_temp_dir);

    rocksdb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = false;
    options.statistics = rocksdb::CreateDBStatistics();
    
    rocksdb::DB* db_ptr = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, rocksdb_temp_dir, &db_ptr);
    if (!status.ok()) {
        throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
    }

    benchmark_db.reset(db_ptr);
    benchmark_cf = benchmark_db->DefaultColumnFamily();
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::CleanupRocksDB() {
    // Destroy the disk index BEFORE closing the database
    // This ensures the index is destroyed while the database is still valid
    if (indices[INDEX_HNSW_DISK].get() != nullptr) {
        indices[INDEX_HNSW_DISK] = IndexPtr();
    }

    if (benchmark_db) {
        benchmark_db.release();
    }

    benchmark_cf = nullptr;

    // Clean up temporary directory
    if (!rocksdb_temp_dir.empty()) {
        try {
            std::filesystem::remove_all(rocksdb_temp_dir);
        } catch (...) {
            // Ignore cleanup errors
        }
        rocksdb_temp_dir.clear();
    }
}


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
#include <filesystem>

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
        // Don't cleanup static RocksDB database here as it's shared across benchmarks
        // CleanupRocksDB();
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

private:
    // RocksDB management for disk-based HNSW indices
    static std::unique_ptr<rocksdb::DB> benchmark_db;
    static rocksdb::ColumnFamilyHandle* benchmark_cf;
    static std::string rocksdb_temp_dir;
    
    static void InitializeRocksDB();
    static void CleanupRocksDB();
    
    static void Initialize();
    static void InsertToQueries(std::ifstream &input);
    static void loadTestVectors(const std::string &test_file, VecSimType type);
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

template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<fp64_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<fp64_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<fp64_index_t>::rocksdb_temp_dir{};

template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<bf16_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<bf16_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<bf16_index_t>::rocksdb_temp_dir{};

template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<fp16_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<fp16_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<fp16_index_t>::rocksdb_temp_dir{};

template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<int8_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<int8_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<int8_index_t>::rocksdb_temp_dir{};

template <>
std::unique_ptr<rocksdb::DB> BM_VecSimIndex<uint8_index_t>::benchmark_db{};
template <>
rocksdb::ColumnFamilyHandle* BM_VecSimIndex<uint8_index_t>::benchmark_cf{};
template <>
std::string BM_VecSimIndex<uint8_index_t>::rocksdb_temp_dir{};

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::Initialize() {
    VecSimType type = index_type_t::get_index_type();
    // dim, block_size, M, EF_C, n_vectors, is_multi, n_queries, hnsw_index_file and
    // test_queries_file are BM_VecSimGeneral static data members that are defined for a specific
    // index type benchmarks.

    if (enabled_index_types & IndexTypeFlags::INDEX_MASK_HNSW) {
        // Initialize and load HNSW index for DBPedia data set.
        indices[INDEX_HNSW] = IndexPtr(HNSWFactory::NewIndex(AttachRootPath(hnsw_index_file)));

        auto *hnsw_index = CastToHNSW(indices[INDEX_HNSW]);
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
        
        // Create disk-based HNSW index with RocksDB
        HNSWParams hnsw_disk_params = {.type = type,
                                       .dim = dim,
                                       .metric = VecSimMetric_Cosine,
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
        abstractInitParams.dataSize = VecSimParams_GetDataSize(type, dim, VecSimMetric_Cosine);
        abstractInitParams.metric = VecSimMetric_Cosine;
        abstractInitParams.blockSize = block_size;
        abstractInitParams.multi = is_multi;
        abstractInitParams.logCtx = nullptr;
        
        // Create index components
        IndexComponents<data_t, dist_t> indexComponents = CreateIndexComponents<data_t, dist_t>(
            abstractInitParams.allocator, VecSimMetric_Cosine, dim, false);
            
        indices[INDEX_HNSW_DISK] = IndexPtr(new (abstractInitParams.allocator) HNSWDiskIndex<data_t, dist_t>(
            &hnsw_disk_params, abstractInitParams, indexComponents, 
            benchmark_db.get(), benchmark_cf));
            
        // Populate the disk index with the same vectors as the HNSW index
        if (enabled_index_types & IndexTypeFlags::INDEX_MASK_HNSW) {
            auto *hnsw_index = CastToHNSW(indices[INDEX_HNSW]);
            size_t n_vectors = hnsw_index->indexSize();
            
            for (size_t i = 0; i < n_vectors; ++i) {
                const char *blob = GetHNSWDataByInternalId(i);
                size_t label = hnsw_index->getExternalLabel(i);
                VecSimIndex_AddVector(indices[INDEX_HNSW_DISK], blob, label);
            }
        }
    }

    if (enabled_index_types & IndexTypeFlags::INDEX_MASK_BF) {
        BFParams bf_params = {.type = type,
                              .dim = dim,
                              .metric = VecSimMetric_Cosine,
                              .multi = is_multi,
                              .blockSize = block_size};
        indices[INDEX_BF] = IndexPtr(CreateNewIndex(bf_params));

        // Currently, we rely on hnsw index to initialize BF index.
        assert(enabled_index_types & IndexTypeFlags::INDEX_MASK_HNSW);
        // Add the same vectors to Flat index.
        for (size_t i = 0; i < n_vectors; ++i) {
            const char *blob = GetHNSWDataByInternalId(i);
            // For multi value indices, the internal id is not necessarily equal the label.
            size_t label = CastToHNSW(indices[INDEX_HNSW])->getExternalLabel(i);
            VecSimIndex_AddVector(indices[INDEX_BF], blob, label);
        }
    }

    // Load the test query vectors from file. Index file path is relative to repository root dir.
    loadTestVectors(AttachRootPath(test_queries_file), type);
    VecSim_SetLogCallbackFunction(nullptr);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::loadTestVectors(const std::string &test_file, VecSimType type) {

    std::ifstream input(test_file, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);

    InsertToQueries(input);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::InsertToQueries(std::ifstream &input) {
    for (size_t i = 0; i < n_queries; i++) {
        std::vector<data_t> query(dim);
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}

// RocksDB initialization and cleanup methods
template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::InitializeRocksDB() {
    if (benchmark_db) {
        return; // Already initialized
    }
    
    // Create a temporary directory for RocksDB
    rocksdb_temp_dir = "/tmp/hnsw_disk_benchmark_" + std::to_string(getpid());
    
    // Ensure the directory exists
    std::filesystem::create_directories(rocksdb_temp_dir);
    
    rocksdb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = false;
    
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
    if (benchmark_cf) {
        benchmark_cf = nullptr;
    }
    
    if (benchmark_db) {
        benchmark_db.reset();
    }
    
    if (!rocksdb_temp_dir.empty()) {
        try {
            std::filesystem::remove_all(rocksdb_temp_dir);
        } catch (...) {
            // Ignore cleanup errors
        }
        rocksdb_temp_dir.clear();
    }
}

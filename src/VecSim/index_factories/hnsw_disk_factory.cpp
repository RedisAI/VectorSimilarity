/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "hnsw_disk_factory.h"
#include "VecSim/algorithms/hnsw/hnsw_disk.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/spaces/space_includes.h"
#include "components/components_factory.h"
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <ctime>
#include <cstdlib>

namespace HNSWDiskFactory {

#ifdef BUILD_TESTS

// RAII wrapper to manage RocksDB database and temporary directory cleanup
class ManagedRocksDB {
private:
    std::unique_ptr<rocksdb::DB> db;
    rocksdb::ColumnFamilyHandle *cf = nullptr;
    std::string temp_dir;
    bool cleanup_temp_dir;  // Whether to delete temp_dir on destruction

public:
    // Constructor for loading from checkpoint (with temp directory for writes)
    // Copies the entire checkpoint to a temp location to ensure the original is never modified
    ManagedRocksDB(const std::string &checkpoint_dir, const std::string &temp_path)
        : temp_dir(temp_path), cleanup_temp_dir(true) {

        // Create temp directory
        std::filesystem::create_directories(temp_dir);

        // Copy the entire checkpoint to temp location to preserve the original
        std::string temp_checkpoint = temp_dir + "/checkpoint_copy";
        try {
            std::filesystem::copy(checkpoint_dir, temp_checkpoint,
                                std::filesystem::copy_options::recursive);
        } catch (const std::filesystem::filesystem_error &e) {
            // Clean up temp dir if copy failed
            std::filesystem::remove_all(temp_dir);
            throw std::runtime_error("Failed to copy checkpoint to temp location: " +
                                   std::string(e.what()));
        }

        // Open RocksDB from the temp checkpoint copy
        // All writes (WAL, SST, MANIFEST, etc.) will go to the temp location
        rocksdb::Options options;
        options.create_if_missing = false;  // Checkpoint copy should exist
        options.error_if_exists = false;
        options.statistics = rocksdb::CreateDBStatistics();

        rocksdb::DB *db_ptr = nullptr;
        rocksdb::Status status = rocksdb::DB::Open(options, temp_checkpoint, &db_ptr);
        if (!status.ok()) {
            // Clean up temp dir if DB open failed
            std::filesystem::remove_all(temp_dir);
            throw std::runtime_error("Failed to open RocksDB from temp checkpoint: " +
                                   status.ToString());
        }

        db.reset(db_ptr);
        cf = db->DefaultColumnFamily();
    }

    // Constructor for creating new index (permanent location, no cleanup)
    ManagedRocksDB(rocksdb::DB *db_ptr, const std::string &db_path)
        : temp_dir(db_path), cleanup_temp_dir(false) {
        db.reset(db_ptr);
        cf = db->DefaultColumnFamily();
    }

    // Destructor: closes DB and optionally cleans up temp directory
    ~ManagedRocksDB() {
        // Close DB first (unique_ptr handles this automatically)
        db.reset();

        // Delete temp directory only if it's actually temporary
        if (cleanup_temp_dir && !temp_dir.empty() && std::filesystem::exists(temp_dir)) {
            std::filesystem::remove_all(temp_dir);
        }
    }

    // Disable copy and move to prevent resource management issues
    ManagedRocksDB(const ManagedRocksDB&) = delete;
    ManagedRocksDB& operator=(const ManagedRocksDB&) = delete;
    ManagedRocksDB(ManagedRocksDB&&) = delete;
    ManagedRocksDB& operator=(ManagedRocksDB&&) = delete;

    // Accessors
    rocksdb::DB* getDB() const { return db.get(); }
    rocksdb::ColumnFamilyHandle* getCF() const { return cf; }
    const std::string& getTempDir() const { return temp_dir; }
};

// Static managed RocksDB instance for benchmark convenience wrapper
// The destructor will automatically clean up the temp directory when:
// 1. A new benchmark run replaces this with a new instance
// 2. The program exits (static destructor is called)
static std::unique_ptr<ManagedRocksDB> managed_rocksdb;

// Helper function to create AbstractIndexInitParams from VecSimParams
static AbstractIndexInitParams NewAbstractInitParams(const VecSimParams *params) {
    const HNSWParams *hnswParams = &params->algoParams.hnswParams;

    size_t dataSize =
        VecSimParams_GetDataSize(hnswParams->type, hnswParams->dim, hnswParams->metric);
    AbstractIndexInitParams abstractInitParams = {.allocator =
                                                      VecSimAllocator::newVecsimAllocator(),
                                                  .dim = hnswParams->dim,
                                                  .vecType = hnswParams->type,
                                                  .dataSize = dataSize,
                                                  .metric = hnswParams->metric,
                                                  .blockSize = hnswParams->blockSize,
                                                  .multi = hnswParams->multi,
                                                  .logCtx = params->logCtx};
    return abstractInitParams;
}

// Helper template to create the appropriate HNSWDiskIndex based on data type
template <typename DataType, typename DistType = DataType>
inline VecSimIndex *NewDiskIndex_ChooseMultiOrSingle(std::ifstream &input,
                                                     const HNSWParams *params,
                                                     const AbstractIndexInitParams &abstractInitParams,
                                                     IndexComponents<DataType, DistType> &components,
                                                     rocksdb::DB *db,
                                                     rocksdb::ColumnFamilyHandle *cf,
                                                     Serializer::EncodingVersion version) {
    // Note: HNSWDiskIndex doesn't have separate Single/Multi variants like in-memory HNSW
    // It handles both cases internally based on params->multi
    return new (abstractInitParams.allocator) HNSWDiskIndex<DataType, DistType>(
        input, params, abstractInitParams, components, db, cf, version);
}

// Initialize @params from file (same format as regular HNSW)
static void InitializeParams(std::ifstream &source_params, HNSWParams &params) {
    Serializer::readBinaryPOD(source_params, params.dim);
    Serializer::readBinaryPOD(source_params, params.type);
    Serializer::readBinaryPOD(source_params, params.metric);
    Serializer::readBinaryPOD(source_params, params.blockSize);
    Serializer::readBinaryPOD(source_params, params.multi);
    Serializer::readBinaryPOD(source_params, params.initialCapacity);
}

std::string GetCheckpointDir(const std::string &folder_path) {
    std::filesystem::path path(folder_path);
    return (path / "rocksdb").string();
}

VecSimIndex *NewIndex(const VecSimParams *params) {
    const HNSWDiskParams *hnswDiskParams = &params->algoParams.hnswDiskParams;
    const HNSWParams *hnswParams = &params->algoParams.hnswParams;

    // Get the dbPath from params
    std::string dbPath = std::string(hnswDiskParams->dbPath);
    std::cerr << "RocksDB path: " << dbPath << std::endl;

    // Ensure the directory exists
    std::filesystem::create_directories(dbPath);

    // Open RocksDB
    rocksdb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = false;
    options.statistics = rocksdb::CreateDBStatistics();

    rocksdb::DB *db_ptr = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, dbPath, &db_ptr);
    if (!status.ok()) {
        throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
    }

    // Store in RAII wrapper (will close DB on exit, but won't delete directory)
    managed_rocksdb = std::make_unique<ManagedRocksDB>(db_ptr, dbPath);

    // Create AbstractIndexInitParams
    AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(params);

    // Create the appropriate disk index based on data type
    if (hnswParams->type == VecSimType_FLOAT32) {
        IndexComponents<float, float> indexComponents = CreateIndexComponents<float, float>(
            abstractInitParams.allocator, hnswParams->metric, abstractInitParams.dim, false);
        return new (abstractInitParams.allocator) HNSWDiskIndex<float, float>(
            hnswParams, abstractInitParams, indexComponents, managed_rocksdb->getDB(),
            managed_rocksdb->getCF(), dbPath);
    } else {
        auto bad_name = VecSimType_ToString(hnswParams->type);
        if (bad_name == nullptr) {
            bad_name = "Unknown (corrupted file?)";
        }
        throw std::runtime_error(std::string("Cannot create disk index: bad index data type: ") +
                                 bad_name);
    }
}

VecSimIndex *NewIndex(const std::string &folder_path, rocksdb::DB *db,
                     rocksdb::ColumnFamilyHandle *cf, bool is_normalized) {
    // Get the checkpoint directory path
    std::string checkpoint_dir = HNSWDiskFactory::GetCheckpointDir(folder_path);

    if (!std::filesystem::exists(checkpoint_dir)) {
        throw std::runtime_error(
            "Checkpoint directory not found: " + checkpoint_dir +
            "\nMake sure the index was saved with the checkpoint-based format.");
    }

    // Verify that the DB was opened from the correct checkpoint directory
    // Accept either the original checkpoint directory OR a temp copy (ends with /checkpoint_copy)
    std::string db_name = db->GetName();
    bool is_valid_db = (db_name == checkpoint_dir) ||
                       (db_name.size() > 16 && db_name.substr(db_name.size() - 16) == "/checkpoint_copy");

    if (!is_valid_db) {
        throw std::runtime_error(
            "RocksDB instance was not opened from a valid checkpoint location.\n"
            "Expected DB to be opened from: " + checkpoint_dir + "\n"
            "Or from a temp copy ending with: /checkpoint_copy\n"
            "But DB was opened from: " + db_name + "\n\n"
            "Please open RocksDB from the checkpoint directory (or temp copy) before loading the index.");
    }

    // Construct the index metadata file path (folder_path/index.hnsw_disk_v1)
    std::filesystem::path index_file_path = std::filesystem::path(folder_path) / "index.hnsw_disk_v1";
    std::ifstream input(index_file_path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file: " + index_file_path.string());
    }

    // Read encoding version
    Serializer::EncodingVersion version = Serializer::ReadVersion(input);

    // Read and validate algorithm type
    VecSimAlgo algo = VecSimAlgo_HNSWLIB_DISK;
    Serializer::readBinaryPOD(input, algo);
    if (algo != VecSimAlgo_HNSWLIB_DISK) {
        input.close();
        auto bad_name = VecSimAlgo_ToString(algo);
        if (bad_name == nullptr) {
            bad_name = "Unknown (corrupted file?)";
        }
        throw std::runtime_error(
            std::string("Cannot load disk index: Expected HNSW_DISK file but got algorithm type: ") +
            bad_name);
    }

    // Read index parameters from file
    HNSWParams params;
    InitializeParams(input, params);

    VecSimParams vecsimParams = {.algo = VecSimAlgo_HNSWLIB_DISK,
                                 .algoParams = {.hnswParams = HNSWParams{params}}};

    AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(&vecsimParams);

    // Create the appropriate disk index based on data type
    if (params.type == VecSimType_FLOAT32) {
        IndexComponents<float, float> indexComponents = CreateIndexComponents<float, float>(
            abstractInitParams.allocator, params.metric, abstractInitParams.dim, is_normalized);
        return NewDiskIndex_ChooseMultiOrSingle<float>(input, &params, abstractInitParams,
                                                       indexComponents, db, cf, version);
    } else {
        auto bad_name = VecSimType_ToString(params.type);
        if (bad_name == nullptr) {
            bad_name = "Unknown (corrupted file?)";
        }
        throw std::runtime_error(std::string("Cannot load disk index: bad index data type: ") +
                                 bad_name);
    }
}

VecSimIndex *NewIndex(const std::string &folder_path, bool is_normalized) {
    // Get the checkpoint directory path
    std::string checkpoint_dir = GetCheckpointDir(folder_path);

    if (!std::filesystem::exists(checkpoint_dir)) {
        throw std::runtime_error(
            "Checkpoint directory not found: " + checkpoint_dir +
            "\nMake sure the index was saved with the checkpoint-based format.");
    }

    // Create a temporary directory for the checkpoint copy
    // Using PID and timestamp to ensure uniqueness across multiple benchmark runs
    std::string temp_dir = "/tmp/hnsw_disk_benchmark_" + std::to_string(getpid()) +
                          "_" + std::to_string(std::time(nullptr));

    // Create RAII-managed RocksDB instance
    // This will:
    // 1. Copy the entire checkpoint to temp_dir/checkpoint_copy
    // 2. Open RocksDB from the temp copy (all writes go to temp location)
    // 3. Auto-cleanup temp_dir on destruction via RAII:
    //    - When a new benchmark run replaces managed_rocksdb
    //    - When the program exits (static destructor)
    managed_rocksdb = std::make_unique<ManagedRocksDB>(checkpoint_dir, temp_dir);

    // Call the main NewIndex function with the managed database
    return NewIndex(folder_path, managed_rocksdb->getDB(), managed_rocksdb->getCF(), is_normalized);
}

#endif // BUILD_TESTS

}; // namespace HNSWDiskFactory


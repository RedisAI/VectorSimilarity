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
#include <sys/wait.h>
#include <ctime>
#include <cstdlib>
#include <array>
#include <memory>
#include <random>

namespace HNSWDiskFactory {

#ifdef BUILD_TESTS

/**
 * @brief Generate a random alphanumeric string of the specified length
 * @param length The length of the string to generate
 * @return A random string containing only alphanumeric characters
 */
static std::string generate_random_string(size_t length) {
    static const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    static const size_t charset_size = sizeof(charset) - 1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, charset_size - 1);

    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result += charset[dist(gen)];
    }
    return result;
}

/**
 * @brief Check if a file is a zip archive by examining its magic bytes
 * @param file_path Path to the file to check
 * @return true if the file starts with zip magic bytes (PK\x03\x04)
 */
static bool isZipFile(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    // ZIP files start with "PK\x03\x04" (0x504B0304)
    std::array<char, 4> magic{};
    file.read(magic.data(), 4);
    return file.gcount() == 4 && magic[0] == 'P' && magic[1] == 'K' &&
           magic[2] == '\x03' && magic[3] == '\x04';
}

/**
 * @brief Extract a zip file to a target directory using the system unzip command
 * @param zip_path Path to the zip file
 * @param target_dir Directory where the zip contents will be extracted
 * @throws std::runtime_error if extraction fails
 */
static void extractZipToDirectory(const std::string &zip_path, const std::string &target_dir) {
    // Create the target directory if it doesn't exist
    std::filesystem::create_directories(target_dir);

    // Build the unzip command
    // -o: overwrite files without prompting
    // -q: quiet mode
    // -d: extract to specified directory
    std::string command = "unzip -o -q \"" + zip_path + "\" -d \"" + target_dir + "\"";

    // Execute the command
    int status = std::system(command.c_str());
    if (status != 0) {
        throw std::runtime_error("Failed to extract zip file: " + zip_path +
                                 " (exit code: " + std::to_string(status) + ")");
    }
}

// RAII wrapper to manage RocksDB database and temporary directory cleanup
class ManagedRocksDB {
private:
    std::unique_ptr<rocksdb::DB> db;
    rocksdb::ColumnFamilyHandle *cf = nullptr;
    std::string temp_dir;
    std::string extracted_folder_path;  // Path to the extracted index folder within temp_dir
    bool cleanup_temp_dir;  // Whether to delete temp_dir on destruction

    // Private constructor - use static factory methods
    ManagedRocksDB() : cleanup_temp_dir(false) {}

public:
    // Factory method for loading from a zip file (extracts to temp directory)
    static std::unique_ptr<ManagedRocksDB> fromZipFile(const std::string &zip_path,
                                                        const std::string &temp_path) {
        auto instance = std::unique_ptr<ManagedRocksDB>(new ManagedRocksDB());
        instance->temp_dir = temp_path;
        instance->extracted_folder_path = temp_path;
        instance->cleanup_temp_dir = true;

        // Create temp directory
        std::filesystem::create_directories(instance->temp_dir);

        // Extract the zip file to temp directory
        try {
            extractZipToDirectory(zip_path, instance->temp_dir);
        } catch (const std::exception &e) {
            std::filesystem::remove_all(instance->temp_dir);
            throw std::runtime_error("Failed to extract zip file: " + std::string(e.what()));
        }

        // Find the extracted folder - it should contain index.hnsw_disk_v1 and rocksdb/
        // The zip might contain the folder at root level or directly contain the files
        std::string index_file = instance->temp_dir + "/index.hnsw_disk_v1";
        std::string rocksdb_dir = instance->temp_dir + "/rocksdb";

        if (!std::filesystem::exists(index_file) || !std::filesystem::exists(rocksdb_dir)) {
            // Check if there's a single subdirectory containing the files
            for (const auto &entry : std::filesystem::directory_iterator(instance->temp_dir)) {
                if (entry.is_directory()) {
                    std::string sub_index = entry.path().string() + "/index.hnsw_disk_v1";
                    std::string sub_rocksdb = entry.path().string() + "/rocksdb";
                    if (std::filesystem::exists(sub_index) &&
                        std::filesystem::exists(sub_rocksdb)) {
                        instance->extracted_folder_path = entry.path().string();
                        break;
                    }
                }
            }
        }

        // Verify the structure exists
        if (!std::filesystem::exists(instance->extracted_folder_path + "/index.hnsw_disk_v1") ||
            !std::filesystem::exists(instance->extracted_folder_path + "/rocksdb")) {
            std::filesystem::remove_all(instance->temp_dir);
            throw std::runtime_error(
                "Invalid zip structure: expected index.hnsw_disk_v1 and rocksdb/ directory");
        }

        // Open RocksDB from the extracted checkpoint
        std::string checkpoint_dir = instance->extracted_folder_path + "/rocksdb";
        rocksdb::Options options;
        options.create_if_missing = false;
        options.error_if_exists = false;
        options.statistics = rocksdb::CreateDBStatistics();

        rocksdb::DB *db_ptr = nullptr;
        rocksdb::Status status = rocksdb::DB::Open(options, checkpoint_dir, &db_ptr);
        if (!status.ok()) {
            std::filesystem::remove_all(instance->temp_dir);
            throw std::runtime_error("Failed to open RocksDB from extracted checkpoint: " +
                                     status.ToString());
        }

        instance->db.reset(db_ptr);
        instance->cf = instance->db->DefaultColumnFamily();
        return instance;
    }

    // Factory method for loading from checkpoint directory (copies to temp location)
    static std::unique_ptr<ManagedRocksDB> fromCheckpointDir(const std::string &checkpoint_dir,
                                                              const std::string &temp_path) {
        auto instance = std::unique_ptr<ManagedRocksDB>(new ManagedRocksDB());
        instance->temp_dir = temp_path;
        instance->cleanup_temp_dir = true;

        // Create temp directory
        std::filesystem::create_directories(instance->temp_dir);

        // Copy the entire checkpoint to temp location to preserve the original
        std::string temp_checkpoint = instance->temp_dir + "/checkpoint_copy";
        try {
            std::filesystem::copy(checkpoint_dir, temp_checkpoint,
                                std::filesystem::copy_options::recursive);
        } catch (const std::filesystem::filesystem_error &e) {
            // Clean up temp dir if copy failed
            std::filesystem::remove_all(instance->temp_dir);
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
            std::filesystem::remove_all(instance->temp_dir);
            throw std::runtime_error("Failed to open RocksDB from temp checkpoint: " +
                                   status.ToString());
        }

        instance->db.reset(db_ptr);
        instance->cf = instance->db->DefaultColumnFamily();
        return instance;
    }

    // Factory method for creating new index (permanent location, no cleanup)
    static std::unique_ptr<ManagedRocksDB> fromExistingDB(rocksdb::DB *db_ptr,
                                                          const std::string &db_path) {
        auto instance = std::unique_ptr<ManagedRocksDB>(new ManagedRocksDB());
        instance->temp_dir = db_path;
        instance->cleanup_temp_dir = false;
        instance->db.reset(db_ptr);
        instance->cf = instance->db->DefaultColumnFamily();
        return instance;
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
    const std::string& getExtractedFolderPath() const { return extracted_folder_path; }
};

// Static managed RocksDB instance for benchmark convenience wrapper
// The destructor will automatically clean up the temp directory when:
// 1. A new benchmark run replaces this with a new instance
// 2. The program exits (static destructor is called)
static std::unique_ptr<ManagedRocksDB> managed_rocksdb;

// Helper function to create AbstractIndexInitParams from VecSimParams
static AbstractIndexInitParams NewAbstractInitParams(const VecSimParams *params) {
    const HNSWParams *hnswParams = &params->algoParams.hnswParams;

    size_t dataSize = hnswParams->dim * sizeof(int8_t); // Quantized storage
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
    managed_rocksdb = ManagedRocksDB::fromExistingDB(db_ptr, dbPath);

    // Create AbstractIndexInitParams
    AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(params);

    // Create the appropriate disk index based on data type
    if (hnswParams->type == VecSimType_FLOAT32) {
        IndexComponents<float, float> indexComponents = CreateQuantizedIndexComponents<float, float>(
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
        IndexComponents<float, float> indexComponents = CreateQuantizedIndexComponents<float, float>(
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
    // Create a temporary directory
    // Using PID and timestamp to ensure uniqueness across multiple benchmark runs
    std::string temp_dir = (std::filesystem::temp_directory_path() /
        ("hnsw_disk_benchmark_" + std::to_string(getpid()) + "_" +
        std::to_string(std::time(nullptr)) + "_" + generate_random_string(8))).string();

    // Check if the input is a zip file
    if (isZipFile(folder_path)) {
        // Load from zip file - extract and open RocksDB from extracted location
        managed_rocksdb = ManagedRocksDB::fromZipFile(folder_path, temp_dir);

        // Use the extracted folder path for loading the index
        std::string extracted_path = managed_rocksdb->getExtractedFolderPath();
        return NewIndex(extracted_path, managed_rocksdb->getDB(), managed_rocksdb->getCF(),
                        is_normalized);
    }

    // Not a zip file - treat as folder path (original behavior)
    std::string checkpoint_dir = GetCheckpointDir(folder_path);

    if (!std::filesystem::exists(checkpoint_dir)) {
        throw std::runtime_error(
            "Checkpoint directory not found: " + checkpoint_dir +
            "\nMake sure the index was saved with the checkpoint-based format.");
    }

    managed_rocksdb = ManagedRocksDB::fromCheckpointDir(checkpoint_dir, temp_dir);

    return NewIndex(folder_path, managed_rocksdb->getDB(), managed_rocksdb->getCF(), is_normalized);
}

#endif // BUILD_TESTS

}; // namespace HNSWDiskFactory


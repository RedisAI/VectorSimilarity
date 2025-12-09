/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr
#include <string>

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // HNSWParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator
#include "VecSim/vec_sim_index.h"
#include "rocksdb/db.h"
#include "rocksdb/statistics.h"

namespace HNSWDiskFactory {

#ifdef BUILD_TESTS

/**
 * Get the checkpoint directory path for a given index file.
 *
 * @param location Path to the index file (.hnsw_disk_v1)
 * @return std::string Path to the RocksDB checkpoint directory
 *
 * @note The checkpoint directory is named <index_stem>_rocksdb
 *       For example: "index.hnsw_disk_v1" -> "index_rocksdb"
 */
std::string GetCheckpointDir(const std::string &location);

/**
 * Factory function to load a serialized disk-based HNSW index from a file.
 *
 * @param folder_path Path to the folder containing the index
 * @param db RocksDB database instance opened from the checkpoint directory
 * @param cf RocksDB column family handle (typically db->DefaultColumnFamily())
 * @param is_normalized Whether vectors are already normalized (for Cosine metric optimization)
 * @return VecSimIndex* Pointer to the loaded HNSWDiskIndex, or throws on error
 *
 * @throws std::runtime_error if:
 *   - File cannot be opened
 *   - File has invalid/deprecated encoding version
 *   - File contains wrong algorithm type (not HNSWLIB_DISK)
 *   - File has unsupported data type
 *   - Checkpoint directory does not exist
 *
 * @note The caller is responsible for:
 *   - Opening RocksDB from the checkpoint directory (use GetCheckpointDir() to get the path)
 *   - Managing the lifetime of the RocksDB database (must outlive the index)
 *   - Deleting the returned index when done
 *
 * @example
 *   std::string checkpoint_dir = HNSWDiskFactory::GetCheckpointDir(folder_path);
 *   rocksdb::DB *db;
 *   rocksdb::Options options;
 *   rocksdb::DB::Open(options, checkpoint_dir, &db);
 *   auto *index = HNSWDiskFactory::NewIndex(folder_path, db, db->DefaultColumnFamily());
 *   // Use index...
 *   delete index;
 *   delete db;
 */
VecSimIndex *NewIndex(const std::string &folder_path, rocksdb::DB *db,
                     rocksdb::ColumnFamilyHandle *cf, bool is_normalized = false);

/**
 * Convenience wrapper to load a disk-based HNSW index with automatic database management.
 * Opens the checkpoint database and loads the index from the specified folder or zip file.
 * The original checkpoint is NEVER modified - all operations use a temporary copy.
 *
 * @param folder_path Path to the folder containing the index, OR path to a zip file
 *                    containing the index folder structure
 * @param is_normalized Whether vectors are already normalized (for Cosine metric optimization)
 * @return VecSimIndex* Pointer to the loaded HNSWDiskIndex, or throws on error
 *
 * @note ZIP FILE SUPPORT:
 *       - If folder_path points to a zip file (detected by magic bytes), it will be
 *         extracted to a temporary directory before loading
 *       - The zip file should contain:
 *           - index.hnsw_disk_v1 (index metadata file)
 *           - rocksdb/ (RocksDB checkpoint directory)
 *         Either at root level or inside a single subdirectory
 *
 * @note CHECKPOINT PRESERVATION:
 *       - For folder input: The checkpoint is copied to <temp_dir>/hnsw_disk_benchmark_<pid>_<timestamp>_<random>/
 *       - For zip input: The zip is extracted to <temp_dir>/hnsw_disk_benchmark_<pid>_<timestamp>_<random>/
 *       - <temp_dir> is the system temporary directory (std::filesystem::temp_directory_path())
 *       - All RocksDB operations use the temporary copy
 *       - The original files remain completely unchanged
 *
 * @note CLEANUP GUARANTEES:
 *       - Temporary directory is automatically cleaned up via RAII (ManagedRocksDB destructor)
 *       - Each benchmark run creates a new temp directory (using PID + timestamp for uniqueness)
 *       - Temp directories are removed when:
 *         1. A new benchmark run starts (replaces the static managed_rocksdb)
 *         2. The program exits (static destructor is called automatically)
 *
 * @note THREAD SAFETY:
 *       - This function is NOT thread-safe due to static variable usage
 *       - Intended for single-threaded benchmark scenarios only
 */
VecSimIndex *NewIndex(const std::string &folder_path, bool is_normalized = false);

VecSimIndex *NewIndex(const VecSimParams *params);

#endif

}; // namespace HNSWDiskFactory


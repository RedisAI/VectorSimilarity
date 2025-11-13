/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv3); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include "VecSim/utils/serializer.h"
#include "VecSim/containers/data_blocks_container.h"
#include "hnsw_serialization_utils.h"
#include <rocksdb/write_batch.h>
#include <rocksdb/utilities/checkpoint.h>
#include <filesystem>


/********************************** Constructor (Deserialization) **********************************/

/**
 * @brief Deserialize HNSW Disk index from file and RocksDB checkpoint
 *
 * This constructor restores an HNSW Disk index from a previously saved state.
 * The restoration process involves:
 * 1. Reading index metadata and configuration from the index file
 * 2. Restoring element metadata and label mappings
 * 3. Loading graph structure and vectors from RocksDB checkpoint (passed as db parameter)
 *
 * IMPORTANT THREAD SAFETY NOTES:
 * - This constructor assumes exclusive access to the index file and RocksDB instance
 * - The RocksDB instance must be opened from the checkpoint directory before calling this constructor
 * - No concurrent modifications should occur during deserialization
 *
 * SERIALIZATION FORMAT:
 * - All in-memory state is stored in the index file (index.hnsw_disk_v1)
 * - Graph structure and raw vectors are stored in RocksDB checkpoint
 * - Pending/staged updates are NOT serialized (must be flushed before saveIndex)
 * - Caches are NOT serialized (rebuilt on demand during queries)
 *
 * @param input Input file stream positioned after the version header
 * @param params HNSW parameters (currently unused, kept for API compatibility)
 * @param abstractInitParams Base index initialization parameters
 * @param components Index components (calculator and preprocessors)
 * @param db RocksDB database instance (must be opened from checkpoint)
 * @param cf RocksDB column family handle
 * @param version Encoding version of the serialized data
 */
template <typename DataType, typename DistType>
HNSWDiskIndex<DataType, DistType>::HNSWDiskIndex(
    std::ifstream &input, const HNSWParams *params,
    const AbstractIndexInitParams &abstractInitParams,
    const IndexComponents<DataType, DistType> &components, rocksdb::DB *db,
    rocksdb::ColumnFamilyHandle *cf, EncodingVersion version)
    : VecSimIndexAbstract<DataType, DistType>(abstractInitParams, components),
      idToMetaData(this->allocator), labelToIdMap(this->allocator), db(db), cf(cf), dbPath(""),
      indexDataGuard(), visitedNodesHandlerPool(INITIAL_CAPACITY, this->allocator),
      delta_list(), new_elements_meta_data(this->allocator), batchThreshold(0), // Will be restored from file
      pendingVectorIds(this->allocator), pendingMetadata(this->allocator),
      pendingVectorCount(0), stagedGraphUpdates(this->allocator),
      stagedNeighborUpdates(this->allocator) {

    // Restore index fields from file (including batchThreshold)
    this->restoreIndexFields(input);

    // Validate the restored fields
    this->fieldsValidation();

    // Initialize level generator with seed based on curElementCount for better distribution
    // Using curElementCount ensures different sequences for indexes with different sizes
    // Add a constant offset to avoid seed=0 for empty indexes
    this->levelGenerator.seed(200 + this->curElementCount);

    // Restore graph and vectors from file
    this->restoreGraph(input, version);
    this->restoreVectors(version);
}

/**
 * @brief Restore vectors from RocksDB checkpoint to this->vectors
 *
 * This method reconstructs the in-memory processed vectors container from the raw vectors
 * stored in RocksDB. During normal operation, vectors are stored in two forms:
 * 1. Raw vectors embedded in level-0 graph values in RocksDB
 * 2. Processed vectors in the this->vectors container for fast distance calculations
 *
 * During deserialization, we need to:
 * 1. Iterate through all elements (0 to curElementCount-1)
 * 2. For each element, retrieve the raw vector from RocksDB (level-0 graph value)
 * 3. Preprocess the raw vector to get the storage blob
 * 4. Add the processed vector to this->vectors container
 *
 * This ensures that getDataByInternalId() can retrieve processed vectors during queries.
 *
 * @param version Encoding version (currently unused, for future compatibility)
 */
template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::restoreVectors(EncodingVersion version) {
    // Iterate through all elements and restore their processed vectors
    for (idType id = 0; id < this->curElementCount; id++) {
        // Retrieve the raw vector from RocksDB (stored in level-0 graph value)
        GraphKey graphKey(id, 0);
        std::string level0_graph_value;
        rocksdb::Status status = this->db->Get(rocksdb::ReadOptions(), this->cf,
                                               graphKey.asSlice(), &level0_graph_value);

        if (!status.ok()) {
            throw std::runtime_error("Failed to retrieve vector for id " + std::to_string(id) +
                                   " during deserialization: " + status.ToString());
        }

        // Extract raw vector data from the graph value
        // Format: [raw_vector_data][neighbor_count][neighbor_ids...]
        const void* raw_vector_data = this->getVectorFromGraphValue(level0_graph_value);
        if (raw_vector_data == nullptr) {
            throw std::runtime_error("Invalid graph value format for id " + std::to_string(id) +
                                   " during deserialization");
        }

        // Copy the raw vector data to ensure it remains valid during preprocessing
        // This is necessary because level0_graph_value will be destroyed at the end of this iteration
        std::vector<char> raw_vector_copy(this->inputBlobSize);
        std::memcpy(raw_vector_copy.data(), raw_vector_data, this->inputBlobSize);

        // Preprocess the copied raw vector to get the storage blob
        auto processed_blob = this->preprocessForStorage(raw_vector_copy.data());

        // Add the processed vector to the vectors container
        // The container ID should match the element ID
        size_t containerId = this->vectors->size();
        if (containerId != id) {
            throw std::runtime_error("Container ID mismatch during deserialization: expected " +
                                   std::to_string(id) + ", got " + std::to_string(containerId));
        }
        this->vectors->addElement(processed_blob.get(), containerId);
    }

    this->log(VecSimCommonStrings::LOG_VERBOSE_STRING,
             "Restored %zu processed vectors from RocksDB checkpoint", this->curElementCount);
}

/**
 * @brief Internal implementation of index serialization
 *
 * This method performs the actual serialization of the index state to a file.
 * It is called by saveIndex() after setting up the output file.
 *
 * CRITICAL REQUIREMENTS:
 * 1. All pending updates MUST be flushed to RocksDB before serialization
 * 2. No concurrent modifications should occur during this operation
 * 3. The index must be in a consistent state (all batches processed)
 *
 * WHAT IS SERIALIZED:
 * - Index configuration (M, efConstruction, ef, epsilon, etc.)
 * - Index state (curElementCount, maxLevel, entrypointNode, etc.)
 * - Element metadata (labels, topLevels, flags)
 * - Label-to-ID mappings
 * - Batch processing configuration (batchThreshold)
 *
 * WHAT IS NOT SERIALIZED (stored in RocksDB checkpoint):
 * - Graph structure (neighbor lists)
 * - Raw vector data
 *
 * WHAT IS NOT SERIALIZED (runtime-only state):
 * - Pending batches (must be empty after flush)
 * - Staged updates (must be empty after flush)
 * - Caches (rebuilt on demand)
 * - Thread synchronization primitives
 *
 * @param output Output file stream for writing serialized data
 * @throws std::runtime_error if any pending state is not empty after flushing
 */
template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::saveIndexIMP(std::ofstream &output) {
    // Flush any pending updates before saving to ensure consistent snapshot
    this->flushStagedUpdates();
    this->flushBatch();

    // Verify that all pending state has been flushed
    // These assertions ensure data integrity during serialization
    if (!pendingVectorIds.empty()) {
        throw std::runtime_error("Serialization error: pendingVectorIds not empty after flush");
    }
    if (!stagedGraphUpdates.empty()) {
        throw std::runtime_error("Serialization error: stagedGraphUpdates not empty after flush");
    }
    if (!stagedNeighborUpdates.empty()) {
        throw std::runtime_error("Serialization error: stagedNeighborUpdates not empty after flush");
    }
    if (!rawVectorsInRAM.empty()) {
        throw std::runtime_error("Serialization error: rawVectorsInRAM not empty after flush");
    }
    if (this->vectors->size() != 0) {
        throw std::runtime_error("Serialization error: vectors container not empty after flush");
    }
    if (pendingVectorCount != 0) {
        throw std::runtime_error("Serialization error: pendingVectorCount not zero after flush");
    }
    // Note: delta_list and new_elements_meta_data are currently unused legacy variables
    // but we verify them for future-proofing
    if (!delta_list.empty()) {
        throw std::runtime_error("Serialization error: delta_list not empty after flush");
    }
    if (!new_elements_meta_data.empty()) {
        throw std::runtime_error("Serialization error: new_elements_meta_data not empty after flush");
    }

    // Save index metadata and graph (in-memory data only)
    this->saveIndexFields(output);
    this->saveGraph(output);
}

template <typename DataType, typename DistType>
std::string HNSWDiskIndex<DataType, DistType>::getCheckpointDir(const std::string &location) {
    // If location is a directory, return location/rocksdb
    // If location is a file, return parent_dir/rocksdb
    std::filesystem::path path(location);
    if (std::filesystem::is_directory(path)) {
        return (path / "rocksdb").string();
    } else {
        return (path.parent_path() / "rocksdb").string();
    }
}

/**
 * @brief Save the HNSW Disk index to a location
 *
 * This method saves the complete index state to disk using a two-part approach:
 * 1. In-memory metadata is saved to an index file (index.hnsw_disk_v1)
 * 2. Graph structure and vectors are saved via RocksDB checkpoint
 *
 * THREAD SAFETY:
 * - This method is NOT thread-safe and requires exclusive access to the index
 * - Caller must ensure no concurrent addVector/deleteVector operations occur
 * - The method will flush all pending updates before serialization
 *
 * LOCATION PARAMETER:
 * - If location is a directory (or doesn't exist with no extension):
 *   Creates: location/index.hnsw_disk_v1 and location/rocksdb/
 * - If location is a file path:
 *   Creates: location (metadata) and parent_dir/rocksdb/ (checkpoint)
 *
 * SERIALIZATION GUARANTEES:
 * - All pending batches are flushed to RocksDB before creating checkpoint
 * - All staged updates are written to RocksDB before creating checkpoint
 * - The resulting checkpoint + metadata file contain complete index state
 * - No data loss occurs if the process completes successfully
 *
 * ERROR HANDLING:
 * - Throws std::runtime_error if file cannot be opened
 * - Throws std::runtime_error if RocksDB checkpoint creation fails
 * - Throws std::runtime_error if any pending state remains after flushing
 *
 * @param location Directory or file path where index should be saved
 * @throws std::runtime_error on serialization errors
 */
template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::saveIndex(const std::string &location) {
    // Override Serializer::saveIndex to use checkpoint-based approach

    // Determine if location is a directory or file path
    std::filesystem::path path(location);
    std::string metadata_file;

    if (std::filesystem::is_directory(path) || (!std::filesystem::exists(path) && path.extension().empty())) {
        // Location is a directory - create index.hnsw_disk_v1 inside it
        std::filesystem::create_directories(path);
        metadata_file = (path / "index.hnsw_disk_v1").string();
    } else {
        // Location is a file path - use it directly
        metadata_file = location;
    }

    // First, save the metadata file
    std::ofstream output(metadata_file, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + metadata_file);
    }

    // Write version
    EncodingVersion version = EncodingVersion_V4;
    Serializer::writeBinaryPOD(output, version);

    // Save in-memory metadata
    saveIndexIMP(output);
    output.close();

    // Create checkpoint directory path
    std::string checkpoint_dir = getCheckpointDir(metadata_file);

    // Remove existing checkpoint directory if it exists
    if (std::filesystem::exists(checkpoint_dir)) {
        std::filesystem::remove_all(checkpoint_dir);
    }

    // Create RocksDB checkpoint
    rocksdb::Checkpoint* checkpoint_ptr;
    rocksdb::Status status = rocksdb::Checkpoint::Create(this->db, &checkpoint_ptr);
    if (!status.ok()) {
        throw std::runtime_error("Failed to create checkpoint object: " + status.ToString());
    }
    std::unique_ptr<rocksdb::Checkpoint> checkpoint(checkpoint_ptr);

    status = checkpoint->CreateCheckpoint(checkpoint_dir);
    if (!status.ok()) {
        throw std::runtime_error("Failed to create checkpoint: " + status.ToString());
    }
}

/********************************** Validation **********************************/

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::fieldsValidation() const {
    if (this->M > UINT16_MAX / 2)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    if (this->M <= 1)
        throw std::runtime_error("HNSW index parameter M cannot be 1 or 0");
    // if (this->M0 != this->M * 2)
    //     throw std::runtime_error("HNSW index parameter M0 should be 2*M");
}

template <typename DataType, typename DistType>
HNSWIndexMetaData HNSWDiskIndex<DataType, DistType>::checkIntegrity() const {
    HNSWIndexMetaData res = {.valid_state = false,
                             .memory_usage = -1,
                             .double_connections = HNSW_INVALID_META_DATA,
                             .unidirectional_connections = HNSW_INVALID_META_DATA,
                             .min_in_degree = HNSW_INVALID_META_DATA,
                             .max_in_degree = HNSW_INVALID_META_DATA,
                             .connections_to_repair = 0};

    // Save current memory usage
    res.memory_usage = this->getAllocationSize();

    // Track connections
    size_t connections_checked = 0;
    size_t double_connections = 0;
    size_t num_deleted = 0;
    size_t max_level_in_graph = 0;

    // Build in-degree map: node_id -> level -> in_degree_count
    std::unordered_map<idType, std::unordered_map<size_t, size_t>> inbound_connections_num;

    // First pass: count deleted and max level
    for (idType id = 0; id < this->curElementCount; id++) {
        if (this->isMarkedDeleted(id)) {
            num_deleted++;
        }
        if (this->idToMetaData[id].topLevel > max_level_in_graph) {
            max_level_in_graph = this->idToMetaData[id].topLevel;
        }
    }

    // Validate deleted count
    if (num_deleted != this->numMarkedDeleted) {
        return res;
    }

    // Second pass: validate graph connections
    auto readOptions = rocksdb::ReadOptions();
    readOptions.fill_cache = false;
    readOptions.prefix_same_as_start = true;

    auto it = this->db->NewIterator(readOptions, this->cf);

    for (it->Seek(GraphKeyPrefix); it->Valid() && it->key().starts_with(GraphKeyPrefix);
         it->Next()) {
        // Parse GraphKey from key
        const char *keyData = it->key().data() + 3; // Skip "GK\0" prefix
        const GraphKey *gk = reinterpret_cast<const GraphKey *>(keyData);

        auto neighborsData = it->value();
        size_t numNeighbors = neighborsData.size() / sizeof(idType);
        const idType *neighbors = reinterpret_cast<const idType *>(neighborsData.data());

        std::unordered_set<idType> uniqueNeighbors;

        for (size_t i = 0; i < numNeighbors; i++) {
            idType neighborId = neighbors[i];

            // Check for invalid neighbor
            if (neighborId >= this->curElementCount || neighborId == gk->id) {
                delete it;
                return res; // Invalid state
            }

            // Check for duplicate neighbors
            if (!uniqueNeighbors.insert(neighborId).second) {
                delete it;
                return res; // Duplicate neighbor found
            }

            // Count connections to deleted nodes
            if (this->isMarkedDeleted(neighborId)) {
                res.connections_to_repair++;
            }

            // Track in-degree
            inbound_connections_num[neighborId][gk->level]++;
            connections_checked++;

            // Check if connection is bidirectional
            GraphKey reverseKey(neighborId, gk->level);
            std::string reverseNeighborsData;
            auto status = this->db->Get(readOptions, this->cf, reverseKey.asSlice(),
                                       &reverseNeighborsData);

            if (status.ok()) {
                size_t reverseNumNeighbors = reverseNeighborsData.size() / sizeof(idType);
                const idType *reverseNeighbors =
                    reinterpret_cast<const idType *>(reverseNeighborsData.data());

                for (size_t j = 0; j < reverseNumNeighbors; j++) {
                    if (reverseNeighbors[j] == gk->id) {
                        double_connections++;
                        break;
                    }
                }
            }
        }
    }

    delete it;

    // Calculate min/max in-degree
    size_t min_in_degree = SIZE_MAX;
    size_t max_in_degree = 0;

    for (const auto &[id, levelMap] : inbound_connections_num) {
        for (const auto &[level, count] : levelMap) {
            if (count > max_in_degree)
                max_in_degree = count;
            if (count < min_in_degree)
                min_in_degree = count;
        }
    }

    // If no connections found, set min to 0
    if (min_in_degree == SIZE_MAX) {
        min_in_degree = 0;
    }

    res.double_connections = double_connections;
    res.unidirectional_connections = 0; // Disk index doesn't track unidirectional edges separately
    res.min_in_degree = min_in_degree;
    res.max_in_degree = max_in_degree;
    res.valid_state = true;

    return res;
}


template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::restoreIndexFields(std::ifstream &input) {

    // Restore HNSW build parameters
    Serializer::readBinaryPOD(input, this->M);
    Serializer::readBinaryPOD(input, this->M0);
    Serializer::readBinaryPOD(input, this->efConstruction);

    // Restore HNSW search parameters
    Serializer::readBinaryPOD(input, this->ef);
    Serializer::readBinaryPOD(input, this->epsilon);

    // Restore index metadata
    Serializer::readBinaryPOD(input, this->mult);

    // Restore index state
    Serializer::readBinaryPOD(input, this->curElementCount);
    Serializer::readBinaryPOD(input, this->numMarkedDeleted);
    Serializer::readBinaryPOD(input, this->maxLevel);
    Serializer::readBinaryPOD(input, this->entrypointNode);

    // Restore batch processing configuration
    Serializer::readBinaryPOD(input, this->batchThreshold);

    // Restore dbPath (string: length + data)
    size_t dbPathLength;
    Serializer::readBinaryPOD(input, dbPathLength);
    this->dbPath.resize(dbPathLength);
    input.read(&this->dbPath[0], dbPathLength);
}

/**
 * @brief Restore graph structure and element metadata from serialized file
 *
 * This method is called during deserialization to restore the in-memory portions
 * of the index state. It reads element metadata and label mappings from the file,
 * while graph structure and vectors are loaded from the RocksDB checkpoint.
 *
 * RESTORATION PHASES:
 * Phase 1: Restore element metadata (label, topLevel, flags) for all elements
 * Phase 2: Restore label-to-ID mapping
 * Phase 3: Graph structure is already in RocksDB (loaded from checkpoint)
 * Phase 4: Vectors are already in RocksDB (loaded on-demand during queries)
 *
 * MEMORY MANAGEMENT:
 * - Element metadata is loaded into RAM (idToMetaData vector)
 * - Label mappings are loaded into RAM (labelToIdMap hash map)
 * - Vectors are NOT pre-loaded (loaded on-demand from RocksDB)
 * - Graph structure is NOT pre-loaded (accessed from RocksDB as needed)
 *
 * PENDING STATE INITIALIZATION:
 * - All pending/staged update structures are cleared (must be empty after deserialization)
 * - Caches are empty (will be populated on-demand during queries)
 * - Visited nodes handler pool is resized to accommodate all elements
 *
 * @param input Input file stream positioned after index fields
 * @param version Encoding version (currently unused, for future compatibility)
 */
template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::restoreGraph(std::ifstream &input,
                                                     EncodingVersion version) {
    // Phase 1: Restore metadata for all elements
    this->idToMetaData.resize(this->curElementCount);

    for (idType id = 0; id < this->curElementCount; id++) {
        labelType label;
        size_t topLevel;
        elementFlags flags;

        Serializer::readBinaryPOD(input, label);
        Serializer::readBinaryPOD(input, topLevel);
        Serializer::readBinaryPOD(input, flags);

        this->idToMetaData[id] = DiskElementMetaData(label, topLevel);
        this->idToMetaData[id].flags = flags;
    }

    // Phase 2: Restore label to id mapping
    size_t labelMapSize;
    Serializer::readBinaryPOD(input, labelMapSize);

    this->labelToIdMap.clear();
    this->labelToIdMap.reserve(labelMapSize);

    for (size_t i = 0; i < labelMapSize; i++) {
        labelType label;
        idType id;
        Serializer::readBinaryPOD(input, label);
        Serializer::readBinaryPOD(input, id);
        this->labelToIdMap[label] = id;
    }

    // Phases 3 & 4: RocksDB data is loaded from checkpoint (not from this file)
    // The checkpoint should already be loaded into the RocksDB instance
    //
    // NOTE: The current HNSW Disk implementation does NOT pre-load vectors into RAM.
    // Vectors are loaded on-demand from RocksDB during queries via getRawVector().
    // This keeps memory usage low and allows the index to handle datasets larger than RAM.
    //
    // The graph structure (neighbor lists) is stored in RocksDB with keys prefixed by "GK\0".
    // Raw vector data is embedded at the beginning of each level-0 graph value.
    // Format: [raw_vector_data][neighbor_count][neighbor_ids...]

    this->log(VecSimCommonStrings::LOG_VERBOSE_STRING,
             "RocksDB checkpoint loaded. Vectors will be loaded on-demand from disk during queries.");

    // Clear any pending state (must be empty after deserialization)
    this->pendingVectorIds.clear();
    this->pendingMetadata.clear();
    this->pendingVectorCount = 0;
    this->stagedGraphUpdates.clear();
    this->stagedNeighborUpdates.clear();

    // Resize visited nodes handler pool
    this->visitedNodesHandlerPool.resize(this->curElementCount);
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::saveIndexFields(std::ofstream &output) const {
    // Save index type
    Serializer::writeBinaryPOD(output, VecSimAlgo_HNSWLIB_DISK);

    // Save VecSimIndex fields
    Serializer::writeBinaryPOD(output, this->dim);
    Serializer::writeBinaryPOD(output, this->vecType);
    Serializer::writeBinaryPOD(output, this->metric);
    Serializer::writeBinaryPOD(output, this->blockSize);
    Serializer::writeBinaryPOD(output, this->isMulti);
    Serializer::writeBinaryPOD(output, this->curElementCount); // Use curElementCount as initial capacity

    // Save HNSW build parameters
    Serializer::writeBinaryPOD(output, this->M);
    Serializer::writeBinaryPOD(output, this->M0);
    Serializer::writeBinaryPOD(output, this->efConstruction);

    // Save HNSW search parameters
    Serializer::writeBinaryPOD(output, this->ef);
    Serializer::writeBinaryPOD(output, this->epsilon);

    // Save index metadata
    Serializer::writeBinaryPOD(output, this->mult);

    // Save index state
    Serializer::writeBinaryPOD(output, this->curElementCount);
    Serializer::writeBinaryPOD(output, this->numMarkedDeleted);
    Serializer::writeBinaryPOD(output, this->maxLevel);
    Serializer::writeBinaryPOD(output, this->entrypointNode);

    // Save batch processing configuration
    Serializer::writeBinaryPOD(output, this->batchThreshold);

    // Save dbPath (string: length + data)
    size_t dbPathLength = this->dbPath.length();
    Serializer::writeBinaryPOD(output, dbPathLength);
    output.write(this->dbPath.c_str(), dbPathLength);
}

template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::saveGraph(std::ofstream &output) const {
    // Phase 1: Save metadata for all elements
    for (idType id = 0; id < this->curElementCount; id++) {
        labelType label = this->idToMetaData[id].label;
        size_t topLevel = this->idToMetaData[id].topLevel;
        elementFlags flags = this->idToMetaData[id].flags;

        Serializer::writeBinaryPOD(output, label);
        Serializer::writeBinaryPOD(output, topLevel);
        Serializer::writeBinaryPOD(output, flags);
    }

    // Phase 2: Save label to id mapping
    size_t labelMapSize = this->labelToIdMap.size();
    Serializer::writeBinaryPOD(output, labelMapSize);

    for (const auto &[label, id] : this->labelToIdMap) {
        Serializer::writeBinaryPOD(output, label);
        Serializer::writeBinaryPOD(output, id);
    }

}

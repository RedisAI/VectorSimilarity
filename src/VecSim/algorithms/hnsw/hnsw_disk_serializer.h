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

template <typename DataType, typename DistType>
HNSWDiskIndex<DataType, DistType>::HNSWDiskIndex(
    std::ifstream &input, const HNSWParams *params,
    const AbstractIndexInitParams &abstractInitParams,
    const IndexComponents<DataType, DistType> &components, rocksdb::DB *db,
    rocksdb::ColumnFamilyHandle *cf, EncodingVersion version)
    : VecSimIndexAbstract<DataType, DistType>(abstractInitParams, components),
      idToMetaData(this->allocator), labelToIdMap(this->allocator), db(db), cf(cf), dbPath(""),
      indexDataGuard(), visitedNodesHandlerPool(INITIAL_CAPACITY, this->allocator),
      delta_list(), new_elements_meta_data(this->allocator), batchThreshold(10),
      pendingVectorIds(this->allocator), pendingMetadata(this->allocator),
      pendingVectorCount(0), stagedGraphUpdates(this->allocator),
      stagedNeighborUpdates(this->allocator) {

    // Restore index fields from file
    this->restoreIndexFields(input);

    // Validate the restored fields
    this->fieldsValidation();

    // Initialize level generator with seed (use 200 like in-memory version)
    this->levelGenerator.seed(200);

    // Restore graph and vectors from file
    this->restoreGraph(input, version);
}


template <typename DataType, typename DistType>
void HNSWDiskIndex<DataType, DistType>::saveIndexIMP(std::ofstream &output) {
    // Flush any pending updates before saving to ensure consistent snapshot
    this->flushStagedUpdates();
    this->flushBatch();

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

    // Restore dbPath (string: length + data)
    size_t dbPathLength;
    Serializer::readBinaryPOD(input, dbPathLength);
    this->dbPath.resize(dbPathLength);
    input.read(&this->dbPath[0], dbPathLength);
}

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
    // We just need to populate the in-memory vector cache from RocksDB

    auto readOptions = rocksdb::ReadOptions();
    readOptions.fill_cache = false;
    readOptions.prefix_same_as_start = true;

    auto it = this->db->NewIterator(readOptions, this->cf);
    std::vector<char> vectorData(this->dataSize);

    for (it->Seek(RawVectorKeyPrefix); it->Valid() && it->key().starts_with(RawVectorKeyPrefix);
         it->Next()) {
        // Extract id from key: "RV\0" + id
        const char *keyData = it->key().data() + 3; // Skip "RV\0" prefix
        idType id = *reinterpret_cast<const idType *>(keyData);

        // Load vector data into in-memory cache
        auto vectorValue = it->value();
        this->vectors->addElement(vectorValue.data(), id);
    }
    delete it;

    // Clear any pending state
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

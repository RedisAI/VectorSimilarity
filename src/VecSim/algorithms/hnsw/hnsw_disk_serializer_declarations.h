/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

// Serializing and tests functions.
public:
HNSWDiskIndex(std::ifstream &input, const HNSWParams *params,
              const AbstractIndexInitParams &abstractInitParams,
              const IndexComponents<DataType, DistType> &components, rocksdb::DB *db,
              rocksdb::ColumnFamilyHandle *cf, EncodingVersion version);

// Validates the connections between vectors
HNSWIndexMetaData checkIntegrity() const;

// Override saveIndex to use RocksDB checkpoint
void saveIndex(const std::string &location);

// Index memory size might be changed during index saving.
virtual void saveIndexIMP(std::ofstream &output) override;

// Get checkpoint directory path for a given index file
static std::string getCheckpointDir(const std::string &location);

// used by index factory to load nodes connections
void restoreGraph(std::ifstream &input, EncodingVersion version);

// used by index factory to restore processed vectors from RocksDB
void restoreVectors(EncodingVersion version);

private:
// Functions for index saving.
void saveIndexFields(std::ofstream &output) const;

void saveGraph(std::ofstream &output) const;

// Functions for index loading.
void restoreIndexFields(std::ifstream &input);
void fieldsValidation() const;


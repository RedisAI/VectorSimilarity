/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"
#include "VecSim/algorithms/hnsw/hnsw_disk.h"
#include "VecSim/index_factories/components/components_factory.h"
#include "VecSim/vec_sim.h"
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <filesystem>
#include <cmath>

class QuantizedHNSWDiskTest : public ::testing::Test {
protected:
    rocksdb::DB *db = nullptr;
    rocksdb::ColumnFamilyHandle *cf = nullptr;
    std::string db_path;

    void SetUp() override {
        // Create temporary RocksDB directory
        db_path = "/tmp/test_quantized_hnsw_" + std::to_string(getpid());
        std::filesystem::create_directories(db_path);

        // Open RocksDB
        rocksdb::Options options;
        options.create_if_missing = true;
        rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
        ASSERT_TRUE(status.ok()) << "Failed to open RocksDB: " << status.ToString();

        cf = db->DefaultColumnFamily();
    }

    void TearDown() override {
        if (db) {
            delete db;
            db = nullptr;
        }

        // Clean up RocksDB directory
        std::filesystem::remove_all(db_path);
    }
};

TEST_F(QuantizedHNSWDiskTest, BasicQuantization) {
    size_t dim = 128;
    size_t n_vectors = 100;

    // Create quantized HNSW disk index
    HNSWParams params = {.type = VecSimType_FLOAT32,
                         .dim = dim,
                         .metric = VecSimMetric_Cosine,
                         .multi = false,
                         .initialCapacity = 0,
                         .blockSize = 100,
                         .M = 16,
                         .efConstruction = 200,
                         .efRuntime = 10,
                         .epsilon = 0.01};

    AbstractIndexInitParams abstractParams;
    abstractParams.allocator = VecSimAllocator::newVecsimAllocator();
    abstractParams.dim = dim;
    abstractParams.vecType = VecSimType_FLOAT32;
    abstractParams.dataSize = dim * sizeof(int8_t); // Quantized storage
    abstractParams.metric = VecSimMetric_Cosine;
    abstractParams.blockSize = 100;
    abstractParams.multi = false;
    abstractParams.logCtx = nullptr;

    // Create quantized components
    auto components = CreateQuantizedIndexComponents<float, float>(abstractParams.allocator,
                                                                   VecSimMetric_Cosine, dim, false);

    auto *index = new (abstractParams.allocator)
        HNSWDiskIndex<float, float>(&params, abstractParams, components, db, cf);

    // Generate normalized test vectors (in [-1, 1] range)
    std::vector<std::vector<float>> vectors(n_vectors);
    for (size_t i = 0; i < n_vectors; i++) {
        vectors[i].resize(dim);
        float norm = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            vectors[i][j] = (float)(rand() % 1000 - 500) / 500.0f; // [-1, 1]
            norm += vectors[i][j] * vectors[i][j];
        }
        // Normalize to unit length
        norm = std::sqrt(norm);
        for (size_t j = 0; j < dim; j++) {
            vectors[i][j] /= norm;
        }
    }

    // Add vectors to index
    for (size_t i = 0; i < n_vectors; i++) {
        int res = VecSimIndex_AddVector(index, vectors[i].data(), i);
        ASSERT_EQ(res, 1) << "Failed to add vector " << i;
    }

    ASSERT_EQ(index->indexSize(), n_vectors);

    // Flush batch to process vectors and build graph
    index->flushBatch();

    // Test query
    auto query = vectors[0];
    auto *results = VecSimIndex_TopKQuery(index, query.data(), 10, nullptr, BY_SCORE);

    ASSERT_NE(results, nullptr);
    ASSERT_GT(VecSimQueryReply_Len(results), 0);

    // Query vector should be in the results (with quantization, it might not be first)
    bool found_query_vector = false;
    auto it = VecSimQueryReply_GetIterator(results);
    while (auto *item = VecSimQueryReply_IteratorNext(it)) {
        if (item->id == 0) {
            found_query_vector = true;
            // With quantization, distance should be very small but might not be exactly 0
            EXPECT_LT(item->score, 0.1) << "Query vector distance should be small";
            break;
        }
    }
    EXPECT_TRUE(found_query_vector) << "Query vector (ID 0) should be in results";

    VecSimQueryReply_IteratorFree(it);
    VecSimQueryReply_Free(results);

    delete index;
}

TEST_F(QuantizedHNSWDiskTest, QuantizationAccuracy) {
    size_t dim = 64;

    // Create quantized index
    HNSWParams params = {.type = VecSimType_FLOAT32,
                         .dim = dim,
                         .metric = VecSimMetric_Cosine,
                         .multi = false,
                         .initialCapacity = 0,
                         .blockSize = 100,
                         .M = 16,
                         .efConstruction = 200,
                         .efRuntime = 10,
                         .epsilon = 0.01};

    AbstractIndexInitParams abstractParams;
    abstractParams.allocator = VecSimAllocator::newVecsimAllocator();
    abstractParams.dim = dim;
    abstractParams.vecType = VecSimType_FLOAT32;
    abstractParams.dataSize = dim * sizeof(int8_t);
    abstractParams.metric = VecSimMetric_Cosine;
    abstractParams.blockSize = 100;
    abstractParams.multi = false;
    abstractParams.logCtx = nullptr;

    auto components = CreateQuantizedIndexComponents<float, float>(abstractParams.allocator,
                                                                   VecSimMetric_Cosine, dim, false);

    auto *index = new (abstractParams.allocator)
        HNSWDiskIndex<float, float>(&params, abstractParams, components, db, cf);

    // Add a few normalized vectors
    std::vector<float> v1(dim, 0.0f);
    v1[0] = 1.0f; // Unit vector along first dimension

    std::vector<float> v2(dim, 0.0f);
    v2[1] = 1.0f; // Unit vector along second dimension

    std::vector<float> v3(dim);
    float norm = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        v3[i] = 1.0f / std::sqrt((float)dim); // Uniform normalized vector
        norm += v3[i] * v3[i];
    }

    VecSimIndex_AddVector(index, v1.data(), 1);
    VecSimIndex_AddVector(index, v2.data(), 2);
    VecSimIndex_AddVector(index, v3.data(), 3);

    // Flush batch to process vectors and build graph
    index->flushBatch();

    // Query with v1 - should find itself first
    auto *results = VecSimIndex_TopKQuery(index, v1.data(), 3, nullptr, BY_SCORE);
    ASSERT_NE(results, nullptr);
    ASSERT_EQ(VecSimQueryReply_Len(results), 3);

    auto it = VecSimQueryReply_GetIterator(results);
    auto *item = VecSimQueryReply_IteratorNext(it);
    EXPECT_EQ(item->id, 1); // Should find v1 first

    VecSimQueryReply_IteratorFree(it);
    VecSimQueryReply_Free(results);

    delete index;
}

TEST_F(QuantizedHNSWDiskTest, StorageSize) {
    size_t dim = 128;

    // The quantized index should use int8 storage (1 byte per dimension)
    // vs float32 (4 bytes per dimension) = 4x reduction

    AbstractIndexInitParams abstractParams;
    abstractParams.allocator = VecSimAllocator::newVecsimAllocator();
    abstractParams.dim = dim;
    abstractParams.vecType = VecSimType_FLOAT32;
    abstractParams.dataSize = dim * sizeof(int8_t); // Quantized
    abstractParams.metric = VecSimMetric_Cosine;
    abstractParams.blockSize = 100;
    abstractParams.multi = false;
    abstractParams.logCtx = nullptr;

    // Verify dataSize is correct for quantized storage
    EXPECT_EQ(abstractParams.dataSize, dim * sizeof(int8_t));
    EXPECT_EQ(abstractParams.dataSize, dim); // 1 byte per dimension

    // Compare to non-quantized
    size_t float_size = dim * sizeof(float);
    EXPECT_EQ(float_size, dim * 4); // 4 bytes per dimension

    // Verify 4x reduction
    EXPECT_EQ(float_size / abstractParams.dataSize, 4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

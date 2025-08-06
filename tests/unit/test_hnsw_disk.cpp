#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <random>
#include <filesystem>
#include <rocksdb/db.h>
#include <rocksdb/options.h>

#include "VecSim/algorithms/hnsw/hnsw_disk.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/index_factories/components/components_factory.h"
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/memory/vecsim_malloc.h"

#include "unit_test_utils.h"

using namespace std;

class HNSWDiskIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for RocksDB
        temp_dir = "/tmp/hnsw_disk_test_" + std::to_string(getpid());
        
        // Ensure the directory exists
        std::filesystem::create_directories(temp_dir);
        
        rocksdb::Options options;
        options.create_if_missing = true;
        options.error_if_exists = false;
        
        rocksdb::Status status = rocksdb::DB::Open(options, temp_dir, &db);
        if (!status.ok()) {
            std::cerr << "Failed to open RocksDB: " << status.ToString() << std::endl;
            std::cerr << "Directory: " << temp_dir << std::endl;
            std::cerr << "Directory exists: " << std::filesystem::exists(temp_dir) << std::endl;
            std::cerr << "Directory is directory: " << std::filesystem::is_directory(temp_dir) << std::endl;
        }
        ASSERT_TRUE(status.ok()) << "Failed to open RocksDB database: " << status.ToString();
    }

    void TearDown() override {
        if (db) {
            delete db;
        }
        // Clean up temporary directory
        try {
            std::filesystem::remove_all(temp_dir);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to remove temp directory: " << e.what() << std::endl;
        }
    }

    // Helper function to create test vectors
    std::vector<float> createRandomVector(size_t dim, std::mt19937& rng) {
        std::vector<float> vec(dim);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < dim; ++i) {
            vec[i] = dist(rng);
        }
        return vec;
    }

    // Helper function to normalize a vector (for cosine similarity)
    void normalizeVector(std::vector<float>& vec) {
        float norm = 0.0f;
        for (float val : vec) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float& val : vec) {
                val /= norm;
            }
        }
    }

    std::string temp_dir;
    rocksdb::DB* db = nullptr;
};

TEST_F(HNSWDiskIndexTest, BasicConstruction) {
    // Test basic construction of HNSWDiskIndex
    const size_t dim = 128;
    const size_t maxElements = 1000;
    
    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 16;
    params.efConstruction = 200;
    params.efRuntime = 100;
    params.epsilon = 0.01;
    
    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
    // Basic assertions - check that the index was created successfully
    // Note: We can't access protected members directly, so we'll test through public methods
    EXPECT_TRUE(&index != nullptr);
    
    // Clean up - shared_ptr will handle deallocation automatically
}

// TEST_F(HNSWDiskIndexTest, VectorOperations) {
//     // Test adding and querying vectors
//     const size_t dim = 64;
//     const size_t maxElements = 100;
    
//     // Create HNSW parameters
//     HNSWParams params;
//     params.dim = dim;
//     params.type = VecSimType_FLOAT32;
//     params.metric = VecSimMetric_L2;
//     params.multi = false;
//     params.M = 8;
//     params.efConstruction = 100;
//     params.efRuntime = 50;
//     params.epsilon = 0.01;
    
//     // Create abstract init parameters
//     AbstractIndexInitParams abstractInitParams;
//     abstractInitParams.dim = dim;
//     abstractInitParams.dataSize = dim * sizeof(float);
//     abstractInitParams.multi = false;
//     abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
//     // Create index components
//     IndexComponents<float, float> components = CreateIndexComponents<float, float>(
//         abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
//     // Create HNSWDiskIndex - use default column family handle
//     rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
//     HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
//     // Create test vectors
//     std::mt19937 rng(42);
//     std::vector<std::vector<float>> vectors;
//     std::vector<std::pair<labelType, const void*>> elements;
    
//     for (size_t i = 0; i < 10; ++i) {
//         auto vec = createRandomVector(dim, rng);
//         normalizeVector(vec);
//         vectors.push_back(vec);
//         elements.emplace_back(i, vectors.back().data());
//     }
    
//     // Add vectors to index
//     std::vector<labelType> deleted_labels;
//     index.batchUpdate(elements, deleted_labels);
    
//     // Query the index
//     auto query_vec = createRandomVector(dim, rng);
//     normalizeVector(query_vec);
    
//     VecSimQueryParams queryParams;
//     queryParams.hnswRuntimeParams.efRuntime = 50;
    
//     auto results = index.topKQuery(query_vec.data(), 5, &queryParams);
    
//     // Basic assertions
//     EXPECT_TRUE(results != nullptr);
//     EXPECT_LE(results->results.size(), 5);
    
//     // Clean up
//     delete results;
//     // shared_ptr will handle deallocation automatically
// }

// TEST_F(HNSWDiskIndexTest, BatchOperations) {
//     // Test batch add and delete operations
//     const size_t dim = 32;
//     const size_t maxElements = 50;
    
//     // Create HNSW parameters
//     HNSWParams params;
//     params.dim = dim;
//     params.type = VecSimType_FLOAT32;
//     params.metric = VecSimMetric_L2;
//     params.multi = false;
//     params.M = 4;
//     params.efConstruction = 50;
//     params.efRuntime = 25;
//     params.epsilon = 0.01;
    
//     // Create abstract init parameters
//     AbstractIndexInitParams abstractInitParams;
//     abstractInitParams.dim = dim;
//     abstractInitParams.dataSize = dim * sizeof(float);
//     abstractInitParams.multi = false;
//     abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
//     // Create index components
//     IndexComponents<float, float> components = CreateIndexComponents<float, float>(
//         abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
//     // Create HNSWDiskIndex - use default column family handle
//     rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
//     HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
//     // Create test vectors
//     std::mt19937 rng(42);
//     std::vector<std::vector<float>> vectors;
//     std::vector<std::pair<labelType, const void*>> elements;
    
//     for (size_t i = 0; i < 20; ++i) {
//         auto vec = createRandomVector(dim, rng);
//         normalizeVector(vec);
//         vectors.push_back(vec);
//         elements.emplace_back(i, vectors.back().data());
//     }
    
//     // Add vectors to index
//     std::vector<labelType> deleted_labels;
//     index.batchUpdate(elements, deleted_labels);
    
//     // Delete some vectors
//     std::vector<labelType> to_delete = {5, 10, 15};
//     std::vector<std::pair<labelType, const void*>> empty_elements;
//     index.batchUpdate(empty_elements, to_delete);
    
//     // Query the index
//     auto query_vec = createRandomVector(dim, rng);
//     normalizeVector(query_vec);
    
//     VecSimQueryParams queryParams;
//     queryParams.hnswRuntimeParams.efRuntime = 25;
    
//     auto results = index.topKQuery(query_vec.data(), 3, &queryParams);
    
//     // Basic assertions
//     EXPECT_TRUE(results != nullptr);
//     EXPECT_LE(results->results.size(), 3);
    
//     // Clean up
//     delete results;
//     // shared_ptr will handle deallocation automatically
// }

// TEST_F(HNSWDiskIndexTest, EmptyIndexQuery) {
//     // Test querying an empty index
//     const size_t dim = 16;
//     const size_t maxElements = 10;
    
//     // Create HNSW parameters
//     HNSWParams params;
//     params.dim = dim;
//     params.type = VecSimType_FLOAT32;
//     params.metric = VecSimMetric_L2;
//     params.multi = false;
//     params.M = 4;
//     params.efConstruction = 20;
//     params.efRuntime = 10;
//     params.epsilon = 0.01;
    
//     // Create abstract init parameters
//     AbstractIndexInitParams abstractInitParams;
//     abstractInitParams.dim = dim;
//     abstractInitParams.dataSize = dim * sizeof(float);
//     abstractInitParams.multi = false;
//     abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
//     // Create index components
//     IndexComponents<float, float> components = CreateIndexComponents<float, float>(
//         abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
//     // Create HNSWDiskIndex - use default column family handle
//     rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
//     HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
//     // Query empty index
//     std::mt19937 rng(42);
//     auto query_vec = createRandomVector(dim, rng);
//     normalizeVector(query_vec);
    
//     VecSimQueryParams queryParams;
//     queryParams.hnswRuntimeParams.efRuntime = 10;
    
//     auto results = index.topKQuery(query_vec.data(), 5, &queryParams);
    
//     // Should return empty results
//     EXPECT_TRUE(results != nullptr);
//     EXPECT_EQ(results->results.size(), 0);
    
//     // Clean up
//     delete results;
//     // shared_ptr will handle deallocation automatically
// }

// TEST_F(HNSWDiskIndexTest, RocksDBIntegration) {
//     // Test RocksDB integration specifically
//     const size_t dim = 64;
    
//     // Create HNSW parameters
//     HNSWParams params;
//     params.dim = dim;
//     params.type = VecSimType_FLOAT32;
//     params.metric = VecSimMetric_L2;
//     params.multi = false;
//     params.M = 8;
//     params.efConstruction = 100;
//     params.efRuntime = 50;
//     params.epsilon = 0.01;
    
//     // Create abstract init parameters
//     AbstractIndexInitParams abstractInitParams;
//     abstractInitParams.dim = dim;
//     abstractInitParams.vecType = params.type;
//     abstractInitParams.dataSize = dim * sizeof(float);
//     abstractInitParams.multi = false;
//     abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
//     // Create index components
//     IndexComponents<float, float> components = CreateIndexComponents<float, float>(
//         abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
//     // Create HNSWDiskIndex - use default column family handle
//     rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
//     HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
//     // Test that RocksDB operations work through the index
//     std::mt19937 rng(42);
//     auto vec = createRandomVector(dim, rng);
//     normalizeVector(vec);
    
//     // Add a single vector
//     std::vector<std::pair<labelType, const void*>> elements = {{1, vec.data()}};
//     std::vector<labelType> deleted_labels;
//     index.batchUpdate(elements, deleted_labels);
    
//     // Query the index
//     VecSimQueryParams queryParams;
//     queryParams.hnswRuntimeParams.efRuntime = 50;
    
//     auto results = index.topKQuery(vec.data(), 1, &queryParams);
    
//     // Should find the vector we just added
//     EXPECT_TRUE(results != nullptr);
//     EXPECT_GE(results->results.size(), 0);
    
//     // Clean up
//     delete results;
//     // shared_ptr will handle deallocation automatically
// }

TEST_F(HNSWDiskIndexTest, SimpleTest) {
    // Simple test to isolate the issue
    const size_t dim = 64;
    
    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 8;
    params.efConstruction = 100;
    params.efRuntime = 50;
    params.epsilon = 0.01;
    
    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
    // Just test that the index was created successfully
    EXPECT_TRUE(&index != nullptr);
    
    // Test basic properties
    EXPECT_EQ(index.indexSize(), 0);
    EXPECT_EQ(index.indexLabelCount(), 0);
}

TEST_F(HNSWDiskIndexTest, BasicStoreVectorTest) {
    // Test just the basic storeVector functionality without RocksDB
    const size_t dim = 64;
    
    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 8;
    params.efConstruction = 100;
    params.efRuntime = 50;
    params.epsilon = 0.01;
    
    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
    // Create a test vector
    std::mt19937 rng(42);
    auto test_vector = createRandomVector(dim, rng);
    normalizeVector(test_vector);
    
    // Test that we can access basic properties
    EXPECT_EQ(index.indexSize(), 0);
    EXPECT_EQ(index.indexLabelCount(), 0);
    
    // Test that we can call getRandomLevel
    double mult = 1.0 / log(1.0 * params.M);
    size_t level = index.getRandomLevel(mult);
    EXPECT_GE(level, 0);
    
    // Test that we can access the vector data
    EXPECT_EQ(index.getDim(), dim);
    EXPECT_EQ(index.getDataSize(), dim * sizeof(float));
}

TEST_F(HNSWDiskIndexTest, StoreVectorTest) {
    // Test just the storeVector method
    const size_t dim = 64;
    
    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 8;
    params.efConstruction = 100;
    params.efRuntime = 50;
    params.epsilon = 0.01;
    
    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
    // Create a test vector
    std::mt19937 rng(42);
    auto test_vector = createRandomVector(dim, rng);
    normalizeVector(test_vector);
    
    // Test storeVector method
    labelType label = 100;
    HNSWAddVectorState state = index.storeVector(test_vector.data(), label);
    
    // Verify the state
    EXPECT_EQ(state.newElementId, 0); // Should be the first element
    EXPECT_GE(state.elementMaxLevel, 0); // Should have a valid level
    EXPECT_EQ(state.currEntryPoint, INVALID_ID); // Should be invalid initially
    EXPECT_EQ(state.currMaxLevel, HNSW_INVALID_LEVEL); // Should be invalid initially
    
    // Test that the index size increased
    EXPECT_EQ(index.indexSize(), 1);
    EXPECT_EQ(index.indexLabelCount(), 1);
}

TEST_F(HNSWDiskIndexTest, SimpleAddVectorTest) {
    // Test just the addVector method
    const size_t dim = 64;
    
    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 8;
    params.efConstruction = 100;
    params.efRuntime = 50;
    params.epsilon = 0.01;
    
    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1; // Use small block size for testing
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
    // Create a test vector
    std::mt19937 rng(42);
    auto test_vector = createRandomVector(dim, rng);
    normalizeVector(test_vector);
    
    // Test addVector method
    labelType label = 100;
    int result = index.addVector(test_vector.data(), label);
    EXPECT_EQ(result, 1); // Should return success
    
    // Verify the index size increased
    EXPECT_EQ(index.indexSize(), 1);
    EXPECT_EQ(index.indexLabelCount(), 1);
}

TEST_F(HNSWDiskIndexTest, AddVectorTest) {
    // Test adding a single vector using the separated approach
    const size_t dim = 64;
    
    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 8;
    params.efConstruction = 100;
    params.efRuntime = 50;
    params.epsilon = 0.01;
    
    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1; // Use small block size for testing
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
    // Create a test vector
    std::mt19937 rng(42);
    auto test_vector = createRandomVector(dim, rng);
    normalizeVector(test_vector);
    
    // Test 1: Add vector using appendVector (combined approach)
    labelType label1 = 100;
    int result1 = index.addVector(test_vector.data(), label1);
    EXPECT_EQ(result1, 1); // Should return success
    
    // Test 2: Add vector using batching approach
    labelType label2 = 200;
    auto test_vector2 = createRandomVector(dim, rng);
    normalizeVector(test_vector2);
    
    // Add vector to batch (should be stored in memory)
    index.addVector(test_vector2.data(), label2);
    
    // Force flush the batch to disk
    index.flushBatch();
    
    // Test 3: Query to verify both vectors were added correctly
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 50;
    
    auto results = index.topKQuery(test_vector.data(), 2, &queryParams);
    
    EXPECT_TRUE(results != nullptr);
    EXPECT_EQ(results->code, VecSim_OK);
    EXPECT_GE(results->results.size(), 1); // Should find at least the first vector
    
    // Verify that both labels are in the results
    bool found_label1 = false;
    bool found_label2 = false;
    for (const auto& result : results->results) {
        if (result.id == label1) found_label1 = true;
        if (result.id == label2) found_label2 = true;
    }
    
    EXPECT_TRUE(found_label1) << "Label " << label1 << " not found in results";
    EXPECT_TRUE(found_label2) << "Label " << label2 << " not found in results";
    
    // Test 4: Verify index size
    EXPECT_EQ(index.indexSize(), 2);
    EXPECT_EQ(index.indexLabelCount(), 2);
    
    // Test 5: Test with a third vector using the combined approach
    labelType label3 = 300;
    auto test_vector3 = createRandomVector(dim, rng);
    normalizeVector(test_vector3);
    
    index.appendVector(test_vector3.data(), label3);
    
    // Verify the third vector was added
    auto results2 = index.topKQuery(test_vector3.data(), 3, &queryParams);
    EXPECT_TRUE(results2 != nullptr);
    EXPECT_EQ(results2->code, VecSim_OK);
    EXPECT_GE(results2->results.size(), 1);
    
    // Check that all three labels are present
    bool found_label3 = false;
    for (const auto& result : results2->results) {
        if (result.id == label3) found_label3 = true;
    }
    EXPECT_TRUE(found_label3) << "Label " << label3 << " not found in results";
    
    // Verify final index size
    EXPECT_EQ(index.indexSize(), 3);
    EXPECT_EQ(index.indexLabelCount(), 3);
    
    // Clean up
    delete results;
    delete results2;
    // shared_ptr will handle deallocation automatically
}

TEST_F(HNSWDiskIndexTest, BatchingTest) {
    // Test the batching functionality
    const size_t dim = 64;
    
    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 8;
    params.efConstruction = 100;
    params.efRuntime = 50;
    params.epsilon = 0.01;
    
    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1; // Use small block size for testing
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle* default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, default_cf);
    
    // Create test vectors
    std::mt19937 rng(42);
    std::vector<std::vector<float>> test_vectors;
    std::vector<labelType> labels;
    
    // Add 15 vectors (more than the batch threshold of 10)
    for (int i = 0; i < 15; i++) {
        auto test_vector = createRandomVector(dim, rng);
        normalizeVector(test_vector);
        test_vectors.push_back(test_vector);
        labels.push_back(1000 + i);
        
        index.addVector(test_vector.data(), labels[i]);
    }
    
    // Verify that vectors are in memory (pending)
    EXPECT_EQ(index.indexSize(), 15);
    EXPECT_EQ(index.indexLabelCount(), 15);
    
    // Force flush to process the batch
    index.flushBatch();
    
    // Verify that vectors are now on disk
    EXPECT_EQ(index.indexSize(), 15);
    EXPECT_EQ(index.indexLabelCount(), 15);
    
    // Test query to verify vectors are accessible
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 50;
    
    auto results = index.topKQuery(test_vectors[0].data(), 10, &queryParams);
    EXPECT_TRUE(results != nullptr);
    EXPECT_EQ(results->code, VecSim_OK);
    EXPECT_GE(results->results.size(), 1);
    
    delete results;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
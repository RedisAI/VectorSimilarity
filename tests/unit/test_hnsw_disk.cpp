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
#include "mock_thread_pool.h"

using namespace std;

class HNSWDiskIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Disable all debug logging
        VecSimIndexInterface::setLogCallbackFunction(
            [](void *ctx, const char *level, const char *message) {
                // Only show warnings and errors
                if (std::string_view{level} == VecSimCommonStrings::LOG_WARNING_STRING) {
                    std::cout << "[" << level << "] " << message << std::endl;
                }
            });
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
            std::cerr << "Directory is directory: " << std::filesystem::is_directory(temp_dir)
                      << std::endl;
        }
        ASSERT_TRUE(status.ok()) << "Failed to open RocksDB database: " << status.ToString();
    }

    void TearDown() override {
        // Note: HNSWDiskIndex objects created in tests should go out of scope
        // and be destroyed before this method is called.
        // If any index objects still exist, they would be accessing deleted database.

        if (db) {
            db.reset();
            db = nullptr;
        }

        // Clean up temporary directory
        try {
            std::filesystem::remove_all(temp_dir);
        } catch (const std::exception &e) {
            std::cerr << "Warning: Failed to remove temp directory: " << e.what() << std::endl;
        }
    }

    // Helper function to create test vectors
    std::vector<float> createRandomVector(size_t dim, std::mt19937 &rng) {
        std::vector<float> vec(dim);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < dim; ++i) {
            vec[i] = dist(rng);
        }
        return vec;
    }

    // Helper function to normalize a vector (for cosine similarity)
    void normalizeVector(std::vector<float> &vec) {
        float norm = 0.0f;
        for (float val : vec) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float &val : vec) {
                val /= norm;
            }
        }
    }

    std::string temp_dir;
    std::unique_ptr<rocksdb::DB> db;
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
    abstractInitParams.dataSize = dim * sizeof(int8_t);
    abstractInitParams.blockSize = 1;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

    // Basic assertions - check that the index was created successfully
    // Note: We can't access protected members directly, so we'll test through public methods
    EXPECT_TRUE(&index != nullptr);

    // Clean up - shared_ptr will handle deallocation automatically
}

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
    abstractInitParams.dataSize = dim * sizeof(int8_t);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

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
    abstractInitParams.dataSize = dim * sizeof(int8_t);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

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
    EXPECT_EQ(index.getDataSize(), dim * sizeof(int8_t));
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
    abstractInitParams.dataSize = dim * sizeof(int8_t);
    abstractInitParams.blockSize = 1;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

    // Create a test vector
    std::mt19937 rng(42);
    auto test_vector = createRandomVector(dim, rng);
    normalizeVector(test_vector);

    // Test addVector method
    labelType label = 100;
    int result = index.addVector(test_vector.data(), label);

    // Verify the result
    EXPECT_EQ(result, 1); // Should return success

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
    abstractInitParams.dataSize = dim * sizeof(int8_t);
    abstractInitParams.blockSize = 1; // Use small block size for testing
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateQuantizedIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

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
    abstractInitParams.dataSize = dim * sizeof(int8_t);
    abstractInitParams.blockSize = 1; // Use small block size for testing
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateQuantizedIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

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

    std::cout << "About to call topKQuery..." << std::endl;
    auto results = index.topKQuery(test_vector.data(), 2, &queryParams);
    std::cout << "topKQuery returned, results size: " << (results ? results->results.size() : 0)
              << std::endl;

    EXPECT_TRUE(results != nullptr);
    EXPECT_EQ(results->code, VecSim_OK);
    EXPECT_GE(results->results.size(), 1); // Should find at least the first vector

    // Verify that both labels are in the results
    bool found_label1 = false;
    bool found_label2 = false;
    for (const auto &result : results->results) {
        if (result.id == label1)
            found_label1 = true;
        if (result.id == label2)
            found_label2 = true;
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

    index.addVector(test_vector3.data(), label3);

    // Verify the third vector was added
    auto results2 = index.topKQuery(test_vector3.data(), 3, &queryParams);
    EXPECT_TRUE(results2 != nullptr);
    EXPECT_EQ(results2->code, VecSim_OK);
    EXPECT_GE(results2->results.size(), 1);

    // Check that all three labels are present
    bool found_label3 = false;
    for (const auto &result : results2->results) {
        if (result.id == label3)
            found_label3 = true;
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
    abstractInitParams.dataSize = dim * sizeof(int8_t);
    abstractInitParams.blockSize = 1; // Use small block size for testing
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex - use default column family handle
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

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

TEST_F(HNSWDiskIndexTest, HierarchicalSearchTest) {
    // Test the hierarchical search functionality
    const size_t dim = 64;

    // Create HNSW parameters with smaller values for testing
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 4; // Small M for testing
    params.efConstruction = 50;
    params.efRuntime = 20;
    params.epsilon = 0.01;

    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(int8_t);
    abstractInitParams.blockSize = 1;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateQuantizedIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

    // Create test vectors with known relationships
    std::mt19937 rng(42);
    std::vector<std::vector<float>> test_vectors;
    std::vector<labelType> labels;

    // Create a base vector
    auto base_vector = createRandomVector(dim, rng);
    normalizeVector(base_vector);
    test_vectors.push_back(base_vector);
    labels.push_back(100);

    // Create similar vectors (small variations of base)
    for (int i = 1; i < 10; i++) {
        auto similar_vector = base_vector;
        // Add small random noise
        std::uniform_real_distribution<float> noise_dist(-0.1f, 0.1f);
        for (size_t j = 0; j < dim; j++) {
            similar_vector[j] += noise_dist(rng);
        }
        normalizeVector(similar_vector);
        test_vectors.push_back(similar_vector);
        labels.push_back(100 + i);
    }

    // Create some different vectors
    for (int i = 10; i < 20; i++) {
        auto different_vector = createRandomVector(dim, rng);
        normalizeVector(different_vector);
        test_vectors.push_back(different_vector);
        labels.push_back(200 + i);
    }

    std::cout << "\n=== Adding vectors to index ===" << std::endl;

    // Add vectors to index
    for (size_t i = 0; i < test_vectors.size(); i++) {
        std::cout << "Adding vector " << i << " with label " << labels[i] << std::endl;
        index.addVector(test_vectors[i].data(), labels[i]);
    }

    // Force flush to process all vectors
    index.flushBatch();

    std::cout << "\n=== Index state after adding vectors ===" << std::endl;
    std::cout << "Index size: " << index.indexSize() << std::endl;
    std::cout << "Index label count: " << index.indexLabelCount() << std::endl;

    // Debug: Print graph structure
    index.debugPrintGraphStructure();

    // Test hierarchical search with different query vectors
    std::cout << "\n=== Testing hierarchical search ===" << std::endl;

    // Test 1: Query with base vector (should find similar vectors first)
    std::cout << "\n--- Test 1: Query with base vector ---" << std::endl;
    VecSimQueryParams queryParams1;
    queryParams1.hnswRuntimeParams.efRuntime = 20;

    auto results1 = index.topKQuery(test_vectors[0].data(), 5, &queryParams1);
    EXPECT_TRUE(results1 != nullptr) << "Query 1 failed to return results";
    EXPECT_EQ(results1->code, VecSim_OK) << "Query 1 returned error code: " << results1->code;
    EXPECT_GE(results1->results.size(), 1) << "Query 1 returned no results";

    std::cout << "Query 1 results (" << results1->results.size() << "):" << std::endl;
    for (size_t i = 0; i < results1->results.size(); i++) {
        std::cout << "  Result " << i << ": id=" << results1->results[i].id
                  << ", score=" << results1->results[i].score << std::endl;
    }

    // Test 2: Query with a different vector
    std::cout << "\n--- Test 2: Query with different vector ---" << std::endl;
    auto results2 = index.topKQuery(test_vectors[15].data(), 5, &queryParams1);
    EXPECT_TRUE(results2 != nullptr) << "Query 2 failed to return results";
    EXPECT_EQ(results2->code, VecSim_OK) << "Query 2 returned error code: " << results2->code;
    EXPECT_GE(results2->results.size(), 1) << "Query 2 returned no results";

    std::cout << "Query 2 results (" << results2->results.size() << "):" << std::endl;
    for (size_t i = 0; i < results2->results.size(); i++) {
        std::cout << "  Result " << i << ": id=" << results2->results[i].id
                  << ", score=" << results2->results[i].score << std::endl;
    }

    // Test 3: Query with larger k to test multiple results
    std::cout << "\n--- Test 3: Query with larger k ---" << std::endl;
    auto results3 = index.topKQuery(test_vectors[0].data(), 10, &queryParams1);
    EXPECT_TRUE(results3 != nullptr) << "Query 3 failed to return results";
    EXPECT_EQ(results3->code, VecSim_OK) << "Query 3 returned error code: " << results3->code;
    EXPECT_GE(results3->results.size(), 5) << "Query 3 should return at least 5 results";

    std::cout << "Query 3 results (" << results3->results.size() << "):" << std::endl;
    for (size_t i = 0; i < results3->results.size(); i++) {
        std::cout << "  Result " << i << ": id=" << results3->results[i].id
                  << ", score=" << results3->results[i].score << std::endl;
    }

    // Verify that results are ordered by distance (lower score = better)
    for (size_t i = 1; i < results3->results.size(); i++) {
        EXPECT_LE(results3->results[i - 1].score, results3->results[i].score)
            << "Results not properly ordered by distance";
    }

    // Test 4: Query with a completely new vector
    std::cout << "\n--- Test 4: Query with new vector ---" << std::endl;
    auto new_query_vector = createRandomVector(dim, rng);
    normalizeVector(new_query_vector);

    auto results4 = index.topKQuery(new_query_vector.data(), 5, &queryParams1);
    EXPECT_TRUE(results4 != nullptr) << "Query 4 failed to return results";
    EXPECT_EQ(results4->code, VecSim_OK) << "Query 4 returned error code: " << results4->code;
    EXPECT_GE(results4->results.size(), 1) << "Query 4 returned no results";

    std::cout << "Query 4 results (" << results4->results.size() << "):" << std::endl;
    for (size_t i = 0; i < results4->results.size(); i++) {
        std::cout << "  Result " << i << ": id=" << results4->results[i].id
                  << ", score=" << results4->results[i].score << std::endl;
    }

    // Clean up
    delete results1;
    delete results2;
    delete results3;
    delete results4;

    std::cout << "\n=== Hierarchical search test completed ===" << std::endl;
}

TEST_F(HNSWDiskIndexTest, RawVectorStorageAndRetrieval) {
    // Test raw vector storage and retrieval
    const size_t dim = 128;

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
    abstractInitParams.dataSize = dim * sizeof(int8_t);
    abstractInitParams.blockSize = 1;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

    // Create test vectors
    std::mt19937 rng(42);
    std::vector<std::vector<float>> test_vectors;
    std::vector<labelType> labels;

    const size_t num_vectors = 10;
    for (size_t i = 0; i < num_vectors; ++i) {
        auto vec = createRandomVector(dim, rng);
        normalizeVector(vec);
        test_vectors.push_back(vec);
        labels.push_back(i);
    }

    // Add vectors to the index
    for (size_t i = 0; i < num_vectors; ++i) {
        int result = index.addVector(test_vectors[i].data(), labels[i]);
        EXPECT_EQ(result, 1) << "Failed to add vector " << i;
    }

    // Verify that vectors were stored on disk
    std::vector<float> buffer(dim);
    for (size_t i = 0; i < num_vectors; ++i) {
        index.getRawVector(i, buffer.data());

        // Check that the data matches (approximately, due to preprocessing)
        const float* retrieved_data = reinterpret_cast<const float*>(buffer.data());
        for (size_t j = 0; j < dim; ++j) {
            EXPECT_FLOAT_EQ(retrieved_data[j], test_vectors[i][j])
                << "Vector " << i << " element " << j << " mismatch";
        }
    }

    std::cout << "Raw vector storage and retrieval test passed!" << std::endl;
}

TEST_F(HNSWDiskIndexTest, RawVectorRetrievalInvalidId) {
    // Test that getRawVector handles invalid IDs gracefully
    const size_t dim = 128;

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
    abstractInitParams.blockSize = 1;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

    // Try to retrieve a vector with an invalid ID (index is empty)
    std::vector<float> buffer(dim);
    memset(buffer.data(), 0, dim * sizeof(float));
    index.getRawVector(0, buffer.data());
    for (size_t j = 0; j < dim; ++j) {
        EXPECT_FLOAT_EQ(buffer[j], 0.0f) << "Invalid ID retrieval returned non-zero data";
    }

    std::cout << "Invalid ID retrieval test passed!" << std::endl;
}

TEST_F(HNSWDiskIndexTest, RawVectorMultipleRetrievals) {
    // Test that we can retrieve the same vector multiple times
    const size_t dim = 64;

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
    abstractInitParams.blockSize = 1;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

    // Create and add a test vector
    std::mt19937 rng(42);
    auto test_vector = createRandomVector(dim, rng);
    normalizeVector(test_vector);

    int result = index.addVector(test_vector.data(), 0);
    EXPECT_EQ(result, 1) << "Failed to add vector";

    // Retrieve the vector multiple times
    const size_t num_retrievals = 5;
    std::vector<std::vector<float>> retrieved_vectors;

    for (size_t i = 0; i < num_retrievals; ++i) {
        std::vector<float> buffer(dim);
        index.getRawVector(0, buffer.data());
        retrieved_vectors.push_back(buffer);
    }

    // Verify all retrievals have the same data
    for (size_t i = 1; i < num_retrievals; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            EXPECT_FLOAT_EQ(retrieved_vectors[0][j], retrieved_vectors[i][j])
                << "Retrieval " << i << " element " << j << " differs from first retrieval";
        }
    }

    // Verify the data matches the original
    for (size_t j = 0; j < dim; ++j) {
        EXPECT_FLOAT_EQ(retrieved_vectors[0][j], test_vector[j])
            << "Retrieved vector element " << j << " mismatch";
    }

    std::cout << "Multiple retrievals test passed!" << std::endl;
}


TEST_F(HNSWDiskIndexTest, markDelete) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 16;
    params.efConstruction = 200;
    params.efRuntime = 50;
    params.epsilon = 0.01;

    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateQuantizedIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf);

    // Try marking a non-existing label (should return empty vector)
    ASSERT_EQ(index.markDelete(0), vecsim_stl::vector<idType>(abstractInitParams.allocator));

    // Add vectors to the index
    for (size_t i = 0; i < n; i++) {
        std::vector<float> vec(dim);
        for (size_t j = 0; j < dim; j++) {
            vec[j] = static_cast<float>(i)/static_cast<float>(n);
        }
        int result = index.addVector(vec.data(), i);
        EXPECT_EQ(result, 1) << "Failed to add vector " << i;
    }
    ASSERT_EQ(index.indexSize(), n);

    // Create query vector around the middle
    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; j++) {
        query[j] = 0.5f;
    }

    // Search for k results around the middle. Expect to find them.
    auto verify_res = [&](size_t id, double score, size_t result_index) {
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (result_index + 1) / 2);
        // ASSERT_EQ(score, (4 * ((result_index + 1) / 2) * ((result_index + 1) / 2)));
    };

    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 50;

    // Run initial search test
    auto results = index.topKQuery(query.data(), k, &queryParams);
    ASSERT_TRUE(results != nullptr);
    ASSERT_EQ(results->code, VecSim_OK);
    ASSERT_EQ(results->results.size(), k);

    for (size_t i = 0; i < results->results.size(); i++) {
        verify_res(results->results[i].id, results->results[i].score, i);
    }
    delete results;

    // Get the entrypoint to determine which vectors to delete
    auto [ep_id, max_level] = index.safeGetEntryPointState();
    unsigned char ep_reminder = ep_id % 2;

    // Mark as deleted half of the vectors, including the entrypoint
    for (labelType label = 0; label < n; label++) {
        if (label % 2 == ep_reminder) {
            auto deleted_ids = index.markDelete(label);
            ASSERT_EQ(deleted_ids.size(), 1);
            // In this test, labels are sequential starting from 0, so internal ID == label
            ASSERT_EQ(deleted_ids[0], label);
        }
    }

    ASSERT_EQ(index.getNumMarkedDeleted(), n / 2);
    // indexSize() returns active elements (curElementCount - numMarkedDeleted) in disk mode
    ASSERT_EQ(index.indexSize(), n / 2);

    // Search for k results around the middle. Expect to find only non-deleted results.
    auto verify_res_half = [&](size_t id, double score, size_t result_index) {
        // Verify the result is not from the deleted set
        ASSERT_NE(id % 2, ep_reminder);
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        // The results alternate between below and above 50, with pairs at the same distance
        // If ep_reminder == 0 (deleted even), remaining are odd: 49,51 (diff=1), 47,53 (diff=3), etc.
        //   Pattern: expected_diff = (result_index + 1) | 1 = make it odd, starting from 1
        // If ep_reminder == 1 (deleted odd), remaining are even: 50 (diff=0), 48,52 (diff=2), etc.
        //   Pattern: expected_diff = ((result_index + 1) / 2) * 2 = pairs of even numbers
        size_t expected_diff;
        if (ep_reminder == 0) {
            // Deleted even, remaining odd
            expected_diff = (result_index + 1) | 1;
        } else {
            // Deleted odd, remaining even
            expected_diff = ((result_index + 1) / 2) * 2;
        }
        ASSERT_EQ(diff_id, expected_diff);
    };

    // Run search test after marking deleted
    results = index.topKQuery(query.data(), k, &queryParams);
    ASSERT_TRUE(results != nullptr);
    ASSERT_EQ(results->code, VecSim_OK);
    ASSERT_EQ(results->results.size(), k);

    for (size_t i = 0; i < results->results.size(); i++) {
        verify_res_half(results->results[i].id, results->results[i].score, i);
    }
    delete results;

    // Add a new vector, make sure it has no link to a deleted vector
    std::vector<float> new_vec(dim);
    for (size_t j = 0; j < dim; j++) {
        new_vec[j] = 1.0f;
    }
    index.addVector(new_vec.data(), n);

    // Re-add the previously marked vectors (under new internal ids)
    for (labelType label = 0; label < n; label++) {
        if (label % 2 == ep_reminder) {
            std::vector<float> vec(dim);
            for (size_t j = 0; j < dim; j++) {
                vec[j] = static_cast<float>(label)/static_cast<float>(n);
            }
            index.addVector(vec.data(), label);
        }
    }

    // indexSize() returns active elements (curElementCount - numMarkedDeleted)
    // curElementCount = n + 1 + n/2 = 151 (original + 1 new + re-added)
    // numMarkedDeleted = n/2 = 50
    // indexSize() = 151 - 50 = 101 = n + 1
    ASSERT_EQ(index.indexSize(), n + 1);
    ASSERT_EQ(index.getNumMarkedDeleted(), n / 2);

    // Search for k results around the middle again. Expect to find the same results we
    // found in the first search
    results = index.topKQuery(query.data(), k, &queryParams);
    ASSERT_TRUE(results != nullptr);
    ASSERT_EQ(results->code, VecSim_OK);
    ASSERT_EQ(results->results.size(), k);

    for (size_t i = 0; i < results->results.size(); i++) {
        verify_res(results->results[i].id, results->results[i].score, i);
    }
    delete results;
}

TEST_F(HNSWDiskIndexTest, BatchedDeletionTest) {
    // Test batched deletion functionality
    const size_t dim = 64;
    const size_t n = 150; // More than deleteBatchThreshold (10)

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
    abstractInitParams.blockSize = 1024; // Set block size
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf, temp_dir);

    // Add vectors to the index
    std::mt19937 rng(42);
    for (labelType label = 0; label < n; label++) {
        auto vec = createRandomVector(dim, rng);
        index.addVector(vec.data(), label);
    }

    // Flush any pending batches
    index.flushBatch();

    // Verify all vectors were added
    ASSERT_EQ(index.indexSize(), n);
    ASSERT_EQ(index.indexLabelCount(), n);

    // Delete vectors in batches (delete every other vector)
    // This should trigger batch processing when we reach deleteBatchThreshold
    size_t deleted_count = 0;
    for (labelType label = 0; label < n; label += 2) {
        int result = index.deleteVector(label);
        ASSERT_EQ(result, 1); // Deletion should succeed
        deleted_count++;
    }

    // Manually flush any remaining deletes
    index.flushDeleteBatch();

    // Verify the index size and label count
    // indexSize() returns active elements (curElementCount - numMarkedDeleted)
    // deleted_count = n/2
    ASSERT_EQ(index.indexSize(), n - deleted_count);
    ASSERT_EQ(index.indexLabelCount(), n - deleted_count);

    // Verify that deleted vectors cannot be deleted again (they don't exist)
    for (labelType label = 0; label < n; label += 2) {
        int result = index.deleteVector(label);
        ASSERT_EQ(result, 0) << "Deleted vector " << label << " should not be found";
    }

    // Get the set of labels to verify which ones exist
    auto labels_set = index.getLabelsSet();

    // Verify that deleted vectors are not in the labels set
    for (labelType label = 0; label < n; label += 2) {
        ASSERT_EQ(labels_set.count(label), 0) << "Deleted label " << label << " should not be in labels set";
    }

    // Verify that non-deleted vectors are in the labels set
    for (labelType label = 1; label < n; label += 2) {
        ASSERT_EQ(labels_set.count(label), 1) << "Non-deleted label " << label << " should be in labels set";
    }

    // Perform a search to verify graph connectivity is maintained
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 50;

    // Search using a non-deleted vector
    auto query_vec = createRandomVector(dim, rng);
    size_t k = 10;
    auto results = index.topKQuery(query_vec.data(), k, &queryParams);

    ASSERT_TRUE(results != nullptr);
    ASSERT_EQ(results->code, VecSim_OK);
    ASSERT_LE(results->results.size(), k);

    // Verify that all returned results are non-deleted vectors (odd labels)
    for (size_t i = 0; i < results->results.size(); i++) {
        labelType result_label = results->results[i].id;
        ASSERT_EQ(result_label % 2, 1) << "Found deleted vector in search results: " << result_label;
    }

    delete results;
}

// Test interleaved insertions and deletions to verify separated staging areas
TEST_F(HNSWDiskIndexTest, InterleavedInsertDeleteTest) {
    const size_t dim = 64;
    const size_t initial_count = 100;

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
    abstractInitParams.blockSize = 1024;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf, temp_dir);

    // Phase 1: Add initial vectors (0-99)
    std::vector<std::vector<float>> vectors;
    std::mt19937 rng(42);

    for (labelType label = 0; label < initial_count; label++) {
        auto vec = createRandomVector(dim, rng);
        vectors.push_back(vec);
        int ret = index.addVector(vec.data(), label);
        ASSERT_EQ(ret, 1) << "Failed to add vector " << label;
    }

    // Flush any pending batches
    index.flushBatch();

    ASSERT_EQ(index.indexSize(), initial_count);
    ASSERT_EQ(index.indexLabelCount(), initial_count);

    // Phase 2: Interleave deletions and insertions
    // Delete vectors 0-19 (20 deletions)
    // Add vectors 100-119 (20 insertions)
    // This tests that both staging areas work independently

    size_t delete_start = 0;
    size_t delete_count = 20;
    size_t insert_start = 100;
    size_t insert_count = 20;

    // Interleave: delete one, insert one, delete one, insert one, etc.
    for (size_t i = 0; i < delete_count; i++) {
        // Delete a vector
        labelType delete_label = delete_start + i;
        int delete_ret = index.deleteVector(delete_label);
        ASSERT_EQ(delete_ret, 1) << "Failed to delete vector " << delete_label;

        // Insert a new vector
        labelType insert_label = insert_start + i;
        auto new_vec = createRandomVector(dim, rng);
        vectors.push_back(new_vec);
        int insert_ret = index.addVector(new_vec.data(), insert_label);
        ASSERT_EQ(insert_ret, 1) << "Failed to add vector " << insert_label;
    }

    // Flush any pending batches
    index.flushBatch();
    index.flushDeleteBatch();

    // Verify index state
    // indexSize() returns active elements (curElementCount - numMarkedDeleted)
    // curElementCount = initial_count + insert_count (no ID recycling)
    // numMarkedDeleted = delete_count
    // indexSize() = initial_count + insert_count - delete_count
    ASSERT_EQ(index.indexSize(), initial_count + insert_count - delete_count);
    // indexLabelCount() returns labelToIdMap.size() which reflects active (non-deleted) labels
    // So it should be: initial_count - delete_count + insert_count = 100 - 20 + 20 = 100
    ASSERT_EQ(index.indexLabelCount(), initial_count - delete_count + insert_count);

    // Phase 3: Verify deleted vectors are gone
    for (size_t i = delete_start; i < delete_start + delete_count; i++) {
        int ret = index.deleteVector(i);
        ASSERT_EQ(ret, 0) << "Vector " << i << " should already be deleted";
    }

    // Phase 4: Verify new vectors are searchable
    auto labels = index.getLabelsSet();

    // Check deleted vectors are not in the set
    for (size_t i = delete_start; i < delete_start + delete_count; i++) {
        ASSERT_EQ(labels.count(i), 0) << "Deleted vector " << i << " still in labels set";
    }

    // Check new vectors are in the set
    for (size_t i = insert_start; i < insert_start + insert_count; i++) {
        ASSERT_EQ(labels.count(i), 1) << "New vector " << i << " not in labels set";
    }

    // Phase 5: Perform a search to verify graph integrity
    size_t k = 10;
    auto *results = index.topKQuery(vectors[50].data(), k, nullptr);
    ASSERT_TRUE(results != nullptr);
    ASSERT_EQ(results->code, VecSim_OK);
    ASSERT_LE(results->results.size(), k);

    // Verify no deleted vectors appear in results
    for (size_t i = 0; i < results->results.size(); i++) {
        labelType result_label = results->results[i].id;
        ASSERT_FALSE(result_label >= delete_start && result_label < delete_start + delete_count)
            << "Found deleted vector in search results: " << result_label;
    }

    delete results;

    // Phase 6: More aggressive interleaving - multiple operations before batch flush
    // Delete vectors 20-29 and add vectors 120-129
    delete_start = 20;
    delete_count = 10;
    insert_start = 120;
    insert_count = 10;

    for (size_t i = 0; i < std::max(delete_count, insert_count); i++) {
        if (i < delete_count) {
            labelType delete_label = delete_start + i;
            int delete_ret = index.deleteVector(delete_label);
            ASSERT_EQ(delete_ret, 1) << "Failed to delete vector " << delete_label;
        }

        if (i < insert_count) {
            labelType insert_label = insert_start + i;
            auto new_vec = createRandomVector(dim, rng);
            vectors.push_back(new_vec);
            int insert_ret = index.addVector(new_vec.data(), insert_label);
            ASSERT_EQ(insert_ret, 1) << "Failed to add vector " << insert_label;
        }
    }

    // Flush any pending batches
    index.flushBatch();
    index.flushDeleteBatch();

    // Final verification
    // indexSize() returns active elements (curElementCount - numMarkedDeleted)
    // curElementCount = initial_count + 30 (total insertions)
    // numMarkedDeleted = 30 (total deletions)
    // indexSize() = initial_count + 30 - 30 = initial_count
    ASSERT_EQ(index.indexSize(), initial_count);
    // indexLabelCount() = initial_count - total_deletes + total_inserts = 100 - 30 + 30 = 100
    size_t expected_label_count = initial_count - 30 + 30; // deleted 30 total, added 30 total
    ASSERT_EQ(index.indexLabelCount(), expected_label_count);

    // Verify graph is still searchable
    results = index.topKQuery(vectors[80].data(), k, nullptr);
    ASSERT_TRUE(results != nullptr);
    ASSERT_EQ(results->code, VecSim_OK);
    ASSERT_GT(results->results.size(), 0) << "Search returned no results after interleaved operations";

    delete results;
}

TEST_F(HNSWDiskIndexTest, StagedRepairTest) {
    const size_t dim = 64;
    const size_t n = 50;

    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 8;  // Small M to ensure neighbors are interconnected
    params.efConstruction = 100;
    params.efRuntime = 50;
    params.epsilon = 0.01;

    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1024;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf, temp_dir);

    // Add vectors to the index - use sequential vectors so they have predictable neighbors
    std::mt19937 rng(42);
    std::vector<std::vector<float>> vectors;
    for (labelType label = 0; label < n; label++) {
        auto vec = createRandomVector(dim, rng);
        vectors.push_back(vec);
        int ret = index.addVector(vec.data(), label);
        ASSERT_EQ(ret, 1) << "Failed to add vector " << label;
    }

    // Flush to disk so all graph data is persisted
    index.flushBatch();

    ASSERT_EQ(index.indexSize(), n);
    ASSERT_EQ(index.indexLabelCount(), n);

    // Delete some vectors (e.g., every 3rd vector)
    // This creates stale edges: nodes that point to deleted nodes
    std::vector<labelType> deleted_labels;
    for (labelType label = 0; label < n; label += 3) {
        int ret = index.deleteVector(label);
        ASSERT_EQ(ret, 1) << "Failed to delete vector " << label;
        deleted_labels.push_back(label);
    }

    // Flush the delete batch to mark vectors as deleted
    index.flushDeleteBatch();

    size_t num_deleted = deleted_labels.size();
    ASSERT_EQ(index.getNumMarkedDeleted(), num_deleted);

    // Now perform searches - this will trigger getNeighbors which should:
    // 1. Filter out deleted nodes from neighbor lists
    // 2. Stage the cleaned lists for repair (opportunistic cleanup)
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 50;

    // Do multiple searches to access different parts of the graph
    for (size_t i = 0; i < 10; i++) {
        auto results = index.topKQuery(vectors[i * 3 + 1].data(), 5, &queryParams);
        ASSERT_TRUE(results != nullptr);
        ASSERT_EQ(results->code, VecSim_OK);

        // Verify no deleted vectors in results
        for (const auto &result : results->results) {
            bool is_deleted = std::find(deleted_labels.begin(), deleted_labels.end(),
                                        result.id) != deleted_labels.end();
            ASSERT_FALSE(is_deleted) << "Deleted vector " << result.id << " found in search results";
        }

        delete results;
    }

    // Flush staged repair updates (triggered by next batch operation)
    // The repairs are flushed along with delete batch
    index.flushDeleteBatch();

    // Verify the index is still functional after repairs
    auto final_results = index.topKQuery(vectors[1].data(), 10, &queryParams);
    ASSERT_TRUE(final_results != nullptr);
    ASSERT_EQ(final_results->code, VecSim_OK);
    ASSERT_GT(final_results->results.size(), 0);

    // Verify all results are non-deleted vectors
    for (const auto &result : final_results->results) {
        bool is_deleted = std::find(deleted_labels.begin(), deleted_labels.end(),
                                    result.id) != deleted_labels.end();
        ASSERT_FALSE(is_deleted) << "Deleted vector " << result.id << " found in final results";
    }

    delete final_results;

    // Additional verification: re-query to ensure cleaned neighbor lists work correctly
    // After staged repair flush, the disk should have cleaned neighbor lists
    for (size_t i = 0; i < 5; i++) {
        size_t query_idx = (i * 7 + 2) % n;
        // Skip if this vector was deleted
        if (query_idx % 3 == 0) query_idx++;

        auto results = index.topKQuery(vectors[query_idx].data(), 5, &queryParams);
        ASSERT_TRUE(results != nullptr);
        ASSERT_EQ(results->code, VecSim_OK);

        for (const auto &result : results->results) {
            bool is_deleted = std::find(deleted_labels.begin(), deleted_labels.end(),
                                        result.id) != deleted_labels.end();
            ASSERT_FALSE(is_deleted) << "Deleted vector " << result.id
                                     << " found after repair flush";
        }

        delete results;
    }
}

// Test that verifies bidirectional edge updates during graph repair
TEST_F(HNSWDiskIndexTest, GraphRepairBidirectionalEdges) {
    size_t n = 50;
    size_t dim = 4;

    // Create HNSW parameters with small M to make bidirectional updates more likely
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 4;  // Small M to test edge capacity limits
    params.efConstruction = 50;
    params.efRuntime = 20;
    params.epsilon = 0.01;

    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1024;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf, temp_dir);

    // Add vectors in a clustered pattern to create predictable neighbor relationships
    std::mt19937 rng(12345);
    std::vector<std::vector<float>> vectors;

    // Create 5 clusters of 10 vectors each
    for (size_t cluster = 0; cluster < 5; cluster++) {
        std::vector<float> cluster_center(dim);
        for (size_t d = 0; d < dim; d++) {
            cluster_center[d] = static_cast<float>(cluster * 10);
        }

        for (size_t i = 0; i < 10; i++) {
            std::vector<float> vec(dim);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t d = 0; d < dim; d++) {
                vec[d] = cluster_center[d] + dist(rng);
            }
            vectors.push_back(vec);
            labelType label = cluster * 10 + i;
            int ret = index.addVector(vec.data(), label);
            ASSERT_EQ(ret, 1) << "Failed to add vector " << label;
        }
    }

    // Flush to disk
    index.flushBatch();
    ASSERT_EQ(index.indexSize(), n);

    // Delete a vector from the middle of a cluster (should trigger repair)
    // Delete vector 15 (middle of cluster 1)
    labelType deleted_label = 15;
    int ret = index.deleteVector(deleted_label);
    ASSERT_EQ(ret, 1) << "Failed to delete vector " << deleted_label;

    // Flush the delete batch - this triggers graph repair
    index.flushDeleteBatch();
    ASSERT_EQ(index.getNumMarkedDeleted(), 1);

    // Verify the index is still functional
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 20;

    // Query with vectors from the same cluster
    for (size_t i = 10; i < 20; i++) {
        if (i == deleted_label) continue;

        auto results = index.topKQuery(vectors[i].data(), 5, &queryParams);
        ASSERT_TRUE(results != nullptr);
        ASSERT_EQ(results->code, VecSim_OK);
        ASSERT_GT(results->results.size(), 0);

        // Verify deleted vector is not in results
        for (const auto &result : results->results) {
            ASSERT_NE(result.id, deleted_label) << "Deleted vector found in search results";
        }

        delete results;
    }

    // Delete multiple vectors to test batch repair with bidirectional updates
    std::vector<labelType> deleted_labels = {5, 15, 25, 35};
    for (labelType label : deleted_labels) {
        if (label == 15) continue; // Already deleted
        ret = index.deleteVector(label);
        ASSERT_EQ(ret, 1) << "Failed to delete vector " << label;
    }

    // Flush and verify
    index.flushDeleteBatch();
    ASSERT_EQ(index.getNumMarkedDeleted(), 4);

    // Verify graph connectivity is maintained
    for (size_t i = 0; i < n; i++) {
        if (std::find(deleted_labels.begin(), deleted_labels.end(), i) != deleted_labels.end()) {
            continue;
        }

        auto results = index.topKQuery(vectors[i].data(), 3, &queryParams);
        ASSERT_TRUE(results != nullptr);
        ASSERT_EQ(results->code, VecSim_OK);

        // Should find at least some neighbors
        ASSERT_GT(results->results.size(), 0) << "No neighbors found for vector " << i;

        // Verify no deleted vectors in results
        for (const auto &result : results->results) {
            bool is_deleted = std::find(deleted_labels.begin(), deleted_labels.end(),
                                        result.id) != deleted_labels.end();
            ASSERT_FALSE(is_deleted) << "Deleted vector " << result.id << " found in results";
        }

        delete results;
    }
}

// Test that verifies unidirectional edges are cleaned up via opportunistic repair
TEST_F(HNSWDiskIndexTest, UnidirectionalEdgeCleanup) {
    size_t n = 30;
    size_t dim = 4;

    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 8;
    params.efConstruction = 100;
    params.efRuntime = 30;
    params.epsilon = 0.01;

    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1024;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf, temp_dir);

    // Add vectors
    std::mt19937 rng(99999);
    std::vector<std::vector<float>> vectors;
    for (labelType label = 0; label < n; label++) {
        auto vec = createRandomVector(dim, rng);
        vectors.push_back(vec);
        int ret = index.addVector(vec.data(), label);
        ASSERT_EQ(ret, 1) << "Failed to add vector " << label;
    }

    // Flush to disk
    index.flushBatch();
    ASSERT_EQ(index.indexSize(), n);

    // Delete some vectors - this may create unidirectional dangling edges
    // (nodes that point to deleted nodes but are not in the deleted node's neighbor list)
    std::vector<labelType> deleted_labels = {5, 10, 15, 20};
    for (labelType label : deleted_labels) {
        int ret = index.deleteVector(label);
        ASSERT_EQ(ret, 1) << "Failed to delete vector " << label;
    }

    // Flush delete batch
    index.flushDeleteBatch();
    ASSERT_EQ(index.getNumMarkedDeleted(), deleted_labels.size());

    // Perform searches - this triggers getNeighbors which will:
    // 1. Filter out deleted nodes from neighbor lists
    // 2. Stage repairs for nodes with dangling edges (opportunistic cleanup)
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 30;

    // Do multiple searches to traverse the graph and trigger opportunistic repair
    for (size_t i = 0; i < n; i++) {
        if (std::find(deleted_labels.begin(), deleted_labels.end(), i) != deleted_labels.end()) {
            continue;
        }

        auto results = index.topKQuery(vectors[i].data(), 5, &queryParams);
        ASSERT_TRUE(results != nullptr);
        ASSERT_EQ(results->code, VecSim_OK);

        // Verify no deleted vectors in results
        for (const auto &result : results->results) {
            bool is_deleted = std::find(deleted_labels.begin(), deleted_labels.end(),
                                        result.id) != deleted_labels.end();
            ASSERT_FALSE(is_deleted) << "Deleted vector " << result.id << " found in search results";
        }

        delete results;
    }

    // Flush staged repair updates (cleanup of unidirectional dangling edges)
    // This happens automatically during the next batch operation
    index.flushDeleteBatch();

    // Verify the graph is still functional after cleanup
    for (size_t i = 0; i < 10; i++) {
        if (std::find(deleted_labels.begin(), deleted_labels.end(), i) != deleted_labels.end()) {
            continue;
        }

        auto results = index.topKQuery(vectors[i].data(), 5, &queryParams);
        ASSERT_TRUE(results != nullptr);
        ASSERT_EQ(results->code, VecSim_OK);
        ASSERT_GT(results->results.size(), 0);

        // Verify all results are non-deleted
        for (const auto &result : results->results) {
            bool is_deleted = std::find(deleted_labels.begin(), deleted_labels.end(),
                                        result.id) != deleted_labels.end();
            ASSERT_FALSE(is_deleted) << "Deleted vector found after opportunistic cleanup";
        }

        delete results;
    }
}

// Test graph repair with heuristic selection
TEST_F(HNSWDiskIndexTest, GraphRepairWithHeuristic) {
    size_t n = 40;
    size_t dim = 8;

    // Create HNSW parameters with small M to force heuristic selection
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 3;  // Very small M to force heuristic pruning
    params.efConstruction = 50;
    params.efRuntime = 20;
    params.epsilon = 0.01;

    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1024;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(),
                                      default_cf, temp_dir);

    // Add vectors in a dense cluster to create many neighbor relationships
    std::mt19937 rng(54321);
    std::vector<std::vector<float>> vectors;

    for (labelType label = 0; label < n; label++) {
        std::vector<float> vec(dim);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (size_t d = 0; d < dim; d++) {
            vec[d] = dist(rng);
        }
        vectors.push_back(vec);
        int ret = index.addVector(vec.data(), label);
        ASSERT_EQ(ret, 1) << "Failed to add vector " << label;
    }

    // Flush to disk
    index.flushBatch();
    ASSERT_EQ(index.indexSize(), n);

    // Delete a vector that likely has many neighbors
    // This will trigger repair with heuristic selection (candidates > max_M)
    labelType deleted_label = 20;
    int ret = index.deleteVector(deleted_label);
    ASSERT_EQ(ret, 1);

    // Flush delete batch - triggers graph repair with heuristic
    index.flushDeleteBatch();
    ASSERT_EQ(index.getNumMarkedDeleted(), 1);

    // Verify search quality is maintained (heuristic selected good neighbors)
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 20;

    for (size_t i = 0; i < n; i++) {
        if (i == deleted_label) continue;

        auto results = index.topKQuery(vectors[i].data(), 5, &queryParams);
        ASSERT_TRUE(results != nullptr);
        ASSERT_EQ(results->code, VecSim_OK);

        // Should find neighbors despite small M
        ASSERT_GT(results->results.size(), 0) << "No neighbors found for vector " << i;

        // Verify deleted vector not in results
        for (const auto &result : results->results) {
            ASSERT_NE(result.id, deleted_label);
        }

        delete results;
    }

    // Delete multiple vectors to stress test the heuristic repair
    std::vector<labelType> deleted_labels = {5, 10, 15, 20, 25, 30};
    for (labelType label : deleted_labels) {
        if (label == 20) continue; // Already deleted
        ret = index.deleteVector(label);
        ASSERT_EQ(ret, 1);
    }

    index.flushDeleteBatch();
    ASSERT_EQ(index.getNumMarkedDeleted(), 6);

    // Verify graph is still navigable
    size_t successful_queries = 0;
    for (size_t i = 0; i < n; i++) {
        if (std::find(deleted_labels.begin(), deleted_labels.end(), i) != deleted_labels.end()) {
            continue;
        }

        auto results = index.topKQuery(vectors[i].data(), 3, &queryParams);
        ASSERT_TRUE(results != nullptr);
        ASSERT_EQ(results->code, VecSim_OK);

        if (results->results.size() > 0) {
            successful_queries++;
        }

        delete results;
    }

    // Most queries should succeed (graph should remain connected)
    ASSERT_GT(successful_queries, (n - deleted_labels.size()) / 2)
        << "Too many queries failed - graph may be disconnected";
}

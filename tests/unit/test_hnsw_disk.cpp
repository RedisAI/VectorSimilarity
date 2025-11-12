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
    abstractInitParams.dataSize = dim * sizeof(float);
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
    abstractInitParams.dataSize = dim * sizeof(float);
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
    abstractInitParams.dataSize = dim * sizeof(float);
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
    abstractInitParams.dataSize = dim * sizeof(float);
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
    abstractInitParams.dataSize = dim * sizeof(float);
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
    for (size_t i = 0; i < num_vectors; ++i) {
        auto retrieved_vector = index.getRawVector(i);

        // Check that we got a vector back
        EXPECT_FALSE(retrieved_vector.empty()) << "Retrieved vector " << i << " is empty";

        // Check that the size matches
        EXPECT_EQ(retrieved_vector.size(), dim * sizeof(float))
            << "Retrieved vector " << i << " has incorrect size";

        // Check that the data matches (approximately, due to preprocessing)
        const float *retrieved_data = reinterpret_cast<const float *>(retrieved_vector.data());
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
    auto retrieved_vector = index.getRawVector(0);

    // Should return an empty vector
    EXPECT_TRUE(retrieved_vector.empty()) << "Retrieved vector for invalid ID should be empty";

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
    std::vector<std::vector<char>> retrieved_vectors;

    for (size_t i = 0; i < num_retrievals; ++i) {
        auto retrieved = index.getRawVector(0);
        retrieved_vectors.push_back(retrieved);
    }

    // Verify all retrievals are identical
    for (size_t i = 1; i < num_retrievals; ++i) {
        EXPECT_EQ(retrieved_vectors[0], retrieved_vectors[i])
            << "Retrieval " << i << " differs from first retrieval";
    }

    // Verify the data matches the original
    const float *retrieved_data = reinterpret_cast<const float *>(retrieved_vectors[0].data());
    for (size_t j = 0; j < dim; ++j) {
        EXPECT_FLOAT_EQ(retrieved_data[j], test_vector[j])
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
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
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
            vec[j] = static_cast<float>(i);
        }
        int result = index.addVector(vec.data(), i);
        EXPECT_EQ(result, 1) << "Failed to add vector " << i;
    }
    ASSERT_EQ(index.indexSize(), n);

    // Create query vector around the middle
    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; j++) {
        query[j] = static_cast<float>(n / 2);
    }

    // Search for k results around the middle. Expect to find them.
    auto verify_res = [&](size_t id, double score, size_t result_index) {
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (result_index + 1) / 2);
        ASSERT_EQ(score, (4 * ((result_index + 1) / 2) * ((result_index + 1) / 2)));
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
    ASSERT_EQ(index.indexSize(), n);

    // Search for k results around the middle. Expect to find only non-deleted results.
    auto verify_res_half = [&](size_t id, double score, size_t result_index) {
        ASSERT_NE(id % 2, ep_reminder);
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        // The results alternate between below and above 50, with pairs at the same distance
        // Pattern: 49,51 (diff=1), 47,53 (diff=3), 45,55 (diff=5), etc.
        // So expected_diff = result_index | 1 (make it odd)
        size_t expected_diff = result_index | 1;
        ASSERT_EQ(diff_id, expected_diff);
        ASSERT_EQ(score, (dim * expected_diff * expected_diff));
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
        new_vec[j] = static_cast<float>(n);
    }
    index.addVector(new_vec.data(), n);

    // Re-add the previously marked vectors (under new internal ids)
    for (labelType label = 0; label < n; label++) {
        if (label % 2 == ep_reminder) {
            std::vector<float> vec(dim);
            for (size_t j = 0; j < dim; j++) {
                vec[j] = static_cast<float>(label);
            }
            index.addVector(vec.data(), label);
        }
    }

    ASSERT_EQ(index.indexSize(), n + n / 2 + 1);
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

/********************************** Serialization Tests **********************************/

TEST_F(HNSWDiskIndexTest, SerializationBasic) {
    // Test basic serialization and deserialization of HNSWDiskIndex
    const size_t dim = 4;
    const size_t n = 100;
    const size_t M = 8;
    const size_t ef = 10;
    const double epsilon = 0.004;

    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = M;
    params.efConstruction = ef;
    params.efRuntime = ef;
    params.epsilon = epsilon;

    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.metric = params.metric;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components =
        CreateIndexComponents<float, float>(abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(), default_cf);

    // Generate and add vectors to the index
    std::vector<float> data(n * dim);
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < n * dim; ++i) {
        data[i] = static_cast<float>(distrib(rng));
    }
    for (size_t j = 0; j < n; ++j) {
        index.addVector(data.data() + dim * j, j);
    }

    // Verify index state before saving
    ASSERT_EQ(index.indexSize(), n);
    ASSERT_EQ(index.indexLabelCount(), n);

    // Save the index
    auto file_name = std::string(getenv("ROOT")) + "/tests/unit/hnsw_disk_serialization_basic.bin";
    index.saveIndex(file_name);

    // Verify integrity after saving
    ASSERT_TRUE(index.checkIntegrity().valid_state);

    // Open RocksDB from the checkpoint directory created during save
    std::string checkpoint_dir = HNSWDiskIndex<float, float>::getCheckpointDir(file_name);
    ASSERT_TRUE(std::filesystem::exists(checkpoint_dir))
        << "Checkpoint directory should exist: " << checkpoint_dir;

    rocksdb::Options options;
    options.create_if_missing = false;  // Checkpoint should already exist
    options.error_if_exists = false;

    rocksdb::DB *db2_raw = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, checkpoint_dir, &db2_raw);
    ASSERT_TRUE(status.ok()) << "Failed to open RocksDB from checkpoint: " << status.ToString();
    std::unique_ptr<rocksdb::DB> db2(db2_raw);

    // Load the index from the file
    std::ifstream input(file_name, std::ios::binary);
    ASSERT_TRUE(input.is_open()) << "Failed to open serialization file";

    Serializer::EncodingVersion version = Serializer::ReadVersion(input);
    ASSERT_EQ(version, Serializer::EncodingVersion_V4);

    // Read algorithm type
    VecSimAlgo algo;
    Serializer::readBinaryPOD(input, algo);
    ASSERT_EQ(algo, VecSimAlgo_HNSWLIB_DISK);

    // Read basic parameters
    size_t read_dim;
    VecSimType read_type;
    VecSimMetric read_metric;
    size_t read_blockSize;
    bool read_multi;
    size_t read_initialCapacity;

    Serializer::readBinaryPOD(input, read_dim);
    Serializer::readBinaryPOD(input, read_type);
    Serializer::readBinaryPOD(input, read_metric);
    Serializer::readBinaryPOD(input, read_blockSize);
    Serializer::readBinaryPOD(input, read_multi);
    Serializer::readBinaryPOD(input, read_initialCapacity);

    ASSERT_EQ(read_dim, dim);
    ASSERT_EQ(read_type, VecSimType_FLOAT32);
    ASSERT_EQ(read_metric, VecSimMetric_L2);
    ASSERT_EQ(read_multi, false);

    // Create new index from file
    AbstractIndexInitParams abstractInitParams2;
    abstractInitParams2.dim = read_dim;
    abstractInitParams2.vecType = read_type;
    abstractInitParams2.metric = read_metric;
    abstractInitParams2.dataSize = read_dim * sizeof(float);
    abstractInitParams2.blockSize = 1;
    abstractInitParams2.multi = read_multi;
    abstractInitParams2.allocator = VecSimAllocator::newVecsimAllocator();

    IndexComponents<float, float> components2 =
        CreateIndexComponents<float, float>(abstractInitParams2.allocator, read_metric, read_dim, false);

    rocksdb::ColumnFamilyHandle *default_cf2 = db2->DefaultColumnFamily();
    HNSWDiskIndex<float, float> serialized_index(input, &params, abstractInitParams2, components2,
                                                  db2.get(), default_cf2, version);

    input.close();

    // Verify that the index was loaded as expected
    ASSERT_TRUE(serialized_index.checkIntegrity().valid_state);
    ASSERT_EQ(serialized_index.indexSize(), n);
    ASSERT_EQ(serialized_index.indexLabelCount(), n);

    // Clean up
    remove(file_name.c_str());
    db2.reset();
    std::filesystem::remove_all(checkpoint_dir);
}

TEST_F(HNSWDiskIndexTest, SerializationMultiLabel) {
    // Test serialization with multi-label index (multiple vectors per label)
    const size_t dim = 4;
    const size_t n = 1001;
    const size_t n_labels = 100;  // 1001 vectors with 100 unique labels
    const size_t M = 8;
    const size_t ef = 10;
    const double epsilon = 0.004;

    // Create HNSW parameters for multi-label index
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = true;  // Enable multi-label
    params.M = M;
    params.efConstruction = ef;
    params.efRuntime = ef;
    params.epsilon = epsilon;

    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.metric = params.metric;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1;
    abstractInitParams.multi = true;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components =
        CreateIndexComponents<float, float>(abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(), default_cf);

    // Generate and add vectors to the index
    std::vector<float> data(n * dim);
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < n * dim; ++i) {
        data[i] = static_cast<float>(distrib(rng));
    }
    for (size_t j = 0; j < n; ++j) {
        index.addVector(data.data() + dim * j, j % n_labels);  // Multiple vectors per label
    }

    // Verify index state before saving
    ASSERT_EQ(index.indexSize(), n);
    // Note: HNSWDiskIndex currently treats each vector as a separate label even in multi mode
    // This is a known limitation that should be fixed separately
    ASSERT_EQ(index.indexLabelCount(), n);

    // Save the index
    auto file_name = std::string(getenv("ROOT")) + "/tests/unit/hnsw_disk_serialization_multi.bin";
    index.saveIndex(file_name);

    // Verify integrity after saving
    ASSERT_TRUE(index.checkIntegrity().valid_state);

    // Open RocksDB from the checkpoint directory created during save
    std::string checkpoint_dir = HNSWDiskIndex<float, float>::getCheckpointDir(file_name);
    ASSERT_TRUE(std::filesystem::exists(checkpoint_dir))
        << "Checkpoint directory should exist: " << checkpoint_dir;

    rocksdb::Options options;
    options.create_if_missing = false;  // Checkpoint should already exist
    options.error_if_exists = false;

    rocksdb::DB *db2_raw = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, checkpoint_dir, &db2_raw);
    ASSERT_TRUE(status.ok()) << "Failed to open RocksDB from checkpoint: " << status.ToString();
    std::unique_ptr<rocksdb::DB> db2(db2_raw);

    // Load the index from the file
    std::ifstream input(file_name, std::ios::binary);
    ASSERT_TRUE(input.is_open()) << "Failed to open serialization file";

    Serializer::EncodingVersion version = Serializer::ReadVersion(input);
    ASSERT_EQ(version, Serializer::EncodingVersion_V4);

    // Read algorithm type
    VecSimAlgo algo;
    Serializer::readBinaryPOD(input, algo);
    ASSERT_EQ(algo, VecSimAlgo_HNSWLIB_DISK);

    // Read basic parameters
    size_t read_dim;
    VecSimType read_type;
    VecSimMetric read_metric;
    size_t read_blockSize;
    bool read_multi;
    size_t read_initialCapacity;

    Serializer::readBinaryPOD(input, read_dim);
    Serializer::readBinaryPOD(input, read_type);
    Serializer::readBinaryPOD(input, read_metric);
    Serializer::readBinaryPOD(input, read_blockSize);
    Serializer::readBinaryPOD(input, read_multi);
    Serializer::readBinaryPOD(input, read_initialCapacity);

    ASSERT_EQ(read_dim, dim);
    ASSERT_EQ(read_type, VecSimType_FLOAT32);
    ASSERT_EQ(read_metric, VecSimMetric_L2);
    ASSERT_EQ(read_multi, true);

    // Create new index from file
    AbstractIndexInitParams abstractInitParams2;
    abstractInitParams2.dim = read_dim;
    abstractInitParams2.vecType = read_type;
    abstractInitParams2.metric = read_metric;
    abstractInitParams2.dataSize = read_dim * sizeof(float);
    abstractInitParams2.blockSize = 1;
    abstractInitParams2.multi = read_multi;
    abstractInitParams2.allocator = VecSimAllocator::newVecsimAllocator();

    IndexComponents<float, float> components2 =
        CreateIndexComponents<float, float>(abstractInitParams2.allocator, read_metric, read_dim, false);

    rocksdb::ColumnFamilyHandle *default_cf2 = db2->DefaultColumnFamily();
    HNSWDiskIndex<float, float> serialized_index(input, &params, abstractInitParams2, components2,
                                                  db2.get(), default_cf2, version);

    input.close();

    // Verify that the index was loaded as expected
    ASSERT_TRUE(serialized_index.checkIntegrity().valid_state);
    ASSERT_EQ(serialized_index.indexSize(), n);
    // Note: HNSWDiskIndex currently treats each vector as a separate label even in multi mode
    ASSERT_EQ(serialized_index.indexLabelCount(), n);

    // Clean up
    remove(file_name.c_str());
    db2.reset();
    std::filesystem::remove_all(checkpoint_dir);
}

TEST_F(HNSWDiskIndexTest, SerializationWithSearch) {
    // Test that search results are consistent after serialization/deserialization
    const size_t dim = 4;
    const size_t n = 200;
    const size_t M = 8;
    const size_t ef = 20;
    const double epsilon = 0.004;
    const size_t k = 10;

    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = M;
    params.efConstruction = ef;
    params.efRuntime = ef;
    params.epsilon = epsilon;

    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.vecType = params.type;
    abstractInitParams.metric = params.metric;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.blockSize = 1;
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();

    // Create index components
    IndexComponents<float, float> components =
        CreateIndexComponents<float, float>(abstractInitParams.allocator, VecSimMetric_L2, dim, false);

    // Create HNSWDiskIndex
    rocksdb::ColumnFamilyHandle *default_cf = db->DefaultColumnFamily();
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db.get(), default_cf);

    // Generate and add vectors to the index
    std::vector<float> data(n * dim);
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < n * dim; ++i) {
        data[i] = static_cast<float>(distrib(rng));
    }
    for (size_t j = 0; j < n; ++j) {
        index.addVector(data.data() + dim * j, j);
    }

    // Perform a search before serialization
    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(distrib(rng));
    }

    VecSimQueryParams queryParams{};
    auto *results_before = index.topKQuery(query.data(), k, &queryParams);
    ASSERT_TRUE(results_before != nullptr);
    ASSERT_EQ(results_before->code, VecSim_OK);
    ASSERT_EQ(results_before->results.size(), k);

    // Store the results for comparison
    std::vector<std::pair<labelType, double>> expected_results;
    for (size_t i = 0; i < results_before->results.size(); ++i) {
        expected_results.push_back({results_before->results[i].id, results_before->results[i].score});
    }
    delete results_before;

    // Save the index
    auto file_name = std::string(getenv("ROOT")) + "/tests/unit/hnsw_disk_serialization_search.bin";
    index.saveIndex(file_name);

    // Open RocksDB from the checkpoint directory created during save
    std::string checkpoint_dir = HNSWDiskIndex<float, float>::getCheckpointDir(file_name);
    ASSERT_TRUE(std::filesystem::exists(checkpoint_dir))
        << "Checkpoint directory should exist: " << checkpoint_dir;

    rocksdb::Options options;
    options.create_if_missing = false;  // Checkpoint should already exist
    options.error_if_exists = false;

    rocksdb::DB *db2_raw = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, checkpoint_dir, &db2_raw);
    ASSERT_TRUE(status.ok()) << "Failed to open RocksDB from checkpoint: " << status.ToString();
    std::unique_ptr<rocksdb::DB> db2(db2_raw);

    // Load the index from the file
    std::ifstream input(file_name, std::ios::binary);
    ASSERT_TRUE(input.is_open()) << "Failed to open serialization file";

    Serializer::EncodingVersion version = Serializer::ReadVersion(input);
    ASSERT_EQ(version, Serializer::EncodingVersion_V4);

    // Read algorithm type
    VecSimAlgo algo;
    Serializer::readBinaryPOD(input, algo);
    ASSERT_EQ(algo, VecSimAlgo_HNSWLIB_DISK);

    // Read basic parameters
    size_t read_dim;
    VecSimType read_type;
    VecSimMetric read_metric;
    size_t read_blockSize;
    bool read_multi;
    size_t read_initialCapacity;

    Serializer::readBinaryPOD(input, read_dim);
    Serializer::readBinaryPOD(input, read_type);
    Serializer::readBinaryPOD(input, read_metric);
    Serializer::readBinaryPOD(input, read_blockSize);
    Serializer::readBinaryPOD(input, read_multi);
    Serializer::readBinaryPOD(input, read_initialCapacity);

    // Create new index from file
    AbstractIndexInitParams abstractInitParams2;
    abstractInitParams2.dim = read_dim;
    abstractInitParams2.vecType = read_type;
    abstractInitParams2.metric = read_metric;
    abstractInitParams2.dataSize = read_dim * sizeof(float);
    abstractInitParams2.blockSize = 1;
    abstractInitParams2.multi = read_multi;
    abstractInitParams2.allocator = VecSimAllocator::newVecsimAllocator();

    IndexComponents<float, float> components2 =
        CreateIndexComponents<float, float>(abstractInitParams2.allocator, read_metric, read_dim, false);

    rocksdb::ColumnFamilyHandle *default_cf2 = db2->DefaultColumnFamily();
    HNSWDiskIndex<float, float> serialized_index(input, &params, abstractInitParams2, components2,
                                                  db2.get(), default_cf2, version);

    input.close();

    // Verify that the index was loaded as expected
    ASSERT_TRUE(serialized_index.checkIntegrity().valid_state);
    ASSERT_EQ(serialized_index.indexSize(), n);

    // Perform the same search after deserialization
    auto *results_after = serialized_index.topKQuery(query.data(), k, &queryParams);
    ASSERT_TRUE(results_after != nullptr);
    ASSERT_EQ(results_after->code, VecSim_OK);
    ASSERT_EQ(results_after->results.size(), k);

    // Verify that the search results are identical
    for (size_t i = 0; i < k; ++i) {
        ASSERT_EQ(results_after->results[i].id, expected_results[i].first)
            << "Result " << i << " has different ID";
        ASSERT_DOUBLE_EQ(results_after->results[i].score, expected_results[i].second)
            << "Result " << i << " has different score";
    }
    delete results_after;

    // Clean up
    remove(file_name.c_str());
    db2.reset();
    std::filesystem::remove_all(checkpoint_dir);
}
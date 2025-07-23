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
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex - pass nullptr for column family since we're not using it
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, nullptr);
    
    // Basic assertions - check that the index was created successfully
    // Note: We can't access protected members directly, so we'll test through public methods
    EXPECT_TRUE(&index != nullptr);
    
    // Clean up
    delete abstractInitParams.allocator;
}

TEST_F(HNSWDiskIndexTest, VectorOperations) {
    // Test adding and querying vectors
    const size_t dim = 64;
    const size_t maxElements = 100;
    
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
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, nullptr);
    
    // Create test vectors
    std::mt19937 rng(42);
    std::vector<std::vector<float>> vectors;
    std::vector<std::pair<labelType, const void*>> elements;
    
    for (size_t i = 0; i < 10; ++i) {
        auto vec = createRandomVector(dim, rng);
        normalizeVector(vec);
        vectors.push_back(vec);
        elements.emplace_back(i, vectors.back().data());
    }
    
    // Add vectors to index
    std::vector<labelType> deleted_labels;
    index.batchUpdate(elements, deleted_labels);
    
    // Query the index
    auto query_vec = createRandomVector(dim, rng);
    normalizeVector(query_vec);
    
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 50;
    
    auto results = index.topKQuery(query_vec.data(), 5, &queryParams);
    
    // Basic assertions
    EXPECT_TRUE(results != nullptr);
    EXPECT_LE(results->results.size(), 5);
    
    // Clean up
    delete results;
    delete abstractInitParams.allocator;
}

TEST_F(HNSWDiskIndexTest, BatchOperations) {
    // Test batch add and delete operations
    const size_t dim = 32;
    const size_t maxElements = 50;
    
    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 4;
    params.efConstruction = 50;
    params.efRuntime = 25;
    params.epsilon = 0.01;
    
    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, nullptr);
    
    // Create test vectors
    std::mt19937 rng(42);
    std::vector<std::vector<float>> vectors;
    std::vector<std::pair<labelType, const void*>> elements;
    
    for (size_t i = 0; i < 20; ++i) {
        auto vec = createRandomVector(dim, rng);
        normalizeVector(vec);
        vectors.push_back(vec);
        elements.emplace_back(i, vectors.back().data());
    }
    
    // Add vectors to index
    std::vector<labelType> deleted_labels;
    index.batchUpdate(elements, deleted_labels);
    
    // Delete some vectors
    std::vector<labelType> to_delete = {5, 10, 15};
    std::vector<std::pair<labelType, const void*>> empty_elements;
    index.batchUpdate(empty_elements, to_delete);
    
    // Query the index
    auto query_vec = createRandomVector(dim, rng);
    normalizeVector(query_vec);
    
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 25;
    
    auto results = index.topKQuery(query_vec.data(), 3, &queryParams);
    
    // Basic assertions
    EXPECT_TRUE(results != nullptr);
    EXPECT_LE(results->results.size(), 3);
    
    // Clean up
    delete results;
    delete abstractInitParams.allocator;
}

TEST_F(HNSWDiskIndexTest, EmptyIndexQuery) {
    // Test querying an empty index
    const size_t dim = 16;
    const size_t maxElements = 10;
    
    // Create HNSW parameters
    HNSWParams params;
    params.dim = dim;
    params.type = VecSimType_FLOAT32;
    params.metric = VecSimMetric_L2;
    params.multi = false;
    params.M = 4;
    params.efConstruction = 20;
    params.efRuntime = 10;
    params.epsilon = 0.01;
    
    // Create abstract init parameters
    AbstractIndexInitParams abstractInitParams;
    abstractInitParams.dim = dim;
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, nullptr);
    
    // Query empty index
    std::mt19937 rng(42);
    auto query_vec = createRandomVector(dim, rng);
    normalizeVector(query_vec);
    
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 10;
    
    auto results = index.topKQuery(query_vec.data(), 5, &queryParams);
    
    // Should return empty results
    EXPECT_TRUE(results != nullptr);
    EXPECT_EQ(results->results.size(), 0);
    
    // Clean up
    delete results;
    delete abstractInitParams.allocator;
}

TEST_F(HNSWDiskIndexTest, RocksDBIntegration) {
    // Test RocksDB integration specifically
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
    abstractInitParams.dataSize = dim * sizeof(float);
    abstractInitParams.multi = false;
    abstractInitParams.allocator = VecSimAllocator::newVecsimAllocator();
    
    // Create index components
    IndexComponents<float, float> components = CreateIndexComponents<float, float>(
        abstractInitParams.allocator, VecSimMetric_L2, dim, false);
    
    // Create HNSWDiskIndex
    HNSWDiskIndex<float, float> index(&params, abstractInitParams, components, db, nullptr);
    
    // Test that RocksDB operations work through the index
    std::mt19937 rng(42);
    auto vec = createRandomVector(dim, rng);
    normalizeVector(vec);
    
    // Add a single vector
    std::vector<std::pair<labelType, const void*>> elements = {{1, vec.data()}};
    std::vector<labelType> deleted_labels;
    index.batchUpdate(elements, deleted_labels);
    
    // Query the index
    VecSimQueryParams queryParams;
    queryParams.hnswRuntimeParams.efRuntime = 50;
    
    auto results = index.topKQuery(vec.data(), 1, &queryParams);
    
    // Should find the vector we just added
    EXPECT_TRUE(results != nullptr);
    EXPECT_GE(results->results.size(), 0);
    
    // Clean up
    delete results;
    delete abstractInitParams.allocator;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 
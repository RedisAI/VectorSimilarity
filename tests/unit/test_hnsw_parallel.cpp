/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "test_utils.h"
#include "VecSim/query_result_struct.h"
#include <unistd.h>
#include <random>
#include <thread>
#include <atomic>

// Helper macro to get the closest even number which is equal or lower than x.
#define FLOOR_EVEN(x) ((x) - ((x)&1))

template <typename index_type_t>
class HNSWTestParallel : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    VecSimIndex *CreateNewIndex(HNSWParams &params, bool is_multi = false) {
        return test_utils::CreateNewIndex(params, index_type_t::get_index_type(), is_multi);
    }
    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex<data_t, dist_t> *>(index);
    }
    HNSWIndex_Single<data_t, dist_t> *CastToHNSW_Single(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex_Single<data_t, dist_t> *>(index);
    }

    /* Helper methods for testing repair jobs:
     * Collect all the nodes that require repair due to the deletions, from top level down, and
     * insert them into a queue.
     */
    void CollectRepairJobs(HNSWIndex<data_t, dist_t> *hnsw_index,
                           std::vector<pair<idType, size_t>> &jobQ) {
        size_t n = hnsw_index->indexSize();
        for (labelType element_id = 0; element_id < n; element_id++) {
            if (!hnsw_index->isMarkedDeleted(element_id)) {
                continue;
            }
            ElementGraphData *element_data = hnsw_index->getGraphDataByInternalId(element_id);
            for (size_t level = 0; level <= element_data->toplevel; level++) {
                LevelData &cur_level_data = hnsw_index->getLevelData(element_data, level);

                // Go over the neighbours of the element in a specific level.
                for (size_t i = 0; i < cur_level_data.numLinks; i++) {
                    idType cur_neighbor = cur_level_data.links[i];
                    LevelData &neighbor_level_data = hnsw_index->getLevelData(cur_neighbor, level);
                    for (size_t j = 0; j < neighbor_level_data.numLinks; j++) {
                        // If the edge is bidirectional, do repair for this neighbor
                        if (neighbor_level_data.links[j] == element_id) {
                            jobQ.emplace_back(cur_neighbor, level);
                            break;
                        }
                    }
                }
                // Next, go over the rest of incoming edges (the ones that are not bidirectional)
                // and make repairs.
                for (auto incoming_edge : *cur_level_data.incomingEdges) {
                    jobQ.emplace_back(incoming_edge, level);
                }
            }
        }
    }
};

// DataTypeSet, TEST_DATA_T and TEST_DIST_T are defined in test_utils.h

TYPED_TEST_SUITE(HNSWTestParallel, DataTypeSet);

TYPED_TEST(HNSWTestParallel, parallelSearchKnn) {
    size_t n = 20000;
    size_t k = 11;
    size_t dim = 45;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 64,
                         .efConstruction = 200,
                         .efRuntime = n};
    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    size_t n_threads = MIN(8, std::thread::hardware_concurrency());
    std::atomic_int successful_searches(0);
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    // Run parallel searches where every searching thread expects to get different labels as results
    // (determined by the thread id), which are labels in the range [50+myID-5, 50+myID+5].
    auto parallel_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            // We expect to get the results with increasing order of the distance between the res
            // label and the query val (query_val, query_val-1, query_val+1, query_val-2,
            // query_val+2, ...) The score is the L2 distance between the vectors that correspond
            // the ids.
            size_t diff_id = (id > query_val) ? (id - query_val) : (query_val - id);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, (dim * (diff_id * diff_id)));
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    size_t memory_before = index->info().commonInfo.memory;
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_search, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(successful_searches, n_threads);

    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), 1);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()), 1);
    // Make sure that we properly update the allocator atomically during the searches. The expected
    // Memory delta should only be the visited nodes handler added to the pool.
    size_t max_elements = this->CastToHNSW(index)->maxElements;
    size_t expected_memory =
        memory_before + (index->info().hnswInfo.visitedNodesPoolSize - 1) *
                            (sizeof(VisitedNodesHandler) + sizeof(tag_t) * max_elements +
                             2 * sizeof(size_t) + sizeof(void *));
    ASSERT_EQ(expected_memory, index->info().commonInfo.memory);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTestParallel, parallelSearchKNNMulti) {
    size_t dim = 45;
    size_t n = 20000;
    size_t n_labels = 1000;
    size_t k = 11;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n, .M = 64, .efRuntime = n};
    VecSimIndex *index = this->CreateNewIndex(params, true);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i % n_labels, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    size_t n_threads = MIN(8, std::thread::hardware_concurrency());
    std::atomic_int successful_searches(0);
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    // Run parallel searches where every searching thread expects to get different label as results
    // (determined by the thread id), which are labels in the range [50+myID-5, 50+myID+5].
    auto parallel_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            size_t diff_id = (id > query_val) ? (id - query_val) : (query_val - id);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, (dim * ((res_index + 1) / 2) * ((res_index + 1) / 2)));
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_search, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(successful_searches, n_threads);
    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), 1);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()), 1);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTestParallel, parallelSearchCombined) {
    size_t n = 10000;
    size_t k = 11;
    size_t dim = 64;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 64,
                         .efConstruction = 200,
                         .efRuntime = n};
    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    size_t n_threads = MIN(15, std::thread::hardware_concurrency());
    std::atomic_int successful_searches(0);
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    /* Run parallel searches of three kinds: KNN, range, and batched search. */

    // In knn, we expect to get different labels as results (determined by the thread id), which are
    // labels in the range [50+myID-5, 50+myID+5].
    auto parallel_knn_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            // We expect to get the results with increasing order of the distance between the res
            // label and the query val (query_val, query_val-1, query_val+1, query_val-2,
            // query_val+2, ...) The score is the L2 distance between the vectors that correspond
            // the ids.
            size_t diff_id = std::abs(id - query_val);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, (dim * (diff_id * diff_id)));
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    auto parallel_range_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T pivot_id = 100 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, pivot_id);
        auto verify_res_by_score = [&](size_t id, double score, size_t res_index) {
            size_t diff_id = std::abs(id - pivot_id);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, dim * (diff_id * diff_id));
        };
        uint expected_num_results = 11;
        // To get 11 results in the range [pivot_id-5, pivot_id+5], set the radius as the L2 score
        // in the boundaries.
        double radius = (double)dim * pow((double)expected_num_results / 2, 2);
        runRangeQueryTest(index, query, radius, verify_res_by_score, expected_num_results,
                          BY_SCORE);
        successful_searches++;
    };

    auto parallel_batched_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
        size_t iteration_num = 0;

        // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet
        // in every iteration. The results order should be sorted by their score (distance from the
        // query vector), which means sorted from the largest id to the lowest.
        // Run different number of iterations for every thread id.
        size_t total_iterations = myID;
        size_t n_res = 5;
        while (VecSimBatchIterator_HasNext(batchIterator) && iteration_num < total_iterations) {
            std::vector<size_t> expected_ids(n_res);
            for (size_t i = 0; i < n_res; i++) {
                expected_ids[i] = (n - iteration_num * n_res - i - 1);
            }
            auto verify_res = [&](size_t id, double score, size_t res_index) {
                ASSERT_TRUE(expected_ids[res_index] == id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num, total_iterations);
        VecSimBatchIterator_Free(batchIterator);
        successful_searches++;
    };

    std::thread thread_objs[n_threads];
    size_t memory_before = index->info().commonInfo.memory;
    for (size_t i = 0; i < n_threads; i++) {
        if (i % 3 == 0) {
            thread_objs[i] = std::thread(parallel_knn_search, i);
        } else if (i % 3 == 1) {
            thread_objs[i] = std::thread(parallel_range_search, i);
        } else {
            thread_objs[i] = std::thread(parallel_batched_search, i);
        }
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(successful_searches, n_threads);
    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), 1);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()), 1);

    // Make sure that we properly update the allocator atomically during the searches.
    // Memory delta should only be the visited nodes handler added to the pool.
    size_t max_elements = this->CastToHNSW(index)->maxElements;
    size_t expected_memory =
        memory_before + (index->info().hnswInfo.visitedNodesPoolSize - 1) *
                            (sizeof(VisitedNodesHandler) + sizeof(tag_t) * max_elements +
                             2 * sizeof(size_t) + sizeof(void *));
    ASSERT_EQ(expected_memory, index->info().commonInfo.memory);
    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTestParallel, parallelInsert) {
    size_t n = 10000;
    size_t k = 11;
    size_t dim = 32;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *parallel_index = this->CreateNewIndex(params);
    size_t n_threads = 10;

    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    auto parallel_insert = [&](int myID) {
        for (labelType label = myID; label < n; label += n_threads) {
            completed_tasks[myID]++;
            GenerateAndAddVector<TEST_DATA_T>(parallel_index, dim, label, label);
        }
    };
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_insert, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(VecSimIndex_IndexSize(parallel_index), n);
    // Validate that every thread executed n/n_threads jobs.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), n / n_threads);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()),
              ceil((double)n / n_threads));

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, (TEST_DATA_T)n / 2);
    auto verify_res = [&](size_t id, double score, size_t res_index) {
        // We expect to get the results with increasing order of the distance between the res
        // label and the query val (n/2, n/2-1, n/2+1, n/2-2, n/2+2, ...) The score is the L2
        // distance between the vectors that correspond the ids.
        size_t diff_id = std::abs(int(id - n / 2));
        ASSERT_EQ(diff_id, (res_index + 1) / 2);
        ASSERT_EQ(score, (dim * (diff_id * diff_id)));
    };
    runTopKSearchTest(parallel_index, query, k, verify_res);
    VecSimIndex_Free(parallel_index);
}

TYPED_TEST(HNSWTestParallel, parallelInsertMulti) {
    size_t n = 10000;
    size_t n_labels = 1000;
    size_t per_label = n / n_labels;
    size_t k = 11;
    size_t dim = 32;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *parallel_index = this->CreateNewIndex(params, true);
    size_t n_threads = 10;

    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);
    auto parallel_insert = [&](int myID) {
        for (size_t i = myID; i < n; i += n_threads) {
            completed_tasks[myID]++;
            GenerateAndAddVector<TEST_DATA_T>(parallel_index, dim, i % n_labels, i);
        }
    };
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_insert, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(VecSimIndex_IndexSize(parallel_index), n);
    // Validate that every thread executed n/n_threads jobs.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), n / n_threads);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()),
              ceil((double)n / n_threads));

    TEST_DATA_T query[dim];
    TEST_DATA_T query_val = (TEST_DATA_T)n / 2 + 10;
    GenerateVector<TEST_DATA_T>(query, dim, (TEST_DATA_T)query_val);
    auto verify_res = [&](size_t id, double score, size_t res_index) {
        // We expect to get the results with increasing order of the distance between the res
        // label and query_val%n_labels (that is ids 10, 9, 11, ... for the current arguments).
        // The score is the L2 distance between the vectors that correspond the ids.
        size_t diff_id = std::abs(int(id - (size_t)query_val % n_labels));
        ASSERT_EQ(diff_id, (res_index + 1) / 2);
        ASSERT_EQ(score, (dim * (diff_id * diff_id)));
    };
    runTopKSearchTest(parallel_index, query, k, verify_res);
    VecSimIndex_Free(parallel_index);
}

TYPED_TEST(HNSWTestParallel, parallelInsertSearch) {
    size_t n = 10000;
    size_t k = 11;
    size_t dim = 32;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 64,
                         .efConstruction = 200,
                         .efRuntime = n};

    for (bool is_multi : {true, false}) {
        VecSimIndex *parallel_index = this->CreateNewIndex(params, is_multi);
        size_t n_threads = MIN(10, FLOOR_EVEN(std::thread::hardware_concurrency()));
        // Save the number fo tasks done by thread i in the i-th entry.
        std::vector<size_t> completed_tasks(n_threads, 0);

        auto parallel_insert = [&](int myID) {
            for (labelType label = myID; label < n; label += n_threads / 2) {
                completed_tasks[myID]++;
                GenerateAndAddVector<TEST_DATA_T>(parallel_index, dim, label, label);
            }
        };

        TEST_DATA_T query_val = (TEST_DATA_T)n / 4;
        std::atomic_int successful_searches(0);
        auto parallel_knn_search = [&](int myID) {
            completed_tasks[myID]++;
            // Make sure were still indexing in parallel to the search (at most 90% if the vectors
            // were already indexed).
            ASSERT_LT(VecSimIndex_IndexSize(parallel_index), 0.9 * n);
            TEST_DATA_T query[dim];
            GenerateVector<TEST_DATA_T>(query, dim, query_val);
            auto verify_res = [&](size_t id, double score, size_t res_index) {
                // We expect to get the results with increasing order of the distance between the
                // res label and the query val (n/4, n/4-1, n/4+1, n/4-2, n/4+2, ...) The score is
                // the L2 distance between the vectors that correspond the ids.
                size_t diff_id = std::abs(int(id - query_val));
                ASSERT_EQ(diff_id, (res_index + 1) / 2);
                ASSERT_EQ(score, (dim * (diff_id * diff_id)));
            };
            runTopKSearchTest(parallel_index, query, k, verify_res);
            successful_searches++;
        };

        auto hnsw_index = this->CastToHNSW(parallel_index);
        std::thread thread_objs[n_threads];
        for (size_t i = 0; i < n_threads; i++) {
            if (i < n_threads / 2) {
                thread_objs[i] = std::thread(parallel_insert, i);
            } else {
                // Search threads are waiting in bust wait until the vectors of the query results
                // are done being indexed.
                bool wait_for_results = true;
                while (wait_for_results) {
                    wait_for_results = false;
                    for (labelType res_label = query_val - k / 2; res_label <= query_val + k / 2;
                         res_label++) {
                        if (!hnsw_index->safeCheckIfLabelExistsInIndex(res_label, true)) {
                            wait_for_results = true;
                            break; // results are not ready yet, restart the check.
                        }
                    }
                }
                thread_objs[i] = std::thread(parallel_knn_search, i);
            }
        }
        for (size_t i = 0; i < n_threads; i++) {
            thread_objs[i].join();
        }
        ASSERT_EQ(VecSimIndex_IndexSize(parallel_index), n);
        ASSERT_EQ(successful_searches, ceil(double(n_threads) / 2));
        // Validate that every insertion thread executed n/(n_threads/2_ jobs).
        ASSERT_EQ(
            *std::min_element(completed_tasks.begin(), completed_tasks.begin() + n_threads / 2),
            n / (n_threads / 2));
        ASSERT_EQ(
            *std::max_element(completed_tasks.begin(), completed_tasks.begin() + n_threads / 2),
            ceil((double)n / (n_threads / 2)));
        // Validate that every search thread executed a single job.
        ASSERT_EQ(*std::min_element(completed_tasks.begin() + n_threads / 2, completed_tasks.end()),
                  1);
        ASSERT_EQ(*std::max_element(completed_tasks.begin() + n_threads / 2, completed_tasks.end()),
                  1);
        VecSimIndex_Free(parallel_index);
    }
}

TYPED_TEST(HNSWTestParallel, parallelRepairs) {
    size_t n = 1000;
    size_t dim = 32;

    HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n};

    auto *hnsw_index = this->CastToHNSW(this->CreateNewIndex(params));
    size_t n_threads = MIN(10, std::thread::hardware_concurrency());
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    // Create some random vectors and insert them to the index.
    std::srand(10); // create pseudo random generator with ana arbitrary seed.
    for (size_t i = 0; i < n; i++) {
        TEST_DATA_T vector[dim];
        for (size_t j = 0; j < dim; j++) {
            vector[j] = std::rand() / (TEST_DATA_T)RAND_MAX;
        }
        VecSimIndex_AddVector(hnsw_index, vector, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(hnsw_index), n);

    // Queue of repair jobs, each job is represented as {id, level}
    auto jobQ = std::vector<pair<idType, size_t>>();

    // Collect all the nodes that require repairment due to the deletions, from top level down.
    for (size_t element_id = 0; element_id < n; element_id += 2) {
        hnsw_index->markDelete(element_id);
    }
    ASSERT_EQ(hnsw_index->getNumMarkedDeleted(), n / 2);
    // Every that every deleted node should have at least 2 connections to repair.
    auto report = hnsw_index->checkIntegrity();
    ASSERT_GE(report.connections_to_repair, n);

    this->CollectRepairJobs(hnsw_index, jobQ);
    size_t n_jobs = jobQ.size();
    ASSERT_EQ(report.connections_to_repair, n_jobs);

    auto executeRepairJobs = [&](int myID) {
        for (size_t i = myID; i < n_jobs; i += n_threads) {
            auto job = jobQ[i];
            hnsw_index->repairNodeConnections(job.first, job.second); // {element_id, level}
            completed_tasks[myID]++;
        }
    };

    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(executeRepairJobs, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    // Check index integrity, also make sure that no node is pointing to a deleted node.
    report = hnsw_index->checkIntegrity();
    ASSERT_TRUE(report.valid_state);
    ASSERT_EQ(report.connections_to_repair, 0);

    // Validate that the tasks are spread among the threads uniformly.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()),
              floorf((float)n_jobs / n_threads));
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()),
              ceilf((float)n_jobs / n_threads));
    VecSimIndex_Free(hnsw_index);
}

TYPED_TEST(HNSWTestParallel, parallelRepairSearch) {
    size_t n = 10000;
    size_t k = 10;
    size_t dim = 32;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n, .efRuntime = n};

    auto *hnsw_index = this->CastToHNSW(this->CreateNewIndex(params));
    size_t n_threads = MIN(10, FLOOR_EVEN(std::thread::hardware_concurrency()));
    // Save the number of tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(hnsw_index), n);

    // Queue of repair jobs, each job is represented as {id, level}
    auto jobQ = std::vector<pair<idType, size_t>>();

    for (size_t element_id = 0; element_id < n; element_id += 2) {
        hnsw_index->markDelete(element_id);
    }
    ASSERT_EQ(hnsw_index->getNumMarkedDeleted(), n / 2);
    // Every deleted node i should have at least 2 connection to repair (to i-1 and i+1), except for
    // 0 and n-1 that has at least one connection to repair.
    ASSERT_GE(hnsw_index->checkIntegrity().connections_to_repair, n - 2);

    // Collect all the nodes that require repairment due to the deletions, from top level down.
    this->CollectRepairJobs(hnsw_index, jobQ);
    size_t n_jobs = jobQ.size();

    auto executeRepairJobs = [&](int myID) {
        for (size_t i = myID; i < n_jobs; i += n_threads / 2) {
            auto job = jobQ[i];
            hnsw_index->repairNodeConnections(job.first, job.second); // {element_id, level}
            completed_tasks[myID]++;
        }
    };

    bool run_queries = true;
    auto parallel_knn_search = [&](int myID) {
        TEST_DATA_T query_val = (TEST_DATA_T)n / 4 + 2 * myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            // We expect to get the results with increasing order of the distance between the
            // res label and the query val and only odd labels (query_val-1, query_val+1,
            // query_val-3, query_val+3, ...) The score is the L2 distance between the vectors that
            // correspond the ids.
            size_t diff_id = std::abs(int(id - query_val));
            ASSERT_EQ(diff_id, res_index + (1 - res_index % 2));
            ASSERT_EQ(score, (dim * (diff_id * diff_id)));
        };
        do {
            runTopKSearchTest(hnsw_index, query, k, verify_res);
            completed_tasks[myID]++;
        } while (run_queries);
    };

    std::thread thread_objs[n_threads];
    // Run queries, expect to get only non-deleted vector as results.
    for (size_t i = n_threads / 2; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_knn_search, i);
    }

    // Run the repair jobs.
    for (size_t i = 0; i < n_threads / 2; i++) {
        thread_objs[i] = std::thread(executeRepairJobs, i);
    }
    for (size_t i = 0; i < n_threads / 2; i++) {
        thread_objs[i].join();
    }
    // Once all the repair jobs are done, signal the query threads to finish.
    run_queries = false;
    for (size_t i = n_threads / 2; i < n_threads; i++) {
        thread_objs[i].join();
    }

    // Check index integrity, also make sure that no node is pointing to a deleted node.
    auto report = hnsw_index->checkIntegrity();
    ASSERT_TRUE(report.valid_state);
    ASSERT_EQ(report.connections_to_repair, 0);

    // Validate that every search thread ran at least one job.
    ASSERT_GE(*std::min_element(completed_tasks.begin() + n_threads / 2, completed_tasks.end()), 1);
    // Validate that the repair tasks are spread among the threads uniformly.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.begin() + n_threads / 2),
              floorf((float)n_jobs / (n_threads / 2.0)));
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.begin() + n_threads / 2),
              ceilf((float)n_jobs / (n_threads / 2.0)));
    VecSimIndex_Free(hnsw_index);
}

TYPED_TEST(HNSWTestParallel, parallelRepairInsert) {
    size_t n = 1000;
    size_t k = 11;
    size_t dim = 32;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n, .efRuntime = n};

    auto *hnsw_index = this->CastToHNSW(this->CreateNewIndex(params));
    size_t n_threads = MIN(8, FLOOR_EVEN(std::thread::hardware_concurrency()));
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    // Insert n/2 vectors to the index.
    for (size_t i = 0; i < n / 2; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(hnsw_index), n / 2);

    // Queue of repair jobs, each job is represented as {id, level}
    auto jobQ = std::vector<pair<idType, size_t>>();
    for (size_t element_id = 0; element_id < n / 2; element_id += 2) {
        hnsw_index->markDelete(element_id);
    }
    ASSERT_EQ(hnsw_index->getNumMarkedDeleted(), n / 4);
    // Every deleted node i should have at least 2 connection to repair (to i-1 and i-1), except for
    // 0 that has at least one connection to repair.
    ASSERT_GE(hnsw_index->checkIntegrity().connections_to_repair, n / 2 - 1);

    // Collect all the nodes that require repairment due to the deletions, from top level down.
    this->CollectRepairJobs(hnsw_index, jobQ);
    size_t n_jobs = jobQ.size();

    auto executeRepairJobs = [&](int myID) {
        for (size_t i = myID - n_threads / 2; i < n_jobs; i += n_threads / 2) {
            auto job = jobQ[i];
            hnsw_index->repairNodeConnections(job.first, job.second); // {element_id, level}
            completed_tasks[myID]++;
        }
    };

    auto parallel_insert = [&](int myID) {
        // Reinsert the even ids that were deleted, and n/4 more even ids.
        for (labelType label = 2 * myID; label < n; label += n_threads) {
            completed_tasks[myID]++;
            GenerateAndAddVector<TEST_DATA_T>(hnsw_index, dim, label, label);
        }
    };

    std::thread thread_objs[n_threads];

    // Insert n/2 new vectors while we repair connections.
    for (size_t i = 0; i < n_threads / 2; i++) {
        thread_objs[i] = std::thread(parallel_insert, i);
    }
    for (size_t i = n_threads / 2; i < n_threads; i++) {
        thread_objs[i] = std::thread(executeRepairJobs, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    // Check index integrity, also make sure that no node is pointing to a deleted node.
    ASSERT_EQ(hnsw_index->indexSize(), n);
    auto report = hnsw_index->checkIntegrity();
    ASSERT_TRUE(report.valid_state);
    ASSERT_EQ(report.connections_to_repair, 0);

    // Validate that the repair tasks are spread among the threads uniformly.
    ASSERT_EQ(*std::min_element(completed_tasks.begin() + n_threads / 2, completed_tasks.end()),
              floorf((float)n_jobs / (n_threads / 2.0)));
    ASSERT_EQ(*std::max_element(completed_tasks.begin() + n_threads / 2, completed_tasks.end()),
              ceilf((float)n_jobs / (n_threads / 2.0)));

    // Run queries to validate the index new state.
    TEST_DATA_T query[dim];
    // Around 3n/4 we only have even numbers vectors.
    size_t query_val = 3 * n / 4;
    GenerateVector<TEST_DATA_T>(query, dim, query_val);
    auto verify_res_even = [&](size_t id, double score, size_t res_index) {
        // We expect to get the results with increasing order of the distance between the
        // res label and the query val (3n/4, 3n/4 - 2, 3n/4 + 2, 3n/4 - 4 3n/4 + 4, ...) The score
        // is the L2 distance between the vectors that correspond the ids.
        size_t diff_id = std::abs(int(id - query_val));
        ASSERT_EQ(diff_id, res_index % 2 ? res_index + 1 : res_index);
        ASSERT_EQ(score, (dim * (diff_id * diff_id)));
    };
    runTopKSearchTest(hnsw_index, query, k, verify_res_even);

    // Around n/4 we should have all vectors (even and odd).
    query_val = n / 4;
    GenerateVector<TEST_DATA_T>(query, dim, query_val);
    auto verify_res = [&](size_t id, double score, size_t res_index) {
        // We expect to get the results with increasing order of the distance between the
        // res label and the query val (n/4, n/4 - 1, n/4 + 1, n/4 - 2 n/4 + 2, ...) The score
        // is the L2 distance between the vectors that correspond the ids.
        size_t diff_id = std::abs(int(id - query_val));
        ASSERT_EQ(diff_id, (res_index + 1) / 2);
        ASSERT_EQ(score, (dim * (diff_id * diff_id)));
    };
    runTopKSearchTest(hnsw_index, query, k, verify_res);
    VecSimIndex_Free(hnsw_index);
}

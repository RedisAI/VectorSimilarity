/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <functional>
#include <cmath>
#include <exception>
#include <thread>

#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "gtest/gtest.h"

// IndexType is used to define indices unit tests
template <VecSimType type, typename DataType, typename DistType = DataType>
struct IndexType {
    static VecSimType get_index_type() { return type; }
    typedef DataType data_t;
    typedef DistType dist_t;
};

#define TEST_DATA_T typename TypeParam::data_t
#define TEST_DIST_T typename TypeParam::dist_t

using DataTypeSet =
    ::testing::Types<IndexType<VecSimType_FLOAT32, float>, IndexType<VecSimType_FLOAT64, double>>;

// Define index type for tests that can be automatically generated for single and multi.
template <VecSimType type, bool IsMulti, typename DataType, typename DistType = DataType>
struct IndexTypeExtended {
    static VecSimType get_index_type() { return type; }
    static bool isMulti() { return IsMulti; }
    typedef DataType data_t;
    typedef DistType dist_t;
};

using DataTypeSetExtended = ::testing::Types<IndexTypeExtended<VecSimType_FLOAT32, false, float>,
                                             IndexTypeExtended<VecSimType_FLOAT32, true, float>,
                                             IndexTypeExtended<VecSimType_FLOAT64, false, double>,
                                             IndexTypeExtended<VecSimType_FLOAT64, true, double>>;

template <typename data_t>
static void GenerateVector(data_t *output, size_t dim, data_t value = 1.0) {
    for (size_t i = 0; i < dim; i++) {
        output[i] = (data_t)value;
    }
}

template <typename data_t>
int GenerateAndAddVector(VecSimIndex *index, size_t dim, size_t id, data_t value = 1.0) {
    data_t v[dim];
    GenerateVector(v, dim, value);
    return VecSimIndex_AddVector(index, v, id);
}

inline VecSimParams CreateParams(const HNSWParams &hnsw_params) {
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB, .hnswParams = hnsw_params};
    return params;
}

inline VecSimParams CreateParams(const BFParams &bf_params) {
    VecSimParams params{.algo = VecSimAlgo_BF, .bfParams = bf_params};
    return params;
}

inline VecSimParams CreateParams(const TieredIndexParams &tiered_params) {
    VecSimParams params{.algo = VecSimAlgo_TIERED, .tieredParams = tiered_params};
    return params;
}

namespace test_utils {
template <typename IndexParams>
inline VecSimIndex *CreateNewIndex(IndexParams &index_params, VecSimType type,
                                   bool is_multi = false) {
    index_params.type = type;
    index_params.multi = is_multi;
    VecSimParams params = CreateParams(index_params);
    return VecSimIndex_New(&params);
}

extern VecsimQueryType query_types[4];

} // namespace test_utils
template <typename IndexParams>
size_t EstimateInitialSize(const IndexParams &index_params) {
    VecSimParams params = CreateParams(index_params);
    return VecSimIndex_EstimateInitialSize(&params);
}

template <typename IndexParams>
size_t EstimateElementSize(const IndexParams &index_params) {
    VecSimParams params = CreateParams(index_params);
    return VecSimIndex_EstimateElementSize(&params);
}

VecSimQueryParams CreateQueryParams(const HNSWRuntimeParams &RuntimeParams);

inline void ASSERT_TYPE_EQ(double arg1, double arg2) { ASSERT_DOUBLE_EQ(arg1, arg2); }

inline void ASSERT_TYPE_EQ(float arg1, float arg2) { ASSERT_FLOAT_EQ(arg1, arg2); }
void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k, size_t expected_res_num,
                       std::function<void(size_t, double, size_t)> ResCB,
                       VecSimQueryParams *params = nullptr,
                       VecSimQueryResult_Order order = BY_SCORE);
void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k,
                       std::function<void(size_t, double, size_t)> ResCB,
                       VecSimQueryParams *params = nullptr,
                       VecSimQueryResult_Order order = BY_SCORE);

void runBatchIteratorSearchTest(VecSimBatchIterator *batch_iterator, size_t n_res,
                                std::function<void(size_t, double, size_t)> ResCB,
                                VecSimQueryResult_Order order = BY_SCORE,
                                size_t expected_n_res = -1);

void compareCommonInfo(CommonInfo info1, CommonInfo info2);
void compareFlatInfo(bfInfoStruct info1, bfInfoStruct info2);
void compareHNSWInfo(hnswInfoStruct info1, hnswInfoStruct info2);

void compareFlatIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter);

void compareHNSWIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter);

void runRangeQueryTest(VecSimIndex *index, const void *query, double radius,
                       const std::function<void(size_t, double, size_t)> &ResCB,
                       size_t expected_res_num, VecSimQueryResult_Order order = BY_ID,
                       VecSimQueryParams *params = nullptr);

size_t getLabelsLookupNodeSize();

inline double GetInfVal(VecSimType type) {
    if (type == VecSimType_FLOAT64) {
        return exp(500);
    } else if (type == VecSimType_FLOAT32) {
        return exp(60);
    } else {
        throw std::invalid_argument("This type is not supported");
    }
}

// Test a specific exception type is thrown and prints the right message.
#define ASSERT_EXCEPTION_MESSAGE(VALUE, EXCEPTION_TYPE, MESSAGE)                                   \
    try {                                                                                          \
        VALUE;                                                                                     \
        FAIL() << "exception '" << MESSAGE << "' not thrown at all!";                              \
    } catch (const EXCEPTION_TYPE &e) {                                                            \
        EXPECT_EQ(std::string(MESSAGE), e.what())                                                  \
            << " exception message is incorrect. Expected the following "                          \
               "message:\n\n"                                                                      \
            << MESSAGE << "\n";                                                                    \
    } catch (...) {                                                                                \
        FAIL() << "exception '" << MESSAGE << "' not thrown with expected type '"                  \
               << #EXCEPTION_TYPE << "'!";                                                         \
    }

namespace tiered_index_mock {

typedef struct RefManagedJob {
    AsyncJob *job;
    std::weak_ptr<VecSimIndex> index_weak_ref;
} RefManagedJob;

struct SearchJobMock : public AsyncJob {
    void *query; // The query vector. ownership is passed to the job in the constructor.
    size_t k;    // The number of results to return.
    size_t n;    // The number of vectors in the index (might be useful for the mock)
    size_t dim;  // The dimension of the vectors in the index (might be useful for the mock)
    std::atomic_int &successful_searches; // A reference to a shared counter that counts the number
                                          // of successful searches.
    SearchJobMock(std::shared_ptr<VecSimAllocator> allocator, JobCallback searchCB,
                  VecSimIndex *index_, void *query_, size_t k_, size_t n_, size_t dim_,
                  std::atomic_int &successful_searches_)
        : AsyncJob(allocator, HNSW_SEARCH_JOB, searchCB, index_), query(query_), k(k_), n(n_),
          dim(dim_), successful_searches(successful_searches_) {}
    ~SearchJobMock() { this->allocator->free_allocation(query); }
};

using JobQueue = std::queue<RefManagedJob>;
int submit_callback(void *job_queue, AsyncJob **jobs, size_t len, void *index_ctx);
int update_mem_callback(void *mem_ctx, size_t mem);

typedef struct IndexExtCtx {
    std::shared_ptr<VecSimIndex> index_strong_ref;
} IndexExtCtx;

static const size_t THREAD_POOL_SIZE = MIN(16, std::thread::hardware_concurrency());
extern std::vector<std::thread> thread_pool;
extern std::mutex queue_guard;
extern std::condition_variable queue_cond;

void thread_main_loop(JobQueue &jobQ, bool &run_thread);

void thread_pool_join(JobQueue &jobQ, bool &run_thread);

} // namespace tiered_index_mock

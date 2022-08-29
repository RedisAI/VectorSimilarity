#pragma once

#include <functional>
#include "VecSim/vec_sim.h"

void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k,
                       std::function<void(size_t, double, size_t)> ResCB,
                       VecSimQueryParams *params = nullptr,
                       VecSimQueryResult_Order order = BY_SCORE);

void runBatchIteratorSearchTest(VecSimBatchIterator *batch_iterator, size_t n_res,
                                std::function<void(size_t, double, size_t)> ResCB,
                                VecSimQueryResult_Order order = BY_SCORE,
                                size_t expected_n_res = -1);

void compareFlatIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter);

void compareHNSWIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter);

void runRangeQueryTest(VecSimIndex *index, const void *query, float radius,
                       const std::function<void(size_t, double, size_t)> &ResCB,
                       size_t expected_res_num, VecSimQueryResult_Order order = BY_ID,
                       VecSimQueryParams *params = nullptr);

size_t getLabelsLookupNodeSize();

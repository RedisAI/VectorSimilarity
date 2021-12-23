#pragma once

#include <functional>
#include "VecSim/vec_sim.h"

void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k,
                       std::function<void(int, float, int)> ResCB,
                       VecSimQueryParams *params = nullptr,
                       VecSimQueryResult_Order order = BY_SCORE);

void runBatchIteratorSearchTest(VecSimBatchIterator *batch_iterator, size_t n_res,
                                std::function<void(int, float, int)> ResCB,
                                VecSimQueryResult_Order order = BY_SCORE,
                                size_t expected_n_res = -1);

void compareFlatIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter);

void compareHNSWIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter);

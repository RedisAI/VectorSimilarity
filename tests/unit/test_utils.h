#pragma once

#include <functional>
#include "VecSim/vec_sim.h"

void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k,
                       std::function<void(int, float, int)> ResCB,
                       VecSimQueryParams *params = nullptr,
                       Index_TopKQuery TopKQueryFunc = VecSimIndex_TopKQuery);
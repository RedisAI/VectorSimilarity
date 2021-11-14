#pragma once

#include <stdlib.h>
#include <VecSim/query_results.h>

void float_vector_normalize(float *x, size_t dim);

void sort_results_by_id(VecSimQueryResult_List results);

void sort_results_by_score(VecSimQueryResult_List results);

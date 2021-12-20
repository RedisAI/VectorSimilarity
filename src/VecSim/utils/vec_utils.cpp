#include "vec_utils.h"
#include "VecSim/query_result_struct.h"
#include <cmath>
#include <cassert>

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void *, const void *);
#endif

int cmpVecSimQueryResultById(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    return (int)(VecSimQueryResult_GetId(res1) - VecSimQueryResult_GetId(res2));
}

int cmpVecSimQueryResultByScore(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    assert(!std::isnan(VecSimQueryResult_GetScore(res1)) &&
           !std::isnan(VecSimQueryResult_GetScore(res2)));
    // Compare floats
    return (VecSimQueryResult_GetScore(res1) - VecSimQueryResult_GetScore(res2)) >= 0.0 ? 1 : -1;
}

void float_vector_normalize(float *x, size_t dim) {
    float sum = 0;
    for (size_t i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    float norm = sqrt(sum);
    if (norm == 0)
        return;
    for (size_t i = 0; i < dim; i++) {
        x[i] = x[i] / norm;
    }
}

void sort_results_by_id(VecSimQueryResult_List results) {
    qsort(results, VecSimQueryResult_Len(results), sizeof(VecSimQueryResult),
          (__compar_fn_t)cmpVecSimQueryResultById);
}

void sort_results_by_score(VecSimQueryResult_List results) {
    qsort(results, VecSimQueryResult_Len(results), sizeof(VecSimQueryResult),
          (__compar_fn_t)cmpVecSimQueryResultByScore);
}

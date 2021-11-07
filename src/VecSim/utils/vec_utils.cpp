#include "vec_utils.h"
#include "VecSim/query_result_struct.h"
#include <math.h>

int cmpVecSimQueryResult(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    return VecSimQueryResult_GetId(res1) > VecSimQueryResult_GetId(res2)
               ? 1
               : (VecSimQueryResult_GetId(res1) < VecSimQueryResult_GetId(res2) ? -1 : 0);
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
          (__compar_fn_t)cmpVecSimQueryResult);
}

#include "vec_utils.h"
#include "OpenBLAS/cblas.h"

void cpu_float_vector_normalize(float *x, int dim) {
    int step = 1;
    float norm = 1.0f / snrm2(&dim, x, &step);
    sscal(&dim, &norm, x, &step);
}

float cpu_l2(float *x, float *y, int dim) {
    int step = 1;
    float xMinusY[dim];
    saxpby scopy(&dim, y, &step, s)
}

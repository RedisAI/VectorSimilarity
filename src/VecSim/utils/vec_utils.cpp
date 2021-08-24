#include "vec_utils.h"
#include "OpenBLAS/cblas.h"

void float_vector_normalize(float *x, int dim) {
    float norm = cblas_snrm2(dim, x, 1);
    norm = norm == 0.0 ? 0 : 1.0f / norm;
    cblas_sscal(dim, norm, x, 1);
}

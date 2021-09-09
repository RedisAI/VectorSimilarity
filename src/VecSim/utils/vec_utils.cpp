#include "vec_utils.h"
#include "math.h"

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

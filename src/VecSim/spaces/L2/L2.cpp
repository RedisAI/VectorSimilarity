#include "L2.h"

float L2Sqr(const float *pVect1v, const float *pVect2v, size_t qty) {

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = pVect1v[i] - pVect2v[i];
        res += t * t;
    }
    return res;
}

double L2Sqr(const double *pVect1v, const double *pVect2v, size_t qty) {

    double res = 0;
    for (size_t i = 0; i < qty; i++) {
        double t = pVect1v[i] - pVect2v[i];
        res += t * t;
    }
    return res;
}

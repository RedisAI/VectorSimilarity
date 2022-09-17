#include "IP.h"

float InnerProduct_impl(const float *pVect1, const float *pVect2, size_t qty) {
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += pVect1[i] * pVect2[i];
    }
    return res;
}

float InnerProduct(const float *pVect1, const float *pVect2, size_t qty) {
    return 1.0f - InnerProduct_impl(pVect1, pVect2, qty);
}

double InnerProduct_impl(const double *pVect1, const double *pVect2, size_t qty) {
    double res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += pVect1[i] * pVect2[i];
    }
    return res;
}

double InnerProduct(const double *pVect1, const double *pVect2, size_t qty) {
    return 1.0 - InnerProduct_impl(pVect1, pVect2, qty);
}

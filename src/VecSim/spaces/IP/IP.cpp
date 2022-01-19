
#include "IP.h"

#include <stdlib.h>

float InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    double res = 0;
    double e1, e2;
    for (unsigned i = 0; i < qty; i++) {
        e1 = ((float *)pVect1)[i];
        e2 = ((float *)pVect2)[i];
        res += e1 * e2;
    }
    return 1.0f - res;
}

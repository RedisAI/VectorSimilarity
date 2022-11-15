/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "info_iterator_struct.h"

extern "C" size_t VecSimInfoIterator_NumberOfFields(VecSimInfoIterator *infoIterator) {
    return infoIterator->numberOfFields();
}

extern "C" bool VecSimInfoIterator_HasNextField(VecSimInfoIterator *infoIterator) {
    return infoIterator->hasNext();
}

extern "C" VecSim_InfoField *VecSimInfoIterator_NextField(VecSimInfoIterator *infoIterator) {
    if (infoIterator->hasNext()) {
        return infoIterator->next();
    }
    return NULL;
}

extern "C" void VecSimInfoIterator_Free(VecSimInfoIterator *infoIterator) {
    if (infoIterator != NULL) {
        delete infoIterator;
    }
}

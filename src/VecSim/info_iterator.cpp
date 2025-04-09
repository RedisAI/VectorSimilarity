/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "info_iterator_struct.h"

extern "C" size_t VecSimDebugInfoIterator_NumberOfFields(VecSimDebugInfoIterator *infoIterator) {
    return infoIterator->numberOfFields();
}

extern "C" bool VecSimDebugInfoIterator_HasNextField(VecSimDebugInfoIterator *infoIterator) {
    return infoIterator->hasNext();
}

extern "C" VecSim_InfoField *
VecSimDebugInfoIterator_NextField(VecSimDebugInfoIterator *infoIterator) {
    if (infoIterator->hasNext()) {
        return infoIterator->next();
    }
    return NULL;
}

extern "C" void VecSimDebugInfoIterator_Free(VecSimDebugInfoIterator *infoIterator) {
    if (infoIterator != NULL) {
        delete infoIterator;
    }
}

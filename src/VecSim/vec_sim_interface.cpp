/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/vec_sim_interface.h"
#include "VecSim/utils/vec_utils.h"
#include <cstdarg>
#include <iostream>

// Print log messages to stdout
void Vecsim_Log(void *ctx, const char *message) { std::cout << message << std::endl; }

timeoutCallbackFunction VecSimIndexInterface::timeoutCallback = [](void *ctx) { return 0; };
logCallbackFunction VecSimIndexInterface::logCallback = Vecsim_Log;

int VecSimIndexInterface::addVector(const void *blob, labelType label, void *auxiliaryCtx) {
    const void *processed_blob = processBlob(blob);
    int ret = addVectorImp(processed_blob, label, auxiliaryCtx);
    returnProcessedBlob(processed_blob);
    return ret;
}

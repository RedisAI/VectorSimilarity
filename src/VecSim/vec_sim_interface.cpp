/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "VecSim/vec_sim_interface.h"
#include <cstdarg>
#include <iostream>

// Print log messages to stdout
void Vecsim_Log(void *ctx, const char *level, const char *message) {
    std::cout << level << ": " << message << std::endl;
}

timeoutCallbackFunction VecSimIndexInterface::timeoutCallback = [](void *ctx) { return 0; };
logCallbackFunction VecSimIndexInterface::logCallback = Vecsim_Log;
VecSimWriteMode VecSimIndexInterface::asyncWriteMode = VecSim_WriteAsync;

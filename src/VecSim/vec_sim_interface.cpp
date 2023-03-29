/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/vec_sim_interface.h"
#include <cstdarg>
#include <iostream>

// Print log messages to stdout
void Vecsim_Log(void *ctx, const char *message) { std::cout << message << std::endl; }

timeoutCallbackFunction VecSimIndexInterface::timeoutCallback = [](void *ctx) { return 0; };
logCallbackFunction VecSimIndexInterface::logCallback = Vecsim_Log;
void *VecSimIndexInterface::logCallbackCtx = nullptr;

void VecSimIndexInterface::log(const char *fmt, ...) const {
    if (logCallback) {
        // Format the message and call the callback
        va_list args;
        va_start(args, fmt);
        int len = vsnprintf(NULL, 0, fmt, args);
        va_end(args);
        char *buf = new char[len + 1];
        va_start(args, fmt);
        vsnprintf(buf, len + 1, fmt, args);
        va_end(args);
        logCallback(logCallbackCtx, buf);
        delete[] buf;
    }
}

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
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <filesystem>

// Global variable to store the current log context
static const char* g_log_context = nullptr;

void VecSim_SetLogContext(const char* context) {
    g_log_context = context;
}

// writes the logs to a file
void Vecsim_Log(void *ctx, const char *level, const char *message) {
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    // Format timestamp
    std::tm* tm_info = std::localtime(&time_t);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
    
    // Format log entry
    std::string log_entry = "[" + std::string(timestamp) + "." + 
                           std::string(3 - std::to_string(ms.count()).length(), '0') + 
                           std::to_string(ms.count()) + "] [" + level + "] " + message;
    
    // Use provided context or fall back to global context
    const char* log_context = ctx ? static_cast<const char*>(ctx) : g_log_context;
    
    // Default log path if no context is provided
    std::string log_path = "logs/tests/flow/vecsim.log";
    
    if (log_context && strlen(log_context) > 0) {
        std::string context_str(log_context);
        
        // Extract just the filename without extension from the full path
        size_t last_slash = context_str.find_last_of("/\\");
        size_t filename_start = (last_slash == std::string::npos) ? 0 : last_slash + 1;
        
        size_t extension_pos = context_str.find_last_of(".");
        std::string filename;
        
        if (extension_pos != std::string::npos && extension_pos > filename_start) {
            filename = context_str.substr(filename_start, extension_pos - filename_start);
        } else {
            filename = context_str.substr(filename_start);
        }
        
        log_path = "logs/tests/flow/" + filename + ".log";
    }
    
    // Write to file
    std::ofstream log_file(log_path, std::ios::app);
    if (log_file.is_open()) {
        log_file << log_entry << std::endl;
        log_file.close();
    }

}

timeoutCallbackFunction VecSimIndexInterface::timeoutCallback = [](void *ctx) { return 0; };
logCallbackFunction VecSimIndexInterface::logCallback = Vecsim_Log;
VecSimWriteMode VecSimIndexInterface::asyncWriteMode = VecSim_WriteAsync;

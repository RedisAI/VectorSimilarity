/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <cstdlib>

/**
 * This file contains the headers to be used internally for creating an array of results in
 * TopKQuery methods.
 */
struct VecSimQueryResult {
    size_t id;
    double score;
};

/**
 * @brief Sets result's id (to use from index TopKQuery method)
 */
void VecSimQueryResult_SetId(VecSimQueryResult &result, size_t id);

/**
 * @brief Sets result score (to use from index TopKQuery method)
 */
void VecSimQueryResult_SetScore(VecSimQueryResult &result, double score);

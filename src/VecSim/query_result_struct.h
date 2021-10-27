#pragma once

#include <cstdlib>
#include <limits>

#define INVALID_ID    -1
#define INVALID_SCORE std::numeric_limits<float>::min()

/**
 * This file contains the headers to be used internally for creating an array of results in
 * TopKQuery methods.
 */
struct VecSimQueryResult {
    size_t id;
    float score;
};

/**
 * @brief Create a new empty result (to use from index TopKQuery method)
 * After creation, use SetId() and SetScore() so that the result will have valid fields.
 */
VecSimQueryResult VecSimQueryResult_Create();

/**
 * @brief Sets result's id (to use from index TopKQuery method)
 */
void VecSimQueryResult_SetId(VecSimQueryResult &result, size_t id);

/**
 * @brief Sets result score (to use from index TopKQuery method)
 */
void VecSimQueryResult_SetScore(VecSimQueryResult &result, float score);

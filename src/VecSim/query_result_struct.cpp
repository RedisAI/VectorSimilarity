/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "query_result_struct.h"
#include "query_results.h"
#include <cassert>
#include <math.h>

void VecSimQueryResult_SetId(VecSimQueryResult &result, size_t id) { result.id = id; }

void VecSimQueryResult_SetScore(VecSimQueryResult &result, double score) { result.score = score; }

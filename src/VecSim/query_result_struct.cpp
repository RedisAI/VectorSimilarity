#include "query_result_struct.h"
#include "query_results.h"
#include <cassert>
#include <cmath>

void VecSimQueryResult_SetId(VecSimQueryResult &result, size_t id) { result.id = id; }

void VecSimQueryResult_SetScore(VecSimQueryResult &result, float score) {
    assert(!isnan(score));
    result.score = score;
}

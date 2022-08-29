#include "query_result_struct.h"
#include "query_results.h"
#include <cassert>
#include <math.h>

void VecSimQueryResult_SetId(VecSimQueryResult &result, size_t id) { result.id = id; }

void VecSimQueryResult_SetScore(VecSimQueryResult &result, double score) { result.score = score; }

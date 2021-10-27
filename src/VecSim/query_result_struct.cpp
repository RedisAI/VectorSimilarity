#include "query_result_struct.h"
#include "query_results.h"

VecSimQueryResult VecSimQueryResult_Create() {
    return VecSimQueryResult{(size_t)INVALID_ID, INVALID_SCORE};
}
void VecSimQueryResult_SetId(VecSimQueryResult &result, size_t id) { result.id = id; }

void VecSimQueryResult_SetScore(VecSimQueryResult &result, float score) { result.score = score; }

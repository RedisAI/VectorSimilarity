#include "query_results.h"

struct VecSimQueryResult {
    size_t id;
    float score;
};

VecSimQueryResult VecSimQueryResult_Create(size_t id, float score) {
    return VecSimQueryResult{id, score};
}

void VecSimQueryResult_SetId(VecSimQueryResult result, size_t id) {
    result.id = id;
}

void VecSimQueryResult_SetScore(VecSimQueryResult result, float score) {
    result.score = score;
}

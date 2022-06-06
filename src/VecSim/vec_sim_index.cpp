#include "vec_sim_index.h"

timeoutCallbackFunction VecSimIndex::timeoutCallback = [](void *ctx) { return 0; };

void VecSimIndex::setTimeoutCallbackFunction(timeoutCallbackFunction callback) {
    VecSimIndex::timeoutCallback = callback;
}

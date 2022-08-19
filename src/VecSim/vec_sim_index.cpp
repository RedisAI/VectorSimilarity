#include "vec_sim_index.h"

timeoutCallbackFunction VecSimIndexAbstract::timeoutCallback = [](void *ctx) { return 0; };

void VecSimIndexAbstract::setTimeoutCallbackFunction(timeoutCallbackFunction callback) {
    VecSimIndexAbstract::timeoutCallback = callback;
}

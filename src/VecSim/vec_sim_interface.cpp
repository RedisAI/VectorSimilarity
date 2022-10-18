#include "VecSim/vec_sim_interface.h"

timeoutCallbackFunction VecSimIndexInterface::timeoutCallback = [](void *ctx) { return 0; };

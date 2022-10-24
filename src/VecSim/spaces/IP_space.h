#pragma once
#include "VecSim/spaces/spaces.h"
namespace spaces {
dist_func_t<float> IP_FP32_GetDistFunc(size_t dim);
dist_func_t<double> IP_FP64_GetDistFunc(size_t dim);

} // namespace spaces

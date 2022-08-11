#pragma once
#include "VecSim/spaces/spaces.h"

namespace Spaces {
dist_func_ptr_ty<float> L2_FLOAT_GetDistFunc(size_t dim);
dist_func_ptr_ty<double> L2_DOUBLE_GetDistFunc(size_t dim);

} // namespace Spaces

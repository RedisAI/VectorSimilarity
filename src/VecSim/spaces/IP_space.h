#pragma once
#include "VecSim/spaces/spaces.h"
namespace Spaces {

dist_func_ptr_ty<float> IP_FLOAT_GetDistFunc(size_t dim);
dist_func_ptr_ty<double> IP_DOUBLE_GetDistFunc(size_t dim);
/* void IP_SetDistFunc(size_t dim, dist_func_ptr_ty<float> *index_dist_func);
void IP_SetDistFunc(size_t dim, dist_func_ptr_ty<double> *index_dist_func); */
} // namespace Spaces

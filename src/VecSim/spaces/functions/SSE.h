/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "spaces.h"

namespace spaces {

dist_func_t<float> Choose_IP_implementation_SSE(size_t dim);
dist_func_t<double> Choose_IP_implementation_SSE(size_t dim);

dist_func_t<float> Choose_L2_implementation_SSE(size_t dim);
dist_func_t<double> Choose_L2_implementation_SSE(size_t dim);

} // namespace spaces

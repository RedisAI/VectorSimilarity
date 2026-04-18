/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "TQ.h"

namespace spaces {

tq_symmetric_polar_estimate_func_t Choose_TQ_SymmetricPolarEstimate_implementation_NEON(void);

tq_packed_sign_dot_func_t Choose_TQ_PackedSignDot_implementation_NEON(void);

tq_packed_residual_sign_dot_func_t
Choose_TQ_PackedResidualSignDot_implementation_NEON(size_t projections);

tq_symmetric_polar_func_t Choose_TQ_SymmetricPolar_implementation_NEON(size_t pairs);

} // namespace spaces

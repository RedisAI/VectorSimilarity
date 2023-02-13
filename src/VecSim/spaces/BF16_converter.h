/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "VecSim/spaces/spaces.h"

namespace spaces {
fp32_to_bf16_converter_t Get_FP32_to_BF16_Converter(size_t dim, const Arch_Optimization arch_opt, bool big_endian);

bf16_to_fp32_converter_t Get_BF16_to_FP32_Converter(size_t dim, const Arch_Optimization arch_opt, bool big_endian);

} // namespace spaces
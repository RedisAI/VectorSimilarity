/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#include "VecSim/spaces/BF16_converter.h"
#include "VecSim/spaces/converters/bf16/BF16_converter_impl.h"
#include "VecSim/spaces/converters/bf16/BF16_converter_AVX512.h"

namespace spaces {
    fp32_to_bf16_converter_t Get_FP32_to_BF16_Converter(size_t dim, const Arch_Optimization arch_opt) {
        fp32_to_bf16_converter_t converter = (fp32_to_bf16_converter_t)FP32_to_BF16;
        return converter;
    }
} // namespace spaces

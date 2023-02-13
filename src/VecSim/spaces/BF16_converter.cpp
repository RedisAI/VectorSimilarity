/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#include "VecSim/spaces/BF16_converter.h"
#include "VecSim/spaces/converters/bf16/BF16_converter_impl.h"

namespace spaces {
    fp32_to_bf16_converter_t Get_FP32_to_BF16_Converter(size_t dim, const Arch_Optimization arch_opt, bool big_endian) {
        fp32_to_bf16_converter_t converter = big_endian ? (fp32_to_bf16_converter_t)FP32_to_BF16_BigEndian : (fp32_to_bf16_converter_t)FP32_to_BF16_LittleEndian;
        return converter;
    }

    bf16_to_fp32_converter_t Get_BF16_to_FP32_Converter(size_t dim, const Arch_Optimization arch_opt, bool big_endian) {
        bf16_to_fp32_converter_t converter = big_endian ? (bf16_to_fp32_converter_t)BF16_to_FP32_BigEndian : (bf16_to_fp32_converter_t)BF16_to_FP32_LittleEndian;
        return converter;

    }
} // namespace spaces

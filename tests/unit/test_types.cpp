/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include <random>
#include "gtest/gtest.h"
#include "VecSim/types/float16.h"

#ifdef OPT_AVX512_FP16_VL
class FP16Type : public ::testing::Test {};

TEST_F(FP16Type, Test_Float16VSvecsim) {
    static_assert(sizeof(_Float16) == sizeof(vecsim_types::float16));
    size_t runs = 10;

    std::mt19937 gen{};
    std::uniform_real_distribution<float> dis(-0.999999, 0.999999);

    for (size_t i = 0; i < runs; i++) {
        float val = dis(gen);
        _Float16 nativeCast = static_cast<_Float16>(val);
        vecsim_types::float16 vecsimCast = vecsim_types::FP32_to_FP16(val);

        uint16_t nativeBits, vecsimBits;
        std::memcpy(&nativeBits, &nativeCast, sizeof(_Float16));
        std::memcpy(&vecsimBits, &vecsimCast, sizeof(vecsim_types::float16));

        ASSERT_EQ(nativeBits, vecsimBits) << "failed at iteration " << i << "with value " << val;
    }
}

#endif

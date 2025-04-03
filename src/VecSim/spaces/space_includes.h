/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/utils/alignment.h"

#include "cpu_features_macros.h"
#ifdef CPU_FEATURES_ARCH_X86_64
#include "cpuinfo_x86.h"
#endif // CPU_FEATURES_ARCH_X86_64
#ifdef CPU_FEATURES_ARCH_AARCH64
#include "cpuinfo_aarch64.h"
#endif // CPU_FEATURES_ARCH_AARCH64

#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
#if defined(__GNUC__)
#include <x86intrin.h>
#elif defined(__clang__)
#include <xmmintrin.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#include <stdexcept>
#endif

#endif // __AVX512F__ || __AVX__ || __SSE__

/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#if defined(__GNUC__)
#include <x86intrin.h>
#elif defined(__clang__)
#include <xmmintrin.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#include <stdexcept>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PORTABLE_ALIGN16 __attribute__((aligned(16)))
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#elif defined(_MSC_VER)
#define PORTABLE_ALIGN16 __declspec(align(16))
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

#endif // USE_AVX || USE_SSE

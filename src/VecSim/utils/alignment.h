/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define PORTABLE_ALIGN16 __attribute__((aligned(16)))
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#elif defined(_MSC_VER)
#define PORTABLE_ALIGN16 __declspec(align(16))
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

#define PORTABLE_ALIGN PORTABLE_ALIGN64

// TODO: relax the above alignment requirements according to the CPU architecture
#ifndef PORTABLE_ALIGN
#if defined(__AVX512F__)
#define PORTABLE_ALIGN PORTABLE_ALIGN64
#elif defined(__AVX__)
#define PORTABLE_ALIGN PORTABLE_ALIGN32
#elif defined(__SSE__)
#define PORTABLE_ALIGN PORTABLE_ALIGN16
#else
#define PORTABLE_ALIGN
#endif
#endif // ifndef PORTABLE_ALIGN

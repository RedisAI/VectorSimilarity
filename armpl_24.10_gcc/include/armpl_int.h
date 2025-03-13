/*
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 * SPDX-FileCopyrightText: Copyright 2015-2024 NAG
 */

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifndef ARMPL_INT_T
#define ARMPL_INT_T

#ifdef INTEGER64
typedef int64_t  armpl_int_t;
typedef uint64_t armpl_uint_t;
#else
typedef int32_t  armpl_int_t;
typedef uint32_t armpl_uint_t;
#endif
typedef int32_t  armpl_strlen_t;

typedef armpl_int_t lapack_int;
typedef armpl_int_t lapack_logical;

#endif

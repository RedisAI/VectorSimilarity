/*
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

/*
  A complex datatype for use by the C interfaces to ARMPL routines.
  The exact definition can be overridden by manually #define-ing
  armpl_singlecomplex_t and armpl_doublecomplex_t.
*/

#ifndef armpl_singlecomplex_t
#include <complex.h>
#if defined(_WIN32)
typedef _Fcomplex armpl_singlecomplex_t;
#else
typedef float _Complex armpl_singlecomplex_t;
#endif
#define armpl_singlecomplex_t armpl_singlecomplex_t
#endif

#ifndef armpl_doublecomplex_t
#include <complex.h>
#if defined(_WIN32)
typedef _Dcomplex armpl_doublecomplex_t;
#else
typedef double _Complex armpl_doublecomplex_t;
#endif
#define armpl_doublecomplex_t armpl_doublecomplex_t
#endif

/*
  The LAPACKE interface uses the macros lapack_complex_float and lapack_complex_double,
  which are compatible with the corresponding armpl types.
*/
#ifndef lapack_complex_float
#define lapack_complex_float armpl_singlecomplex_t
#endif
#ifndef lapack_complex_double
#define lapack_complex_double armpl_doublecomplex_t
#endif

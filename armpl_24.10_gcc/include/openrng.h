/*
 * SPDX-FileCopyrightText: <text>Copyright 2024 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \example pi.c
 *
 * This example demonstrates usage of OpenRNG to fill a buffer with uniformly
 * distributed random numbers. It uses \ref vslNewStream, \ref vsRngUniform and
 * \ref vslDeleteStream. The behaviour of the example is documented inline.
 *
 * Example output:
 * \code
 * Estimate of pi:        3.142112
 * Number of iterations:  1000000
 * \endcode
 */

/**
 * \example skipahead.c
 *
 * This example demonstrates how to use the 'block-skipping' method to improve
 * performance when parallelism is available.
 *
 * Example output:
 * \code
 * Serial run took 0.021867 seconds
 * Parallel run took 0.002624 seconds
 * All values matched!
 * \endcode
 */

/**
 * \example sobol_joe_kuo.cpp
 *
 * This example demonstrates how to process the direction numbers supplied by S.
 * Joe and F. Y. Kuo (available at https://web.maths.unsw.edu.au/~fkuo/sobol/),
 * and organise the input array such that they can be used with \ref
 * VSL_BRNG_SOBOL. It expects three arguments: a path to the list of numbers,
 * the number of dimensions and the number of elements to generate.
 *
 * Example output:
 * \code
 * $ ./sobol_joe_kuo new-joe-kuo-6.21201 3 30
 * 0.5,	0.5,	0.5
 * 0.75,	0.25,	0.25
 * 0.25,	0.75,	0.75
 * 0.375,	0.375,	0.625
 * 0.875,	0.875,	0.125
 * 0.625,	0.125,	0.875
 * 0.125,	0.625,	0.375
 * 0.1875,	0.3125,	0.9375
 * 0.6875,	0.8125,	0.4375
 * 0.9375,	0.0625,	0.6875
 * \endcode
 */

/**
 * \defgroup groupService Service functions
 *
 * This section documents all service function. The service functions are used
 * for creating, copying, querying, modifying and deleting instances of basic
 * random number generators.
 */

/**
 * \defgroup groupContinuous Continuous distribution functions
 *
 * This section documents all methods for producing continuous distributions.
 */

/**
 * \defgroup groupDiscrete Discrete distribution functions
 *
 * This section documents all methods for producing discrete distributions.
 */

/**
 * \defgroup groupBrng Basic random number generator constants
 *
 * This section describes the different types of Basic random number generators
 * (BRNG) available in OpenRNG. A BRNG is any source of randomness that can
 * produce uniform distributions on `[0, 1)`. OpenRNG supports 3 types of BRNG
 *
 *  * Pseudorandom number generators
 *  * Quasirandom number generators
 *  * Nondeterministic random number generators
 *
 * Each BRNG is identified by a preprocessor define, of the form
 * `VSL_BRNG_<NAME>` and set as a unique integer value. To use a BRNG, you need
 * to create a stream using a service function, and pass in the corresponding
 * preprocessor define. See \ref groupService for the list of all supported
 * service functions.
 */

/**
 * \defgroup groupError Error codes
 *
 * These are the possible error codes returned by the OpenRNG API. See
 * individual pages for descriptions of the error codes.
 */

/**
 * \defgroup groupDistMethods Distribution methods
 *
 * When a distribution is selected, one can further specify which method will be
 * used to generate the particular distribution.
 *
 * The distribution methods can be divided into two categories; fast, and
 * accurate mode. Accurate mode can be identified by the `_ACCURATE` suffix;
 * fast mode has no suffix.
 *
 * Most distributions support an inverse cumulative distribution method (ICDF).
 * The ICDF works by generating a number in the range [0, 1) and mapping it on
 * to the cumulative distribution function. In some cases, the ICDF method can't
 * be computed efficiently, in these cases accept-reject methods can be applied,
 * or a method more specific to the distribution is used.
 */

/**
 * \defgroup groupTypes Data types.
 *
 * The following section documents all data types provided by OpenRNG.
 */

/**
 * \defgroup groupQRNG QRNG flags.
 *
 * The following section documents special flags for initialising QRNGs.
 */

/**
 * \ingroup groupError
 *
 * No error raised. Execution ran with no errors.
 */
#define VSL_STATUS_OK 0

/**
 * \ingroup groupError
 *
 * No error raised. Execution ran with no errors.
 */
#define VSL_ERROR_OK 0

/**
 * \ingroup groupError
 *
 * Error raised when the feature invoked is not implemented.
 */
#define VSL_ERROR_FEATURE_NOT_IMPLEMENTED -1

/**
 * \ingroup groupError
 *
 * Error raised when bad values of arguments passed to
 * function call.
 */
#define VSL_ERROR_BADARGS -2

/**
 * \ingroup groupError
 *
 * Error raised when the system cannot allocate memory.
 */
#define VSL_ERROR_MEM_FAILURE -3

/**
 * \ingroup groupError
 *
 * Error raised when a null pointer is encountered.
 */
#define VSL_ERROR_NULL_PTR -4

/**
 * \ingroup groupError
 *
 * Error raised when the index indicating a Basic
 * Random Number Generator (BRNG) is invalid.
 */
#define VSL_RNG_ERROR_INVALID_BRNG_INDEX -0x1000

/**
 * \ingroup groupError
 *
 * Error raised when leapfrog is requested for a generator that does not
 * support it.
 */
#define VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED -0x1001

/**
 * \ingroup groupError
 *
 * Error raised when skip ahead is requested for a generator that does not
 * support it.
 */
#define VSL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED -0x1002

/**
 * \ingroup groupError
 *
 * Error raised when extended skip ahead is requested for a generator that does
 * not support it.
 */
#define VSL_RNG_ERROR_SKIPAHEADEX_UNSUPPORTED -0x1003

/**
 * \ingroup groupError
 *
 * Error raised when an operation is requested on two incompatible BRNGs.
 */
#define VSL_RNG_ERROR_BRNGS_INCOMPATIBLE -0x1004

/**
 * \ingroup groupError
 *
 * Error raised when the random stream is invalid.
 */
#define VSL_RNG_ERROR_BAD_STREAM -0x1005

/**
 * \ingroup groupError
 *
 * Error raised when the random stream format is unknown.
 */
#define VSL_RNG_ERROR_BAD_MEM_FORMAT -0x1006

/**
 * \ingroup groupError
 *
 * Error raised when the BRNG is not supported by the
 * function.
 */
#define VSL_RNG_ERROR_BRNG_NOT_SUPPORTED -0x1007

/**
 * \ingroup groupError
 *
 * Error raised while trying to open a file.
 */
#define VSL_RNG_ERROR_FILE_OPEN -0x1008

/**
 * \ingroup groupError
 *
 * Error raised while trying to write to a file.
 */
#define VSL_RNG_ERROR_FILE_WRITE -0x1009

/**
 * \ingroup groupError
 *
 * Error raised while trying to read from a file.
 */
#define VSL_RNG_ERROR_FILE_READ -0x100a

/**
 * \ingroup groupError
 *
 * Error raised while trying to close a file.
 */
#define VSL_RNG_ERROR_FILE_CLOSE -0x100b

/**
 * \ingroup groupError
 *
 * Error raised when the nondeterministic random number generator is not
 * supported on the CPU running the application.
 */
#define VSL_RNG_ERROR_NONDETERM_NOT_SUPPORTED -0x100c

/**
 * \ingroup groupQRNG
 *
 * Second element of `params` argument to \ref vslNewStreamEx to indicate custom
 * direction numbers and/or primitive polynomials will be used.
 */
#define VSL_USER_QRNG_INITIAL_VALUES 0x1

/**
 * \ingroup groupQRNG
 *
 * Pass (possibly combined with other flags using `|`) as third element of
 * `params` array to \ref vslNewStreamEx to indicate user-defined primitive
 * polynomials will be used. If \ref VSL_USER_INIT_DIRECTION_NUMBERS is not also
 * set, the number of dimensions must be less than 41.
 */
#define VSL_USER_PRIMITIVE_POLYMS 0x1

/**
 * \ingroup groupQRNG
 *
 * Pass (possibly combined with other flags using `|`) as third element of
 * `params` array to \ref vslNewStreamEx to indicate user-defined initial
 * direction numbers will be used. If \ref VSL_USER_PRIMITIVE_POLYMS is not also
 * set, the number of dimensions must be less than 41. Use \ref
 * VSL_USER_DIRECTION_NUMBERS to pass the full set of 32 direction numbers per
 * dimension instead of relying on polynomials to generate them.
 */
#define VSL_USER_INIT_DIRECTION_NUMBERS 0x2

/**
 * \ingroup groupQRNG
 *
 * Pass (possibly combined with other flags using `|`) as third element of
 * `params` array to \ref vslNewStreamEx to indicate user-defined direction
 * numbers will be used. \ref VSL_USER_PRIMITIVE_POLYMS will be ignored if set -
 * params array must contain the full set of 32 direction numbers per dimension.
 */
#define VSL_USER_DIRECTION_NUMBERS 0x4

/**
 * \ingroup groupQRNG
 *
 * Pass (possibly combined with other flags using `|`) as third element of
 * `params` array to \ref vslNewStreamEx to indicate that the default values for
 * dimension 0 should be overridden by the custom settings (direction numbers
 * for dimension 0 are ordinarily set to the special-case from Bratley & Fox
 * 1988. This option is ignored unless both `VSL_USER_INIT_DIRECTION_NUMBERS`
 * and `VSL_USER_PRIMITIVE_POLYMS` are set, and is implicit for
 * `VSL_USER_DIRECTION_NUMBERS`.
 */
#define VSL_QRNG_OVERRIDE_1ST_DIM_INIT 0x8

/**
 * \ingroup groupBrng
 *
 * A multiplicative congruential pseudorandom number generator. It's a
 * specialized linear congruential generator without a displacement term.
 * MCG31m1 uses the following constants which originated from [1]:
 *
 * \code
 * a    =  1132489760;
 * m    =  pow(2, 31) - 1;
 * \endcode
 *
 * Initialization with vslNewStream
 * ================================
 *
 * First element of the integer sequence `x[0]` calculated from parameter `seed`
 * according to:
 *
 * \code{.cpp}
 * x[0] = max(1, seed % m);
 * \endcode
 *
 * Initialization with vslNewStreamEx
 * ==================================
 *
 * First element of the integer sequence `x[0]` calculated from parameter array
 * `params` of length `n` according to:
 *
 * \code{.cpp}
 * if (n == 0)
 *   x[0] = 1;
 * else
 *   x[0] = max(1, params[0] % m);
 * \endcode
 *
 * Recurrence relation
 * ===================
 *
 * Subsequent elements of the sequence are calculated from the preceding by
 * using:
 *
 * \code{.cpp}
 * x[i] =  a * x[i - 1] % m;
 * \endcode
 *
 * Integer output
 * ==============
 *
 * Output `r[i]` of the integer sequence is defined by
 *
 * \code{.cpp}
 * r[i] = x[i];
 * \endcode
 *
 * Float output
 * ============
 *
 * Output `r[i]` of the floating point sequence is defined by
 *
 * \code{.cpp}
 * r[i] = x[i] / m;
 * \endcode
 *
 * Subsequences
 * ============
 *
 * The following table lists which methods are supported with the BRNG for
 * generating subsequences:
 *
 * Method | Supported?
 * ------------- | -------------
 * \ref vslLeapfrogStream | Yes
 * \ref vslSkipAheadStream | Yes
 * \ref vslSkipAheadStreamEx | No
 *
 * Example
 * =======
 *
 * An example of initializing a stream using MCG31 and a seed of 42:
 *
 * \code
 *   VSLStreamStatePtr stream;
 *   int errcode = vslNewStream(&stream, VSL_BRNG_MCG31, 42);
 * \endcode
 *
 * References
 * ==========
 * [1] L’Ecuyer, P. Tables of Linear Congruential Generators of Different Sizes
 * and Good Lattice Structure. Mathematics of Computation, 68, 225, 249-260,
 * 1999.
 */
#define VSL_BRNG_MCG31 0x1000

/**
 * \ingroup groupBrng
 *
 * A pseudorandom number generator using a linear-feedback shift register [1].
 *
 * Initialization with vslNewStream
 * ================================
 *
 * The first 32-bit element of the sequence is set using `x[0] = seed`, if `seed
 * != 0`, otherwise `x[0] = 1`. The subsequent 249 elements are then initialized
 * according to recurrence relation:
 *
 * \code{.cpp} x[n] = 69069 * x[n - 1]; \endcode
 *
 * The elements `x[7 * k + 3]` for `k == 0, 1, ... , 31` are concatenated and
 * treated as rows in a 32x32 binary matrix. The the under-diagonal bits are set
 * to 0, and diagonal bits are set to 1.
 *
 * Initialization with vslNewStreamEx
 * ==================================
 *
 * R250 is initialized with vslNewStreamEx using the following scheme:
 *
 *  * If `n >= 250`, use the first 250 elements of `params` to initialize the
 *    sequence, ignoring any subsequent elements.
 *  * Otherwise, if n > 0, initialize as for \ref vslNewStream, ignoring all but
 *    the first element of `params`.
 *  * If n == 0, `x[0]` is set to 1.
 *
 * Recurrence relation
 * ===================
 *
 * The first 250 elements are used to seed the generator, but are not included
 * in the output. Subsequent samples are generated by:
 *
 * \code{.cpp} x[n] = x[n - 147] ^ x[n - 250]; \endcode
 *
 * Where `^` denotes the XOR operator.
 *
 * Integer output
 * ==============
 *
 * 32-bit integer output element `z[i]` is element `x[250 + i]` of sequence,
 * because the initial 250-element seed is ignored).
 *
 * Float output
 * ============
 *
 * Output element `r[i]` of the floating point sequence is computed as `z[i] /
 * pow(2, 32)`.
 *
 * Subsequences
 * ============
 *
 * The following table lists which methods are supported with the BRNG for
 * generating subsequences:
 *
 * Method | Supported?
 * ------------- | -------------
 * \ref vslLeapfrogStream | No
 * \ref vslSkipAheadStream | No
 * \ref vslSkipAheadStreamEx | No
 *
 * Example
 * =======
 *
 * An example of initializing a stream using R250 and a seed of 42:
 *
 * \code
 *   VSLStreamStatePtr stream;
 *   int errcode = vslNewStream(&stream, VSL_BRNG_R250, 42);
 * \endcode
 *
 * References
 * ==========
 * [1] Kirkpatrick, S., and Stoll, E. A Very Fast Shift-Register Sequence Random
 * Number Generator. Journal of Computational Physics, V. 40. 517-526, 1981.
 */
#define VSL_BRNG_R250 0x1001

// clang-format off
/**
 * \ingroup groupBrng
 *
 * A combined multiple recursive pseudorandom number generator, introduced in
 * [1], with two components of order 3. The two component sequences are referred
 * to here as `x` and `y`, where each component has its own modulus: `m_1 ==
 * pow(2, 32) - 209` for `x` and `m_2 == pow(2, 32) - 22853` for `y`.
 *
 * Initialization with vslNewStream
 * ================================
 *
 * Set `x[0] = seed % m_1`, `x[1] = x[2] = y[0] = y[1] = y[2] = 1`.
 *
 * Initialization with vslNewStreamEx
 * ==================================
 *
 * The below tables describes how `x` and `y` are initialized based on
 * `n` and `params` array (referred to as `p` for brevity).
 *
 * n | `x[0]` |  `x[1]` |  `x[2]` | `y[0]` |  `y[1]` |  `y[2]`
 * --|--------|------|--------|--------|--------|-------
 * 0 | 1 | 1 | 1 | 1 | 1 | 1
 * 1 | `p[0] % m_1` | 1 | 1 | 1 | 1 | 1
 * 2 | `p[0] % m_1` | `p[1] % m_1` | 1 | 1 | 1 | 1
 * 3 | `p[0] % m_1` | `p[1] % m_1` | `p[2] % m_1` | 1 | 1 | 1
 * 4 | `p[0] % m_1` | `p[1] % m_1` | `p[2] % m_1` | `p[3] % m_2` | 1 | 1
 * 5 | `p[0] % m_1` | `p[1] % m_1` | `p[2] % m_1` | `p[3] % m_2` | `p[4] % m_2` | 1
 * 6 | `p[0] % m_1` | `p[1] % m_1` | `p[2] % m_1` | `p[3] % m_2` | `p[4] % m_2` | `p[5] % m_2`
 *
 * If, after following the above, all initialized elements of either `x` or `y`
 * are zero, the first three elements of the sequence are all set to 1.
 *
 * Recurrence relation
 * ===================
 *
 * Coefficients for advancing `x` and `y` are given below:
 *
 * Sequence | `a[0]` | `a[1]`  | `a[2]`
 * ---------|--------|---------|---------
 * `x`      | 0      | 1403580 | -810728
 * `y`      | 527612 | 0       | -1370589
 *
 * \code{.cpp}
 * // update x
 * x[n] = (x[n - 1] * ax[0] + x[n - 2] * ax[1] + x[n - 3] * ax[2]) % m_1;
 * // update y
 * y[n] = (y[n - 1] * ay[0] + y[n - 2] * ay[2] + y[n - 3] * ay[2]) % m_2;
 * \endcode
 *
 * Integer output
 * ==============
 *
 * Output element `z[i]` is then calculated by:
 *
 * \code{.cpp}
 * z[i] = (x[i] - y[i]) % m_1;
 * \endcode
 *
 * Float output
 * ============
 *
 * Floating-point output element `r[i]` is calculated by `z[i] / m_1`.
 *
 * Subsequences
 * ============
 *
 * The following table lists which methods are supported with the BRNG for
 * generating subsequences:
 *
 * Method | Supported?
 * ------------- | -------------
 * \ref vslLeapfrogStream | No
 * \ref vslSkipAheadStream | Yes
 * \ref vslSkipAheadStreamEx | Yes
 *
 * Example
 * =======
 *
 * An example of initializing a stream using MRG32K3A and a seed of 42:
 *
 * \code
 *   VSLStreamStatePtr stream;
 *   int errcode = vslNewStream(&stream, VSL_BRNG_MRG32K3A, 42);
 * \endcode
 *
 * References
 * ==========
 *
 * [1] Fischer, G., Carmon, Z., Zauberman, G., and L’Ecuyer, P.
 * (1999). Good Parameters and Implementations for Combined Multiple Recursive
 * Random Number Generators. Operations Research. 47. 159-164.
 */
// clang-format on
#define VSL_BRNG_MRG32K3A 0x1002

/**
 * \ingroup groupBrng
 *
 * A 59-bit multiplicative congruential pseudorandom number generator from [1].
 *
 * \code{.cpp}
 * m = pow(2, 59);
 * a = pow(13, 13);
 * \endcode
 *
 * Initialization with vslNewStream
 * ================================
 *
 * The generator state `x[0]` is initialized from parameter `seed` according
 * to:
 *
 * \code{.cpp}
 * x[0] = a * seed % m;
 * \endcode
 *
 * Initialization with vslNewStreamEx
 * ==================================
 *
 * The generator state `x[0]` is initialized the from parameter array `params`
 * according to:
 *
 * \code{.cpp}
 * seed = ((uint64)params[1] << 32) | params[0];
 * x[0] = a * seed % m;
 * \endcode
 *
 * Recurrence relation
 * ===================
 *
 * The state is updated for each sample according to:
 *
 * \code{.cpp}
 * x[i] = a * x[i - 1] % m;
 * \endcode
 *
 * Integer Output
 * ==============
 *
 * MCG59 produces a 32-bit output `x32` by bit-shifting `x` by 27 right
 * \code{.cpp}
 * x32[i] = x[i] >> 27;
 * \endcode
 *
 * It produces a 64-bit output `x64` by concatenating two consecutive 32-bit
 * outputs `x32`
 * \code{.cpp}
 * x64[i] = (x32[2 * i + 1] << 32) | x32[2 * i];
 * \endcode
 *
 * Float Output
 * ============
 *
 * The floating point output `u` is defined as
 * \code{.cpp}
 * u[i] = x[i] / m;
 * \endcode
 *
 * Subsequences
 * ============
 *
 * The following table lists which methods are supported with the BRNG for
 * generating subsequences:
 *
 * Method | Supported?
 * ------------- | -------------
 * \ref vslLeapfrogStream | Yes
 * \ref vslSkipAheadStream | Yes
 * \ref vslSkipAheadStreamEx | No
 *
 * Example
 * =======
 *
 * An example of initializing a stream using MCG59 and a seed of 42:
 *
 * \code
 *   VSLStreamStatePtr stream;
 *   int errcode = vslNewStream(&stream, VSL_BRNG_MCG59, 42);
 * \endcode
 *
 * References
 * ==========
 * [1] NAG Numerical Libraries. https://nag.com/nag-library/
 */
#define VSL_BRNG_MCG59 0x1003

/**
 * \ingroup groupBrng
 *
 * A set of 273 Wichmann-Hill combined multiplicative congruential generators.
 *
 * \warning Not implemented in \armplref.
 */
#define VSL_BRNG_WH 0x1004

// clang-format off
/**
 * \ingroup groupBrng
 *
 * A quasirandom number generator (QRNG) based on Gray codes, implementing
 * Sobol's low-discrepancy sequence. This implementation of SOBOL supports only
 * the direction numbers provided in [1].
 *
 * Initialization with vslNewStream
 * ================================
 *
 * Set the number of dimensions in the output vector to `seed`. A
 * maximum of 40 dimensions are supported - if `seed` is 0 or greater
 * than 40, set dimensions to 1.
 *
 * Initialization with vslNewStreamEx
 * ==================================
 *
 * If `n` is 1, initialize according to \ref vslNewStream. Otherwise,
 * vslNewStreamEx can be used to pass user-defined primitive polynomials and/or
 * initial direction numbers:
 *
 * Polynomials and direction numbers
 * ---------------------------------
 *
 * To register both custom polynomials and initial direction numbers, the layout
 * of the `params` array should be as follows:
 *
 * | Index | Value |
 * |-------|-------|
 * | 0     | `n_dimensions` |
 * | 1     | `\ref VSL_USER_QRNG_INITIAL_VALUES` |
 * | 2     | `VSL_USER_INIT_DIRECTION_NUMBERS \| VSL_USER_PRIMITIVE_POLYMS` |
 * | 3 to 2 + `n_dimensions` | Primitive polynomials |
 * | 2 + `n_dimensions` | Max degree of any polynomial (`max_deg`) |
 * | 3 + `n_dimensions` to 3 + `n_dimensions` + `max_deg` * (`n_dimensions` - 1) | Initial direction numbers, in dimension-major order and zero-padded to `max_deg` elements each |
 *
 * See \ref sobol_joe_kuo.cpp for an example of how to initialise a
 * stream using this layout.
 *
 * By default dimension 0 is still initialised using default
 * parameters. set `params[2] |= VSL_QRNG_OVERRIDE_1ST_DIM_INIT` to
 * override dimension 0 - layout then has to be:
 *
 * | Index | Value |
 * |-------|-------|
 * | 0     | `n_dimensions` |
 * | 1     | `\ref VSL_USER_QRNG_INITIAL_VALUES` |
 * | 2     | `VSL_USER_INIT_DIRECTION_NUMBERS \| VSL_USER_PRIMITIVE_POLYMS \| VSL_QRNG_OVERRIDE_1ST_DIM_INIT` |
 * | 3 to 3 + `n_dimensions` | Primitive polynomials |
 * | 3 + `n_dimensions` | Max degree of any polynomial (`max_deg`) |
 * | 4 + `n_dimensions` to 4 + (`max_deg` + 1) * `n_dimensions`  | Initial direction numbers, in dimension-major order and zero-padded to `max_deg` elements each |
 *
 * In other words, one more polynomial and set of initial direction
 * numbers must be passed to account for the extra dimension.
 *
 * Polynomials only
 * ----------------
 *
 * To register only custom polynomials, and use the default table of
 * initial direction numbers, the layout of the `params` array should
 * be as follows:
 *
 * | Index | Value |
 * |-------|-------|
 * | 0     | `n_dimensions` |
 * | 1     | `\ref VSL_USER_QRNG_INITIAL_VALUES` |
 * | 2     | `VSL_USER_PRIMITIVE_POLYMS` |
 * | 3 to 2 + `n_dimensions` | Primitive polynomials |
 *
 * Note it is assumed that the maximum degree is 8 - using a higher
 * degree than this without supplying initial direction numbers is not
 * supported. `n_dimensions` must be less than 41.
 *
 * Note VSL_QRNG_OVERRIDE_1ST_DIM_INIT is ignored for this option.
 *
 * Initial direction numbers only
 * ------------------------------
 *
 * To register only initial direction numbers, and use VSL's default
 * polynomials to generate the full table, layout should be as
 * follows:
 *
 * | Index | Value |
 * |-------|-------|
 * | 0     | `n_dimensions` |
 * | 1     | `\ref VSL_USER_QRNG_INITIAL_VALUES` |
 * | 2     | `VSL_USER_INIT_DIRECTION_NUMBERS` |
 * | 3     | `max_deg` |
 * | 4 to 4 + `max_deg` * (`n_dimensions` - 1) | Initial direction numbers, in dimension-major order and zero-padded to `max_deg` elements each |
 *
 * Note VSL_QRNG_OVERRIDE_1ST_DIM_INIT is ignored for this option.
 *
 * Direction numbers only
 * ----------------------
 *
 * To register only the full table of direction numbers, layout should be as
 * follows:
 *
 * | Index | Value |
 * |-------|-------|
 * | 0     | `n_dimensions` |
 * | 1     | `\ref VSL_USER_QRNG_INITIAL_VALUES` |
 * | 2     | `VSL_USER_DIRECTION_NUMBERS` |
 * | 3 to 3 + 32 * `n_dimensions` | Direction numbers, in dimension-major order |
 *
 * Note VSL_QRNG_OVERRIDE_1ST_DIM_INIT is implicitly set for this option.
 *
 * Recurrence relation
 * ===================
 *
 * Element `i` of the Sobol sequence is the result of XOR-ing (`^`) together
 * direction numbers `V[j]`, for all `j` for which bit `j` is set in
 * the Gray code representation of `i`. Hence element `x[n]` of the
 * sequence is calculated from `x[n - 1]` by:
 *
 * \code{.cpp}
 * x[n] = x[n - 1] ^ v[Ji];
 * \endcode
 *
 * where `v` is a direction number and `Ji` is the index of the
 * right-most zero bit in `n - 1`, or equivalently the number of
 * trailing zeroes in `n`. Note that the sequence is
 * multi-dimensional, so `x[n]` may be a vector rather than a single
 * integer. In this case, each dimension should have it its own set of
 * direction numbers.
 *
 * Integer output
 * ==============
 *
 * Integer output is vector-by-vector, i.e. element `y[n]` of the
 * stream is element `n % d` of output vector `x[n / d]` (rounded
 * down), where `d` is the number of dimensions.
 *
 * Float output
 * ============
 *
 * Floating-point output element `n` is calculated by
 * \code{.cpp}
 * r[n] = y[n] / pow(2, 32);
 * \endcode
 *
 * Where `y[n]` is element `n` of the 32-bit integer output.
 *
 * Subsequences
 * ============
 *
 * The following table lists which methods are supported with the BRNG for
 * generating subsequences:
 *
 * Method | Supported?
 * ------------- | -------------
 * \ref vslLeapfrogStream | Yes
 * \ref vslSkipAheadStream | Yes
 * \ref vslSkipAheadStreamEx | No
 *
 * Note that skip-ahead of `n` skips `n` elements, not vectors, i.e. to skip `m`
 * full vectors it is required to pass `m * d` to \ref vslSkipAheadStream, where
 * `d` is the number of dimensions. If the number of skips is not a multiple of
 * `d`, a subsequent sample will begin part-way through the vector.
 *
 * Leapfrog is used to select a single dimension of the output
 * vector. The `nstreams`  parameter is ignored. `k` indicates the selected
 * dimension (must be less than the number of dimensions in the vector). The
 * resultant stream is not quite the same as a single-dimension stream, as
 * skip-ahead still operates by elements of the full vector.
 *
 * Example
 * =======
 *
 * An example of initializing a stream using SOBOL and a dimension of 3:
 *
 * \code
 *   VSLStreamStatePtr stream;
 *   int errcode = vslNewStream(&stream, VSL_BRNG_SOBOL, 3);
 * \endcode
 *
 * References
 * ==========
 *
 * [1] Bratley, P., and Fox, B.L., Sobol's quasirandom sequence generator for
 * multivariate quadrature and optimization, ACM TOMS 14 (1988) 88-100
 *
 */
// clang-format on
#define VSL_BRNG_SOBOL 0x1005

/**
 * \ingroup groupBrng
 *
 * A quasirandom number generator (QRNG) based on Gray codes.
 *
 * \warning Not implemented in \armplref.
 */
#define VSL_BRNG_NIEDERR 0x1006

/**
 * \ingroup groupBrng
 *
 * A Mersenne Twister pseudorandom number generator with a period of
 * \f$2^{19937}-1\f$, as formulated in [1], with improved initialization
 * procedure from [2].
 *
 * Recurrence relation
 * ===================
 *
 * In the following, `M` is 397, `N` is 624. Successive elements of the MT19937
 * sequence are calculated by:
 *
 * \code{.cpp}
 * x[n] = (x[n - N] & 0x80000000) | (x[n - N + 1] & 0x7FFFFFFF);
 * is_odd = x[n] & 0x1 == 1;
 * x[n] >>= 1;
 * if (is_odd)
 *   x[n] = x[n] ^ 0x9908B0DF;
 * x[n] = x[n] ^ x[n - (N - M)];
 * \endcode
 *
 * This is expressed in [1] using multiplication by a binary matrix `A`. Note
 * that `^` denotes the XOR operator.
 *
 * Integer output
 * ==============
 *
 * Integer output element `y[n]` is calculated from the MT19937 sequence element
 * `x[n]` by a 'tempering' step:
 *
 * \code{.cpp}
 * y[n] = x[n];
 * y[n] = y[n] ^ (y[n] >> 11);
 * y[n] = y[n] ^ ((y[n] << 7) & 0x9D2C5680);
 * y[n] = y[n] ^ ((y[n] << 15) & 0xEFC60000);
 * y[n] = y[n] ^ (y[n] >> 18);
 * \endcode
 *
 * Floating point output
 * =====================
 *
 * Floating point element `r[n]` calculated from `y[n]` by `r[n] = y[n] /
 * pow(2, 32)`.
 *
 * Initialization with vslNewStream
 * ================================
 *
 * Use `seed` as the seed, then initialize the generator according to the
 * initialization-by-array procedure from [2]. Note that initialization-by-array
 * procedure is always used regardless of whether multiple seed values were
 * provided.
 *
 * Initialization with vslNewStreamEx
 * ==================================
 *
 * Follow the initialization-by-array procedure as described in [2] using the
 * `params` array. If `n == 0`, initialize as though `n` was 1 and the seed
 * was 1.
 *
 * Subsequences
 * ============
 *
 * The following table lists which methods are supported with the BRNG for
 * generating subsequences:
 *
 * Method | Supported?
 * ------------- | -------------
 * \ref vslLeapfrogStream | No
 * \ref vslSkipAheadStream | Yes
 * \ref vslSkipAheadStreamEx | No
 *
 * \note MT19937 skip ahead is missing some important performance optimizations
 * in \armplref. This will be rectified in a later release.
 *
 * Example
 * =======
 *
 * An example of initializing a stream using MT19937 and a seed of 42:
 *
 * \code
 *   VSLStreamStatePtr stream;
 *   int errcode = vslNewStream(&stream, VSL_BRNG_MT19937, 42);
 * \endcode
 *
 * References
 * ==========
 *
 * [1] Matsumoto, M., and Nishimura, T., Mersenne twister: a
 * 623-dimensionally equidistributed uniform pseudo-random number generator.,
 * ACM Trans. Model. Comput. Simul. 8 (1998)
 *
 * [2] Matsumoto, M., and Nishimura, T., Mersenne Twister with improved
 * initialization,
 * http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MT2002/emt19937ar.html
 */
#define VSL_BRNG_MT19937 0x1007

/**
 * \ingroup groupBrng
 *
 * A set of 6024 Mersenne Twister pseudorandom number generators.
 *
 * \warning Not implemented in \armplref.
 */
#define VSL_BRNG_MT2203 0x1008

/**
 * \ingroup groupBrng
 *
 * An abstract random number generator for integer arrays.
 *
 * \warning Not implemented in \armplref.
 */
#define VSL_BRNG_IABSTRACT 0x1009

/**
 * \ingroup groupBrng
 *
 * An abstract random number generator for double-precision floating-point
 * arrays.
 *
 * \warning Not implemented in \armplref.
 */
#define VSL_BRNG_DABSTRACT 0x100a

/**
 * \ingroup groupBrng
 *
 * An abstract random number generator for single-precision floating-point
 * arrays.
 *
 * \warning Not implemented in \armplref.
 */
#define VSL_BRNG_SABSTRACT 0x100b

/**
 * \ingroup groupBrng
 *
 * A SIMD-oriented Mersenne Twister pseudorandom number generator, with a period
 * of \f$2^{19937}-1\f$. It's a modified version of \ref VSL_BRNG_MT19937 which
 * is designed to perform well on SIMD architectures.
 *
 * See [1] for the recurrence relation and computation of integer/floating-point
 * output.
 *
 * Initialization with vslNewStream
 * ================================
 *
 * Use `seed` as the seed, then initialize the generator according to the
 * initialization-by-array procedure from [1]. Note that initialization-by-array
 * procedure is always used regardless of whether multiple seed values were
 * provided.
 *
 * Initialization with vslNewStreamEx
 * ==================================
 *
 * Follow the initialization-by-array procedure as described in [1] using the
 * `params` array. If `n == 0`, initialize as though `n` was 1 and the seed was
 * 1.
 *
 * Subsequences
 * ============
 *
 * The following table lists which methods are supported with the BRNG for
 * generating subsequences:
 *
 * Method | Supported?
 * ------------- | -------------
 * \ref vslLeapfrogStream | No
 * \ref vslSkipAheadStream | Yes (not implemented)
 * \ref vslSkipAheadStreamEx | No
 *
 * Example
 * =======
 *
 * An example of initializing a stream using SFMT19937 and a seed of 42:
 *
 * \code
 *   VSLStreamStatePtr stream;
 *   int errcode = vslNewStream(&stream, VSL_BRNG_SFMT19937, 42);
 * \endcode
 *
 * References
 * ==========
 *
 * [1] Saito, M., Matsumoto, M. (2008). SIMD-Oriented Fast Mersenne Twister: a
 * 128-bit Pseudorandom Number Generator. In: Keller, A., Heinrich, S.,
 * Niederreiter, H. (eds) Monte Carlo and Quasi-Monte Carlo Methods 2006.
 * Springer, Berlin, Heidelberg.
 */
#define VSL_BRNG_SFMT19937 0x100c

/**
 * \ingroup groupBrng
 *
 * A nondeterministic number generator based on architectural features for
 * nondeterminism.
 *
 * Initialization with vslNewStream and vslNewStreamEx
 * ===================================================
 *
 * All initialization parameters are ignored. If nondeterminism is not
 * supported by the architecture, initialization will return
 * \ref VSL_RNG_ERROR_NONDETERM_NOT_SUPPORTED.
 *
 * Integers
 * ========
 *
 * Integer output is nondeterministic.
 *
 * Floats
 * ======
 *
 * Integer output `x[i]` transformed to float `u[i]` by the following:
 * \code{.cpp}
 * u[i] = (int)x[i] / pow(2, 32) + 0.5;
 * \endcode
 *
 * Subsequences
 * ============
 *
 * The following table lists which methods are supported with the BRNG for
 * generating subsequences:
 *
 * Method | Supported?
 * ------------- | -------------
 * \ref vslLeapfrogStream | No
 * \ref vslSkipAheadStream | No
 * \ref vslSkipAheadStreamEx | No
 *
 * Example
 * =======
 *
 * An example of initializing a stream using NONDETERM:
 *
 * \code
 *   VSLStreamStatePtr stream;
 *   int errcode = vslNewStream(&stream, VSL_BRNG_NONDETERM, 0);
 * \endcode
 *
 * Note the third parameter is ignored when using NONDETERM.
 */
#define VSL_BRNG_NONDETERM 0x100d

/**
 * \ingroup groupBrng
 *
 * An ARS-5 counter-based pseudorandom number generator.
 *
 * \warning Not implemented in \armplref.
 */
#define VSL_BRNG_ARS5 0x100e

/**
 * \ingroup groupBrng
 *
 * A counter-based pseudorandom number generator, using a 128-bit counter `c`
 * and two 32-bit integer 'keys'; `k0` and `k1`.
 *
 * Initialization with vslNewStream
 * ================================
 *
 * Initialize counter and keys from parameter `seed` according to:
 *
 * \code{.cpp}
 * k0 = seed;
 * k1 = 0;
 * c = 0;
 * \endcode
 *
 * Initialization with vslNewStreamEx
 * ==================================
 *
 * The below tables describes how `k0`, `k1` and the four constituent
 * words of `c` are initialized based on `n` and `params` array
 * (referred to as `p` for brevity).
 *
 * n | `k0` | `k1` | `c[0]` | `c[1]` | `c[2]` | `c[3]`
 * --|------|------|--------|--------|--------|-------
 * 0 | 0    | 0    | 0      | 0      | 0      | 0
 * 1 | `p[0]` | 0 | 0 | 0 | 0 | 0
 * 2 | `p[0]` | `p[1]` | 0 | 0 | 0 | 0
 * 3 | `p[0]` | `p[1]` | `p[2]` | 0 | 0 | 0
 * 4 | `p[0]` | `p[1]` | `p[2]` | `p[3]` | 0 | 0
 * 5 | `p[0]` | `p[1]` | `p[2]` | `p[3]` | `p[4]` | 0
 * >= 6 | `p[0]` | `p[1]` | `p[2]` | `p[3]` | `p[4]` | `p[5]`
 *
 * (Note `c` is assumed here to be little-endian, so `c[0]` is the
 * least significant word of `c`. For example, for `n == 4`:
 *
 * \code{.cpp}
 * c = params[2] + pow(2, 32) * params[3];
 * \endcode
 *
 * Recurrence relation
 * ===================
 *
 * State is advanced every fourth 32-bit sample by:
 *
 * \code{.cpp}
 * c = c + 1;
 * \endcode
 *
 * Integer output
 * ==============
 *
 * Integer output is then calculated by performing 10 rounds of Philox
 * on `c`, where one round of Philox is defined in [1] as:
 *
 * \code{.cpp}
 * R0 = c[0]; L0 = c[1]; R1 = c[2]; L1 = c[3];
 * L1_ = mullo(R0, 0xD2511F53);
 * R1_ = mulhi(R0, 0xD2511F53) ^ k1 ^ L1;
 * L0_ = mullo(R1, 0xCD9E8D57);
 * R0_ = mulhi(R1, 0xCD9E8D57) ^ k0 ^ L0;
 * c   = {R0_, L0_, R1_, L1_};
 * \endcode
 *
 * Where `mullo` is the low word of the result of a 64-bit
 * multiplication, `mulhi` is the high word, and `^` denotes the XOR operator.
 * `k0` and `k1` are updated between subsequent Philox rounds using constants
 * defined in [1], but these updates do not persist after a state update. The
 * Philox-10 calculation always begins with the seeded values of `k0`
 * and `k1`, and can be expressed as:
 *
 * \code
 * Input: c, k0, k1
 * for i from 1 to 10:
 *   update c = philox(c, k0, k1)
 *   update k0 = k0 + 0x9E3779B9
 *   update k1 = k1 + 0xBB67AE85
 * return c
 * \endcode
 *
 * Output `y[i]` of the integer sequence is hence element `i % 4` of
 * `philox_10(c0 + i/4)`, where `c0` is the seeded value of `c` and
 * `i/4` is rounded down.
 *
 * Float output
 * ============
 *
 * Output `r[i]` of the floating point sequence is defined by
 *
 * \code{.py}
 * r[i] = (int)y[i] / pow(2, 32) + 0.5
 * \endcode
 *
 * Subsequences
 * ============
 *
 * The following table lists which methods are supported with the BRNG for
 * generating subsequences:
 *
 * Method | Supported?
 * ------------- | -------------
 * \ref vslLeapfrogStream | No
 * \ref vslSkipAheadStream | Yes
 * \ref vslSkipAheadStreamEx | Yes
 *
 * Example
 * =======
 *
 * An example of initializing a stream using PHILOX4X32X10 and a seed of 42:
 *
 * \code
 *   VSLStreamStatePtr stream;
 *   int errcode = vslNewStream(&stream, VSL_BRNG_PHILOX4X32X10, 42);
 * \endcode
 *
 * References
 * ==========
 *
 * [1] J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random
 * numbers: As easy as 1, 2, 3, " SC '11: Proceedings of 2011 International
 * Conference for High Performance Computing, Networking, Storage and Analysis.
 *
 */
#define VSL_BRNG_PHILOX4X32X10 0x100f

/**
 * \ingroup groupDistMethods
 *
 * This will initially invoke the generation of a sequence of uniformly
 * distributed numbers within the interval \f$[0, 1)\f$ through the chosen
 * generator, see \ref groupBrng.
 *
 * In the case of single or double precision floating point uniform
 * distribution, this will transform the generated floating point numbers into
 * numbers within the domain \f$[a, b)\f$, with \f$a\f$ and \f$b\f$ both in
 * \f$\mathbb{R}\f$. For some particular values of the lower and upper bounds of
 * the domain, the generated values might be outside the interval \f$[a, b)\f$.
 * To prevent this from happening, and in case where it is very important that
 * every generated value is within the domain, the `ACCURATE` version of this
 * method has to be used.
 *
 * For the integer uniform distribution, this method will invoke the floor
 * operation on the generated, possibly scaled and translated, random number
 * \f$u\in[a, b)\f$.
 */
#define VSL_RNG_METHOD_UNIFORM_STD 0

/**
 * \ingroup groupDistMethods
 *
 * This will initially invoke the generation of a sequence of uniformly
 * distributed numbers within the interval \f$[0, 1)\f$ through the chosen
 * generator, see \ref groupBrng.
 *
 * In the case of single or double precision floating point uniform
 * distribution, this will transform the generated floating point numbers into
 * numbers within the domain \f$[a, b)\f$, with \f$a\f$ and \f$b\f$ both in
 * \f$\mathbb{R}\f$. This method will guarantee that every generated value is
 * within the domain \f$[a, b)\f$.
 *
 * For the integer uniform distribution, this method is not supported.
 */
#define VSL_RNG_METHOD_UNIFORM_STD_ACCURATE 1

/**
 * \ingroup groupDistMethods
 *
 * As part of the generation of a uniformly distributed sequence of
 * numbers within the \f$[0, 1)\f$ domain, each BRNG (see \ref groupBrng)
 * produces a sequence of integer numbers, which then get normalized
 * to return a floating point number between \f$[0, 1)\f$. Using this
 * method will return the interpretation of each integer value as 32-bit
 * integer output using the integer recurrence relation of each BRNG.
 */
#define VSL_RNG_METHOD_UNIFORMBITS_STD 0

/**
 * \ingroup groupDistMethods
 *
 * Given a BRNG (see \ref groupBrng), for which this method is
 * supported, this method transforms the underlying BRNG integer
 * recurrence so that all bits in 32-bit chunks are uniformly
 * distributed.
 *
 * This method is not supported by the following BRNGs:
 *   - \ref VSL_BRNG_MCG31
 *   - \ref VSL_BRNG_R250
 *   - \ref VSL_BRNG_MRG32K3A
 *   - \ref VSL_BRNG_SOBOL
 */
#define VSL_RNG_METHOD_UNIFORMBITS32_STD 0

/**
 * \ingroup groupDistMethods
 *
 * Similarly to the \ref VSL_RNG_METHOD_UNIFORMBITS32_STD, the use of
 * this method transforms the underlying BRNG (see \ref groupBrng) integer
 * recurrence so that all bits in 64-bit chunks are uniformly distributed.
 *
 * This method is not supported by the following BRNGs:
 *   - \ref VSL_BRNG_MCG31
 *   - \ref VSL_BRNG_R250
 *   - \ref VSL_BRNG_MRG32K3A
 *   - \ref VSL_BRNG_SOBOL
 */
#define VSL_RNG_METHOD_UNIFORMBITS64_STD 0

/**
 * \ingroup groupDistMethods
 *
 * This invokes the BoxMuller method to generate a normally distributed
 * random number \f$x\f$ from a pair of uniformly distributed numbers
 * \f$u_1\f$ \f$u_2\f$, with \f$u_1\f$ and \f$u_2\f$ in \f$[0, 1)\f$,
 * according to the formulas:
 * \f[
 *     x = \sqrt{(-2\ln{u_1})} \sin(2\pi u_2)
 * \f]
 */
#define VSL_RNG_METHOD_GAUSSIAN_BOXMULLER 0

/**
 * \ingroup groupDistMethods
 *
 * This invokes the BoxMuller2 method to generate normally distributed
 * random numbers \f$x_1\f$ and \f$x_2\f$ from a pair of uniformly
 * distributed numbers \f$u_1\f$ \f$u_2\f$, with \f$u_1\f$ and \f$u_2\f$
 * in \f$[0, 1)\f$, according to the formulas:
 * \f[
 *  \begin{array}{ll}
 *     x_1 = \sqrt{(-2\ln{u_1})} \sin(2\pi u_2) \\
 *     x_2 = \sqrt{(-2\ln{u_1})} \cos(2\pi u_2) \\
 *  \end{array}
 * \f]
 */
#define VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2 1

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method
 * to calculate the Gaussian distribution from a set of uniformly
 * distributed numbers in the interval \f$[0, 1)\f$.
 */
#define VSL_RNG_METHOD_GAUSSIAN_ICDF 2

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method
 * to calculate the Laplace distribution from a set of uniformly
 * distributed numbers in the interval \f$[0, 1)\f$.
 */
#define VSL_RNG_METHOD_LAPLACE_ICDF 0

/**
 * \ingroup groupDistMethods
 *
 * This invokes the generation of Gaussian distributed numbers through
 * the Inverse Cumulative Distribution Function method, from a set of
 * uniformly distributed numbers in the interval \f$[0, 1)\f$, to which
 * then apply an appropriate transformation to produce random numbers
 * distributed following a lognormal distribution.
 */
#define VSL_RNG_METHOD_LOGNORMAL_ICDF 0

/**
 * \ingroup groupDistMethods
 *
 * This invokes the generation of Gaussian distributed numbers through
 * the BoxMuller2 method, from a set of uniformly distributed numbers in
 * the interval \f$[0, 1)\f$, to which the apply an appropriate
 * transformation to produce random numbers distributed following a
 * lognormal distribution.
 */
#define VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2 2

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method to calculate
 * the exponential distribution from a set of uniformly distributed numbers in
 * the interval \f$[0, 1)\f$. It may happen that for certain values of the
 * displacement quantity and scale factor for the exponential distribution, the
 * generated values are outside the domain. To fix this issue, use the
 * `ACCURATE` version of this method.
 */
#define VSL_RNG_METHOD_EXPONENTIAL_ICDF 0

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method to calculate
 * the exponential distribution from a set of uniformly distributed numbers in
 * the interval \f$[0, 1)\f$. It may happen that for certain values of the
 * displacement quantity and scale factor for the exponential distribution, the
 * generated values are outside the domain. Using this method will invoke the
 * ICDF method to generate values that follow an exponential distribution and
 * will enforce that all returned values are within the domain.
 */
#define VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE 1

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method to calculate
 * the Rayleigh distribution from a set of uniformly distributed numbers in the
 * interval \f$[0, 1)\f$. It may happen that for certain values of the shape
 * parameter, displacement quantity, and scale factor for the Rayleigh
 * distribution, the generated values are outside the domain. To fix this issue,
 * use the `ACCURATE` version of this method.
 */
#define VSL_RNG_METHOD_RAYLEIGH_ICDF 0

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method to calculate
 * the Rayleigh distribution from a set of uniformly distributed numbers in the
 * interval \f$[0, 1)\f$. It may happen that, for certain values of the
 * distribution parameters, the generated values are outside the domain. Using
 * this method will invoke the ICDF method to generate values that follow a
 * Rayleigh distribution and will enforce that all returned values are within
 * the domain.
 */
#define VSL_RNG_METHOD_RAYLEIGH_ICDF_ACCURATE 1

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method to calculate
 * the Weibull distribution from a set of uniformly distributed numbers in the
 * interval \f$[0, 1)\f$. It may happen that for certain values of the shape
 * parameter, displacement quantity, and scale factor for the Weibull
 * distribution, the generated values are outside the domain. To fix this issue,
 * use the `ACCURATE` version of this method.
 */
#define VSL_RNG_METHOD_WEIBULL_ICDF 0

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method to calculate
 * the Weibull distribution from a set of uniformly distributed numbers in the
 * interval \f$[0, 1)\f$. It may happen that, for certain values of the
 * distribution parameters, the generated values are outside the domain. Using
 * this method will invoke the ICDF method to generate values that follow a
 * Weibull distribution and will enforce that all returned values are within the
 * domain.
 */
#define VSL_RNG_METHOD_WEIBULL_ICDF_ACCURATE 1

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method
 * to calculate the Gumbel distribution from a set of uniformly
 * distributed numbers in the interval \f$[0, 1)\f$.
 */
#define VSL_RNG_METHOD_GUMBEL_ICDF 0

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method
 * to calculate the Bernoulli distribution from a set of uniformly
 * distributed numbers in the interval \f$[0, 1)\f$.
 */
#define VSL_RNG_METHOD_BERNOULLI_ICDF 0

/**
 * \ingroup groupDistMethods
 *
 * This will invoke the BTPE (Binomial, Triangle, Parallelogram,
 * Exponential) method, see [1], using accept/reject approach, to
 * transform a sequence of uniformly distributed numbers between
 * \f$[0, 1)\f$ into a sequence following a binomial distribution.
 *
 * References
 * ==========
 *
 * [1] Kachitvichyanukul, Voratas and Schmeise, Bruce W.,
 * "Binomial Random Variate Generation",
 * Communications of the ACM Vol. 31, 1988,
 * https://dl.acm.org/doi/pdf/10.1145/42372.42381
 */
#define VSL_RNG_METHOD_BINOMIAL_BTPE 0

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method
 * to calculate the geometric distribution from a set of uniformly
 * distributed numbers in the interval \f$[0, 1)\f$.
 */
#define VSL_RNG_METHOD_GEOMETRIC_ICDF 0

/**
 * \ingroup groupDistMethods
 *
 * This invokes the Inverse Cumulative Distribution Function method
 * to calculate the Cauchy distribution from a set of uniformly
 * distributed numbers in the interval \f$[0, 1)\f$.
 */
#define VSL_RNG_METHOD_CAUCHY_ICDF 0

/**
 * \ingroup groupDistMethods
 *
 * Poisson PTPE algorithm
 */
#define VSL_RNG_METHOD_POISSON_PTPE 0

/**
 * \ingroup groupDistMethods
 *
 * Poisson from normally distributed random numbers
 */
#define VSL_RNG_METHOD_POISSON_POISNORM 1

/**
 * \ingroup groupTypes
 *
 * \typedef openrng_int_t
 *
 * Generalized signed integer type.
 *
 *  * 32 bits if built in LP64 mode.
 *  * 64 bits if built in ILP64 mode.
 *
 * The following syntax section assumes LP64. See \int64libs for the different
 * integer modes in \armplref.
 */

/**
 * \ingroup groupTypes
 *
 * \typedef openrng_uint_t
 *
 * Generalized unsigned integer type.
 *
 *  * 32 bits if built in LP64 mode.
 *  * 64 bits if built in ILP64 mode.
 *
 * The following syntax section assumes LP64. See \int64libs for the different
 * integer modes in \armplref.
 */

#if defined(OPENRNG_ILP64) || defined(INTEGER64)
typedef long long openrng_int_t;
typedef unsigned long long openrng_uint_t;
#else
typedef int32_t openrng_int_t;
typedef uint32_t openrng_uint_t;
#endif

/**
 * \ingroup groupTypes
 *
 * 64 bit unsigned integer type.
 */
typedef unsigned long long openrng_uint64_t;

/**
 * \ingroup groupTypes
 *
 * Opaque pointer to a stream handle.
 */
typedef void *VSLStreamStatePtr;

/**
 * \ingroup groupTypes
 *
 * Not used in \armplref.
 */
typedef int (*InitStreamPtr)(int method, VSLStreamStatePtr stream, int n,
                             const unsigned int params[]);

/**
 * \ingroup groupTypes
 *
 * Not used in \armplref.
 */
typedef int (*sBRngPtr)(VSLStreamStatePtr stream, int n, float r[], float a,
                        float b);

/**
 * \ingroup groupTypes
 *
 * Not used in \armplref.
 */
typedef int (*dBRngPtr)(VSLStreamStatePtr stream, int n, double r[], double a,
                        double b);

/**
 * \ingroup groupTypes
 *
 * Not used in \armplref.
 */
typedef int (*iBRngPtr)(VSLStreamStatePtr stream, int n, unsigned int r[]);

/**
 * \ingroup groupTypes
 *
 * Not used in \armplref.
 */
typedef int (*iUpdateFuncPtr)(VSLStreamStatePtr stream, int *n,
                              unsigned int ibuf[], int *nmin, int *nmax,
                              int *idx);

/**
 * \ingroup groupTypes
 *
 * Not used in \armplref.
 */
typedef int (*dUpdateFuncPtr)(VSLStreamStatePtr stream, int *n, double dbuf[],
                              int *nmin, int *nmax, int *idx);

/**
 * \ingroup groupTypes
 *
 * Not used in \armplref.
 */
typedef int (*sUpdateFuncPtr)(VSLStreamStatePtr stream, int *n, float sbuf[],
                              int *nmin, int *nmax, int *idx);

/**
 * \ingroup groupTypes
 *
 * Struct returned by \ref vslGetBrngProperties. It contains runtime information
 * about the BRNG. As of \armplref, the only field with a reliable value is
 * `WordSize`, which is needed for the correct usage of \ref viRngUniformBits.
 */
typedef struct {
  int StreamStateSize;      ///< Not used.
  int NSeeds;               ///< Not used.
  int IncludesZero;         ///< Not used.
  int WordSize;             ///< Number of bytes produced per iteration.
  int NBits;                ///< Not used.
  InitStreamPtr InitStream; ///< Not used.
  sBRngPtr sBRng;           ///< Not used.
  dBRngPtr dBRng;           ///< Not used.
  iBRngPtr iBRng;           ///< Not used.
} VSLBRngProperties;

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the beta
 * distribution.
 *
 * The parameters for this distribution are the shape parameters for
 * the beta distribution \f$p\f$ and \f$q\f$, a displacement quantity
 * \f$\alpha\f$, and, a scale factor \f$\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; p, q, \alpha, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{1}{B(p, q)\beta^{p+q-1}}(x-\alpha)^{p-1}
 *      (\beta+\alpha-x)^q-1, & \alpha < x < \alpha + \beta \\
 *      0, & x < \alpha, x > \alpha + \beta,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$, where \f$B(p, q)\f$ is the complete
 * beta function. Parameters \f$p, q, \alpha, \beta \in \mathbb{R}\f$,
 * with \f$p >0, q>0\f$, and \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Beta distribution.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Beta distribution.
 * \param[in] p Shape parameter for the Beta function.
 * Constraint: `p` > 0.
 * \param[in] q Shape parameter for the Beta function.
 * Constraint: `q` > 0.
 * \param[in] alpha Displacement quantity.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vdRngBeta(const openrng_int_t method, VSLStreamStatePtr stream,
              const openrng_int_t n, double r[], const double p, const double q,
              const double alpha, const double beta);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the Cauchy
 * distribution. The parameters for this distribution are a displacement
 * quantity \f$a\f$ and a scale factor \f$\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \beta) = \displaystyle \frac{1}{\displaystyle
 *      \pi\beta\left(1 + \left(\frac{x-a}{\beta}\right)^2\right)},
 * \f]
 *
 * for \f$-\infty < x < \infty\f$. Parameters \f$\alpha, \beta
 * \in\mathbb{R}\f$, with \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Cauchy distribution. Valid value for `method` is:
 *   - \ref VSL_RNG_METHOD_CAUCHY_ICDF.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Cauchy distribution.
 * \param[in] a Displacement quantity.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vdRngCauchy(const openrng_int_t method, VSLStreamStatePtr stream,
                const openrng_int_t n, double r[], const double a,
                const double beta);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the
 * chi-squared distribution.
 *
 * \warning Not implemented in \armplref.
 *
 * The parameter for this distribution is the number of degrees of freedom,
 * \f$v\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; v) = \frac{x^{(v-2)/2} \exp(-x/2)}{2^{v/2} \Gamma(v/2)},
 * \f]
 *
 * for \f$x\in[0, \infty)\f$. Parameter \f$v\in\mathbb{N}\f$, with \f$v >
 * 0\f$.
 *
 * \param[in] method Method used for transforming from a uniform
 * distribution to a chi-squared distribution.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a chi-squared distribution.
 * \param[in] v Numbers of degrees of freedom.
 * Constraint: `v` > 0.
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vdRngChiSquare(const openrng_int_t method, VSLStreamStatePtr stream,
                   const openrng_int_t n, double r[], const int v);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the
 * exponential distribution. The parameters for this distribution are a
 * displacement quantity \f$a\f$, and a scale factor \f$\beta>0\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{1}{\beta} \exp\left(
 *      \displaystyle -\frac{x-a}{\beta}\right), & x \geq a \\
 *      0, & x < a,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$. Parameters \f$\alpha, \beta \in
 * \mathbb{R}\f$, with \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to an exponential distribution. Valid values for
 * `method` are:
 *  - \ref VSL_RNG_METHOD_EXPONENTIAL_ICDF.
 *  - \ref VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of an exponential distribution.
 * \param[in] a Displacement factor.
 * \param[in] beta Scaling factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vdRngExponential(const openrng_int_t method, VSLStreamStatePtr stream,
                     const openrng_int_t n, double r[], const double a,
                     const double beta);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the Gamma
 * distribution.
 *
 * \warning Not implemented in \armplref.
 *
 * The parameters of this distribution are its shape parameter
 * \f$\alpha\f$, a displacement \f$a\f$, and, a scale factor
 * \f$\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; \alpha, a, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{1}{\Gamma(\alpha)\beta^{\alpha}}
 *      (x-a)^{\alpha-1} \exp\left( \displaystyle -\frac{(x-a)}
 *      {\beta}\right), & x \geq a \\
 *      0, & x < a,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$, where \f$\alpha, \beta, a \in
 * \mathbb{R}\f$ and \f$\alpha>0\f$ and \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Gamma distribution.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Gamma distribution.
 * \param[in] a Displacement quantity.
 * \param[in] alpha Shape parameter for the Gamma function.
 * Constraint: `alpha` > 0.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vdRngGamma(const openrng_int_t method, VSLStreamStatePtr stream,
               const openrng_int_t n, double r[], const double a,
               const double alpha, const double beta);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the
 * Gaussian distribution. The parameters of this distribution are its mean value
 * \f$a\f$ and its standard deviation \f$\sigma\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \sigma) = \displaystyle \frac{1}{\sigma\sqrt{2\pi}}
 *      \exp\left(\displaystyle -\frac{(x-a)^2}{2\sigma^2}\right),
 * \f]
 *
 * for \f$-\infty < x < \infty\f$, where \f$a, \sigma \in \mathbb{R}\f$,
 * with \f$\sigma > 0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Gaussian distribution. Valid values for `method`
 * are:
 *   - \ref VSL_RNG_METHOD_GAUSSIAN_ICDF.
 *   - \ref VSL_RNG_METHOD_GAUSSIAN_BOXMULLER.
 *   - \ref VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Gaussian distribution.
 * \param[in] a Mean value.
 * \param[in] sigma Standard deviation.
 * Constraint: `sigma` > 0.
 * \return See \ref groupError.
 */
int vdRngGaussian(const openrng_int_t method, VSLStreamStatePtr stream,
                  const openrng_int_t n, double r[], const double a,
                  const double sigma);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the
 * multivariate Gaussian distribution of dimension \f$d\f$.
 *
 * \warning Not implemented in \armplref.
 *
 * The parameters of this distribution are a vector \f$a\f$ of
 * length \f$d\f$ of mean values, a variance-covariance matrix
 * \f$C\f$, of dimension \f$d\times d\f$. The matrix \f$C\f$ is
 * symmetric and positive-definite.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, C) = \displaystyle \frac{1}{\sqrt{\text{det}(2\pi C)}}
 *      \exp\left(\displaystyle -\frac{1}{2(x-a)^T C^{-1} (x-a)}\right),
 * \f]
 *
 * with \f$x\in\mathbb{R}^d\f$, and \f$-\infty < x_i < \infty\f$, for
 * each \f$i\in\{1, \ldots, d\}\f$. In each dimension \f$i\f$, with
 * \f$i < d\f$, \f$a_i\f$ is the mean value of the Gaussian distribution
 * in that dimension; therefore \f$a \in\mathbb{R}^d\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a multivariate Gaussian distribution.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a multivariate Gaussian
 * distribution.
 * \param[in] dimen Dimension of the multivariate distribution.
 * Constraint: `d` > 0.
 * \param[in] mstorage Storage scheme for lower triangular matrices.
 * \param[in] a Vector of mean values.
 * \param[in] t Elements of the lower triangular part of the
 *   variance-covariance matrix. Elements are store according to the
 *   storage scheme in `mstorage`.
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vdRngGaussianMV(const openrng_int_t method, VSLStreamStatePtr stream,
                    const openrng_int_t n, double r[],
                    const openrng_int_t dimen, const openrng_int_t mstorage,
                    const double *a, const double *t);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the Gumbel
 * distribution. The parameters of this distribution are a displacement quantity
 * \f$a\f$ and a scale factor \f$\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \beta) = \displaystyle \frac{1}{\sqrt{\beta}}
 *      \exp\left(
 *      \displaystyle \frac{x-a}{\beta} - \text{exp}\left(
 *      \displaystyle \frac{x-a}{\beta}\right)\right),
 * \f]
 *
 * for \f$-\infty < x < \infty\f$, where \f$a \in \mathbb{R}\f$, and
 * \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Gumbel distribution. Valid value for `method` is:
 *   - \ref VSL_RNG_METHOD_GUMBEL_ICDF.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Gumbel distribution.
 * \param[in] a Displacement quantity.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vdRngGumbel(const openrng_int_t method, VSLStreamStatePtr stream,
                const openrng_int_t n, double r[], const double a,
                const double beta);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the
 * Laplace distribution. The parameters of this distribution are its mean
 * \f$a\f$ and a scale factor \f$\beta\f$. The standard deviation \f$\sigma\f$
 * of the distribution can be calculated from \f$\beta\f$ by the relation
 * \f$\sigma=\sqrt{2}\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \beta) = \displaystyle \frac{1}{\sqrt{2\beta}} \exp\left(
 *      \displaystyle -\frac{|x-a|}{\beta}\right),
 * \f]
 *
 * for \f$-\infty < x < \infty\f$. The parameters \f$a\f$ and
 * \f$\beta\f$ are such that \f$a, \beta \in\mathbb{R}\f$ and
 * \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Laplace distribution. Valid value for `method` is:
 *   - \ref VSL_RNG_METHOD_LAPLACE_ICDF.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Laplace distribution.
 * \param[in] a Mean of the distribution.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vdRngLaplace(const openrng_int_t method, VSLStreamStatePtr stream,
                 const openrng_int_t n, double r[], const double a,
                 const double beta);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the
 * lognormal distribution.
 *
 * \warning Not implemented in \armplref.
 *
 * The parameters of this distribution are the mean \f$a\f$ and standard
 * deviation \f$\sigma\f$ of the normal distribution obtained from its natural
 * logarithm, a displacement value \f$b\f$ and, scale factor \f$\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \sigma, b, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{1}{\sqrt{2\pi}\sigma(x-b)} \exp\left( \displaystyle
 *                    -\frac{\left[\ln\left(
 *      \displaystyle \frac{x-b}{\beta}\right)-a\right]^2 }{2\sigma^2}\right),
 *      & x > b \\
 *      0, & x \leq b,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$. Above, \f$\alpha, \sigma, b,
 * and \beta \in \mathbb{R}\f$, with \f$\sigma>0\f$ and \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a lognormal distribution.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a lognormal distribution.
 * \param[in] alpha Mean of the underlying normal distribution.
 * \param[in] sigma Standard deviation of the underlying normal distribution.
 * Constraint: `sigma` > 0.
 * \param[in] b Displacement value for the lognormal distribution.
 * \param[in] beta Scale factor for the lognormal distribution.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vdRngLognormal(const openrng_int_t method, VSLStreamStatePtr stream,
                   const openrng_int_t n, double r[], const double alpha,
                   const double sigma, const double b, const double beta);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the
 * Rayleigh distribution. The parameters for this distribution are a
 * displacement quantity \f$a\f$ and a scale factor \f$\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{2(x-a)}{\beta^2} \exp\left( \displaystyle
 *                    -\frac{(x-a)^2}{\beta^2}\right), & x \geq a \\
 *      0, & x < a,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$. The parameters \f$a,
 * \beta\in\mathbb{R}\f$, with \f$\beta>0\f$. The Rayleigh distribution
 * is a special case of the Weibull distribution with shape parameter
 * \f$\alpha=2\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Rayleigh distribution. Valid values for `method`
 * are:
 *   - \ref VSL_RNG_METHOD_RAYLEIGH_ICDF.
 *   - \ref VSL_RNG_METHOD_RAYLEIGH_ICDF_ACCURATE.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Rayleigh distribution.
 * \param[in] a Displacement factor.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vdRngRayleigh(const openrng_int_t method, VSLStreamStatePtr stream,
                  const openrng_int_t n, double r[], const double a,
                  const double beta);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the
 * uniform distribution in the interval \f$[a, b)\f$. The lower and upper bound
 * of the domain have to satisfy \f$a < b\f$.
 *
 * The probability distribution is given by:
 *
 * \f[
 *      f(x; a, b) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{1}{b - a}, & a \leq x < b, \\
 *      0, & \text{otherwise},
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to another, possibly rescaled and translated, uniform
 * distribution. Valid values for `method` are:
 *   - \ref VSL_RNG_METHOD_UNIFORM_STD.
 *   - \ref VSL_RNG_METHOD_UNIFORM_STD_ACCURATE.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a uniform distribution.
 * \param[in] a Lower bound of the interval.
 * Constraint: `a < b`.
 * \param[in] b Upper bound of the interval.
 * Constraint: `b > a`.
 * \return See \ref groupError.
 */
int vdRngUniform(const openrng_int_t method, VSLStreamStatePtr stream,
                 const openrng_int_t n, double r[], const double a,
                 const double b);

/**
 * \ingroup groupContinuous
 *
 * Generates double-precision random numbers distributed according to the
 * Weibull distribution. The parameters for this distribution are a shape
 * parameter \f$\alpha\f$, a displacement quantity \f$a\f$, and, a scale
 * quantity \f$\beta\f$.
 *
 * The probability distribution is given by:
 *
 * \f[
 *      f(x; \alpha, a, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{\alpha}{\beta^{\alpha}} (x-a)^{\alpha-1}
 *      \exp\left(-\left(\displaystyle \frac{x-a}{\beta}\right)^{\alpha}\right),
 *      & x \geq a \\
 *      0, & x < a,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$, where \f$a, \alpha, \beta
 * \in\mathbb{R}\f$, with \f$\alpha > 0\f$, and \f$\beta >0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Weibull distribution. Valid values for `method`
 * are:
 *   - \ref VSL_RNG_METHOD_WEIBULL_ICDF.
 *   - \ref VSL_RNG_METHOD_WEIBULL_ICDF_ACCURATE.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Weibull distribution.
 * \param[in] alpha Shape parameter.
 * Constraint: `alpha` > 0.
 * \param[in] a Displacement quantity.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vdRngWeibull(const openrng_int_t method, VSLStreamStatePtr stream,
                 const openrng_int_t n, double r[], const double alpha,
                 const double a, const double beta);

/**
 * \ingroup groupDiscrete
 *
 * Generates integer variates distributed according to the Bernoulli
 * distribution.
 *
 * The probability function can be defined as
 *
 * \f[
 * P(X=k;\ p)=\begin{cases}
 *            1-p \quad &\text{if} \, k=0 \\
 *            p   \quad &\text{if} \, k=1 \\
 *          \end{cases}
 * \f]
 *
 * for \f$k\in\{0, 1\}\f$. Above, the probability of success for each
 * trial is indicated by the parameters \f$p\f$, with \f$p\in(0, 1)\f$.
 *
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Bernoulli distribution. Valid value for `method`
 * is:
 *   - \ref VSL_RNG_METHOD_BERNOULLI_ICDF.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Bernoulli distribution.
 * \param[in] p Bernoulli trial success probability.
 * Constraint: `p` in `(0, 1)`.
 * \return See \ref groupError.
 */
int viRngBernoulli(const openrng_int_t method, VSLStreamStatePtr stream,
                   const openrng_int_t n, int r[], const double p);

/**
 * \ingroup groupDiscrete
 *
 * Generates integer variates according to the Binomial distribution.
 *
 * The probability function can be defined as
 *
 * \f[
 * P(X=k;\ N, \ p) = \binom{N}{k} p^k (1-p)^{N-k}\,
 * \f]
 *
 * for \f$k\in\{0, 1, 2, ..., N\}\f$. The parameters for this distribution are
 * the number of independent trials \f$N\in\{0, 1, 2, ...\}\f$, and the
 * probability of success in each trial, \f$p\in(0, 1)\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a binomial distribution. Valid value for `method` is:
 *   - \ref VSL_RNG_METHOD_BINOMIAL_BTPE.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a binomial distribution.
 * \param[in] ntrial Number of independent trials.
 * Constraint: `ntrial` > 0.
 * \param[in] p Trial success probability.
 * Constraint: `p` in `(0, 1)`.
 * \return See \ref groupError.
 */
int viRngBinomial(const openrng_int_t method, VSLStreamStatePtr stream,
                  const openrng_int_t n, int r[], const int ntrial,
                  const double p);

/**
 * \ingroup groupDiscrete
 *
 * Generates integer variates according to the Geometric distribution.
 *
 * The probability function can be defined as
 *
 * \f[
 * P(X=k;\ p) = (1-p)^{k}p\,
 * \f]
 *
 * for \f$k\in\{0, 1, 2, ...\}\f$. Above, the probability of success for
 * each individual trial is indicated by the parameter \f$p\f$, with
 * \f$p\in[0, 1]\f$.
 *
 * \param[in] method Method used for transforming from a uniform
 * distribution to a geometric distribution. Valid value for `method`
 * is:
 *   - \ref VSL_RNG_METHOD_GEOMETRIC_ICDF.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[in] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a geometric distribution.
 * \param[in] p Trial success probability.
 * Constraint: `p` in `(0, 1)`.
 * \return See \ref groupError.
 */
int viRngGeometric(const openrng_int_t method, VSLStreamStatePtr stream,
                   const openrng_int_t n, int r[], const double p);

/**
 * \ingroup groupDiscrete
 *
 * \warning Not implemented in \armplref.
 *
 * \param[in] method
 * \param[in] stream
 * \param[in] n
 * \param[in] r
 * \param[in] l
 * \param[in] s
 * \param[in] m
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int viRngHypergeometric(const openrng_int_t method, VSLStreamStatePtr stream,
                        const openrng_int_t n, int r[], const int l,
                        const int s, const int m);

/**
 * \ingroup groupDiscrete
 *
 * \warning Not implemented in \armplref.
 *
 * \param[in] method
 * \param[in] stream
 * \param[in] n
 * \param[in] r
 * \param[in] ntrial
 * \param[in] k
 * \param[in] p
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int viRngMultinomial(const openrng_int_t method, VSLStreamStatePtr stream,
                     const openrng_int_t n, int r[], const int ntrial,
                     const int k, const double p[]);

/**
 * \ingroup groupDiscrete
 *
 * \warning Not implemented in \armplref.
 *
 * \param[in] method
 * \param[in] stream
 * \param[in] n
 * \param[in] r
 * \param[in] a
 * \param[in] p
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int viRngNegBinomial(const openrng_int_t method, VSLStreamStatePtr stream,
                     const openrng_int_t n, int r[], const double a,
                     const double p);

/**
 * \ingroup groupDiscrete
 *
 * \warning Not implemented in \armplref.
 *
 * \param[in] method
 * \param[in] stream
 * \param[in] n
 * \param[in] r
 * \param[in] a
 * \param[in] p
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int viRngNegbinomial(const openrng_int_t method, VSLStreamStatePtr stream,
                     const openrng_int_t n, int r[], const double a,
                     const double p);

/**
 * \ingroup groupDiscrete
 *
 * @brief Generates integer variates according to the Poisson distribution.
 *
 * The probability function can be defined as
 *
 * \f[
 * P(X=k;\ \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}\,
 * \f]
 *
 * for \f$k\in[0, \infty]))\f$.
 *
 * @param[in] method Method used for transforming from a uniform sequence.
 *   Valid values are `VSL_RNG_METHOD_POISSON_PTPE` and
 *   `VSL_RNG_METHOD_POISSON_POISNORM` (not implemented in \armplref).
 * @param[in] stream Pointer to an initialized stream.
 * @param[in] n Number of output elements.
 * @param[in] r Buffer for storing output elements.
 * @param[in] lambda Expected rate of occurrences \f$\lambda>0\f$.
 * @return See \ref groupError.
 */
int viRngPoisson(const openrng_int_t method, VSLStreamStatePtr stream,
                 const openrng_int_t n, int r[], const double lambda);

/**
 * \ingroup groupDiscrete
 *
 * \warning Not implemented in \armplref.
 *
 * \param[in] method
 * \param[in] stream
 * \param[in] n
 * \param[in] r
 * \param[in] lambda
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int viRngPoissonV(const openrng_int_t method, VSLStreamStatePtr stream,
                  const openrng_int_t n, int r[], const double lambda[]);

/**
 * \ingroup groupDiscrete
 *
 * Generates integer random numbers distributed according to the uniform
 * distribution in the interval \f$\{a, a+1, \ldots, b-1\}\f$, with \f$a, b
 * \in\mathbb{N}\f$. The lower and upper bound of the domain have to satisfy
 * \f$a < b\f$.
 *
 * The probability distribution is given by:
 *
 * \f[
 *      P(X=k) = \displaystyle \frac{1}{b - a},
 *      \text{ }k \in \{a, a+1, \ldots, b-1\}.
 * \f]
 *
 * \param[in] method Method used for transforming from a uniform
 * sequence to another, possibly rescaled and translated, uniform
 * distribution. Valid value for `method` is:
 *   - \ref VSL_RNG_METHOD_UNIFORM_STD.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a uniform distribution.
 * \param[in] a Lower bound of the domain.
 * Constraint: `a < b`.
 * \param[in] b Upper bound of the domain.
 * Constraint: `b > a`.
 * \return See \ref groupError.
 */
int viRngUniform(const openrng_int_t method, VSLStreamStatePtr stream,
                 const openrng_int_t n, int r[], const int a, const int b);

/**
 * \ingroup groupDiscrete
 *
 * Generates integer random values with uniform bit distribution. The generators
 * of uniformly distributed numbers can be represented as recurrence relations
 * over integer values in modular arithmetic. This function provides the output
 * of underlying integer recurrence that characterizes a specific generator.
 * However, for pseudorandom generators, the randomness of each individual bit
 * can be violated as some bits in particular regions (e.g. the lower bits) of
 * the vector that represents a number, are less random than in other portions
 * of the same vector. This function does not guarantee uniformity across all
 * individual bits and, therefore, it should be used with care. Alternatively,
 * users are advised that the functions \ref viRngUniformBits32 and \ref
 * viRngUniformBits64 guarantee that all bits are uniformly distributed.
 *
 * \param[in] method Method used for transforming from a uniform sequence. Valid
 * value for `method` is:
 *   - \ref VSL_RNG_METHOD_UNIFORMBITS_STD.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[in, out] buffer On input, a pre-allocated buffer, of at least size
 *   `n*WordSize`, where `WordSize` is dependent on the generator being used. A
 *   generator's `WordSize` can be queried using \ref VSLBRngProperties. On
 *   output, the buffer contains the bit representation of `n` elements from the
 *   generator's integer output. See \ref groupBrng for the definition of each
 *   generator's integer output.
 * \return See \ref groupError.
 */
int viRngUniformBits(const openrng_int_t method, VSLStreamStatePtr stream,
                     const openrng_int_t n, unsigned int buffer[]);

/**
 * \ingroup groupDiscrete
 *
 * Generates uniformly distributed bits in 32-bit chunks. This function is
 * designed to ensure that each bit in the 32-bit chunk is uniformly
 * distributed, unlike \ref viRngUniformBits.
 *
 * \param[in] method Method used for transforming from a uniform sequence.
 * Valid value for `method` is:
 *   - \ref VSL_RNG_METHOD_UNIFORMBITS32_STD.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[in, out] buffer On input, a buffer of length greater or equal to `n`.
 * On output, a buffer of 32-bit random integer numbers with uniformly
 * distributed bits.
 * \return See \ref groupError.
 */
int viRngUniformBits32(const openrng_int_t method, VSLStreamStatePtr stream,
                       const openrng_int_t n, unsigned int buffer[]);

/**
 * \ingroup groupDiscrete
 *
 * Generates uniformly distributed bits in 64-bit chunks. This function is
 * designed to ensure that each bit in the 64-bit chunk is uniformly
 * distributed, unlike the viRngUniformBits.
 *
 * \param[in] method Method used for transforming from a uniform sequence.
 * Valid value for `method` is:
 *   - \ref VSL_RNG_METHOD_UNIFORMBITS64_STD.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] buffer On input, a buffer of length greater or equal to `n`.
 * On output, a buffer of 64-bit random integer numbers with uniformly
 * distributed bits.
 * \return See \ref groupError.
 */
int viRngUniformBits64(const openrng_int_t method, VSLStreamStatePtr stream,
                       const openrng_int_t n, openrng_uint64_t buffer[]);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the beta
 * distribution.
 *
 * \warning Not implemented in \armplref.
 *
 * The parameters for this distribution are the shape parameters for the beta
 * distribution \f$p\f$ and \f$q\f$, a displacement quantity \f$\alpha\f$, and a
 * scale factor \f$\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; p, q, \alpha, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{1}{B(p, q)\beta^{p+q-1}}(x-\alpha)^{p-1}
 *      (\beta+\alpha-x)^q-1, & \alpha < x < \alpha + \beta \\
 *      0, & x < \alpha, x > \alpha + \beta,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$, where \f$B(p, q)\f$ is the complete
 * Beta function. Parameters \f$p, q, \alpha, \beta \in \mathbb{R}\f$,
 * with \f$p >0, q>0\f$, and \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Beta distribution.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Beta distribution.
 * \param[in] p Shape parameter for the Beta function.
 * Constraint: `p` > 0.
 * \param[in] q Shape parameter for the Beta function.
 * Constraint: `q` > 0.
 * \param[in] alpha Displacement quantity.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vsRngBeta(const openrng_int_t method, VSLStreamStatePtr stream,
              const openrng_int_t n, float r[], const float p, const float q,
              const float alpha, const float beta);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the Cauchy
 * distribution. The parameters for this distribution are a displacement
 * quantity \f$a\f$ and a scale factor \f$\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \beta) = \displaystyle \frac{1}{\displaystyle
 *      \pi\beta\left(1 + \left(\frac{x-a}{\beta}\right)^2\right)}
 * \f]
 *
 * for \f$-\infty < x < \infty\f$. Parameters \f$\alpha, \beta
 * \in\mathbb{R}\f$, with \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Cauchy distribution. Valid value for `method` is:
 *   - \ref VSL_RNG_METHOD_CAUCHY_ICDF.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Cauchy distribution.
 * \param[in] a Displacement quantity.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vsRngCauchy(const openrng_int_t method, VSLStreamStatePtr stream,
                const openrng_int_t n, float r[], const float a,
                const float beta);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the
 * chi-squared distribution.
 *
 * \warning Not implemented in \armplref.
 *
 * The parameter for this distribution is the number of degrees of freedom,
 * \f$v\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; v) = \frac{x^{(v-2)/2} \exp(-x/2)}{2^{v/2} \Gamma(v/2)}
 * \f]
 *
 * for \f$x\in[0, \infty)\f$. Parameter \f$v\in\mathbb{N}\f$, with \f$v >
 * 0\f$.
 *
 * \param[in] method Method used for transforming from a uniform
 * distribution to a chi-squared distribution.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a chi-squared distribution.
 * \param[in] v Numbers of degrees of freedom.
 * Constraint: `v` > 0.
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vsRngChiSquare(const openrng_int_t method, VSLStreamStatePtr stream,
                   const openrng_int_t n, float r[], const int v);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the
 * exponential distribution. The parameters for this distribution are a
 * displacement quantity \f$a\f$, and a scale factor \f$\beta>0\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{1}{\beta} \exp\left(
 *      \displaystyle -\frac{x-a}{\beta}\right), & x \geq a \\
 *      0, & x < a,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$. Parameters \f$\alpha, \beta \in
 * \mathbb{R}\f$, with \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to an exponential distribution. Valid values for`method`
 * are:
 *   - \ref VSL_RNG_METHOD_EXPONENTIAL_ICDF.
 *   - \ref VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of an exponential distribution.
 * \param[in] a Displacement factor.
 * \param[in] beta Scaling factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vsRngExponential(const openrng_int_t method, VSLStreamStatePtr stream,
                     const openrng_int_t n, float r[], const float a,
                     const float beta);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the Gamma
 * distribution.
 *
 * \warning Not implemented in \armplref.
 *
 * The parameters of this distribution are: its shape parameter \f$\alpha\f$, a
 * displacement \f$a\f$, and, a scale factor \f$\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; \alpha, a, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{1}{\Gamma(\alpha)\beta^{\alpha}} (x-a)^{\alpha-1}
 *      \exp\left( \displaystyle -\frac{(x-a)}{\beta}\right), & x \geq a \\
 *      0, & x < a,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$, where \f$\alpha, \beta, a \in
 * \mathbb{R}\f$ and \f$\alpha>0\f$ and \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Gamma distribution.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Gamma distribution.
 * \param[in] a Displacement quantity.
 * \param[in] alpha Shape parameter for the Gamma function.
 * Constraint: `alpha` > 0.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vsRngGamma(const openrng_int_t method, VSLStreamStatePtr stream,
               const openrng_int_t n, float r[], const float a,
               const float alpha, const float beta);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the
 * Gaussian distribution. The parameters of this distribution are: its mean
 * value \f$a\f$ and its standard deviation \f$\sigma\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \sigma) = \displaystyle \frac{1}{\sigma\sqrt{2\pi}}\exp\left(
 *      \displaystyle -\frac{(x-a)^2}{2\sigma^2}\right),
 * \f]
 *
 * for \f$-\infty < x < \infty\f$, where \f$a, \sigma \in \mathbb{R}\f$,
 * with \f$\sigma > 0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Gaussian distribution. Valid values for `method`
 * are:
 *   - \ref VSL_RNG_METHOD_GAUSSIAN_ICDF.
 *   - \ref VSL_RNG_METHOD_GAUSSIAN_BOXMULLER.
 *   - \ref VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Gaussian distribution.
 * \param[in] a Mean value.
 * \param[in] sigma Standard deviation.
 * Constraint: `sigma` > 0.
 * \return See \ref groupError.
 */
int vsRngGaussian(const openrng_int_t method, VSLStreamStatePtr stream,
                  const openrng_int_t n, float r[], const float a,
                  const float sigma);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the
 * multivariate Gaussian distribution of dimension \f$d\f$.
 *
 * \warning Not implemented in \armplref.
 *
 * The parameters of this distribution are: a vector \f$a\f$ of length \f$d\f$
 * of mean values, a variance-covariance matrix \f$C\f$, of dimension
 * \f$d\times d\f$. The matrix \f$C\f$ is symmetric and positive-definite.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, C) = \displaystyle \frac{1}{\sqrt{\text{det}(2\pi C)}}
 *      \exp\left(\displaystyle -\frac{1}{2(x-a)^T C^{-1} (x-a)}\right),
 * \f]
 *
 * with \f$x\in\mathbb{R}^d\f$, and \f$-\infty < x_i < \infty\f$, for each
 * \f$i\in\{1, \ldots, d\}\f$. In each dimension \f$i\f$, with \f$i <
 * d\f$, \f$a_i\f$ is the mean value of the Gaussian distribution in
 * that dimension; therefore \f$a \in\mathbb{R}^d\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a multivariate Gaussian distribution.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a multivariate Gaussian
 * distribution.
 * \param[in] dimen Dimension of the multivariate distribution.
 * Constraint: `d` > 0.
 * \param[in] mstorage Storage scheme for lower triangular matrices.
 * \param[in] a Vector of mean values.
 * \param[in] t Elements of the lower triangular part of the
 *   variance-covariance matrix. Elements are store according to the
 *   storage scheme in `mstorage`.
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vsRngGaussianMV(const openrng_int_t method, VSLStreamStatePtr stream,
                    const openrng_int_t n, float r[], const openrng_int_t dimen,
                    const openrng_int_t mstorage, const float *a,
                    const float *t);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the Gumbel
 * distribution. The parameters of this distribution are: a displacement
 * quantity \f$a\f$ and a scale factor \f$\beta\f$
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \beta) = \displaystyle \frac{1}{\sqrt{\beta}} \exp\left(
 *      \displaystyle \frac{x-a}{\beta} - \text{exp}\left(
 *      \displaystyle \frac{x-a}{\beta}\right)\right),
 * \f]
 *
 * for \f$-\infty < x < \infty\f$, where \f$a \in \mathbb{R}\f$, and
 * \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Gumbel distribution. Valid value for `method` is:
 *   - \ref VSL_RNG_METHOD_GUMBEL_ICDF.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Gumbel distribution.
 * \param[in] a Displacement quantity.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vsRngGumbel(const openrng_int_t method, VSLStreamStatePtr stream,
                const openrng_int_t n, float r[], const float a,
                const float beta);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the
 * Laplace distribution. The parameters of this distribution are: its mean
 * \f$a\f$ and a scale factor \f$\beta\f$. The standard deviation \f$\sigma\f$
 * of the distribution can be calculated from \f$\beta\f$ by the relation
 * \f$\sigma=\sqrt{2}\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \beta) = \displaystyle \frac{1}{\sqrt{2\beta}} \exp\left(
 *      \displaystyle -\frac{|x-a|}{\beta}\right),
 * \f]
 *
 * for \f$-\infty < x < \infty\f$. The parameters \f$a\f$ and
 * \f$\beta\f$ are such that \f$a, \beta \in\mathbb{R}\f$ and
 * \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Laplace distribution. Valid value for `method` is:
 *   - \ref VSL_RNG_METHOD_LAPLACE_ICDF.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Laplace distribution.
 * \param[in] a Mean of the distribution.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vsRngLaplace(const openrng_int_t method, VSLStreamStatePtr stream,
                 const openrng_int_t n, float r[], const float a,
                 const float beta);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the
 * lognormal distribution.
 *
 * The parameters of this distribution are: the mean \f$a\f$ and standard
 * deviation \f$\sigma\f$ of the normal distribution obtained from the logarithm
 * of the lognormal distribution, a displacement value \f$b\f$ and, scale factor
 * \f$\beta\f$.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \sigma, b, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{1}{\sqrt{2\pi}\sigma(x-b)} \exp\left( \displaystyle
 *                    -\frac{\left[\ln\left(
 *      \displaystyle \frac{x-b}{\beta}\right)-a\right]^2 }{2\sigma^2}\right),
 *      & x > b \\
 *      0, & x \leq b,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$. Above, \f$\alpha, \sigma, b,
 * \beta \in \mathbb{R}\f$, with \f$\sigma>0\f$ and \f$\beta>0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a lognormal distribution.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a lognormal distribution.
 * \param[in] alpha Mean of the underlying normal distribution.
 * \param[in] sigma Standard deviation of the underlying normal distribution.
 * Constraint: `sigma` > 0.
 * \param[in] b Displacement value for the lognormal distribution.
 * \param[in] beta Scale factor for the lognormal distribution.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vsRngLognormal(const openrng_int_t method, VSLStreamStatePtr stream,
                   const openrng_int_t n, float r[], const float alpha,
                   const float sigma, const float b, const float beta);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the
 * Rayleigh distribution.
 *
 * The probability density function is given by:
 *
 * \f[
 *      f(x; a, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{2(x-a)}{\beta^2} \exp\left( \displaystyle
 *                    -\frac{(x-a)^2}{\beta^2}\right), & x \geq a \\
 *      0, & x < a,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$. The parameters \f$a,
 * \beta\in\mathbb{R}\f$, with \f$\beta>0\f$. The Rayleigh distribution
 * is a special case of the Weibull distribution with shape parameter
 * \f$\alpha=2\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Rayleigh distribution. Valid values for `method`
 * are:
 *   - \ref VSL_RNG_METHOD_RAYLEIGH_ICDF.
 *   - \ref VSL_RNG_METHOD_RAYLEIGH_ICDF_ACCURATE.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Rayleigh distribution.
 * \param[in] a Displacement factor.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vsRngRayleigh(const openrng_int_t method, VSLStreamStatePtr stream,
                  const openrng_int_t n, float r[], const float a,
                  const float beta);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the
 * uniform distribution in the interval \f$[a, b)\f$. The lower and upper bound
 * of the domain have to satisfy \f$a < b\f$.
 *
 * The probability distribution is given by:
 *
 * \f[
 *      f(x; a, b) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{1}{b - a}, & a \leq x < b, \\
 *      0, & \text{otherwise},
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to another, possibly rescaled and translated, uniform
 * distribution.
 * Valid values for `method` are:
 *   - \ref VSL_RNG_METHOD_UNIFORM_STD.
 *   - \ref VSL_RNG_METHOD_UNIFORM_STD_ACCURATE.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a uniform distribution.
 * \param[in] a Lower bound of the interval.
 * Constraint: `a < b`.
 * \param[in] b Upper bound of the interval.
 * Constraint: `b > a`.
 * \return See \ref groupError.
 */
int vsRngUniform(const openrng_int_t method, VSLStreamStatePtr stream,
                 const openrng_int_t n, float r[], const float a,
                 const float b);

/**
 * \ingroup groupContinuous
 *
 * Generates single-precision random numbers distributed according to the
 * Weibull distribution. The parameters for this distribution are: a shape
 * parameter \f$\alpha\f$, a displacement quantity \f$a\f$, and, a scale
 * quantity \f$\beta\f$.
 *
 * The probability distribution is given by:
 *
 * \f[
 *      f(x; \alpha, a, \beta) = \left\{\begin{array}{ll}
 *      \displaystyle \frac{\alpha}{\beta^{\alpha}} (x-a)^{\alpha-1}
 *      \exp\left(-\left(\displaystyle \frac{x-a}{\beta}\right)^{\alpha}\right),
 *      & x \geq a \\
 *      0, & x < a,
 *      \end{array}\right.
 * \f]
 *
 * for \f$-\infty < x < \infty\f$, where \f$a, \alpha, \beta
 * \in\mathbb{R}\f$, with \f$\alpha > 0\f$, and \f$\beta >0\f$.
 * \param[in] method Method used for transforming from a uniform
 * distribution to a Weibull distribution. Valid values for `method`
 * are:
 *   - \ref VSL_RNG_METHOD_WEIBULL_ICDF.
 *   - \ref VSL_RNG_METHOD_WEIBULL_ICDF_ACCURATE.
 * \param[in] stream Pointer to an initialized stream.
 * \param[in] n Number of output elements.
 * \param[out] r Pre-allocated buffer, of at least size `n`. On
 * return, this contains `n` elements of a Weibull distribution.
 * \param[in] alpha Shape parameter.
 * Constraint: `alpha` > 0.
 * \param[in] a Displacement quantity.
 * \param[in] beta Scale factor.
 * Constraint: `beta` > 0.
 * \return See \ref groupError.
 */
int vsRngWeibull(const openrng_int_t method, VSLStreamStatePtr stream,
                 const openrng_int_t n, float r[], const float alpha,
                 const float a, const float beta);

/**
 * \ingroup groupService
 *
 * Creates a stream using the same BRNG and state as `srcstream`. On output,
 * `newStream` will point to a newly allocated and initialized stream. See \ref
 * vslCopyStreamState if you want to copy the state into a stream that has
 * previously been created with the same BRNG. Any resources can be later
 * released with \ref vslDeleteStream.
 *
 * \param[out] newstream On output, pointer to a newly created stream.
 * \param[in] srcstream An initialized stream.
 * \return See \ref groupError.
 */
int vslCopyStream(VSLStreamStatePtr *newstream,
                  const VSLStreamStatePtr srcstream);

/**
 * \ingroup groupService
 *
 * Copies the state of `srcstream` into the state of `dststream`. `dststream`
 * should have been previously created with the same BRNG as `srcstream`. See
 * \ref vslCopyStream if you also want to create a new stream.
 *
 * \param[in, out] dststream On input, a pre-existing stream with the same BRNG
 * as `srcstream`. On output, a copy of `srcstream`.
 * \param[in] srcstream An initialized stream.
 * \return See \ref groupError.
 */
int vslCopyStreamState(VSLStreamStatePtr dststream,
                       const VSLStreamStatePtr srcstream);

/**
 * \ingroup groupService
 *
 * Deletes a stream and releases all associated resources.
 *
 * \param[in, out] stream On input, a valid stream. On output, a pointer to
 * `NULL`.
 * \return See \ref groupError.
 */
int vslDeleteStream(VSLStreamStatePtr *stream);

/**
 * \ingroup groupService
 *
 * Retrieves the properties of a given basic generator.
 *
 * Note that only some properties are currently implemented. See
 * \ref VSLBRngProperties for the available properties.
 *
 * \param[in] brngId See \ref groupBrng for the list of supported BRNGs.
 * \param[out] properties On output, a pointer to the BRNG's properties, see
 * \ref VSLBRngProperties.
 * \return See \ref groupError.
 */
int vslGetBrngProperties(const int brngId, VSLBRngProperties *const properties);

/**
 * \ingroup groupService
 *
 * \warning Not implemented in \armplref.
 *
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vslGetNumRegBrngs(void);

/**
 * \ingroup groupService
 *
 * This function is used to compute the minimum buffer size needed for storing
 * the state of a stream in a memory buffer, see \ref vslSaveStreamM.
 *
 * \param[in] stream An initialized random stream.
 * \return On success, the required size in bytes. On failure, a negative value
 * corresponding to an error code, see \ref groupError.
 */
int vslGetStreamSize(const VSLStreamStatePtr stream);

/**
 * \ingroup groupService
 *
 * Retrieve the index of the BRNG associated with a given stream.
 *
 * \param[in] stream An initialized random stream.
 * \return On success, the index of the stream's BRNG. On failure, a negative
 * value corresponding to an error code, see \ref groupError.
 */
int vslGetStreamStateBrng(const VSLStreamStatePtr stream);

/**
 * \ingroup groupService
 *
 * Initializes the leapfrog method on a stream. The leapfrog method allows for
 * the splitting of a random stream into multiple interleaving subsequences. It
 * works by specifying a non-unit stride, `nstreams`, into the sequence,
 * alongside an initial index, `k`.
 *
 * If the sequence returned by a stream without leapfrog is:
 *
 * \code x[0], x[1], x[2], ... \endcode
 *
 * The sequence returned by a stream after leapfrog is enabled, would be:
 *
 * \code x[k], x[k + nstreams], x[k + 2 * nstreams], ... \endcode
 *
 * Leapfrog is only supported by a subset of BRNGs. See the corresponding
 * subsequence table of each BRNG in \ref groupBrng.
 *
 * \param[in] stream An initialized random stream.
 * \param[in] k The index of the current computational node.
 * \param[in] nstreams The stride, or total number of streams.
 * \return See \ref groupError.
 */
int vslLeapfrogStream(VSLStreamStatePtr stream, const openrng_int_t k,
                      const openrng_int_t nstreams);

/**
 * \ingroup groupService
 *
 * Creates a `stream` with data from a file. The file should have been created
 * by the \ref vslSaveStreamF function from the same version of OpenRNG. Any
 * resources can be later released with \ref vslDeleteStream.
 *
 * \param[out] stream On output, a pointer to a random stream.
 * \param[in] fname File containing stream data.
 * \return See \ref groupError.
 */
int vslLoadStreamF(VSLStreamStatePtr *stream, const char *fname);

/**
 * \ingroup groupService
 *
 * Creates a `stream` with data from a memory buffer. The buffer should have
 * been filled using the \ref vslSaveStreamM function from the same version of
 * OpenRNG. Any resources can be later released with \ref vslDeleteStream.
 *
 * \param[out] stream On output, a pointer to a random stream.
 * \param[in] memptr Memory buffer containing stream data.
 * \return See \ref groupError.
 */
int vslLoadStreamM(VSLStreamStatePtr *stream, const char *memptr);

/**
 * \ingroup groupService
 *
 * Creates a stream of type `brng` with initial seed of `seed`. For the list of
 * BRNGs and the meaning of their seed, see \ref groupBrng. For more complex
 * initialization involving multiple seeds/parameters, see \ref vslNewStreamEx.
 * Any resources can be later released with \ref vslDeleteStream.
 *
 * \param[out] stream On output, an initialized random stream.
 * \param[in] brng See \ref groupBrng for the list of BRNGs supported.
 * \param[in] seed The meaning of seed depends on the BRNG being initialized.
 * \return See \ref groupError.
 */
int vslNewStream(VSLStreamStatePtr *stream, const openrng_int_t brng,
                 const openrng_uint_t seed);

/**
 * \ingroup groupService
 *
 * Creates a stream of type `brng` with an array of `n` parameters, `params`.
 * For the list of BRNGs and the meaning of `params`, see \ref groupBrng. In
 * general, `params` is used to store multiple seed values when more data is
 * required than is permitted by \ref vslNewStream. Any resources can be later
 * released with \ref vslDeleteStream.
 *
 * \param[out] stream On output, an initialized random stream.
 * \param[in] brng See \ref groupBrng for the list of BRNGs supported.
 * \param[in] n Number of elements in params.
 * \param[in] params The meaning of `params[n]` depends on the BRNG being
 * initialized.
 * \return See \ref groupError.
 */
int vslNewStreamEx(VSLStreamStatePtr *stream, const openrng_int_t brng,
                   const openrng_int_t n, const unsigned int params[]);

/**
 * \ingroup groupService
 *
 * \warning Not implemented in \armplref.
 *
 * \param[in] properties
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vslRegisterBrng(const VSLBRngProperties *properties);

/**
 * \ingroup groupService
 *
 * Saves `stream` and its corresponding state to the file path specified by
 * `fname`.
 *
 * \param[in] stream An initialized random stream.
 * \param[in] fname Path of output file.
 * \return See \ref groupError.
 */
int vslSaveStreamF(const VSLStreamStatePtr stream, const char *fname);

/**
 * \ingroup groupService
 *
 * Saves `stream` and its associated state to a memory buffer. The size of the
 * buffer must be at least the number of bytes reported by \ref
 * vslGetStreamSize.
 *
 * \param[in] stream An initialized random stream.
 * \param[in, out] memptr On input, a buffer of at least the number of bytes
 * reported by \ref vslGetStreamSize. On output, an opaque buffer containing the
 * stream's state.
 * \return See \ref groupError.
 */
int vslSaveStreamM(const VSLStreamStatePtr stream, char *memptr);

/**
 * \ingroup groupService
 *
 * Skips `nskip` elements of the provided stream. This can be used to form
 * non-overlapping sequences per stream, by passing in a value of `nskip` larger
 * than the amount of elements required by any other stream, provided the period
 * of the generator has not been exceeded. By extension, this can be used to
 * guarantee non-overlapping sequences per thread, if each thread skips ahead a
 * unique amount. Skip ahead is only supported by a subset of BRNGs, see the
 * subsequence support tables in \ref groupBrng. If you need to skip ahead by
 * more than \f$2^{63}\f$ elements, see \ref vslSkipAheadStreamEx. See \ref
 * skipahead.c for an example use of this method.
 *
 * \param[in] stream An initialized stream.
 * \param[in] nskip The number of elements to skip.
 * \return See \ref groupError.
 */
int vslSkipAheadStream(VSLStreamStatePtr stream, const long long int nskip);

/**
 * \ingroup groupService
 *
 * Skips a stream by the number of elements specified by the `nskip` array. See
 * \ref vslSkipAheadStream for a description of how skip ahead works. This
 * function is useful for skipping ahead in a stream by more than \f$2^{63}\f$
 * elements.
 *
 * The nskip array is interpreted as follows:
 *
 * \code{.cpp} nskip[0] + nskip[1] * pow(2, 64) + nskip[2] * pow(2, 128) ...
 *   + nskip[i] * pow(2, 64 * i) + ... \endcode
 *
 * \param[in] stream An initialized random stream.
 * \param[in] n The size of the provided `nskip` array.
 * \param[in] nskip An array specifying the number of elements to skip.
 * \return See \ref groupError.
 */
int vslSkipAheadStreamEx(VSLStreamStatePtr stream, const openrng_int_t n,
                         const openrng_uint64_t nskip[]);

/**
 * \ingroup groupService
 *
 * \warning Not implemented in \armplref.
 *
 * \param[out] stream
 * \param[in] n
 * \param[in] dbuf
 * \param[in] a
 * \param[in] b
 * \param[in] dcallback
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vsldNewAbstractStream(VSLStreamStatePtr *stream, const openrng_int_t n,
                          const double dbuf[], const double a, const double b,
                          const dUpdateFuncPtr dcallback);

/**
 * \ingroup groupService
 *
 * \warning Not implemented in \armplref.
 *
 * \param[out] stream
 * \param[in] n
 * \param[in] ibuf
 * \param[in] a
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vsliNewAbstractStream(VSLStreamStatePtr *stream, const openrng_int_t n,
                          const unsigned int ibuf[], const iUpdateFuncPtr a);

/**
 * \ingroup groupService
 *
 * \warning Not implemented in \armplref.
 *
 * \param[out] stream
 * \param[in] n
 * \param[in] sbuf
 * \param[in] a
 * \param[in] b
 * \param[in] scallback
 * \return \ref VSL_ERROR_FEATURE_NOT_IMPLEMENTED
 */
int vslsNewAbstractStream(VSLStreamStatePtr *stream, const openrng_int_t n,
                          const float sbuf[], const float a, const float b,
                          const sUpdateFuncPtr scallback);

#ifdef __cplusplus
}
#endif

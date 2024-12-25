/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

/*
 * This file contains macros magic to choose the implementation of a function based on the
 * dimension's remainder. It is used to collapse large and repetitive switch statements that are
 * used to choose and define the templated values of the implementation of the distance functions.
 * We assume that we are dealing with 512-bit blocks, so we define a chunk size of 32 for 16-bit
 * elements, 16 for 32-bit elements, and a chunk size of 8 for 64-bit elements. The main macro is
 * CHOOSE_IMPLEMENTATION, and it's the one that should be used.
 */

// Macro for a single case. Sets __ret_dist_func to the function with the given remainder.
#define X(N, func)                                                                                 \
    case (N):                                                                                      \
        __ret_dist_func = func<(N)>;                                                               \
        break;

// Macros for folding cases of a switch statement, for easier readability.
// Each macro expands into a sequence of cases, from 0 to N-1, doubling the previous macro.
#define C2(X, func, N)  X(2 * (N), func) X(2 * (N) + 1, func)
#define C4(X, func, N)  C2(X, func, 2 * (N)) C2(X, func, 2 * (N) + 1)
#define C8(X, func, N)  C4(X, func, 2 * (N)) C4(X, func, 2 * (N) + 1)
#define C16(X, func, N) C8(X, func, 2 * (N)) C8(X, func, 2 * (N) + 1)
#define C32(X, func, N) C16(X, func, 2 * (N)) C16(X, func, 2 * (N) + 1)
#define C64(X, func, N) C32(X, func, 2 * (N)) C32(X, func, 2 * (N) + 1)

// Macros for 8, 16, 32 and 64 cases. Used to collapse the switch statement.
// Expands into 0-7, 0-15, 0-31 or 0-63 cases respectively.
#define CASES8(X, func)  C8(X, func, 0)
#define CASES16(X, func) C16(X, func, 0)
#define CASES32(X, func) C32(X, func, 0)
#define CASES64(X, func) C64(X, func, 0)

// Main macro. Expands into a switch statement that chooses the implementation based on the
// dimension's remainder.
// @params:
// out:     The output variable that will be set to the chosen implementation.
// dim:     The dimension.
// chunk:   The chunk size. Can be 64, 32, 16 or 8. 64 for 8-bit elements, 32 for 16-bit elements,
// 16 for 32-bit elements, 8 for 64-bit elements. func:    The templated function that we want to
// choose the implementation for.
#define CHOOSE_IMPLEMENTATION(out, dim, chunk, func)                                               \
    do {                                                                                           \
        decltype(out) __ret_dist_func;                                                             \
        switch ((dim) % (chunk)) { CASES##chunk(X, func) }                                         \
        out = __ret_dist_func;                                                                     \
    } while (0)

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
 * We assume that we are dealing with 512-bit blocks, so we define a chunk size of 16 for 32-bit
 * elements, a chunk size of 8 for 64-bit elements, and a chunk size of 32 for 16-bit elements.
 * The main macro is CHOOSE_IMPLEMENTATION, and it's the one that should be used.
 */

// Macro for a single case. Sets __ret_dist_func to the function with the given remainder.
#define X(N, func)                                                                                 \
    case (N):                                                                                      \
        __ret_dist_func = func<(N)>;                                                               \
        break;

// Macro for 4 cases. Used to collapse the switch statement. For a given N, expands to 4 X macros
// of 4N, 4N+1, 4N+2, 4N+3.
#define C4(X, func, N) X(4 * N, func) X(4 * N + 1, func) X(4 * N + 2, func) X(4 * N + 3, func)
// Macro for 8 cases. Used to collapse the switch statement. For a given N, expands to 8 X macros
// of 4N, 4N+1, 4N+2, 4N+3, 4N+4, 4N+5, 4N+6, 4N+7.
#define C8(X, func, N) C4(X, func, (N)) C4(X, func, (N + 4))

// Macros for 8, 16 and 32 cases. Used to collapse the switch statement.
// Expands into 0-7, 0-15 and 0-31 cases respectively.
#define CASES32(x, func) C8(X, func, 0) C8(X, func, 1) C8(X, func, 2) C8(X, func, 3)
#define CASES16(X, func) C4(X, func, 0) C4(X, func, 1) C4(X, func, 2) C4(X, func, 3)
#define CASES8(X, func)  C4(X, func, 0) C4(X, func, 1)

// Main macro. Expands into a switch statement that chooses the implementation based on the
// dimension's remainder.
// @params:
// out:     The output variable that will be set to the chosen implementation.
// dim:     The dimension.
// chunk:   The chunk size (defined to be 512 / <element size in bits>).
// func:    The templated function that we want to choose the implementation for.
#define CHOOSE_IMPLEMENTATION(out, dim, chunk, func)                                               \
    do {                                                                                           \
        decltype(out) __ret_dist_func;                                                             \
        switch (dim % chunk) { CASES##chunk(X, func) }                                             \
        out = __ret_dist_func;                                                                     \
    } while (0)

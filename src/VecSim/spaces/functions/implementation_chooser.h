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
#define C1(func, N)                                                                                \
    case (N):                                                                                      \
        __ret_dist_func = func<(N)>;                                                               \
        break;

// Macros for folding cases of a switch statement, for easier readability.
// Each macro expands into a sequence of cases, from 0 to N-1, doubling the previous macro.
#define C2(func, N)  C1(func, 2 * (N)) C1(func, 2 * (N) + 1)
#define C4(func, N)  C2(func, 2 * (N)) C2(func, 2 * (N) + 1)
#define C8(func, N)  C4(func, 2 * (N)) C4(func, 2 * (N) + 1)
#define C16(func, N) C8(func, 2 * (N)) C8(func, 2 * (N) + 1)
#define C32(func, N) C16(func, 2 * (N)) C16(func, 2 * (N) + 1)
#define C64(func, N) C32(func, 2 * (N)) C32(func, 2 * (N) + 1)

// Macros for 8, 16, 32 and 64 cases. Used to collapse the switch statement.
// Expands into 0-7, 0-15, 0-31 or 0-63 cases respectively.
#define CASES1(func)  C1(func, 1)
#define CASES4(func)  C4(func, 1)
#define CASES8(func)  C8(func, 0)
#define CASES16(func) C16(func, 0)
#define CASES32(func) C32(func, 0)
#define CASES64(func) C64(func, 0)

// Main macro. Expands into a switch statement that chooses the implementation based on the
// dimension's remainder.
// @params:
// out:     The output variable that will be set to the chosen implementation.
// dim:     The dimension.
// func:    The templated function that we want to choose the implementation for.
// chunk:   The chunk size. Can be 64, 32, 16 or 8. Should be the number of elements of the expected
//          type fitting in the expected register size.
//          chunk == 1 means that there's no use of the residual, and we can use the function
//          directly.
#define CHOOSE_IMPLEMENTATION(out, dim, chunk, func)                                               \
    do {                                                                                           \
        decltype(out) __ret_dist_func;                                                             \
        if ((chunk) == 1) {                                                                        \
            /* Handle the case where chunk is 0 */                                                 \
            __ret_dist_func = func<1>;                                                             \
        } else {                                                                                   \
            switch ((dim) % (chunk)) { CASES##chunk(func) }                                        \
        }                                                                                          \
        out = __ret_dist_func;                                                                     \
    } while (0)

#define CHOOSE_RUNTIME_IMPLEMENTATION(ret_dist_func, dim, reg_size, func_template) \
switch (reg_size) {                                             \
    case 4:                                                    \
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 4, func_template); \
        break;                                                  \
    case 8:                                                     \
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 8, func_template); \
        break;                                                  \
    case 16:                                                    \
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, func_template); \
        break;                                                  \
    case 32:                                                    \
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 32, func_template); \
        break;                                                  \
    case 64:                                                    \
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 64, func_template); \
        break;                                                  \
}


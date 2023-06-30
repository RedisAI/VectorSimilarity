/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#define C4(X, func, N)                                                                             \
    X((4 * N), func) X((4 * N + 1), func) X((4 * N + 2), func) X((4 * N + 3), func)
#define X(N, func)                                                                                 \
    case (N):                                                                                      \
        ret_dist_func = func<(1 << (N)) - 1>;                                                      \
        break;

#define CASES16(X, func) C4(X, func, 0) C4(X, func, 1) C4(X, func, 2) C4(X, func, 3)
#define CASES8(X, func)  C4(X, func, 0) C4(X, func, 1)

#define CHOOSE_IMPLEMENTATION(dim, chunk, func)                                                    \
    {                                                                                              \
        switch (dim % chunk) { CASES##chunk(X, func) }                                             \
    }                                                                                              \
    break

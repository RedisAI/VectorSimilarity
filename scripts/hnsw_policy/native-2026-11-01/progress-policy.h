/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once
#include <cstddef>
// Experimental continuation policy. Kept outside production until measured.
bool progressPolicy(size_t k, size_t n, size_t d, size_t m, float r, size_t remaining) {
    if (k < 13 || k > 500 || n < 35000 || n > 300000 || d < 76 || d > 1536 || m < 11 || m > 64)
        return true;
    if (remaining) k = remaining;
    if (r < .1f) return !(remaining && ((k <= 31 && r >= .07f) || k <= 31 * r));
    if (n <= 55000 && ((k > 350 && r <= .625f) || (k > 125 && r <= .375f))) return true;
    if (r > .375f) return false;
    if (r > .175f) return k > 350 && n <= 112500;
    if (k <= 31) return false;
    if (k <= 125) return n <= 112500;
    return n <= 112500 || m <= 24;
}

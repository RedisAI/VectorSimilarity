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

inline bool preferHnswAdhocMeasured(size_t k, size_t n, size_t dim, size_t m, float ratio,
                                    size_t k_remaining = 0) {
    // Keep the historical leaf outside the measured feature ranges.
    if (k < 13 || k > 500 || n < 35000 || n > 300000 || dim < 76 || dim > 1536 || m < 11 ||
        m > 64 || !(ratio > 0.0f && ratio <= 1.0f)) {
        return true;
    }
    // Continuing only needs the missing hits; restarting BF still scans the whole filter.
    if (k_remaining && k_remaining < k) {
        k = k_remaining;
    }
    // Avoid restarting for a tiny final batch; falling estimates increase its cost and restore BF.
    if (ratio < 0.07f) {
        return !(k_remaining && k <= 31 * ratio);
    }
    if (ratio < 0.1f && !(k_remaining && k <= 31)) {
        return true;
    }
    if (n <= 55000 && ((k > 350 && ratio <= 0.625f) || (k > 125 && ratio <= 0.375f))) {
        return true;
    }
    if (ratio > 0.375f) {
        return false;
    }
    if (ratio > 0.175f) {
        return k > 350 && n <= 112500;
    }
    if (k <= 31) {
        return false;
    }
    if (k <= 125) {
        return n <= 112500;
    }
    return n <= 112500 || m <= 24;
}

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

// Generated policy; training code and measurements are preserved at:
// https://github.com/RedisAI/VectorSimilarity/tree/omer-mod18264-policy-measurements/scripts/hnsw_policy
// Measurement SHA-256: e024433df41f370994c36fc0581740c984a52adc9c735370d1ed609018131686
// Called only from the affected leaf of HNSWIndex::preferAdHocSearch.
inline bool preferHnswAdhocMeasured(size_t k, size_t n, size_t dim, size_t m, float ratio) {
    // Keep the historical leaf outside the measured feature ranges.
    if (!(static_cast<float>(k) >= 13 && static_cast<float>(k) <= 500) ||
        !(static_cast<float>(n) >= 35000 && static_cast<float>(n) <= 300000) ||
        !(static_cast<float>(dim) >= 76 && static_cast<float>(dim) <= 1536) ||
        !(static_cast<float>(m) >= 11 && static_cast<float>(m) <= 64) ||
        !(static_cast<float>(ratio) >= 0.10000000149011612 && static_cast<float>(ratio) <= 1)) {
        return true;
    }
    if (static_cast<float>(ratio) <= 0.17500000074505806) {
        if (static_cast<float>(k) <= 125) {
            if (static_cast<float>(n) <= 55000) {
                if (static_cast<float>(k) <= 31.5) {
                    return false;
                } else {
                    return true;
                }
            } else {
                if (static_cast<float>(n) <= 112500) {
                    if (static_cast<float>(k) <= 31.5) {
                        return false;
                    } else {
                        return true;
                    }
                } else {
                    return false;
                }
            }
        } else {
            if (static_cast<float>(n) <= 112500) {
                return true;
            } else {
                if (static_cast<float>(m) <= 24) {
                    return true;
                } else {
                    return false;
                }
            }
        }
    } else {
        if (static_cast<float>(ratio) <= 0.375) {
            if (static_cast<float>(k) <= 350) {
                return false;
            } else {
                if (static_cast<float>(n) <= 112500) {
                    return true;
                } else {
                    return false;
                }
            }
        } else {
            return false;
        }
    }
}

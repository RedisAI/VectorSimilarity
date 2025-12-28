/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include "VecSim/spaces/spaces.h"

namespace spaces {
dist_func_t<float> IP_SQ8_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                      const void *arch_opt = nullptr);

dist_func_t<float> IP_FP32_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                       const void *arch_opt = nullptr);
dist_func_t<double> IP_FP64_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                        const void *arch_opt = nullptr);
dist_func_t<float> IP_BF16_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                       const void *arch_opt = nullptr);
dist_func_t<float> IP_FP16_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                       const void *arch_opt = nullptr);
dist_func_t<float> IP_INT8_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                       const void *arch_opt = nullptr);
dist_func_t<float> Cosine_INT8_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                           const void *arch_opt = nullptr);
dist_func_t<float> IP_UINT8_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                        const void *arch_opt = nullptr);
dist_func_t<float> Cosine_UINT8_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                            const void *arch_opt = nullptr);
dist_func_t<float> Cosine_SQ8_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                          const void *arch_opt = nullptr);
// SQ8-to-SQ8 distance functions (both vectors are uint8 quantized)
dist_func_t<float> IP_SQ8_SQ8_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                          const void *arch_opt = nullptr);
dist_func_t<float> Cosine_SQ8_SQ8_GetDistFunc(size_t dim, unsigned char *alignment = nullptr,
                                              const void *arch_opt = nullptr);
} // namespace spaces

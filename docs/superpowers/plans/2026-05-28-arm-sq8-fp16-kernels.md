# SQ8↔FP16 ARM SIMD Distance Kernels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQ8↔FP16 asymmetric distance kernels (IP, L2, Cosine) for ARM ISA tiers — NEON_HP, SVE, SVE2 — plugged into the existing dispatcher. Mirrors the x86 work delivered in PR #970.

**Architecture:** Header-only SIMD kernel templates (one per metric × ISA), instantiated via the existing `CHOOSE_IMPLEMENTATION` / `CHOOSE_SVE_IMPLEMENTATION` macros inside ISA-specific TUs (`NEON_HP.cpp`, `SVE.cpp`, `SVE2.cpp`). Wiring lives in `IP_space.cpp` and `L2_space.cpp` under a `#ifdef CPU_FEATURES_ARCH_AARCH64` block that parallels the existing x86 block. L2 reuses the IP `_IMP` template via the algebraic identity `L2² = x_sum_sq + y_sum_sq − 2·IP`. Scalar fallback already on `main` is unchanged and stays as the reference for every tier.

**Tech Stack:** C++20, ARM NEON intrinsics (`arm_neon.h`), ARM SVE/SVE2 intrinsics (`arm_sve.h`), GoogleTest, Google Benchmark, cpu_features.

**Branch:** `dor-forer-sq8-fp16-arm-kernels-mod-14972` (stacked on PR #970 / `dor-forer-sq8-fp16-x86-kernels-mod-14954`).

**Build / test loop:** The user runs `make build` (per project memory). After each build cycle confirmed, the assistant runs `make unit_test` / ASan / benchmarks on the appropriate host (ARM hardware or cross-compile/qemu — coordinate with user). Each task ends in a commit; commits are pushed only when explicitly requested.

**Spec:** [`docs/superpowers/specs/2026-05-28-arm-sq8-fp16-design.md`](../specs/2026-05-28-arm-sq8-fp16-design.md)

---

## File Structure

### Files created

| Path | Responsibility |
|------|----------------|
| `src/VecSim/spaces/IP/IP_NEON_SQ8_FP16.h` | NEON IP kernel template (`SQ8_FP16_InnerProductSIMD16_NEON_HP_IMP` + thin wrappers) |
| `src/VecSim/spaces/L2/L2_NEON_SQ8_FP16.h` | NEON L2 kernel template (calls NEON IP impl, applies L2 identity) |
| `src/VecSim/spaces/IP/IP_SVE_SQ8_FP16.h` | SVE IP kernel template (`SQ8_FP16_InnerProductSIMD_SVE_IMP` + wrappers); also `#include`d from SVE2.cpp |
| `src/VecSim/spaces/L2/L2_SVE_SQ8_FP16.h` | SVE L2 kernel template; also `#include`d from SVE2.cpp |

### Files modified

| Path | Change |
|------|--------|
| `src/VecSim/spaces/functions/NEON_HP.h` | +3 chooser declarations (IP, L2, Cosine) |
| `src/VecSim/spaces/functions/NEON_HP.cpp` | +#include kernel headers; +3 chooser definitions |
| `src/VecSim/spaces/functions/SVE.h` | +3 chooser declarations |
| `src/VecSim/spaces/functions/SVE.cpp` | +#include kernel headers; +3 chooser definitions |
| `src/VecSim/spaces/functions/SVE2.h` | +3 chooser declarations |
| `src/VecSim/spaces/functions/SVE2.cpp` | +#include SVE kernel headers; +3 chooser definitions (own symbols, templates instantiated under SVE2 compile flags) |
| `src/VecSim/spaces/IP_space.cpp` | +#ifdef AArch64 block in `IP_SQ8_FP16_GetDistFunc` and `Cosine_SQ8_FP16_GetDistFunc` (2 dispatcher blocks) |
| `src/VecSim/spaces/L2_space.cpp` | +#ifdef AArch64 block in `L2_SQ8_FP16_GetDistFunc` (1 dispatcher block) |
| `tests/unit/test_spaces.cpp` | retarget `GetDistFuncSQ8FP16Asymmetric` to dim=15; add dim=0 test; extend the three `SQ8_FP16_SpacesOptimizationTest` test bodies with ARM tier walks; extend `SQ8_FP16_SIMD_TierCoverage.ReportTiersExercised` with AArch64 tier reporting |
| `tests/benchmark/spaces_benchmarks/bm_spaces_sq8_fp16.cpp` | +AArch64 `cpu_features` block; +ARM ISA benchmark registrations |

### Files NOT modified

`src/VecSim/spaces/CMakeLists.txt` — zero CMake changes. Existing TU flags (`-march=armv8.2-a+fp16fml` for NEON_HP, `-march=armv8-a+sve` for SVE, `-march=armv9-a+sve2` for SVE2) already carry everything the new kernels need.

---

## Task 1: Retarget the scalar-fallback dispatcher test

**Why first:** Builds and runs on x86 today, has nothing to do with the ARM kernels, and tightens the contract the rest of the plan relies on (the dispatcher returns scalar for `dim < 16`).

**Files:**
- Modify: `tests/unit/test_spaces.cpp` — locate test named `GetDistFuncSQ8FP16Asymmetric` (added by PR #970; currently asserts `dim=128` returns the scalar fallback)

- [ ] **Step 1: Locate the existing test**

Run:
```bash
grep -n 'GetDistFuncSQ8FP16Asymmetric' tests/unit/test_spaces.cpp
```
Expected: one or more line hits pointing at the `TEST(...)` block.

- [ ] **Step 2: Modify the test to cover dim=0 and dim=15 instead of dim=128**

Replace the body of the existing `TEST(..., GetDistFuncSQ8FP16Asymmetric)` so it walks two below-threshold dims and asserts the scalar fallback for each of L2 / IP / Cosine. Drop in this exact body (rename the test fixture symbol to match what is already there if it differs):

```cpp
TEST_F(SpacesTest, GetDistFuncSQ8FP16Asymmetric) {
    // SQ8 storage with FP16 query (asymmetric) - should return SQ8_FP16 functions.
    // Per-ISA dispatcher walk coverage lives in the SQ8_FP16 SpacesOptimizationTest below.
    //
    // Walk two below-threshold dims (0 and 15) so the assertions hold regardless of which
    // SIMD tiers the host advertises: dim < 16 must always short-circuit to scalar fallback.
    // The template-mapping form (spaces::GetDistFunc<sq8, float, float16>) and the direct
    // *_SQ8_FP16_GetDistFunc form must agree for every dim, and both must match the scalar
    // reference at sub-threshold dims.
    for (size_t dim : {static_cast<size_t>(0), static_cast<size_t>(15)}) {
        auto l2_func = spaces::GetDistFunc<sq8, float, float16>(VecSimMetric_L2, dim, nullptr);
        auto ip_func = spaces::GetDistFunc<sq8, float, float16>(VecSimMetric_IP, dim, nullptr);
        auto cosine_func =
            spaces::GetDistFunc<sq8, float, float16>(VecSimMetric_Cosine, dim, nullptr);

        ASSERT_EQ(l2_func, L2_SQ8_FP16_GetDistFunc(dim, nullptr))
            << "Template mapping disagrees with direct dispatcher for L2 at dim=" << dim;
        ASSERT_EQ(ip_func, IP_SQ8_FP16_GetDistFunc(dim, nullptr))
            << "Template mapping disagrees with direct dispatcher for IP at dim=" << dim;
        ASSERT_EQ(cosine_func, Cosine_SQ8_FP16_GetDistFunc(dim, nullptr))
            << "Template mapping disagrees with direct dispatcher for Cosine at dim=" << dim;

        ASSERT_EQ(l2_func, SQ8_FP16_L2Sqr)
            << "dim=" << dim << " must short-circuit to scalar L2 fallback";
        ASSERT_EQ(ip_func, SQ8_FP16_InnerProduct)
            << "dim=" << dim << " must short-circuit to scalar IP fallback";
        ASSERT_EQ(cosine_func, SQ8_FP16_Cosine)
            << "dim=" << dim << " must short-circuit to scalar Cosine fallback";
    }
}
```

- [ ] **Step 3: User builds**

Ask the user to run `make build` (their normal x86 build is sufficient — this test is host-agnostic).

- [ ] **Step 4: Run the test**

Run:
```bash
./bin/<host-triple>/unit_tests --gtest_filter='SpacesTest.GetDistFuncSQ8FP16Asymmetric'
```
(Use `find bin -name unit_tests -type f` if the host-triple subdir is unknown.)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_spaces.cpp
git commit -m "Retarget SQ8↔FP16 scalar-fallback dispatcher test to dim=0/15 [MOD-14972]"
```

---

## Task 2: NEON IP kernel header

**Files:**
- Create: `src/VecSim/spaces/IP/IP_NEON_SQ8_FP16.h`

- [ ] **Step 1: Author the kernel file**

Create exactly this file (modeled on `IP_NEON_SQ8_FP32.h` + the NEON FP16 widening pattern from `IP_NEON_FP16.h`):

```cpp
/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include "VecSim/types/sq8.h"
#include "VecSim/types/float16.h"
#include <arm_neon.h>
#include <cassert>

using sq8 = vecsim_types::sq8;
using float16 = vecsim_types::float16;

/*
 * Optimised asymmetric SQ8<->FP16 inner product using the algebraic identity:
 *
 *   IP(x, y) = sum(x_i * y_i)
 *            ~= sum((min + delta * q_i) * y_i)
 *            = min * y_sum + delta * sum(q_i * y_i)
 *
 * The hot loop only accumulates sum(q_i * y_i) - no per-element dequantisation.
 * FP16 query lanes are widened to FP32 via vcvt_f32_f16 per 16-lane chunk.
 */

// Helper: 16 lanes per call, four FP32 accumulators (one per quarter).
static inline void
SQ8_FP16_InnerProductStep_NEON_HP(const uint8_t *&pVect1, const float16 *&pVect2,
                                  float32x4_t &sum0, float32x4_t &sum1,
                                  float32x4_t &sum2, float32x4_t &sum3) {
    // SQ8 storage: 16 * uint8 -> 4 * float32x4_t
    uint8x16_t v1_u8 = vld1q_u8(pVect1);
    uint16x8_t v1_lo = vmovl_u8(vget_low_u8(v1_u8));
    uint16x8_t v1_hi = vmovl_u8(vget_high_u8(v1_u8));
    float32x4_t v1_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v1_lo)));
    float32x4_t v1_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v1_lo)));
    float32x4_t v1_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v1_hi)));
    float32x4_t v1_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v1_hi)));

    // FP16 query: 16 * f16 -> 4 * float32x4_t via vcvt_f32_f16
    const float16_t *q = reinterpret_cast<const float16_t *>(pVect2);
    float16x8_t q_lo = vld1q_f16(q);
    float16x8_t q_hi = vld1q_f16(q + 8);
    float32x4_t v2_0 = vcvt_f32_f16(vget_low_f16(q_lo));
    float32x4_t v2_1 = vcvt_f32_f16(vget_high_f16(q_lo));
    float32x4_t v2_2 = vcvt_f32_f16(vget_low_f16(q_hi));
    float32x4_t v2_3 = vcvt_f32_f16(vget_high_f16(q_hi));

    sum0 = vfmaq_f32(sum0, v1_0, v2_0);
    sum1 = vfmaq_f32(sum1, v1_1, v2_1);
    sum2 = vfmaq_f32(sum2, v1_2, v2_2);
    sum3 = vfmaq_f32(sum3, v1_3, v2_3);

    pVect1 += 16;
    pVect2 += 16;
}

// pVect1v = SQ8 storage, pVect2v = FP16 query
template <unsigned char residual> // 0..15
float SQ8_FP16_InnerProductSIMD16_NEON_HP_IMP(const void *pVect1v, const void *pVect2v,
                                              size_t dimension) {
    assert(dimension >= 16 && "kernel precondition: dispatcher must guard dim >= 16");

    const uint8_t *pVect1 = static_cast<const uint8_t *>(pVect1v); // SQ8 storage
    const float16 *pVect2 = static_cast<const float16 *>(pVect2v); // FP16 query

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    const size_t num_of_chunks = dimension / 16;
    for (size_t i = 0; i < num_of_chunks; i++) {
        SQ8_FP16_InnerProductStep_NEON_HP(pVect1, pVect2, sum0, sum1, sum2, sum3);
    }

    // Residual handling: dim % 16 lanes.
    // residual >= 8: one safe 8-lane SQ8 + 8-lane FP16 load (FP16 trailer is wide enough).
    // residual <  8: scalar-only - a 4-lane FP16 load would overread y_sum metadata.
    constexpr unsigned char r = residual;
    if constexpr (r >= 8) {
        uint8x8_t v1_u8 = vld1_u8(pVect1);
        uint16x8_t v1_u16 = vmovl_u8(v1_u8);
        float32x4_t v1_a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v1_u16)));
        float32x4_t v1_b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v1_u16)));
        float16x8_t q_h = vld1q_f16(reinterpret_cast<const float16_t *>(pVect2));
        float32x4_t v2_a = vcvt_f32_f16(vget_low_f16(q_h));
        float32x4_t v2_b = vcvt_f32_f16(vget_high_f16(q_h));
        sum0 = vfmaq_f32(sum0, v1_a, v2_a);
        sum1 = vfmaq_f32(sum1, v1_b, v2_b);
        pVect1 += 8;
        pVect2 += 8;
    }
    // Lane-by-lane scalar for the final 0..7 (residual % 8) elements.
    constexpr unsigned char tail = r & 0x7;
    float scalar_dot = 0.0f;
    for (unsigned char k = 0; k < tail; ++k) {
        scalar_dot += static_cast<float>(pVect1[k]) * vecsim_types::FP16_to_FP32(pVect2[k]);
    }

    // Reduce the four NEON accumulators.
    float32x4_t sum_lo = vaddq_f32(sum0, sum1);
    float32x4_t sum_hi = vaddq_f32(sum2, sum3);
    float quantized_dot = vaddvq_f32(vaddq_f32(sum_lo, sum_hi)) + scalar_dot;

    // Metadata loads - use load_unaligned because odd dim leaves trailers unaligned.
    const uint8_t *params_bytes = static_cast<const uint8_t *>(pVect1v) + dimension;
    const float min_val =
        load_unaligned<float>(params_bytes + sq8::MIN_VAL * sizeof(float));
    const float delta =
        load_unaligned<float>(params_bytes + sq8::DELTA * sizeof(float));
    const uint8_t *query_meta_bytes =
        reinterpret_cast<const uint8_t *>(static_cast<const float16 *>(pVect2v) + dimension);
    const float y_sum =
        load_unaligned<float>(query_meta_bytes + sq8::SUM_QUERY * sizeof(float));

    return min_val * y_sum + delta * quantized_dot;
}

template <unsigned char residual>
float SQ8_FP16_InnerProductSIMD16_NEON_HP(const void *pVect1v, const void *pVect2v,
                                          size_t dimension) {
    return 1.0f -
           SQ8_FP16_InnerProductSIMD16_NEON_HP_IMP<residual>(pVect1v, pVect2v, dimension);
}

template <unsigned char residual>
float SQ8_FP16_CosineSIMD16_NEON_HP(const void *pVect1v, const void *pVect2v, size_t dimension) {
    // Cosine = 1 - IP (vectors are pre-normalised); reuses the IP wrapper.
    return SQ8_FP16_InnerProductSIMD16_NEON_HP<residual>(pVect1v, pVect2v, dimension);
}
```

- [ ] **Step 2: Header-only smoke (no build yet)**

Run:
```bash
grep -n 'load_unaligned\|FP16_to_FP32' src/VecSim/spaces/space_includes.h \
    src/VecSim/spaces/IP/IP.cpp src/VecSim/types/float16.h 2>/dev/null
```
Expected: confirm the global `load_unaligned<T>` is reachable through `space_includes.h` (matches the include path used by `IP_NEON_SQ8_FP32.h`) and `FP16_to_FP32` is reachable through `VecSim/types/float16.h`. If either include is missing, add it.

- [ ] **Step 3: Commit**

```bash
git add src/VecSim/spaces/IP/IP_NEON_SQ8_FP16.h
git commit -m "Add NEON_HP SQ8↔FP16 IP kernel header [MOD-14972]"
```

---

## Task 3: NEON L2 kernel header

**Files:**
- Create: `src/VecSim/spaces/L2/L2_NEON_SQ8_FP16.h`

- [ ] **Step 1: Author the kernel file**

```cpp
/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP/IP_NEON_SQ8_FP16.h"
#include "VecSim/types/sq8.h"
#include "VecSim/types/float16.h"

using sq8 = vecsim_types::sq8;
using float16 = vecsim_types::float16;

/*
 * Optimised asymmetric SQ8<->FP16 L2 squared distance using the algebraic identity:
 *
 *   ||x - y||^2 = sum(x_i^2) - 2 * IP(x, y) + sum(y_i^2)
 *               = x_sum_squares - 2 * IP(x, y) + y_sum_squares
 *
 * IP is computed by SQ8_FP16_InnerProductSIMD16_NEON_HP_IMP; metadata is FP32.
 */

template <unsigned char residual> // 0..15
float SQ8_FP16_L2SqrSIMD16_NEON_HP(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float ip =
        SQ8_FP16_InnerProductSIMD16_NEON_HP_IMP<residual>(pVect1v, pVect2v, dimension);

    const uint8_t *params_bytes = static_cast<const uint8_t *>(pVect1v) + dimension;
    const float x_sum_sq =
        load_unaligned<float>(params_bytes + sq8::SUM_SQUARES * sizeof(float));

    const uint8_t *query_meta_bytes = reinterpret_cast<const uint8_t *>(
        static_cast<const float16 *>(pVect2v) + dimension);
    const float y_sum_sq =
        load_unaligned<float>(query_meta_bytes + sq8::SUM_SQUARES_QUERY * sizeof(float));

    return x_sum_sq + y_sum_sq - 2.0f * ip;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/VecSim/spaces/L2/L2_NEON_SQ8_FP16.h
git commit -m "Add NEON_HP SQ8↔FP16 L2 kernel header [MOD-14972]"
```

---

## Task 4: NEON_HP dispatcher TU additions

**Files:**
- Modify: `src/VecSim/spaces/functions/NEON_HP.h` — add 3 declarations
- Modify: `src/VecSim/spaces/functions/NEON_HP.cpp` — add 3 chooser definitions

- [ ] **Step 1: Add chooser declarations to NEON_HP.h**

In `src/VecSim/spaces/functions/NEON_HP.h`, inside `namespace spaces { ... }`, append these three declarations alongside the existing `Choose_FP16_*_implementation_NEON_HP`:

```cpp
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_NEON_HP(size_t dim);
dist_func_t<float> Choose_SQ8_FP16_L2_implementation_NEON_HP(size_t dim);
dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_NEON_HP(size_t dim);
```

- [ ] **Step 2: Add chooser definitions to NEON_HP.cpp**

In `src/VecSim/spaces/functions/NEON_HP.cpp`, add the kernel `#include`s alongside the existing FP16 includes:

```cpp
#include "VecSim/spaces/IP/IP_NEON_SQ8_FP16.h"
#include "VecSim/spaces/L2/L2_NEON_SQ8_FP16.h"
```

Then inside `namespace spaces { ... }` (between `#include "implementation_chooser.h"` and `#include "implementation_chooser_cleanup.h"`), append:

```cpp
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_NEON_HP(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, SQ8_FP16_InnerProductSIMD16_NEON_HP);
    return ret_dist_func;
}

dist_func_t<float> Choose_SQ8_FP16_L2_implementation_NEON_HP(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, SQ8_FP16_L2SqrSIMD16_NEON_HP);
    return ret_dist_func;
}

dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_NEON_HP(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, SQ8_FP16_CosineSIMD16_NEON_HP);
    return ret_dist_func;
}
```

- [ ] **Step 3: Commit**

```bash
git add src/VecSim/spaces/functions/NEON_HP.h src/VecSim/spaces/functions/NEON_HP.cpp
git commit -m "Wire NEON_HP SQ8↔FP16 choosers [MOD-14972]"
```

---

## Task 5: NEON_HP dispatcher wiring in IP_space.cpp + L2_space.cpp

**Files:**
- Modify: `src/VecSim/spaces/IP_space.cpp` — `IP_SQ8_FP16_GetDistFunc` + `Cosine_SQ8_FP16_GetDistFunc`
- Modify: `src/VecSim/spaces/L2_space.cpp` — `L2_SQ8_FP16_GetDistFunc`

Each of those three `_GetDistFunc` functions currently has an `#ifdef CPU_FEATURES_ARCH_X86_64` block with an early `if (dim < 16) return ret_dist_func;` guard followed by per-tier dispatch. We append an `#ifdef CPU_FEATURES_ARCH_AARCH64` block with the matching shape. Only NEON_HP is wired in this task; SVE/SVE2 land in a later task.

- [ ] **Step 1: Confirm the #include for NEON_HP.h is present**

Run:
```bash
grep -n 'functions/NEON_HP.h' src/VecSim/spaces/IP_space.cpp src/VecSim/spaces/L2_space.cpp
```
Expected: both files already `#include "VecSim/spaces/functions/NEON_HP.h"`. If a file is missing it, add the include.

- [ ] **Step 2: Wire IP_SQ8_FP16_GetDistFunc**

In `src/VecSim/spaces/IP_space.cpp`, locate `IP_SQ8_FP16_GetDistFunc`. After the closing `#endif // x86_64`, insert a parallel AArch64 block immediately before the trailing `return ret_dist_func;`:

```cpp
#ifdef CPU_FEATURES_ARCH_AARCH64
    if (dim < 16) {
        return ret_dist_func;
    }
#ifdef OPT_NEON_HP
    if (features.asimdhp) {
        // No alignment write: the locked spec and the sister ARM SQ8_FP32 dispatchers
        // leave *alignment untouched on ARM tiers. The corresponding tests assert
        // 0xFF passthrough on the scalar path and do not assert any non-zero value here.
        return Choose_SQ8_FP16_IP_implementation_NEON_HP(dim);
    }
#endif
#endif // CPU_FEATURES_ARCH_AARCH64
```

- [ ] **Step 3: Wire Cosine_SQ8_FP16_GetDistFunc**

In the same file, locate `Cosine_SQ8_FP16_GetDistFunc`. Insert the same block, swapping `Choose_SQ8_FP16_IP_implementation_NEON_HP` for `Choose_SQ8_FP16_Cosine_implementation_NEON_HP`.

- [ ] **Step 4: Wire L2_SQ8_FP16_GetDistFunc**

In `src/VecSim/spaces/L2_space.cpp`, locate `L2_SQ8_FP16_GetDistFunc`. Insert the same block, swapping the call for `Choose_SQ8_FP16_L2_implementation_NEON_HP`.

- [ ] **Step 5: User builds**

Ask the user to run `make build` — first time the new NEON_HP TU additions compile. If they have ARM hardware or a cross-compile target, that build path; otherwise the x86 build must at least confirm the new headers don't accidentally break non-ARM compilation (the new headers are only `#include`d from `NEON_HP.cpp`, which is excluded on non-ARM hosts, so x86 builds should be clean).

- [ ] **Step 6: Commit**

```bash
git add src/VecSim/spaces/IP_space.cpp src/VecSim/spaces/L2_space.cpp
git commit -m "Dispatch SQ8↔FP16 to NEON_HP tier on AArch64 [MOD-14972]"
```

---

## Task 6: Extend `SQ8_FP16_SpacesOptimizationTest` with NEON_HP tier-walk

**Files:**
- Modify: `tests/unit/test_spaces.cpp` — three test bodies (`SQ8_FP16_L2SqrTest`, `SQ8_FP16_InnerProductTest`, `SQ8_FP16_CosineTest`)

After the existing `#ifdef OPT_SSE4` block in each test, append:

- [ ] **Step 1: Add NEON_HP tier to L2 test**

In `SQ8_FP16_L2SqrTest`, immediately after the closing `#endif` that follows the SSE4 block and before `// Scalar fallback`:

```cpp
#ifdef OPT_NEON_HP
    if (optimization.asimdhp) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_NEON_HP(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "NEON_HP with dim " << dim;
        optimization.asimdhp = 0;
    }
#endif
```

- [ ] **Step 2: Add NEON_HP tier to IP test**

In `SQ8_FP16_InnerProductTest`, append the same block but swap `L2_SQ8_FP16_GetDistFunc` → `IP_SQ8_FP16_GetDistFunc` and `Choose_SQ8_FP16_L2_implementation_NEON_HP` → `Choose_SQ8_FP16_IP_implementation_NEON_HP`.

- [ ] **Step 3: Add NEON_HP tier to Cosine test**

In `SQ8_FP16_CosineTest`, append the same block with `Cosine_SQ8_FP16_GetDistFunc` and `Choose_SQ8_FP16_Cosine_implementation_NEON_HP`.

- [ ] **Step 4: Confirm the include path for the NEON_HP chooser declarations**

Run:
```bash
grep -n 'functions/NEON_HP.h' tests/unit/test_spaces.cpp
```
Expected: include present. If not, add `#include "VecSim/spaces/functions/NEON_HP.h"` near the other space-function includes at the top of the file.

- [ ] **Step 5: User builds (ARM target)**

Ask the user to run `make build` for an ARM target (hardware or cross-compile). On x86 the new test code is gated by `#ifdef OPT_NEON_HP` and stays inert.

- [ ] **Step 6: Run NEON_HP tests**

Once the ARM build is reported clean, run:
```bash
./bin/<arm-triple>/unit_tests --gtest_filter='SQ8_FP16_*Test*'
```
Expected: all parametrized cases PASS, including the dims-16..32 and high-dim suites.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_spaces.cpp
git commit -m "Extend SQ8↔FP16 tier-walk tests with NEON_HP [MOD-14972]"
```

---

## Task 7: SVE IP kernel header

**Files:**
- Create: `src/VecSim/spaces/IP/IP_SVE_SQ8_FP16.h`

- [ ] **Step 1: Author the kernel file**

Modeled on `IP_SVE_SQ8_FP32.h`. The shape: an `InnerProductStep` helper that consumes `chunk = svcntw()` FP32 lanes per call (FP16 query loaded under a `b16` predicate, SQ8 storage under a `b32` predicate that drives uint8→uint32 widening), then a templated `_IMP` over `<bool partial_chunk, unsigned char additional_steps>`.

```cpp
/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include "VecSim/types/sq8.h"
#include "VecSim/types/float16.h"
#include <arm_sve.h>
#include <cassert>

using sq8 = vecsim_types::sq8;
using float16 = vecsim_types::float16;

/*
 * Optimised asymmetric SQ8<->FP16 inner product using the algebraic identity:
 *
 *   IP(x, y) ~= min * y_sum + delta * sum(q_i * y_i)
 *
 * Hot loop accumulates sum(q_i * y_i) only; FP16 query lanes are widened to FP32
 * inside each step via svcvt_f32_f16_x. Metadata loads use load_unaligned<float>.
 */

// Helper: one SVE-vector-width-of-FP32 step.
//   chunk = svcntw() - number of FP32 lanes per step.
//   pg    = svptrue_b32() - predicate for FP32 lanes.
static inline void
SQ8_FP16_InnerProductStep_SVE(const uint8_t *pVect1, const float16 *pVect2, size_t &offset,
                              svfloat32_t &sum, svbool_t pg, size_t chunk) {
    // SQ8 -> uint32 (widen on load), then to FP32.
    svuint32_t v1_u32 = svld1ub_u32(pg, pVect1 + offset);
    svfloat32_t v1_f = svcvt_f32_u32_x(pg, v1_u32);

    // FP16 query -> FP32. svld1_f16 uses a b16 predicate sized to `chunk` half lanes.
    svbool_t pg16 = svwhilelt_b16(uint32_t(0), uint32_t(chunk));
    svfloat16_t q_h =
        svld1_f16(pg16, reinterpret_cast<const float16_t *>(pVect2) + offset);
    svfloat32_t v2_f = svcvt_f32_f16_x(pg, q_h);

    sum = svmla_f32_x(pg, sum, v1_f, v2_f);
    offset += chunk;
}

// pVect1v = SQ8 storage, pVect2v = FP16 query
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP16_InnerProductSIMD_SVE_IMP(const void *pVect1v, const void *pVect2v,
                                        size_t dimension) {
    assert(dimension >= 16 && "kernel precondition: dispatcher must guard dim >= 16");

    const uint8_t *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const float16 *pVect2 = static_cast<const float16 *>(pVect2v);
    size_t offset = 0;
    svbool_t pg = svptrue_b32();
    const size_t chunk = svcntw();

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    // Partial chunk for dim % chunk lanes. Use _z form so inactive lanes are zero -
    // the final reduction below walks all lanes via svptrue_b32().
    if constexpr (partial_chunk) {
        size_t remaining = dimension % chunk;
        if (remaining > 0) {
            svbool_t pg_partial =
                svwhilelt_b32(uint32_t(0), uint32_t(remaining));
            svbool_t pg16_partial =
                svwhilelt_b16(uint32_t(0), uint32_t(remaining));
            svuint32_t v1_u32 = svld1ub_u32(pg_partial, pVect1 + offset);
            svfloat32_t v1_f = svcvt_f32_u32_z(pg_partial, v1_u32);
            svfloat16_t q_h = svld1_f16(
                pg16_partial, reinterpret_cast<const float16_t *>(pVect2) + offset);
            svfloat32_t v2_f = svcvt_f32_f16_z(pg_partial, q_h);
            sum0 = svmla_f32_z(pg_partial, sum0, v1_f, v2_f);
            offset += remaining;
        }
    }

    // Main loop: 4 chunks per iteration via 4 accumulators.
    const size_t chunk_size = 4 * chunk;
    const size_t number_of_chunks =
        (dimension - (partial_chunk ? dimension % chunk : 0)) / chunk_size;
    for (size_t i = 0; i < number_of_chunks; i++) {
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum0, pg, chunk);
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum1, pg, chunk);
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum2, pg, chunk);
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum3, pg, chunk);
    }

    // Additional steps 0..3.
    if constexpr (additional_steps > 0)
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum0, pg, chunk);
    if constexpr (additional_steps > 1)
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum1, pg, chunk);
    if constexpr (additional_steps > 2)
        SQ8_FP16_InnerProductStep_SVE(pVect1, pVect2, offset, sum2, pg, chunk);

    svfloat32_t sum = svadd_f32_x(pg, sum0, sum1);
    sum = svadd_f32_x(pg, sum, sum2);
    sum = svadd_f32_x(pg, sum, sum3);
    float quantized_dot = svaddv_f32(pg, sum);

    // Metadata loads - unaligned because odd dim leaves trailers unaligned.
    const uint8_t *params_bytes = static_cast<const uint8_t *>(pVect1v) + dimension;
    const float min_val =
        load_unaligned<float>(params_bytes + sq8::MIN_VAL * sizeof(float));
    const float delta =
        load_unaligned<float>(params_bytes + sq8::DELTA * sizeof(float));
    const uint8_t *query_meta_bytes = reinterpret_cast<const uint8_t *>(
        static_cast<const float16 *>(pVect2v) + dimension);
    const float y_sum =
        load_unaligned<float>(query_meta_bytes + sq8::SUM_QUERY * sizeof(float));

    return min_val * y_sum + delta * quantized_dot;
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP16_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v,
                                    size_t dimension) {
    return 1.0f - SQ8_FP16_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(
                      pVect1v, pVect2v, dimension);
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP16_CosineSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return SQ8_FP16_InnerProductSIMD_SVE<partial_chunk, additional_steps>(
        pVect1v, pVect2v, dimension);
}
```

**Note for the implementer:** `svcvt_f32_f16_x(pg, q_h)` widens *the lower half of `q_h`'s lanes* to FP32 (one widening, b32-predicated). If the ACLE on the target toolchain rejects this pairing (e.g. ARM RVCT vs LLVM disagreement), verify the FP16->FP32 widening sequence against the actual ARM build output and adjust as needed (potential alternatives: explicit `svunpklo_*` unpack-then-widen, or operating on the lower half lanes by reinterpretation). Commit only after the build is clean. Do not blindly copy `IP_SVE_FP16.h`'s pattern - that file accumulates in FP16 and is not a direct widening reference.

- [ ] **Step 2: Commit**

```bash
git add src/VecSim/spaces/IP/IP_SVE_SQ8_FP16.h
git commit -m "Add SVE SQ8↔FP16 IP kernel header [MOD-14972]"
```

---

## Task 8: SVE L2 kernel header

**Files:**
- Create: `src/VecSim/spaces/L2/L2_SVE_SQ8_FP16.h`

- [ ] **Step 1: Author the kernel file**

```cpp
/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP/IP_SVE_SQ8_FP16.h"
#include "VecSim/types/sq8.h"
#include "VecSim/types/float16.h"

using sq8 = vecsim_types::sq8;
using float16 = vecsim_types::float16;

/*
 * SVE SQ8<->FP16 L2 squared distance:
 *   ||x - y||^2 = x_sum_squares - 2 * IP(x, y) + y_sum_squares
 * IP is computed by SQ8_FP16_InnerProductSIMD_SVE_IMP; metadata is FP32.
 */

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP16_L2SqrSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float ip = SQ8_FP16_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(
        pVect1v, pVect2v, dimension);

    const uint8_t *params_bytes = static_cast<const uint8_t *>(pVect1v) + dimension;
    const float x_sum_sq =
        load_unaligned<float>(params_bytes + sq8::SUM_SQUARES * sizeof(float));
    const uint8_t *query_meta_bytes = reinterpret_cast<const uint8_t *>(
        static_cast<const float16 *>(pVect2v) + dimension);
    const float y_sum_sq =
        load_unaligned<float>(query_meta_bytes + sq8::SUM_SQUARES_QUERY * sizeof(float));

    return x_sum_sq + y_sum_sq - 2.0f * ip;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/VecSim/spaces/L2/L2_SVE_SQ8_FP16.h
git commit -m "Add SVE SQ8↔FP16 L2 kernel header [MOD-14972]"
```

---

## Task 9: SVE + SVE2 dispatcher TU additions

**Files:**
- Modify: `src/VecSim/spaces/functions/SVE.h` — +3 declarations
- Modify: `src/VecSim/spaces/functions/SVE.cpp` — +#includes; +3 chooser definitions
- Modify: `src/VecSim/spaces/functions/SVE2.h` — +3 declarations
- Modify: `src/VecSim/spaces/functions/SVE2.cpp` — +#includes; +3 chooser definitions (own symbols, template instantiated under SVE2 flags)

- [ ] **Step 1: Declarations in SVE.h**

Inside `namespace spaces { ... }`, alongside the existing `Choose_SQ8_FP32_*_SVE` declarations:

```cpp
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_SVE(size_t dim);
dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_SVE(size_t dim);
dist_func_t<float> Choose_SQ8_FP16_L2_implementation_SVE(size_t dim);
```

- [ ] **Step 2: Definitions in SVE.cpp**

Add includes alongside the existing SQ8_FP32 includes:

```cpp
#include "VecSim/spaces/IP/IP_SVE_SQ8_FP16.h"
#include "VecSim/spaces/L2/L2_SVE_SQ8_FP16.h"
```

Inside `namespace spaces { ... }` (between `implementation_chooser.h` and `implementation_chooser_cleanup.h`), append:

```cpp
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_SVE(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, SQ8_FP16_InnerProductSIMD_SVE, dim, svcntw);
    return ret_dist_func;
}

dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_SVE(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, SQ8_FP16_CosineSIMD_SVE, dim, svcntw);
    return ret_dist_func;
}

dist_func_t<float> Choose_SQ8_FP16_L2_implementation_SVE(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, SQ8_FP16_L2SqrSIMD_SVE, dim, svcntw);
    return ret_dist_func;
}
```

- [ ] **Step 3: Declarations in SVE2.h**

```cpp
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_SVE2(size_t dim);
dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_SVE2(size_t dim);
dist_func_t<float> Choose_SQ8_FP16_L2_implementation_SVE2(size_t dim);
```

- [ ] **Step 4: Definitions in SVE2.cpp**

Add includes alongside the existing SQ8_FP32 includes — note the SVE header is included from SVE2 (SVE2 instantiates the template under SVE2 compile flags):

```cpp
#include "VecSim/spaces/IP/IP_SVE_SQ8_FP16.h" // SVE2 implementation is identical to SVE
#include "VecSim/spaces/L2/L2_SVE_SQ8_FP16.h" // SVE2 implementation is identical to SVE
```

Inside `namespace spaces { ... }`, append:

```cpp
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, SQ8_FP16_InnerProductSIMD_SVE, dim, svcntw);
    return ret_dist_func;
}

dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, SQ8_FP16_CosineSIMD_SVE, dim, svcntw);
    return ret_dist_func;
}

dist_func_t<float> Choose_SQ8_FP16_L2_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, SQ8_FP16_L2SqrSIMD_SVE, dim, svcntw);
    return ret_dist_func;
}
```

- [ ] **Step 5: Commit**

```bash
git add src/VecSim/spaces/functions/SVE.h src/VecSim/spaces/functions/SVE.cpp \
        src/VecSim/spaces/functions/SVE2.h src/VecSim/spaces/functions/SVE2.cpp
git commit -m "Wire SVE/SVE2 SQ8↔FP16 choosers [MOD-14972]"
```

---

## Task 10: SVE + SVE2 dispatcher wiring in IP_space.cpp + L2_space.cpp

The NEON_HP block added in Task 5 lives inside `#ifdef CPU_FEATURES_ARCH_AARCH64`. Extend the same block in all three `_GetDistFunc` functions with SVE2 and SVE tiers — ordered SVE2 → SVE → NEON_HP, matching every other SQ8/FP32 dispatcher in the file.

**Files:**
- Modify: `src/VecSim/spaces/IP_space.cpp` (two functions)
- Modify: `src/VecSim/spaces/L2_space.cpp` (one function)

- [ ] **Step 1: Confirm the SVE/SVE2 dispatcher includes are present**

Run:
```bash
grep -n 'functions/SVE\.h\|functions/SVE2\.h' src/VecSim/spaces/IP_space.cpp src/VecSim/spaces/L2_space.cpp
```
Expected: both files already include both headers. If not, add them.

- [ ] **Step 2: Extend IP_SQ8_FP16_GetDistFunc**

Inside the AArch64 block of `IP_SQ8_FP16_GetDistFunc`, after the `if (dim < 16) return ret_dist_func;` guard and **before** the existing `#ifdef OPT_NEON_HP`, prepend:

```cpp
#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_SQ8_FP16_IP_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_SQ8_FP16_IP_implementation_SVE(dim);
    }
#endif
```

(SVE/SVE2 paths don't compute alignment hints — the SVE vector width is runtime-variable, so the SQ8_FP32 sister doesn't set `*alignment` here either. Mirror that.)

- [ ] **Step 3: Extend Cosine_SQ8_FP16_GetDistFunc**

Same as Step 2, with `Cosine` in the chooser names.

- [ ] **Step 4: Extend L2_SQ8_FP16_GetDistFunc**

Same as Step 2, with `L2` in the chooser names.

- [ ] **Step 5: User builds (ARM target)**

Ask user to run `make build` for an ARM target.

- [ ] **Step 6: Commit**

```bash
git add src/VecSim/spaces/IP_space.cpp src/VecSim/spaces/L2_space.cpp
git commit -m "Dispatch SQ8↔FP16 to SVE/SVE2 tiers on AArch64 [MOD-14972]"
```

---

## Task 11: Extend `SQ8_FP16_SpacesOptimizationTest` with SVE2 + SVE tier-walks

**Files:**
- Modify: `tests/unit/test_spaces.cpp` — the same three test bodies extended in Task 6

For each test (L2, IP, Cosine), inside the existing `#ifdef CPU_FEATURES_ARCH_AARCH64` region (which currently holds only NEON_HP from Task 6), **prepend** SVE2 and SVE blocks so the dispatch-precedence order is SVE2 → SVE → NEON_HP. If the existing NEON_HP block is not yet inside an AArch64 outer ifdef, wrap all three together.

- [ ] **Step 1: Wrap and extend the L2 test**

Replace the NEON_HP-only AArch64 block in `SQ8_FP16_L2SqrTest` with:

```cpp
#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (optimization.sve2) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_SVE2(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SVE2 with dim " << dim;
        optimization.sve2 = 0;
    }
#endif
#ifdef OPT_SVE
    if (optimization.sve) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_SVE(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "SVE with dim " << dim;
        optimization.sve = 0;
    }
#endif
#ifdef OPT_NEON_HP
    if (optimization.asimdhp) {
        unsigned char alignment = 0;
        arch_opt_func = L2_SQ8_FP16_GetDistFunc(dim, &alignment, &optimization);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_L2_implementation_NEON_HP(dim))
            << "Unexpected distance function chosen for dim " << dim;
        ASSERT_NEAR(baseline, arch_opt_func(v2_compressed.data(), v1_query.data(), dim), 0.01)
            << "NEON_HP with dim " << dim;
        optimization.asimdhp = 0;
    }
#endif
#endif // CPU_FEATURES_ARCH_AARCH64
```

- [ ] **Step 2: Same for IP test**

Replicate the block in `SQ8_FP16_InnerProductTest` with `IP_SQ8_FP16_GetDistFunc` and `Choose_SQ8_FP16_IP_implementation_<TIER>`.

- [ ] **Step 3: Same for Cosine test**

Replicate with `Cosine_SQ8_FP16_GetDistFunc` and `Choose_SQ8_FP16_Cosine_implementation_<TIER>`.

- [ ] **Step 4: User builds**

ARM target build.

- [ ] **Step 5: Run the optimization tests**

```bash
./bin/<arm-triple>/unit_tests --gtest_filter='SQ8_FP16_SpacesOptimizationTest.*'
```
Expected: all parametrized cases PASS — dims 16..32 + high-dim suite (64..1024) — exercising whichever ARM tiers the host advertises.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_spaces.cpp
git commit -m "Extend SQ8↔FP16 tier-walk tests with SVE/SVE2 [MOD-14972]"
```

---

## Task 12: Extend `SQ8_FP16_SIMD_TierCoverage.ReportTiersExercised` with ARM rows

**Files:**
- Modify: `tests/unit/test_spaces.cpp` — `TEST(SQ8_FP16_SIMD_TierCoverage, ReportTiersExercised)`

The existing test body has an outer `#ifdef CPU_FEATURES_ARCH_X86_64` block that loops over each x86 tier and logs presence to stderr. Add a sibling `#ifdef CPU_FEATURES_ARCH_AARCH64` block with the same shape.

- [ ] **Step 1: Append the AArch64 reporting block**

Locate the trailing `#endif // CPU_FEATURES_ARCH_X86_64` and immediately after, insert:

```cpp
#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (opt.sve2) {
        std::cerr << "[SQ8_FP16] SVE2 tier exercised\n";
        any_simd = true;
    } else {
        std::cerr << "[SQ8_FP16] SVE2 tier NOT exercised on this host\n";
    }
#endif
#ifdef OPT_SVE
    if (opt.sve) {
        std::cerr << "[SQ8_FP16] SVE tier exercised\n";
        any_simd = true;
    } else {
        std::cerr << "[SQ8_FP16] SVE tier NOT exercised on this host\n";
    }
#endif
#ifdef OPT_NEON_HP
    if (opt.asimdhp) {
        std::cerr << "[SQ8_FP16] NEON_HP tier exercised\n";
        any_simd = true;
    } else {
        std::cerr << "[SQ8_FP16] NEON_HP tier NOT exercised on this host\n";
    }
#endif
#endif // CPU_FEATURES_ARCH_AARCH64
```

(The trailing `if (!any_simd) { GTEST_SKIP() << ...; }` already at the bottom of the existing test handles the all-quiet case across both archs.)

- [ ] **Step 2: Build + run on an ARM host**

Ask the user to build for ARM, then run:
```bash
./bin/<arm-triple>/unit_tests --gtest_filter='SQ8_FP16_SIMD_TierCoverage.*'
```
Expected: stderr shows at least one ARM tier marked "exercised", test PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_spaces.cpp
git commit -m "Report ARM tiers in SQ8↔FP16 tier-coverage test [MOD-14972]"
```

---

## Task 13: Microbench AArch64 block

**Files:**
- Modify: `tests/benchmark/spaces_benchmarks/bm_spaces_sq8_fp16.cpp`

The existing file already opens `#ifdef CPU_FEATURES_ARCH_X86_64` and pulls `cpu_features::X86Features opt = cpu_features::GetX86Info().features;`. Add the parallel AArch64 block at the end of that `#endif // CPU_FEATURES_ARCH_X86_64`.

- [ ] **Step 1: Append the AArch64 bench block**

After the closing `#endif // CPU_FEATURES_ARCH_X86_64` (or after the last x86 `INITIALIZE_BENCHMARKS_SET_*` macro if no such comment exists), insert:

```cpp
#ifdef CPU_FEATURES_ARCH_AARCH64
cpu_features::Aarch64Features arm_opt = cpu_features::GetAarch64Info().features;

#ifdef OPT_SVE2
bool sve2_supported = arm_opt.sve2;
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, SVE2, 16, sve2_supported);
INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, SVE2, 16, sve2_supported);
#endif

#ifdef OPT_SVE
bool sve_supported = arm_opt.sve;
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, SVE, 16, sve_supported);
INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, SVE, 16, sve_supported);
#endif

#ifdef OPT_NEON_HP
bool neon_hp_supported = arm_opt.asimdhp;
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, NEON_HP, 16, neon_hp_supported);
INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, NEON_HP, 16,
                                  neon_hp_supported);
#endif
#endif // CPU_FEATURES_ARCH_AARCH64
```

Verify the exact `cpu_features` helper name during build. If the toolchain uses `Aarch64Info` vs `Aarch64Features` vs `ArmFeatures`, adjust to match the sister x86 block.

- [ ] **Step 2: Update the file-header comment**

The current file-header comment (around the top) ends with `ARM kernels land via MOD-14972.` — change that line to `ARM kernels (NEON_HP / SVE / SVE2) are registered below.` so the doc stays accurate.

- [ ] **Step 3: User builds (ARM target)**

- [ ] **Step 4: Run the bench on ARM**

```bash
./bin/<arm-triple>/bm_spaces_sq8_fp16 --benchmark_filter='SQ8_FP16_.*(SVE2|SVE|NEON_HP)'
```
Expected: per-ISA throughput rows for L2, IP, Cosine. If no rows match, list all benchmarks first with `--benchmark_list_tests` to see the exact generated names, then adjust the regex.

- [ ] **Step 5: Side-by-side compare against SQ8_FP32**

```bash
./bin/<arm-triple>/bm_spaces_sq8_fp32 --benchmark_filter='SQ8_FP32_.*(SVE2|SVE|NEON)'
```
Compare matched-ISA rows manually. Acceptance per Jira: per-ISA throughput data captured.

- [ ] **Step 6: Commit**

```bash
git add tests/benchmark/spaces_benchmarks/bm_spaces_sq8_fp16.cpp
git commit -m "Register ARM SQ8↔FP16 microbenchmarks [MOD-14972]"
```

---

## Task 14: ASan + final pre-PR verification

- [ ] **Step 1: Full unit-test pass on ARM host (no filter)**

```bash
./bin/<arm-triple>/unit_tests
```
Expected: all tests PASS.

- [ ] **Step 2: ASan build + run**

Ask user to run `make build SAN=address` (or the repo's equivalent — verify against `Makefile`). After confirmed:

```bash
./bin/<arm-triple>-asan/unit_tests --gtest_filter='SQ8_FP16_*'
```
Expected: zero ASan reports; all SQ8_FP16 tests PASS.

- [ ] **Step 3: x86 sanity build**

User runs `make build` on x86 (no ARM target). Confirms the new test extensions and dispatcher AArch64 ifdefs stay inert on x86 and the build is clean.

- [ ] **Step 4: Push branch (ASK USER FIRST)**

Pushes are user-gated. Confirm with the user before running:

```bash
git push -u origin dor-forer-sq8-fp16-arm-kernels-mod-14972
```

- [ ] **Step 5: Open PR against PR #970 (ASK USER FIRST)**

PR creation is user-gated. Confirm with the user before running:

```bash
gh pr create \
  --base dor-forer-sq8-fp16-x86-kernels-mod-14954 \
  --title 'Add SQ8↔FP16 ARM SIMD distance kernels [MOD-14972]' \
  --body "$(cat <<'EOF'
## Summary

- Add asymmetric SQ8↔FP16 distance kernels (IP, L2, Cosine) for ARM NEON_HP, SVE, SVE2 tiers
- Wire kernels into the existing dispatcher (`IP_space.cpp`, `L2_space.cpp`)
- Extend `SQ8_FP16_SpacesOptimizationTest` and `SQ8_FP16_SIMD_TierCoverage` with ARM tiers
- Register per-ISA microbenchmarks for cross-arch throughput comparison

Stacked on PR #970 (MOD-14954 x86 kernels); retarget to `main` once #970 merges.

Spec: `docs/superpowers/specs/2026-05-28-arm-sq8-fp16-design.md`

## Test plan

- [ ] Unit tests on ARM host pass — `SQ8_FP16_SpacesOptimizationTest` (dims 16..32 + 64..1024), `SQ8_FP16_SIMD_TierCoverage`, `GetDistFuncSQ8FP16Asymmetric`
- [ ] ASan build on ARM host clean across SQ8_FP16 tests
- [ ] x86 build remains clean (new AArch64 dispatcher block + tests stay inert)
- [ ] Microbench output captured for SVE2 / SVE / NEON_HP, compared against matched SQ8_FP32 ARM rows
EOF
)"
```

- [ ] **Step 6: Retarget once #970 merges (ASK USER FIRST)**

When PR #970 lands on `main`, change this PR's base to `main`:

```bash
gh pr edit <PR-number> --base main
```

---

## Self-review checklist

- [x] **Spec coverage:** every requirement in `2026-05-28-arm-sq8-fp16-design.md` is covered:
  - Kernel headers (4 new): Tasks 2, 3, 7, 8
  - Wrapper symbols: Tasks 4 (NEON_HP), 9 (SVE/SVE2)
  - Dispatcher wiring: Tasks 5 (NEON_HP), 10 (SVE/SVE2)
  - Tier-walk tests: Tasks 6 (NEON_HP), 11 (SVE/SVE2)
  - TierCoverage report: Task 12
  - Scalar-fallback edge tests (dim=0, dim=15): Task 1
  - Microbench: Task 13
  - ASan + verification: Task 14
- [x] **No CMake changes** — confirmed in file structure table.
- [x] **Zero placeholders** — every code block is concrete; ambiguous spots (SVE FP16 widening ACLE) are called out with the fallback strategy spelled in-task.
- [x] **Type/symbol consistency:**
  - NEON kernel template names: `SQ8_FP16_InnerProductSIMD16_NEON_HP_IMP` / `…NEON_HP` / `SQ8_FP16_L2SqrSIMD16_NEON_HP` / `SQ8_FP16_CosineSIMD16_NEON_HP` — match across kernel header, NEON_HP chooser, dispatcher call, and test.
  - SVE kernel template names: `SQ8_FP16_InnerProductSIMD_SVE_IMP` / `…SVE` / `SQ8_FP16_L2SqrSIMD_SVE` / `SQ8_FP16_CosineSIMD_SVE` — match across kernel header, SVE chooser, SVE2 chooser, dispatcher call, and test.
  - Chooser symbol names: `Choose_SQ8_FP16_{IP,L2,Cosine}_implementation_{NEON_HP,SVE,SVE2}` — match across `.h` declarations, `.cpp` definitions, dispatcher calls, tests, and bench.
  - Test fixture: `SQ8_FP16_SpacesOptimizationTest` already exists on base (PR #970); we extend the three test methods inside it, no rename.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-28-arm-sq8-fp16-kernels.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

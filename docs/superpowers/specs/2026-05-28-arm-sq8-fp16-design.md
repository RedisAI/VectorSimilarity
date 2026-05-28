# SQ8↔FP16 ARM SIMD Distance Kernels — Design Spec

- **Ticket**: [MOD-14972](https://redislabs.atlassian.net/browse/MOD-14972)
- **Branch**: `dor-forer-sq8-fp16-arm-kernels-mod-14972`
- **Base**: `dor-forer-sq8-fp16-x86-kernels-mod-14954` (PR #970) — stacked
- **Sibling**: MOD-14954 / PR #970 delivers x86 SIMD kernels (AVX-512, AVX2, SSE4) for the same operation

## Goal

Add SQ8↔FP16 SIMD distance kernels for IP and L2 on the ARM ISA tiers (NEON_HP, SVE, SVE2). FP16 is the query data type; SQ8 is the stored vector representation. Match the contract and structure of the x86 kernels delivered in PR #970 so dispatch tables, metadata layout, and acceptance criteria stay symmetric across architectures.

The scalar fallback (`SQ8_FP16_InnerProduct`, `SQ8_FP16_L2Sqr`, `SQ8_FP16_Cosine` in `src/VecSim/spaces/IP/IP.cpp` and `src/VecSim/spaces/L2/L2.cpp`) already exists on `main`. This spec does not modify it; it serves as the reference implementation for all platforms.

## Algebraic identity (shared with x86 PR + SQ8_FP32 sister)

```
IP(x, y) ≈ min · y_sum + delta · Σ(q_i · y_i)
L2(x, y) = x_sum_sq + y_sum_sq - 2 · IP(x, y)
```

Hot loop accumulates `Σ(q_i · y_i)` only. No per-element dequantization. FP16 query lanes are widened to FP32 per SIMD chunk; everything in the hot loop is FP32.

## Metadata layout

```
SQ8 storage (pVect1): [uint8 × dim] [min_val] [delta] [x_sum] [x_sum_squares]
FP16 query   (pVect2): [float16 × dim] [y_sum] [y_sum_squares]
```

Both metadata trailers are FP32 scalars. Storage metadata is not 4-byte aligned whenever `dim % 4 != 0`; query metadata is not 4-byte aligned whenever `dim` is odd. The blanket rule: every FP32 metadata read uses the global `load_unaligned<float>` helper, matching scalar `_Impl` in `IP.cpp` / `L2.cpp`. `sq8` namespace constants: `MIN_VAL`, `DELTA`, `SUM_QUERY`, `SUM_SQUARES`, `SUM_SQUARES_QUERY`.

## File layout

```
src/VecSim/spaces/IP/
  IP_NEON_SQ8_FP16.h     (new)
  IP_SVE_SQ8_FP16.h      (new) — also #included from SVE2.cpp
src/VecSim/spaces/L2/
  L2_NEON_SQ8_FP16.h     (new)
  L2_SVE_SQ8_FP16.h      (new) — also #included from SVE2.cpp
src/VecSim/spaces/functions/
  NEON_HP.cpp            (+ Choose_SQ8_FP16_{IP,L2,Cosine}_implementation_NEON_HP)
  NEON_HP.h              (+ 3 declarations)
  SVE.cpp                (+ Choose_SQ8_FP16_*_implementation_SVE)
  SVE.h                  (+ 3 declarations)
  SVE2.cpp               (+ Choose_SQ8_FP16_*_implementation_SVE2; owns its own chooser symbols; instantiates SVE kernel templates under SVE2 compile flags)
  SVE2.h                 (+ 3 declarations)
src/VecSim/spaces/
  IP_space.cpp           (2 dispatcher block edits: IP, Cosine)
  L2_space.cpp           (1 dispatcher block edit)
```

**Zero CMake changes.** Existing TU flags carry exactly what we need:

| TU | Flags |
|----|-------|
| `NEON_HP.cpp` | `-march=armv8.2-a+fp16fml` (covers fp16 cvt + fma) |
| `SVE.cpp` | `-march=armv8-a+sve` (SVE includes f16↔f32 cvt) |
| `SVE2.cpp` | `-march=armv9-a+sve2` |

## Dispatcher tier order

Same precedence as existing SQ8_FP32 ARM dispatch:

```cpp
#ifdef OPT_SVE2
    if (features.sve2 && dim >= 16) {
        return Choose_SQ8_FP16_IP_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve && dim >= 16) {
        return Choose_SQ8_FP16_IP_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON_HP
    if (features.asimdhp && dim >= 16) {
        return Choose_SQ8_FP16_IP_implementation_NEON_HP(dim);
    }
#endif
// dim < 16 or no ARM SIMD → scalar fallback (existing return at function tail)
```

The `dim >= 16` guard in the dispatcher is what lets each SIMD kernel hold an internal `assert(dim >= 16)` as a real precondition. Edge cases for `dim < 16` are routed to scalar.

## NEON kernel design

### Header: `IP_NEON_SQ8_FP16.h`

Template signature mirrors SQ8_FP32 NEON sister:

```cpp
template <unsigned char residual>   // 0..15
float SQ8_FP16_InnerProductSIMD16_NEON_HP_IMP(const void *pVect1v, const void *pVect2v, size_t dimension);
```

Hot loop — 16 lanes per iteration, 4 FP32 accumulators:

```cpp
// SQ8 load: 16 × uint8 → 4 × float32x4_t
uint8x16_t v1_u8 = vld1q_u8(pVect1);
uint16x8_t v1_lo = vmovl_u8(vget_low_u8(v1_u8));
uint16x8_t v1_hi = vmovl_u8(vget_high_u8(v1_u8));
float32x4_t v1_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v1_lo)));
float32x4_t v1_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v1_lo)));
float32x4_t v1_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v1_hi)));
float32x4_t v1_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v1_hi)));

// FP16 query load: 16 × f16 → 4 × float32x4_t via vcvt_f32_f16
float16x8_t q_lo = vld1q_f16(pVect2);
float16x8_t q_hi = vld1q_f16(pVect2 + 8);
float32x4_t v2_0 = vcvt_f32_f16(vget_low_f16(q_lo));
float32x4_t v2_1 = vcvt_f32_f16(vget_high_f16(q_lo));
float32x4_t v2_2 = vcvt_f32_f16(vget_low_f16(q_hi));
float32x4_t v2_3 = vcvt_f32_f16(vget_high_f16(q_hi));

// 4-accumulator FMA
sum0 = vfmaq_f32(sum0, v1_0, v2_0);
sum1 = vfmaq_f32(sum1, v1_1, v2_1);
sum2 = vfmaq_f32(sum2, v1_2, v2_2);
sum3 = vfmaq_f32(sum3, v1_3, v2_3);
```

Residual ladder (`dim % 16`, residual 0..15):

- **`residual >= 8`**: one 8-lane safe load each side — `vld1_u8` (8 bytes) for SQ8 and `vld1q_f16` (8 × FP16 = 16 bytes, fits before query metadata) for FP16. Convert + FMA. Remaining `residual - 8` lanes handled scalar.
- **`residual < 8`**: full scalar residual loop using `vecsim_types::FP16_to_FP32`.

Rationale: a 16-byte SQ8 load (`vld1q_u8`) or a 16-byte FP16 load (`vld1q_f16` past the 8-lane boundary) on a residual < 8 would overread past valid query data into metadata — `y_sum` is only 4 bytes for IP and `y_sum_sq` adds 4 more for L2, not enough headroom for an 8-lane FP16 load.

Final reduction: `vaddvq_f32(sum0 + sum1 + sum2 + sum3)`, then return `min_val * y_sum + delta * quantized_dot`.

`assert(dim >= 16)` at the top.

### Header: `L2_NEON_SQ8_FP16.h`

Calls `SQ8_FP16_InnerProductSIMD16_NEON_HP_IMP<residual>(...)` to compute raw IP, then returns `x_sum_sq + y_sum_sq - 2.0f * ip`. Mirrors `L2_NEON_SQ8_FP32.h` exactly.

### Wrapper symbols (NEON_HP.cpp)

```cpp
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_NEON_HP(size_t dim) {
    dist_func_t<float> ret;
    CHOOSE_IMPLEMENTATION(ret, dim, 16, SQ8_FP16_InnerProductSIMD16_NEON_HP);
    return ret;
}
// L2 + Cosine identical shape (Cosine reuses IP wrapper per repo convention)
```

## SVE kernel design

### Header: `IP_SVE_SQ8_FP16.h`

Template signature mirrors SVE SQ8_FP32 sister:

```cpp
template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP16_InnerProductSIMD_SVE_IMP(const void *pVect1v, const void *pVect2v, size_t dimension);
```

Inner step (one SVE vector width `svcntw()` lanes of FP32):

```cpp
svbool_t pg = svptrue_b32();
// SQ8: zero-extend uint8 → uint32 (predicated b32 load)
svuint32_t v1_u32 = svld1ub_u32(pg, pVect1 + offset);
svfloat32_t v1_f  = svcvt_f32_u32_x(pg, v1_u32);
// FP16: load chunk fp16 lanes, widen to fp32
svbool_t pg16 = svwhilelt_b16(uint32_t(0), uint32_t(chunk));
svfloat16_t q_h = svld1_f16(pg16, pVect2 + offset);
svfloat32_t v2_f = svcvt_f32_f16_x(pg, q_h);   // verify exact ACLE/packing during impl
sum = svmla_f32_x(pg, sum, v1_f, v2_f);
offset += chunk;
```

**ACLE caveat**: exact f16→f32 widening intrinsic and lane packing — confirm `svcvt_f32_f16_x(pg, q_h)` compiles cleanly against the loaded `svfloat16_t`. If lane packing needs an unpack/interleave step, verify against `IP_SVE_FP16.h`.

4 accumulators `sum0..sum3`; main loop processes 4 chunks via 4 `InnerProductStep` calls. `partial_chunk` template branch handles `dim % chunk` via `svwhilelt_b32`.

Inactive-lane discipline on the partial path: the predicated `svld1_f16` / `svld1ub_u32` cover lane *liveness*, but the final reduction with `svaddv_f32(svptrue_b32(), ...)` walks *all* lanes. To keep inactive lanes from contributing garbage, the partial step uses the zeroing form `svmla_f32_z(pg_partial, sum0, v1_f, v2_f)` (matches `IP_SVE_SQ8_FP32.h` partial-chunk pattern). Alternative: reduce only active lanes via `svaddv_f32(pg_partial, sum0)` for the partial-step accumulator, then sum into the main reduction. The `_z` form is the simpler choice and is what the SQ8_FP32 SVE sister already does.

Predicate widths on the partial path: FP32 math (load/widen/mla) uses a `b32` predicate sized to `remaining` 32-bit lanes (`svwhilelt_b32(0, remaining)`); the FP16 query load needs its own `b16` predicate sized to the same `remaining` half lanes (`svwhilelt_b16(0, remaining)`) since `svld1_f16` is governed by a 16-bit predicate. SQ8 load via `svld1ub_u32` is governed by the `b32` predicate (it widens uint8 → uint32 lanewise).

Final reduction: `svaddv_f32(svptrue_b32(), sum0 + sum1 + sum2 + sum3)`.

### Header: `L2_SVE_SQ8_FP16.h`

Calls `SQ8_FP16_InnerProductSIMD_SVE_IMP<partial_chunk, additional_steps>(...)` then returns `x_sum_sq + y_sum_sq - 2.0f * ip`. Mirrors `L2_SVE_SQ8_FP32.h`.

### Wrapper symbols

`SVE.cpp`:

```cpp
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_SVE(size_t dim) {
    dist_func_t<float> ret;
    CHOOSE_SVE_IMPLEMENTATION(ret, SQ8_FP16_InnerProductSIMD_SVE, dim, svcntw);
    return ret;
}
// L2 + Cosine identical shape
```

`SVE2.cpp`:

```cpp
#include "VecSim/spaces/IP/IP_SVE_SQ8_FP16.h"  // SVE2 implementation is identical to SVE
#include "VecSim/spaces/L2/L2_SVE_SQ8_FP16.h"

dist_func_t<float> Choose_SQ8_FP16_IP_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret;
    CHOOSE_SVE_IMPLEMENTATION(ret, SQ8_FP16_InnerProductSIMD_SVE, dim, svcntw);
    return ret;
}
// L2 + Cosine identical shape
```

SVE2 owns its own chooser symbols (does **not** call the SVE chooser); template instantiated under SVE2 compile flags.

## Tests

### Class

Branch base is PR #970. During implementation, verify whether the base branch already exposes `SQ8_FP16_SpacesOptimizationTest` (extend) or only `SQ8_FP16_NoOptimizationSpacesTest` (add the optimization class here mirroring `SQ8_FP32_SpacesOptimizationTest`).

### Tier-walk pattern

Per-tier `if (features.<flag>)` block; **unset higher flag** after each block so the next tier is exercised on hosts that support multiple ISAs. Do not use `GTEST_SKIP()` here — it would abort the entire walk.

```cpp
auto expected = SQ8_FP16_InnerProduct;  // scalar reference

#ifdef OPT_SVE2
    if (features.sve2) {
        arch_opt_func = IP_SQ8_FP16_GetDistFunc(dim, &alignment, &features);
        ASSERT_EQ(arch_opt_func, Choose_SQ8_FP16_IP_implementation_SVE2(dim))
            << "SVE2 dispatch mismatch";
        ASSERT_NEAR(arch_opt_func(v1, v2, dim), expected(v1, v2, dim), 0.01);
        features.sve2 = 0;   // exercise next tier
    }
#endif
#ifdef OPT_SVE
    if (features.sve) { /* same shape */ features.sve = 0; }
#endif
#ifdef OPT_NEON_HP
    if (features.asimdhp) { /* same shape */ features.asimdhp = 0; }
#endif
// final fallback assertion: IP_SQ8_FP16_GetDistFunc(...) == SQ8_FP16_InnerProduct (scalar)
```

Three dispatch entry points exercised per tier: `IP_SQ8_FP16_GetDistFunc`, `L2_SQ8_FP16_GetDistFunc`, `Cosine_SQ8_FP16_GetDistFunc`.

### Scalar-fallback tests

`GetDistFuncSQ8FP16Asymmetric` — currently asserts `dim=128` returns scalar; that assertion breaks once SIMD dispatch lands. Change to `dim=15` (below the `dim >= 16` SIMD threshold). Add a small `dim=0` (empty) scalar-fallback assertion to cover the Jira "empty" edge case.

### Dim parameterization

Base branch already has both parameterized suites against `SQ8_FP16_SpacesOptimizationTest`:
- `SQ8_FP16_SIMD` — `testing::Range(16UL, 33UL)` (dims 16..32; residual + threshold boundaries)
- `SQ8_FP16_SIMD_HighDim` — `64, 128, 256, 512, 1024` (multi-iteration main loop)

Both suites pick up the ARM tier-walk additions automatically since the test class body is what's extended. No new instantiation needed.

### Tier coverage report

`SQ8_FP16_SIMD_TierCoverage.ReportTiersExercised` (test_spaces.cpp) currently reports only x86 tiers. Extend it with ARM tier entries (SVE2 / SVE / NEON_HP) so an ARM-only SIMD host reports its exercised tiers instead of going silent.

## Microbench

`tests/benchmark/spaces_benchmarks/bm_spaces_sq8_fp16.cpp` already registers x86 ISA benchmarks. Add ARM registrations under `#ifdef OPT_*` guards using the existing `bm_spaces.h` macros:

```cpp
#ifdef CPU_FEATURES_ARCH_AARCH64
    cpu_features::Aarch64Features opt = cpu_features::GetAarch64Info().features;
    bool sve2_supported    = opt.sve2;
    bool sve_supported     = opt.sve;
    bool neon_hp_supported = opt.asimdhp;
#ifdef OPT_SVE2
    INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, SVE2, 16, sve2_supported);
    INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, SVE2, 16, sve2_supported);
#endif
#ifdef OPT_SVE
    INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, SVE, 16, sve_supported);
    INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, SVE, 16, sve_supported);
#endif
#ifdef OPT_NEON_HP
    INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, NEON_HP, 16, neon_hp_supported);
    INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, NEON_HP, 16, neon_hp_supported);
#endif
#endif // CPU_FEATURES_ARCH_AARCH64
```

Verify exact `cpu_features` helper names against the x86 sister block already in `bm_spaces_sq8_fp16.cpp` (e.g. `GetX86Info`).

`bm_spaces_sq8_fp16` and `bm_spaces_sq8_fp32` are separate executables; the per-ISA throughput comparison requested by Jira is done by running both benches and comparing matched ISA rows.

## Acceptance criteria (Jira MOD-14972 → spec mapping)

| Jira requirement | Where this spec delivers it |
|------------------|------------------------------|
| Kernels: IP + L2 for NEON | NEON_HP TU hosts kernel headers + chooser symbols |
| Kernels: IP + L2 for SVE | SVE TU hosts kernel headers + chooser symbols |
| Kernels: IP + L2 for SVE2 | SVE2 TU includes SVE headers, instantiates templates under SVE2 flags |
| Scalar fallback (reference for all platforms) | Already present in `IP.cpp` / `L2.cpp`; unchanged |
| FP16 query → FP32 per SIMD chunk | `vcvt_f32_f16` (NEON), `svcvt_f32_f16_x` (SVE) |
| FP32 metadata + correction terms | `load_unaligned<float>` for all FP32 trailer scalars |
| Wire into dispatch table per ISA flag | `IP_space.cpp` (2 blocks), `L2_space.cpp` (1 block), `OPT_SVE2/SVE/NEON_HP` |
| Unit tests vs. scalar reference per ISA | Tier-walk in `SQ8_FP16_SpacesOptimizationTest` |
| Edge cases (empty, dim-alignment boundaries) | `dim=0` + `dim=15` scalar tests; `dim=16..32` SIMD boundary param suite |
| Microbench per ISA throughput vs. SQ8↔FP32 | ARM registrations in `bm_spaces_sq8_fp16.cpp`; matched-ISA comparison vs. `bm_spaces_sq8_fp32` |

## Diff size estimate

| Area | Files | LoC (rough) |
|------|-------|-------------|
| Kernel headers | 4 new | ~600 |
| Dispatcher TU additions | NEON_HP.cpp/h, SVE.cpp/h, SVE2.cpp/h | ~80 |
| Dispatcher wiring | IP_space.cpp, L2_space.cpp | ~45 |
| Tests | test_spaces.cpp | ~80 |
| Bench | bm_spaces_sq8_fp16.cpp | ~25 |
| CMakeLists.txt | none | 0 |
| **Total** | **~10 files** | **~830** |

## PR mechanics

- **Branch**: `dor-forer-sq8-fp16-arm-kernels-mod-14972`
- **Base branch**: `dor-forer-sq8-fp16-x86-kernels-mod-14954` (PR #970)
- **PR target**: opens against PR #970 head; retarget to `main` once #970 merges
- **Commit prefix**: `[MOD-14972]` (matches repo convention)
- **PR title**: `Add SQ8↔FP16 ARM SIMD distance kernels [MOD-14972]`

## Verification gates before opening PR

1. **x86 host build clean** — verifies generic dispatch and tests remain clean; ARM kernels require ARM build or cross-compile, so the kernels themselves are not exercised here.
2. **ARM host build + unit tests** — NEON_HP / SVE / SVE2 paths exercised. Requires coordination with the user for ARM hardware or a cross-compile setup.
3. **ASan clean** on every host that runs unit tests.
4. **Microbench compiles + runs on ARM host.**

## Out of scope (deferred, separate PRs)

- Dispatcher-routed edge-case tests (`ZeroQueryTest`, `ConstantStorageTest`, `MixedSignQueryTest`) — they currently bypass the dispatcher and call scalar directly; cross-arch debt, also PR #970 H1.
- Multi-accumulator ILP tuning beyond the 4-accumulator baseline established here.
- Unrelated x86 review-feedback fixes (M1–M4, H1–H2 on x86 files from PR #970 review). This ARM PR will modify some files that PR #970 also touches (dispatchers, test class, bench), but only with ARM-relevant additions — x86 review fixes land in #970.

## Inheritance from PR #970 review findings

The following lessons from the PR #970 review are baked into this design so they do not need to be re-flagged on ARM kernels:

- `assert(dim >= 16)` at the top of every kernel template (paired with dispatcher `dim >= 16` guard).
- 4-accumulator ILP in both NEON and SVE hot loops.
- Algebraic-identity formula anchor comment at the top of each kernel header.
- `load_unaligned<float>` for all FP32 metadata reads (matches scalar).
- Dispatcher-routed tier-walk test pattern (no scalar-bypass).
- Per-ISA microbench registration alongside SQ8↔FP32 sister for direct comparison.

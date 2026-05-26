# SQ8↔FP16 SIMD distance kernels — Intel x86 (MOD-14954)

## Goal

Add asymmetric SQ8 (storage) ↔ FP16 (query) distance kernels for Inner
Product, Cosine, and L2² on Intel x86 across four ISA tiers:

- AVX-512 (F + BW + VL + VNNI bundle already used for SQ8_FP32)
- AVX2 + FMA
- AVX2 (no FMA)
- SSE4.1

Each kernel converts FP16 query lanes to FP32 per SIMD chunk; the inner
multiply-accumulate runs in FP32. SQ8 metadata and FP32 query metadata
(precomputed sums) stay scalar and are read with the same algebraic
identity used by the SQ8_FP32 kernels:

```text
IP(x, y) = min · y_sum + delta · Σ(q_i · y_i)
L2²(x, y) = x_sum_squares + y_sum_squares − 2 · IP(x, y)
```

Wire the new kernels into the dispatcher tables so
`{IP,Cosine,L2}_SQ8_FP16_GetDistFunc` returns the best SIMD path
available at runtime instead of the scalar fallback delivered by
MOD-15141.

## Non-goals

- No new metric (only IP / Cosine / L2²).
- No change to scalar `SQ8_FP16_*` reference; existing tests against
  `SQ8_FP16_NotOptimized_*` remain the correctness baseline.
- No ARM kernels (MOD-14972 covers ARM).
- No SQ8↔FP32 changes; existing kernels untouched.

## Scope and constraints

- FP16 query layout is `[float16 values (dim)] [y_sum (float)]
  [y_sum_squares (float, L2 only)]`. Trailing metadata is FP32 and may
  sit at an offset that is not a multiple of 4 when `dim` is odd; use
  `load_unaligned<float>` to read it (mirrors scalar `SQ8_FP16_Impl`).
- All four ISA tiers need a way to widen FP16 → FP32. The 512-bit
  variant (`_mm512_cvtph_ps`) is in AVX512F. The 256-bit and 128-bit
  variants (`_mm256_cvtph_ps`, `_mm_cvtph_ps`) require the F16C
  extension. F16C is its own ISA flag; AVX2/SSE4.1 do not imply it.
- Existing dispatcher source files (`AVX2_FMA.cpp`, `AVX2.cpp`,
  `SSE4.cpp`) are compiled without `-mf16c`. We add `-mf16c` to those
  files in CMake (conditional on `CXX_F16C`), guard the new SQ8_FP16
  symbols behind `#ifdef OPT_F16C`, and add `features.f16c &&` to the
  dispatch gates for the AVX2/SSE4 tiers. The AVX-512 tier needs no
  F16C gate.
- `dim` must be ≥ 16 for the AVX-512/AVX2 SIMD paths and ≥ 16 for SSE4
  (matches existing SQ8_FP32 contract).
- SQ8 storage is read as `uint8_t`; alignment hint returned by
  `*_GetDistFunc` continues to refer to the SQ8 (first) operand. Hints:
  16 / 8 / 8 / 4 bytes for AVX-512 / AVX2+FMA / AVX2 / SSE4 when
  `dim % chunk == 0`, else 0.

## File-level design

### New SIMD headers (8 files)

Per ISA tier × {IP, L2}:

```text
src/VecSim/spaces/IP/IP_AVX512F_BW_VL_VNNI_SQ8_FP16.h
src/VecSim/spaces/IP/IP_AVX2_FMA_SQ8_FP16.h
src/VecSim/spaces/IP/IP_AVX2_SQ8_FP16.h
src/VecSim/spaces/IP/IP_SSE4_SQ8_FP16.h
src/VecSim/spaces/L2/L2_AVX512F_BW_VL_VNNI_SQ8_FP16.h
src/VecSim/spaces/L2/L2_AVX2_FMA_SQ8_FP16.h
src/VecSim/spaces/L2/L2_AVX2_SQ8_FP16.h
src/VecSim/spaces/L2/L2_SSE4_SQ8_FP16.h
```

Each IP header exposes:

- `template <unsigned char residual> float SQ8_FP16_InnerProductImp_<tier>(const void*, const void*, size_t)` — raw inner product (no `1 -`), used by both InnerProduct/Cosine wrappers and the L2 kernel.
- `template <unsigned char residual> float SQ8_FP16_InnerProductSIMD16_<tier>(...)` — returns `1.0f - Imp`.
- `template <unsigned char residual> float SQ8_FP16_CosineSIMD16_<tier>(...)` — aliases InnerProduct (vectors are pre-normalised, mirrors SQ8_FP32 pattern).

Each L2 header `#include`s the matching IP header and exposes:

- `template <unsigned char residual> float SQ8_FP16_L2SqrSIMD16_<tier>(...)` — computes `x_sum_sq + y_sum_sq − 2·Imp(...)`.

`<tier>` strings:

- `AVX512F_BW_VL_VNNI`
- `AVX2_FMA`
- `AVX2`
- `SSE4`

All four headers' inner loops:

1. Load 16 SQ8 bytes (one chunk) and widen to 16×FP32.
2. Load 16 FP16 query lanes and widen to 16×FP32 (`_mm512_cvtph_ps`,
   two `_mm256_cvtph_ps` calls, two `_mm256_cvtph_ps` for plain AVX2,
   or four `_mm_cvtph_ps` for SSE4 — chunk granularity matches the
   existing SQ8_FP32 layout for that tier).
3. Fuse-multiply-add (or mul + add for SSE4 and plain AVX2) into the
   FP32 accumulator(s).
4. After the loop, horizontal-reduce and apply
   `min_val · y_sum + delta · quantized_dot`.

L2 kernels additionally read `x_sum_squares` from SQ8 metadata and
`y_sum_squares` from query metadata, return
`x_sum_sq + y_sum_sq − 2·ip`. **Both** the SQ8 storage metadata
(`min_val`, `delta`, `x_sum_squares`) and the FP16 query metadata
(`y_sum`, `y_sum_squares`) are read with `load_unaligned<float>`. SQ8
metadata starts at byte offset `dim` after the quantised lanes — for
odd `dim` that offset is not 4-byte aligned. FP16 query metadata
starts at byte offset `2*dim` after the FP16 lanes — odd `dim` leaves
it 2-byte aligned. Mirrors the scalar `SQ8_FP16_InnerProduct_Impl`
pattern in `src/VecSim/spaces/IP/IP.cpp`.

Residual handling:

- **AVX-512** (residual 0..15): load the full 256-bit FP16 chunk
  (`_mm256_loadu_si256` over 32 bytes; the chunk is always within the
  query blob since `dim >= 16` and the FP16 metadata follows), convert with
  `_mm512_cvtph_ps`, then mask away unused lanes via
  `_mm512_maskz_mov_ps(mask, v2_f)` (or fold the mask into the
  FP32 multiply with `_mm512_maskz_mul_ps`). The SQ8 side uses
  `_mm_loadu_si128` + `_mm512_cvtepu8_epi32` + `_mm512_cvtepi32_ps`
  and is also masked.
- **AVX2+FMA / AVX2** (residual 0..15, split into a 0..7 head plus a
  conditional 8-wide pre-step): for the 0..7 head, load the full
  128-bit FP16 block (`_mm_loadu_si128`), convert with
  `_mm256_cvtph_ps`, then zero out unused lanes via
  `_mm256_blend_ps(_mm256_setzero_ps(), v2_f, residuals_mask)` —
  mirroring the existing F16C `FP16_InnerProductSIMD32_F16C` blend
  pattern. The SQ8 side uses `_mm_loadl_epi64` (8 bytes) +
  `_mm256_cvtepu8_epi32` + `_mm256_cvtepi32_ps`. When residual ≥ 8,
  one extra full 8-wide step runs before the do-while loop, matching
  the SQ8_FP32 AVX2[+FMA] residual layout.
- **SSE4** (residual 0..15, split into 4-wide pre-steps): for the
  0..3 head, materialise the FP32 lanes via `_mm_set_ps(0, ..., 0,
  FP16_to_FP32(pVec2[k]), ...)` paired with `_mm_set_ps` on the SQ8
  side — mirrors the existing SSE4 SQ8_FP32 `_mm_set_ps` residual
  path. For residual ≥ 4 / ≥ 8 / ≥ 12, run 1 / 2 / 3 extra 4-wide
  steps before the do-while loop. Each 4-wide step loads 8 bytes of
  FP16 (`_mm_loadl_epi64`), converts with `_mm_cvtph_ps`, and loads
  4 SQ8 bytes via `_mm_cvtsi32_si128` + `_mm_cvtepu8_epi32` +
  `_mm_cvtepi32_ps`.

### Dispatcher edits

Per existing ISA dispatcher (no new dispatcher files):

| File | Add declarations / definitions |
| --- | --- |
| `src/VecSim/spaces/functions/AVX512F_BW_VL_VNNI.{h,cpp}` | `Choose_SQ8_FP16_{IP,Cosine,L2}_implementation_AVX512F_BW_VL_VNNI` |
| `src/VecSim/spaces/functions/AVX2_FMA.{h,cpp}` | `Choose_SQ8_FP16_{IP,Cosine,L2}_implementation_AVX2_FMA`, guarded by `#ifdef OPT_F16C` |
| `src/VecSim/spaces/functions/AVX2.{h,cpp}` | `Choose_SQ8_FP16_{IP,Cosine,L2}_implementation_AVX2`, guarded by `#ifdef OPT_F16C` |
| `src/VecSim/spaces/functions/SSE4.{h,cpp}` | `Choose_SQ8_FP16_{IP,Cosine,L2}_implementation_SSE4`, guarded by `#ifdef OPT_F16C` |

Each `Choose_*` uses the existing `CHOOSE_IMPLEMENTATION(out, dim, 16,
func)` macro (16-element residual table — matches SQ8_FP32 contract).

`src/VecSim/spaces/IP_space.cpp` — extend `IP_SQ8_FP16_GetDistFunc` and
`Cosine_SQ8_FP16_GetDistFunc`. `L2_space.cpp` — extend
`L2_SQ8_FP16_GetDistFunc`. New body shape (IP shown; L2/Cosine
identical):

```cpp
dist_func_t<float> ret_dist_func = SQ8_FP16_InnerProduct;
[[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_X86_64
if (dim < 16) {
    return ret_dist_func;
}
#ifdef OPT_AVX512_F_BW_VL_VNNI
if (features.avx512f && features.avx512bw && features.avx512vl && features.avx512vnni) {
    if (dim % 16 == 0) *alignment = 16 * sizeof(uint8_t);
    return Choose_SQ8_FP16_IP_implementation_AVX512F_BW_VL_VNNI(dim);
}
#endif
#ifdef OPT_AVX2_FMA
#ifdef OPT_F16C
if (features.avx2 && features.fma3 && features.f16c) {
    if (dim % 8 == 0) *alignment = 8 * sizeof(uint8_t);
    return Choose_SQ8_FP16_IP_implementation_AVX2_FMA(dim);
}
#endif
#endif
#ifdef OPT_AVX2
#ifdef OPT_F16C
if (features.avx2 && features.f16c) {
    if (dim % 8 == 0) *alignment = 8 * sizeof(uint8_t);
    return Choose_SQ8_FP16_IP_implementation_AVX2(dim);
}
#endif
#endif
#ifdef OPT_SSE4
#ifdef OPT_F16C
// F16C instructions are VEX-encoded — require AVX as well, matching the
// existing FP16/F16C dispatcher gate in IP_space.cpp.
if (features.sse4_1 && features.f16c && features.avx) {
    if (dim % 4 == 0) *alignment = 4 * sizeof(uint8_t);
    return Choose_SQ8_FP16_IP_implementation_SSE4(dim);
}
#endif
#endif
#endif // x86_64
return ret_dist_func;
```

ARM block (`OPT_SVE2` / `OPT_SVE` / `OPT_NEON`) is left as-is — the
SQ8_FP16 ARM kernels arrive via MOD-14972.

### CMake change

`src/VecSim/spaces/CMakeLists.txt` — when both `CXX_F16C` and the
parent ISA flag are present, add `-mf16c` to the dispatcher file:

```cmake
if(CXX_AVX2 AND CXX_FMA)
    set(_avx2_fma_flags "-mavx2 -mfma")
    if(CXX_F16C)
        set(_avx2_fma_flags "${_avx2_fma_flags} -mf16c")
    endif()
    set_source_files_properties(functions/AVX2_FMA.cpp PROPERTIES COMPILE_FLAGS "${_avx2_fma_flags}")
    list(APPEND OPTIMIZATIONS functions/AVX2_FMA.cpp)
endif()

if(CXX_AVX2)
    set(_avx2_flags "-mavx2")
    if(CXX_F16C)
        set(_avx2_flags "${_avx2_flags} -mf16c")
    endif()
    set_source_files_properties(functions/AVX2.cpp PROPERTIES COMPILE_FLAGS "${_avx2_flags}")
    list(APPEND OPTIMIZATIONS functions/AVX2.cpp)
endif()

if(CXX_SSE4)
    set(_sse4_flags "-msse4.1")
    if(CXX_F16C)
        set(_sse4_flags "${_sse4_flags} -mf16c")
    endif()
    set_source_files_properties(functions/SSE4.cpp PROPERTIES COMPILE_FLAGS "${_sse4_flags}")
    list(APPEND OPTIMIZATIONS functions/SSE4.cpp)
endif()
```

AVX-512 dispatcher (`AVX512F_BW_VL_VNNI.cpp`) needs no flag change —
`-mavx512f` already enables `_mm512_cvtph_ps`.

`-mf16c` does not alter the emitted code for the existing SQ8_FP32
sources, since those sources contain no F16C intrinsics.

### Tests (`tests/unit/test_spaces.cpp`)

1. New parameterised class `SQ8_FP16_SpacesOptimizationTest` mirroring
   `SQ8_FP32_SpacesOptimizationTest`. Three test bodies for L2 / IP /
   Cosine, each comparing the chosen optimised function against the
   scalar `SQ8_FP16_*` baseline (`ASSERT_NEAR ... 0.01`). Walks down
   AVX512 → AVX2_FMA → AVX2 → SSE4 → scalar by zeroing feature flags
   between assertions, exactly like `SQ8_FP32_SpacesOptimizationTest`.
   `INSTANTIATE_TEST_SUITE_P` with `testing::Range(16UL, 16 * 2UL + 1)`.

2. Update existing `SpacesTest.GetDistFunc_*_SQ8_FP16` assertions at
   lines ~563–575: when running on x86, the dispatcher now returns the
   SIMD `Choose_*` symbol instead of the scalar. AVX-512 selection
   depends on `avx512f && avx512bw && avx512vl && avx512vnni` only
   (no F16C requirement — 512-bit `_mm512_cvtph_ps` is part of
   AVX512F). AVX2+FMA, AVX2, and SSE4 selection additionally requires
   `features.f16c` (and `features.avx` for the SSE4 gate). The tests
   should call `getCpuOptimizationFeatures()` and assert the expected
   `Choose_*` for the host's highest supported tier (same shape used
   by `SQ8_FP32_SpacesOptimizationTest`).

3. Reuse existing helpers: `populate_sq8_fp16_query`,
   `populate_float_vec_to_sq8_with_metadata`,
   `SQ8_FP16_NotOptimized_{InnerProduct,Cosine,L2Sqr}`.

### Benchmarks (`tests/benchmark/spaces_benchmarks/bm_spaces_sq8_fp16.cpp`)

Add per-ISA benches mirroring `bm_spaces_sq8_fp32.cpp`:

```cpp
#ifdef CPU_FEATURES_ARCH_X86_64
cpu_features::X86Features opt = cpu_features::GetX86Info().features;

#ifdef OPT_AVX512_F_BW_VL_VNNI
bool avx512_f_bw_vl_vnni_supported = opt.avx512f && opt.avx512bw && opt.avx512vl && opt.avx512vnni;
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, AVX512F_BW_VL_VNNI, 16, avx512_f_bw_vl_vnni_supported);
INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, AVX512F_BW_VL_VNNI, 16, avx512_f_bw_vl_vnni_supported);
#endif

#ifdef OPT_F16C
#ifdef OPT_AVX2_FMA
bool avx2_fma3_f16c_supported = opt.avx2 && opt.fma3 && opt.f16c;
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, AVX2_FMA, 16, avx2_fma3_f16c_supported);
INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, AVX2_FMA, 16, avx2_fma3_f16c_supported);
#endif

#ifdef OPT_AVX2
bool avx2_f16c_supported = opt.avx2 && opt.f16c;
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, AVX2, 16, avx2_f16c_supported);
INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, AVX2, 16, avx2_f16c_supported);
#endif

#ifdef OPT_SSE4
bool sse4_f16c_supported = opt.sse4_1 && opt.f16c && opt.avx;
INITIALIZE_BENCHMARKS_SET_L2_IP(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, SSE4, 16, sse4_f16c_supported);
INITIALIZE_BENCHMARKS_SET_Cosine(BM_VecSimSpaces_SQ8_FP16, SQ8_FP16, SSE4, 16, sse4_f16c_supported);
#endif
#endif // OPT_F16C
#endif // x86_64
```

Naive bench lines stay (covers the scalar fallback case).

## Validation strategy

1. Unit tests (`SQ8_FP16_SpacesOptimizationTest`) assert numerical
   parity against the scalar baseline for all dims in `[16, 32]`
   (covers every residual class for the 16-wide chunk). Existing
   `SQ8_FP16_NoOpt` parameterised suite continues to exercise small
   and odd dims for the scalar reference; combined with the new
   optimisation tests this covers each SIMD residual class plus the
   scalar fallback.
2. Existing edge-case tests (`SQ8_FP16_EdgeCases.ZeroQueryTest`,
   `SQ8_FP16_l2sqr_odd_dim_unaligned_metadata_test`) keep running
   against the scalar implementation directly — they exercise
   alignment-sensitive paths that are deliberately scalar-only.
3. Microbenchmarks compare per-ISA SQ8_FP16 throughput to the matching
   SQ8_FP32 throughput on the same machine. Acceptance: SQ8_FP16
   should be within ~1.0–1.5× of SQ8_FP32 (one extra widening per
   chunk, no extra memory pressure since the FP16 query is half the
   size of FP32).
4. CI: x86 jobs already exist; verifies the CMake change keeps
   building. No new toolchain requirement (binutils 2.34+ already
   covers F16C, no AVX-512 FP16 dependency).

## Risk register

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Adding `-mf16c` to AVX2_FMA.cpp / AVX2.cpp / SSE4.cpp accidentally enables F16C codegen elsewhere | Low | Those sources contain only SQ8_FP32 / SQ8_SQ8 / INT8 / UINT8 code; no F16C intrinsics — compiler cannot synthesise F16C without an explicit intrinsic. |
| Older toolchain without F16C support | Low | `CXX_F16C` already detected; `-mf16c` only appended when present. Dispatcher symbols guarded by `#ifdef OPT_F16C`; missing → falls through to scalar. |
| Backport branches diverge in dispatcher | Medium | Change is additive (new headers, new symbols, new gates). No SQ8_FP32 path touched. CMake change is conditional. Backport just cherry-picks the commit. |
| Pre-Ivy Bridge SSE4-only CPUs lose a SIMD tier (no F16C) | Negligible | Fall through to scalar SQ8_FP16. Such CPUs are out of practical support anyway. |
| Numerical drift between FP16→FP32 widening and the scalar `FP16_to_FP32` software path | Low | `vcvtph2ps` follows IEEE 754 half→single conversion exactly; the scalar `FP16_to_FP32` in `float16.h` is bit-faithful for finite values. Tests use `ASSERT_NEAR ... 0.01` slack. |

## Out-of-scope follow-ups

- AVX512FP16-native kernels (would use `__m512h` and `vfmadd*ph`
  directly on 32 FP16 lanes per 512-bit register, skipping the
  widen-to-FP32 step). Deferred for four concrete reasons, not just
  "lower priority":
    1. **Deployment baseline.** AVX512FP16 is Sapphire Rapids and
       newer (Intel server 2023+) plus very recent AMD parts. Most
       production hosts running this library do not have it. The
       AVX-512F path delivered here is the right default for the
       widely-deployed AVX-512 tier, and a Sapphire-Rapids-only
       variant would land underneath the same gating tree, not as a
       replacement.
    2. **Numerical fit is awkward for SQ8↔FP16.** The kernel computes
       `Σ(q_i · y_i)` where `q_i ∈ [0,255]` (uint8) and `y_i` is
       FP16. Each lane product can be as large as
       `255 · 65504 ≈ 1.67e7`, which is well above the FP16 finite
       range (`±65504`). A pure FP16 accumulator would overflow on
       realistic data; the only safe path is to accumulate in FP32
       after a per-chunk `vcvtph2ps`-equivalent — which is exactly
       what the AVX-512F path already does. AVX512FP16 mainly buys
       FP16-native multiply-add, which we cannot safely use here.
    3. **Marginal speedup over the AVX-512F path proposed here.**
       The widening cost is one `_mm512_cvtph_ps` per 16-element
       chunk against a kernel that is already memory-bandwidth-bound
       (16 bytes of SQ8 storage + 32 bytes of FP16 query per chunk).
       Eliminating that one conversion saves a few cycles per chunk
       on a path that is gated on memory, not arithmetic throughput.
    4. **Ticket scope.** MOD-14954 enumerates AVX-512, AVX2+FMA, and
       SSE4; the plain-AVX2 tier was added during brainstorming as
       free coverage. An AVX512FP16 variant is its own ISA tier with
       its own gating column in the dispatcher and its own residual
       table, and warrants a separate design / benchmarking pass
       once the deployment baseline justifies the maintenance cost.
  Pure FP16↔FP16 (no SQ8 involved) already has an AVX512FP16_VL path
  at `src/VecSim/spaces/functions/AVX512FP16_VL.cpp`; that file is the
  natural home should we revisit this later.
- ARM SQ8_FP16 (MOD-14972).
- Reranking flow integration tests under HNSW (separate ticket).

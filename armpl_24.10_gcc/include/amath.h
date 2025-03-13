/*
 * ARMPL version ARMPLRELEASENUMBER Copyright ARM, NAG THEYEAR
 */

/*
  This is the AMATH header file. It contains function prototypes
  to allow a C/C++ programmer to call AMATH scalar or vector maths
  functions via their C/C++ interface.
*/

#ifndef _AMATH_H
#define _AMATH_H

#include <arm_neon.h>

// Attribute for AArch64 Vector PCS (AVPCS)
#if (__clang__ && __clang_major__ >= 7) || __GNUC__ >= 9
#define __amath_vpcs __attribute__((__aarch64_vector_pcs__))
#else
#define __amath_vpcs
#endif

#if defined(__ARM_FEATURE_SVE) &&                                                                            \
    ((__clang__ && __clang_major__ >= 5) || __GNUC__ >= 10 || defined(__NVCOMPILER))
#define _AMATH_SVE 1
#include <arm_sve.h>
#else
#define _AMATH_SVE 0
#endif

/**
 * For scalar routines the original GNU names are used.
 *
 * For vector routines the mangled name is made up as
 * `_ZGV<isa><mask><vlen><params>_<name>`. For details,
 * see the name mangling function described in
 * https://developer.arm.com/docs/101129/latest.
 *
 * Additionally, an ArmPL-specific mangling is used
 * where the mangled name is made up as
 * `armpl_v<name>q_f<prec>` for Neon routines,
 * `armpl_sv<name>_f<prec>_x` for SVE routines,
 * `armpl_<name>_f<prec>` for scalar routines,
 * `<name>` is the name of the function in libm,
 * `<prec>` is the precision expressed in bits.
 */

#ifdef __THROW
#define __amath_throw __THROW
#else
#define __amath_throw
#endif

#ifndef __cplusplus
#include <math.h>
#else
#include <cmath>

extern "C" {

#endif

// List of scalar symbols using GNU or ZGV mangling.
double acos(double) __amath_throw;
float acosf(float) __amath_throw;
double acosh(double) __amath_throw;
float acoshf(float) __amath_throw;
double asin(double) __amath_throw;
float asinf(float) __amath_throw;
double asinh(double) __amath_throw;
float asinhf(float) __amath_throw;
double atan(double) __amath_throw;
double atan2(double, double) __amath_throw;
float atan2f(float, float) __amath_throw;
float atanf(float) __amath_throw;
double atanh(double) __amath_throw;
float atanhf(float) __amath_throw;
double cbrt(double) __amath_throw;
float cbrtf(float) __amath_throw;
float cosf(float) __amath_throw;
double cosh(double) __amath_throw;
float coshf(float) __amath_throw;
double cospi(double) __amath_throw;
float cospif(float) __amath_throw;
double erf(double) __amath_throw;
double erfc(double) __amath_throw;
float erfcf(float) __amath_throw;
float erff(float) __amath_throw;
double exp(double) __amath_throw;
double exp10(double) __amath_throw;
double exp2(double) __amath_throw;
float exp2f(float) __amath_throw;
float expf(float) __amath_throw;
double expm1(double) __amath_throw;
float expm1f(float) __amath_throw;
double log(double) __amath_throw;
double log10(double) __amath_throw;
float log10f(float) __amath_throw;
double log1p(double) __amath_throw;
float log1pf(float) __amath_throw;
double log2(double) __amath_throw;
float log2f(float) __amath_throw;
float logf(float) __amath_throw;
double pow(double, double) __amath_throw;
float powf(float, float) __amath_throw;
void sincosf(float, float *, float *) __amath_throw;
void sincospi(double, double *, double *) __amath_throw;
void sincospif(float, float *, float *) __amath_throw;
float sinf(float) __amath_throw;
double sinh(double) __amath_throw;
float sinhf(float) __amath_throw;
double sinpi(double) __amath_throw;
float sinpif(float) __amath_throw;
float tanf(float) __amath_throw;
double tanh(double) __amath_throw;
float tanhf(float) __amath_throw;
// List of Neon symbols using GNU or ZGV mangling.
__amath_vpcs float64x2_t _ZGVnN2v_acos(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_acosh(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_asin(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_asinh(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_atan(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_atanh(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_cbrt(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_ceil(float64x2_t);
__amath_vpcs float64x2x2_t _ZGVnN2v_cexpi(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_cos(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_cosh(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_cospi(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_erf(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_erfc(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_exp(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_exp10(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_exp2(float64x2_t);
__amath_vpcs int64x2_t _ZGVnN2v_expfrexp(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_expm1(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_fabs(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_floor(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_frfrexp(float64x2_t);
__amath_vpcs int64x2_t _ZGVnN2v_ilogb(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_lgamma(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_log(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_log10(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_log1p(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_log2(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_rint(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_round(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_sin(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_sinh(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_sinpi(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_sqrt(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_tan(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_tanh(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_tgamma(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2v_trunc(float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vl8_modf(float64x2_t, double *);
__amath_vpcs void _ZGVnN2vl8l8_sincos(float64x2_t, double *, double *);
__amath_vpcs void _ZGVnN2vl8l8_sincospi(float64x2_t, double *, double *);
__amath_vpcs float64x2_t _ZGVnN2vv_atan2(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vv_copysign(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vv_fdim(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vv_fmax(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vv_fmin(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vv_fmod(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vv_hypot(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vv_ldexp(float64x2_t, int64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vv_nextafter(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vv_pow(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vv_remainder(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t _ZGVnN2vvv_fma(float64x2_t, float64x2_t, float64x2_t);
__amath_vpcs float32x4_t _ZGVnN4v_acosf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_acoshf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_asinf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_asinhf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_atanf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_atanhf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_cbrtf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_ceilf(float32x4_t);
__amath_vpcs float32x4x2_t _ZGVnN4v_cexpif(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_cosf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_coshf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_cospif(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_erfcf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_erff(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_exp10f(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_exp2f(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_expf(float32x4_t);
__amath_vpcs int32x4_t _ZGVnN4v_expfrexpf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_expm1f(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_fabsf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_floorf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_frfrexpf(float32x4_t);
__amath_vpcs int32x4_t _ZGVnN4v_ilogbf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_lgammaf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_log10f(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_log1pf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_log2f(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_logf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_rintf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_roundf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_sinf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_sinhf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_sinpif(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_sqrtf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_tanf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_tanhf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_tgammaf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4v_truncf(float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vl4_modff(float32x4_t, float *);
__amath_vpcs void _ZGVnN4vl4l4_sincosf(float32x4_t, float *, float *);
__amath_vpcs void _ZGVnN4vl4l4_sincospif(float32x4_t, float *, float *);
__amath_vpcs float32x4_t _ZGVnN4vv_atan2f(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vv_copysignf(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vv_fdimf(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vv_fmaxf(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vv_fminf(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vv_fmodf(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vv_hypotf(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vv_ldexpf(float32x4_t, int32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vv_nextafterf(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vv_powf(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vv_remainderf(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t _ZGVnN4vvv_fmaf(float32x4_t, float32x4_t, float32x4_t);
// List of SVE symbols using GNU or ZGV mangling.
#if _AMATH_SVE
svfloat64_t _ZGVsMxv_acos(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_acosf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_acosh(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_acoshf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_asin(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_asinf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_asinh(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_asinhf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_atan(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_atanf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_atanh(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_atanhf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_cbrt(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_cbrtf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_ceil(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_ceilf(svfloat32_t, svbool_t);
svfloat64x2_t _ZGVsMxv_cexpi(svfloat64_t, svbool_t);
svfloat32x2_t _ZGVsMxv_cexpif(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_cos(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_cosf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_cosh(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_coshf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_cospi(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_cospif(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_erf(svfloat64_t, svbool_t);
svfloat64_t _ZGVsMxv_erfc(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_erfcf(svfloat32_t, svbool_t);
svfloat32_t _ZGVsMxv_erff(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_exp(svfloat64_t, svbool_t);
svfloat64_t _ZGVsMxv_exp10(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_exp10f(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_exp2(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_exp2f(svfloat32_t, svbool_t);
svfloat32_t _ZGVsMxv_expf(svfloat32_t, svbool_t);
svint64_t _ZGVsMxv_expfrexp(svfloat64_t, svbool_t);
svint32_t _ZGVsMxv_expfrexpf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_expm1(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_expm1f(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_fabs(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_fabsf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_floor(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_floorf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_frfrexp(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_frfrexpf(svfloat32_t, svbool_t);
svint64_t _ZGVsMxv_ilogb(svfloat64_t, svbool_t);
svint32_t _ZGVsMxv_ilogbf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_lgamma(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_lgammaf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_log(svfloat64_t, svbool_t);
svfloat64_t _ZGVsMxv_log10(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_log10f(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_log1p(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_log1pf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_log2(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_log2f(svfloat32_t, svbool_t);
svfloat32_t _ZGVsMxv_logf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_rint(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_rintf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_round(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_roundf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_sin(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_sinf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_sinh(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_sinhf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_sinpi(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_sinpif(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_sqrt(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_sqrtf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_tan(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_tanf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_tanh(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_tanhf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_tgamma(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_tgammaf(svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxv_trunc(svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxv_truncf(svfloat32_t, svbool_t);
svfloat32_t _ZGVsMxvl4_modff(svfloat32_t, float *, svbool_t);
void _ZGVsMxvl4l4_sincosf(svfloat32_t, float *, float *, svbool_t);
void _ZGVsMxvl4l4_sincospif(svfloat32_t, float *, float *, svbool_t);
svfloat64_t _ZGVsMxvl8_modf(svfloat64_t, double *, svbool_t);
void _ZGVsMxvl8l8_sincos(svfloat64_t, double *, double *, svbool_t);
void _ZGVsMxvl8l8_sincospi(svfloat64_t, double *, double *, svbool_t);
svfloat64_t _ZGVsMxvv_atan2(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvv_atan2f(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxvv_copysign(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvv_copysignf(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxvv_fdim(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvv_fdimf(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxvv_fmax(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvv_fmaxf(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxvv_fmin(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvv_fminf(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxvv_fmod(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvv_fmodf(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxvv_hypot(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvv_hypotf(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxvv_ldexp(svfloat64_t, svint64_t, svbool_t);
svfloat32_t _ZGVsMxvv_ldexpf(svfloat32_t, svint32_t, svbool_t);
svfloat64_t _ZGVsMxvv_nextafter(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvv_nextafterf(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxvv_pow(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvv_powf(svfloat32_t, svfloat32_t, svbool_t);
svfloat32_t _ZGVsMxvv_powi(svfloat32_t, svint32_t, svbool_t);
svfloat64_t _ZGVsMxvv_powk(svfloat64_t, svint64_t, svbool_t);
svfloat64_t _ZGVsMxvv_remainder(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvv_remainderf(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t _ZGVsMxvvv_fma(svfloat64_t, svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t _ZGVsMxvvv_fmaf(svfloat32_t, svfloat32_t, svfloat32_t, svbool_t);
#endif
// List of special symbols using ArmPL mangling.
void armpl_get_libamath_identity();
void print_lib_details();
// List of scalar symbols using ArmPL mangling.
double armpl_acos_f64(double);
float armpl_acos_f32(float);
double armpl_acosh_f64(double);
float armpl_acosh_f32(float);
double armpl_asin_f64(double);
float armpl_asin_f32(float);
double armpl_asinh_f64(double);
float armpl_asinh_f32(float);
double armpl_atan_f64(double);
double armpl_atan2_f64(double, double);
float armpl_atan2_f32(float, float);
float armpl_atan_f32(float);
double armpl_atanh_f64(double);
float armpl_atanh_f32(float);
double armpl_cbrt_f64(double);
float armpl_cbrt_f32(float);
float armpl_cos_f32(float);
double armpl_cosh_f64(double);
float armpl_cosh_f32(float);
double armpl_cospi_f64(double);
float armpl_cospi_f32(float);
double armpl_erf_f64(double);
double armpl_erfc_f64(double);
float armpl_erfc_f32(float);
float armpl_erf_f32(float);
double armpl_exp_f64(double);
double armpl_exp10_f64(double);
double armpl_exp2_f64(double);
float armpl_exp2_f32(float);
float armpl_exp_f32(float);
double armpl_expm1_f64(double);
float armpl_expm1_f32(float);
double armpl_log_f64(double);
double armpl_log10_f64(double);
float armpl_log10_f32(float);
double armpl_log1p_f64(double);
float armpl_log1p_f32(float);
double armpl_log2_f64(double);
float armpl_log2_f32(float);
float armpl_log_f32(float);
double armpl_pow_f64(double, double);
float armpl_pow_f32(float, float);
void armpl_sincos_f32(float, float *, float *);
void armpl_sincospi_f64(double, double *, double *);
void armpl_sincospi_f32(float, float *, float *);
float armpl_sin_f32(float);
double armpl_sinh_f64(double);
float armpl_sinh_f32(float);
double armpl_sinpi_f64(double);
float armpl_sinpi_f32(float);
float armpl_tan_f32(float);
double armpl_tanh_f64(double);
float armpl_tanh_f32(float);
// List of Neon symbols using ArmPL mangling.
__amath_vpcs float64x2_t armpl_vacosq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vacoshq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vasinq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vasinhq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vatanq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vatanhq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vcbrtq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vceilq_f64(float64x2_t);
__amath_vpcs float64x2x2_t armpl_vcexpiq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vcosq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vcoshq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vcospiq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_verfq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_verfcq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vexpq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vexp10q_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vexp2q_f64(float64x2_t);
__amath_vpcs int64x2_t armpl_vexpfrexpq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vexpm1q_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vfabsq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vfloorq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vfrfrexpq_f64(float64x2_t);
__amath_vpcs int64x2_t armpl_vilogbq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vlgammaq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vlogq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vlog10q_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vlog1pq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vlog2q_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vrintq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vroundq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vsinq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vsinhq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vsinpiq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vsqrtq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vtanq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vtanhq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vtgammaq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vtruncq_f64(float64x2_t);
__amath_vpcs float64x2_t armpl_vmodfq_f64(float64x2_t, double *);
__amath_vpcs void armpl_vsincosq_f64(float64x2_t, double *, double *);
__amath_vpcs void armpl_vsincospiq_f64(float64x2_t, double *, double *);
__amath_vpcs float64x2_t armpl_vatan2q_f64(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t armpl_vcopysignq_f64(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t armpl_vfdimq_f64(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t armpl_vfmaxq_f64(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t armpl_vfminq_f64(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t armpl_vfmodq_f64(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t armpl_vhypotq_f64(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t armpl_vldexpq_f64(float64x2_t, int64x2_t);
__amath_vpcs float64x2_t armpl_vnextafterq_f64(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t armpl_vpowq_f64(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t armpl_vremainderq_f64(float64x2_t, float64x2_t);
__amath_vpcs float64x2_t armpl_vfmaq_f64(float64x2_t, float64x2_t, float64x2_t);
__amath_vpcs float32x4_t armpl_vacosq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vacoshq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vasinq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vasinhq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vatanq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vatanhq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vcbrtq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vceilq_f32(float32x4_t);
__amath_vpcs float32x4x2_t armpl_vcexpiq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vcosq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vcoshq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vcospiq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_verfcq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_verfq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vexp10q_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vexp2q_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vexpq_f32(float32x4_t);
__amath_vpcs int32x4_t armpl_vexpfrexpq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vexpm1q_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vfabsq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vfloorq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vfrfrexpq_f32(float32x4_t);
__amath_vpcs int32x4_t armpl_vilogbq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vlgammaq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vlog10q_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vlog1pq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vlog2q_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vlogq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vrintq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vroundq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vsinq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vsinhq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vsinpiq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vsqrtq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vtanq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vtanhq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vtgammaq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vtruncq_f32(float32x4_t);
__amath_vpcs float32x4_t armpl_vmodfq_f32(float32x4_t, float *);
__amath_vpcs void armpl_vsincosq_f32(float32x4_t, float *, float *);
__amath_vpcs void armpl_vsincospiq_f32(float32x4_t, float *, float *);
__amath_vpcs float32x4_t armpl_vatan2q_f32(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t armpl_vcopysignq_f32(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t armpl_vfdimq_f32(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t armpl_vfmaxq_f32(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t armpl_vfminq_f32(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t armpl_vfmodq_f32(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t armpl_vhypotq_f32(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t armpl_vldexpq_f32(float32x4_t, int32x4_t);
__amath_vpcs float32x4_t armpl_vnextafterq_f32(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t armpl_vpowq_f32(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t armpl_vremainderq_f32(float32x4_t, float32x4_t);
__amath_vpcs float32x4_t armpl_vfmaq_f32(float32x4_t, float32x4_t, float32x4_t);
// List of SVE symbols using ArmPL mangling.
#if _AMATH_SVE
svfloat64_t armpl_svacos_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svacos_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svacosh_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svacosh_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svasin_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svasin_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svasinh_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svasinh_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svatan_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svatan_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svatanh_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svatanh_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svcbrt_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svcbrt_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svceil_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svceil_f32_x(svfloat32_t, svbool_t);
svfloat64x2_t armpl_svcexpi_f64_x(svfloat64_t, svbool_t);
svfloat32x2_t armpl_svcexpi_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svcos_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svcos_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svcosh_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svcosh_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svcospi_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svcospi_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_sverf_f64_x(svfloat64_t, svbool_t);
svfloat64_t armpl_sverfc_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_sverfc_f32_x(svfloat32_t, svbool_t);
svfloat32_t armpl_sverf_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svexp_f64_x(svfloat64_t, svbool_t);
svfloat64_t armpl_svexp10_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svexp10_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svexp2_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svexp2_f32_x(svfloat32_t, svbool_t);
svfloat32_t armpl_svexp_f32_x(svfloat32_t, svbool_t);
svint64_t armpl_svexpfrexp_f64_x(svfloat64_t, svbool_t);
svint32_t armpl_svexpfrexp_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svexpm1_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svexpm1_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svfabs_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svfabs_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svfloor_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svfloor_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svfrfrexp_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svfrfrexp_f32_x(svfloat32_t, svbool_t);
svint64_t armpl_svilogb_f64_x(svfloat64_t, svbool_t);
svint32_t armpl_svilogb_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svlgamma_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svlgamma_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svlog_f64_x(svfloat64_t, svbool_t);
svfloat64_t armpl_svlog10_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svlog10_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svlog1p_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svlog1p_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svlog2_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svlog2_f32_x(svfloat32_t, svbool_t);
svfloat32_t armpl_svlog_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svrint_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svrint_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svround_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svround_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svsin_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svsin_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svsinh_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svsinh_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svsinpi_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svsinpi_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svsqrt_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svsqrt_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svtan_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svtan_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svtanh_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svtanh_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svtgamma_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svtgamma_f32_x(svfloat32_t, svbool_t);
svfloat64_t armpl_svtrunc_f64_x(svfloat64_t, svbool_t);
svfloat32_t armpl_svtrunc_f32_x(svfloat32_t, svbool_t);
svfloat32_t armpl_svmodf_f32_x(svfloat32_t, float *, svbool_t);
void armpl_svsincos_f32_x(svfloat32_t, float *, float *, svbool_t);
void armpl_svsincospi_f32_x(svfloat32_t, float *, float *, svbool_t);
svfloat64_t armpl_svmodf_f64_x(svfloat64_t, double *, svbool_t);
void armpl_svsincos_f64_x(svfloat64_t, double *, double *, svbool_t);
void armpl_svsincospi_f64_x(svfloat64_t, double *, double *, svbool_t);
svfloat64_t armpl_svatan2_f64_x(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svatan2_f32_x(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t armpl_svcopysign_f64_x(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svcopysign_f32_x(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t armpl_svfdim_f64_x(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svfdim_f32_x(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t armpl_svfmax_f64_x(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svfmax_f32_x(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t armpl_svfmin_f64_x(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svfmin_f32_x(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t armpl_svfmod_f64_x(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svfmod_f32_x(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t armpl_svhypot_f64_x(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svhypot_f32_x(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t armpl_svldexp_f64_x(svfloat64_t, svint64_t, svbool_t);
svfloat32_t armpl_svldexp_f32_x(svfloat32_t, svint32_t, svbool_t);
svfloat64_t armpl_svnextafter_f64_x(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svnextafter_f32_x(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t armpl_svpow_f64_x(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svpow_f32_x(svfloat32_t, svfloat32_t, svbool_t);
svfloat32_t armpl_svpowi_f32_x(svfloat32_t, svint32_t, svbool_t);
svfloat64_t armpl_svpowk_f64_x(svfloat64_t, svint64_t, svbool_t);
svfloat64_t armpl_svremainder_f64_x(svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svremainder_f32_x(svfloat32_t, svfloat32_t, svbool_t);
svfloat64_t armpl_svfma_f64_x(svfloat64_t, svfloat64_t, svfloat64_t, svbool_t);
svfloat32_t armpl_svfma_f32_x(svfloat32_t, svfloat32_t, svfloat32_t, svbool_t);
#endif

#ifdef __cplusplus

} // extern "C"

#endif

#endif // _AMATH_H

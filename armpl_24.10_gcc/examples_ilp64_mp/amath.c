/*
 * amath - Elementary mathematical routines
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include "round_eps_to_zero.h"
#include <amath.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

/* In order to enable SVE, please compile this example with
   CFLAGS="-march=armv8-a+sve" enabled.
//LIN
   Make sure to link to libamath and libm by adding "-lamath -lm" to CLINKLIBS.
//LIN
   For versions of glibc older than glibc-2.34, lazy binding needs to be
   disabled by setting `LD_BIND_NOW=1` at runtime. Alternatively, the
   executable can be linked statically CLINK="$(CC) -static".
*/
int main(void) {

	// Scalar exp in single and double precision
	double x = 0.0;
	float xf = 1.0f;
	double e = exp(x);
	float ef = expf(xf);

	// Double precision Neon sin and cos
	double pc[2], ps[2], px[2];
	px[0] = 0.0;
	px[1] = M_PI / 2.0;
	// Neon cos and sin using sincos on Linux
	float64x2_t vx = vld1q_f64 (px);
	_ZGVnN2vl8l8_sincos(vx, ps, pc);

#if defined(__ARM_FEATURE_SVE)
	// Unary and binary SVE math routines
	// single precision pow
	svbool_t pg32 = svptrue_b32 ();
	svfloat32_t svx = svdup_n_f32(2.0f);
	svfloat32_t svy = svdup_n_f32(3.0f);
	svfloat32_t svz = _ZGVsMxvv_powf(svx, svy, pg32);
	// double precision erf
	svbool_t pg64 = svptrue_b64();
	svfloat64_t svw = svdup_n_f64(20.0);
	svfloat64_t sve = _ZGVsMxv_erf(svw, pg64);
	// Compute a scalar quantity to ensure result does not depend on vector length
	float err_powf = round_eps_to_zero_f(svaddv_f32(pg32, svsub_n_f32_x(pg32, svz, 8.0f)));
	double err_erf = round_eps_to_zero_d(svaddv_f64(pg64, svsub_n_f64_x(pg64, sve, 1.0)));
#else
	// Scalar math routines
	float err_powf = 0.0f;
	double err_erf = 0.0;
#endif

	printf("AMath example: Elementary mathematical routines\n");
	printf("-----------------------------------------------\n\n");

	printf("  exp(%.2e) = %.2e\n", x, e);
	printf("  expf(%.2e) = %.2e\n", xf, ef);
	printf("  sin({%.2e, %.2e}) = {%.2e, %.2e}\n", px[0], px[1], round_eps_to_zero_d(ps[0]), ps[1]);
	printf("  cos({%.2e, %.2e}) = {%.2e, %.2e}\n", px[0], px[1], pc[0], round_eps_to_zero_d(pc[1]));
	printf("  powf(2.0f, 3.0f) - 8.0f = %.2e\n", err_powf);
	printf("  erf(20.0) - 1.0 = %.2e\n", err_erf);

	return 0;
}


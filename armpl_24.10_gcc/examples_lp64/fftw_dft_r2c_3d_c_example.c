/*
 * fftw_dft_r2c_3d: FFT of a 3d real sequence
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>

#include "round_eps_to_zero.h"

int main(void) {
#define MMAX 10
#define NMAX 10
#define LMAX 10
	double xx[MMAX * NMAX * LMAX];
	double x[MMAX * NMAX * LMAX];
	// The output vector is of size m * n * (l/2+1) as it is Hermitian
	fftw_complex y[MMAX * NMAX * (LMAX / 2 + 1)];

	printf("ARMPL example: FFT of a 3d real sequence using "
	       "fftw_plan_dft_r2c_3d\n");
	printf("-------------------------------------------------------------------"
	       "\n");
	printf("\n");

	/* The 3d sequence of double data */
	int m = 2;
	int n = 2;
	int l = 5;
	int ix = 0;
	int i, j, k;
	for (i = 1; i <= m; i++) {
		for (j = 1; j <= n; j++) {
			for (k = 1; k <= l; k++) {
				x[ix] = 1.0 + cos(ix + 1);
				xx[ix] = x[ix];
				ix++;
			}
		}
	}

	fftw_plan f = fftw_plan_dft_r2c_3d(m, n, l, x, y, FFTW_ESTIMATE);
	fftw_plan b = fftw_plan_dft_c2r_3d(m, n, l, y, x, FFTW_ESTIMATE);
	fftw_execute(f);
	fftw_destroy_plan(f);

	printf("Components of discrete Fourier transform:\n");
	printf("\n");
	ix = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			for (k = 0; k <= l / 2; k++) {
				double scale = 1.0 / sqrt(m * n * l);
				double y_real = round_eps_to_zero_d(creal(y[ix]) * scale);
				double y_imag = round_eps_to_zero_d(cimag(y[ix]) * scale);
				printf("%4d,%d,%d   (%7.4f+%7.4fi)\n", i + 1, j + 1, k + 1, y_real, y_imag);
				ix++;
			}
		}
	}
	fftw_execute(b);
	fftw_destroy_plan(b);

	printf("\n");
	printf("Original sequence as restored by inverse transform:\n");
	printf("\n");
	printf("       Original  Restored\n");
	ix = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			for (k = 0; k < l; k++) {
				double scale = 1.0 / (m * n * l);
				double xx_ix = round_eps_to_zero_d(xx[ix]);
				double x_ix = round_eps_to_zero_d(x[ix] * scale);
				printf("%4d,%d,%d   %7.4f   %7.4f\n", i + 1, j + 1, k + 1, xx_ix, x_ix);
				ix++;
			}
		}
	}

	return 0;
}

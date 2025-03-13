/*
 * fftw_dft_r2c_2d: FFT of a 2d real sequence
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
#define MMAX 20
#define NMAX 20
	double xx[MMAX * NMAX];
	double x[MMAX * NMAX];
	// The output vector is of size m * (n/2+1) as it is Hermitian
	fftw_complex y[MMAX * (NMAX / 2) + 1];

	printf("ARMPL example: FFT of a 2d real sequence using "
	       "fftw_plan_dft_r2c_2d\n");
	printf("-------------------------------------------------------------------"
	       "\n");
	printf("\n");

	/* The 2d sequence of double data */
	int m = 5;
	int n = 2;
	int ix = 0;
	int i, j;
	for (i = 1; i <= m; i++) {
		for (j = 1; j <= n; j++) {
			x[ix] = 1.0 + cos(j + 2 * i);
			// Copy the values into another array to preserve input
			xx[ix] = x[ix];
			ix++;
		}
	}

	// Initialise a plan for a real-to-complex 2d transform from x->y
	fftw_plan f = fftw_plan_dft_r2c_2d(m, n, x, y, FFTW_ESTIMATE);
	// Initialise a plan for a complex-to-real 2d transform from y->x (inverse)
	fftw_plan b = fftw_plan_dft_c2r_2d(m, n, y, x, FFTW_ESTIMATE);

	// Execute the forward plan and then deallocate the plan
	/* NOTE: FFTW does NOT compute a normalised transform -
	 * returned array will contain unscaled values */
	fftw_execute(f);
	fftw_destroy_plan(f);

	printf("Components of discrete Fourier transform:\n");
	printf("\n");
	ix = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j <= n / 2; j++) {
			// Scale factor of 1/sqrt(m*n) to output normalised data
			double scale = 1.0 / sqrt(m * n);
			double y_real = round_eps_to_zero_d(creal(y[ix]) * scale);
			double y_imag = round_eps_to_zero_d(cimag(y[ix]) * scale);
			printf("%4d,%d   (%7.4f+%7.4fi)\n", i + 1, j + 1, y_real, y_imag);
			ix++;
		}
	}

	// Execute the reverse plan and then deallocate the plan
	/* NOTE: FFTW does NOT compute a normalised transform -
	 * returned array will contain unscaled values */
	fftw_execute(b);
	fftw_destroy_plan(b);

	printf("\n");
	printf("Original sequence as restored by inverse transform:\n");
	printf("\n");
	printf("       Original  Restored\n");
	ix = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			// Scale factor of 1/m*n to output normalised data
			double scale = 1.0 / (m * n);
			double xx_ix = round_eps_to_zero_d(xx[ix]);
			double x_ix = round_eps_to_zero_d(x[ix] * scale);
			printf("%4d,%d   %7.4f   %7.4f\n", i + 1, j + 1, xx_ix, x_ix);
			ix++;
		}
	}

	return 0;
}

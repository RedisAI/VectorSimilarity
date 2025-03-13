/*
 * fftwf_dft_r2c_1d: FFT of a single-precision real sequence
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <armpl.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>

#include "round_eps_to_zero.h"

int main(void) {
#define NMAX 20
	float xx[NMAX];
	float x[NMAX];
	// The output vector is of size (n/2)+1 as it is Hermitian
	fftwf_complex y[NMAX / 2 + 1];

	printf("ARMPL example: FFT of a single-precision real sequence using "
	       "fftwf_plan_dft_r2c_1d\n");
	printf("-------------------------------------------------------------------"
	       "---------------\n");
	printf("\n");

	/* The sequence of single-precision data */
	int n = 7;
	x[0] = 0.34907;
	x[1] = 0.54890;
	x[2] = 0.74776;
	x[3] = 0.94459;
	x[4] = 1.13850;
	x[5] = 1.32850;
	x[6] = 1.51370;

	// Use scopy to copy the values into another array (preserve input)
	cblas_scopy(n, x, 1, xx, 1);

	// Initialise a plan for a real-to-complex 1d transform from x->y
	fftwf_plan forward_plan = fftwf_plan_dft_r2c_1d(n, x, y, FFTW_ESTIMATE);
	// Initialise a plan for a complex-to-real 1d transform from y->x (inverse)
	fftwf_plan inverse_plan = fftwf_plan_dft_c2r_1d(n, y, x, FFTW_ESTIMATE);

	// Execute the forward plan and then deallocate the plan
	/* NOTE: fftwf does NOT compute a normalised transform -
	 * returned array will contain unscaled values */
	fftwf_execute(forward_plan);
	fftwf_destroy_plan(forward_plan);

	printf("Components of discrete Fourier transform:\n");
	printf("\n");
	int j;
	for (j = 0; j <= n / 2; j++) {
		// Scale factor of 1/sqrt(n) to output normalised data
		float y_real = round_eps_to_zero_f(creal(y[j]) / sqrt(n));
		float y_imag = round_eps_to_zero_f(cimag(y[j]) / sqrt(n));
		printf("%4d   (%7.4f%7.4f)\n", j + 1, y_real, y_imag);
	}

	// Execute the reverse plan and then deallocate the plan
	/* NOTE: fftwf does NOT compute a normalised transform -
	 * returned array will contain unscaled values */
	fftwf_execute(inverse_plan);
	fftwf_destroy_plan(inverse_plan);

	printf("\n");
	printf("Original sequence as restored by inverse transform:\n");
	printf("\n");
	printf("       Original  Restored\n");
	for (j = 0; j < n; j++) {
		float xx_j = round_eps_to_zero_f(xx[j]);
		// Scale factor of 1/n to output normalised data
		float x_j = round_eps_to_zero_f(x[j] / n);
		printf("%4d   %7.4f   %7.4f\n", j + 1, xx_j, x_j);
	}

	return 0;
}

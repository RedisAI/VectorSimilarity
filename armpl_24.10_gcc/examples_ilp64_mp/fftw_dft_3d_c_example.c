/*
 * fftw_dft_3d: FFT of a 3d complex sequence
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
	fftw_complex x[MMAX * NMAX * LMAX];
	fftw_complex xx[MMAX * NMAX * LMAX];

	printf("ARMPL example: FFT of a complex sequence using fftw_plan_dft_3d\n");
	printf("---------------------------------------------------------------\n");
	printf("\n");

	/* The 3d sequence of complex data */
	int m = 2;
	int n = 2;
	int l = 5;
	int ix = 0;

	int i, j, k;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			for (k = 0; k < l; k++) {
				x[ix] = 1 + cos(ix + 1) + I * sin(ix + 1);
				// Copy the values into another array to preserve input
				xx[ix] = x[ix];
				ix++;
			}
		}
	}

	// Initialise a plan for an in-place complex-to-complex 3d forward transform
	// (x->x')
	fftw_plan forward = fftw_plan_dft_3d(m, n, l, x, x, FFTW_FORWARD, FFTW_ESTIMATE);
	// Initialise a plan for an in-place complex-to-complex 3d backward
	// transform (x'->x)
	fftw_plan reverse = fftw_plan_dft_3d(m, n, l, x, x, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the forward plan and then deallocate the plan
	fftw_execute(forward);
	fftw_destroy_plan(forward);

	printf("Components of discrete Fourier transform:\n");
	printf("\n");
	printf("          Real    Imag\n");
	ix = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			for (k = 0; k < l; k++) {
				double x_real = round_eps_to_zero_d(creal(x[ix]));
				double x_imag = round_eps_to_zero_d(cimag(x[ix]));
				printf("%4d,%d,%d   (%7.4f,%7.4f)\n", i + 1, j + 1, k + 1, x_real, x_imag);
				ix++;
			}
		}
	}

	// Execute the reverse plan and then deallocate the plan
	/* NOTE: FFTW does NOT compute a normalised transform -
	 * returned array will contain unscaled values */
	fftw_execute(reverse);
	fftw_destroy_plan(reverse);

	printf("\n");
	printf("Original sequence as restored by inverse transform:\n");
	printf("\n");
	printf("            Original            Restored\n");
	printf("          Real    Imag        Real    Imag\n");
	ix = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			for (k = 0; k < l; k++) {
				double xx_real = round_eps_to_zero_d(creal(xx[ix]));
				double xx_imag = round_eps_to_zero_d(cimag(xx[ix]));
				// Scale factor of 1/m*n*k to output normalised data
				double scale = 1.0 / (m * n * l);
				double x_real = round_eps_to_zero_d(creal(x[ix]) * scale);
				double x_imag = round_eps_to_zero_d(cimag(x[ix]) * scale);

				printf("%4d,%d,%d   (%7.4f,%7.4f)   (%7.4f,%7.4f)\n", i + 1, j + 1, k + 1, xx_real, xx_imag,
				       x_real, x_imag);
				ix++;
			}
		}
	}
	return 0;
}

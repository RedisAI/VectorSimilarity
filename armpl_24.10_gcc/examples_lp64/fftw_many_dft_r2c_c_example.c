/*
 * fftw_many_dft_r2c: This example program fills an array with two sequences of
 * real numbers, then uses fftw_plan_many_dft_r2c to do a batch real-to-complex
 * fourier transform on them.
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>

#include "round_eps_to_zero.h"

#define HOWMANY 2
#define N 5
#define ISTRIDE 2
#define OSTRIDE 3
#define IDIST N * ISTRIDE
#define ODIST N * OSTRIDE
#define ISIZE HOWMANY * IDIST
#define OSIZE HOWMANY * ODIST

int main(void) {
	// We are computing 2 sequences of 5 elements each in one plan
	int howmany = HOWMANY;
	int n = N;

	// istride and ostride are the steps between elements in respective input
	// and output arrays
	int istride = ISTRIDE;
	int ostride = OSTRIDE;
	// idist and odist are the steps between the first element of one sequence
	// to the first element of the next
	int idist = IDIST;
	int odist = ODIST;

	double x[ISIZE];
	double xx[ISIZE];
	fftw_complex y[OSIZE / 2 + ODIST];

	printf("ARMPL example: FFT of many real sequences using "
	       "fftw_plan_many_dft_r2c\n");
	printf("-------------------------------------------------------------------"
	       "---\n");
	printf("\n");

	int i, j, ix;
	for (i = 0; i < howmany; i++) {
		/* A sequence of real double data */
		for (j = 0; j < n; j++) {
			ix = j * istride + i * idist;
			x[ix] = 1 + cos(i + j * istride + 2);
			// Copy the values into another array to preserve input
			xx[ix] = x[ix];
		}
	}

	// Initialise a plan for a real-to-complex transform from x->y on howmany
	// sequences
	fftw_plan forward = fftw_plan_many_dft_r2c(1, &n, howmany, x, NULL, istride, idist, y, NULL, ostride,
	                                           odist, FFTW_ESTIMATE);
	// Initialise a plan for a reverse complex-to-real transform from y->x
	// (inverse) on howmany sequences
	fftw_plan reverse = fftw_plan_many_dft_c2r(1, &n, howmany, y, NULL, ostride, odist, x, NULL, istride,
	                                           idist, FFTW_ESTIMATE);

	// Execute the forward transform and deallocate the plan
	/* NOTE: FFTW does NOT compute a normalised transform -
	 * returned array will contain unscaled values */
	fftw_execute(forward);
	fftw_destroy_plan(forward);

	for (i = 0; i < howmany; i++) {
		printf("Components of discrete Fourier transform for sequence %d:\n", i + 1);
		printf("\n");
		printf("          Real    Imag\n");
		for (j = 0; j <= n / 2; j++) {
			int index = j * ostride + i * odist;
			double y_real = round_eps_to_zero_d(creal(y[index]) / sqrt(n));
			double y_imag = round_eps_to_zero_d(cimag(y[index]) / sqrt(n));
			// Scale by 1/sqrt(N) to output normalised values
			printf("%4d   (%7.4f,%7.4f)\n", index, y_real, y_imag);
		}
	}
	// Execute the reverse transform and deallocate the plan
	/* NOTE: FFTW does NOT compute a normalised transform -
	 * returned array will contain unscaled values */
	fftw_execute(reverse);
	fftw_destroy_plan(reverse);

	for (i = 0; i < howmany; i++) {
		printf("\n");
		printf("Original sequence %d as restored by inverse transform:\n", i + 1);
		printf("\n");
		printf("       Original  Restored\n");
		for (j = 0; j < n; j++) {
			int index = j * istride + i * idist;
			double xx_i = round_eps_to_zero_d(xx[index]);
			// Scale by 1/N to output normalised values
			double x_i = round_eps_to_zero_d(x[index] / n);
			printf("%4d   %7.4f   %7.4f\n", index, xx_i, x_i);
		}
	}

	return 0;
}

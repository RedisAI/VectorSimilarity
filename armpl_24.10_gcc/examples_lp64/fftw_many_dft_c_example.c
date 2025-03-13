/*
 * fftw_many_dft: This example program fills an array with two sequences of
 * complex numbers, then uses fftw_plan_many_dft to do a batch discrete fourier
 * transform on them.
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

	fftw_complex x[ISIZE];
	fftw_complex xx[ISIZE];
	fftw_complex y[OSIZE];

	printf("ARMPL example: FFT of many complex sequences using "
	       "fftw_plan_many_dft\n");
	printf("-------------------------------------------------------------------"
	       "--\n");
	printf("\n");

	int i, j, ix;
	for (i = 0; i < howmany; i++) {
		/* A sequence of complex double data */
		for (j = 0; j < n; j++) {
			ix = j * istride + i * idist;
			x[ix] = 1 + cos(i + j * istride + 2) + sin(i - j * istride) * I;
			// Copy the values into another array to preserve values
			xx[ix] = x[ix];
		}
	}

	// Initialise a plan for a forward complex-to-complex transform from x->y on
	// howmany sequences
	fftw_plan forward = fftw_plan_many_dft(1, &n, howmany, x, NULL, istride, idist, y, NULL, ostride, odist,
	                                       FFTW_FORWARD, FFTW_ESTIMATE);
	// Initialise a plan for a reverse complex-to-complex transform from y->x
	// (inverse) on howmany sequences
	fftw_plan reverse = fftw_plan_many_dft(1, &n, howmany, y, NULL, ostride, odist, x, NULL, istride, idist,
	                                       FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the forward transform and deallocate the plan
	fftw_execute(forward);
	fftw_destroy_plan(forward);

	for (i = 0; i < howmany; i++) {
		printf("Components of discrete Fourier transform for sequence %d:\n", i + 1);
		printf("\n");
		printf("          Real    Imag\n");
		for (j = 0; j < n; j++) {
			int index = j * ostride + i * odist;
			double y_real = round_eps_to_zero_d(creal(y[index]));
			double y_imag = round_eps_to_zero_d(cimag(y[index]));
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
		printf("            Original            Restored\n");
		printf("          Real    Imag        Real    Imag\n");
		for (j = 0; j < n; j++) {
			int index = j * istride + i * idist;
			double xx_real = round_eps_to_zero_d(creal(xx[index]));
			double xx_imag = round_eps_to_zero_d(cimag(xx[index]));
			// Scale by 1/N to display normalised values
			double x_real = round_eps_to_zero_d(creal(x[index]) / n);
			double x_imag = round_eps_to_zero_d(cimag(x[index]) / n);
			printf("%4d   (%7.4f,%7.4f)   (%7.4f,%7.4f)\n", index, xx_real, xx_imag, x_real, x_imag);
		}
	}

	return 0;
}

/*
 * fftwf_dft_1d: FFT of a single-precision complex sequence
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
	fftwf_complex x[NMAX];
	fftwf_complex xx[NMAX];

	printf("ARMPL example: FFT of a single-precision complex sequence using "
	       "fftwf_plan_dft_1d\n");
	printf("-------------------------------------------------------------------"
	       "--------------\n");
	printf("\n");

	/* The sequence of complex data */
	int n = 7;
	x[0] = 0.34907 + -0.37168 * I;
	x[1] = 0.54890 + -0.35669 * I;
	x[2] = 0.74776 + -0.31174 * I;
	x[3] = 0.94459 + -0.23703 * I;
	x[4] = 1.13850 + -0.13274 * I;
	x[5] = 1.32850 + 0.00074 * I;
	x[6] = 1.51370 + 0.16298 * I;

	// Use ccopy to copy the values into another array (preserve input)
	cblas_ccopy(n, (armpl_singlecomplex_t *) x, 1, (armpl_singlecomplex_t *) xx, 1);

	// Initialise a plan for an in-place complex-to-complex 1d forward transform
	// (x->x')
	fftwf_plan forward = fftwf_plan_dft_1d(n, x, x, FFTW_FORWARD, FFTW_ESTIMATE);
	// Initialise a plan for an in-place complex-to-complex 1d backward
	// transform (x'->x)
	fftwf_plan reverse = fftwf_plan_dft_1d(n, x, x, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the forward plan and then deallocate the plan
	fftwf_execute(forward);
	fftwf_destroy_plan(forward);

	printf("Components of discrete Fourier transform:\n");
	printf("\n");
	printf("          Real    Imag\n");
	int j;
	for (j = 0; j < n; j++) {
		float x_real = round_eps_to_zero_f(creal(x[j]) / sqrt(n));
		float x_imag = round_eps_to_zero_f(cimag(x[j]) / sqrt(n));
		printf("%4d   (%7.4f,%7.4f)\n", j, x_real, x_imag);
	}

	// Execute the reverse plan and then deallocate the plan
	/* NOTE: FFTW does NOT compute a normalised transform -
	 * returned array will contain unscaled values */
	fftwf_execute(reverse);
	fftwf_destroy_plan(reverse);

	printf("\n");
	printf("Original sequence as restored by inverse transform:\n");
	printf("\n");
	printf("            Original            Restored\n");
	printf("          Real    Imag        Real    Imag\n");
	for (j = 0; j < n; j++) {
		float xx_real = round_eps_to_zero_f(creal(xx[j]));
		float xx_imag = round_eps_to_zero_f(cimag(xx[j]));
		// Scale factor of 1/n to output normalised data
		float x_real = round_eps_to_zero_f(creal(x[j]) / n);
		float x_imag = round_eps_to_zero_f(cimag(x[j]) / n);
		printf("%4d   (%7.4f,%7.4f)   (%7.4f,%7.4f)\n", j, xx_real, xx_imag, x_real, x_imag);
	}
	return 0;
}

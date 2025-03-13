/*
 * fftw_dft_1d: FFT of a complex sequence
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <armpl.h>
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>

#include "round_eps_to_zero.h"

int main(void) {
#define NMAX 20
	fftw_complex x[NMAX];
	fftw_complex xx[NMAX];

	printf("ARMPL example: FFT of a complex sequence using fftw_plan_dft_1d\n");
	printf("---------------------------------------------------------------\n");
	printf("\n");

	/* The sequence of complex data */
	int n = 7;
	x[0] = 0.34907 + -0.37168 * I;
	x[1] = 0.54890 + -0.35669 * I;
	x[2] = 0.74776 + -0.31174 * I;
	x[3] = 0.94459 + -0.23703 * I;
	x[4] = 1.13870 + -0.13274 * I;
	x[5] = 1.32870 + 0.00074 * I;
	x[6] = 1.51370 + 0.16298 * I;

	// Use zcopy to copy the values into another array (preserve input)
	cblas_zcopy(n, (armpl_doublecomplex_t *) x, 1, (armpl_doublecomplex_t *) xx, 1);

	// Initialise a plan for an in-place complex-to-complex 1d forward transform
	// (x->x')
	fftw_plan forward = fftw_plan_dft_1d(n, x, x, FFTW_FORWARD, FFTW_ESTIMATE);
	// Initialise a plan for an in-place complex-to-complex 1d backward
	// transform (x'->x)
	fftw_plan reverse = fftw_plan_dft_1d(n, x, x, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the forward plan and then deallocate the plan
	fftw_execute(forward);
	fftw_destroy_plan(forward);

	printf("Components of discrete Fourier transform:\n");
	printf("\n");
	printf("          Real    Imag\n");
	int j;
	for (j = 0; j < n; j++) {
		double x_real = round_eps_to_zero_d(creal(x[j]));
		double x_imag = round_eps_to_zero_d(cimag(x[j]));
		printf("%4d   (%7.4f,%7.4f)\n", j, x_real, x_imag);
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
	for (j = 0; j < n; j++) {
		double xx_real = round_eps_to_zero_d(creal(xx[j]));
		double xx_imag = round_eps_to_zero_d(cimag(xx[j]));
		// Scale factor of 1/n to output normalised data
		double x_real = round_eps_to_zero_d(creal(x[j]) / n);
		double x_imag = round_eps_to_zero_d(cimag(x[j]) / n);

		printf("%4d   (%7.4f,%7.4f)   (%7.4f,%7.4f)\n", j, xx_real, xx_imag, x_real, x_imag);
	}
	return 0;
}

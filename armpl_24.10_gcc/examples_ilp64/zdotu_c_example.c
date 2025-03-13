/*
 * Double complex dot product
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <stdio.h>
#include <stdlib.h>
#include "armpl.h"

#define N 4

int main()
{
	const int64_t n = N;
	const int64_t incx = 1;
	const int64_t incy = 1;

	double complex x[N], y[N];
	double complex res;

	printf("ARMPL example: double complex dot product using ZDOTU\n");
	printf("-----------------------------------------------------\n");

	/* Initialize vectors x and y */
	for (int64_t i = 0; i < n; ++i) {
		x[i] = (double)(i+1) + (double)(i+2) * I;
		y[i] = (double)(i+1) + (double)(-i-2) * I;
	}

	printf("\nVector x:\n");
	for (int64_t i = 0; i < n; ++i) {
		printf(" %10.3E%+10.3Ei\n", creal(x[i]), cimag(x[i]));
	}

	printf("\nVector y:\n");
	for (int64_t i = 0; i < n; ++i) {
		printf(" %10.3E%+10.3Ei\n", creal(y[i]), cimag(y[i]));
	}

#if FORTRAN_COMPLEX_RETTYPE == 1
	zdotu_(&res, &n, &x[0], &incx, &y[0], &incy);
#else
	res = zdotu_(&n, &x[0], &incx, &y[0], &incy);
#endif

	printf("\nResult:\n");
	printf(" %10.3E%+10.3Ei\n", creal(res), cimag(res));

	return 0;
}

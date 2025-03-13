/*
 * Double precision hermitian packed matrix-vector multiplication example
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include "armpl.h"

#define N 4

int main()
{
	const char uplo = 'U';

	const int n = N;
	const int incx = 1;
	const int incy = 1;

	const double complex alpha = 1.1 + 1.2 * I;
	const double complex beta = 1.3 + 1.4 * I;

	double complex ap[N*(N+1)/2], x[N], y[N];

	printf("ARMPL example: double complex matrix-vector multiplication using ZHPMV\n");
	printf("----------------------------------------------------------------------\n");
	printf("\n");

	/* Initialize matrix A in packed format */
	for (int i = 0; i < n*(n+1)/2; ++i) {
		ap[i] = (double)(i+1) + (double)(i+2) * I;
	}

	/* Initialize vector x and y */
	for (int i = 0; i < n; ++i) {
		x[i] = (double)(2*i+1) + (double)(2*i+2) * I;
		y[i] = (double)(3*i+1) + (double)(3*i+2) * I;
	}

	printf("alpha : %5.3F%+5.3Fi\n", creal(alpha), cimag(alpha));
	printf("beta  : %5.3F%+5.3Fi\n", creal(beta), cimag(beta));
	printf("\n");

	printf("Matrix A:\n");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			int idx = j < i ? j + i*(i+1)/2 : i + j*(j+1)/2;
			if (j<i) {
				printf(" %10.3E%+10.3Ei", creal(ap[idx]), -cimag(ap[idx]));
			}
			else if (j>i) {
				printf(" %10.3E%+10.3Ei", creal(ap[idx]), cimag(ap[idx]));
			}
			else if (j==i) {
				printf(" %10.3E%+10.3Ei", creal(ap[idx]), 0.0);
			}
		}
		printf("\n");
	}

	printf("\n");
	printf("Vector x:\n");
	for (int i = 0; i < n; ++i) {
		printf(" %10.3E%+10.3Ei\n", creal(x[i]), cimag(x[i]));
	}

	printf("\n");
	printf("Vector y:\n");
	for (int i = 0; i < n; ++i) {
		printf(" %10.3E%+10.3Ei\n", creal(y[i]), cimag(y[i]));
	}

	zhpmv_(&uplo, &n, &alpha, &ap[0], &x[0], &incx, &beta, &y[0], &incy);

	printf("\n");
	printf("Updated vector y = alpha*A*x + beta*y:\n");
	for (int i = 0; i < n; ++i) {
		printf(" %10.3E%+10.3Ei\n", creal(y[i]), cimag(y[i]));
	}

	return 0;
}

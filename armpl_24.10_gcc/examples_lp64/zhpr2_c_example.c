/*
 * Double complex hermitian rank 2 operation in packed format
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
	const char uplo = 'U';

	const int n = N;
	const int incx = 1;
	const int incy = 1;

	const double complex alpha = 1.1 + 1.2 * I;

	double complex ap[N*(N+1)/2], x[N], y[N];

	printf("ARMPL example: double complex hermitian rank 2 operation using ZHPR2\n");
	printf("--------------------------------------------------------------------\n");
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

	zhpr2_(&uplo, &n, &alpha, &x[0], &incx, &y[0], &incy, &ap[0]);

	printf("\n");
	printf("Updated vector y = alpha*A*x + beta*y:\n");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			int idx = j < i ? j + i*(i+1)/2 : i + j*(j+1)/2;
			if (j<i) {
				printf(" %10.3E%+10.3Ei", creal(ap[idx]), -cimag(ap[idx]));
			}
			else if (j>=i) {
				printf(" %10.3E%+10.3Ei", creal(ap[idx]), cimag(ap[idx]));
			}
		}
		printf("\n");
	}

	return 0;
}

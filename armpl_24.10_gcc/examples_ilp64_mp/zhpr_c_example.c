/*
 * Double complex hermitian rank 1 operation in packed format
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

	const int64_t n = N;
	const int64_t incx = 1;

	const double alpha = 1.1;

	double complex ap[N*(N+1)/2], x[N];

	printf("ARMPL example: double complex hermitian rank 1 operation using ZHPR\n");
	printf("-------------------------------------------------------------------\n");
	printf("\n");

	/* Initialize matrix A in packed format */
	for (int64_t i = 0; i < n*(n+1)/2; ++i) {
		ap[i] = (double)(i+1) + (double)(i+2) * I;
	}

	/* Initialize vector x */
	for (int64_t i = 0; i < n; ++i) {
		x[i] = (double)(2*i+1) + (double)(2*i+2) * I;
	}

	printf("alpha : %5.3F\n", alpha);
	printf("\n");

	printf("Matrix A:\n");
	for (int64_t i = 0; i < n; ++i) {
		for (int64_t j = 0; j < n; ++j) {
			int64_t idx = j < i ? j + i*(i+1)/2 : i + j*(j+1)/2;
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
	for (int64_t i = 0; i < n; ++i) {
		printf(" %10.3E%+10.3Ei\n", creal(x[i]), cimag(x[i]));
	}

	zhpr_(&uplo, &n, &alpha, &x[0], &incx, &ap[0]);

	printf("\n");
	printf("Updated matrix A = alpha*x*x**H + A:\n");
	for (int64_t i = 0; i < n; ++i) {
		for (int64_t j = 0; j < n; ++j) {
			int64_t idx = j < i ? j + i*(i+1)/2 : i + j*(j+1)/2;
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

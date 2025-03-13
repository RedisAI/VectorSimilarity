/*
 * Double precision packed matrix-vector multiplication example
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
	const char trans = 'N';
	const char diag = 'N';

	const int64_t n = N;
	const int64_t incx = 1;

	double ap[N*(N+1)/2], x[N];

	printf("ARMPL example: double precision matrix-vector multiplication using DTPMV\n");
	printf("------------------------------------------------------------------------\n");
	printf("\n");

	/* Initialize matrix A in packed format */
	for (int64_t i = 0; i < n*(n+1)/2; ++i) {
		ap[i] = (double)(i+1);
	}

	/* Initialize vector x */
	for (int64_t i = 0; i < n; ++i) {
		x[i] = (double)(2*i+1);
	}

	printf("Matrix A:\n");
	for (int64_t i = 0; i < n; ++i) {
		for (int64_t j = 0; j < n; ++j) {
			int64_t idx = i + j*(j+1)/2;
			printf(" %13.3E", j < i ? 0. : ap[idx]);
		}
		printf("\n");
	}

	printf("\n");
	printf("Vector x:\n");
	for (int64_t i = 0; i < n; ++i) {
		printf(" %13.3E\n", x[i]);
	}

	dtpmv_(&uplo, &trans, &diag, &n, &ap[0], &x[0], &incx);

	printf("\n");
	printf("Updated vector x = A*x:\n");
	for (int64_t i = 0; i < n; ++i) {
		printf(" %13.3E\n", x[i]);
	}

	return 0;
}

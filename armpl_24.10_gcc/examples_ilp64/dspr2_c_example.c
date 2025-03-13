/*
 * Double precision symmetric rank 2 operation in packed format
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
	const int64_t incy = 1;

	const double alpha = 1.1;

	double ap[N*(N+1)/2], x[N], y[N];

	printf("ARMPL example: double precision symmetric rank 2 operation using DSPR2\n");
	printf("----------------------------------------------------------------------\n");
	printf("\n");

	/* Initialize matrix A in packed format */
	for (int64_t i = 0; i < n*(n+1)/2; ++i) {
		ap[i] = (double)(i+1);
	}

	/* Initialize vector x and y */
	for (int64_t i = 0; i < n; ++i) {
		x[i] = (double)(2*i+1);
		y[i] = (double)(3*i+1);
	}

	printf("alpha : %5.3F\n", alpha);
	printf("\n");

	printf("Matrix A:\n");
	for (int64_t i = 0; i < n; ++i) {
		for (int64_t j = 0; j < n; ++j) {
			int64_t idx = j < i ? j + i*(i+1)/2 : i + j*(j+1)/2;
			printf(" %13.3E", ap[idx]);
		}
		printf("\n");
	}

	printf("\n");
	printf("Vector x:\n");
	for (int64_t i = 0; i < n; ++i) {
		printf(" %13.3E\n", x[i]);
	}

	printf("\n");
	printf("Vector y:\n");
	for (int64_t i = 0; i < n; ++i) {
		printf(" %13.3E\n", y[i]);
	}

	dspr2_(&uplo, &n, &alpha, &x[0], &incx, &y[0], &incy, &ap[0]);

	printf("\n");
	printf("Updated matrix A = alpha*x*y**T + alpha*y*x**T + A:\n");
	for (int64_t i = 0; i < n; ++i) {
		for (int64_t j = 0; j < n; ++j) {
			int64_t idx = j < i ? j + i*(i+1)/2 : i + j*(j+1)/2;
			printf(" %13.3E", ap[idx]);
		}
		printf("\n");
	}

	return 0;
}

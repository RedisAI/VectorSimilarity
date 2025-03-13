/*
 * Double precision plane rotation of sparse vectors example
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <stdio.h>
#include <stdlib.h>
#include "armpl.h"

#define NNZ 4
#define N 9

int main()
{
	/* 1. Set-up local sparse and dense vectors */
	armpl_spvec_t armpl_vec_x;
	int creation_flags = 0;

	int index_base = 0;
	int n = N;
	int nnz = NNZ;
	double vals[NNZ] = {1., 2., 3., 4.};
	int indx[NNZ] = {2, 5, 6, 8};
	double y[N] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
	const double c = 0.;
	const double s = 1.;

	printf("Rotation matrix:\n");
	printf("%4.1f %4.1f\n", c, s);
	printf("%4.1f %4.1f\n", -s, c);

	printf("Sparse vector x:\n");
	printf("[");
	int cnt = 0;
	for (int i=0; i<N; ++i) {
		printf("%4.1f ", i == indx[cnt] ? vals[cnt++] : 0.);
	}
	printf("]\n");
	printf("Dense vector y:\n");
	printf("[");
	for (int i=0; i<N; ++i) {
		printf("%4.1f ", y[i]);
	}
	printf("]\n");

	/* 2. Set-up Arm Performance Libraries sparse vector object for x */
	armpl_status_t info = armpl_spvec_create_d(&armpl_vec_x, index_base, n, nnz, &indx[0], &vals[0], creation_flags);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spvec_print_err(armpl_vec_x); return (int)info; }

	/* 3. Do SpRot */
	info = armpl_sprot_exec_d(armpl_vec_x, &y[0], c, s);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spvec_print_err(armpl_vec_x); return (int)info; }

	/* 4. Export results from Arm Performance Libraries sparse vector object */
	info = armpl_spvec_export_d(armpl_vec_x, &index_base, &n, &nnz, &indx[0], &vals[0]);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spvec_print_err(armpl_vec_x); return (int)info; }

	printf("Rotated vector x:\n");
	printf("[");
	cnt = 0;
	for (int i=0; i<N; ++i) {
		printf("%4.1f ", i == indx[cnt] ? vals[cnt++] : 0.);
	}
	printf("]\n");
	printf("Rotated vector y:\n");
	printf("[");
	for (int i=0; i<N; ++i) {
		printf("%4.1f ", y[i]);
	}
	printf("]\n");

	/* 5. Destroy created sparse vector object */
	info = armpl_spvec_destroy(armpl_vec_x);
	if (info!=ARMPL_STATUS_SUCCESS) printf("ERROR: armpl_spvec_destroy returned %d\n", info);

	return (int)info;
}

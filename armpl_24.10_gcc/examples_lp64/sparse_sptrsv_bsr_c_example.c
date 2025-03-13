/*
 * Double precision sparse matrix triangular solve example
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <stdio.h>
#include <stdlib.h>
#include "armpl.h"

#define NNZB 18
#define NNZ 72
#define N 16

int main()
{
	/* 1. Set-up local BSR structure */
	armpl_spmat_t armpl_mat;
	const double alpha = 2.0;
	int creation_flags = 0;
	int block_size = 2;
	double vals[NNZ] = {1., 0., 0., 1.,
	                    1., 0., 0., 1.,
	                    1., 0., 0., 1.,
	                    1., 0., 0., 1.,
	                    1., 0., 0., 1.,
	                    2., 0., 0., 2.,
	                    3., 0., 0., 3.,
	                    1., 0., 0., 1.,
	                    3., 0., 0., 3.,
	                    4., 0., 0., 4.,
	                    1., 0., 0., 1.,
	                    4., 0., 0., 4.,
	                    5., 0., 0., 5.,
	                    1., 0., 0., 1.,
	                    11., 0., 0., 11.,
	                    14., 0., 0., 14.,
	                    16., 0., 0., 16.,
	                    18., 0., 0., 18.};
	// Note: one-based indexing used here, since row_ptr[0] = 1.
	int row_ptr[N+1] = {1, 5, 8, 11, 14, 16, 17, 18, 19};
	int col_indx[NNZB] = {1, 3, 6, 8, 2, 4, 7, 3, 5, 6, 4, 5, 7, 5, 8, 6, 7, 8};

	/* 2. Set-up Arm Performance Libraries sparse matrix object */
	armpl_status_t info = armpl_spmat_create_bsr_d(&armpl_mat, ARMPL_ROW_MAJOR, N, N, block_size, row_ptr, col_indx, vals, creation_flags);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spmat_print_err(armpl_mat); return (int)info; }

	/* 3. Supply any hints that are about the SpTRSV calculations to be performed */
	info = armpl_spmat_hint(armpl_mat, ARMPL_SPARSE_HINT_SPSV_OPERATION, ARMPL_SPARSE_OPERATION_NOTRANS);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spmat_print_err(armpl_mat); return (int)info; }

	/* 4. Call an optimization process that will learn from the hints you have previously supplied */
	info = armpl_spsv_optimize(armpl_mat);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spmat_print_err(armpl_mat); return (int)info; }

	/* 5. Setup input and output vectors and then do Spsv and print result.*/
	double *x = (double *)malloc(block_size*N*sizeof(double));
	double *y = (double *)malloc(block_size*N*sizeof(double));
	for (int i=0; i<N/block_size; i++) {
		for (int j=0; j<block_size; j++) {
			y[i*block_size + j] = 2.0 + (double)(i);
		}
	}

	info = armpl_spsv_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS, armpl_mat, x, alpha, y);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spmat_print_err(armpl_mat); return (int)info; }

	printf("Input RHS vector y:\n");
	for (int i=0; i<N; i++) {
		printf("  %4.1f\n", y[i]);
	}

	printf("Computed solution vector x:\n");
	for (int i=0; i<N; i++) {
		printf("  %4.1f\n", x[i]);
	}

	/* 6. Destroy created matrix to free any memory */
	info = armpl_spmat_destroy(armpl_mat);
	if (info!=ARMPL_STATUS_SUCCESS) printf("ERROR: armpl_spmat_destroy returned %d\n", info);

	/* 7. Free user allocated storage */
	free(x); free(y);

	return (int)info;
}

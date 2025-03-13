/*
 * Double precision sparse triangular solve error return example
 *
 * The purpose of this example is to demonstrate what happens
 * when an attempt is made to perform a sparse triangular solve
 * using a matrix that is rectangular in shape (i.e. not upper
 * or lower triangular).

 * The setup is the same as in sparse_sptrsv_c_example.c, except
 * that here there is a non-zero element in the lower half of the
 * matrix in addition to the upper triangular matrix.
 *
 * The fact that there are non-zero elements in both halves of
 * the matrix is detected by the library and reported during
 * the armpl_spsv_optimize call.
 *
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <stdio.h>
#include <stdlib.h>
#include "armpl.h"

#define NNZ 19
#define N 8

int main()
{
	/* 1. Set-up local CSR structure */
	armpl_spmat_t armpl_mat;
	const double alpha = 2.0;
	int64_t creation_flags = 0;
	double vals[NNZ] = {1., 1., 1., 1., 1., 2., 3., 1., 3., 4., 1., 4., 5., 1., 11., 14., 16., 18.};
	int64_t row_ptr[N+1] = {0, 4, 7, 10, 14, 16, 17, 17, 19};
	int64_t col_indx[NNZ] = {0, 2, 5, 7, 1, 3, 6, 0, 2, 4, 5, 3, 4, 6, 4, 7, 5, 6, 7};

	/* 2. Set-up Arm Performance Libraries sparse matrix object */
	armpl_status_t info = armpl_spmat_create_csr_d(&armpl_mat, N, N, row_ptr, col_indx, vals, creation_flags);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spmat_print_err(armpl_mat); return (int)info; }

	/* 3. Supply any hints that are about the SpTRSV calculations to be performed */
	info = armpl_spmat_hint(armpl_mat, ARMPL_SPARSE_HINT_SPSV_OPERATION, ARMPL_SPARSE_OPERATION_NOTRANS);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spmat_print_err(armpl_mat); return (int)info; }

	/*
	   4. Call an optimization process that will learn from the hints you have previously supplied

	      NOTE: this call should return an error because the matrix is not upper or lower triangular.
	            Hence, we exit the example with 0 on successful detection of the deliberate error.
	            If this optimization call is skipped, the same error will be returned by
	            armpl_spsv_exec_d instead.
	*/
	info = armpl_spsv_optimize(armpl_mat);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spmat_print_err(armpl_mat); return 0; }

	/* 5. Setup input and output vectors and then do Spsv and print result.*/
	double *x = (double *)malloc(N*sizeof(double));
	double *y = (double *)malloc(N*sizeof(double));
	for (int i=0; i<N; i++) {
		y[i] = 2.0 + (double)(i);
	}

	info = armpl_spsv_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS, armpl_mat, x, alpha, y);
	if (info!=ARMPL_STATUS_SUCCESS) { armpl_spmat_print_err(armpl_mat); return (int)info; }

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

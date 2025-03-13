/*
 * zgemm_batch Example Program Test
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <armpl.h>
#include <stdio.h>
#include <stdlib.h>
#define NMAX 25

int main(void) {
	const CBLAS_LAYOUT layout = CblasRowMajor;
	const int64_t group_count = 3;

	printf("ARMPL example: matrix multiplication using cblas_zgemm_batch\n");
	printf("------------------------------------------------------------\n");

	int64_t *group_size;
	group_size = (int64_t*)malloc((group_count)*sizeof(int64_t));

	// Set group sizes
	group_size[0] = 2;
	group_size[1] = 3;
	group_size[2] = 4;

	// Calculate total batch count as the sum of all group sizes
	int64_t total_batch_count = 0;
	for (int64_t i=0; i<group_count; i++) {
		 total_batch_count += group_size[i];
	}

	// Allocate memory for all the arrays used
	CBLAS_TRANSPOSE *transA, *transB;
	transA = (CBLAS_TRANSPOSE*)malloc((group_count)*sizeof(CBLAS_TRANSPOSE));
	transB = (CBLAS_TRANSPOSE*)malloc((group_count)*sizeof(CBLAS_TRANSPOSE));

	int64_t *m, *n, *k;
	m = (int64_t*)malloc((group_count)*sizeof(int64_t));
	n = (int64_t*)malloc((group_count)*sizeof(int64_t));
	k = (int64_t*)malloc((group_count)*sizeof(int64_t));

	int64_t *lda, *ldb, *ldc;
	lda = (int64_t*)malloc((group_count)*sizeof(int64_t));
	ldb = (int64_t*)malloc((group_count)*sizeof(int64_t));
	ldc = (int64_t*)malloc((group_count)*sizeof(int64_t));

	armpl_doublecomplex_t *alpha, *beta;
	alpha = (armpl_doublecomplex_t*)malloc((group_count)*sizeof(armpl_doublecomplex_t));
	beta = (armpl_doublecomplex_t*)malloc((group_count)*sizeof(armpl_doublecomplex_t));

	// Allocate memory for the matrices A, B, C
	armpl_doublecomplex_t **A, **B, **C;
	A = (armpl_doublecomplex_t**)malloc(total_batch_count*sizeof(armpl_doublecomplex_t*));
	A[0] = (armpl_doublecomplex_t*)malloc(NMAX*NMAX*sizeof(armpl_doublecomplex_t));

	B = (armpl_doublecomplex_t**)malloc(total_batch_count*sizeof(armpl_doublecomplex_t*));
	B[0] = (armpl_doublecomplex_t*)malloc(NMAX*NMAX*sizeof(armpl_doublecomplex_t));

	C = (armpl_doublecomplex_t**)malloc(total_batch_count*sizeof(armpl_doublecomplex_t*));
	C[0] = (armpl_doublecomplex_t*)malloc(NMAX*NMAX*sizeof(armpl_doublecomplex_t));

	/* Set values to matrices A, B and C */
	for (int64_t k = 0; k < NMAX; k++) {
		for (int64_t j = 0; j < NMAX; j++) {
			A[0][k*NMAX + j] = (k + j) + k*I;
			B[0][k*NMAX + j] = (k - j) + j*I;
			C[0][k*NMAX + j] = 0 + 0*I;
		}
	}

	// Set transA options
	transA[0] = CblasNoTrans;
	transA[1] = CblasTrans;
	transA[2] = CblasConjTrans;

	// Set trans B options
	transB[0] = CblasNoTrans;
	transB[1] = CblasTrans;
	transB[2] = CblasConjTrans;

	// Set size matrices
	// m array
	m[0] = 2;
	m[1] = 2;
	m[2] = 3;

	// k array
	k[0] = 2;
	k[1] = 3;
	k[2] = 2;

	// n array
	n[0] = 3;
	n[1] = 2;
	n[2] = 3;

	// Set alpha array
	alpha[0] = 1.0 + -1.0*I;
	alpha[1] = 1.0 + -1.0*I;
	alpha[2] = 1.0 + -1.0*I;

	// Set beta array
	beta[0] = 0.0 + 0.0*I;
	beta[1] = 0.0 + 0.0*I;
	beta[2] = 0.0 + 0.0*I;

	// Set lda
	lda[0] = NMAX;
	lda[1] = NMAX;
	lda[2] = NMAX;

	// Set ldb
	ldb[0] = NMAX;
	ldb[1] = NMAX;
	ldb[2] = NMAX;

	// Set ldc
	ldc[0] = NMAX;
	ldc[1] = NMAX;
	ldc[2] = NMAX;

	// Set pointers to matrices
	// For matrices A
	// for group 0
	// transA = N, k=2
	A[0] = &A[0][0];
	A[1] = &A[0][2];
	// for group 1
	// transA = T, m=2
	A[2] = &A[0][4];
	A[3] = &A[0][6];
	A[4] = &A[0][8];
	// for group 2
	// transA = C, m=3
	A[5] = &A[0][10];
	A[6] = &A[0][13];
	A[7] = &A[0][16];
	A[8] = &A[0][19];

	// For matrices B
	// group 0
	// transB = N, n = 3
	B[0]= &B[0][0];
	B[1]= &B[0][3];
	// group 1
	// transB = T, k= 3
	B[2] = &B[0][6];
	B[3] = &B[0][9];
	B[4] = &B[0][12];
	// group 2
	// transB = C, k = 2
	B[5] = &B[0][15];
	B[6] = &B[0][17];
	B[7] = &B[0][19];
	B[8] = &B[0][21];

	// For matrices C
	// n=3
	C[0] = &C[0][0];
	C[1] = &C[0][3];
	// n=2
	C[2] = &C[0][6];
	C[3] = &C[0][8];
	C[4] = &C[0][10];
	// n=3
	C[5] = &C[0][12];
	C[6] = &C[0][15];
	C[7] = &C[0][18];
	C[8] = &C[0][21];

	// Call the function
	cblas_zgemm_batch(layout, transA, transB, m, n, k, alpha,
	                  (const void *const *)A, lda,
	                  (const void *const *)B, ldb,
	                  beta, (void *const *)C, ldc, group_count, group_size);

	// Print every group of matrices A, B and C
	int64_t imat = 0;
	for (int64_t i = 0; i < group_count; i++) {
		for (int64_t j = 0; j < group_size[i]; j++) {

				printf("\n****** Group %lld\n", (long long)i);

				printf("* alpha : (%7.4F, %7.4Fi)\n", creal(alpha[i]), cimag(alpha[i]));
				printf("* beta : (%7.4F, %7.4Fi)\n", creal(beta[i]), cimag(beta[i]));

				printf("\n*** Matrix A\n");

				int64_t ma = 0;
				int64_t na = 0;
				switch(transA[i])
				{
					case CblasNoTrans:
						printf("* TransA: CblasNoTrans\n");
						ma = m[i];
						na = k[i];
						break;
					case CblasTrans:
						printf("* TransA: CblasTrans\n");
						ma = k[i];
						na = m[i];
						break;
					case CblasConjTrans:
						printf("* TransA: CblasConjTrans\n");
						ma = k[i];
						na = m[i];
						break;
				}

				for (int64_t ii = 0; ii < ma; ii++) {
					for (int64_t jj = 0; jj < na; jj++) {
						printf("(%7.4f, %7.4fi)\t", creal(A[imat][ii*lda[i] + jj]),
								 cimag(A[imat][ii*lda[i] + jj]));
					}
					printf("\n");
				}

				printf("\n*** Matrix B\n");

				int64_t mb = 0;
				int64_t nb = 0;
				switch(transB[i])
				{
					case CblasNoTrans:
						printf("* TransB: CblasNoTrans\n");
						mb = k[i];
						nb = n[i];
						break;
					case CblasTrans:
						printf("* TransB: CblasTrans\n");
						mb = n[i];
						nb = k[i];
						break;
					case CblasConjTrans:
						printf("* TransB: CblasConjTrans\n");
						mb = n[i];
						nb = k[i];
						break;
				}

				for (int64_t ii = 0; ii < mb; ii++) {
					for (int64_t jj = 0; jj < nb; jj++) {
						 printf("(%7.4f, %7.4fi)\t", creal(B[imat][ii*ldb[i] + jj]),
								cimag(B[imat][i*ldb[i] + jj]));
					}
					 printf("\n");
				}

				printf("\n*** Matrix C\n");
				for (int64_t ii = 0; ii < m[i]; ii++) {
					for (int64_t jj = 0; jj < n[i]; jj++) {
						printf("(%7.4f, %7.4fi)\t", creal(C[imat][ii*ldc[i] + jj]),
								cimag(C[imat][ii*ldc[i] + jj]));
				}
				printf("\n");
			}

			imat++;

			printf("==========end entry\n");
		}
	}
	printf("==========end group==========\n");

	free(group_size);
	free(transA);
	free(transB);
	free(m);
	free(n);
	free(k);
	free(lda);
	free(ldb);
	free(ldc);
	free(alpha);
	free(beta);
	free(A[0]);
	free(A);
	free(B[0]);
	free(B);
	free(C[0]);
	free(C);

}

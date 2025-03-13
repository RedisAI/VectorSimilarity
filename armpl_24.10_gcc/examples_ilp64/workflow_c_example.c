/*
 * Workflow Example Program Text
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <armpl.h>
#include <fftw3.h>

void handle_armpl_rng_error(int info, const char *message) {
	if (info != VSL_ERROR_OK) {
		printf("Error: %s\n", message);
		exit(EXIT_FAILURE);
	}
}

void handle_armpl_spmat_error(armpl_status_t info, armpl_spmat_t armpl_mat) {
	if (info != ARMPL_STATUS_SUCCESS) {
		armpl_spmat_print_err(armpl_mat);
		exit(EXIT_FAILURE);
	}
}

// We use the random number generator from openrng.
// Declare openrng stream.
static VSLStreamStatePtr stream;

// Computes a complex to real FFT
void perform_fft(double complex *in_data, double *out, int64_t n) {

	fftw_plan plan = fftw_plan_dft_c2r_1d(n, in_data, out, FFTW_ESTIMATE);
	fftw_execute(plan);

	// Free memory
	fftw_destroy_plan(plan);
}

// Generate_csr_format create row_ptr and col_indx
// given m, n, and nnz
void generate_csr_format(int64_t m, int64_t n, int64_t nnz, int64_t *col_indx,
                         int64_t *row_ptr) {
	int64_t i, k, row, col;
	// Tracking position (row, col)
	int64_t **filled = (int64_t **) malloc((m + 1) * sizeof(int *));
	for (i = 0; i <= m; i++) {
		filled[i] = (int64_t *) calloc((n + 1), sizeof(int64_t));
	}

	// Assign nnz positions
	col_indx[0] = 0;
	for (i = 0; i <= m; i++) {
		row_ptr[i] = 0;
	}
	for (k = 0; k < nnz; k++) {
		do {
			double random_number[2];
			int errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2, random_number, 0, 1);
			handle_armpl_rng_error(errcode, "vdRngUniform failed");
			row = random_number[0] * (m - 1);
			col = random_number[1] * (n - 1);
		} while (filled[row][col]); // Check if the position is already filled

		filled[row][col] = 1; // Mark as filled
		col_indx[k] = col;
		row_ptr[row]++;
	}

	// Convert row counts into starting indices
	for (i = 1; i <= m; i++) {
		row_ptr[i] += row_ptr[i - 1];
	}
	row_ptr[1] = 0;

	// Free the filled array
	for (i = 0; i <= m; i++) {
		free(filled[i]);
	}
	free(filled);
}

void spmv_csr(armpl_spmat_t armpl_mat, double alpha, double beta, const double *x, double *y) {

	armpl_status_t info =
	    armpl_spmat_hint(armpl_mat, ARMPL_SPARSE_HINT_SPMV_OPERATION, ARMPL_SPARSE_OPERATION_NOTRANS);
	handle_armpl_spmat_error(info, armpl_mat);

	info = armpl_spmv_optimize(armpl_mat);
	handle_armpl_spmat_error(info, armpl_mat);

	info = armpl_spmv_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS, alpha, armpl_mat, x, beta, y);
	handle_armpl_spmat_error(info, armpl_mat);
}

int main() {

	int errcode = vslNewStream(&stream, VSL_BRNG_PHILOX4X32X10, 42);
	handle_armpl_rng_error(errcode, "vslNewStream failed");

	// Step 1. FFT
	// This step will generate a double real vector
	const size_t n = 1000;
	// Generate random complex input
	double complex *in = (double complex *) malloc(n * sizeof(double complex));
	errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2 * n, (double *) in, 0, 1);
	handle_armpl_rng_error(errcode, "vdRngUniform failed");

	double *x = (double *) malloc(n * sizeof(double));
	perform_fft(in, x, n);
	free(in);

	// Step 2. SPMV
	// Use the output from FFT as x in the SPMV
	// Generate a sparse matrix and perform an SPMV
	const size_t m = n;
	const size_t nnz = 400;

	double *y = (double *) malloc(n * sizeof(double));
	errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, y, 0, 1);
	handle_armpl_rng_error(errcode, "vdRngUniform failed");
	double *vals = (double *) malloc(nnz * sizeof(double));
	errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, nnz, vals, 0, 1);
	handle_armpl_rng_error(errcode, "vdRngUniform failed");

	int64_t *col_indx = (int64_t *) malloc(nnz * sizeof(int64_t));
	int64_t *row_ptr = (int64_t *) malloc((m + 1) * sizeof(int64_t));
	generate_csr_format(m, n, nnz, col_indx, row_ptr);
	// Create opaque sparse matrix
	armpl_spmat_t sparse_mat;
	const int64_t flags = 0;
	armpl_status_t info = armpl_spmat_create_csr_d(&sparse_mat, m, n, row_ptr, col_indx, vals, flags);
	handle_armpl_spmat_error(info, sparse_mat);

	const double alpha = 1.0;
	const double beta = 1.0;
	spmv_csr(sparse_mat, alpha, beta, x, y);

	info = armpl_spmat_destroy(sparse_mat);
	handle_armpl_spmat_error(info, sparse_mat);

	free(vals);
	free(row_ptr);
	free(col_indx);

	// Step 3. GEMV
	// Use the output vectors from FFT and SPMV in GEMV
	const int64_t lda = m;
	double *A = (double *) malloc(lda * m * sizeof(double));
	errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, lda * m, A, 0, 1);
	handle_armpl_rng_error(errcode, "vdRngUniform failed");

	const size_t incx = 1;
	const size_t incy = 1;
	cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, A, lda, x, incx, beta, y, incy);

	// Step 4. GELS
	// Use the output from GEMV as a rhs in a least squares system
	const int64_t ldb = lda;
	const size_t nrhs = 1;
	LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', m, m, nrhs, A, lda, y, ldb);

	free(A);

	// Step 5. cos from libamath
	// Compute cos on the entries of the output of GELS
	for (int64_t i = 0; i < m; i++) {
		y[i] = cos(y[i]);
	}

	// Step 6. memcpy from libastring
	// Copy the vector transformed at step 5 into a new vector
	// using memcpy
	double *copied_array = (double *) malloc(m * sizeof(double));
	memcpy(copied_array, y, m * sizeof(double));

	// Step 7. dot product
	// To finish, perform a dot produt on the outpt
	// to have a scalar
	double result = cblas_ddot(n, copied_array, incx, y, incy);

	free(x);
	free(y);
	free(copied_array);

	// The exact result is pre-computed and serve as a reference
	double exact_result = 479.733703668257135178;
	double error = fabs(result - exact_result) / fabs(n * exact_result);
	double tolerance = 1e-10;
	const char *test_status = error < tolerance ? "PASSED" : "FAILED";
	printf("TEST  %s\n", test_status);
	return 0;
}

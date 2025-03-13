/*
 * Wrappers for exporting sparse matrices in Fortran using the C
 * interface. These are required when using a version of libgfortran
 * which does not match the version with which libarmpl has been built.
 *
 * See examples/sparse_spadd_f_example.f90 and examples/sparse_spmm_f_example.f90
 * for examples of how to call these functions from Fortran.
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <armpl_sparse.h>
#include <stdlib.h>

/*
	Compile this file with your GCC toolchain and call this
	function to determine whether or not you should call into the
	sparse matrix export/deallocate wrappers defined in this file.
*/
void query_mismatch_(armpl_int_t *mismatch) {
// Only detect mismatch if we're actually gcc (not LLVM/clang) and earlier than gcc 8
#if !defined(__clang__) && __GNUC__ < 8
	*mismatch = 1; // Mismatch is true. Wrappers below are required.
#else
	// Wrappers are not required; you should use the documented Fortran
	// sparse export functions and you can disregard this file.
	*mismatch = 0;
#endif
}


// Deallocation of arrays returned by the export functions below
void armpl_spmat_export_deallocate_wrapper_(void **p) {
	free(*p);
}


/* ---- CSR exports ---- */
void armpl_spmat_export_csr_wrapper_s_(armpl_const_spmat_t *A, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n,
                                       armpl_int_t **row_ptr,
                                       armpl_int_t **col_indx,
                                       float **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_csr_s(*A, *index_base, m, n, row_ptr, col_indx, vals);
}

void armpl_spmat_export_csr_wrapper_d_(armpl_const_spmat_t *A, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n,
                                       armpl_int_t **row_ptr,
                                       armpl_int_t **col_indx,
                                       double **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_csr_d(*A, *index_base, m, n, row_ptr, col_indx, vals);
}

void armpl_spmat_export_csr_wrapper_c_(armpl_const_spmat_t *A, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n,
                                       armpl_int_t **row_ptr,
                                       armpl_int_t **col_indx,
                                       armpl_singlecomplex_t **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_csr_c(*A, *index_base, m, n, row_ptr, col_indx, vals);
}

void armpl_spmat_export_csr_wrapper_z_(armpl_const_spmat_t *A, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n,
                                       armpl_int_t **row_ptr,
                                       armpl_int_t **col_indx,
                                       armpl_doublecomplex_t **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_csr_z(*A, *index_base, m, n, row_ptr, col_indx, vals);
}


/* ---- CSC exports ---- */
void armpl_spmat_export_csc_wrapper_s_(armpl_const_spmat_t *A, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n,
                                       armpl_int_t **row_indx,
                                       armpl_int_t **col_ptr,
                                       float **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_csc_s(*A, *index_base, m, n, row_indx, col_ptr, vals);
}

void armpl_spmat_export_csc_wrapper_d_(armpl_const_spmat_t *A, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n,
                                       armpl_int_t **row_indx,
                                       armpl_int_t **col_ptr,
                                       double **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_csc_d(*A, *index_base, m, n, row_indx, col_ptr, vals);
}

void armpl_spmat_export_csc_wrapper_c_(armpl_const_spmat_t *A, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n,
                                       armpl_int_t **row_indx,
                                       armpl_int_t **col_ptr,
                                       armpl_singlecomplex_t **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_csc_c(*A, *index_base, m, n, row_indx, col_ptr, vals);
}

void armpl_spmat_export_csc_wrapper_z_(armpl_const_spmat_t *A, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n,
                                       armpl_int_t **row_indx,
                                       armpl_int_t **col_ptr,
                                       armpl_doublecomplex_t **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_csc_z(*A, *index_base, m, n, row_indx, col_ptr, vals);
}


/* ---- BSR exports ---- */
void armpl_spmat_export_bsr_wrapper_s_(armpl_const_spmat_t *A, enum armpl_dense_layout *layout, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n, armpl_int_t *block_size,
                                       armpl_int_t **row_ptr,
                                       armpl_int_t **col_indx,
                                       float **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_bsr_s(*A, *layout, *index_base, m, n, block_size, row_ptr, col_indx, vals);
}

void armpl_spmat_export_bsr_wrapper_d_(armpl_const_spmat_t *A, enum armpl_dense_layout *layout, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n, armpl_int_t *block_size,
                                       armpl_int_t **row_ptr,
                                       armpl_int_t **col_indx,
                                       double **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_bsr_d(*A, *layout, *index_base, m, n, block_size, row_ptr, col_indx, vals);
}

void armpl_spmat_export_bsr_wrapper_c_(armpl_const_spmat_t *A, enum armpl_dense_layout *layout, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n, armpl_int_t *block_size,
                                       armpl_int_t **row_ptr,
                                       armpl_int_t **col_indx,
                                       armpl_singlecomplex_t **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_bsr_c(*A, *layout, *index_base, m, n, block_size, row_ptr, col_indx, vals);
}

void armpl_spmat_export_bsr_wrapper_z_(armpl_const_spmat_t *A, enum armpl_dense_layout *layout, armpl_int_t *index_base,
                                       armpl_int_t *m, armpl_int_t *n, armpl_int_t *block_size,
                                       armpl_int_t **row_ptr,
                                       armpl_int_t **col_indx,
                                       armpl_doublecomplex_t **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_bsr_z(*A, *layout, *index_base, m, n, block_size, row_ptr, col_indx, vals);
}


/* ---- COO exports ---- */
void armpl_spmat_export_coo_wrapper_s_(armpl_const_spmat_t *A,
                                       armpl_int_t *m, armpl_int_t *n, armpl_int_t *nnz,
                                       armpl_int_t **row_indx,
                                       armpl_int_t **col_indx,
                                       float **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_coo_s(*A, m, n, nnz, row_indx, col_indx, vals);
}

void armpl_spmat_export_coo_wrapper_d_(armpl_const_spmat_t *A,
                                       armpl_int_t *m, armpl_int_t *n, armpl_int_t *nnz,
                                       armpl_int_t **row_indx,
                                       armpl_int_t **col_indx,
                                       double **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_coo_d(*A, m, n, nnz, row_indx, col_indx, vals);
}

void armpl_spmat_export_coo_wrapper_c_(armpl_const_spmat_t *A,
                                       armpl_int_t *m, armpl_int_t *n, armpl_int_t *nnz,
                                       armpl_int_t **row_indx,
                                       armpl_int_t **col_indx,
                                       armpl_singlecomplex_t **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_coo_c(*A, m, n, nnz, row_indx, col_indx, vals);
}

void armpl_spmat_export_coo_wrapper_z_(armpl_const_spmat_t *A,
                                       armpl_int_t *m, armpl_int_t *n, armpl_int_t *nnz,
                                       armpl_int_t **row_indx,
                                       armpl_int_t **col_indx,
                                       armpl_doublecomplex_t **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_coo_z(*A, m, n, nnz, row_indx, col_indx, vals);
}


/* ---- Dense exports ---- */
void armpl_spmat_export_dense_wrapper_s_(armpl_const_spmat_t *A, enum armpl_dense_layout *layout,
                                         armpl_int_t *m, armpl_int_t *n,
                                         float **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_dense_s(*A, *layout, m, n, vals);
}

void armpl_spmat_export_dense_wrapper_d_(armpl_const_spmat_t *A, enum armpl_dense_layout *layout,
                                         armpl_int_t *m, armpl_int_t *n,
                                         double **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_dense_d(*A, *layout, m, n, vals);
}

void armpl_spmat_export_dense_wrapper_c_(armpl_const_spmat_t *A, enum armpl_dense_layout *layout,
                                         armpl_int_t *m, armpl_int_t *n,
                                         armpl_singlecomplex_t **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_dense_c(*A, *layout, m, n, vals);
}

void armpl_spmat_export_dense_wrapper_z_(armpl_const_spmat_t *A, enum armpl_dense_layout *layout,
                                         armpl_int_t *m, armpl_int_t *n,
                                         armpl_doublecomplex_t **vals, armpl_status_t *info) {

	*info = armpl_spmat_export_dense_z(*A, *layout, m, n, vals);
}

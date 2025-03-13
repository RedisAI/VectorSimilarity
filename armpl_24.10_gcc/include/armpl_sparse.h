/*
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 * SPDX-FileCopyrightText: Copyright 2015-2024 NAG
 */

#ifndef ARMPL_SPARSE_H
#define ARMPL_SPARSE_H

#include "armpl_int.h"
#include "armpl_status.h"
#include "armpl_complex.h"

/* Structures */

typedef struct armpl_spmat_top_t *armpl_spmat_t;
typedef const struct armpl_spmat_top_t *armpl_const_spmat_t;
typedef struct armpl_spvec_top_t *armpl_spvec_t;
typedef const struct armpl_spvec_top_t *armpl_const_spvec_t;

/* ENUMs */

enum armpl_sparse_hint_type {
	ARMPL_SPARSE_HINT_STRUCTURE=50,
	ARMPL_SPARSE_HINT_SPMV_OPERATION=60,
	ARMPL_SPARSE_HINT_SPMM_OPERATION=61,
	ARMPL_SPARSE_HINT_SPADD_OPERATION=62,
	ARMPL_SPARSE_HINT_SPSV_OPERATION=63,
	ARMPL_SPARSE_HINT_SPMM_STRATEGY=64,
	ARMPL_SPARSE_HINT_MEMORY=70,
	ARMPL_SPARSE_HINT_SPMV_INVOCATIONS=80,
	ARMPL_SPARSE_HINT_SPMM_INVOCATIONS=81,
	ARMPL_SPARSE_HINT_SPADD_INVOCATIONS=82,
	ARMPL_SPARSE_HINT_SPSV_INVOCATIONS=83
};

enum armpl_dense_layout {
	ARMPL_COL_MAJOR=90,
	ARMPL_ROW_MAJOR=91
};

enum armpl_sparse_hint_value  {
	/* Structure hints */
	ARMPL_SPARSE_STRUCTURE_DENSE=100,
	ARMPL_SPARSE_STRUCTURE_UNSTRUCTURED=101,
	ARMPL_SPARSE_STRUCTURE_SYMMETRIC=110,
	ARMPL_SPARSE_STRUCTURE_DIAGONAL=120,
	ARMPL_SPARSE_STRUCTURE_BLOCKDIAGONAL=130,
	ARMPL_SPARSE_STRUCTURE_BANDED=140,
	ARMPL_SPARSE_STRUCTURE_TRIANGULAR=150,
	ARMPL_SPARSE_STRUCTURE_BLOCKTRIANGULAR=160,
	ARMPL_SPARSE_STRUCTURE_HERMITIAN=170,
	ARMPL_SPARSE_STRUCTURE_HPCG=180,
	/* Memory alocation allowed? */
	ARMPL_SPARSE_MEMORY_NOALLOCS=200,
	ARMPL_SPARSE_MEMORY_ALLOCS=201,
	/* SPMV transpose operation */
	ARMPL_SPARSE_OPERATION_NOTRANS=300,
	ARMPL_SPARSE_OPERATION_TRANS=310,
	ARMPL_SPARSE_OPERATION_CONJTRANS=320,
	/* SPMV execution count estimate */
	ARMPL_SPARSE_INVOCATIONS_SINGLE=400,
	ARMPL_SPARSE_INVOCATIONS_FEW=410,
	ARMPL_SPARSE_INVOCATIONS_MANY=420,
	/* Scalar hints */
	ARMPL_SPARSE_SCALAR_ONE=500,
	ARMPL_SPARSE_SCALAR_ZERO=501,
	ARMPL_SPARSE_SCALAR_ANY=502,
	/* SpMM Strategies */
	ARMPL_SPARSE_SPMM_STRAT_UNSET=600,
	ARMPL_SPARSE_SPMM_STRAT_OPT_NO_STRUCT=601,
	ARMPL_SPARSE_SPMM_STRAT_OPT_PART_STRUCT=602,
	ARMPL_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT=603
};

/* Flags for matrix creation */
#define ARMPL_SPARSE_CREATE_NOCOPY 1

/* C function prototypes */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

armpl_status_t armpl_spmat_create_csr_s(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, const armpl_int_t *row_ptr, const armpl_int_t *col_indx, const float *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_csr_d(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, const armpl_int_t *row_ptr, const armpl_int_t *col_indx, const double *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_csr_c(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, const armpl_int_t *row_ptr, const armpl_int_t *col_indx, const armpl_singlecomplex_t *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_csr_z(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, const armpl_int_t *row_ptr, const armpl_int_t *col_indx, const armpl_doublecomplex_t *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_csc_s(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, const armpl_int_t *row_indx, const armpl_int_t *col_ptr, const float *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_csc_d(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, const armpl_int_t *row_indx, const armpl_int_t *col_ptr, const double *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_csc_c(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, const armpl_int_t *row_indx, const armpl_int_t *col_ptr, const armpl_singlecomplex_t *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_csc_z(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, const armpl_int_t *row_indx, const armpl_int_t *col_ptr, const armpl_doublecomplex_t *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_coo_s(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, armpl_int_t nnz, const armpl_int_t *row_indx, const armpl_int_t *col_indx, const float *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_coo_d(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, armpl_int_t nnz, const armpl_int_t *row_indx, const armpl_int_t *col_indx, const double *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_coo_c(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, armpl_int_t nnz, const armpl_int_t *row_indx, const armpl_int_t *col_indx, const armpl_singlecomplex_t *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_coo_z(armpl_spmat_t *A, armpl_int_t m, armpl_int_t n, armpl_int_t nnz, const armpl_int_t *row_indx, const armpl_int_t *col_indx, const armpl_doublecomplex_t *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_dense_s(armpl_spmat_t *A, enum armpl_dense_layout layout, armpl_int_t m, armpl_int_t n, armpl_int_t lda, const float *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_dense_d(armpl_spmat_t *A, enum armpl_dense_layout layout, armpl_int_t m, armpl_int_t n, armpl_int_t lda, const double *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_dense_c(armpl_spmat_t *A, enum armpl_dense_layout layout, armpl_int_t m, armpl_int_t n, armpl_int_t lda, const armpl_singlecomplex_t *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_dense_z(armpl_spmat_t *A, enum armpl_dense_layout layout, armpl_int_t m, armpl_int_t n, armpl_int_t lda, const armpl_doublecomplex_t *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_bsr_s(armpl_spmat_t *A, enum armpl_dense_layout block_layout, armpl_int_t m, armpl_int_t n, armpl_int_t block_size, const armpl_int_t *row_ptr, const armpl_int_t *col_indx, const float *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_bsr_d(armpl_spmat_t *A, enum armpl_dense_layout block_layout, armpl_int_t m, armpl_int_t n, armpl_int_t block_size, const armpl_int_t *row_ptr, const armpl_int_t *col_indx, const double *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_bsr_c(armpl_spmat_t *A, enum armpl_dense_layout block_layout, armpl_int_t m, armpl_int_t n, armpl_int_t block_size, const armpl_int_t *row_ptr, const armpl_int_t *col_indx, const armpl_singlecomplex_t *vals, armpl_int_t flags);
armpl_status_t armpl_spmat_create_bsr_z(armpl_spmat_t *A, enum armpl_dense_layout block_layout, armpl_int_t m, armpl_int_t n, armpl_int_t block_size, const armpl_int_t *row_ptr, const armpl_int_t *col_indx, const armpl_doublecomplex_t *vals, armpl_int_t flags);
armpl_spmat_t armpl_spmat_create_null(armpl_int_t m, armpl_int_t n);
armpl_spmat_t armpl_spmat_create_identity(armpl_int_t n);

armpl_status_t armpl_spmat_destroy(armpl_spmat_t A);

armpl_status_t armpl_spmat_query(armpl_spmat_t A, armpl_int_t *index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t *nnz);

armpl_status_t armpl_spmat_hint(armpl_spmat_t A, enum armpl_sparse_hint_type hint, enum armpl_sparse_hint_value value);

armpl_status_t armpl_spvec_create_s(armpl_spvec_t *x, armpl_int_t index_base, armpl_int_t n, armpl_int_t nnz, const armpl_int_t *indx, const float *vals, armpl_int_t flags);
armpl_status_t armpl_spvec_create_d(armpl_spvec_t *x, armpl_int_t index_base, armpl_int_t n, armpl_int_t nnz, const armpl_int_t *indx, const double *vals, armpl_int_t flags);
armpl_status_t armpl_spvec_create_c(armpl_spvec_t *x, armpl_int_t index_base, armpl_int_t n, armpl_int_t nnz, const armpl_int_t *indx, const armpl_singlecomplex_t *vals, armpl_int_t flags);
armpl_status_t armpl_spvec_create_z(armpl_spvec_t *x, armpl_int_t index_base, armpl_int_t n, armpl_int_t nnz, const armpl_int_t *indx, const armpl_doublecomplex_t *vals, armpl_int_t flags);

armpl_status_t armpl_spvec_query(armpl_spvec_t x, armpl_int_t *index_base, armpl_int_t *n, armpl_int_t *nnz);

armpl_status_t armpl_spvec_destroy(armpl_spvec_t x);

armpl_status_t armpl_spvec_export_s(armpl_spvec_t x, armpl_int_t *index_base, armpl_int_t *n, armpl_int_t *nnz, armpl_int_t *indx, float *vals);
armpl_status_t armpl_spvec_export_d(armpl_spvec_t x, armpl_int_t *index_base, armpl_int_t *n, armpl_int_t *nnz, armpl_int_t *indx, double *vals);
armpl_status_t armpl_spvec_export_c(armpl_spvec_t x, armpl_int_t *index_base, armpl_int_t *n, armpl_int_t *nnz, armpl_int_t *indx, armpl_singlecomplex_t *vals);
armpl_status_t armpl_spvec_export_z(armpl_spvec_t x, armpl_int_t *index_base, armpl_int_t *n, armpl_int_t *nnz, armpl_int_t *indx, armpl_doublecomplex_t *vals);

armpl_status_t armpl_spvec_gather_s(const float *x_d, armpl_int_t index_base, armpl_int_t n, armpl_spvec_t *x_s, armpl_int_t flags);
armpl_status_t armpl_spvec_gather_d(const double *x_d, armpl_int_t index_base, armpl_int_t n, armpl_spvec_t *x_s, armpl_int_t flags);
armpl_status_t armpl_spvec_gather_c(const armpl_singlecomplex_t *x_d, armpl_int_t index_base, armpl_int_t n, armpl_spvec_t *x_s, armpl_int_t flags);
armpl_status_t armpl_spvec_gather_z(const armpl_doublecomplex_t *x_d, armpl_int_t index_base, armpl_int_t n, armpl_spvec_t *x_s, armpl_int_t flags);

armpl_status_t armpl_spvec_scatter_s(armpl_spvec_t x_s, float *x_d);
armpl_status_t armpl_spvec_scatter_d(armpl_spvec_t x_s, double *x_d);
armpl_status_t armpl_spvec_scatter_c(armpl_spvec_t x_s, armpl_singlecomplex_t *x_d);
armpl_status_t armpl_spvec_scatter_z(armpl_spvec_t x_s, armpl_doublecomplex_t *x_d);

armpl_status_t armpl_spvec_update_s(armpl_spvec_t x, armpl_int_t n_updates, const armpl_int_t *indx, const float *vals);
armpl_status_t armpl_spvec_update_d(armpl_spvec_t x, armpl_int_t n_updates, const armpl_int_t *indx, const double *vals);
armpl_status_t armpl_spvec_update_c(armpl_spvec_t x, armpl_int_t n_updates, const armpl_int_t *indx, const armpl_singlecomplex_t *vals);
armpl_status_t armpl_spvec_update_z(armpl_spvec_t x, armpl_int_t n_updates, const armpl_int_t *indx, const armpl_doublecomplex_t *vals);

armpl_status_t armpl_spdot_exec_s(armpl_spvec_t x, const float *y, float *result);
armpl_status_t armpl_spdot_exec_d(armpl_spvec_t x, const double *y, double *result);
armpl_status_t armpl_spdotu_exec_c(armpl_spvec_t x, const armpl_singlecomplex_t *y, armpl_singlecomplex_t *result);
armpl_status_t armpl_spdotu_exec_z(armpl_spvec_t x, const armpl_doublecomplex_t *y, armpl_doublecomplex_t *result);
armpl_status_t armpl_spdotc_exec_c(armpl_spvec_t x, const armpl_singlecomplex_t *y, armpl_singlecomplex_t *result);
armpl_status_t armpl_spdotc_exec_z(armpl_spvec_t x, const armpl_doublecomplex_t *y, armpl_doublecomplex_t *result);

armpl_status_t armpl_spaxpby_exec_s(const float alpha, armpl_spvec_t x, const float beta, float *y);
armpl_status_t armpl_spaxpby_exec_d(const double alpha, armpl_spvec_t x, const double beta, double *y);
armpl_status_t armpl_spaxpby_exec_c(const armpl_singlecomplex_t alpha, armpl_spvec_t x, const armpl_singlecomplex_t beta, armpl_singlecomplex_t *y);
armpl_status_t armpl_spaxpby_exec_z(const armpl_doublecomplex_t alpha, armpl_spvec_t x, const armpl_doublecomplex_t beta, armpl_doublecomplex_t *y);

armpl_status_t armpl_spwaxpby_exec_s(const float alpha, armpl_spvec_t x, const float beta, const float *y, float *w);
armpl_status_t armpl_spwaxpby_exec_d(const double alpha, armpl_spvec_t x, const double beta, const double *y, double *w);
armpl_status_t armpl_spwaxpby_exec_c(const armpl_singlecomplex_t alpha, armpl_spvec_t x, const armpl_singlecomplex_t beta, const armpl_singlecomplex_t *y, armpl_singlecomplex_t *w);
armpl_status_t armpl_spwaxpby_exec_z(const armpl_doublecomplex_t alpha, armpl_spvec_t x, const armpl_doublecomplex_t beta, const armpl_doublecomplex_t *y, armpl_doublecomplex_t *w);

armpl_status_t armpl_spmv_optimize(armpl_spmat_t A);
armpl_status_t armpl_spmm_optimize(enum armpl_sparse_hint_value transA, enum armpl_sparse_hint_value transB, enum armpl_sparse_hint_value alpha, armpl_spmat_t A, armpl_spmat_t B, enum armpl_sparse_hint_value beta, armpl_spmat_t C);
armpl_status_t armpl_spadd_optimize(enum armpl_sparse_hint_value transA, enum armpl_sparse_hint_value transB, enum armpl_sparse_hint_value alpha, armpl_spmat_t A, enum armpl_sparse_hint_value beta, armpl_spmat_t B, armpl_spmat_t C);
armpl_status_t armpl_spsv_optimize(armpl_spmat_t A);

armpl_status_t armpl_spmv_exec_s(enum armpl_sparse_hint_value trans, float alpha, armpl_spmat_t A, const float *x, float beta, float *y);
armpl_status_t armpl_spmv_exec_d(enum armpl_sparse_hint_value trans, double alpha, armpl_spmat_t A, const double *x, double beta, double *y);
armpl_status_t armpl_spmv_exec_c(enum armpl_sparse_hint_value trans, armpl_singlecomplex_t alpha, armpl_spmat_t A, const armpl_singlecomplex_t *x, armpl_singlecomplex_t beta, armpl_singlecomplex_t *y);
armpl_status_t armpl_spmv_exec_z(enum armpl_sparse_hint_value trans, armpl_doublecomplex_t alpha, armpl_spmat_t A, const armpl_doublecomplex_t *x, armpl_doublecomplex_t beta, armpl_doublecomplex_t *y);

armpl_status_t armpl_spmm_exec_s(enum armpl_sparse_hint_value transA, enum armpl_sparse_hint_value transB, float alpha, armpl_spmat_t A, armpl_spmat_t B, float beta, armpl_spmat_t C);
armpl_status_t armpl_spmm_exec_d(enum armpl_sparse_hint_value transA, enum armpl_sparse_hint_value transB, double alpha, armpl_spmat_t A, armpl_spmat_t B, double beta, armpl_spmat_t C);
armpl_status_t armpl_spmm_exec_c(enum armpl_sparse_hint_value transA, enum armpl_sparse_hint_value transB, armpl_singlecomplex_t alpha, armpl_spmat_t A, armpl_spmat_t B, armpl_singlecomplex_t beta, armpl_spmat_t C);
armpl_status_t armpl_spmm_exec_z(enum armpl_sparse_hint_value transA, enum armpl_sparse_hint_value transB, armpl_doublecomplex_t alpha, armpl_spmat_t A, armpl_spmat_t B, armpl_doublecomplex_t beta, armpl_spmat_t C);

armpl_status_t armpl_spadd_exec_s(enum armpl_sparse_hint_value transA, enum armpl_sparse_hint_value transB, float alpha, armpl_spmat_t A, float beta, armpl_spmat_t B, armpl_spmat_t C);
armpl_status_t armpl_spadd_exec_d(enum armpl_sparse_hint_value transA, enum armpl_sparse_hint_value transB, double alpha, armpl_spmat_t A, double beta, armpl_spmat_t B, armpl_spmat_t C);
armpl_status_t armpl_spadd_exec_c(enum armpl_sparse_hint_value transA, enum armpl_sparse_hint_value transB, armpl_singlecomplex_t alpha, armpl_spmat_t A, armpl_singlecomplex_t beta, armpl_spmat_t B, armpl_spmat_t C);
armpl_status_t armpl_spadd_exec_z(enum armpl_sparse_hint_value transA, enum armpl_sparse_hint_value transB, armpl_doublecomplex_t alpha, armpl_spmat_t A, armpl_doublecomplex_t beta, armpl_spmat_t B, armpl_spmat_t C);

armpl_status_t armpl_spsv_exec_s(enum armpl_sparse_hint_value transA, armpl_spmat_t A, float *x, float alpha, const float *y);
armpl_status_t armpl_spsv_exec_d(enum armpl_sparse_hint_value transA, armpl_spmat_t A, double *x, double alpha, const double *y);
armpl_status_t armpl_spsv_exec_c(enum armpl_sparse_hint_value transA, armpl_spmat_t A, armpl_singlecomplex_t *x, armpl_singlecomplex_t alpha, const armpl_singlecomplex_t *y);
armpl_status_t armpl_spsv_exec_z(enum armpl_sparse_hint_value transA, armpl_spmat_t A, armpl_doublecomplex_t *x, armpl_doublecomplex_t alpha, const armpl_doublecomplex_t *y);

armpl_status_t armpl_spmat_update_s(armpl_spmat_t A, armpl_int_t n_updates, const armpl_int_t *row_indx, const armpl_int_t *col_indx, const float *vals);
armpl_status_t armpl_spmat_update_d(armpl_spmat_t A, armpl_int_t n_updates, const armpl_int_t *row_indx, const armpl_int_t *col_indx, const double *vals);
armpl_status_t armpl_spmat_update_c(armpl_spmat_t A, armpl_int_t n_updates, const armpl_int_t *row_indx, const armpl_int_t *col_indx, const armpl_singlecomplex_t *vals);
armpl_status_t armpl_spmat_update_z(armpl_spmat_t A, armpl_int_t n_updates, const armpl_int_t *row_indx, const armpl_int_t *col_indx, const armpl_doublecomplex_t *vals);

armpl_status_t armpl_spmat_export_csr_s(armpl_const_spmat_t A, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t **row_ptr, armpl_int_t **col_indx, float **vals);
armpl_status_t armpl_spmat_export_csr_d(armpl_const_spmat_t A, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t **row_ptr, armpl_int_t **col_indx, double **vals);
armpl_status_t armpl_spmat_export_csr_c(armpl_const_spmat_t A, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t **row_ptr, armpl_int_t **col_indx, armpl_singlecomplex_t **vals);
armpl_status_t armpl_spmat_export_csr_z(armpl_const_spmat_t A, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t **row_ptr, armpl_int_t **col_indx, armpl_doublecomplex_t **vals);
armpl_status_t armpl_spmat_export_csc_s(armpl_const_spmat_t A, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t **row_indx, armpl_int_t **col_ptr, float **vals);
armpl_status_t armpl_spmat_export_csc_d(armpl_const_spmat_t A, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t **row_indx, armpl_int_t **col_ptr, double **vals);
armpl_status_t armpl_spmat_export_csc_c(armpl_const_spmat_t A, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t **row_indx, armpl_int_t **col_ptr, armpl_singlecomplex_t **vals);
armpl_status_t armpl_spmat_export_csc_z(armpl_const_spmat_t A, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t **row_indx, armpl_int_t **col_ptr, armpl_doublecomplex_t **vals);
armpl_status_t armpl_spmat_export_coo_s(armpl_const_spmat_t A, armpl_int_t *m, armpl_int_t *n, armpl_int_t *nnz, armpl_int_t **row_indx, armpl_int_t **col_indx, float **vals);
armpl_status_t armpl_spmat_export_coo_d(armpl_const_spmat_t A, armpl_int_t *m, armpl_int_t *n, armpl_int_t *nnz, armpl_int_t **row_indx, armpl_int_t **col_indx, double **vals);
armpl_status_t armpl_spmat_export_coo_c(armpl_const_spmat_t A, armpl_int_t *m, armpl_int_t *n, armpl_int_t *nnz, armpl_int_t **row_indx, armpl_int_t **col_indx, armpl_singlecomplex_t **vals);
armpl_status_t armpl_spmat_export_coo_z(armpl_const_spmat_t A, armpl_int_t *m, armpl_int_t *n, armpl_int_t *nnz, armpl_int_t **row_indx, armpl_int_t **col_indx, armpl_doublecomplex_t **vals);
armpl_status_t armpl_spmat_export_dense_s(armpl_const_spmat_t A, enum armpl_dense_layout layout, armpl_int_t *m, armpl_int_t *n, float **vals);
armpl_status_t armpl_spmat_export_dense_d(armpl_const_spmat_t A, enum armpl_dense_layout layout, armpl_int_t *m, armpl_int_t *n, double **vals);
armpl_status_t armpl_spmat_export_dense_c(armpl_const_spmat_t A, enum armpl_dense_layout layout, armpl_int_t *m, armpl_int_t *n, armpl_singlecomplex_t **vals);
armpl_status_t armpl_spmat_export_dense_z(armpl_const_spmat_t A, enum armpl_dense_layout layout, armpl_int_t *m, armpl_int_t *n, armpl_doublecomplex_t **vals);
armpl_status_t armpl_spmat_export_bsr_s(armpl_const_spmat_t A, enum armpl_dense_layout block_layout, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t *block_size, armpl_int_t **row_ptr, armpl_int_t **col_indx, float **vals);
armpl_status_t armpl_spmat_export_bsr_d(armpl_const_spmat_t A, enum armpl_dense_layout block_layout, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t *block_size, armpl_int_t **row_ptr, armpl_int_t **col_indx, double **vals);
armpl_status_t armpl_spmat_export_bsr_c(armpl_const_spmat_t A, enum armpl_dense_layout block_layout, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t *block_size, armpl_int_t **row_ptr, armpl_int_t **col_indx, armpl_singlecomplex_t **vals);
armpl_status_t armpl_spmat_export_bsr_z(armpl_const_spmat_t A, enum armpl_dense_layout block_layout, armpl_int_t index_base, armpl_int_t *m, armpl_int_t *n, armpl_int_t *block_size, armpl_int_t **row_ptr, armpl_int_t **col_indx, armpl_doublecomplex_t **vals);

armpl_status_t armpl_sprot_exec_s(armpl_spvec_t x, float *y, float c, float s);
armpl_status_t armpl_sprot_exec_d(armpl_spvec_t x, double *y, double c, double s);
armpl_status_t armpl_sprot_exec_c(armpl_spvec_t x, armpl_singlecomplex_t *y, float c, armpl_singlecomplex_t s);
armpl_status_t armpl_sprot_exec_cs(armpl_spvec_t x, armpl_singlecomplex_t *y, float c, float s);
armpl_status_t armpl_sprot_exec_z(armpl_spvec_t x, armpl_doublecomplex_t *y, double c, armpl_doublecomplex_t s);
armpl_status_t armpl_sprot_exec_zd(armpl_spvec_t x, armpl_doublecomplex_t *y, double c, double s);

void armpl_spmat_print_err(armpl_spmat_t A);
void armpl_spvec_print_err(armpl_spvec_t x);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

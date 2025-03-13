/*
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 * SPDX-FileCopyrightText: Copyright 2015-2024 NAG
 */

#ifndef ARMPL_XBLAS_H
#define ARMPL_XBLAS_H

#ifndef BLAS_ENUM_H
#define BLAS_ENUM_H

/* Enumerated types */

enum blas_order_type {
            blas_rowmajor = 101,
            blas_colmajor = 102 };

enum blas_trans_type {
            blas_no_trans   = 111,
            blas_trans      = 112,
            blas_conj_trans = 113 };

enum blas_uplo_type  {
            blas_upper = 121,
            blas_lower = 122 };

enum blas_diag_type {
            blas_non_unit_diag = 131,
            blas_unit_diag     = 132 };

enum blas_side_type {
            blas_left_side  = 141,
            blas_right_side = 142 };

enum blas_cmach_type {
            blas_base      = 151,
            blas_t         = 152,
            blas_rnd       = 153,
            blas_ieee      = 154,
            blas_emin      = 155,
            blas_emax      = 156,
            blas_eps       = 157,
            blas_prec      = 158,
            blas_underflow = 159,
            blas_overflow  = 160,
            blas_sfmin     = 161};

enum blas_norm_type {
            blas_one_norm       = 171,
            blas_real_one_norm  = 172,
            blas_two_norm       = 173,
            blas_frobenius_norm = 174,
            blas_inf_norm       = 175,
            blas_real_inf_norm  = 176,
            blas_max_norm       = 177,
            blas_real_max_norm  = 178 };

enum blas_sort_type {
            blas_increasing_order = 181,
            blas_decreasing_order = 182 };

enum blas_conj_type {
            blas_conj    = 191,
            blas_no_conj = 192 };

enum blas_jrot_type {
            blas_jrot_inner  = 201,
            blas_jrot_outer  = 202,
            blas_jrot_sorted = 203 };

enum blas_prec_type {
            blas_prec_single     = 211,
            blas_prec_double     = 212,
            blas_prec_indigenous = 213,
            blas_prec_extra      = 214 };

enum blas_base_type {
            blas_zero_base = 221,
            blas_one_base  = 222 };

enum blas_symmetry_type {
            blas_general          = 231,
            blas_symmetric        = 232,
            blas_hermitian        = 233,
            blas_triangular       = 234,
            blas_lower_triangular = 235,
            blas_upper_triangular = 236 };

enum blas_field_type {
            blas_complex          = 241,
            blas_real             = 242,
            blas_double_precision = 243,
            blas_single_precision = 244  };

enum blas_size_type {
            blas_num_rows      = 251,
            blas_num_cols      = 252,
            blas_num_nonzeros  = 253  };

enum blas_handle_type{
            blas_invalid_handle = 261,
			blas_new_handle     = 262,
			blas_open_handle    = 263,
			blas_closed_handle  = 264};

enum blas_sparsity_optimization_type {
            blas_regular       = 271,
            blas_irregular     = 272,
            blas_block         = 273,
            blas_unassembled   = 274 };

void BLAS_ddot_d_s(enum blas_conj_type conj, armpl_int_t n, double alpha, const double *x, armpl_int_t incx, double beta, const float *y, armpl_int_t incy, double *r);
void BLAS_ddot_s_d(enum blas_conj_type conj, armpl_int_t n, double alpha, const float *x, armpl_int_t incx, double beta, const double *y, armpl_int_t incy, double *r);
void BLAS_ddot_s_s(enum blas_conj_type conj, armpl_int_t n, double alpha, const float *x, armpl_int_t incx, double beta, const float *y, armpl_int_t incy, double *r);
void BLAS_zdot_z_c(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r);
void BLAS_zdot_c_z(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r);
void BLAS_zdot_c_c(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r);
void BLAS_cdot_c_s(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const float *y, armpl_int_t incy, void *r);
void BLAS_cdot_s_c(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const float *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r);
void BLAS_cdot_s_s(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const float *x, armpl_int_t incx, const void *beta, const float *y, armpl_int_t incy, void *r);
void BLAS_zdot_z_d(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const double *y, armpl_int_t incy, void *r);
void BLAS_zdot_d_z(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const double *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r);
void BLAS_zdot_d_d(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const double *x, armpl_int_t incx, const void *beta, const double *y, armpl_int_t incy, void *r);
void BLAS_sdot_x(enum blas_conj_type conj, armpl_int_t n, float alpha, const float *x, armpl_int_t incx, float beta, const float *y, armpl_int_t incy, float *r, enum blas_prec_type prec);
void BLAS_ddot_x(enum blas_conj_type conj, armpl_int_t n, double alpha, const double *x, armpl_int_t incx, double beta, const double *y, armpl_int_t incy, double *r, enum blas_prec_type prec);
void BLAS_cdot_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r, enum blas_prec_type prec);
void BLAS_zdot_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r, enum blas_prec_type prec);
void BLAS_ddot_d_s_x(enum blas_conj_type conj, armpl_int_t n, double alpha, const double *x, armpl_int_t incx, double beta, const float *y, armpl_int_t incy, double *r, enum blas_prec_type prec);
void BLAS_ddot_s_d_x(enum blas_conj_type conj, armpl_int_t n, double alpha, const float *x, armpl_int_t incx, double beta, const double *y, armpl_int_t incy, double *r, enum blas_prec_type prec);
void BLAS_ddot_s_s_x(enum blas_conj_type conj, armpl_int_t n, double alpha, const float *x, armpl_int_t incx, double beta, const float *y, armpl_int_t incy, double *r, enum blas_prec_type prec);
void BLAS_zdot_z_c_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r, enum blas_prec_type prec);
void BLAS_zdot_c_z_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r, enum blas_prec_type prec);
void BLAS_zdot_c_c_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r, enum blas_prec_type prec);
void BLAS_cdot_c_s_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const float *y, armpl_int_t incy, void *r, enum blas_prec_type prec);
void BLAS_cdot_s_c_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const float *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r, enum blas_prec_type prec);
void BLAS_cdot_s_s_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const float *x, armpl_int_t incx, const void *beta, const float *y, armpl_int_t incy, void *r, enum blas_prec_type prec);
void BLAS_zdot_z_d_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const double *y, armpl_int_t incy, void *r, enum blas_prec_type prec);
void BLAS_zdot_d_z_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const double *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *r, enum blas_prec_type prec);
void BLAS_zdot_d_d_x(enum blas_conj_type conj, armpl_int_t n, const void *alpha, const double *x, armpl_int_t incx, const void *beta, const double *y, armpl_int_t incy, void *r, enum blas_prec_type prec);


void BLAS_ssum_x(armpl_int_t n, const float *x, armpl_int_t incx, float *sum, enum blas_prec_type prec);
void BLAS_dsum_x(armpl_int_t n, const double *x, armpl_int_t incx, double *sum, enum blas_prec_type prec);
void BLAS_csum_x(armpl_int_t n, const void *x, armpl_int_t incx, void *sum, enum blas_prec_type prec);
void BLAS_zsum_x(armpl_int_t n, const void *x, armpl_int_t incx, void *sum, enum blas_prec_type prec);


void BLAS_daxpby_s(armpl_int_t n, double alpha, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_caxpby_s(armpl_int_t n, const void *alpha, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zaxpby_c(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zaxpby_d(armpl_int_t n, const void *alpha, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_saxpby_x(armpl_int_t n, float alpha, const float *x, armpl_int_t incx, float beta, float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_daxpby_x(armpl_int_t n, double alpha, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_caxpby_x(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zaxpby_x(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_daxpby_s_x(armpl_int_t n, double alpha, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zaxpby_c_x(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_caxpby_s_x(armpl_int_t n, const void *alpha, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zaxpby_d_x(armpl_int_t n, const void *alpha, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_dwaxpby_d_s(armpl_int_t n, double alpha, const double *x, armpl_int_t incx, double beta, const float *y, armpl_int_t incy, double *w, armpl_int_t incw);
void BLAS_dwaxpby_s_d(armpl_int_t n, double alpha, const float *x, armpl_int_t incx, double beta, const double *y, armpl_int_t incy, double *w, armpl_int_t incw);
void BLAS_dwaxpby_s_s(armpl_int_t n, double alpha, const float *x, armpl_int_t incx, double beta, const float *y, armpl_int_t incy, double *w, armpl_int_t incw);
void BLAS_zwaxpby_z_c(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw);
void BLAS_zwaxpby_c_z(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw);
void BLAS_zwaxpby_c_c(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw);
void BLAS_cwaxpby_c_s(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const float *y, armpl_int_t incy, void *w, armpl_int_t incw);
void BLAS_cwaxpby_s_c(armpl_int_t n, const void *alpha, const float *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw);
void BLAS_cwaxpby_s_s(armpl_int_t n, const void *alpha, const float *x, armpl_int_t incx, const void *beta, const float *y, armpl_int_t incy, void *w, armpl_int_t incw);
void BLAS_zwaxpby_z_d(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const double *y, armpl_int_t incy, void *w, armpl_int_t incw);
void BLAS_zwaxpby_d_z(armpl_int_t n, const void *alpha, const double *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw);
void BLAS_zwaxpby_d_d(armpl_int_t n, const void *alpha, const double *x, armpl_int_t incx, const void *beta, const double *y, armpl_int_t incy, void *w, armpl_int_t incw);
void BLAS_swaxpby_x(armpl_int_t n, float alpha, const float *x, armpl_int_t incx, float beta, const float *y, armpl_int_t incy, float *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_dwaxpby_x(armpl_int_t n, double alpha, const double *x, armpl_int_t incx, double beta, const double *y, armpl_int_t incy, double *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_cwaxpby_x(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_zwaxpby_x(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_dwaxpby_d_s_x(armpl_int_t n, double alpha, const double *x, armpl_int_t incx, double beta, const float *y, armpl_int_t incy, double *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_dwaxpby_s_d_x(armpl_int_t n, double alpha, const float *x, armpl_int_t incx, double beta, const double *y, armpl_int_t incy, double *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_dwaxpby_s_s_x(armpl_int_t n, double alpha, const float *x, armpl_int_t incx, double beta, const float *y, armpl_int_t incy, double *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_zwaxpby_z_c_x(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_zwaxpby_c_z_x(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_zwaxpby_c_c_x(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_cwaxpby_c_s_x(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const float *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_cwaxpby_s_c_x(armpl_int_t n, const void *alpha, const float *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_cwaxpby_s_s_x(armpl_int_t n, const void *alpha, const float *x, armpl_int_t incx, const void *beta, const float *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_zwaxpby_z_d_x(armpl_int_t n, const void *alpha, const void *x, armpl_int_t incx, const void *beta, const double *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_zwaxpby_d_z_x(armpl_int_t n, const void *alpha, const double *x, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);
void BLAS_zwaxpby_d_d_x(armpl_int_t n, const void *alpha, const double *x, armpl_int_t incx, const void *beta, const double *y, armpl_int_t incy, void *w, armpl_int_t incw, enum blas_prec_type prec);


void BLAS_dgemv_d_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dgemv_s_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dgemv_s_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_zgemv_z_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgemv_c_z(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgemv_c_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgemv_c_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgemv_s_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgemv_s_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgemv_z_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgemv_d_z(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgemv_d_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_sgemv_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, float alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, float beta, float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgemv_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgemv_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgemv_d_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgemv_s_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgemv_s_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv_z_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv_c_z_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv_c_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgemv_c_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgemv_s_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgemv_s_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv_z_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv_d_z_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv_d_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_dge_sum_mv_d_s(enum blas_order_type order, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, const double *b, armpl_int_t ldb, double *y, armpl_int_t incy);
void BLAS_dge_sum_mv_s_d(enum blas_order_type order, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, const float *b, armpl_int_t ldb, double *y, armpl_int_t incy);
void BLAS_dge_sum_mv_s_s(enum blas_order_type order, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, const float *b, armpl_int_t ldb, double *y, armpl_int_t incy);
void BLAS_zge_sum_mv_z_c(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy);
void BLAS_zge_sum_mv_c_z(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy);
void BLAS_zge_sum_mv_c_c(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy);
void BLAS_cge_sum_mv_c_s(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy);
void BLAS_cge_sum_mv_s_c(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const float *b, armpl_int_t ldb, void *y, armpl_int_t incy);
void BLAS_cge_sum_mv_s_s(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, const float *b, armpl_int_t ldb, void *y, armpl_int_t incy);
void BLAS_zge_sum_mv_z_d(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy);
void BLAS_zge_sum_mv_d_z(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const double *b, armpl_int_t ldb, void *y, armpl_int_t incy);
void BLAS_zge_sum_mv_d_d(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, const double *b, armpl_int_t ldb, void *y, armpl_int_t incy);
void BLAS_sge_sum_mv_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, float alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, float beta, const float *b, armpl_int_t ldb, float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dge_sum_mv_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, const double *b, armpl_int_t ldb, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cge_sum_mv_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zge_sum_mv_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dge_sum_mv_d_s_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, const double *b, armpl_int_t ldb, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dge_sum_mv_s_d_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, const float *b, armpl_int_t ldb, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dge_sum_mv_s_s_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, const float *b, armpl_int_t ldb, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zge_sum_mv_z_c_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zge_sum_mv_c_z_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zge_sum_mv_c_c_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cge_sum_mv_c_s_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cge_sum_mv_s_c_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const float *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cge_sum_mv_s_s_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, const float *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zge_sum_mv_z_d_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, const void *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zge_sum_mv_d_z_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, const double *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zge_sum_mv_d_d_x(enum blas_order_type order, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, const double *b, armpl_int_t ldb, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_dgbmv_d_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const double *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dgbmv_s_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const float *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dgbmv_s_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_zgbmv_z_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgbmv_c_z(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgbmv_c_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgbmv_c_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgbmv_s_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const float *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgbmv_s_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgbmv_z_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgbmv_d_z(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const double *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgbmv_d_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_sgbmv_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, float alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, float beta, float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgbmv_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgbmv_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgbmv_d_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const double *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgbmv_s_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const float *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgbmv_s_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv_z_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv_c_z_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv_c_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgbmv_c_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgbmv_s_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const float *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgbmv_s_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv_z_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv_d_z_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const double *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv_d_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_dsymv_d_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dsymv_s_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dsymv_s_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_zsymv_z_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsymv_c_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsymv_c_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_csymv_c_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_csymv_s_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_csymv_s_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsymv_z_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsymv_d_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsymv_d_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_ssymv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, float alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, float beta, float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsymv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csymv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsymv_d_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsymv_s_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsymv_s_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv_z_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv_c_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv_c_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csymv_c_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csymv_s_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csymv_s_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv_z_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv_d_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv_d_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_dspmv_d_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const double *ap, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dspmv_s_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *ap, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dspmv_s_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *ap, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_zspmv_z_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zspmv_c_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zspmv_c_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cspmv_c_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cspmv_s_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cspmv_s_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *ap, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zspmv_z_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zspmv_d_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zspmv_d_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *ap, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_sspmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, float alpha, const float *ap, const float *x, armpl_int_t incx, float beta, float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dspmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const double *ap, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cspmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zspmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dspmv_d_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const double *ap, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dspmv_s_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *ap, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dspmv_s_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *ap, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zspmv_z_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zspmv_c_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zspmv_c_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cspmv_c_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cspmv_s_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cspmv_s_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *ap, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zspmv_z_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zspmv_d_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zspmv_d_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *ap, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_dsbmv_d_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, double alpha, const double *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dsbmv_s_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, double alpha, const float *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dsbmv_s_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, double alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_zsbmv_z_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsbmv_c_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsbmv_c_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_csbmv_c_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_csbmv_s_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const float *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_csbmv_s_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsbmv_z_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsbmv_d_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const double *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsbmv_d_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_ssbmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, float alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, float beta, float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsbmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, double alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csbmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsbmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsbmv_d_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, double alpha, const double *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsbmv_s_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, double alpha, const float *a, armpl_int_t lda, const double *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsbmv_s_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, double alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsbmv_z_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsbmv_c_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsbmv_c_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csbmv_c_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csbmv_s_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const float *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csbmv_s_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const float *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsbmv_z_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsbmv_d_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const double *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsbmv_d_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const double *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_zhemv_z_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zhemv_c_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zhemv_c_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_chemv_c_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zhemv_z_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_chemv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhemv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhemv_z_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhemv_c_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhemv_c_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_chemv_c_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhemv_z_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_zhpmv_z_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zhpmv_c_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zhpmv_c_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_chpmv_c_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zhpmv_z_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_chpmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhpmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhpmv_z_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhpmv_c_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhpmv_c_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_chpmv_c_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhpmv_z_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *ap, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_zhbmv_z_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zhbmv_c_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zhbmv_c_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_chbmv_c_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zhbmv_z_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_chbmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhbmv_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhbmv_z_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhbmv_c_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhbmv_c_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_chbmv_c_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const float *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhbmv_z_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const double *x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_dtrmv_s(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, double alpha, const float *T, armpl_int_t ldt, double *x, armpl_int_t incx);
void BLAS_ztrmv_c(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *T, armpl_int_t ldt, void *x, armpl_int_t incx);
void BLAS_ctrmv_s(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const float *T, armpl_int_t ldt, void *x, armpl_int_t incx);
void BLAS_ztrmv_d(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const double *T, armpl_int_t ldt, void *x, armpl_int_t incx);
void BLAS_strmv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, float alpha, const float *T, armpl_int_t ldt, float *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_dtrmv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, double alpha, const double *T, armpl_int_t ldt, double *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ctrmv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *T, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztrmv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *T, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_dtrmv_s_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, double alpha, const float *T, armpl_int_t ldt, double *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztrmv_c_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *T, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ctrmv_s_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const float *T, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztrmv_d_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const double *T, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);


void BLAS_dtpmv_s(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, double alpha, const float *tp, double *x, armpl_int_t incx);
void BLAS_ztpmv_c(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *tp, void *x, armpl_int_t incx);
void BLAS_ctpmv_s(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const float *tp, void *x, armpl_int_t incx);
void BLAS_ztpmv_d(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const double *tp, void *x, armpl_int_t incx);
void BLAS_stpmv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, float alpha, const float *tp, float *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_dtpmv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, double alpha, const double *tp, double *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ctpmv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *tp, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztpmv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *tp, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_dtpmv_s_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, double alpha, const float *tp, double *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztpmv_c_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *tp, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ctpmv_s_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const float *tp, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztpmv_d_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const double *tp, void *x, armpl_int_t incx, enum blas_prec_type prec);


void BLAS_dtrsv_s(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, double alpha, const float *T, armpl_int_t ldt, double *x, armpl_int_t incx);
void BLAS_ztrsv_c(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *T, armpl_int_t ldt, void *x, armpl_int_t incx);
void BLAS_ctrsv_s(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const float *T, armpl_int_t ldt, void *x, armpl_int_t incx);
void BLAS_ztrsv_d(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const double *T, armpl_int_t ldt, void *x, armpl_int_t incx);
void BLAS_strsv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, float alpha, const float *T, armpl_int_t ldt, float *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_dtrsv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, double alpha, const double *T, armpl_int_t ldt, double *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_dtrsv_s_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, double alpha, const float *T, armpl_int_t ldt, double *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ctrsv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *T, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztrsv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *T, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztrsv_c_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const void *T, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ctrsv_s_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const float *T, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztrsv_d_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, const void *alpha, const double *T, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);


void BLAS_dtbsv_s(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, double alpha, const float *t, armpl_int_t ldt, double *x, armpl_int_t incx);
void BLAS_ztbsv_c(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, const void *alpha, const void *t, armpl_int_t ldt, void *x, armpl_int_t incx);
void BLAS_ctbsv_s(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, const void *alpha, const float *t, armpl_int_t ldt, void *x, armpl_int_t incx);
void BLAS_ztbsv_d(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, const void *alpha, const double *t, armpl_int_t ldt, void *x, armpl_int_t incx);
void BLAS_stbsv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, float alpha, const float *t, armpl_int_t ldt, float *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_dtbsv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, double alpha, const double *t, armpl_int_t ldt, double *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_dtbsv_s_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, double alpha, const float *t, armpl_int_t ldt, double *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ctbsv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, const void *alpha, const void *t, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztbsv_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, const void *alpha, const void *t, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztbsv_c_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, const void *alpha, const void *t, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ctbsv_s_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, const void *alpha, const float *t, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);
void BLAS_ztbsv_d_x(enum blas_order_type order, enum blas_uplo_type uplo, enum blas_trans_type trans, enum blas_diag_type diag, armpl_int_t n, armpl_int_t k, const void *alpha, const double *t, armpl_int_t ldt, void *x, armpl_int_t incx, enum blas_prec_type prec);


void BLAS_dgemm_d_s(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, double alpha, const double *a, armpl_int_t lda, const float *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc);
void BLAS_dgemm_s_d(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, double alpha, const float *a, armpl_int_t lda, const double *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc);
void BLAS_dgemm_s_s(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, double alpha, const float *a, armpl_int_t lda, const float *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc);
void BLAS_zgemm_z_c(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zgemm_c_z(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zgemm_c_c(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_cgemm_c_s(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const float *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_cgemm_s_c(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const float *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_cgemm_s_s(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const float *a, armpl_int_t lda, const float *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zgemm_z_d(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const double *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zgemm_d_z(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const double *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zgemm_d_d(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const double *a, armpl_int_t lda, const double *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_sgemm_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, float alpha, const float *a, armpl_int_t lda, const float *b, armpl_int_t ldb, float beta, float *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_dgemm_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, double alpha, const double *a, armpl_int_t lda, const double *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_cgemm_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zgemm_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_dgemm_d_s_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, double alpha, const double *a, armpl_int_t lda, const float *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_dgemm_s_d_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, double alpha, const float *a, armpl_int_t lda, const double *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_dgemm_s_s_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, double alpha, const float *a, armpl_int_t lda, const float *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zgemm_z_c_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zgemm_c_z_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zgemm_c_c_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_cgemm_c_s_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const float *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_cgemm_s_c_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const float *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_cgemm_s_s_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const float *a, armpl_int_t lda, const float *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zgemm_z_d_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const void *a, armpl_int_t lda, const double *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zgemm_d_z_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const double *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zgemm_d_d_x(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb, armpl_int_t m, armpl_int_t n, armpl_int_t k, const void *alpha, const double *a, armpl_int_t lda, const double *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);


void BLAS_dsymm_d_s(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc);
void BLAS_dsymm_s_d(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc);
void BLAS_dsymm_s_s(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc);
void BLAS_zsymm_z_c(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zsymm_c_z(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zsymm_c_c(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_csymm_c_s(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_csymm_s_c(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_csymm_s_s(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zsymm_z_d(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zsymm_d_z(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zsymm_d_d(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_ssymm_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, float alpha, const float *a, armpl_int_t lda, const float *b, armpl_int_t ldb, float beta, float *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_dsymm_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const double *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_csymm_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zsymm_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_dsymm_d_s_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_dsymm_s_d_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_dsymm_s_s_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *b, armpl_int_t ldb, double beta, double *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zsymm_z_c_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zsymm_c_z_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zsymm_c_c_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_csymm_c_s_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_csymm_s_c_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_csymm_s_s_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zsymm_z_d_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zsymm_d_z_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zsymm_d_d_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);


void BLAS_zhemm_z_c(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zhemm_c_z(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zhemm_c_c(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_chemm_c_s(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_zhemm_z_d(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc);
void BLAS_chemm_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zhemm_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zhemm_z_c_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zhemm_c_z_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zhemm_c_c_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_chemm_c_s_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);
void BLAS_zhemm_z_d_x(enum blas_order_type order, enum blas_side_type side, enum blas_uplo_type uplo, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *b, armpl_int_t ldb, const void *beta, void *c, armpl_int_t ldc, enum blas_prec_type prec);


void BLAS_dgemv2_d_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dgemv2_s_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dgemv2_s_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_zgemv2_z_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgemv2_c_z(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgemv2_c_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgemv2_c_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgemv2_s_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgemv2_s_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgemv2_z_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgemv2_d_z(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgemv2_d_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_sgemv2_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, float alpha, const float *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, float beta, float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgemv2_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgemv2_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv2_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgemv2_d_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgemv2_s_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgemv2_s_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv2_z_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv2_c_z_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv2_c_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgemv2_c_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgemv2_s_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgemv2_s_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv2_z_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv2_d_z_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgemv2_d_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_dsymv2_d_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dsymv2_s_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *x_head, const double *x_tail, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dsymv2_s_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_zsymv2_z_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsymv2_c_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsymv2_c_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_csymv2_c_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_csymv2_s_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_csymv2_s_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsymv2_z_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x_head, const double *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsymv2_d_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zsymv2_d_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *x_head, const double *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_ssymv2_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, float alpha, const float *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, float beta, float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsymv2_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const double *x_head, const double *x_tail, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csymv2_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv2_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsymv2_d_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const double *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsymv2_s_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const double *x_head, const double *x_tail, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dsymv2_s_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, double alpha, const float *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv2_z_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv2_c_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv2_c_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csymv2_c_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csymv2_s_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_csymv2_s_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const float *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv2_z_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x_head, const double *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv2_d_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zsymv2_d_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const double *a, armpl_int_t lda, const double *x_head, const double *x_tail, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_zhemv2_z_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy);
void BLAS_zhemv2_c_z(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy);
void BLAS_zhemv2_c_c(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy);
void BLAS_chemv2_c_s(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, const void *beta, const float *y, armpl_int_t incy);
void BLAS_zhemv2_z_d(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x_head, const double *x_tail, armpl_int_t incx, const void *beta, const double *y, armpl_int_t incy);
void BLAS_chemv2_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhemv2_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhemv2_z_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhemv2_c_z_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhemv2_c_c_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const void *x_head, const void *x_tail, armpl_int_t incx, const void *beta, const void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_chemv2_c_s_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const float *x_head, const float *x_tail, armpl_int_t incx, const void *beta, const float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zhemv2_z_d_x(enum blas_order_type order, enum blas_uplo_type uplo, armpl_int_t n, const void *alpha, const void *a, armpl_int_t lda, const double *x_head, const double *x_tail, armpl_int_t incx, const void *beta, const double *y, armpl_int_t incy, enum blas_prec_type prec);


void BLAS_dgbmv2_d_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const double *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dgbmv2_s_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const float *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_dgbmv2_s_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const float *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy);
void BLAS_zgbmv2_z_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgbmv2_c_z(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgbmv2_c_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgbmv2_c_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgbmv2_s_c(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const float *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_cgbmv2_s_s(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const float *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgbmv2_z_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgbmv2_d_z(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const double *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_zgbmv2_d_d(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const double *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy);
void BLAS_sgbmv2_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, float alpha, const float *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, float beta, float *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgbmv2_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const double *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgbmv2_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv2_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgbmv2_d_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const double *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgbmv2_s_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const float *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_dgbmv2_s_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, double alpha, const float *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, double beta, double *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv2_z_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv2_c_z_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv2_c_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgbmv2_c_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgbmv2_s_c_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const float *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_cgbmv2_s_s_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const float *a, armpl_int_t lda, const float *head_x, const float *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv2_z_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const void *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv2_d_z_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const double *a, armpl_int_t lda, const void *head_x, const void *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
void BLAS_zgbmv2_d_d_x(enum blas_order_type order, enum blas_trans_type trans, armpl_int_t m, armpl_int_t n, armpl_int_t kl, armpl_int_t ku, const void *alpha, const double *a, armpl_int_t lda, const double *head_x, const double *tail_x, armpl_int_t incx, const void *beta, void *y, armpl_int_t incy, enum blas_prec_type prec);
armpl_int_t BLAS_fpinfo_x(enum blas_cmach_type cmach, enum blas_prec_type prec);
void BLAS_error(const char *rname, armpl_int_t iflag, armpl_int_t ival, const char *form, ...);

#endif /* BLAS_ENUM_H */
#endif /* ARMPL_XBLAS_H */

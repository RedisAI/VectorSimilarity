#ifndef ARMPL_INTERLEAVE_BATCH_H
#define ARMPL_INTERLEAVE_BATCH_H

#include "armpl_status.h"
#include "armpl_int.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

/*
  User-visible naming conventions:
    Matrices are uppercase, vectors lowercase
    Pointers to BLAS matrices use the identifier only (A), as in BLAS
    Pointers to Interleaved-batch structures are appended with _p (A_p) (similar to MKL's convention)
    Column strides are istrd_x
    Row strides are jstrd_x
	Vectors have column strides, but not row strides
    Batch strides are bstrd_x

  These conventions are not enforced beyond the header file.
*/

/**
 *  Copies a single matrix @f$A@f$ with arbitrary column and row strides into into @f$B^{(l)}@f$, i.e. the @f$l^{\textrm{th}}@f$ matrix in an
 *  interleaved data structure consisting of @f$B^{(i)}@f$ for @f$i = 1, .. ninter@f$. Users should call this function iteratively to
 *  build up interleaved-batch data structures.
 *
 * @param ninter   [in] The number of matrices interleaved. This should be a small multiple of your machine's native vector length.
 * @param l        [in] The index of the matrix in the interleaving, 0 <= l < ninter.
 * @param m        [in] The number of rows of each of the matrices @f$A@f$ and @f$B^{(l)}@f$.
 * @param n        [in] The number of columns of each of the matrices @f$A@f$ and @f$B^{(l)}@f$.
 * @param A        [in] The input matrix.
 * @param istrd_A  [in] The stride between successive elements in the same column of the matrix in A.
 * @param jstrd_A  [in] The stride between successive elements in the same row of the matrix in A.
 * @param B_p      [out] Pointer to the start of the array which contains the matrix @f$B^{(l)}@f$ laid out in interleaved-batch format.
 * @param istrd_B  [in] The stride between successive elements in the same column of the same matrix in B_p.
 * @param jstrd_B  [in] The stride between successive elements in the same row of the same matrix in B_p.
 * @return Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t
armpl_dge_interleave(armpl_int_t ninter, armpl_int_t l,
                     armpl_int_t m, armpl_int_t n,
                     const double *A, armpl_int_t istrd_A, armpl_int_t jstrd_A,
                     double *B_p,     armpl_int_t istrd_B, armpl_int_t jstrd_B);

/**
 * Extracts a single matrix @f$A@f$ from @f$B^{(l)}@f$, i.e. the @f$l^{\textrm{th}}@f$ matrix in an interleaved data structure consisting of @f$B^{(i)}@f$
 * for @f$i = 1, .. ninter@f$. @f$A@f$ may be written with arbitrary row and column strides. Users should call this function iteratively to
 * extract each matrix stored within interleaved-batch data structures.
 *
 * @param ninter      [in] The number of matrices interleaved. This should be a small multiple of your machine's native vector length.
 * @param l           [in] The index of the matrix in the interleaving, 0 <= l < ninter.
 * @param m           [in] The number of rows of each of the matrices @f$A@f$ and @f$B^{(l)}@f$.
 * @param n           [in] The number of columns of each of the matrices @f$A@f$ and @f$B^{(l)}@f$.
 * @param A           [out] The output matrix.
 * @param istrd_A     [in] The stride between successive elements in the same column of the matrix in A.
 * @param jstrd_A     [in] The stride between successive elements in the same row of the matrix in A.
 * @param B_p         [in] Pointer to the start of the array which contains the matrix @f$B^{(l)}@f$ laid out in interleaved-batch format.
 * @param istrd_B     [in] The stride between successive elements in the same column of the same matrix in B_p.
 * @param jstrd_B     [in] The stride between successive elements in the same row of the same matrix in B_p.
 * @return Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dge_deinterleave(armpl_int_t ninter, armpl_int_t l,
                                      armpl_int_t m, armpl_int_t n,
                                            double *A,   armpl_int_t istrd_A, armpl_int_t jstrd_A,
                                      const double *B_p, armpl_int_t istrd_B, armpl_int_t jstrd_B);

/**
 * Computes:
 *   @f[
        \zeta^{(i)} = \sum_{j=1}^n x_j^{(i)} y_j^{(i)}
 *   @f]
 * @f$\zeta^{(i)}@f$ are scalars and @f$x^{(i)}@f$ and @f$y^{(i)}@f$ are vectors, for @f$i = 1, .. np@f$.
 *
 * @param ninter  [in] The number of matrices interleaved. This should be a small multiple of your machine's native vector length.
 * @param nbatch  [in] The number of batches of ninter matrices. Note that np = ninter*nbatch.
 * @param n       [in] The length of each vector.
 * @param x_p     [in] Pointer to the start of the array which contains vectors @f$x^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_x [in] The stride between corresponding elements in successive batches in x_p.
 * @param istrd_x [in] The stride between successive elements in the same vector in x_p.
 * @param y_p     [in] Pointer to the start of the array which contains vectors @f$y^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_y [in] The stride between corresponding elements in successive batches in y_p.
 * @param istrd_y [in] The stride between successive elements in the same vector in y_p.
 * @param z_p     [out] Pointer to @f$np@f$ dot products, @f$\zeta^{(i)}@f$. Must be an array of length at least bstrd_z*nbatch elements allocated by the user.
 * @param bstrd_z [in] The stride between corresponding results in successive batches in z_p.
 * @return Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_ddot_interleave_batch(armpl_int_t ninter, armpl_int_t nbatch, armpl_int_t n,
                                           const double *x_p, armpl_int_t bstrd_x, armpl_int_t istrd_x,
                                           const double *y_p, armpl_int_t bstrd_y, armpl_int_t istrd_y,
                                                 double *z_p, armpl_int_t bstrd_z);

/**
 * Computes:
 *   @f[
        A^{(i)} = \alpha x^{(i)} (y^{(i)})^T + A^{(i)}
 *   @f]
 * @f$A^{(i)}@f$ are matrices, @f$x^{(i)}@f$ and @f$y^{(i)}@f$ are vectors, for @f$i = 1, .. np@f$, and @f$\alpha@f$ is a scalar.
 *
 * @param ninter  [in] The number of matrices interleaved. This should be a small multiple of your machine's native vector length.
 * @param nbatch  [in] The number of batches of ninter matrices. Note that np = ninter*nbatch.
 * @param m       [in] The number of rows of each of the matrices @f$A^{(i)}@f$.
 * @param n       [in] The number of columns of each of the matrices @f$A^{(i)}@f$.
 * @param alpha   [in] The scalar parameter @f$\alpha@f$.
 * @param x_p     [in] Pointer to the start of the array which contains vectors @f$x^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_x [in] The stride between corresponding elements in successive batches in x_p.
 * @param istrd_x [in] The stride between successive elements in the same vector in x_p.
 * @param y_p     [in] Pointer to the start of the array which contains vectors @f$y^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_y [in] The stride between corresponding elements in successive batches in y_p.
 * @param istrd_y [in] The stride between successive elements in the same vector in y_p.
 * @param A_p     [in,out] Pointer to the start of the interleaved matrix A. On exit A is overwritten by the updated interleave batch matrix.
 * @param bstrd_A [in] The stride between corresponding elements in successive batches in A_p.
 * @param istrd_A [in] The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrd_A [in] The stride between successive elements in the same row of the same matrix in A_p.
 * @return Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dger_interleave_batch(
	armpl_int_t ninter,
	armpl_int_t nbatch,
	armpl_int_t m, armpl_int_t n,
	double alpha,
	const double *x_p, armpl_int_t bstrd_x, armpl_int_t istrd_x,
	const double *y_p, armpl_int_t bstrd_y, armpl_int_t istrd_y,
	      double *A_p, armpl_int_t bstrd_A, armpl_int_t istrd_A, armpl_int_t jstrd_A);

/**
 * Computes:
 * @f[
 * C^{(i)} := \alpha op(A^{(i)}) op(B^{(i)}) + \beta C^{(i)}
 * @f]
 *
 * where @f$op(X) =X^T@f$, if @f$toupper(transX) = ‘T’@f$, or @f$op(X) = X@f$ if @f$toupper(transX) = ‘N’@f$.
 *
 * @f$A^{(i)}@f$ are matrices for @f$i = 1..np@f$, and @f$\alpha@f$ and @f$\beta@f$ are scalars.
 * @param ninter [in] The number of matrices interleaved. This should be a small multiple of your machine’s native vector length.
 * @param nbatch [in] The number of batches ofnintermatrices. Note that np=ninter*nbatch.
 * @param transA [in] Whether the @f$A^{(i)}@f$ are transposed (@f$toupper(transA) = ‘T’@f$) or not (@f$toupper(transA) = ‘N’@f$).
 * @param transB [in] Whether the @f$B^{(i)}@f$ are transposed (@f$toupper(transB) = ‘T’@f$) or not (@f$toupper(transB) = ‘N’@f$).
 * @param m [in] The number of rows of each of the matrices @f$op(A^{(i)})@f$ and Ci.
 * @param n [in] The number of columns of each of the matrices @f$op(B^{(i)})@f$ andCi.
 * @param k [in] The number of columns of each of the matrices @f$op(A^{(i)})@f$ and the number of rows of each of the matrices @f$op(B^{(i)})@f$.
 * @param alpha [in] The scalar parameter @f$\beta@f$.
 * @param A_p [in] Pointer to the start of the array which contains matrices @f$A^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_A [in] The stride between corresponding elements in successive batches in A_p.
 * @param istrd_A [in] The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrd_A [in] The stride between successive elements in the same row of the same matrix in A_p.
 * @param B_p [in] Pointer to the start of the array which contains matrices @f$B^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_B [in] The stride between corresponding elements in successive batches in B_p.
 * @param istrd_B [in] The stride between successive elements in the same column of the same matrix in B_p.
 * @param jstrd_B [in] The stride between successive elements in the same row of the same matrix in B_p.
 * @param beta [in] The scalar parameter @f$\beta@f$.
 * @param C_p [in,out] Pointer to the start of the array which contains matrices Ci laid out in interleaved-batch format.
 * @param bstrd_C [in] The stride between corresponding elements in successive batches in C_p.
 * @param istrd_C [in] The stride between successive elements in the same column of the same matrix in C_p.
 * @param jstrd_C [in] The stride between successive elements in the same row of the same matrix in C_p.
 * @returns Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dgemm_interleave_batch(
	armpl_int_t ninter, armpl_int_t nbatch,
	char transa,
	char transb,
	armpl_int_t m, armpl_int_t n, armpl_int_t k,
	double alpha,
	const double *A_p, armpl_int_t bstrdA, armpl_int_t istrdA, armpl_int_t jstrdA,
	const double *B_p, armpl_int_t bstrdB, armpl_int_t istrdB, armpl_int_t jstrdB,
	double beta,
	      double *C_p, armpl_int_t bstrdC, armpl_int_t istrdC, armpl_int_t jstrdC);

/**
 * Computes:
 *  @f[
 *  y^{(i)} := \alpha op(A^{(i)})x^{(i)} + \beta y^{(i)}
 *  @f]
 *
 * where @f$op(X) = Xi^T@f$, if @f$toupper(transX) = ‘T’@f$, or @f$op(X) = X@f$ if @f$toupper(transX) = ‘N’@f$.
 *
 * @f$A^{(i)}@f$ are an m by n matrices, and @f$x^{(i)}@f$ and @f$y^{(i)}@f$ are vectors, for @f$i = 1..np@f$, and @f$\alpha@f$ and @f$\beta@f$ are scalars.
 *
 * @param ninter [in] The number of matrices interleaved. This should be a small multiple of your machine’s native vector length.
 * @param nbatch [in] The number of batches of ninter matrices. Note that np = ninter*nbatch.
 * @param transA [in] Whether the @f$A^{(i)}@f$ are transposed (@f$toupper(transA) = ‘T’@f$) or not (@f$toupper(transA) = ‘N’@f$).
 * @param m [in] The number of rows of each of the matrices @f$op(A^{(i)})@f$.
 * @param n [in] The number of columns of each of the matrices @f$op(A^{(i)})@f$.
 * @param alpha [in] The scalar parameter @f$\alpha@f$.
 * @param A_p [in] Pointer to the start of the array which contains matrices @f$A^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_A [in] The stride between corresponding elements in successive batches in A_p.
 * @param istrd_A [in] The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrd_A [in] The stride between successive elements in the same row of the same matrix in A_p.
 * @param x_p [in] Pointer to the start of the array which contains vectors @f$x^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_x [in] The stride between corresponding elements in successive batches in x_p.
 * @param istrd_x [in] The stride between successive elements in the same vector in x_p.
 * @param beta [in] The scalar parameter @f$\beta@f$.
 * @param y_p [in,out] Pointer to the start of the array which contains vectors @f$Ay^{(i)}@f$A laid out in interleaved-batch format.
 * @param bstrd_y [in] The stride between corresponding elements in successive batches in y_p.
 * @param istrd_y [in] The stride between successive elements in the same vector in y_p.
 * @returns Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dgemv_interleave_batch(
	armpl_int_t ninter , armpl_int_t nbatch ,
	char transA ,
	armpl_int_t m, armpl_int_t n,
	double alpha,
	const double *A_p, armpl_int_t bstrd_A, armpl_int_t istrd_A, armpl_int_t jstrd_A,
	const double *x_p, armpl_int_t bstrd_x, armpl_int_t istrd_x,
	double beta,
	double *y_p , armpl_int_t bstrd_y, armpl_int_t istrd_y);

/**
 * Interleaved-batch QR factorization with column pivoting. Computes:
 *  @f[
 *  A^{(i)} P^{(i)} = Q^{(i)} R^{(i)}
 *  @f]
 *
 * for @f$i = 1..np@f$ where @f$A^{(i)}@f$ are input matrices, @f$Q^{(i)}@f$ are orthogonal matrices, @f$R^{(i)}@f$ are upper triangular
 * matrices and @f$P^{(i)}@f$ are permutation matrices. This is rank-revealing QR factorization; the ranks of the
 * @f$np@f$ matrices @f$A^{(i)}@f$ are returned. @f$Q^{(i)}@f$ are returned as the product of rank elementary
 * reflectors. @f$Q^{(i)} = H^{(i)}_1 H^{(i)}_2..H^{(i)}_{rank}@f$, where @f$H^{(i)}_j = (I - \tau^{(i)}_j v^{(i)}_j (v^{(i)}_j)^T)@f$ for
 * @f$i= 1..np@f$ and @f$j = 1..rank@f$.
 *
 * @param ninter     [in] The number of matrices interleaved. This should be a small multiple of your machine’s
 *                        native vector length.
 * @param nbatch     [in] The number of batches of ninter matrices. Note that @f$np@f$=ninter*nbatch.
 * @param m          [in] The number of rows of each of the matrices @f$A^{(i)}@f$.
 * @param n          [in] The number of columns of each of the matrices @f$A^{(i)}@f$.
 * @param A_p        [in,out] On input: Pointer to the start of the array which contains matrices @f$A^{(i)}@f$ laid out
 *                                      in interleaved-batch format.
 *                            On output: the upper triangular part of each matrix contains the min(m,n)-by-n upper
 *                                       trapezoidal matrix @f$R^{(i)}@f$, in interleaved-batch format. Elements below
 *                                       the diagonal of each matrix in interleaved-batch format represent vectors
 *                                       @f$v^{(i)}@f$, which, together with the array tau_p, represent the orthogonal
 *                                       matrix @f$Q^{(i)}@f$ as a product of min(m,n) elementary reflectors.
 * @param bstrd_A    [in] The stride between corresponding elements in successive batches in A_p.
 * @param istrd_A    [in] The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrd_A    [in] The stride between successive elements in the same row of the same matrix in A_p.
 * @param jpvt_p     [out] Encodes the column permutations that have been applied. jpvt_p is stored in
 *                         interleaved-batch format. If jpvt_p[bstrd_jpvt*i + istrd_jpvt*j] = k then the @f$j^{\textrm{th}}@f$
 *                         column of @f$A^{(i)} P^{(i)}@f$ was the @f$k^{\textrm{th}}@f$ column of @f$A^{(i)}@f$.
 * @param bstrd_jpvt [in] The stride between corresponding elements in successive batches in jpvt_p.
 * @param istrd_jpvt [in] The stride between successive elements in the same vector in jpvt_p.
 * @param tau_p      [out] Pointer to the start of the array which contains vectors @f$\tau^{(i)}@f$ laid out in
 *                         interleaved-batch format.
 * @param bstrd_tau  [in] The stride between corresponding elements in successive batches in tau_p.
 * @param istrd_tau  [in] The stride between successive elements in the same vector in tau_p.
 * @param rank       [out] Array of length @f$np@f$, where rank[i] is the rank of @f$A^{(i)}@f$.
 * @returns Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dgeqrfrr_interleave_batch(armpl_int_t ninter, armpl_int_t nbatch, armpl_int_t m,
                                               armpl_int_t n, double *A_p, armpl_int_t bstrd_A,
                                               armpl_int_t istrd_A, armpl_int_t jstrd_A, armpl_int_t *jpvt_p,
                                               armpl_int_t bstrd_jpvt, armpl_int_t istrd_jpvt, double *tau_p,
                                               armpl_int_t bstrd_tau, armpl_int_t istrd_tau, armpl_int_t *rank);

/**
 * Interleaved-batch LU factorization with threshold pivoting. Forms:
 * @f[
 *    A^{(i)} = P^{(i)} L^{(i)} U^{(i)}
 * @f]
 * for @f$i = 1..np@f$ where @f$A^{(i)}@f$ are input matrices, @f$L^{(i)}@f$ are lower triangular matrices, @f$U^{(i)}@f$ are upper triangular matrices
 * and @f$P^{(i)}@f$ are permutation matrices.
 *
 * Threshold pivoting is used: if, during step @f$j@f$ of matrix @f$k@f$, @f$|A^{(k)}_{ij}| \geq |\theta|@f$ then row @f$i@f$
 * @f$(j+ 1 \geq i \geq m)@f$ is exchanged with row @f$j@f$ and no further rows are considered for pivoting on the current step.
 * If @f$|\theta| = 0@f$ then no pivoting is performed.
 * If @f$|\theta| > ||A^{(k)}_{ij}||_{\infty} (k+ 1 \geq j \geq n)@f$ (i.e.\ @f$\theta@f$ is larger than largest absolute
 * value in column @f$j@f$)  then  @f$|A^{(k)}_{ij}| = ||A^{(k)}_{ij}||_{\infty}@f$ is chosen as the pivot value. This is
 * equivalent to LAPACK GETRF's pivoting strategy.
 *
 * @param ninter      [in]     The number of matrices interleaved. This should be a small multiple of your machine's native vector length.
 * @param nbatch      [in]     The number of batches of ninter matrices. Note that np=ninter*nbatch.
 * @param m           [in]     The number of rows of each of the matrices @f$A^{(i)}@f$.
 * @param n           [in]     The number of columns of each of the matrices @f$A^{(i)}@f$.
 * @param A_p         [in,out] On input: Pointer to the start of the array which contains matrices @f$A^{(i)}@f$ laid out in interleaved-batch format.
 *                             On output: The upper triangular parts of each @f$A^{(i)}@f$ in interleaved-batch layout contain the matrices @f$U^{(i)}@f$;
 *                             the strictly lower triangular parts contain matrices @f$L^{(i)}@f$. The diagonal elements of @f$L^{(i)}@f$ are not stored.
 * @param bstrd_A     [in]     The stride between corresponding elements in successive batches in A_p.
 * @param istrd_A     [in]     The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrd_A     [in]     The stride between successive elements in the same row of the same matrix in A_p.
 * @param theta       [in]     The threshold value @f$\theta@f$.
 * @param ipvt_p      [out]    Encodes the row permutations that have been applied. ipvt_p is stored in interleaved-batch format.
 *                             If ipvt_p[bstrd_ipvt*i + istrd_ipvt*j] = k then the @f$j^{th}@f$ row of @f$P^{(i)}A^{(i)}@f$ was the @f$k^{th}@f$ row of @f$A^{(i)}@f$.
 * @param bstrd_ipvt  [in]     The stride between corresponding elements in successive batches in ipivt_p.
 * @param istrd_ipivt [in]     The stride between successive elements in the same column of the ipivt_p.
 * Returns:                    Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 *                                           Set to ARMPL_STATUS_EXECUTION_FAILURE if one or more diagonal entries of @f$U^{(i)}@f$ are zero.
 *                                           The routine has computed all the factorizations but using them to solve systems of linear equations will
 *                                           cause divisions by zero. This is analogous to LAPACK GETRF returning with INFO > 0.
*/
armpl_status_t armpl_dgetrftp_interleave_batch(armpl_int_t ninter, armpl_int_t nbatch, armpl_int_t m,
                                               armpl_int_t n, double *A_p, armpl_int_t bstrd_A,
                                               armpl_int_t istrd_A, armpl_int_t jstrd_A, double theta,
                                               armpl_int_t *ipvt_p, armpl_int_t bstrd_ipvt,
                                               armpl_int_t istrd_ipvt);

/**
 * Form the @f$Q^{(i)}@f$ matrices from the Householder transformations generated by armpl_dgeqrfrr_interleave_batch:
 *  @f[
 *      Q^{(i)}  =  H^{(i)}_1 H^{(i)}_2 . . . H^{(i)}_k
 *  @f]
 * for @f$i = 1..np@f$.
 * @param ninter   [in] The number of input vectors and output matrices interleaved.
 * @param nbatch   [in] The number of batches of ninter vectors and matrices.
 * @param m        [in] The length of each vector x and the number of rows in each matrix @f$Q^{(i)}@f$.
 * @param n        [in] The length of each vector y and the number of columns in each matrix @f$Q^{(i)}@f$.
 * @param nk        [in] Array of length np, where nk[i] is the number of elementary reflectors whose product defines @f$Q^{(i)}@f$.
 * @param A_p      [in,out] On input: parameter A_p, as output from armpl_dgeqrfrr_interleave_batch.
 *                          On return: the m-by-n matrices Q, in interleaved-batch format.
 * @param bstrdA   [in] The stride between corresponding elements in successive batches in A_p.
 * @param istrdA   [in] The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrdA   [in] The stride between successive elements in the same row of the same matrix in A_p.
 * @param tau_p    [in] The scalar factors of the elementary reflectors.
 * @param bstrdtau [in] The stride between corresponding elements in successive batches in tau_p.
 * @param istrdtau [in] The stride between successive elements in the same vector in tau_p.
 * @returns: Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dorgqr_interleave_batch(armpl_int_t ninter, armpl_int_t nbatch, armpl_int_t m,
                                             armpl_int_t n, const armpl_int_t *nk, double *A_p,
                                             armpl_int_t bstrdA, armpl_int_t istrdA, armpl_int_t jstrdA,
                                             const double *tau_p, armpl_int_t bstrdtau, armpl_int_t istrdtau);

/**
 * Interleaved-batch multiply by @f$Q@f$ matrix. Computes:
 *  @f[
 *      C^{(i)} := \mathrm{op}(Q^{(i)}) C^{(i)},
 *  @f]
 * or:
 *  @f[
 *      C^{(i)} := C^{(i)} \mathrm{op}(Q^{(i)}),
 *  @f]
 * where @f$op(X) =X^T@f$ if transX = 'T' or @f$op(X) =X@f$ if transX = 'N'.
 * The matrices @f$Q^{(i)}@f$ are taken as output from armpl_dgeqrfrr_interleave_batch and @f$C^{(i)}@f$ are m by n
 * matrices, for  @f$i= 1..np @f$
 * @param ninter   [in] The number of input vectors and output matrices interleaved.
 * @param nbatch   [in] The number of batches of ninter vectors and matrices.
 * @param side     [in] Whether @f$Q^{(i)}@f$ appear on the left (toupper(side) = ‘L’) or right (toupper(side) = ‘R’) of @f$C^{(i)}@f$.
 * @param transQ   [in] Whether the @f$Q^{(i)}@f$ are transposed (toupper(transQ) = ‘T’) or not (toupper(transQ) = ‘N’).
 * @param m        [in] The length of each vector x and the number of rows in each matrix C.
 * @param n        [in] The length of each vector y and the number of columns in each matrix C.
 * @param nk       [in] Array of length np, where nk[i] is the number of elementary reflectors whose product defines Q^{(i)}.
 * @param A_p      [in] A_p, as output from armpl_dgeqrfrr_interleave_batch.
 * @param bstrdA   [in] The stride between corresponding elements in successive batches in A_p.
 * @param istrdA   [in] The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrdA   [in] The stride between successive elements in the same row of the same matrix in A_p.
 * @param tau_p    [in] The scalar factors of the elementary reflectors.
 * @param bstrdtau [in] The stride between corresponding elements in successive batches in tau_p.
 * @param istrdtau [in] The stride between successive elements in the same vector in tau_p.
 * @param C_p      [in,out] Pointer to the start of the array which contains matrices @f$C^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrdC   [in] The stride between corresponding elements in successive batches in C_p.
 * @param istrdC   [in] The stride between successive elements in the same column of the same matrix in C_p.
 * @param jstrdC   [in] The stride between successive elements in the same row of the same matrix in C_p.
 * @returns: Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dormqr_interleave_batch(armpl_int_t ninter, armpl_int_t nbatch, char side, char transQ,
                                             armpl_int_t m, armpl_int_t n, const armpl_int_t *nk,
                                             const double *A_p, armpl_int_t bstrdA, armpl_int_t istrdA,
                                             armpl_int_t jstrdA, const double *tau_p, armpl_int_t bstrdtau,
                                             armpl_int_t istrdtau, double *C_p, armpl_int_t bstrdC,
                                             armpl_int_t istrdC, armpl_int_t jstrdC);

/**
 * Interleaved-batch Cholesky factorization. Forms:
 * @f[
 *    A^{(i)} = L^{(i)} (L^{(i)})^T
 * @f]
 * if toupper(uplo) = ‘L’, or forms:
 * @f[
 *    A^{(i)} = (U^{(i)})^T U^{(i)}
 * @f]
 * if toupper(uplo) = ‘U’, for @f$i = 1..np@f$ where @f$A^{(i)}@f$ are input symmetric positive-definite square matrices,
 * @f$L^{(i)}@f$ are lower triangular matrices, and @f$U^{(i)}=(L^{(i)})^T@f$ are upper triangular matrices.
 *
 * @param ninter      [in]     The number of matrices interleaved. This should be a small multiple of your machine's native vector length.
 * @param nbatch      [in]     The number of batches of ninter matrices. Note that np=ninter*nbatch.
 * @param n           [in]     The number of rows/columns of each of the matrices @f$A^{(i)}@f$.
 * @param A_p         [in,out] On input: Pointer to the start of the array which contains matrices @f$A^{(i)}@f$ laid out in interleaved-batch format.
 *                             On output: If toupper(uplo) = ‘U’, the upper triangular parts of each @f$A^{(i)}@f$ in interleaved-batch layout contain
 *                             the matrices @f$U^{(i)}@f$, otherwise (toupper(uplo) = ‘L’) the lower triangular parts contain matrices @f$L^{(i)}@f$.
 * @param bstrd_A     [in]     The stride between corresponding elements in successive batches in A_p.
 * @param istrd_A     [in]     The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrd_A     [in]     The stride between successive elements in the same row of the same matrix in A_p.
 * Returns:                    Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 *                                           No error will be returned if any @f$A^{(i)}@f$ are not SPD, unlike in LAPACK's POTRF.
*/
armpl_status_t armpl_dpotrf_interleave_batch(armpl_int_t ninter, armpl_int_t nbatch, char uplo,
                                             armpl_int_t n, double *A_p, armpl_int_t bstrd_A,
                                             armpl_int_t istrd_A, armpl_int_t jstrd_A);

/**
 * Scale a batch of @f$np@f$ vectors @f$x^{(i)}@f$ by a scalar:
 *  @f[
 *      x^{(i)}  =  \alpha^{(i)} x^{(i)}
 *  @f]
 * for @f$i = 1..np@f$.
 * @param ninter   [in] The number of vectors interleaved.
 * @param nbatch   [in] The number of batches of ninter vectors.
 * @param n        [in] The length of each vector @f$x@f$.
 * @param alpha    [in] The scalar parameter @f$\alpha@f$.
 * @param x_p      [in,out] Pointer to the start of the interleaved-batch vectors, @f$x@f$.
 * @param bstrd_x  [in] The stride between corresponding elements in successive batches in x_p.
 * @param istrd_x  [in] The stride between successive elements in the same vector in x_p.
 * @return Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dscal_interleave_batch(armpl_int_t ninter, armpl_int_t nbatch, armpl_int_t n,
                                            double alpha, double *x_p, armpl_int_t bstrd_x,
                                            armpl_int_t istrd_x);

/**
 * Interleave-batch triangular matrix-matrix multiplication.
 *
 * Computes:
 * @f[
 *     B^{(i)} = \alpha op(A^{(i)})B^{(i)}
 * @f]
 * or:
 * @f[
 *     B^{(i)} = \alpha B^{(i)} op(A^{(i)})
 * @f]
 * for @f$i = 1..np@f$, where @f$op(X) =X^T@f$ if toupper(transX) = ‘T’, or @f$op(X) = X@f$ if toupper(transX) = ‘N’.
 * @f$B^a{(i)}@f$ are m-by-n matrices, @f$A^{(i)}@f$ are upper or lower triangular, unit or non-unit diagnonal matrices,
 * and @f$\alpha@f$ is a scalar.
 *
 * @param ninter  [in] The number of matrices interleaved. This should be a small multiple of your machine’s native vector length.
 * @param nbatch  [in] The number of batches of ninter matrices. Note that np=ninter*nbatch.
 * @param side    [in] Whether @f$A^{(i)}@f$ appear on the left (toupper(side) = ‘L’) or right (toupper(side) = ‘R’) of @f$B^{(i)}@f$.
 * @param uplo    [in] Whether @f$A^{(i)}@f$ are upper (toupper(uplo) = ‘U’) or lower (toupper(uplo) = ‘L’) triangular matrices.
 * @param transA  [in] Whether the @f$A^{(i)}@f$ are transposed (toupper(transA) = ‘T’) or not (toupper(transA) = ‘N’).
 * @param diag    [in] Whether @f$A^{(i)}@f$ have unit (toupper(diag) = ‘U’) or non-unit (toupper(diag) = ‘N’) diagonals.
 * @param m       [in] The number of rows of each of the matrices @f$B^{(i)}@f$.
 * @param n       [in] The number of columns of each of the matrices @f$B^{(i)}@f$.
 * @param alpha   [in] The scalar parameter @f$\alpha@f$.
 * @param A_p     [in] Pointer to the start of the array which contains matrices @f$A^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_A [in] The stride between corresponding elements in successive batches in A_p.
 * @param istrd_A [in] The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrd_A [in] The stride between successive elements in the same row of the same matrix in A_p.
 * @param B_p     [in,out] Pointer to the start of the array which contains matrices @f$B^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_B [in] The stride between corresponding elements in successive batches in B_p.
 * @param istrd_B [in] The stride between successive elements in the same column of the same matrix in B_p.
 * @param jstrd_B [in] The stride between successive elements in the same row of the same matrix in B_p.
 * @return Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dtrmm_interleave_batch(armpl_int_t ninter, armpl_int_t nbatch, char side, char uplo,
                                            char transA, char diag, armpl_int_t m, armpl_int_t n,
                                            double alpha, const double *A_p, armpl_int_t bstrd_A,
                                            armpl_int_t istrd_A, armpl_int_t jstrd_A, double *B_p,
                                            armpl_int_t bstrd_B, armpl_int_t istrd_B, armpl_int_t jstrd_B);

/**
 * Interleave-batch triangular matrix-matrix solve.
 *
 * Finds @f$X^{(i)}@f$ such that:
 * @f[
 *     op(A^{(i)})X^{(i)}=\alpha B^{(i)}
 * @f]
 * or:
 * @f[
 *     X^{(i)}op(A^{(i)}) =\alpha B^{(i)}
 * @f]
 * for @f$i = 1..np@f$, where @f$op(X) =X^T@f$ if toupper(transX) = ‘T’, or @f$op(X) = X@f$ if toupper(transX) = ‘N’.
 * @f$A^{(i)}@f$ are upper or lower triangular, unit or non-unit diagnonal input matrices, @f$B^{(i)}@f$ are m-by-n input
 * matrices and @f$X^{(i)}@f$ are output matrices.
 *
 * @param ninter  [in] The number of matrices interleaved. This should be a small multiple of your machine’s native vector length.
 * @param nbatch  [in] The number of batches of ninter matrices. Note that np=ninter*nbatch.
 * @param side    [in] Whether @f$A^{(i)}@f$ appear on the left (toupper(side) = ‘L’) or right (toupper(side) = ‘R’) of @f$X^{(i)}@f$.
 * @param uplo    [in] Whether @f$A^{(i)}@f$ are upper (toupper(uplo) = ‘U’) or lower (toupper(uplo) = ‘L’) triangular matrices.
 * @param transA  [in] Whether the @f$A^{(i)}@f$ are transposed (toupper(transA) = ‘T’) or not (toupper(transA) = ‘N’).
 * @param diag    [in] Whether @f$A^{(i)}@f$ have unit (toupper(diag) = ‘U’) or non-unit (toupper(diag) = ‘N’) diagonals.
 * @param m       [in] The number of rows of each of the matrices @f$B^{(i)}@f$.
 * @param n       [in] The number of columns of each of the matrices @f$B^{(i)}@f$.
 * @param alpha   [in] The scalar parameter @f$\alpha@f$.
 * @param A_p     [in] Pointer to the start of the array which contains matrices @f$A^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_A [in] The stride between corresponding elements in successive batches in A_p.
 * @param istrd_A [in] The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrd_A [in] The stride between successive elements in the same row of the same matrix in A_p.
 * @param B_p     [in,out] On input: Pointer to the start of the array which contains matrices @f$B^{(i)}@f$ laid out in interleaved-batch format.
 *                         On output: Pointer to the start of the array which contains matrices @f$X^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_B [in] The stride between corresponding elements in successive batches in B_p.
 * @param istrd_B [in] The stride between successive elements in the same column of the same matrix in B_p.
 * @param jstrd_B [in] The stride between successive elements in the same row of the same matrix in B_p.
 * @return Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dtrsm_interleave_batch(armpl_int_t ninter, armpl_int_t nbatch, char side, char uplo,
                                            char transA, char diag, armpl_int_t m, armpl_int_t n,
                                            double alpha, const double *A_p, armpl_int_t bstrd_A,
                                            armpl_int_t istrd_A, armpl_int_t jstrd_A, double *B_p,
                                            armpl_int_t bstrd_B, armpl_int_t istrd_B, armpl_int_t jstrd_B);

/**
 * Interleave-batch triangular matrix-vector solve.
 *
 * Finds @f$x^{(i)}@f$ such that:
 * @f[
 *     op(A^{(i)})x^{(i)} = b^{(i)}
 * @f]
 * for @f$i = 1..np@f$, where @f$op(X) =X^T@f$ if toupper(transX) = ‘T’, or @f$op(X) = X@f$ if toupper(transX) = ‘N’.
 * @f$A^{(i)}@f$ are upper or lower triangular, unit or non-unit diagnonal input matrices, @f$b^{(i)}@f$ are input
 * vectors and @f$x^{(i)}@f$ are output vectors.
 *
 * @param ninter  [in] The number of matrices interleaved. This should be a small multiple of your machine’s native vector length.
 * @param nbatch  [in] The number of batches of ninter matrices. Note that np=ninter*nbatch.
 * @param uplo    [in] Whether @f$A^{(i)}@f$ are upper (toupper(uplo) = ‘U’) or lower (toupper(uplo) = ‘L’) triangular matrices.
 * @param transA  [in] Whether the @f$A^{(i)}@f$ are transposed (toupper(transA) = ‘T’) or not (toupper(transA) = ‘N’).
 * @param diag    [in] Whether @f$A^{(i)}@f$ have unit (toupper(diag) = ‘U’) or non-unit (toupper(diag) = ‘N’) diagonals.
 * @param n       [in] The number of rows and columns of each of the matrices @f$A^{(i)}@f$.
 * @param A_p     [in] Pointer to the start of the array which contains matrices @f$A^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_A [in] The stride between corresponding elements in successive batches in A_p.
 * @param istrd_A [in] The stride between successive elements in the same column of the same matrix in A_p.
 * @param jstrd_A [in] The stride between successive elements in the same row of the same matrix in A_p.
 * @param x_p     [in,out] On input: Pointer to the start of the array which contains vectors @f$b^{(i)}@f$ laid out in interleaved-batch format.
 *                         On output: Pointer to the start of the array which contains vectors @f$x^{(i)}@f$ laid out in interleaved-batch format.
 * @param bstrd_x [in] The stride between corresponding elements in successive batches in x_p.
 * @param istrd_x [in] The stride between successive elements in the same vector in x_p.
 * @return Error status. Set to ARMPL_STATUS_SUCCESS if no errors occurred.
 */
armpl_status_t armpl_dtrsv_interleave_batch(armpl_int_t ninter, armpl_int_t nbatch, char uplo,
                                            char transA, char diag, armpl_int_t n,
                                            const double *A_p, armpl_int_t bstrd_A,
                                            armpl_int_t istrd_A, armpl_int_t jstrd_A,
                                            double *x_p, armpl_int_t bstrd_x, armpl_int_t istrd_x);

#ifdef __cplusplus
} //extern "C"
#endif

#endif //ARMPL_INTERLEAVE_BATCH_H

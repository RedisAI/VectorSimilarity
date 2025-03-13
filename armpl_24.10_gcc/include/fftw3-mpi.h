/*
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#ifndef ARMPL_FFTW_MPI_H
#define ARMPL_FFTW_MPI_H

#include <fftw3.h>
#ifndef ARMPL_FFTW_MPI_INTERNAL
#include <mpi.h>
#include <armpl_mpi_wrapper.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

//
// Sizing
//

ptrdiff_t fftw_mpi_local_size_many(int rnk, const ptrdiff_t *n, ptrdiff_t howmany, ptrdiff_t block0,
                                   MPI_Comm comm, ptrdiff_t *local_n0, ptrdiff_t *local_0_start);
ptrdiff_t fftw_mpi_local_size(int rnk, const ptrdiff_t *n, MPI_Comm comm, ptrdiff_t *local_n0,
                              ptrdiff_t *local_0_start);
ptrdiff_t fftw_mpi_local_size_1d(ptrdiff_t N0, MPI_Comm comm, int sign, unsigned flags, ptrdiff_t *local_ni,
                                 ptrdiff_t *local_i_start, ptrdiff_t *local_no, ptrdiff_t *local_o_start);
ptrdiff_t fftw_mpi_local_size_2d(ptrdiff_t N0, ptrdiff_t N1, MPI_Comm comm, ptrdiff_t *local_n0,
                                 ptrdiff_t *local_0_start);
ptrdiff_t fftw_mpi_local_size_3d(ptrdiff_t N0, ptrdiff_t N1, ptrdiff_t N2, MPI_Comm comm, ptrdiff_t *local_n0,
                                 ptrdiff_t *local_0_start);
ptrdiff_t fftw_mpi_local_size_2d_transposed(ptrdiff_t N0, ptrdiff_t N1, MPI_Comm comm, ptrdiff_t *local_n0,
                                            ptrdiff_t *local_0_start, ptrdiff_t *local_n1,
                                            ptrdiff_t *local_1_start);
ptrdiff_t fftw_mpi_local_size_3d_transposed(ptrdiff_t N0, ptrdiff_t N1, ptrdiff_t N2, MPI_Comm comm,
                                            ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                            ptrdiff_t *local_n1, ptrdiff_t *local_1_start);
ptrdiff_t fftw_mpi_local_size_transposed(int rnk, const ptrdiff_t *Ns, MPI_Comm comm, ptrdiff_t *local_n0,
                                         ptrdiff_t *local_0_start, ptrdiff_t *local_n1,
                                         ptrdiff_t *local_1_start);
ptrdiff_t fftw_mpi_local_size_many_transposed(int rnk, const ptrdiff_t *Ns, ptrdiff_t howmany,
                                              ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
                                              ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                              ptrdiff_t *local_n1, ptrdiff_t *local_1_start);
ptrdiff_t fftw_mpi_local_size_many_1d(ptrdiff_t n0, ptrdiff_t howmany, MPI_Comm comm, int sign,
                                      unsigned flags, ptrdiff_t *local_ni, ptrdiff_t *local_i_start,
                                      ptrdiff_t *local_no, ptrdiff_t *local_o_start);
//
// Planning
//
fftw_plan fftw_mpi_plan_dft(int rnk, const ptrdiff_t *n, fftw_complex *in, fftw_complex *out, MPI_Comm comm,
                            int sign, unsigned flags);
fftw_plan fftw_mpi_plan_dft_1d(ptrdiff_t N0, fftw_complex *inData, fftw_complex *outData, MPI_Comm comm,
                               int sign, unsigned flags);
fftw_plan fftw_mpi_plan_dft_2d(ptrdiff_t N0, ptrdiff_t N1, fftw_complex *inData, fftw_complex *outData,
                               MPI_Comm comm, int sign, unsigned flags);
fftw_plan fftw_mpi_plan_dft_3d(ptrdiff_t N0, ptrdiff_t N1, ptrdiff_t N2, fftw_complex *inData,
                               fftw_complex *outData, MPI_Comm comm, int sign, unsigned flags);
fftw_plan fftw_mpi_plan_many_dft(int rnk, const ptrdiff_t *n, ptrdiff_t howmany, ptrdiff_t block,
                                 ptrdiff_t tblock, fftw_complex *in, fftw_complex *out, MPI_Comm comm,
                                 int sign, unsigned flags);
fftw_plan fftw_mpi_plan_dft_r2c(int rnk, const ptrdiff_t *Ns, double *inData, fftw_complex *outData,
                                MPI_Comm comm, unsigned flags);
fftw_plan fftw_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1, double *in, fftw_complex *out, MPI_Comm comm,
                                   unsigned flags);
fftw_plan fftw_mpi_plan_dft_r2c_3d(ptrdiff_t N0, ptrdiff_t N1, ptrdiff_t N2, double *inData,
                                   fftw_complex *outData, MPI_Comm comm, unsigned flags);
fftw_plan fftw_mpi_plan_many_dft_r2c(int rnk, const ptrdiff_t *n, ptrdiff_t howmany, ptrdiff_t iblock,
                                     ptrdiff_t oblock, double *in, fftw_complex *out, MPI_Comm comm,
                                     unsigned flags);
fftw_plan fftw_mpi_plan_dft_c2r_2d(ptrdiff_t n0, ptrdiff_t n1, fftw_complex *in, double *out, MPI_Comm comm,
                                   unsigned flags);
fftw_plan fftw_mpi_plan_dft_c2r_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2, fftw_complex *in, double *out,
                                   MPI_Comm comm, unsigned flags);
fftw_plan fftw_mpi_plan_dft_c2r(int rnk, const ptrdiff_t *n, fftw_complex *in, double *out, MPI_Comm comm,
                                unsigned flags);
fftw_plan fftw_mpi_plan_many_dft_c2r(int rnk, const ptrdiff_t *n, ptrdiff_t howmany, ptrdiff_t iblock,
                                     ptrdiff_t oblock, fftw_complex *in, double *out, MPI_Comm comm,
                                     unsigned flags);
fftw_plan fftw_mpi_plan_transpose(ptrdiff_t n0, ptrdiff_t n1, double *in, double *out, MPI_Comm comm,
                                  unsigned flags);

//
//
// Execute plans
//
void fftw_mpi_execute_dft(fftw_plan plan, fftw_complex *in, fftw_complex *out);
void fftw_mpi_execute_dft_r2c(fftw_plan p, double *in, fftw_complex *out);
void fftw_mpi_execute_dft_c2r(fftw_plan p, fftw_complex *in, double *out);

void fftw_mpi_gather_wisdom(MPI_Comm comm);
void fftw_mpi_broadcast_wisdom(MPI_Comm comm);

void fftw_mpi_cleanup(void);

//
// Sizing
//
ptrdiff_t fftwf_mpi_local_size_many(int rnk, const ptrdiff_t *n, ptrdiff_t howmany, ptrdiff_t block0,
                                    MPI_Comm comm, ptrdiff_t *local_n0, ptrdiff_t *local_0_start);
ptrdiff_t fftwf_mpi_local_size(int rnk, const ptrdiff_t *n, MPI_Comm comm, ptrdiff_t *local_n0,
                               ptrdiff_t *local_0_start);
ptrdiff_t fftwf_mpi_local_size_1d(ptrdiff_t N0, MPI_Comm comm, int sign, unsigned flags, ptrdiff_t *local_ni,
                                  ptrdiff_t *local_i_start, ptrdiff_t *local_no, ptrdiff_t *local_o_start);
ptrdiff_t fftwf_mpi_local_size_2d(ptrdiff_t N0, ptrdiff_t N1, MPI_Comm comm, ptrdiff_t *local_n0,
                                  ptrdiff_t *local_0_start);
ptrdiff_t fftwf_mpi_local_size_3d(ptrdiff_t N0, ptrdiff_t N1, ptrdiff_t N2, MPI_Comm comm,
                                  ptrdiff_t *local_n0, ptrdiff_t *local_0_start);
ptrdiff_t fftwf_mpi_local_size_2d_transposed(ptrdiff_t N0, ptrdiff_t N1, MPI_Comm comm, ptrdiff_t *local_n0,
                                             ptrdiff_t *local_0_start, ptrdiff_t *local_n1,
                                             ptrdiff_t *local_1_start);
ptrdiff_t fftwf_mpi_local_size_3d_transposed(ptrdiff_t N0, ptrdiff_t N1, ptrdiff_t N2, MPI_Comm comm,
                                             ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                             ptrdiff_t *local_n1, ptrdiff_t *local_1_start);
ptrdiff_t fftwf_mpi_local_size_transposed(int rnk, const ptrdiff_t *Ns, MPI_Comm comm, ptrdiff_t *local_n0,
                                          ptrdiff_t *local_0_start, ptrdiff_t *local_n1,
                                          ptrdiff_t *local_1_start);
ptrdiff_t fftwf_mpi_local_size_many_transposed(int rnk, const ptrdiff_t *Ns, ptrdiff_t howmany,
                                               ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
                                               ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                               ptrdiff_t *local_n1, ptrdiff_t *local_1_start);
ptrdiff_t fftwf_mpi_local_size_many_1d(ptrdiff_t n0, ptrdiff_t howmany, MPI_Comm comm, int sign,
                                       unsigned flags, ptrdiff_t *local_ni, ptrdiff_t *local_i_start,
                                       ptrdiff_t *local_no, ptrdiff_t *local_o_start);
//
// Planning
//
fftwf_plan fftwf_mpi_plan_dft(int rnk, const ptrdiff_t *n, fftwf_complex *in, fftwf_complex *out,
                              MPI_Comm comm, int sign, unsigned flags);
fftwf_plan fftwf_mpi_plan_dft_1d(ptrdiff_t N0, fftwf_complex *inData, fftwf_complex *outData, MPI_Comm comm,
                                 int sign, unsigned flags);
fftwf_plan fftwf_mpi_plan_dft_2d(ptrdiff_t N0, ptrdiff_t N1, fftwf_complex *inData, fftwf_complex *outData,
                                 MPI_Comm comm, int sign, unsigned flags);
fftwf_plan fftwf_mpi_plan_dft_3d(ptrdiff_t N0, ptrdiff_t N1, ptrdiff_t N2, fftwf_complex *inData,
                                 fftwf_complex *outData, MPI_Comm comm, int sign, unsigned flags);
fftwf_plan fftwf_mpi_plan_many_dft(int rnk, const ptrdiff_t *n, ptrdiff_t howmany, ptrdiff_t block,
                                   ptrdiff_t tblock, fftwf_complex *in, fftwf_complex *out, MPI_Comm comm,
                                   int sign, unsigned flags);
fftwf_plan fftwf_mpi_plan_dft_c2r(int rnk, const ptrdiff_t *Ns, fftwf_complex *inData, float *outData,
                                  MPI_Comm comm, unsigned flags);
fftwf_plan fftwf_mpi_plan_dft_c2r_2d(ptrdiff_t n0, ptrdiff_t n1, fftwf_complex *in, float *out, MPI_Comm comm,
                                     unsigned flags);
fftwf_plan fftwf_mpi_plan_dft_c2r_3d(ptrdiff_t N0, ptrdiff_t N1, ptrdiff_t N2, fftwf_complex *inData,
                                     float *outData, MPI_Comm comm, unsigned flags);
fftwf_plan fftwf_mpi_plan_many_dft_c2r(int rnk, const ptrdiff_t *n, ptrdiff_t howmany, ptrdiff_t iblock,
                                       ptrdiff_t oblock, fftwf_complex *in, float *out, MPI_Comm comm,
                                       unsigned flags);
fftwf_plan fftwf_mpi_plan_dft_r2c(int rnk, const ptrdiff_t *Ns, float *inData, fftwf_complex *outData,
                                  MPI_Comm comm, unsigned flags);
fftwf_plan fftwf_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1, float *in, fftwf_complex *out, MPI_Comm comm,
                                     unsigned flags);
fftwf_plan fftwf_mpi_plan_dft_r2c_3d(ptrdiff_t N0, ptrdiff_t N1, ptrdiff_t N2, float *inData,
                                     fftwf_complex *outData, MPI_Comm comm, unsigned flags);
fftwf_plan fftwf_mpi_plan_many_dft_r2c(int rnk, const ptrdiff_t *n, ptrdiff_t howmany, ptrdiff_t iblock,
                                       ptrdiff_t oblock, float *in, fftwf_complex *out, MPI_Comm comm,
                                       unsigned flags);
fftwf_plan fftwf_mpi_plan_transpose(ptrdiff_t n0, ptrdiff_t n1, double *in, double *out, MPI_Comm comm,
                                    unsigned flags);

//
//
// Execute plans
//
void fftwf_mpi_execute_dft(fftwf_plan plan, fftwf_complex *in, fftwf_complex *out);
void fftwf_mpi_execute_dft_r2c(fftwf_plan p, float *in, fftwf_complex *out);
void fftwf_mpi_execute_dft_c2r(fftwf_plan p, fftwf_complex *in, float *out);

void fftwf_mpi_gather_wisdom(MPI_Comm comm);
void fftwf_mpi_broadcast_wisdom(MPI_Comm comm);

void fftwf_mpi_cleanup(void);

/* MPI-specific flags */
#define FFTW_MPI_DEFAULT_BLOCK (0)
#define FFTW_MPI_SCRAMBLED_IN (1U << 27)
#define FFTW_MPI_SCRAMBLED_OUT (1U << 28)
#define FFTW_MPI_TRANSPOSED_IN (1U << 29)
#define FFTW_MPI_TRANSPOSED_OUT (1U << 30)

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

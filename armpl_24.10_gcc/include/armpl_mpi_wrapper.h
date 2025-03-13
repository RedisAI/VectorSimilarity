/*
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 * SPDX-FileCopyrightText: Copyright 2015-2024 NAG
 */

#ifndef ARMPL_MPI_WRAPPERS_H
#define ARMPL_MPI_WRAPPERS_H

#include <mpi.h>
#include <stdint.h>

static int armpl_MPI_allGatherI(const void *sendbuf, int sendcount, void *recvbuf, int recvcount,
                                long armpl_comm) {
	return MPI_Allgather(sendbuf, sendcount, MPI_INT, recvbuf, recvcount, MPI_INT, (MPI_Comm)armpl_comm);
}
static int armpl_MPI_allGatherL(const void *sendbuf, int sendcount, void *recvbuf, int recvcount,
                                long armpl_comm) {
	return MPI_Allgather(sendbuf, sendcount, MPI_LONG, recvbuf, recvcount, MPI_LONG, (MPI_Comm)armpl_comm);
}
static int armpl_MPI_gatherI(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                             long armpl_comm) {
	return MPI_Gather(sendbuf, sendcount, MPI_INT, recvbuf, recvcount, MPI_INT, root, (MPI_Comm)armpl_comm);
}
static int armpl_MPI_gatherL(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                             long armpl_comm) {
	return MPI_Gather(sendbuf, sendcount, MPI_LONG, recvbuf, recvcount, MPI_LONG, root, (MPI_Comm)armpl_comm);
}
static int armpl_MPI_gatherD(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                             long armpl_comm) {
	return MPI_Gather(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcount, MPI_DOUBLE, root,
	                  (MPI_Comm)armpl_comm);
}
static int armpl_MPI_gathervD(const void *sendbuf, int sendcount, void *recvbuf, const int *recvcounts,
                              const int *displs, int root, long armpl_comm) {
	return MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcounts, displs, MPI_DOUBLE, root,
	                   (MPI_Comm)armpl_comm);
}
static int armpl_MPI_scatterD(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                              long armpl_comm) {
	return MPI_Scatter(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcount, MPI_DOUBLE, root,
	                   (MPI_Comm)armpl_comm);
}
static int armpl_MPI_scattervD(const void *sendbuf, const int *sendcounts, const int *displs, void *recvbuf,
                               int recvcount, int root, long armpl_comm) {
	return MPI_Scatterv(sendbuf, sendcounts, displs, MPI_DOUBLE, recvbuf, recvcount, MPI_DOUBLE, root,
	                    (MPI_Comm)armpl_comm);
}
static int armpl_MPI_gatherF(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                             long armpl_comm) {
	return MPI_Gather(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, root,
	                  (MPI_Comm)armpl_comm);
}
static int armpl_MPI_gathervF(const void *sendbuf, int sendcount, void *recvbuf, const int *recvcounts,
                              const int *displs, int root, long armpl_comm) {
	return MPI_Gatherv(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcounts, displs, MPI_FLOAT, root,
	                   (MPI_Comm)armpl_comm);
}
static int armpl_MPI_scatterF(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                              long armpl_comm) {
	return MPI_Scatter(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, root,
	                   (MPI_Comm)armpl_comm);
}
static int armpl_MPI_scattervF(const void *sendbuf, const int *sendcounts, const int *displs, void *recvbuf,
                               int recvcount, int root, long armpl_comm) {
	return MPI_Scatterv(sendbuf, sendcounts, displs, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, root,
	                    (MPI_Comm)armpl_comm);
}
static int armpl_MPI_alltoallvD(const void *sendbuf, const int *sendcounts, const int *sdispls, void *recvbuf,
                                const int *recvcounts, const int *rdispls, long armpl_comm) {
	return MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_DOUBLE, recvbuf, recvcounts, rdispls, MPI_DOUBLE,
	                     MPI_COMM_WORLD);
}
static int armpl_MPI_alltoallvF(const void *sendbuf, const int *sendcounts, const int *sdispls, void *recvbuf,
                                const int *recvcounts, const int *rdispls, long armpl_comm) {
	return MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_FLOAT, recvbuf, recvcounts, rdispls, MPI_FLOAT,
	                     MPI_COMM_WORLD);
}
static int armpl_MPI_alltoallD(const void *sendbuf, const int sendcount, void *recvbuf, const int recvcount,
                               long armpl_comm) {
	return MPI_Alltoall(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
}
static int armpl_MPI_alltoallF(const void *sendbuf, const int sendcount, void *recvbuf, const int recvcount,
                               long armpl_comm) {
	return MPI_Alltoall(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, MPI_COMM_WORLD);
}
static int armpl_MPI_bcastI(void *buffer, int count, int root, long armpl_comm) {
	return MPI_Bcast(buffer, count, MPI_INT, root, (MPI_Comm)armpl_comm);
}
static int armpl_MPI_comm_rank(long armpl_comm, int *rank) {
	return MPI_Comm_rank((MPI_Comm)armpl_comm, rank);
}
static int armpl_MPI_comm_size(long armpl_comm, int *size) {
	return MPI_Comm_size((MPI_Comm)armpl_comm, size);
}
static int armpl_MPI_finalize(void) { return MPI_Finalize(); }

// To convert Fortran MPI_Com into C MPI_Comm
static long armpl_MPI_comm_f2c(int32_t comm) {
#if OPEN_MPI
	return (long)MPI_Comm_f2c(comm);
#else
	return (long)comm;
#endif
}

// Function pointers, set by user code calling fftw_mpi_init() which in turn call fftw_mpi_init_internal()
typedef int (*armpl_MPI_1_t)(const void *sendbuf, int sendcount, void *recvbuf, int recvcount,
                             long armpl_comm);
typedef int (*armpl_MPI_2_t)(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                             long armpl_comm);
typedef int (*armpl_MPI_3_t)(const void *sendbuf, int sendcount, void *recvbuf, const int *recvcounts,
                             const int *displs, int root, long armpl_comm);
typedef int (*armpl_MPI_4_t)(const void *sendbuf, const int *sendcounts, const int *displs, void *recvbuf,
                             int recvcount, int root, long armpl_comm);
typedef int (*armpl_MPI_5_t)(void *buffer, int count, int root, long armpl_comm);
typedef int (*armpl_MPI_6_t)(long armpl_comm, int *size);
typedef int (*armpl_MPI_7_t)(void);
typedef long (*armpl_MPI_8_t)(int32_t fortran_comm);
typedef int (*armpl_MPI_10_t)(const void *sendbuf, const int *sendcounts, const int *sdispls, void *recvbuf,
                              const int *recvcounts, const int *rdispls, long armpl_comm);

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void
fftw_mpi_init_internal(armpl_MPI_1_t allGatherI_Ptr, armpl_MPI_1_t allGatherL_Ptr, armpl_MPI_2_t gatherI_Ptr,
                       armpl_MPI_2_t gatherL_Ptr, armpl_MPI_2_t gatherD_Ptr, armpl_MPI_2_t scatterD_Ptr,
                       armpl_MPI_3_t gathervD_Ptr, armpl_MPI_4_t scattervD_Ptr, armpl_MPI_10_t alltoallvD_Ptr,
                       armpl_MPI_1_t alltoallD_Ptr, armpl_MPI_2_t gatherF_Ptr, armpl_MPI_2_t scatterF_Ptr,
                       armpl_MPI_3_t gathervF_Ptr, armpl_MPI_4_t scattervF_Ptr, armpl_MPI_10_t alltoallvF_Ptr,
                       armpl_MPI_1_t alltoallF_Ptr, armpl_MPI_5_t bcastI_Ptr, armpl_MPI_6_t comm_size_Ptr,
                       armpl_MPI_6_t comm_rank_Ptr, armpl_MPI_7_t finalize_Ptr, armpl_MPI_8_t comm_f2c_Ptr);

#ifdef __cplusplus
}
#endif /* __cplusplus */

static inline void fftw_mpi_init(void) {

	fftw_mpi_init_internal(armpl_MPI_allGatherI, armpl_MPI_allGatherL, armpl_MPI_gatherI, armpl_MPI_gatherL,
	                       armpl_MPI_gatherD, armpl_MPI_scatterD, armpl_MPI_gathervD, armpl_MPI_scattervD,
	                       armpl_MPI_alltoallvD, armpl_MPI_alltoallD, armpl_MPI_gatherF, armpl_MPI_scatterF,
	                       armpl_MPI_gathervF, armpl_MPI_scattervF, armpl_MPI_alltoallvF, armpl_MPI_alltoallF,
	                       armpl_MPI_bcastI, armpl_MPI_comm_size, armpl_MPI_comm_rank, armpl_MPI_finalize,
	                       armpl_MPI_comm_f2c);
}

static inline void fftwf_mpi_init(void) { fftw_mpi_init(); }

#endif

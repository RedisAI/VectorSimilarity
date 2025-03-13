/*
 * dgesdd Example Program Text
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 * SPDX-FileCopyrightText: Copyright 2015-2024 NAG
 */

#include <armpl.h>
#include <stdio.h>

int main(void)
{
#define MMAX 8
#define NMAX 8
  int64_t lda;
  int64_t i, info, j, m, n, minmn;
  double a[MMAX*NMAX], s[MMAX+NMAX], u[1], vt[1];
  int64_t matrix_layout;

#ifdef USE_ROW_ORDER
  /* This macro allows access to a 1-d array as though
     it is a 2-d array stored in row-major order */
  #define A(I,J) a[((I)-1)*lda+(J)-1]
  matrix_layout = LAPACK_ROW_MAJOR;
  lda = MMAX;
#else
  /* This macro allows access to a 1-d array as though
     it is a 2-d array stored in column-major order */
  #define A(I,J) a[((J)-1)*lda+(I)-1]
  matrix_layout = LAPACK_COL_MAJOR;
  lda = NMAX;
#endif

  printf("ARMPL example: SVD of a matrix A using dgesdd\n");
  printf("---------------------------------------------\n");
  printf("\n");

  /* Initialize matrix A */
  m = 4;
  n = 4;
  A(1,1) = -0.57;
  A(1,2) = -1.28;
  A(1,3) = -0.39;
  A(1,4) = 0.25;
  A(2,1) = -1.93;
  A(2,2) = 1.08;
  A(2,3) = -0.31;
  A(2,4) = -2.14;
  A(3,1) = 2.30;
  A(3,2) = 0.24;
  A(3,3) = 0.40;
  A(3,4) = -0.35;
  A(4,1) = -1.93;
  A(4,2) = 0.64;
  A(4,3) = -0.66;
  A(4,4) = 0.08;

  printf("Matrix A:\n");
  for (i = 1; i <= m; i++)
    {
      for (j = 1; j <= n; j++)
        printf("%8.4f ", A(i,j));
      printf("\n");
    }

  /* Compute singular values of A */
  info = LAPACKE_dgesdd(matrix_layout,'n',m,n,a,lda,s,u,1,vt,1);

  /* Print solution */
  if (m < n)
    minmn = m;
  else
    minmn = n;
  printf("\n");
  printf("Singular values of matrix A:\n");
  for (i = 0; i < minmn; i++)
    printf("%8.4f ", s[i]);
  printf("\n");

  return 0;
}

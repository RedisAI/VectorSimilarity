/*
 * dgetrf Example Program Text
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 * SPDX-FileCopyrightText: Copyright 2015-2024 NAG
 */

#include <armpl.h>
#include <stdio.h>

int main(void)
{
#define NMAX 8
#define NRHMAX 8
  int lda, ldb, matrix_layout;
  int i, info, j, n, nrhs;
  double a[NMAX*NMAX], b[NMAX*NRHMAX];
  int ipiv[NMAX];

#ifdef USE_ROW_ORDER
  /* These macros allows access to a 1-D array as though
     they are 2-D arrays stored in row-major order */
  #define A(I,J) a[((I)-1)*lda+(J)-1]
  #define B(I,J) b[((I)-1)*ldb+(J)-1]
  matrix_layout = LAPACK_ROW_MAJOR;
  lda = NMAX;
  ldb = NMAX;
#else
  /* These macros allows access to a 1-D array as though
     they are 2-D arrays stored in column-major order */
  #define A(I,J) a[((J)-1)*lda+(I)-1]
  #define B(I,J) b[((J)-1)*ldb+(I)-1]
  matrix_layout = LAPACK_COL_MAJOR;
  lda = NMAX;
  ldb = NMAX;
#endif

  printf("ARMPL example: solution of linear equations using dgetrf/dgetrs\n");
  printf("---------------------------------------------------------------\n");
  printf("\n");

  /* Initialize matrix A */
  n = 4;
  A(1,1) = 1.80;
  A(1,2) = 2.88;
  A(1,3) = 2.05;
  A(1,4) = -0.89;
  A(2,1) = 5.25;
  A(2,2) = -2.95;
  A(2,3) = -0.95;
  A(2,4) = -3.80;
  A(3,1) = 1.58;
  A(3,2) = -2.69;
  A(3,3) = -2.90;
  A(3,4) = -1.04;
  A(4,1) = -1.11;
  A(4,2) = -0.66;
  A(4,3) = -0.59;
  A(4,4) = 0.80;

  /* Initialize right-hand-side matrix B */
  nrhs = 2;
  B(1,1) = 9.52;
  B(1,2) = 18.47;
  B(2,1) = 24.35;
  B(2,2) = 2.25;
  B(3,1) = 0.77;
  B(3,2) = -13.28;
  B(4,1) = -6.22;
  B(4,2) = -6.21;

  printf("Matrix A:\n");
  for (i = 1; i <= n; i++)
    {
      for (j = 1; j <= n; j++)
        printf("%8.4f ", A(i,j));
      printf("\n");
    }

  printf("\n");
  printf("Right-hand-side matrix B:\n");
  for (i = 1; i <= n; i++)
    {
      for (j = 1; j <= nrhs; j++)
        printf("%8.4f ", B(i,j));
      printf("\n");
    }

  /* Factorize A */
  info = LAPACKE_dgetrf(matrix_layout,n,n,a,lda,ipiv);

  printf("\n");
  if (info == 0)
    {
      /* Compute solution */
      info = LAPACKE_dgetrs(matrix_layout,'n',n,nrhs,a,lda,ipiv,b,ldb);
      /* Print solution */
      printf("Solution matrix X of equations A*X = B:\n");
      for (i = 1; i <= n; i++)
        {
          for (j = 1; j <= nrhs; j++)
            printf("%8.4f ", B(i,j));
          printf("\n");
        }
    }
  else
    printf("The factor U of matrix A is singular\n");

  return 0;
}

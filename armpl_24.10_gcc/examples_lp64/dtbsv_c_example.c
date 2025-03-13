/*
 * Example Triangular Banded Solve
 *
 * Arm Performance Libraries version 24.10
 * SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
 */

#include <armpl.h>
#include <stdio.h>

int main(void)
{
#define NMAX 8
  int i, j;
  double a[NMAX*NMAX], x[NMAX];

#ifdef USE_ROW_ORDER
  /* These macros allows access to a 1-D array as though
     they are 2-D arrays stored in row-major order */
  #define A(I,J) a[((I)-1)*lda+(J)-1]
  #define X(I) x[((I)-1)*incx]
  const int lda = NMAX;
  const int incx = 1;
#else
  /* These macros allows access to a 1-D array as though
     they are 2-D arrays stored in column-major order */
  #define A(I,J) a[((J)-1)*lda+(I)-1]
  #define X(I) x[((I)-1)*incx]
  const int lda = NMAX;
  const int incx = 1;
#endif

  const int k = 2;

  printf("ARMPL example: Triangular Banded Solve (dtbsv)\n");
  printf("---------------------------------------------------------------\n");
  printf("\n");

  /* Initialize matrix A */
  const int n = 4;
  A(1,1) = 11.0;
  A(1,2) = 12.0;
  A(1,3) = 13.0;
  A(1,4) = 14.0;
  A(2,1) = 21.0;
  A(2,2) = 22.0;
  A(2,3) = 23.0;
  A(2,4) = 24.0;
  A(3,1) = 31.0;
  A(3,2) = 32.0;
  A(3,3) = 33.0;
  A(3,4) = 34.0;

  /* Initialize right-hand-side matrix B */
  X(1) = 100.0;
  X(2) = 200.0;
  X(3) = 300.0;
  X(4) = 400.0;

  const char diag = 'n';
  const char trans = 'n';
  const char uplo = 'u';

  const int kp1 = k + 1;

  printf("Diag : %c\n", diag);
  printf("Trans: %c\n", trans);
  printf("Uplo : %c\n\n", uplo);

  printf("K : %d\n\n", k);

  printf("Matrix A:\n");
  for (i = 1; i <= kp1; i++)
  {
    for (j = 1; j <= n; j++)
      printf("%8.4f ", A(i,j));
    printf("\n");
  }
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

  printf("\nMatrix A (formatted):\n");
  for (i = 1; i <= n; i++)
  {
    for (j = 1; j < i; j++)
    {
      printf("%8.4f ", 0.0);
    }
    for (;j <= MIN(i + k, n); j++)
    {
      printf("%8.4f ", A((n-1+i)-j,j));
    }
    for (; j <= n ; j++)
    {
      printf("%8.4f ", 0.0);
    }
    printf("\n");
  }

  printf("\n");
  printf("Vector X:\n");
  for (i = 1; i <= n; i++)
    printf("%8.4f\n", X(i));

  printf("\n");
  /* Compute solution */
  dtbsv_(&uplo, &trans, &diag, &n, &k, a, &lda, x, &incx);
  /* Print solution */
  printf("Vector b:\n");
  for (i = 1; i <= n; i++)
    printf("%8.4f\n", X(i));

  return 0;
}

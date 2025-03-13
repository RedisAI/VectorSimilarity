!   Double complex hermitian rank 2 operation in packed format
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!
    Program zhpr2_f_example

      Use armpl_library
      Use, Intrinsic                   :: iso_fortran_env, Only: wp => real64

      Implicit None

      Character (Len=1), Parameter     :: uplo = 'U'

      Integer, Parameter               :: n = 4
      Integer, Parameter               :: incx = 1
      Integer, Parameter               :: incy = 1

      Complex (Kind=wp), Parameter     :: alpha = (1.1_wp, 1.2_wp)

      Complex (Kind=wp), Dimension (n*(n+1)/2) :: ap
      Complex (Kind=wp), Dimension (n) :: x
      Complex (Kind=wp), Dimension (n) :: y

      Integer                          :: i, j, idx

      Write (*, 99999) 'ARMPL example: double complex hermitian rank 2 &
        &operation using ZHPR2'
      Write (*, 99999) '-----------------------------------------------&
        &---------------------'
      Write (*, *)

!
!     Initialize matrix A in packed format
      Do i = 1, n*(n+1)/2
        ap(i) = cmplx(i, i+1, kind=wp)
      End Do
!
!     Initialize vector x and y
      Do i = 1, n
        x(i) = cmplx(2*i-1, 2*i, kind=wp)
        y(i) = cmplx(3*i-2, 3*i-1, kind=wp)
      End Do
!
      Write (*, 99998) 'alpha : ', alpha
      Write (*, *)
      Write (*, 99999) 'Matrix A:'
      Do i = 1, n
        Do j = 1, n
          If (j<i) Then
            idx = j + i*(i-1)/2
            Write (*, 99997, Advance='no') real(ap(idx)), -aimag(ap(idx))
          Else If (j>i) Then
            idx = i + j*(j-1)/2
            Write (*, 99997, Advance='no') real(ap(idx)), aimag(ap(idx))
          Else If (j==i) Then
            idx = i + j*(j-1)/2
            Write (*, 99997, Advance='no') real(ap(idx)), 0.0_wp
          End If
        End Do
        Write (*, *)
      End Do
!
      Write (*, *)
      Write (*, 99999) 'Vector x:'
      Do i = 1, n
        Write (*, 99997) x(i)
      End Do
!
      Write (*, *)
      Write (*, 99999) 'Vector y:'
      Do i = 1, n
        Write (*, 99997) y(i)
      End Do
!
      Call zhpr2(uplo, n, alpha, x, incx, y, incy, ap)
!
      Write (*, *)
      Write (*, 99999) 'Updated vector y = alpha*A*x + beta*y:'
      Do i = 1, n
        Do j = 1, n
          If (j<i) Then
            idx = j + i*(i-1)/2
            Write (*, 99997, Advance='no') real(ap(idx)), -aimag(ap(idx))
          Else If (j>=i) Then
            idx = i + j*(j-1)/2
            Write (*, 99997, Advance='no') real(ap(idx)), aimag(ap(idx))
          End If
        End Do
        Write (*, *)
      End Do
!
99999 Format (A)
99998 Format (A, F6.3, Sp, F6.3, 'i')
99997 Format (1X, 1P, E10.3, Sp, E10.3, 'i')
!
    End Program

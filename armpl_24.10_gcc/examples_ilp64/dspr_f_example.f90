!   Double precision symmetric rank 1 operation in packed format
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!
    Program dspr_f_example

      Use armpl_library
      Use, Intrinsic                   :: iso_fortran_env, Only: wp => real64

      Implicit None

      Character (Len=1), Parameter     :: uplo = 'U'

      Integer, Parameter               :: n = 4
      Integer, Parameter               :: incx = 1

      Real (Kind=wp), Parameter        :: alpha = 1.1_wp

      Real (Kind=wp), Dimension (n*(n+1)/2) :: ap
      Real (Kind=wp), Dimension (n)    :: x

      Integer                          :: i, j, idx

      Write (*, 99999) 'ARMPL example: double precision symmetric &
        &rank 1 operation using DSPR'
      Write (*, 99999) '-------------------------------------------------------&
        &--------------'
      Write (*, *)

!
!     Initialize matrix A in packed format
      Do i = 1, n*(n+1)/2
        ap(i) = real(i, kind=wp)
      End Do
!
!     Initialize vector x
      Do i = 1, n
        x(i) = real(2*i-1, kind=wp)
      End Do
!
      Write (*, 99998) 'alpha : ', alpha
      Write (*, *)
      Write (*, 99999) 'Matrix A:'
      Do i = 1, n
        Do j = 1, n
          If (j<i) Then
            idx = j + i*(i-1)/2
            Write (*, 99997, Advance='no') ap(idx)
          Else
            idx = i + j*(j-1)/2
            Write (*, 99997, Advance='no') ap(idx)
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
      Call dspr(uplo, n, alpha, x, incx, ap)
!
      Write (*, *)
      Write (*, 99999) 'Updated matrix A = alpha*x*x**T + A:'
      Do i = 1, n
        Do j = 1, n
          If (j<i) Then
            idx = j + i*(i-1)/2
            Write (*, 99997, Advance='no') ap(idx)
          Else
            idx = i + j*(j-1)/2
            Write (*, 99997, Advance='no') ap(idx)
          End If
        End Do
        Write (*, *)
      End Do
!
99999 Format (A)
99998 Format (A, F5.3)
99997 Format (1X, 1P, E13.3)
!
    End Program

!   Double precision packed matrix-vector multiplication example
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!
    Program dtpmv_f_example

      Use armpl_library
      Use, Intrinsic                   :: iso_fortran_env, Only: wp => real64

      Implicit None

      Character (Len=1), Parameter     :: uplo = 'U'
      Character (Len=1), Parameter     :: trans = 'N'
      Character (Len=1), Parameter     :: diag = 'N'

      Integer, Parameter               :: n = 4
      Integer, Parameter               :: incx = 1

      Real (Kind=wp), Dimension (n*(n+1)/2) :: ap
      Real (Kind=wp), Dimension (n)    :: x

      Integer                          :: i, j, idx

      Write (*, 99999) 'ARMPL example: double precision matrix-vector &
        &multiplication using DTPMV'
      Write (*, 99999) '-------------------------------------------------------&
        &-----------------'
      Write (*, *)

!
!     Initialize matrix A in packed format
      Do i = 1, n*(n+1)/2
        ap(i) = i
      End Do
!
!     Initialize vector x
      Do i = 1, n
        x(i) = 2*i - 1
      End Do
!
      Write (*, 99999) 'Matrix A:'
      Do i = 1, n
        Do j = 1, n
          If (j<i) Then
            Write (*, 99998, Advance='no') 0.0E0_wp
          Else
            idx = i + j*(j-1)/2
            Write (*, 99998, Advance='no') ap(idx)
          End If
        End Do
        Write (*, *)
      End Do
!
      Write (*, *)
      Write (*, 99999) 'Vector x:'
      Do i = 1, n
        Write (*, 99998) x(i)
      End Do
!
      Call dtpmv(uplo, trans, diag, n, ap, x, incx)
!
      Write (*, *)
      Write (*, 99999) 'Updated vector x = A*x:'
      Do i = 1, n
        Write (*, 99998) x(i)
      End Do
!
99999 Format (A)
99998 Format (1X, 1P, E13.3)
!
    End Program

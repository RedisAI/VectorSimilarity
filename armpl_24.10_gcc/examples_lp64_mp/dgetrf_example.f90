!   DGETRF Example Program Text
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!   SPDX-FileCopyrightText: Copyright 2015-2024 NAG
!
    Program main
!     .. Use Statements ..
      Use armpl_library, Only: dgetrf, dgetrs
!     .. Implicit None Statement ..
      Implicit None
!     .. Parameters ..
      Integer, Parameter               :: wp = kind(0.0D0)
      Integer, Parameter               :: nmax = 8
      Integer, Parameter               :: nrhmax = nmax
      Integer, Parameter               :: lda = nmax
      Integer, Parameter               :: ldb = nmax
      Character (1), Parameter         :: trans = 'N'
!     .. Local Scalars ..
      Integer                          :: i, info, j, n, nrhs
!     .. Local Arrays ..
      Real (Kind=wp)                   :: a(lda, nmax), b(ldb, nrhmax)
      Integer                          :: ipiv(nmax)
!     .. Executable Statements ..
!
      Write (*, 99999)                                                         &
        'ARMPL example: solution of linear equations using DGETRF/DGETRS'
      Write (*, 99999)                                                         &
        '---------------------------------------------------------------'
      Write (*, *)
!
!     Initialize matrix A
      n = 4
      a(1, 1) = 1.80E0_wp
      a(1, 2) = 2.88E0_wp
      a(1, 3) = 2.05E0_wp
      a(1, 4) = -0.89E0_wp
      a(2, 1) = 5.25E0_wp
      a(2, 2) = -2.95E0_wp
      a(2, 3) = -0.95E0_wp
      a(2, 4) = -3.80E0_wp
      a(3, 1) = 1.58E0_wp
      a(3, 2) = -2.69E0_wp
      a(3, 3) = -2.90E0_wp
      a(3, 4) = -1.04E0_wp
      a(4, 1) = -1.11E0_wp
      a(4, 2) = -0.66E0_wp
      a(4, 3) = -0.59E0_wp
      a(4, 4) = 0.80E0_wp
!
!     Initialize right-hand-side matrix B
      nrhs = 2
      b(1, 1) = 9.52E0_wp
      b(1, 2) = 18.47E0_wp
      b(2, 1) = 24.35E0_wp
      b(2, 2) = 2.25E0_wp
      b(3, 1) = 0.77E0_wp
      b(3, 2) = -13.28E0_wp
      b(4, 1) = -6.22E0_wp
      b(4, 2) = -6.21E0_wp
!
      Write (*, 99999) 'Matrix A:'
      Do i = 1, n
        Write (*, 99998)(a(i,j), j=1, n)
      End Do
!
      Write (*, *)
      Write (*, 99999) 'Right-hand-side matrix B:'
      Do i = 1, n
        Write (*, 99998)(b(i,j), j=1, nrhs)
      End Do

!     Factorize A
      Call dgetrf(n, n, a, lda, ipiv, info)
!
      Write (*, *)
      If (info==0) Then
!
!        Compute solution
        Call dgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info)
!
!        Print solution
        Write (*, 99999) 'Solution matrix X of equations A*X = B:'
        Do i = 1, n
          Write (*, 99998)(b(i,j), j=1, nrhs)
        End Do
      Else
        Write (*, 99999) 'The factor U of matrix A is singular'
      End If
!
99999 Format (A)
99998 Format (1X, 1P, 6E13.3)
!
    End Program

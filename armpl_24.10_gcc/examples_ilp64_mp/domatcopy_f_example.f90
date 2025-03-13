!   DOMATCOPY Example Program Text
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!
    Program main
!      .. Use Statements ..
      Use armpl_library, Only: domatcopy
!      .. Implicit NONE Statement ..
      Implicit None
!      .. Parameters ..
      Integer, Parameter               :: wp = kind(0.0D0)
      Integer, Parameter               :: nmax = 8
      Integer, Parameter               :: lda = nmax
      Integer, Parameter               :: ldb = nmax
      Character (1), Parameter         :: trans = 'N'
      Character (1), Parameter         :: order = 'C'
      Real (Kind=wp), Parameter        :: alpha = 1.1_wp
!     .. Local Scalars ..
      Integer                          :: i, j, n
!     .. Local Arrays ..
      Real (Kind=wp)                   :: a(lda, nmax), b(ldb, nmax)
!     .. Executable Statements ..
!
      Write (*, 99999)                                                         &
        'ARMPL example for DOMATCOPY'
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
      Write (*, 99997) 'alpha : ', alpha
      Write (*, *)
      Write (*, 99999) 'Matrix A:'
      Do i = 1, n
        Write (*, 99998)(a(i,j), j=1, n)
      End Do

!     Call domatcopy
      Call domatcopy(order, trans, n, n, alpha, a, lda, b, ldb)

!     Print B
      Write (*, 99999) 'Matrix B:'
      Do i = 1, n
         Write (*, 99998)(b(i,j), j=1, n)
      End Do
!
99999 Format (A)
99998 Format (1X, 1P, 6E13.3)
99997 Format (A, F5.3)
!
    End Program

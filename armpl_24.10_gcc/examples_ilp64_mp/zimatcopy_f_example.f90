!   ZIMATCOPY Example Program Text
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!
    Program main
!      .. Use Statements ..
      Use armpl_library, Only: zimatcopy
      Use, Intrinsic                   :: iso_fortran_env, Only: wp => real64
!      .. Implicit NONE Statement ..
      Implicit None
!      .. Parameters ..
      Integer, Parameter               :: m = 4
      Integer, Parameter               :: n = m
      Integer, Parameter               :: lda = m
      Integer, Parameter               :: ldb = m
      Character (1), Parameter         :: trans = 'C'
      Character (1), Parameter         :: order = 'C'

      Complex (Kind=wp), Parameter     :: alpha = (1.0_wp, 0.0_wp)
!     .. Local Scalars ..
      Integer                          :: i, j
!     .. Local Arrays ..
      Complex (Kind=wp), Dimension(n * ldb)     :: ab
!     .. Executable Statements ..
!
      Write (*, 99999)                                                         &
        'ARMPL example for ZIMATCOPY'
      Write (*, 99999)                                                         &
        '---------------------------------------------------------------'
      Write (*, *)
!
!     Initialize matrix AB, note that AB should
!     be big enough for A and B
      Do i = 1, n*ldb
        ab(i) = cmplx(i, i+1, kind=wp)
      End Do

!
!     Printing m, n, lda, ldb, trans, and order
      Write (*, "(A7, I4)") "m    : ", m
      Write (*, "(A7, I4)") "n    : ", n
      Write (*, "(A7, I4)") "lda  : ", lda
      Write (*, "(A7, I4)") "ldb  : ", ldb
      Write (*, "(A7, A4)")  "trans: ", trans
      Write (*, "(A7, A4)")  "order: ", order
      Write (*, *)

      Write (*, 99998) 'alpha : ', alpha
      Write (*, *)
      Write (*, 99999) 'Full Matrix AB before zimatcopy:'
      Do i = 1, ldb
         Do j = 1, n
            Write (*, 99997, advance="no") real(ab((j-1)*ldb + i)), aimag(ab((j-1)*ldb + i))
         End Do
         Write(*, *)
      End Do

!     Call zimatcopy
      Call zimatcopy(order, trans, m, n, alpha, ab, lda, ldb)

!     Print AB
      Write (*, 99999) 'Full Matrix AB after zimatcopy:'
      Do i = 1, ldb
         Do j = 1, n
            Write (*, 99997, advance="no") real(ab((j-1)*ldb + i)), aimag(ab((j-1)*ldb + i))
         End Do
         Write(*, *)
      End Do
!
99999 Format (A)
99998 Format (A, F6.3, Sp, F6.3, 'i')
99997 Format (1X, 1P, E10.3, Sp, E10.3, 'i')
!
    End Program

!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!   SPDX-FileCopyrightText: Copyright 2015-2024 NAG
!
!   Interface blocks for Fortran routines in ARMPL
!
    Module armpl_library
!     .. Use Statements ..
      Use armpl_blas
      Use armpl_lapack
      Use armpl_sparse
!     .. Interface Blocks ..
      Interface
        Subroutine armplinfo
        End Subroutine armplinfo
        Subroutine armplversion(major, minor, patch)
!     .. Implicit None Statement ..
          Implicit None
!     .. Scalar Arguments ..
          Integer :: major, minor, patch
        End Subroutine armplversion
      End Interface
    End Module armpl_library

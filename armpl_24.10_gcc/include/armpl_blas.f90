!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!   SPDX-FileCopyrightText: Copyright 2015-2024 NAG
!
!   Interface blocks for Fortran BLAS routines in ARMPL
!
  Module armpl_blas
! .. Interface Blocks ..
      Interface
        Subroutine caxpy(n, ca, cx, incx, cy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: ca
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: cx(*), cy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine caxpy
        Subroutine ccopy(n, cx, incx, cy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: cx(*), cy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ccopy
        Function cdotc(n, cx, incx, cy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: cx(*), cy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Complex (wp) :: cdotc
        End Function cdotc
        Function cdotu(n, cx, incx, cy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: cx(*), cy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Complex (wp) :: cdotu
        End Function cdotu
        Subroutine cgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, &
          incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, kl, ku, lda, m, n
          Character (1) :: trans
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cgbmv
        Subroutine cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: k, lda, ldb, ldc, m, n
          Character (1) :: transa, transb
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cgemm
        Subroutine cgemmt(uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: n, k, lda, ldb, ldc
          Character (1) :: uplo, transa, transb
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cgemmt
        Subroutine cgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, lda, m, n
          Character (1) :: trans
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cgemv
        Subroutine cgerbc(m, n, alpha, x, incx, y, incy, beta, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cgerbc
        Subroutine cgerbu(m, n, alpha, x, incx, y, incy, beta, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cgerbu
        Subroutine cgerc(m, n, alpha, x, incx, y, incy, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cgerc
        Subroutine cgeru(m, n, alpha, x, incx, y, incy, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cgeru
        Subroutine chbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, k, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine chbmv
        Subroutine chemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: lda, ldb, ldc, m, n
          Character (1) :: side, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine chemm
        Subroutine chemv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine chemv
        Subroutine cher(uplo, n, alpha, x, incx, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cher
        Subroutine cher2(uplo, n, alpha, x, incx, y, incy, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: incx, incy, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cher2
        Subroutine cher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Real (wp) :: beta
          Integer :: k, lda, ldb, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cher2k
        Subroutine cherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: k, lda, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cherk
        Subroutine chpmv(uplo, n, alpha, ap, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: ap(*), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine chpmv
        Subroutine chpr(uplo, n, alpha, x, incx, ap)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine chpr
        Subroutine chpr2(uplo, n, alpha, x, incx, y, incy, ap)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: incx, incy, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: ap(*), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine chpr2
        Subroutine crotg(ca, cb, c, s)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: ca, cb, s
          Real (wp) :: c
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine crotg
        Subroutine cscal(n, ca, cx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: ca
          Integer :: incx, n
! .. Array Arguments ..
          Complex (wp) :: cx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cscal
        Subroutine csrot(n, cx, incx, cy, incy, c, s)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: c, s
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: cx(*), cy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine csrot
        Subroutine csscal(n, sa, cx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: sa
          Integer :: incx, n
! .. Array Arguments ..
          Complex (wp) :: cx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine csscal
        Subroutine cswap(n, cx, incx, cy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: cx(*), cy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cswap
        Subroutine csymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: lda, ldb, ldc, m, n
          Character (1) :: side, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine csymm
        Subroutine csyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: k, lda, ldb, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine csyr2k
        Subroutine csyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: k, lda, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine csyrk
        Subroutine ctbmv(uplo, trans, diag, n, k, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, k, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ctbmv
        Subroutine ctbsv(uplo, trans, diag, n, k, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, k, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ctbsv
        Subroutine ctpmv(uplo, trans, diag, n, ap, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ctpmv
        Subroutine ctpsv(uplo, trans, diag, n, ap, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ctpsv
        Subroutine ctrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, &
          ldb)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: lda, ldb, m, n
          Character (1) :: diag, side, transa, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ctrmm
        Subroutine ctrmv(uplo, trans, diag, n, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ctrmv
        Subroutine ctrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, &
          ldb)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: lda, ldb, m, n
          Character (1) :: diag, side, transa, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ctrsm
        Subroutine ctrsv(uplo, trans, diag, n, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ctrsv
        Function dasum(n, dx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Real (wp) :: dx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: dasum
        End Function dasum
        Subroutine daxpy(n, da, dx, incx, dy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: da
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: dx(*), dy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine daxpy
        Function dcabs1(z)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: z
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: dcabs1
        End Function dcabs1
        Subroutine dcopy(n, dx, incx, dy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: dx(*), dy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dcopy
        Function ddot(n, dx, incx, dy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: dx(*), dy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: ddot
        End Function ddot
        Subroutine dgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, &
          incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, kl, ku, lda, m, n
          Character (1) :: trans
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dgbmv
        Subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: k, lda, ldb, ldc, m, n
          Character (1) :: transa, transb
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dgemm
        Subroutine dgemmt(uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: n, k, lda, ldb, ldc
          Character (1) :: uplo, transa, transb
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dgemmt
        Subroutine dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, lda, m, n
          Character (1) :: trans
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dgemv
        Subroutine dger(m, n, alpha, x, incx, y, incy, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dger
        Subroutine dgerb(m, n, alpha, x, incx, y, incy, beta, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dgerb
        subroutine simatcopy(order, transa, m, n, alpha, ab, lda, ldb)
          implicit none
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: m, n, lda, ldb
          Real (wp) :: alpha
          Character :: order, transa
! .. Array Arguments ..
          Real (wp) :: ab(*)
          Intrinsic :: kind
        end subroutine simatcopy
        subroutine dimatcopy(order, transa, m, n, alpha, ab, lda, ldb)
          implicit none
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: m, n, lda, ldb
          Real (wp) :: alpha
          Character :: order, transa
! .. Array Arguments ..
          Real (wp) :: ab(*)
          Intrinsic :: kind
        end subroutine dimatcopy
        subroutine cimatcopy(order, transa, m, n, alpha, ab, lda, ldb)
          implicit none
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: m, n, lda, ldb
          Complex(wp) :: alpha
          Character :: order, transa
! .. Array Arguments ..
          complex(wp) :: ab(*)
          Intrinsic :: kind
        end subroutine cimatcopy
        subroutine zimatcopy(order, transa, m, n, alpha, ab, lda, ldb)
          implicit none
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: m, n, lda, ldb
          Complex(wp) :: alpha
          Character :: order, transa
! .. Array Arguments ..
          complex(wp) :: ab(*)
          Intrinsic :: kind
        end subroutine zimatcopy
        Function dnrm2(n, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Real (wp) :: x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: dnrm2
        End Function dnrm2
        subroutine somatcopy(order, transa, m, n, alpha, a, lda, b, ldb)
          implicit none
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: m, n, lda, ldb
          Real (wp) :: alpha
          Character :: order, transa
! .. Array Arguments ..
          Real(wp) :: a(lda, *)
          Real(wp) :: b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        end subroutine somatcopy
        subroutine domatcopy(order, transa, m, n, alpha, a, lda, b, ldb)
          implicit none
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: m, n, lda, ldb
          Real (wp) :: alpha
          Character :: order, transa
! .. Array Arguments ..
          Real(wp) :: a(lda, *)
          Real(wp) :: b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        end subroutine domatcopy
        subroutine comatcopy(order, transa, m, n, alpha, a, lda, b, ldb)
          implicit none
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: m, n, lda, ldb
          Complex (wp) :: alpha
          Character :: order, transa
! .. Array Arguments ..
          Complex (wp) :: a(lda, *)
          Complex (wp) :: b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        end subroutine comatcopy
        subroutine zomatcopy(order, transa, m, n, alpha, a, lda, b, ldb)
          implicit none
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: m, n, lda, ldb
          Complex (wp) :: alpha
          Character :: order, transa
! .. Array Arguments ..
          Complex (wp) :: a(lda, *)
          Complex (wp) :: b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        end subroutine zomatcopy
        Subroutine drot(n, dx, incx, dy, incy, c, s)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: c, s
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: dx(*), dy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine drot
        Subroutine drotg(da, db, c, s)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: c, da, db, s
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine drotg
        Subroutine drotm(n, dx, incx, dy, incy, dparam)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: dparam(5), dx(*), dy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine drotm
        Subroutine drotmg(dd1, dd2, dx1, dy1, dparam)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: dd1, dd2, dx1, dy1
! .. Array Arguments ..
          Real (wp) :: dparam(5)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine drotmg
        Subroutine dsbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, k, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dsbmv
        Subroutine dscal(n, da, dx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: da
          Integer :: incx, n
! .. Array Arguments ..
          Real (wp) :: dx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dscal
        Function dsdot(n, sx, incx, sy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: dsdot
        End Function dsdot
        Subroutine dspmv(uplo, n, alpha, ap, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: ap(*), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dspmv
        Subroutine dspr(uplo, n, alpha, x, incx, ap)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dspr
        Subroutine dspr2(uplo, n, alpha, x, incx, y, incy, ap)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, incy, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: ap(*), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dspr2
        Subroutine dswap(n, dx, incx, dy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: dx(*), dy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dswap
        Subroutine dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: lda, ldb, ldc, m, n
          Character (1) :: side, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dsymm
        Subroutine dsymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dsymv
        Subroutine dsyr(uplo, n, alpha, x, incx, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dsyr
        Subroutine dsyr2(uplo, n, alpha, x, incx, y, incy, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, incy, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dsyr2
        Subroutine dsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: k, lda, ldb, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dsyr2k
        Subroutine dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: k, lda, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dsyrk
        Subroutine dtbmv(uplo, trans, diag, n, k, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, k, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dtbmv
        Subroutine dtbsv(uplo, trans, diag, n, k, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, k, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dtbsv
        Subroutine dtpmv(uplo, trans, diag, n, ap, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dtpmv
        Subroutine dtpsv(uplo, trans, diag, n, ap, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dtpsv
        Subroutine dtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, &
          ldb)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: lda, ldb, m, n
          Character (1) :: diag, side, transa, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dtrmm
        Subroutine dtrmv(uplo, trans, diag, n, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dtrmv
        Subroutine dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, &
          ldb)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: lda, ldb, m, n
          Character (1) :: diag, side, transa, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dtrsm
        Subroutine dtrsv(uplo, trans, diag, n, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dtrsv
        Function dzasum(n, zx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Complex (wp) :: zx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: dzasum
        End Function dzasum
        Function dznrm2(n, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Complex (wp) :: x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: dznrm2
        End Function dznrm2
        Function icamax(n, cx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Function Return Value ..
          Integer :: icamax
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Complex (wp) :: cx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Function icamax
        Function idamax(n, dx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Function Return Value ..
          Integer :: idamax
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Real (wp) :: dx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Function idamax
        Function isamax(n, sx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Function Return Value ..
          Integer :: isamax
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Real (wp) :: sx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Function isamax
        Function izamax(n, zx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Function Return Value ..
          Integer :: izamax
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Complex (wp) :: zx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Function izamax
        Function lsame(ca, cb)
! .. Implicit None Statement ..
          Implicit None
! .. Function Return Value ..
          Logical :: lsame
! .. Scalar Arguments ..
          Character (1) :: ca, cb
        End Function lsame
        Function sasum(n, sx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Real (wp) :: sx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: sasum
        End Function sasum
        Subroutine saxpy(n, sa, sx, incx, sy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: sa
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine saxpy
        Function scabs1(z)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: z
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: scabs1
        End Function scabs1
        Function scasum(n, cx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Complex (wp) :: cx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: scasum
        End Function scasum
        Function scnrm2(n, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Complex (wp) :: x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: scnrm2
        End Function scnrm2
        Subroutine scopy(n, sx, incx, sy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine scopy
        Function sdot(n, sx, incx, sy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: sdot
        End Function sdot
        Function sdsdot(n, sb, sx, incx, sy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: sb
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: sdsdot
        End Function sdsdot
        Subroutine sgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, &
          incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, kl, ku, lda, m, n
          Character (1) :: trans
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sgbmv
        Subroutine sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: k, lda, ldb, ldc, m, n
          Character (1) :: transa, transb
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sgemm
        Subroutine sgemmt(uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: n, k, lda, ldb, ldc
          Character (1) :: uplo, transa, transb
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sgemmt
        Subroutine sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, lda, m, n
          Character (1) :: trans
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sgemv
        Subroutine sger(m, n, alpha, x, incx, y, incy, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sger
        Subroutine sgerb(m, n, alpha, x, incx, y, incy, beta, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sgerb
        Function snrm2(n, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, n
! .. Array Arguments ..
          Real (wp) :: x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Real (wp) :: snrm2
        End Function snrm2
        Subroutine srot(n, sx, incx, sy, incy, c, s)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: c, s
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine srot
        Subroutine srotg(sa, sb, c, s)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: c, s, sa, sb
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine srotg
        Subroutine srotm(n, sx, incx, sy, incy, sparam)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: sparam(5), sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine srotm
        Subroutine srotmg(sd1, sd2, sx1, sy1, sparam)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: sd1, sd2, sx1, sy1
! .. Array Arguments ..
          Real (wp) :: sparam(5)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine srotmg
        Subroutine ssbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, k, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ssbmv
        Subroutine sscal(n, sa, sx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: sa
          Integer :: incx, n
! .. Array Arguments ..
          Real (wp) :: sx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sscal
        Subroutine sspmv(uplo, n, alpha, ap, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: ap(*), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sspmv
        Subroutine sspr(uplo, n, alpha, x, incx, ap)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sspr
        Subroutine sspr2(uplo, n, alpha, x, incx, y, incy, ap)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, incy, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: ap(*), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sspr2
        Subroutine sswap(n, sx, incx, sy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine sswap
        Subroutine ssymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: lda, ldb, ldc, m, n
          Character (1) :: side, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ssymm
        Subroutine ssymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: incx, incy, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ssymv
        Subroutine ssyr(uplo, n, alpha, x, incx, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ssyr
        Subroutine ssyr2(uplo, n, alpha, x, incx, y, incy, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, incy, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ssyr2
        Subroutine ssyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: k, lda, ldb, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ssyr2k
        Subroutine ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: k, lda, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ssyrk
        Subroutine stbmv(uplo, trans, diag, n, k, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, k, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine stbmv
        Subroutine stbsv(uplo, trans, diag, n, k, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, k, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine stbsv
        Subroutine stpmv(uplo, trans, diag, n, ap, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine stpmv
        Subroutine stpsv(uplo, trans, diag, n, ap, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine stpsv
        Subroutine strmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, &
          ldb)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: lda, ldb, m, n
          Character (1) :: diag, side, transa, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine strmm
        Subroutine strmv(uplo, trans, diag, n, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine strmv
        Subroutine strsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, &
          ldb)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: lda, ldb, m, n
          Character (1) :: diag, side, transa, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine strsm
        Subroutine strsv(uplo, trans, diag, n, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Integer :: incx, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Real (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine strsv
        Subroutine xerbla(srname, info)
! .. Implicit None Statement ..
          Implicit None
! .. Scalar Arguments ..
          Integer :: info
          Character (*) :: srname
        End Subroutine xerbla
        Subroutine xerbla_array(srname_array, srname_len, info)
! .. Implicit None Statement ..
          Implicit None
! .. Scalar Arguments ..
          Integer :: info, srname_len
! .. Array Arguments ..
          Character (1) :: srname_array(srname_len)
        End Subroutine xerbla_array
        Subroutine zaxpy(n, za, zx, incx, zy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: za
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: zx(*), zy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zaxpy
        Subroutine zcopy(n, zx, incx, zy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: zx(*), zy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zcopy
        Function zdotc(n, zx, incx, zy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: zx(*), zy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Complex (wp) :: zdotc
        End Function zdotc
        Function zdotu(n, zx, incx, zy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: zx(*), zy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
! .. Function Return Value ..
          Complex (wp) :: zdotu
        End Function zdotu
        Subroutine zdrot(n, cx, incx, cy, incy, c, s)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: c, s
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: cx(*), cy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zdrot
        Subroutine zdscal(n, da, zx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: da
          Integer :: incx, n
! .. Array Arguments ..
          Complex (wp) :: zx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zdscal
        Subroutine zgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, &
          incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, kl, ku, lda, m, n
          Character (1) :: trans
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zgbmv
        Subroutine zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: k, lda, ldb, ldc, m, n
          Character (1) :: transa, transb
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zgemm
        Subroutine zgemmt(uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: n, k, lda, ldb, ldc
          Character (1) :: uplo, transa, transb
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zgemmt
        Subroutine zgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, lda, m, n
          Character (1) :: trans
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zgemv
        Subroutine zgerbc(m, n, alpha, x, incx, y, incy, beta, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zgerbc
        Subroutine zgerbu(m, n, alpha, x, incx, y, incy, beta, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zgerbu
        Subroutine zgerc(m, n, alpha, x, incx, y, incy, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zgerc
        Subroutine zgeru(m, n, alpha, x, incx, y, incy, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: incx, incy, lda, m, n
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zgeru
        Subroutine zhbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, k, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zhbmv
        Subroutine zhemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: lda, ldb, ldc, m, n
          Character (1) :: side, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zhemm
        Subroutine zhemv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zhemv
        Subroutine zher(uplo, n, alpha, x, incx, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zher
        Subroutine zher2(uplo, n, alpha, x, incx, y, incy, a, lda)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: incx, incy, lda, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zher2
        Subroutine zher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Real (wp) :: beta
          Integer :: k, lda, ldb, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zher2k
        Subroutine zherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha, beta
          Integer :: k, lda, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zherk
        Subroutine zhpmv(uplo, n, alpha, ap, x, incx, beta, y, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: incx, incy, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: ap(*), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zhpmv
        Subroutine zhpr(uplo, n, alpha, x, incx, ap)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Real (wp) :: alpha
          Integer :: incx, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zhpr
        Subroutine zhpr2(uplo, n, alpha, x, incx, y, incy, ap)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: incx, incy, n
          Character (1) :: uplo
! .. Array Arguments ..
          Complex (wp) :: ap(*), x(*), y(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zhpr2
        Subroutine zrotg(ca, cb, c, s)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: ca, cb, s
          Real (wp) :: c
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zrotg
        Subroutine zscal(n, za, zx, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: za
          Integer :: incx, n
! .. Array Arguments ..
          Complex (wp) :: zx(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zscal
        Subroutine zswap(n, zx, incx, zy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: zx(*), zy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zswap
        Subroutine zsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: lda, ldb, ldc, m, n
          Character (1) :: side, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zsymm
        Subroutine zsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, &
          ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: k, lda, ldb, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zsyr2k
        Subroutine zsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: k, lda, ldc, n
          Character (1) :: trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zsyrk
        Subroutine ztbmv(uplo, trans, diag, n, k, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, k, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ztbmv
        Subroutine ztbsv(uplo, trans, diag, n, k, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, k, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ztbsv
        Subroutine ztpmv(uplo, trans, diag, n, ap, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ztpmv
        Subroutine ztpsv(uplo, trans, diag, n, ap, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: ap(*), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ztpsv
        Subroutine ztrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, &
          ldb)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: lda, ldb, m, n
          Character (1) :: diag, side, transa, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ztrmm
        Subroutine ztrmv(uplo, trans, diag, n, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ztrmv
        Subroutine ztrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, &
          ldb)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha
          Integer :: lda, ldb, m, n
          Character (1) :: diag, side, transa, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ztrsm
        Subroutine ztrsv(uplo, trans, diag, n, a, lda, x, incx)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Integer :: incx, lda, n
          Character (1) :: diag, trans, uplo
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), x(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine ztrsv
        Subroutine sgemm_batch(transa_array, transb_array, m_array, n_array, &
          k_array, alpha_array, a_array, lda_array, b_array, ldb_array, &
          beta_array, c_array, ldc_array, group_count, group_size)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
          Integer, Parameter :: i8 = selected_int_kind(10)
! .. Scalar Arguments ..
          Integer :: group_count
! .. Array Arguments ..
          Character (1) :: transa_array(*), transb_array(*)
          Integer :: m_array(*), n_array(*), k_array(*)
          Integer :: lda_array(*), ldb_array(*), ldc_array(*)
          Integer :: group_size(*)
          Integer(i8) :: a_array(*), b_array(*), c_array(*)
          Real (wp) :: alpha_array(*), beta_array(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind, selected_int_kind
        End Subroutine sgemm_batch
        Subroutine dgemm_batch(transa_array, transb_array, m_array, n_array, &
          k_array, alpha_array, a_array, lda_array, b_array, ldb_array, &
          beta_array, c_array, ldc_array, group_count, group_size)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.D0)
          Integer, Parameter :: i8 = selected_int_kind(10)
! .. Scalar Arguments ..
          Integer :: group_count
! .. Array Arguments ..
          Character (1) :: transa_array(*), transb_array(*)
          Integer :: m_array(*), n_array(*), k_array(*)
          Integer :: lda_array(*), ldb_array(*), ldc_array(*)
          Integer :: group_size(*)
          Integer(i8) :: a_array(*), b_array(*), c_array(*)
          Real (wp) :: alpha_array(*), beta_array(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind, selected_int_kind
        End Subroutine dgemm_batch
        Subroutine cgemm_batch(transa_array, transb_array, m_array, n_array, &
          k_array, alpha_array, a_array, lda_array, b_array, ldb_array, &
          beta_array, c_array, ldc_array, group_count, group_size)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
          Integer, Parameter :: i8 = selected_int_kind(10)
! .. Scalar Arguments ..
          Integer :: group_count
! .. Array Arguments ..
          Character (1) :: transa_array(*), transb_array(*)
          Integer :: m_array(*), n_array(*), k_array(*)
          Integer :: lda_array(*), ldb_array(*), ldc_array(*)
          Integer :: group_size(*)
          Integer(i8) :: a_array(*), b_array(*), c_array(*)
          Complex (wp) :: alpha_array(*), beta_array(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind, selected_int_kind
        End Subroutine cgemm_batch
        Subroutine zgemm_batch(transa_array, transb_array, m_array, n_array, &
          k_array, alpha_array, a_array, lda_array, b_array, ldb_array, &
          beta_array, c_array, ldc_array, group_count, group_size)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.D0)
          Integer, Parameter :: i8 = selected_int_kind(10)
! .. Scalar Arguments ..
          Integer :: group_count
! .. Array Arguments ..
          Character (1) :: transa_array(*), transb_array(*)
          Integer :: m_array(*), n_array(*), k_array(*)
          Integer :: lda_array(*), ldb_array(*), ldc_array(*)
          Integer :: group_size(*)
          Integer(i8) :: a_array(*), b_array(*), c_array(*)
          Complex (wp) :: alpha_array(*), beta_array(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind, selected_int_kind
        End Subroutine zgemm_batch
        Subroutine cgemm3m(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: k, lda, ldb, ldc, m, n
          Character (1) :: transa, transb
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cgemm3m
        Subroutine zgemm3m(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
          c, ldc)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0D0)
! .. Scalar Arguments ..
          Complex (wp) :: alpha, beta
          Integer :: k, lda, ldb, ldc, m, n
          Character (1) :: transa, transb
! .. Array Arguments ..
          Complex (wp) :: a(lda, *), b(ldb, *), c(ldc, *)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zgemm3m
        Subroutine saxpby(n, sa, sx, incx, sb, sy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: sa, sb
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine saxpby
        Subroutine daxpby(n, sa, sx, incx, sb, sy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.D0)
! .. Scalar Arguments ..
          Real (wp) :: sa, sb
          Integer :: incx, incy, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine daxpby
        Subroutine caxpby(n, sa, sx, incx, sb, sy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: sa, sb
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine caxpby
        Subroutine zaxpby(n, sa, sx, incx, sb, sy, incy)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.D0)
! .. Scalar Arguments ..
          Complex (wp) :: sa, sb
          Integer :: incx, incy, n
! .. Array Arguments ..
          Complex (wp) :: sx(*), sy(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zaxpby
        Subroutine swaxpby(n, sa, sx, incx, sb, sy, incy, sw, incw)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Real (wp) :: sa, sb
          Integer :: incx, incy, incw, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*), sw(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine swaxpby
        Subroutine dwaxpby(n, sa, sx, incx, sb, sy, incy, sw, incw)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.D0)
! .. Scalar Arguments ..
          Real (wp) :: sa, sb
          Integer :: incx, incy, incw, n
! .. Array Arguments ..
          Real (wp) :: sx(*), sy(*), sw(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine dwaxpby
        Subroutine cwaxpby(n, sa, sx, incx, sb, sy, incy, sw, incw)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.0)
! .. Scalar Arguments ..
          Complex (wp) :: sa, sb
          Integer :: incx, incy, incw, n
! .. Array Arguments ..
          Complex (wp) :: sx(*), sy(*), sw(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine cwaxpby
        Subroutine zwaxpby(n, sa, sx, incx, sb, sy, incy, sw, incw)
! .. Implicit None Statement ..
          Implicit None
! .. Parameters ..
          Integer, Parameter :: wp = kind(0.D0)
! .. Scalar Arguments ..
          Complex (wp) :: sa, sb
          Integer :: incx, incy, incw, n
! .. Array Arguments ..
          Complex (wp) :: sx(*), sy(*), sw(*)
! .. Intrinsic Procedures ..
          Intrinsic :: kind
        End Subroutine zwaxpby
      End Interface
    End Module armpl_blas

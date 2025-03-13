!   SGEMM Example Program Text
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!
    Program main
!      .. Use Statements ..
      Use armpl_library, Only: sgemm_batch
!      .. Implicit NONE Statement ..
      Implicit None
!      .. Parameters ..
      Integer, Parameter               :: nmax = 250
!      .. Local scalars ..
      Integer                          :: i, j, ii, jj
      Integer                          :: group_count, total_batch_count
!      .. Local Arrays ..
      Integer, Allocatable, Target     :: m(:), n(:), k(:)
      Integer, Allocatable             :: lda(:), ldb(:), ldc(:), &
                                          group_size(:)
      Integer, Pointer                 :: ma, na, mb, nb
      Real, Allocatable                :: alpha(:), beta(:)
      Real, Allocatable                :: a(:, :), b(:, :), c(:, :)
!      .. Arrays for transpose options ..
      Character *1, Allocatable        :: transa(:), transb(:)
!      .. Arrays for pointers to matrices ..
      Integer *8, Allocatable          :: ap(:), bp(:), cp(:)
      Integer                          :: aptr, bptr, cptr, ki
!      .. Executable statements ..

       Write (*, Fmt=99995)                                            &
           'ARMPL example for SGEMM_BATCH'
       Write (*, Fmt=99995)                                            &
           '-------------------------------------------------------'
      Write (*, *) ''

      group_count = 3
      Allocate (m(group_count), n(group_count), k(group_count), &
        lda(group_count), ldb(group_count), ldc(group_count), &
        group_size(group_count), alpha(group_count), beta(group_count), &
        a(nmax,nmax), b(nmax,nmax), c(nmax,nmax), transa(group_count), &
        transb(group_count))
!      Set values to matrices a, b and c
      Do j = 1, nmax
        Do i = 1, nmax
          a(i, j) = i + j
          b(i, j) = i - j
          c(i, j) = 0
        End Do
      End Do

!      Set array with maxtrix A's transpose options
      transa(1) = 'N'
      transa(2) = 'T'
      transa(3) = 'N'

!      Set array with matrix B's transpose options
      transb(1) = 'N'
      transb(2) = 'T'
      transb(3) = 'N'

!      Set sizes for matrices
!      Set values for m array
      m(1) = 2
      m(2) = 2
      m(3) = 3

!      Set values for k array
      k(1) = 3
      k(2) = 2
      k(3) = 3

!      Set values for n array
      n(1) = 2
      n(2) = 3
      n(3) = 2

!      Set values for alpha array
      alpha(1) = 1.0
      alpha(2) = 1.0
      alpha(3) = 1.0

!      Set values for beta array
      beta(1) = 0.0
      beta(2) = 0.0
      beta(3) = 0.0

!      Set values for lda array
      lda(1) = nmax
      lda(2) = nmax
      lda(3) = nmax

!      Set values for ldb array
      ldb(1) = nmax
      ldb(2) = nmax
      ldb(3) = nmax

!      Set values for ldc array
      ldc(1) = nmax
      ldc(2) = nmax
      ldc(3) = nmax

!      Set values for group size array
      group_size(1) = 2
      group_size(2) = 3
      group_size(3) = 4

      total_batch_count = 0
      Do i = 1, group_count
        total_batch_count = total_batch_count + group_size(i)
      End Do

      Allocate (ap(total_batch_count), bp(total_batch_count), &
        cp(total_batch_count))

!      Set pointers for A matrices
!      Pointers to group 1
!      - trans opt = N. m = 2
      ap(1) = loc(a(1,1))
      ap(2) = loc(a(3,1))
!      Pointers to group 2
!      - trans opt = T, k = 2
      ap(3) = loc(a(5,1))
      ap(4) = loc(a(7,1))
      ap(5) = loc(a(9,1))
!      Pointers to group 3
!      - trans opt = N, m = 3
      ap(6) = loc(a(11,1))
      ap(7) = loc(a(14,1))
      ap(8) = loc(a(17,1))
      ap(9) = loc(a(20,1))

!      Set pointers to B matrices
!      Pointers to group 1
!      - trans opt = N, k = 3
      bp(1) = loc(b(1,1))
      bp(2) = loc(b(4,1))
!      Pointers to group 2
!      - trans opt = T, n = 3
      bp(3) = loc(b(7,1))
      bp(4) = loc(b(10,1))
      bp(5) = loc(b(13,1))
!      Pointers to group 3
!      - trans opt = N, k = 3
      bp(6) = loc(b(16,1))
      bp(7) = loc(b(19,1))
      bp(8) = loc(b(22,1))
      bp(9) = loc(b(25,1))

!      Set pointers to C matrices
!      Pointers to group 1
!      m = 2
      cp(1) = loc(c(1,1))
      cp(2) = loc(c(3,1))
!      Pointer to group 2
!      m = 2
      cp(3) = loc(c(5,1))
      cp(4) = loc(c(7,1))
      cp(5) = loc(c(9,1))
!      Pointers to group 3
!      m = 3
      cp(6) = loc(c(11,1))
      cp(7) = loc(c(14,1))
      cp(8) = loc(c(17,1))
      cp(9) = loc(c(20,1))

!      Call the function
      Call sgemm_batch(transa, transb, m, n, k, alpha, ap, lda, bp, ldb, beta, &
        cp, ldc, group_count, group_size)

!
!      Print all the data
!
!      Pointer to matrices
      aptr = 0
      bptr = 0
      cptr = 0
      ki = 1

      Do i = 1, group_count

        If (transa(i)=='N') Then
          ma => m(i)
          na => k(i)
        Else
          ma => k(i)
          na => m(i)
        End If

        If (transb(i)=='N') Then
          mb => k(i)
          nb => n(i)
        Else
          mb => n(i)
          nb => k(i)
        End If

!        Print each entry of each group
        Write (*, Fmt=99999) i
        Do j = 1, group_size(i)
!           Print matrix A
          Write (*, Fmt=99995) '**** Matrix A'
          Write (*, Fmt=99996) transa(i)
          Do ii = 1, ma
            Write (*, Fmt=99997)(a(aptr+ii,jj), jj=1, na)
          End Do
          Write (*, *) ''
!           Print matrix B
          Write (*, Fmt=99995) '**** Matrix B'
          Write (*, Fmt=99996) transb(i)
          Do ii = 1, mb
            Write (*, Fmt=99997)(b(bptr+ii,jj), jj=1, nb)
          End Do
          Write (*, *) ''
          Write (*, Fmt=99998) 'alpha : ', alpha(i)
          Write (*, Fmt=99998) 'beta  : ', beta(i)
          Write (*, *) ''
!           Print matrix C
          Write (*, Fmt=99995) '**** Matrix C'
          Do ii = 1, m(i)
            Write (*, Fmt=99997)(c(cptr+ii,jj), jj=1, n(i))
          End Do
          Write (*, *) ''

!           Update pointer based on transpose option
          If (transa(i)=='N') Then
            aptr = aptr + m(i)
          Else
            aptr = aptr + k(i)
          End If

          If (transb(i)=='N') Then
            bptr = bptr + k(i)
          Else
            bptr = bptr + n(i)
          End If

          cptr = cptr + m(i)

          ki = ki + 1

          Write (*, Fmt=99995) '=============end entry'
        End Do
        Write (*, Fmt=99995) '=============end group============='

      End Do

      Deallocate (m, n, k, lda, ldb, ldc, group_size, alpha, beta, a, b, c, &
        ap, bp, cp, transa, transb)
99999 Format ('******** Group: ', I2)
99998 Format ('** ', A, F7.3)
99997 Format (1X, 1P, 7E13.3)
99996 Format ('** Trans option: ', A1)
99995 Format (A)
    End Program

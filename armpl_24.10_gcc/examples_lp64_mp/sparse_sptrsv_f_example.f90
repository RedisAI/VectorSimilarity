!   Single precision sparse matrix triangular solve example
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!
    Subroutine check_result(armpl_mat, info)
      Use armpl_library
      Integer (Kind=armpl_i8) armpl_mat
      Integer info
      If (info/=armpl_status_success) Then
        Call armpl_spmat_print_err(armpl_mat)
        Stop 'Exiting due to error'
      End If
    End Subroutine check_result

    Program sptrsv_f_example
      Use armpl_library
      Implicit None

!     1. Set-up local CSR structure
      Integer (Kind=armpl_i8) armpl_mat
      Integer, Parameter               :: n = 8
      Integer, Parameter               :: nnz = 18
      Real (Kind=armpl_r32)            :: alpha = 2.0
      Integer                          :: flags = 0
      Real (Kind=armpl_r32), Dimension (nnz) :: vals = (/ 1., 1., 1., 1., 1., &
                                          2., 3., 1., 3., 4., 1., 4., 5., 1., &
                                          11., 14., 16., 18. /)
      Integer, Dimension (n+1)         :: row_ptr = (/ 1, 5, 8, 11, 14, 16, &
                                          17, 18, 19 /)
      Integer, Dimension (nnz)         :: col_indx = (/ 1, 3, 6, 8, 2, 4, 7, &
                                          3, 5, 6, 4, 5, 7, 5, 8, 6, 7, 8 /)
      Real (Kind=armpl_r32), Allocatable, Dimension (:) :: x, y
      Integer                          :: info
      Integer                          :: i

!     2. Set-up Arm Performance Libraries sparse matrix object
      Call armpl_spmat_create_csr(armpl_mat, n, n, row_ptr, col_indx, vals, &
        flags, info)
      Call check_result(armpl_mat, info)

!     3. Supply any hints that are about the SpTRSV calculations to be performed
      Call armpl_spmat_hint(armpl_mat, armpl_sparse_hint_spsv_operation, &
        armpl_sparse_operation_notrans, info)
      Call check_result(armpl_mat, info)

!     4. Call an optimization process that will learn from the hints you have previously supplied
      Call armpl_spsv_optimize(armpl_mat, info)
      Call check_result(armpl_mat, info)

!     5. Setup input and output vectors, then do sparse solve and print result.
      Allocate (x(n))
      Allocate (y(n))
      Do i = 1, n
          y(i) = 2.0 + real(i-1, kind=armpl_r32)
      End Do

      Call armpl_spsv_exec(armpl_sparse_operation_notrans, armpl_mat, &
          x, alpha, y, info)
      Call check_result(armpl_mat, info)

      Write (*, '(a)') 'Computed solution vector x:'
      Do i = 1, n
        Write (*, '(a, f4.1)') '  ', x(i)
      End Do

!     6. Destroy created matrix to free any memory
      Call armpl_spmat_destroy(armpl_mat, info)
      If (info/=armpl_status_success) Stop 'Error in armpl_spmat_destroy'

!     7. Free user allocated storage
      Deallocate (x)
      Deallocate (y)

    End Program

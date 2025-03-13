!   Single precision sparse matrix-vector multiplication example
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

    Program spmv_f_example
      Use armpl_library
      Implicit None

      Integer (Kind=armpl_i8) armpl_mat
      Integer                          :: ntests = 1000
      Integer, Parameter               :: m = 5
      Integer, Parameter               :: n = 5
      Integer, Parameter               :: nnz = 12
      Real (Kind=armpl_r32)            :: alpha = 1.0, beta = 0.0
      Integer                          :: flags = 0
      Real (Kind=armpl_r32), Dimension (nnz) :: vals = (/ 1., 2., 3., 4., 5., &
                                          6., 7., 8., 9., 10., 11., 12. /)
      Integer, Dimension (m+1)         :: row_ptr = (/ 1, 3, 5, 8, 10, 13 /)
      Integer, Dimension (nnz)         :: col_indx = (/ 1, 3, 2, 4, 2, 3, 4, 3 &
                                          , 4, 3, 4, 5 /)
      Real (Kind=armpl_r32), Allocatable, Dimension (:) :: x, y
      Integer                          :: info
      Integer                          :: i

      Call armpl_spmat_create_csr(armpl_mat, m, n, row_ptr, col_indx, vals, &
        flags, info)
      Call check_result(armpl_mat, info)

      Call armpl_spmat_hint(armpl_mat, armpl_sparse_hint_structure, &
        armpl_sparse_structure_unstructured, info)
      Call check_result(armpl_mat, info)

      Call armpl_spmat_hint(armpl_mat, armpl_sparse_hint_spmv_operation, &
        armpl_sparse_operation_notrans, info)
      Call check_result(armpl_mat, info)

      Call armpl_spmat_hint(armpl_mat, armpl_sparse_hint_spmv_invocations, &
        armpl_sparse_invocations_many, info)
      Call check_result(armpl_mat, info)

      Call armpl_spmv_optimize(armpl_mat, info)
      Call check_result(armpl_mat, info)

      Allocate (x(n))
      x(:) = 1.0

      Allocate (y(m))

      Do i = 1, ntests
        Call armpl_spmv_exec(armpl_sparse_operation_notrans, alpha, armpl_mat, &
          x, beta, y, info)
        Call check_result(armpl_mat, info)
      End Do

      Do i = 1, n
        Write (*, '(a, f4.1)') '  ', y(i)
      End Do

      Call armpl_spmat_destroy(armpl_mat, info)
      If (info/=armpl_status_success) Stop 'Error in armpl_spmat_destroy'

      Deallocate (x)
      Deallocate (y)

    End Program

!   Double precision sparse matrix-matrix multiplication example
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

    Program spmm_f_example
      Use armpl_library
      Use iso_c_binding
      Implicit None

!   1. Set-up local CSR structure
      Integer (Kind=armpl_i8) armpl_mat_a, armpl_mat_b, armpl_mat_c
      Integer, Parameter               :: m = 5
      Integer, Parameter               :: n = 5
      Integer, Parameter               :: k = 5
      Integer, Parameter               :: nnz_a = 12
      Integer, Parameter               :: nnz_b = 9
      Real (Kind=armpl_r64)            :: alpha = 2.0, beta = 0.0
      Integer                          :: creation_flags = 0

      Real (Kind=armpl_r64), Dimension (nnz_a) :: vals_a = (/ 1., 2., 3., 4., &
                                          5., 6., 7., 8., 9., 10., 11., 12. /)
      Integer, Dimension (m+1)         :: row_ptr_a = (/ 1, 3, 5, 8, 10, 13 /)
      Integer, Dimension (nnz_a)       :: col_indx_a = (/ 1, 3, 2, 4, 2, 3, 4, &
                                          3, 4, 3, 4, 5 /)

      Real (Kind=armpl_r64), Dimension (nnz_b) :: vals_b = (/ 1., 2., 3., 4., &
                                          5., 6., 7., 8., 9. /)
      Integer, Dimension (k+1)         :: row_ptr_b = (/ 1, 3, 5, 7, 8, 10 /)
      Integer, Dimension (nnz_b)       :: col_indx_b = (/ 1, 3, 2, 4, 1, 2, 5, &
                                          3, 5 /)

      Integer                          :: out_m, out_n
      Integer, Allocatable             :: out_row_ptr(:), out_col_indx(:)
      Real (Kind=armpl_r64), Allocatable :: out_vals(:)

      Integer                          :: info

!     Variables/functions used when building with an older version of gfortran,
!     in which case the recommendation is to use the armpl_spmat_export* functions
!     via the C interfaces provided in src/sparse_export_ftn_libgfortran_ne_5.c
      Integer                          :: libgfortran_mismatch
      External                         :: query_mismatch
      External                         :: armpl_spmat_export_csr_wrapper_d
      External                         :: armpl_spmat_export_deallocate_wrapper
      Type(C_ptr)                      :: c_out_row_ptr, c_out_col_indx
      Type(C_ptr)                      :: c_out_vals
      Integer, Pointer                 :: p_out_row_ptr(:), p_out_col_indx(:)
      Real (Kind=armpl_r64), Pointer   :: p_out_vals(:)
      Integer                          :: index_base, nnz_c

!   2a. Set-up Arm Performance Libraries sparse matrix object for A
      Call armpl_spmat_create_csr(armpl_mat_a, m, k, row_ptr_a, col_indx_a, &
        vals_a, creation_flags, info)
      Call check_result(armpl_mat_a, info)

!   2b. Set-up Arm Performance Libraries sparse matrix object for B
      Call armpl_spmat_create_csr(armpl_mat_b, k, n, row_ptr_b, col_indx_b, &
        vals_b, creation_flags, info)
      Call check_result(armpl_mat_b, info)

!   2c. Set-up Arm Performance Libraries sparse matrix object for C
      armpl_mat_c = armpl_spmat_create_null(m, n)

!   3a. Supply any pertinent information that is known about the matrix A
      Call armpl_spmat_hint(armpl_mat_a, armpl_sparse_hint_structure, &
        armpl_sparse_structure_unstructured, info)
      Call check_result(armpl_mat_a, info)

!   3b. Supply any pertinent information that is known about the matrix B
      Call armpl_spmat_hint(armpl_mat_b, armpl_sparse_hint_structure, &
        armpl_sparse_structure_unstructured, info)
      Call check_result(armpl_mat_b, info)

!   3c. Supply any hints that are about the SpMM calculations to be performed with matrix A
      Call armpl_spmat_hint(armpl_mat_a, armpl_sparse_hint_spmm_operation, &
        armpl_sparse_operation_notrans, info)
      Call check_result(armpl_mat_a, info)

      Call armpl_spmat_hint(armpl_mat_a, armpl_sparse_hint_spmm_invocations, &
        armpl_sparse_invocations_many, info)
      Call check_result(armpl_mat_a, info)

!   3c. Supply any hints that are about the SpMM calculations to be performed with matrix B
      Call armpl_spmat_hint(armpl_mat_b, armpl_sparse_hint_spmm_operation, &
        armpl_sparse_operation_notrans, info)
      Call check_result(armpl_mat_b, info)

      Call armpl_spmat_hint(armpl_mat_b, armpl_sparse_hint_spmm_invocations, &
        armpl_sparse_invocations_many, info)
      Call check_result(armpl_mat_b, info)

!   3d. Hint that the full structure of the output matrix should be set up in the optimize phase
      Call armpl_spmat_hint(armpl_mat_c, armpl_sparse_hint_spmm_strategy, armpl&
        &_sparse_spmm_strat_opt_full_struct, info)
      Call check_result(armpl_mat_c, info)

!   4. Optimize the matrices based on the hints previously supplied
      Call armpl_spmm_optimize(armpl_sparse_operation_notrans, &
        armpl_sparse_operation_notrans, armpl_sparse_scalar_any, armpl_mat_a, &
        armpl_mat_b, armpl_sparse_scalar_zero, armpl_mat_c, info)
      Call check_result(armpl_mat_c, info)

!   5a. Do SpMM
      Call armpl_spmm_exec(armpl_sparse_operation_notrans, &
        armpl_sparse_operation_notrans, alpha, armpl_mat_a, armpl_mat_b, beta, &
        armpl_mat_c, info)
      Call check_result(armpl_mat_c, info)

!   6. Print C - export and write output. Check we are using a recent gfortran
!      (libgfortran.so.5). If not, use the C interfaces for export/deallocation
!      of exported arrays.
      Call query_mismatch(libgfortran_mismatch)
      if (libgfortran_mismatch .eq. 1) then

          Call armpl_spmat_export_csr_wrapper_d(armpl_mat_c, 1, out_m, out_n, c_out_row_ptr, &
            c_out_col_indx, c_out_vals, info)
          Call armpl_spmat_query(armpl_mat_c, index_base, out_m, out_n, nnz_c, info)
          Call c_f_pointer(c_out_row_ptr, p_out_row_ptr, [out_m+1])
          Call c_f_pointer(c_out_col_indx, p_out_col_indx, [nnz_c])
          Call c_f_pointer(c_out_vals, p_out_vals, [nnz_c])

          Call check_result(armpl_mat_c, info)

          Write (*, 99997) 'Matrix C:'
          Write (*, 99997) 'Values:'
          Write (*, 99998) p_out_vals
          Write (*, 99997) 'Column Indices:'
          Write (*, 99999) p_out_col_indx
          Write (*, 99997) 'Row Pointer:'
          Write (*, 99999) p_out_row_ptr

!   Free memory allocated in armpl_spmat_export_csr_wrapper
          Call armpl_spmat_export_deallocate_wrapper(c_out_vals)
          Call armpl_spmat_export_deallocate_wrapper(c_out_col_indx)
          Call armpl_spmat_export_deallocate_wrapper(c_out_row_ptr)

      else

          Call armpl_spmat_export_csr(armpl_mat_c, 1, out_m, out_n, out_row_ptr, &
            out_col_indx, out_vals, info)

          Call check_result(armpl_mat_c, info)

          Write (*, 99997) 'Matrix C:'
          Write (*, 99997) 'Values:'
          Write (*, 99998) out_vals
          Write (*, 99997) 'Column Indices:'
          Write (*, 99999) out_col_indx
          Write (*, 99997) 'Row Pointer:'
          Write (*, 99999) out_row_ptr

!   Free memory allocated in armpl_spmat_export_csr
          Deallocate (out_vals)
          Deallocate (out_col_indx)
          Deallocate (out_row_ptr)

      end if

      Call armpl_spmat_destroy(armpl_mat_a, info)
      If (info/=armpl_status_success) Stop 'Error in armpl_spmat_destroy'

      Call armpl_spmat_destroy(armpl_mat_b, info)
      If (info/=armpl_status_success) Stop 'Error in armpl_spmat_destroy'

      Call armpl_spmat_destroy(armpl_mat_c, info)
      If (info/=armpl_status_success) Stop 'Error in armpl_spmat_destroy'

99999 Format (I3)
99998 Format (F10.1)
99997 Format (A)
    End Program


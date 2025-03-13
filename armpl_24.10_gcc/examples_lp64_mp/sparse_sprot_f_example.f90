!   Double precision plane rotation of sparse vectors example
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!
    Subroutine check_result(armpl_vec, info)
      Use armpl_library
      Integer (Kind=armpl_i8) armpl_vec
      Integer info
      If (info/=armpl_status_success) Then
        Call armpl_spvec_print_err(armpl_vec)
        Stop 'Exiting due to error'
      End If
    End Subroutine check_result

    Program sprot_f_example
      Use armpl_library
      Implicit None

!     1. Set-up local sparse and dense vectors
      Integer (Kind=armpl_i8) :: armpl_vec_x
      Integer, Parameter      :: n = 9
      Integer, Parameter      :: nnz = 4
      Integer                 :: index_base = 1
      Integer                 :: creation_flags = 0

      Real (Kind=armpl_r64), Dimension (nnz) :: vals = (/ 1., 2., 3., 4. /)
      Integer, Dimension (nnz)               :: indx = (/ 3, 6, 7, 9 /)
      Real (Kind=armpl_r64), Dimension (n)   :: y = (/ 1., 2., 3., 4., &
                                                       5., 6., 7., 8., 9. /)
      Real (Kind=armpl_r64), Parameter :: c = 0.
      Real (Kind=armpl_r64), Parameter :: s = 1.

      Integer                          :: info
      Integer                          :: i, cnt, dummy

      Real (Kind=armpl_r64)            :: v

      Write (*, 99999) 'Rotation matrix:'
      Write (*, 99998) c, s
      Write (*, 99998) -s, c
      Write (*, 99999) 'Sparse vector x:'
      Write (*, 99999, Advance='no') '['
      cnt = 1
      Do i = 1, n
          If (i==indx(cnt)) Then
              v = vals(cnt)
              cnt = cnt + 1
          Else
              v = 0.
          End If
          Write (*, 99997, Advance='no') v
      End Do
      Write (*, 99999) ']'
      Write (*, 99999) 'Dense vector y:'
      Write (*, 99999, Advance='no') '['
      Do i = 1, n
          Write (*, 99997, Advance='no') y(i)
      End Do
      Write (*, 99999) ']'

!     2. Set-up Arm Performance Libraries sparse vector object for x
      Call armpl_spvec_create(armpl_vec_x, index_base, n, nnz, indx, vals, creation_flags, info);
      Call check_result(armpl_vec_x, info)

!     3. Do SpRot
      Call armpl_sprot_exec(armpl_vec_x, y, c, s, info)
      Call check_result(armpl_vec_x, info)

!     4. Export results from Arm Performance Libraries sparse vector object
      Call armpl_spvec_export(armpl_vec_x, index_base, dummy, dummy, indx, vals, info);
      Call check_result(armpl_vec_x, info)

      Write (*, 99999) 'Rotated vector x:'
      Write (*, 99999, Advance='no') '['
      cnt = 1
      Do i = 1, n
          If (i==indx(cnt)) Then
              v = vals(cnt)
              cnt = cnt + 1
          Else
              v = 0.
          End If
          Write (*, 99997, Advance='no') v
      End Do
      Write (*, 99999) ']'
      Write (*, 99999) 'Rotated vector y:'
      Write (*, 99999, Advance='no') '['
      Do i = 1, n
          Write (*, 99997, Advance='no') y(i)
      End Do
      Write (*, 99999) ']'

!     5. Destroy created sparse vector object
      Call armpl_spvec_destroy(armpl_vec_x, info)
      If (info/=armpl_status_success) Stop 'Error in armpl_spvec_destroy'

99999 Format (A)
99998 Format (F4.1,1X,F4.1)
99997 Format (F4.1,1X)
    End Program

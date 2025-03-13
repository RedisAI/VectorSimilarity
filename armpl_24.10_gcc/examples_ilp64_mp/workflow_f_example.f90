!   Double precision program calling into different components of Arm PL
!   for showing a workflow featuring different library suites.
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates

module rng
   use armpl_kinds
   implicit none
   Integer(kind=armpl_i8), Parameter :: modulus = 2147483647 ! 2^31-1
   Integer(kind=armpl_i8), Parameter :: multiplier = 16807
   Integer(kind=armpl_i8), Parameter :: increment = 0
   Integer(kind=armpl_i8) :: seed = 666                      ! A default seed
contains
   subroutine init_rng(new_seed)
      Integer(kind=8), Intent(in) :: new_seed
      seed = new_seed
   end subroutine init_rng

   ! Generate a random number
   function rand() result(random_number)
      real :: random_number
      seed = mod(multiplier*seed + increment, modulus)
      random_number = real(seed)/real(modulus)
   end function rand
end module rng

Module helper_subroutines
  Use armpl_library
   Use rng
   implicit none
contains

   ! Geenerate a vector of real numbers
   Subroutine generate_real_vector(vec)
      Real(armpl_r64), intent(out) :: vec(:)
      Integer :: i
      Do i = 1, size(vec)
         vec(i) = rand()
      End Do
   End Subroutine generate_real_vector

   ! Generate a vector of complex numbers
   Subroutine generate_complex_vector(vec)
      Complex(kind=armpl_r64), intent(out) :: vec(:)
      Real(kind=armpl_r64) :: real_parts(size(vec)), imag_parts(size(vec))
      Integer :: i

      Do i = 1, size(vec)
         real_parts(i) = rand()
         imag_parts(i) = rand()
      End Do
      vec = cmplx(real_parts, imag_parts, kind=armpl_r64)
   End Subroutine generate_complex_vector

   Subroutine generate_csr_format(m, n, nnz, col_indx, row_ptr)
      Integer, Intent(in) :: m, n, nnz
      Integer, Dimension(nnz), Intent(out) :: col_indx
      Integer, Dimension(m + 1), Intent(out) :: row_ptr
      Integer :: i, k, row, col
      Real :: random_row, random_col
      Logical, Dimension(m, n) :: filled

      ! Initialize arrays
      col_indx = 0
      row_ptr = 0
      filled = .false.

      ! Assign nnz positions
      Do k = 1, nnz
         Do
            random_col = rand()
            random_row = rand()
            row = 1 + int(random_row*m)
            col = 1 + int(random_col*n)
            If (.not. filled(row, col)) Then
               filled(row, col) = .true.
               Exit
            End if
         End Do
         col_indx(k) = col
         row_ptr(row) = row_ptr(row) + 1
      End Do

      ! Convert row counts into starting indices
      Do i = 2, m + 1
         row_ptr(i) = row_ptr(i) + row_ptr(i - 1)
      End Do
      row_ptr(1) = 1
   End Subroutine generate_csr_format

   Subroutine check_result(armpl_mat, info)
      Use armpl_library
      Integer(Kind=armpl_i8) armpl_mat
      Integer info
      If (info /= armpl_status_success) Then
         Call armpl_spmat_print_err(armpl_mat)
         Stop 'Exiting due to error'
      End If
   End Subroutine check_result
End Module helper_subroutines

Program workflow_example
   Use armpl_library
   Use, Intrinsic :: iso_c_binding
   Use helper_subroutines
   Use rng
   Include 'fftw3.f03'

   !     .. Paramaters ..
   Integer, Parameter :: n = 1000, m = n, lda = n, ldb = n, nrhs = 1
   Integer, Parameter :: nnz = 400, incx = 1, incy = 1, flags = 0, lwork = 2*m
   Integer(kind=4), Parameter :: fft_size = n
   Real(Kind=armpl_r64), Parameter :: alpha = 1.0, beta = 1.0, tolerance = 1e-10
   Real(kind=armpl_r64), Parameter :: exact_result = 76.0048672889510328332
   !     .. Local variables ..
   real(kind=armpl_r64), Allocatable :: A(:), x(:), y(:), vals(:), work(:)
   Complex(kind=armpl_r64), Allocatable :: in(:)
   Integer, Allocatable :: col_indx(:), row_ptr(:)
   Real(kind=armpl_r64)                 :: result, error
   Type(c_ptr)                          :: plan
   Integer(Kind=armpl_i8)               :: armpl_mat

   !  Step 1. Perform complex to real FFT
   !  This step will generate a double real vector
   Allocate (in(n), x(n)) ! x = out
   Call generate_complex_vector(in)
   plan = fftw_plan_dft_c2r_1d(fft_size, in, x, FFTW_ESTIMATE)
   Call dfftw_execute_dft(plan, in, x)
   Call fftw_destroy_plan(plan)
   Deallocate (in)

   !  Step 2. SPMV
   !  Use the output from FFT as x in the SPMV
   !  Generate a sparse matrix and perform an SPMV

   Allocate (y(n), vals(nnz), col_indx(nnz), row_ptr(m + 1))
   Call generate_real_vector(y)
   Call generate_real_vector(vals)
   Call generate_csr_format(m, n, nnz, col_indx, row_ptr)
   Call armpl_spmat_create_csr(armpl_mat, m, n, row_ptr, col_indx, vals, &
                               flags, info)
   Call check_result(armpl_mat, info)

   Call armpl_spmat_hint(armpl_mat, armpl_sparse_hint_structure, &
                         armpl_sparse_structure_unstructured, info)
   Call check_result(armpl_mat, info)

   Call armpl_spmat_hint(armpl_mat, armpl_sparse_hint_spmv_operation, &
                         armpl_sparse_operation_notrans, info)
   Call check_result(armpl_mat, info)

   Call armpl_spmv_optimize(armpl_mat, info)
   Call check_result(armpl_mat, info)

   Call armpl_spmv_exec(armpl_sparse_operation_notrans, alpha, armpl_mat, &
                        x, beta, y, info)
   Call armpl_spmat_destroy(armpl_mat, info)
   Call check_result(armpl_mat, info)

   Deallocate (col_indx, row_ptr, vals)

   !   Step 3. GEMV.
   !   Use the output vector from FFT and SPMV in GEMV
   Allocate (A(lda*n))
   Call generate_real_vector(A)
   Call dgemv('N', m, n, alpha, A, lda, x, incx, beta, y, incy)

   !  Step 4. GELS
   !  Use the output from GEMV as a rhs in a least squares system
   Allocate (work(lwork))
   Call dgels('N', m, n, nrhs, A, lda, y, ldb, work, lwork, info)
   Deallocate (A, work)

   !  Step 5. cos from libamath
   !  Compute cos on the entries of the output of GELS
   Do i = 1, n
      y(i) = cos(y(i))
   End Do

   !  Step 6. dot product
   result = ddot(n, x, incx, y, incy)
   Deallocate (x, y)

   ! Validate result
   error = abs(result - exact_result)/abs(n*exact_result)
   if (error < tolerance) then
      print *, "TEST PASSED"
   else
      print *, "TEST FAILED"
   end if

End Program workflow_example

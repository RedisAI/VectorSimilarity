!   fftw_dft_r2c_1d: FFT of a real sequence
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!
    Program main
!     .. Use Statements ..
      Use armpl_library, Only: dcopy
      Use, Intrinsic                   :: iso_c_binding
      Include 'fftw3.f03'
!     .. Parameters ..
      Integer, Parameter               :: wp = kind(0.0D0)
      Integer, Parameter               :: nmax = 20
!     .. Local Scalars ..
      Integer                          :: j, n
      Type (c_ptr)                     :: forward_plan, inverse_plan
!     .. Local Arrays ..
!     The output vector is of size (n/2)+1 as it is Hermitian packed
      Complex (Kind=wp)                :: y(nmax/2+1)
      Real (Kind=wp)                   :: x(nmax), xx(nmax)
!     .. Executable Statements ..
!
      Write (*, 99998) 'ARMPL example: FFT of a real sequence using &
        &fftw_plan_dft_r2c_1d'
      Write (*, 99998) '--------------------------------------------&
        &--------------------'
      Write (*, *)
!
!     The sequence of real data
      n = 7
      x(1) = 0.34907E0_wp
      x(2) = 0.54890E0_wp
      x(3) = 0.74776E0_wp
      x(4) = 0.94459E0_wp
      x(5) = 1.13870E0_wp
      x(6) = 1.32870E0_wp
      x(7) = 1.51370E0_wp

!     Use dcopy to copy the values into another array (preserve input)
      Call dcopy(n, x, 1, xx, 1)

!     Initialise a plan for a real-to-complex 1d transform from x->y
      forward_plan = fftw_plan_dft_r2c_1d(n, x, y, fftw_estimate)
!     Initialise a plan for a complex-to-real 1d transform from y->x
      inverse_plan = fftw_plan_dft_c2r_1d(n, y, x, fftw_estimate)

!     Execute the forward plan and then deallocate the plan
!     NOTE: FFTW does NOT compute a normalised transform -
!     returned array will contain unscaled values
      Call fftw_execute_dft_r2c(forward_plan, x, y)
      Call fftw_destroy_plan(forward_plan)
!
      Write (*, 99998) 'Components of discrete Fourier transform:'
      Write (*, *)
      Do j = 1, n/2 + 1
!       Scale factor of 1/sqrt(n) to output normalised data
        Write (*, 99999) j, y(j)/sqrt(real(n))
      End Do
!
!     Execute the reverse plan and then deallocate the plan
!     NOTE: FFTW does NOT compute a normalised transform -
!     returned array will contain unscaled values
      Call fftw_execute_dft_c2r(inverse_plan, y, x)
      Call fftw_destroy_plan(inverse_plan)

      Write (*, *)
      Write (*, 99998) 'Original sequence as restored by inverse transform:'
      Write (*, *)
      Write (*, 99998) '       Original  Restored'
      Do j = 1, n
!       Scale factor of 1/n to output normalised data
        Write (*, 99997) j, xx(j), x(j)/n
      End Do
!
99999 Format (1X, I3, 3X, '(', F7.4, F7.4, ')')
99998 Format (A)
99997 Format (1X, I3, 2(:,3X,F7.4))
    End Program

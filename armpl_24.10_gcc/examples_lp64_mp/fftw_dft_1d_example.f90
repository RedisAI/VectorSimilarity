!   fftw_dft_1d: FFT of a complex sequence
!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!
    Program main
!     .. Use Statements ..
      Use armpl_library, Only: zcopy
      Use, Intrinsic                   :: iso_c_binding
      Include 'fftw3.f03'
!     .. Parameters ..
      Integer, Parameter               :: wp = kind(0.0D0)
      Integer, Parameter               :: nmax = 20
!     .. Local Scalars ..
      Integer                          :: i, n
      Type (c_ptr)                     :: forward_plan, reverse_plan
!     .. Local Arrays ..
      Complex (Kind=wp)                :: x(nmax), xx(nmax)
!     .. Executable Statements ..
!
      Write (*, Fmt=99999)                                                     &
        'ARMPL example: FFT of a complex sequence using fftw_plan_dft_1d'
      Write (*, Fmt=99999)                                                     &
        '---------------------------------------------------------------'
      Write (*, Fmt=*)
!
!     The sequence of complex data
      n = 7
      x(1) = (0.34907E0_wp, -0.37170E0_wp)
      x(2) = (0.54890E0_wp, -0.35669E0_wp)
      x(3) = (0.74776E0_wp, -0.31174E0_wp)
      x(4) = (0.94459E0_wp, -0.23702E0_wp)
      x(5) = (1.13870E0_wp, -0.13274E0_wp)
      x(6) = (1.32870E0_wp, 0.00074E0_wp)
      x(7) = (1.51370E0_wp, 0.16298E0_wp)

!     Use zcopy to copy the values into another array (preserve input)
      Call zcopy(n, x, 1, xx, 1)
!
!     Initialise a plan for a complex-to-complex 1d forward transform from x->x'
      forward_plan = fftw_plan_dft_1d(n, x, x, fftw_forward, fftw_estimate)
!     Initialise a plan for a complex-to-complex 1d backward transform from x'->x
      reverse_plan = fftw_plan_dft_1d(n, x, x, fftw_backward, fftw_estimate)

!     Execute the forward plan and then deallocate the plan
      Call fftw_execute_dft(forward_plan, x, x)
      Call fftw_destroy_plan(forward_plan)
!
      Write (*, Fmt=99999) 'Components of discrete Fourier transform:'
      Write (*, Fmt=*)
      Write (*, Fmt=99999) '         Real   Imag'
      Do i = 1, n
        Write (*, Fmt=99998) i, x(i)
      End Do

!     Execute the reverse plan and then deallocate the plan
!     NOTE: FFTW does NOT compute a normalised transform -
!     returned array will contain unscaled values
      Call fftw_execute_dft(reverse_plan, x, x)
      Call fftw_destroy_plan(reverse_plan)
!
      Write (*, Fmt=*)
      Write (*, Fmt=99999)                                                     &
        'Original sequence as restored by inverse transform:'
      Write (*, Fmt=*)
      Write (*, Fmt=99999) '          Original          Restored'
      Write (*, Fmt=99999) '         Real   Imag       Real   Imag'
      Do i = 1, n
!       Scale factor of 1/n to output normalised data
        Write (*, Fmt=99998) i, xx(i), x(i)/n
      End Do
!
99999 Format (A)
99998 Format (1X, I3, 2(:,3X,'(',F7.4,',',F7.4,')'))
    End Program

!=====================================================================

Subroutine PyGLTR(N, f, G, VECTOR, radius, stop_relative, stop_absolute, &
                   itmax, litmax, unitm, ST, boundary, equality,          &
                   fraction_opt , step, multiplier, snorm, niter, nc,     &
                   ierr, initial)

  Use GALAHAD_GLTR_double
  Implicit None

  Integer, Parameter :: wp = Kind(1.0D+0)
  Integer, Intent(In) :: N, itmax, litmax, unitm, ST, boundary, equality, initial
  Integer, Intent(Inout) :: niter, ierr
  Logical, Intent(Inout) :: nc
  Real(kind = wp), Intent(In) :: stop_relative, stop_absolute, fraction_opt
  Real(kind = wp), Intent(Inout) :: f, radius, multiplier, snorm
  Real(kind = wp), Dimension(N), Intent(InOut) :: G, VECTOR, step

  Integer, Parameter :: iout = 6
  Real(kind = wp), Save :: zero = 0.0_wp

  Type(GLTR_data_type), Save :: Data
  Type(GLTR_control_type), Save :: control
  Type(GLTR_info_type), Save :: info

  ! Initialization
  !  initial /= 0 in the initial call only
  If( initial /= 0 ) Then
     Call GLTR_initialize(Data, control) !, info)

     ! Non-default values come here
     control%out = 6
     ! If tolerances are negative, use default GLTR values
     If( stop_relative > zero ) control%stop_relative = stop_relative
     If( stop_absolute > zero ) control%stop_absolute = stop_absolute
     control%print_level = 0
     control%unitm = (unitm /= 0)
     control%steihaug_toint = (ST /= 0)
     control%Lanczos_itmax = litmax
     control%itmax = itmax
     control%fraction_opt = fraction_opt
     control%boundary = (boundary /= 0)
     control%equality_problem = (equality /= 0)
     info%status = 1
     !VECTOR = zero
!      Write(6, *) ' Upon initialization, GLTR has parameters'
!      Write(6, *) '  N = ', N
!      Write(6, *) '  radius = ', radius
!      Write(6, *) '  stop_rel = ', control%stop_relative
!      Write(6, *) '  stop_abs = ', control%stop_absolute
!      Write(6, *) '  itmax = ', control%itmax
!      Write(6, *) '  litmax = ', control%lanczos_itmax
!      Write(6, *) '  unitm = ', control%unitm
!      Write(6, *) '  ST = ', control%steihaug_toint
!      Write(6, *) '  boundary = ', control%boundary
!      Write(6, *) '  equality = ', control%equality_problem
!      Write(6, *) '  fraction = ', control%fraction_opt
     Return
  End If

  ! Iteration to find the minimizer
  Do

     Call GLTR_solve(N, radius, f, step, G, VECTOR, Data, control, info)
     ierr = info%status

     Select Case(info%status)
     Case(2, 6)
        ! Return for preconditioning step
        Return

     Case (3, 7)
        ! Return to form the matrix-vector product VECTOR <-- H . VECTOR
        Return

     Case (5)
        ! Restart with initial gradient
        Return

     Case (- 2 : 0)
        ! Successful return
        niter = info%iter + info%iter_pass2
        !Write(6, *) '   iter_pass1 = ', info%iter, ', iter_pass2 = ', info%iter_pass2
        multiplier = info%multiplier
        nc = info%negative_curvature
        snorm = info%mnormx
        Call GLTR_terminate(Data, control, info)
        Return

     Case DEFAULT
        ! Error return
        !Write(iout, "('Error :: GLTR_solve exit status = ', I6) ") &
        !     info%status
        niter = info%iter + info%iter_pass2
        multiplier = info%multiplier
        nc = info%negative_curvature
        snorm = info%mnormx
        Call GLTR_terminate(Data, control, info)
        Return

     End Select
  End Do

End Subroutine PyGLTR

!=====================================================================

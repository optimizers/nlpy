"""
An implementation of the projected Bi-CGSTAB algorithm
as described in

  D. Orban
  *Projected Krylov Methods for Unsymmetric Augmented Systems*,
  Cahiers du GERAD G-2008-46, GERAD, Montreal, Canada, 2008.

.. moduleauthor:: D. Orban <dominique.orban@gerad.ca>
"""

__docformat__ = 'restructuredtext'

import numpy
from projKrylov import ProjectedKrylov   # Abstract projected Krylov class
from nlpy.tools import norms
from math import sqrt
from nlpy.tools.timing import cputime

class ProjectedBCGSTAB( ProjectedKrylov ):

    def __init__( self, c, **kwargs ):
        """
        Solve saddle-point systems of the form

            [ H   A^T ] [ x ] = [ c ]                                        (1)
            [ A    0  ] [ y ]   [ b ]

        where H need not be symmetric. However, the method will work and be well
        defined when H is symmetric. For symmetric matrices H that arise as a
        second-derivative matrix as in optimization contexts, the projected
        conjugate gradient method may be more appropriate.
    
        Unless A is explicitly specified, we assume that there are no "equality
        constraints" A x = b. In this case, the method reduces to the well-known
        Bi-CGSTAB method described in reference [Van92]_ below.

        Similarly, unless b is explicitly specified, we assume that b = 0.
    
        The matrix H need not be given explicitly. Instead, a
        function named 'matvec' may be provided, with prototype

            v = matvec( q )

        where the return value v is the result of the matrix-vector
        product Hq between the matrix H and the supplied vector q.

        At least one of H and matvec must be supplied. If both are given,
        matvec is used.

        Note that the projected Bi-CGSTAB method does not require matrix-vector
        products with the transpose of H.

        A preconditioner may be given by using the precon keyword. For the time
        being, the preconditioner should be given in assembled form and may not
        be an operator. The preconditioner G is used to assemble and factorize
        the symmetric indefinite projection matrix

            [ G   A^T ]
            [ A    0  ].

        The user should keep in mind that G must be relatively sparse and must
        be **symmetric** and positive definite over the nullspace of A. If no
        preconditioner is given, everything happens as if G = I (the identity
        matrix) were given.

        The algorithm stops as soon as the norm of the projected residual
        falls under

            max( abstol, reltol * r0 )

        where

            abstol      is an absolute stopping tolerance
            reltol      is a relative stopping tolerance
            r0          is the norm of the initial projected residual
                        (or of the preconditioned projected residual, if a
                         preconditioner is given.)

        Optional keyword arguments are given in the following table.

        +----------+-------------------------------------------------+---------+
        | Keyword  | Description                                     | Default |
        +==========+=================================================+=========+
        | A        | the matrix defining linear equality constraints | None    |
        +----------+-------------------------------------------------+---------+
        | rhs      | a nonzero right-hand side of the constraints    | None    |
        +----------+-------------------------------------------------+---------+
        | H        | the explicit matrix H (with matvec method)      | None    |
        +----------+-------------------------------------------------+---------+
        | matvec   | a method to compute H-vector products           | None    |
        +----------+-------------------------------------------------+---------+
        | precon   | a preconditioner G (given explicitly)           | Identity|
        +----------+-------------------------------------------------+---------+
        | Proj     | an existing factorization of the projection     |         |
        |          | matrix conforming to the LBLContext             | None    |
        +----------+-------------------------------------------------+---------+
        | abstol   | the absolute stopping tolerance                 | 1.0e-8  |
        +----------+-------------------------------------------------+---------+
        | reltol   | the relative stopping tolerance                 | 1.0e-6  |
        +----------+-------------------------------------------------+---------+
        | MatvecMax| the maximum number of H-vector products         | 2n      |
        +----------+-------------------------------------------------+---------+
        | max_itref| the maximum number of iterative refinement steps| 3       |
        +----------+-------------------------------------------------+---------+
        | itref_tol| the required threshold for the residual after   |         |
        |          |    iterative refinement                         | 1.0e-6  |
        +----------+-------------------------------------------------+---------+
        | factorize| set to False if calling again with the same A   | True    |
        +----------+-------------------------------------------------+---------+
        | debug    | a boolean indicating debug/verbose mode         | False   |
        +----------+-------------------------------------------------+---------+


        References
        ----------

        .. [Orb08] D. Orban, *Projected Krylov Methods for Unsymmetric Augmented
                   Systems*, Cahiers du GERAD G-2008-46, GERAD, Montreal,
                   Canada, 2008.
        .. [Van92] H. A. Van der Vorst, *Bi-CGSTAB: A fast and smoothly
                   converging variant to Bi-CG for the solution of nonsymmetric
                   systems*, SIAM Journal on Scientific and Statistical
                   Computing, 13, pp. 631--644, 1992.
         """

        ProjectedKrylov.__init__(self, c, **kwargs)
        self.nMatvecMax = kwargs.get('MatvecMax', 2*self.n)
        self.prefix = 'Pbcgstab: '
        self.name = 'Projected Bi-CGSTAB'

        # Initializations
        self.x_feasible = None
        self.x = numpy.zeros(self.n)
        self.rhs = numpy.zeros(self.n + self.m)
        self.p = numpy.zeros(self.n)
        self.r = self.c.copy()
        self.s = numpy.zeros(self.n)
        self.Pp = None     # Will only be used as a 'pointer'
        self.Ps = None     # Will only be used as a 'pointer'
        self.Ap = numpy.empty(self.n)
        self.As = numpy.empty(self.n)
        self.y = numpy.empty(self.m)

        self.nMatvec = 0

        self.residNorm  = None
        self.residNorm0 = None

        # Formats for display
        self.hd_fmt = ' %-6s  %9s  %9s  %9s  %9s\n'
        self.header = self.hd_fmt % ('Matvec', 'resid', '<r,r0>', 'alpha', 'omega')
        self.fmt = ' %-6d  %9.2e  %9.2e  %9.2e  %9.2e\n'

    def Solve( self ):

        # Find feasible solution
        if self.A is not None:
            if self.factorize and not self.factorized: self.Factorize()
            if self.b is not None: self.FindFeasible()

        n = self.n
        m = self.m
        nMatvec = 0
        alpha = beta = omega = 0.0

        self.t_solve = cputime()

        # Obtain fixed vector r0 = projected initial residual
        # (initial x = 0 in homogeneous problem.)
        if self.A is not None:
            self.rhs[:n] = self.r
            self.rhs[n:] = 0.0
            self.Proj.solve( self.rhs )
            r0 = self.Proj.x[:n].copy()
            Btv = self.r - r0
        else:
            r0 = self.c

        # Initialize search direction
        self.p = self.r

        # Further initializations
        rr0 = rr00 = numpy.dot(self.r, r0)
        residNorm = self.residNorm0 = sqrt(rr0)
        stopTol = self.abstol + self.reltol * self.residNorm0
        finished = False

        if self.debug:
            self._write( self.header )
            self._write( '-' * len(self.header) + '\n' )

        if self.debug:
            self._write(self.fmt % (nMatvec, residNorm, rr0, alpha, omega))

        while not finished:

            # Project p
            self.rhs[:n] = self.p
            self.rhs[n:] = 0.0
            self.Proj.solve(self.rhs)
            self.Pp = self.Proj.x[:n]

            # Compute alpha and s
            if self._matvec_found:
                # Here we must copy Ap to prevent it from being overwritten
                # in the next matvec. We still need Ap when we update p below.
                self.Ap = self.matvec( self.Pp ).copy()
            else:
                self.H.matvec( self.Pp, self.Ap )
            nMatvec += 1

            alpha = rr0/numpy.dot(r0, self.Ap)
            self.s = self.r - alpha * self.Ap

            # Project s
            self.rhs[:n] = self.s - Btv              # Iterative semi-refinement
            self.rhs[n:] = 0.0
            self.Proj.solve(self.rhs)
            self.Ps = self.Proj.x[:n].copy()
            Btv = self.s - self.Ps

            residNorm = sqrt(numpy.dot(self.s, self.Ps))

            # Test for termination in the CGS process
            if residNorm <= stopTol or nMatvec > self.nMatvecMax:
                self.x += alpha * self.Pp
                if nMatvec > self.nMatvecMax:
                    reason = 'matvec'
                else:
                    reason = 's small'
                finished = True

            else:

                # Project  A*Ps
                if self._matvec_found:
                    self.As = self.matvec( self.Ps )
                else:
                    self.H.matvec( self.Ps, self.As )
                nMatvec += 1
                self.rhs[:n] = self.As
                self.rhs[n:] = 0.0
                self.Proj.solve(self.rhs)

                # Compute omega and update x
                sAs = numpy.dot(self.Ps, self.As)
                AsPAs = numpy.dot(self.As, self.Proj.x[:n])
                omega = sAs/AsPAs
                self.x += alpha * self.Pp + omega * self.Ps

                # Check for termination
                if nMatvec > self.nMatvecMax:
                    finished = True
                    reason = 'matvec'

                else:

                    # Update residual
                    self.r = self.s - omega * self.As
                    rr0_next = numpy.dot(self.r, r0)
                    beta = alpha/omega * rr0_next/rr0
                    rr0 = rr0_next
                    self.p -= omega * self.Ap
                    self.p *= beta
                    self.p += self.r

                    # Check for termination in the Bi-CGSTAB process
                    if abs(rr0) < 1.0e-12 * rr00:
                        self.rhs[:n] = self.r
                        self.rhs[n:] = 0.0
                        self.Proj.solve(self.rhs)
                        rPr = numpy.dot(self.r, self.Proj.x[:n])
                        if sqrt(rPr) <= stopTol:
                            finished = True
                            reason = 'r small'

            # Display current iteration info
            if self.debug:
                self._write(self.fmt % (nMatvec, residNorm, rr0, alpha, omega))

        # End while

        # Obtain final solution x
        if self.x_feasible is not None:
            self.x += self.x_feasible

        if self.A is not None:
            # Find (weighted) least-squares Lagrange multipliers from
            #   [ G  B^T ] [w]   [c - Hx]
            #   [ B   0  ] [v] = [  0   ]
            if self._matvec_found:
                self.rhs[:n] = -self.matvec( self.x )
            else:
                self.H.matvec( -self.x, self.rhs[:n] )
            self.rhs[:n] += self.c
            self.rhs[n:] = 0.0
            self.Proj.solve( self.rhs )
            self.v = self.Proj.x[n:].copy()

        self.t_solve = cputime() - self.t_solve
        self.converged = (nMatvec < self.nMatvecMax)
        self.nMatvec = nMatvec
        self.residNorm = residNorm
        self.status = reason

        return

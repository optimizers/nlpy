"""
An implementation of the projected conjugate gradient algorithm
as described in

  N.I.M. Gould, M.E. Hribar and J. Nocedal,
  *On the Solution of Equality Constrained Quadratic Programming
  Problems Arising in Optimization*,
  SIAM Journal on Scientific Computing **23** (4), pp. 1376-1395, 2001.

with the addition of an optional trust-region constraint.

.. moduleauthor:: D. Orban <dominique.orban@gerad.ca>
"""

__docformat__ = 'restructuredtext'

import numpy
from nlpy.krylov.projKrylov import ProjectedKrylov
from nlpy.tools import norms
from math import sqrt
from nlpy.tools.timing import cputime


class ProjectedCG( ProjectedKrylov ):

    def __init__( self, c, H, **kwargs ):
        """
        Solve the equality-constrained quadratic programming problem

            minimize     < c, x > + 1/2 < x, Hx >                           (1)
            subject to   A x = b

        using the projected preconditioned conjugate-gradient algorithm.
        Possibly, there may be no linear equality constraints in the problem.

        This module may also be used to solve the equality-constrained
        trust-region subproblem

            minimize     < c, x > + 1/2 < x, Hx >
            subject to   A x = 0                                            (2)
                         sqrt(< x, Mx >)  <=  radius,

        where M is a symmetric positive definite scaling matrix and radius > 0
        is a given trust-region radius. Note that b = 0 in the latter problem.
        For the time being, only M = I is implemented, i.e., the Euclidian norm
        is used. Specifying M is equivalent to specifying a preconditioner. See
        the keyword 'precon'.

        Equivalently, this module is appropriate for solving saddle-point
        systems of the form

            [ H   A^T ] [ x ] = [ -c ]                                      (3)
            [ A    0  ] [ y ]   [  b ]

        where H is a symmetric matrix. If H is positive definite on the
        nullspace of A, then (1) and (3) are equivalent. Otherwise, it is
        possible that no finite solution exists or that there are an infinite
        number of them. A 'trust-region radius' must then be
        specified. If any of the latter two cases apply, this module computes
        an initial x0 satisfying  A x0 = b, solves problem (2) with appropriate
        values of c and H and returns the final solution x+x0.

        Unless A is explicitly specified, we assume that there are no equality
        constraints. In this case, the method reduces to the well-known
        conjugate gradient method. Similarly, unless b is explicitly specified,
        we assume that b = 0.

        The symmetric matrix H need not be given explicitly. Instead, a
        function named 'matvec' may be provided, with prototype

            v = matvec( q )

        where the return value v is the result of the matrix-vector
        product Hq between the matrix H and the supplied vector q.

        At least one of H and matvec must be supplied. If both are given,
        matvec is used.

        A preconditioner may be given by using the precon keyword. For the time
        being, the preconditioner should be given in assembled form and may not
        be an operator. The preconditioner G is used to assemble and factorize
        the symmetric indefinite projection matrix

            [ G   A^T ]
            [ A    0  ].

        The user should keep in mind that G must be relatively sparse and must
        be positive definite over the nullspace of A. If no preconditioner is
        given, everything happens as if G = I (the identity matrix) were given.

        The algorithm stops as soon as the norm of the projected gradient
        falls under

            max( abstol, reltol * g0 )

        where

            abstol      is an absolute stopping tolerance
            reltol      is a relative stopping tolerance
            g0          is the norm of the initial projected gradient
                        (or of the preconditioned projected gradient, if a
                         preconditioner is given.)

        :keywords:
            :A: the matrix defining linear equality constraints (None)
            :rhs: a nonzero right-hand side of the constraints (None)
            :H: the explicit matrix H (with matvec method) (None)
            :matvec: a method to compute H-vector products (None)
            :precon: a preconditioner G (given explicitly) (Identity)
            :Proj: an existing factorization of the projection matrix
                   conforming to the LBLContext (None)
            :abstol: the absolute stopping tolerance (1.0e-8)
            :reltol: the relative stopping tolerance (1.0e-6)
            :maxiter: the maximum number of iterations (2n)
            :max_itref: the maximum number of iterative refinement steps (3)
            :itref_tol: the required threshold for the residual after
                        iterative refinement (1.0e-6)
            :radius: trust-region radius (None)
            :btol: fraction-to-the-boundary factor (None)
            :cur_iter: a vector related to btol (see below) (None)
            :factorize: set to `False` if calling again with the same
                        constraint matrix `A` (True)
            :debug: a boolean indicating debug/verbose mode (False)

        If specified, a positive factor `btol` will cause the algorithm to
        enforce all conjugate gradient iterates to satisfy  sk >= btol.
        Typically, in an interior-point methods calling `ppcg()`, the step
        computed at iteration k must be such that
                dk >= -tau xk
        where 0 < tau < 1, so that
                xk + dk >= (1-tau) xk,
        which prevents the iterates from getting too close to the boundary of
        the nonnegative orthant. In this type of setting, btol should be set
        to
                btol = tau
        and the vector cur_iter should be set to
                cur_iter = xk.
        This ensures that no copy of xk occurs and only a pointer to xk is
        used.

        Upon completion, a few members of the instance are set so a status
        check can be performed. The most important situations are:

        * A point was found where the residual is sufficiently small (whether
          no trust region was present, or its boundary was not encountered).
          This can only happen when `H` is second-order sufficient.
          In this case `onBoundary` is set to `False` and `infDescent` is set
          to `False`.
        * No trust region is present but the problem is not second-order
          sufficient. In this case, an infinite descent direction has been
          identified: `infDescent` is set to `True` and `dir` contains the
          infinite descent direction. `onBoundary` is set to `False`.
        * A trust-region is present and its boundary was hit. If no infinite
          descent direction has been discovered, `infDescent` is set to
          `False`. Otherwise, it is set to `True`. In both cases, `onBoundary`
          is set to `True`.

        Reference
        ---------

        .. [GHN01]  N.I.M. Gould, M.E. Hribar and J. Nocedal, *On the Solution
                    of Equality Constrained Quadratic Programming Problems
                    Arising in Optimization*, SIAM Journal on Scientific
                    Computing **23**(4), pp. 1376-1395, 2001.
        """

        ProjectedKrylov.__init__(self, c, H, **kwargs)

        self.prefix = 'Ppcg: '
        self.name = 'Projected CG'
        self.radius = kwargs.get( 'radius', None )
        #if self.radius is not None and self.b is not None:
        #    raise ValueError, 'Either radius or rhs may be given but not both'

        if self.radius is not None and self.radius <= 0.0:
            raise ValueError, 'Radius must be a positive real number'

        self.btol = kwargs.get( 'btol', None )
        self.cur_iter = kwargs.get( 'cur_iter', None )
        self.precon = kwargs.get('precon', None)

        # Initializations
        self.x_feasible = None
        self.x = numpy.zeros( self.n, 'd' )
        self.step = self.x  # Shortcut for consistency with TruncatedCG
        self.v = None
        self.residNorm  = None
        self.residNorm0 = None
        self.rhs = numpy.zeros( self.n + self.m, 'd' )
        self.iter = self.nMatvec = 0
        self.infiniteDescentDir = None
        self.xNorm2 = 0.0        # Square norm of step, not counting x_feasible
        self.stepNorm = 0.0 # Shortcut for consistency with TruncatedCG
        self.dir = None  # Direction of infinity descent
        self.onBoundary = False
        self.infDescent = False

        # Formats for display
        self.hd_fmt = ' %-5s  %9s  %8s\n'
        self.header = self.hd_fmt % ('Iter', '<r,g>', 'curv')
        self.fmt1 = ' %-5d  %9.2e\n'
        self.fmt = ' %-5d  %9.2e  %8.2e\n'


    def to_boundary(self, s, p, Delta, ss=None):
        """
        Given vectors `s` and `p` and a trust-region radius `Delta` > 0,
        return the positive scalar sigma such that

            || s + sigma * p || = Delta

        in Euclidian norm. If known, supply optional argument `ss` whose value
        should be the squared Euclidian norm of argument `s`.
        """
        if Delta is None:
            raise ValueError, 'Radius value must be positive number.'
        sp = numpy.dot(s,p)
        pp = numpy.dot(p,p)
        if ss is None: ss = numpy.dot(s,s)
        sigma = (-sp + sqrt( sp*sp + pp * ( Delta*Delta - ss ) ) )
        sigma /= pp

        if (self.btol is not None) and (self.cur_iter is not None):
            sigma = min(sigma, self.ftb(s,p))

        return sigma


    def ftb(self, s, p):
        """
        If fraction-to-the-boundary rule is to be enforced, compute step
        length to satisfy  s + t*p >= btol * cur_iter.
        """
        neg_idx = numpy.where(p < 0.0)[0]
        stepLen = 1.0
        for i in neg_idx:
            stepLen = min(stepLen, -(self.btol * self.cur_iter[i] + s[i])/p[i])
        return stepLen


    def Solve(self):
        if self.A is not None:
            if self.factorize and not self.factorized: self.Factorize()
            if self.b is not None: self.FindFeasible()

        n = self.n
        m = self.m
        xNorm2 = 0.0   # Squared norm of current iterate x, not counting x_feas

        # Obtain initial projected residual
        self.t_solve = cputime()
        if self.A is not None:
            if self.b is not None:
                self.rhs[:n] = self.c + self.H * self.x_feasible
                self.rhs[n:] = 0.0
            else:
                self.rhs[:n] = self.c
            self.Proj.solve( self.rhs )
            r = g = self.Proj.x[:n]
            self.v = self.Proj.x[n:]

            #self.CheckAccurate()

        else:
            g = self.c
            r = g.copy()

        # Initialize search direction
        p = -g
        pHp = None

        self.residNorm0 = numpy.dot(r,g)
        rg  = self.residNorm0
        threshold = max( self.abstol, self.reltol * sqrt(self.residNorm0) )
        iter = 0
        onBoundary = False

        if self.debug:
            self._write( self.header )
            self._write( '-' * len(self.header) + '\n' )
            self._write( self.fmt1 % (iter, rg) )

        while sqrt(rg) > threshold and iter < self.maxiter and not onBoundary:

            Hp = self.H * p
            pHp = numpy.dot(p,Hp)

            # Display current iteration info
            if self.debug: self._write( self.fmt % (iter, rg, pHp) )

            if self.radius is not None:
                # Compute steplength to the boundary
                sigma = self.to_boundary(self.x, p, self.radius, ss=xNorm2)
            elif pHp <= 0.0:
                self._write('Problem is not second-order sufficient\n')
                status = 'problem not SOS'
                self.infDescent = True
                self.dir = p
                continue

            alpha = rg/pHp

            if self.radius is not None and (pHp <= 0.0 or alpha > sigma):
                # p is a direction of singularity or negative curvature or
                # next iterate will lie past the boundary of the trust region
                # Move to boundary of trust-region
                self.x += sigma * p
                xNorm2 = self.radius * self.radius
                status = 'on boundary (sigma = %g)' % sigma
                self.infDescent = True
                onBoundary = True
                continue

            # Make sure nonnegativity bounds remain enforced, if requested
            if (self.btol is not None) and (self.cur_iter is not None):
                stepBnd = self.ftb(self.x, p)
                if stepBnd < alpha:
                    self.x += stepBnd * p
                    status = 'on boundary'
                    onBoundary = True
                    continue

            # Move on
            self.x += alpha * p
            r += alpha * Hp

            if self.A is not None:
                # Project current residual
                self.rhs[:n] = r
                self.Proj.solve( self.rhs )

                # Perform actual iterative refinement, if necessary
                #self.Proj.refine( self.rhs, nitref=self.max_itref,
                #                  tol=self.itref_tol )

                # Obtain new projected gradient
                g = self.Proj.x[:n]
                if self.precon is not None:
                    # Prepare for iterative semi-refinement
                    self.A.matvec_transp( self.Proj.x[n:], self.v )
            else:
                g = r

            rg_next = numpy.dot(r,g)
            beta = rg_next/rg
            p = -g + beta * p
            if self.precon is not None:
                # Perform iterative semi-refinement
                r = r - self.v
            else:
                r = g
            rg = rg_next

            if self.radius is not None:
                xNorm2 = numpy.dot( self.x, self.x )
            iter += 1

        # Output info about the last iteration
        if self.debug and iter > 0:
            self._write( self.fmt % (iter, rg, pHp) )

        # Obtain final solution x
        self.xNorm2 = xNorm2
        self.stepNorm = sqrt(xNorm2)
        if self.x_feasible is not None:
            self.x += self.x_feasible

        if self.A is not None:
            # Find (weighted) least-squares Lagrange multipliers
            self.rhs[:n] = - self.c - self.H * self.x
            self.rhs[n:] = 0.0
            self.Proj.solve( self.rhs )
            self.v = self.Proj.x[n:].copy()

        self.t_solve = cputime() - self.t_solve

        self.step = self.x  # Alias for consistency with TruncatedCG.
        self.onBoundary = onBoundary
        self.converged = (iter < self.maxiter)
        if iter < self.maxiter and not onBoundary:
            status = 'residual small'
        elif iter >= self.maxiter:
            status = 'max iter'
        self.iter = iter
        self.nMatvec = iter
        self.residNorm = sqrt(rg)
        self.status = status

        return

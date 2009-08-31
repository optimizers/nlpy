# Pygltr : A Python interface to the GALAHAD package GLTR
# GLTR solves a trust-region subproblem using a preconditioned
# Lanczos iteration.

"""
Solution of a trust-region subproblem using the preconditioned Lanczos method.
"""

from pysparse import spmatrix
from nlpy.krylov import _pygltr
import numpy
import sys

class PyGltrContext:

    def __init__(self, g, **kwargs):
        """
        Create a new instance of a PyGltrContext object, representing a
        context to solve the quadratic problem

            min <g, d> + 1/2 <d, Hd>
            s.t ||d|| <= radius

        where either the Hessian matrix H or a means to compute
        matrix-vector products with H, are to be specified later.

        Arguments of initialization are

         g          the gradient vector
         radius     the trust-region radius (default: 1.0)
         reltol     the relative stopping tolerance (default: sqrt(eps))
         abstol     the absolute stopping tolerance (default: 0.0)
         prec       a function solving preconditioning systems.
                    If M is a preconditioner, prec(v) returns a solution
                    to the linear system of equations Mx = v (default: None)
         itmax      the maximum number of iterations (default: n)
         litmax     the maximum number of Lanczos iterations on the boundary
                    (default: n)
         ST         Use Steihaug-Toint strategy (default: False)
         boundary   Indicates whether the solution is thought to lie on
                    the boundary of the trust region (default: False)
         equality   Require that the solution lie on the boundary (default: False)
         fraction   Fraction of optimality that is acceptable. A value smaller
                    that 1.0 results in a correspondingly sub-optimal solution.
                    (default: 1.0)
         
        See the GLTR spec sheet for more information on these parameters.

        Convergence of the iteration takes place as soon as

         Norm(Hd + l Md + g) <= max(Norm(g) * reltol, abstol)

        where M     is a preconditioner
              l     is an estimate of the Lagrange multipliers
              Norm  is the M^{-1}-norm
        """
        self.n = g.shape[0]

        # Process optional parameters
        self.radius = kwargs.get('radius', 1.0)
        self.reltol = kwargs.get('reltol', -1.0)
        self.abstol = kwargs.get('abstol', -1.0)
        self.prec   = kwargs.get('prec', None)
        self.itmax  = kwargs.get('itmax', -1)
        self.litmax = kwargs.get('litmax', -1)
        self.unitm  = (self.prec == None)
        self.ST       = kwargs.get('ST', False)
        self.boundary = kwargs.get('boundary', False)
        self.equality = kwargs.get('equality', False)
        self.fraction = kwargs.get('fraction', 1.0)
        
        self.debug = False
        self.step   = numpy.zeros(self.n, 'd')
        self.vector = numpy.zeros(self.n, 'd')
        self.context = _pygltr.gltr(g, self.step, self.vector,
                                     self.radius, self.reltol, self.abstol,
                                     self.itmax, self.litmax,
                                     self.unitm, self.ST, self.boundary,
                                     self.equality, self.fraction)

        # Return values
        self.m     = 0.0     # Final model value
        self.mult  = 0.0     # Final Lagrange multiplier
        self.snorm = 0.0     # Preconditioned step norm
        self.niter = 0       # Number of iterations
        self.nc    = False   # Whether negative curvature was encountered
        self.ierr  = 0       # Return error code

    def explicit_solve(self, H):
        """
        Solves the quadratic trust-region subproblem whose data was
        specified upon initialization. During the reverse communication
        phase, matrix vector products with the Hessian H will be
        computed explicitly using the matvec method of the object H.
        For instance, if H is an ll_mat, or csr_mat, products will be
        evaluated using H.matvec(x,y).
        """

        # Check that H has right dimension
        (nH, mH) = H.shape;
        if nH != mH or nH != self.n:
            return None

        done = False
        tmp = numpy.empty(self.n, 'd')

        if self.debug:
            sys.stderr.write(' PyGltr.explicit_solve() called with data\n')
            sys.stderr.write('   radius  reltol  abstol  itmax  litmax')
            sys.stderr.write('  unitm  ST  boundary  equality  fraction\n')
            sys.stderr.write('   %-6g  %-6g  %-6g  %-5d  %-6d' % (self.radius, self.reltol, self.abstol, self.itmax, self.litmax))
            sys.stderr.write('  %-5d  %-2d  %-8d  %-8d  %-8g\n' % (self.unitm, self.ST, self.boundary, self.equality, self.fraction))
            sys.stderr.write(' ..entering gltr [')

        # Main loop
        while not done:
            # Step and Vector are updated at C level
            (m,mult,snorm,niter,nc,ierr) = self.context.solve(self.step,
                                                              self.vector)

            # The case ierr == 5 is treated at C level; we only treat the others
            if ierr == 2 or ierr == 6:
                # Preconditioning step
                if self.prec is not None:
                    tmp = self.prec(self.vector)
                    self.context.reassign(tmp)
        
            elif ierr == 3 or ierr == 7:
                # Form product H * vector and reassign vector to the result
                H.matvec(self.vector, tmp)
                self.context.reassign(tmp)

            elif -2 <= ierr and ierr <= 0:
                # Successful return
                done = True

            else:
                # The problem was not solved to the desired accuracy, either
                # because the trust-region boundary was hit while ST = True,
                # or because the maximum number of iterations was reached.
                done = True

        if self.debug: sys.stderr.write(']\n')

        self.m = m
        self.mult = mult
        self.snorm = snorm
        self.niter = niter
        self.nc = nc
        self.ierr = ierr
        return

    def implicit_solve(self, hessprod):
        """
        Solves the quadratic trust-region subproblem whose data was
        specified upon initialization. During the reverse communication
        phase, matrix vector products with the Hessian H will be
        computed implicitly using the supplied hessprod method.
        Given an array v, hessprod must return an array of the same size
        containing the result of the multiplication H*v.
        For instance, if the problem is from an Ampl model called nlp,
        the hessprod method could be
        
            lambda v: nlp.hprod(z, v)

        for some multiplier estimates z.
        """

        done = False

        if self.debug:
            sys.stderr.write(' PyGltr.implicit_solve() called with data\n')
            sys.stderr.write('   radius  reltol  abstol  itmax  litmax')
            sys.stderr.write('  unitm  ST  boundary  equality  fraction\n')
            sys.stderr.write('   %-6g  %-6g  %-6g  %-5d  %-6d' % (self.radius, self.reltol, self.abstol, self.itmax, self.litmax))
            sys.stderr.write('  %-5d  %-2d  %-8d  %-8d  %-8g\n' % (self.unitm, self.ST, self.boundary, self.equality, self.fraction))
            sys.stderr.write(' ...entering gltr2 [')

        # Main loop
        while not done:
            (m, mult, snorm, niter, nc, ierr) = self.context.solve(self.step, self.vector)

            # The case ierr == 5 is treated at C level. We only treat the others
            if ierr == 2 or ierr == 6:
                # Preconditioning step
                if self.prec is not None:
                    tmp = self.prec(self.vector)
                    self.context.reassign(tmp)
        
            elif ierr == 3 or ierr == 7:
                # Form product vector = H * vector
                tmp = hessprod(self.vector)
                # Reassign vector to the result
                self.context.reassign(tmp)

            elif -2 <= ierr and ierr <= 0:
                # Successful return
                done = True

            else:
                # The problem was not solved to the desired accuracy, either
                # because the trust-region boundary was hit while ST = True,
                # or because the maximum number of iterations was reached.
                done = True

        if self.debug: sys.stderr.write(' ierr = %d ]\n' % ierr)

        self.m = m
        self.mult = mult
        self.snorm = snorm
        self.niter = niter
        self.nc = nc
        self.ierr = ierr
        return

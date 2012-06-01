# Pygltr : A Python interface to the GALAHAD package GLTR
# GLTR solves a trust-region subproblem using a preconditioned
# Lanczos iteration.

"""
Solution of a trust-region subproblem using the preconditioned Lanczos method.
"""

from pysparse.sparse import spmatrix
from nlpy.krylov import _pygltr
import numpy as np
import logging
import sys

import warnings

__docformat__='restructuredtext'


class PyGltrContext(object):
    """
    Create a new instance of a PyGltrContext object, representing a
    context to solve the quadratic problem

        min g'd + 1/2 d'Hd
        s.t d'd <= radius

    where either the Hessian matrix H or a means to compute
    matrix-vector products with H, are to be specified later.

    :parameters:

      :qp:         instance of :class:`QPModel`


    :keywords:

      :radius:      trust-region radius (default: 1.0)
      :reltol:      relative stopping tolerance (default: sqrt(eps))
      :abstol:      absolute stopping tolerance (default: 0.0)
      :prec:        callable for solving preconditioning systems.
                    If M is a preconditioner, `prec(v)` returns a solution
                    to the linear system of equations Mx = v
                    (default: `None`)
      :itmax:       maximum number of iterations (default: `n`)
      :litmax:      maximum number of Lanczos iterations on the boundary
                    (default: `n`)
      :ST:          Use Steihaug-Toint strategy (default: `False`)
      :boundary:    Indicates whether the solution is thought to lie on
                    the boundary of the trust region (default: `False`)
      :equality:    Require that the solution lie on the boundary
                    (default: `False`)
      :fraction:    Fraction of optimality that is acceptable. A value
                    smaller than `1.0` results in a correspondingly
                    sub-optimal solution. (default: `1.0`)
      :logger_name: name of a logger (default: `None`)

    See the GLTR spec sheet for more information on these parameters.

    Convergence of the iteration takes place as soon as

     Norm(Hd + l Md + g) <= max(Norm(g) * reltol, abstol)

    where M       is a preconditioner
          l       is an estimate of the Lagrange multipliers
          Norm()  is the M^{-1}-norm
    """

    def __init__(self, qp, **kwargs):

        self.qp = qp
        self.n = qp.c.shape[0]

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
        logger_name = kwargs.get('logger_name', 'nlpy.pygltr')
        self.log = logging.getLogger(logger_name)
        self.log.propagate = False

        self.step   = np.zeros(self.n, 'd')
        self.vector = np.zeros(self.n, 'd')
        self.context = _pygltr.gltr(qp.c, self.step, self.vector,
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

    def explicit_solve(self):
        """
        Solves the quadratic trust-region subproblem whose data was
        specified upon initialization. During the reverse communication
        phase, matrix vector products with the Hessian H will be
        computed explicitly using the matvec method of the object H.
        For instance, if H is an ll_mat, or csr_mat, products will be
        evaluated using H.matvec(x,y).
        """

        warnings.warn("deprecated---use `implicit_solve()`", DeprecationWarning)

        # Check that H has right dimension
        qp = self.qp
        H = qp.hess(qp.x0)
        (nH, mH) = H.shape;
        if nH != mH or nH != self.n:
            raise ValueError, 'Dimension mismatch'

        done = False
        tmp = np.empty(self.n)

        self.log.debug(' PyGltr.explicit_solve() called with data')
        self.log.debug('   radius  reltol  abstol  itmax  litmax')
        self.log.debug('   %-6g  %-6g  %-6g  %-5d  %-6d' % (self.radius, self.reltol, self.abstol, self.itmax, self.litmax))
        self.log.debug('   unitm  ST  boundary  equality  fraction')
        self.log.debug('   %-5d  %-2d  %-8d  %-8d  %-8g' % (self.unitm, self.ST, self.boundary, self.equality, self.fraction))
        self.log.debug(' ...entering gltr [')

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
                #H.matvec(self.vector, tmp)
                self.context.reassign(H * self.vector)

            elif -2 <= ierr and ierr <= 0:
                # Successful return
                done = True

            else:
                # The problem was not solved to the desired accuracy, either
                # because the trust-region boundary was hit while ST = True,
                # or because the maximum number of iterations was reached.
                done = True

        self.log.debug(']')

        self.m = m
        self.mult = mult
        self.snorm = snorm
        self.niter = niter
        self.nc = nc
        self.ierr = ierr
        return

    def implicit_solve(self):
        """
        Solves the quadratic trust-region subproblem whose data was
        specified upon initialization. During the reverse communication
        phase, matrix vector products with the Hessian H will be
        computed implicitly using the `hprod` method of the `qp` object given
        upon initialization.
        Given an array `v`, `hprod` must return an array of the same size
        containing the result of the multiplication `H*v`.
        For instance, if the problem is from an Ampl model called nlp,
        the hessprod method could be

            lambda v: nlp.hprod(z, v)

        for some multiplier estimates z.
        """

        done = False
        qp = self.qp
        x  = qp.x0
        y  = qp.pi0

        self.log.debug(' PyGltr.implicit_solve() called with data')
        self.log.debug('   radius  reltol  abstol  itmax  litmax')
        self.log.debug('   %-6g  %-6g  %-6g  %-5d  %-6d' % (self.radius, self.reltol, self.abstol, self.itmax, self.litmax))
        self.log.debug('   unitm  ST  boundary  equality  fraction')
        self.log.debug('   %-5d  %-2d  %-8d  %-8d  %-8g' % (self.unitm, self.ST, self.boundary, self.equality, self.fraction))
        self.log.debug(' ...entering gltr [')

        # Main loop
        while not done:
            (m, mult, snorm, niter, nc, ierr) = self.context.solve(self.step,
                                                                   self.vector)

            # The case ierr == 5 is treated at C level. We only treat the others
            if ierr == 2 or ierr == 6:
                # Preconditioning step
                if self.prec is not None:
                    tmp = self.prec(self.vector)
                    self.context.reassign(tmp)

            elif ierr == 3 or ierr == 7:
                # Form product vector = H * vector
                #tmp = hessprod(self.vector)
                # Reassign vector to the result
                self.context.reassign(qp.hprod(x, y, self.vector))

            elif -2 <= ierr and ierr <= 0:
                # Successful return
                done = True

            else:
                # The problem was not solved to the desired accuracy, either
                # because the trust-region boundary was hit while ST = True,
                # or because the maximum number of iterations was reached.
                done = True

        self.log.debug(' ierr = %d ]' % ierr)

        self.m = m
        self.mult = mult
        self.snorm = snorm
        self.niter = niter
        self.nc = nc
        self.ierr = ierr
        return

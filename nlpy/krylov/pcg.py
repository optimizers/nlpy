"""
A pure Python/numpy implementation of the Steihaug-Toint
truncated preconditioned conjugate gradient algorithm as described in

  T. Steihaug, *The conjugate gradient method and trust regions in large scale
  optimization*, SIAM Journal on Numerical Analysis **20** (3), pp. 626-637,
  1983.

.. moduleauthor:: D. Orban <dominique.orban@gerad.ca>
"""

from nlpy.optimize.solvers.lbfgs import InverseLBFGS
from nlpy.tools.exceptions import UserExitRequest
import numpy as np
from math import sqrt
import logging

__docformat__ = 'restructuredtext'


class TruncatedCG(object):

    def __init__(self, qp, **kwargs):
        """
        Solve the quadratic trust-region subproblem

          minimize    g's + 1/2 s'Hs
          subject to  s's  <=  radius

        by means of the truncated conjugate gradient algorithm (aka the
        Steihaug-Toint algorithm). The notation `x'y` denotes the dot
        product of vectors `x` and `y`.

        :parameters:
            :qp:           an instance of the :class:`QPModel` class.
                           The Hessian H must be a symmetric linear
                           operator of appropriate size, but not necessarily
                           positive definite.
            :logger_name:  name of a logger object that can be used during the
                           iterations                         (default None)

        :returns:

          Upon return, the following attributes are set:

          :step:       final step,
          :niter:      number of iterations,
          :stepNorm:   Euclidian norm of the step,
          :dir:        direction of infinite descent (if radius=None and
                       H is not positive definite),
          :onBoundary: set to True if trust-region boundary was hit,
          :infDescent: set to True if a direction of infinite descent was found

        The algorithm stops as soon as the preconditioned norm of the gradient
        falls under

            max( abstol, reltol * g0 )

        where g0 is the preconditioned norm of the initial gradient (or the
        Euclidian norm if no preconditioner is given), or as soon as the
        iterates cross the boundary of the trust region.
        """

        self.qp = qp
        self.n = qp.c.shape[0]

        self.prefix = 'Pcg: '
        self.name = 'Truncated CG'

        self.status = '?'
        self.onBoundary = False
        self.step = None
        self.stepNorm = 0.0
        self.niter = 0
        self.dir = None

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlpy.trunk')
        self.log = logging.getLogger(logger_name)
        self.log.propagate=False

        # Formats for display
        self.hd_fmt = ' %-5s  %9s  %8s\n'
        self.header = self.hd_fmt % ('Iter', '<r,g>', 'curv')
        self.fmt = ' %-5d  %9.2e  %8.2e\n'

        return


    def to_boundary(self, s, p, radius, ss=None):
        """
        Given vectors `s` and `p` and a trust-region radius `radius` > 0,
        return the positive scalar `sigma` such that

          `|| s + sigma * p || = radius`

        in Euclidian norm. If known, supply optional argument `ss` whose value
        should be the squared Euclidian norm of `s`.
        """
        if radius is None:
            raise ValueError, 'Input value radius must be positive number.'
        sp = np.dot(s,p)
        pp = np.dot(p,p)
        if ss is None: ss = np.dot(s,s)
        sigma = (-sp + sqrt(sp*sp + pp * (radius*radius - ss)))
        sigma /= pp
        return sigma

    def post_iteration(self, *args, **kwargs):
        """
        Subclass and override this method to implement custom post-iteration
        actions. This method will be called at the end of each CG iteration.
        """
        pass

    def Solve(self, **kwargs):
        """
        Solve the trust-region subproblem.

        :keywords:

          :s0:         initial guess (default: [0,0,...,0]),
          :radius:     the trust-region radius (default: None),
          :abstol:     absolute stopping tolerance (default: 1.0e-8),
          :reltol:     relative stopping tolerance (default: 1.0e-6),
          :maxiter:    maximum number of iterations (default: 2n),
          :prec:       a user-defined preconditioner.
        """

        radius  = kwargs.get('radius', None)
        abstol  = kwargs.get('absol', 1.0e-8)
        reltol  = kwargs.get('reltol', 1.0e-6)
        maxiter = kwargs.get('maxiter', 2*self.n)
        prec    = kwargs.get('prec', lambda v: v)
        debug   = kwargs.get('debug', False)

        n = self.n
        g = self.qp.c
        H = self.qp.H

        # Initialization
        r = g.copy()
        if 's0' in kwargs:
            s = kwargs['s0']
            snorm2 = np.linalg.norm(s)
            r += H*s                 # r = g + H s0
        else:
            s = np.zeros(n)
            snorm2 = 0.0

        y = prec(r)
        ry = np.dot(r, y)

        exitOptimal = exitIter = exitUser = False

        try:
            sqrtry = sqrt(ry)
        except:
            msg = 'Preconditioned residual = %8.1e\n' % ry
            msg += 'Is preconditioner positive definite?'
            raise ValueError, msg

        stopTol = max(abstol, reltol * sqrtry)

        # Initialize r as a copy of g not to alter the original g
        p = -y                       # p = - preconditioned residual
        k = 0

        onBoundary = False
        infDescent = False

        self.log.info(self.header)
        self.log.info('-' * len(self.header) + '\n')

        while not (exitOptimal or exitIter or exitUser) and \
                not onBoundary and not infDescent:

            k += 1
            Hp  = H * p
            pHp = np.dot(p, Hp)

            self.log.info(self.fmt % (k, ry, pHp))

            # Compute steplength to the boundary.
            if radius is not None:
                sigma = self.to_boundary(s, p, radius, ss=snorm2)

            if pHp <= 0 and radius is None:
                # p is direction of singularity or negative curvature.
                self.status = 'infinite descent'
                snorm2 = 0
                self.dir = p
                infDescent = True
                continue

            # Compute CG steplength.
            alpha = ry/pHp if pHp != 0 else 1

            if radius is not None and (pHp <= 0 or alpha > sigma):
                # p leads past the trust-region boundary. Move to the boundary.
                s += sigma * p
                snorm2 = radius*radius
                #self.status = 'on boundary (sigma = %g)' % sigma
                self.status = 'trust-region boundary active'
                onBoundary = True
                continue

            self.ds = alpha * p
            self.dr = alpha * Hp

            # Move to next iterate.
            s += self.ds
            r += self.dr
            y = prec(r)
            ry_next = np.dot(r, y)
            beta = ry_next/ry
            p = -y + beta * p
            ry = ry_next

            # Transfer useful quantities for post iteration.
            self.pHp = pHp
            self.r = r
            self.y = y
            self.p = p
            self.step = s
            self.stepNorm2 = snorm2
            self.ry = ry
            self.alpha = alpha
            self.beta = beta

            try:
                sqrtry = sqrt(ry)
            except:
                msg = 'Preconditioned residual = %8.1e\n' % ry
                msg += 'Is preconditioner positive definite?'
                raise ValueError, msg

            snorm2 = np.dot(s,s)

            try:
                self.post_iteration()
            except UserExitRequest:
                self.status = 'usr'

            exitUser    = self.status == 'usr'
            exitIter    = k >= maxiter
            exitOptimal = sqrtry <= stopTol

        # Output info about the last iteration.
        self.log.info(self.fmt % (k, ry, pHp))

        if k < maxiter and not onBoundary:
            self.status = 'residual small'
        elif k >= maxiter:
            self.status = 'max iter'
        self.step = s
        self.niter = k
        self.stepNorm = sqrt(snorm2)
        self.onBoundary = onBoundary
        self.infDescent = infDescent
        return


class TruncatedCGLBFGS(TruncatedCG):

    def __init__(self, g, H, **kwargs):
        super(TruncatedCGLBFGS, self).__init__(g, H, **kwargs)
        npairs = kwargs.get('npairs', 5)
        scaling = kwargs.get('scaling', True)
        self.lbfgs = InverseLBFGS(self.n,
                                  npairs=npairs,
                                  scaling=scaling)

    def post_iteration(self):
        self.lbfgs.store(self.ds, self.dr)


def model_value(H, g, s):
    # Return <g,s> + 1/2 <s,Hs>
    return np.dot(g,s) + 0.5 * np.dot(s, H*s)

def model_grad(H, g, s):
    # Return g + Hs
    return g + H*s

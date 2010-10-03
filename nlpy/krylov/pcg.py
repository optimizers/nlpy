"""
A pure Python/numpy implementation of the Steihaug-Toint
truncated preconditioned conjugate gradient algorithm as described in

  T. Steihaug, *The conjugate gradient method and trust regions in large scale
  optimization*, SIAM Journal on Numerical Analysis **20** (3), pp. 626-637,
  1983.

.. moduleauthor:: D. Orban <dominique.orban@gerad.ca>
"""

import numpy as np
from math import sqrt
import sys

__docformat__ = 'restructuredtext'


class TruncatedCG:

    def __init__(self, g, H, **kwargs):
        """
        Solve the quadratic trust-region subproblem

          minimize    < g, s > + 1/2 < s, Hs >
          subject to  < s, s >  <=  radius

        by means of the truncated conjugate gradient algorithm (aka the
        Steihaug-Toint algorithm). The notation `< x, y >` denotes the dot
        product of vectors `x` and `y`. `H` must be a symmetric matrix of
        appropriate size, but not necessarily positive definite.

        :returns:

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

        self.H = H
        self.g = g
        self.n = len(g)

        self.prefix = 'Pcg: '
        self.name = 'Truncated CG'

        self.status = '?'
        self.onBoundary = False
        self.step = None
        self.stepNorm = 0.0
        self.niter = 0
        self.dir = None

        # Formats for display
        self.hd_fmt = ' %-5s  %9s  %8s\n'
        self.header = self.hd_fmt % ('Iter', '<r,g>', 'curv')
        self.fmt = ' %-5d  %9.2e  %8.2e\n'

        return


    def _write( self, msg ):
        sys.stderr.write(self.prefix + msg)


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

    def Solve(self, **kwargs):
        """
        Solve the trust-region subproblem.

        :keywords:

          :radius:     the trust-region radius (default: None),
          :H:          linear operator representing the matrix `H`,
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
        g = self.g
        H = self.H

        # Initialization
        y = prec(g)
        ry = np.dot(g, y)
        sqrtry = sqrt(ry)
        stopTol = max(abstol, reltol * sqrtry)

        s = np.zeros(n) ; snorm2 = 0.0

        # Initialize r as a copy of g not to alter the original g
        r = g.copy()                 # r = g + H s0 = g
        p = -y                       # p = - preconditioned residual
        k = 0

        onBoundary = False
        infDescent = False

        if debug:
            self._write(self.header)
            self._write('-' * len(self.header) + '\n')

        while sqrtry > stopTol and k < maxiter and \
                not onBoundary and not infDescent:

            Hp  = H * p
            pHp = np.dot(p, Hp)

            if debug:
                self._write(self.fmt % (k, ry, pHp))

            # Compute steplength to the boundary.
            if radius is not None:
                sigma = self.to_boundary(s, p, radius, ss=snorm2)

            # Compute CG steplength.
            alpha = ry/pHp

            if pHp <= 0 and radius is None:
                # p is direction of singularity or negative curvature.
                self.status = 'infinite descent'
                snorm2 = 0
                self.dir = p
                infDescent = True
                continue

            if radius is not None and (pHp <= 0 or alpha > sigma):
                # p leads past the trust-region boundary. Move to the boundary.
                s += sigma * p
                snorm2 = radius*radius
                self.status = 'on boundary (sigma = %g)' % sigma
                onBoundary = True
                continue

            # Move to next iterate.
            s += alpha * p
            r += alpha * Hp
            y = prec(r)
            ry_next = np.dot(r, y)
            beta = ry_next/ry
            p = -y + beta * p
            ry = ry_next
            sqrtry = sqrt(ry)
            snorm2 = np.dot(s,s)
            k += 1

        # Output info about the last iteration.
        if debug:
            self._write(self.fmt % (k, ry, pHp))

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


def model_value(H, g, s):
    # Return <g,s> + 1/2 <s,Hs>
    return np.dot(g,s) + 0.5 * np.dot(s, H*s)

def model_grad(H, g, s):
    # Return g + Hs
    return g + H*s


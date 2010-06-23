"""
A pure Python/numpy implementation of the Steihaug-Toint
truncated preconditioned conjugate gradient algorithm as described in

  T. Steihaug,
  *The conjugate gradient method and trust regions in large scale optimization*,
  SIAM Journal on Numerical Analysis **20** (3), pp. 626-637, 1983.

.. moduleauthor:: D. Orban <dominique.orban@gerad.ca>
"""

import numpy as np
from math import sqrt
import sys

__docformat__ = 'restructuredtext'


class TruncatedCG:

    def __init__(self, g, **kwargs):
        """
        Solve the quadratic trust-region subproblem

          minimize    < g, s > + 1/2 < s, Hs >
          subject to  < s, s >  <=  radius

        by means of the truncated conjugate gradient algorithm (aka the
        Steihaug-Toint algorithm). The notation `< x, y >` denotes the dot
        product of vectors `x` and `y`. `H` must be a symmetric matrix of
        appropriate size, but not necessarily positive definite.

        :keywords:

          :radius:     the trust-region radius (default: ||g||),
          :matvec:     a function to compute matrix-vector products with `H`,
          :H:          the explicit matrix `H`,
          :abstol:     absolute stopping tolerance (default: 1.0e-8),
          :reltol:     relative stopping tolerance (default: 1.0e-6),
          :maxiter:    maximum number of iterations (default: 2n),
          :prec:       a user-defined preconditioner.

        :returns:

          :s:          final step,
          :it:         number of iterations,
          :snorm:      Euclidian norm of the step.

        If both `matvec` and `H` are specified, `matvec` takes precedence. If
        `matvec` is specified, it is supposed to receive a vector `p` as input
        and return the matrix-vector product `H*p`. If `H` is given, it must
        have a method named `matvec` to compute matrix-vector products.

        The algorithm stops as soon as the preconditioned norm of the gradient
        falls under

            max( abstol, reltol * g0 )

        where g0 is the preconditioned norm of the initial gradient (or the
        Euclidian norm if no preconditioner is given), or as soon as the
        iterates cross the boundary of the trust region.
        """

        self.g = g
        self.n = len(g)

        self.prefix = 'Pcg: '
        self.name = 'Truncated CG'

        # Grab optional arguments
        self.matvec = kwargs.get('matvec', None)
        if self.matvec is None:
            self.H = kwargs.get('H', None)
            if self.H is None:
                raise ValueError, 'Specify one of matvec or H'

        self.radius = kwargs.get('radius', 1.0)
        self.abstol = kwargs.get('absol', 1.0e-8)
        self.reltol = kwargs.get('reltol', 1.0e-6)
        self.maxiter = kwargs.get('maxiter', 2*self.n)
        self.prec = kwargs.get('prec', lambda v: v)
        self.debug = kwargs.get('debug', False)
        self.status = '?'

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
        sp = np.dot(s,p)
        pp = np.dot(p,p)
        if ss is None: ss = np.dot(s,s)
        sigma = (-sp + sqrt(sp*sp + pp * (radius*radius - ss)))
        sigma /= pp
        return sigma

    def Solve(self, **kwargs):
        """
        Solve the trust-region subproblem.
        """
        n = self.n
        g = self.g
        matvec = self.matvec
        if matvec is None: H = self.H
        prec = self.prec

        # Initialization
        y = prec(g)
        ry = np.dot(g, y)
        sqrtry = sqrt(ry)
        stopTol = max(self.abstol, self.reltol * sqrtry)

        Delta = self.radius
        debug = self.debug

        s = np.zeros(n) ; snorm2 = 0.0

        # Initialize r as a copy of g not to alter the original g
        r = g.copy()                 # r = g + H s0 = g
        p = -y                       # p = - preconditioned residual
        k = 0
        if matvec is None: Hp = np.empty(n)

        onBoundary = False

        if self.debug:
            self._write(self.header)
            self._write('-' * len(self.header) + '\n')

        while sqrtry > stopTol and k < self.maxiter and not onBoundary:

            # Compute matrix-vector product H*p.
            if matvec is not None:
                Hp = matvec(p)
            else:
                H.matvec(p, Hp)
            pHp = np.dot(p, Hp)

            if debug:
                self._write(self.fmt % (k, ry, pHp))

            # Compute steplength to the boundary.
            sigma = self.to_boundary(s,p,Delta,ss=snorm2)

            # Compute CG steplength.
            alpha = ry/pHp

            if pHp <= 0.0 or alpha > sigma:
                # Either p is direction of singularity or negative curvature or
                # leads past the trust-region boundary. Follow p to the boundary.
                s += sigma * p
                snorm2 = Delta*Delta
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

        if k < self.maxiter and not onBoundary:
            self.status = 'residual small'
        elif k >= self.maxiter:
            self.status = 'max iter'
        self.step = s
        self.niter = k
        self.stepNorm = sqrt(snorm2)
        return


def model_value(H, g, s):
    # Return <g,s> + 1/2 <s,Hs>
    n = len(g)
    Hs = np.zeros(n, 'd')
    H.matvec(s, Hs)
    q = 0.5 * np.dot(s, Hs)
    q = np.dot(g,s) + q
    return q

def model_grad(H, g, s):
    # Return g + Hs
    n = len(g)
    Hs = np.zeros(n, 'd')
    H.matvec(s, Hs)
    return g+Hs

def H_prod(H, v):
    # For simulation purposes only
    n = H.shape[0]
    Hv = np.zeros(n, 'd')
    H.matvec(v, Hv)
    return Hv

if __name__ == '__main__':

    from pysparse.sparse import spmatrix
    import precon
    from nlpy_timing import cputime
    #from demo_pygltr import SpecSheet
    #(H,g) = SpecSheet() # The GLTR spec sheet example
    
    t_setup = cputime()
    H = spmatrix.ll_mat_from_mtx('1138bus.mtx')
    n = H.shape[0]
    e = np.ones(n, 'd')
    g = np.empty(n, 'd')
    H.matvec(e, g)
    K = precon.DiagonalPreconditioner(H)
    #K = precon.BandedPreconditioner(H)
    t_setup = cputime() - t_setup

    t_solve = cputime()
    #(s,it,snorm) = trpcg(g, Delta=40.0, matvec = lambda p: H_prod(H,p))
    (s,it,snorm) = trpcg(g,
                         Delta = 33.5,
                         H = H,
                         prec = K.precon
                        )
    t_solve = cputime() - t_solve

    print ' #it = ', it
    print ' snorm = ', snorm
    print ' q(s) = ', model_value(H, g, s)
    print ' setup time: ', t_setup
    print ' solve time: ', t_solve

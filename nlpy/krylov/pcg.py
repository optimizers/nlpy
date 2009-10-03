"""
A pure Python/numpy implementation of the Steihaug-Toint
truncated preconditioned conjugate gradient algorithm as described in

  T. Steihaug,
  *The conjugate gradient method and trust regions in large scale optimization*,
  SIAM Journal on Numerical Analysis **20**(3), pp. 626-637, 1983.

D. Orban, Montreal.
"""

import numpy as np
from math import sqrt

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
          :hprod:      a function to compute matrix-vector products with `H`,
          :H:          the explicit matrix `H`,
          :abstol:     absolute stopping tolerance (default: 1.0e-8),
          :reltol:     relative stopping tolerance (default: 1.0e-6),
          :maxiter:    maximum number of iterations (default: 2n),
          :prec:       a user-defined preconditioner.

        :returns:

          :s:          final step,
          :it:         number of iterations,
          :snorm:      Euclidian norm of the step.

        If both `hprod` and `H` are specified, `hprod` takes precedence. If
        `hprod` is specified, it is supposed to receive a vector `p` as input and
        return the matrix-vector product `H*p`. If `H` is given, it must have a
        method named `matvec` to compute matrix-vector products.

        The algorithm stops as soon as the preconditioned norm of the gradient
        falls under

            max( abstol, reltol * g0 )

        where g0 is the preconditioned norm of the initial gradient (or the
        Euclidian norm if no preconditioner is given), or as soon as the
        iterates cross the boundary of the trust region.
        """

        self.g = g
        self.n = len(g)

        # Grab optional arguments
        self.hprod = kwargs.get('hprod', None)
        if self.hprod is None:
            self.H = kwargs.get('H', None)
            if self.H is None:
                raise ValueError, 'Specify one of hprod or H'

        self.abstol = kwargs.get('absol', 1.0e-8)
        self.reltol = kwargs.get('reltol', 1.0e-6)
        self.maxiter = kwargs.get('maxiter', 2*self.n)
        self.prec = kwargs.get('prec', lambda v: v)
        return


    def to_boundary(self, s, p, radius, ss=None):
        """
        Given vectors `s` and `p` and a trust-region radius `radius` > 0,
        return the positive scalar `sigma` such that
          || s + sigma * p || = radius
        in Euclidian norm. If known, supply optional argument `ss` whose value
        should be the squared Euclidian norm of `s`.
        """
        sp = np.dot(s,p)
        pp = np.dot(p,p)
        if ss is None: ss = np.dot(s,s)
        sigma = (-sp + sqrt(sp*sp + pp * (radius*radius - ss)))
        sigma /= pp
        return sigma

    def solve(self, **kwargs):
        """
        Solve the trust-region subproblem.

        :keywords:

          :radius:     the trust-region radius (default: ||g||),
        """
        n = self.n
        g = self.g
        hprod = self.hprod
        if hprod is None: H = self.H
        prec = self.prec

        # Initialization
        y = prec(g)
        res = sqrt(np.dot(g, y))
        stopTol = max(self.abstol, self.reltol * res)

        Delta = kwargs.get('radius', res)
        s = np.zeros(n) ; snorm = 0.0

        # Initialize r as a copy of g not to alter the original g
        r = g.copy()                 # r = g + H s0 = g
        p = -y                       # p = - preconditioned residual
        k = 0
        Hp = np.empty(n)

        hitBoundary = False

        while res > stopTol and k < self.maxiter and not hitBoundary:

            # Compute matrix-vector product H*p.
            if hprod is not None:
                Hp = hprod(p)
            else:
                H.matvec(p, Hp)

            pHp = np.dot(p, Hp)

            # Compute steplength to the boundary.
            sigma = self.to_boundary(s,p,Delta,ss=snorm*snorm)

            # Compute CG steplength.
            alpha = res*res/pHp

            if pHp <= 0.0 or alpha > sigma:
                # Either p is direction of singularity or negative curvature or
                # leads past the trust-region boundary. Follow p to the boundary.
                s += sigma * p
                snorm = Delta
                self.step = s
                self.niter = k
                self.stepNorm = snorm
                hitBoundary = True
                continue

            # Move to next iterate.
            s += alpha * p
            snorm = sqrt(np.dot(s,s))
            r += alpha * Hp
            y = prec(r)
            res1 = sqrt(np.dot(r, y))
            res1overres = res1/res
            beta = res1overres * res1overres
            p = -y + beta * p
            res = res1
            k += 1

        self.step = s
        self.niter = k
        self.stepNorm = snorm
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

    from pysparse import spmatrix
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
    #(s,it,snorm) = trpcg(g, Delta=40.0, hprod = lambda p: H_prod(H,p))
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

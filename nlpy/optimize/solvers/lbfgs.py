from nlpy.model import InverseLBFGS
from nlpy.optimize.ls.pymswolfe import StrongWolfeLineSearch
from nlpy.tools import norms
from nlpy.tools.timing import cputime

__docformat__ = 'restructuredtext'


class LBFGSFramework(object):
    """
    Class LBFGSFramework provides a framework for solving unconstrained
    optimization problems by means of the limited-memory BFGS method.

    Instantiation is done by

    lbfgs = LBFGSFramework(nlp)

    where nlp is an instance of a nonlinear problem. A solution of the
    problem is obtained by called the solve member function, as in

    lbfgs.solve().

    :keywords:

        :npairs:    the number of (s,y) pairs to store (default: 5)
        :x0:        the starting point (default: nlp.x0)
        :maxiter:   the maximum number of iterations (default: max(10n,1000))
        :abstol:    absolute stopping tolerance (default: 1.0e-6)
        :reltol:    relative stopping tolerance (default: `nlp.stop_d`)

    Other keyword arguments will be passed to InverseLBFGS.

    The linesearch used in this version is Jorge Nocedal's modified More and
    Thuente linesearch, attempting to ensure satisfaction of the strong Wolfe
    conditions. The modifications attempt to limit the effects of rounding
    error inherent to the More and Thuente linesearch.
    """
    def __init__(self, nlp, **kwargs):

        self.nlp = nlp
        self.npairs = kwargs.get('npairs', 5)
        self.silent = kwargs.get('silent', False)
        self.abstol = kwargs.get('abstol', 1.0e-6)
        self.reltol = kwargs.get('reltol', self.nlp.stop_d)
        self.iter = 0
        self.nresets = 0
        self.converged = False

        self.lbfgs = InverseLBFGS(self.nlp.n, **kwargs)

        self.x = kwargs.get('x0', self.nlp.x0)
        self.f = self.nlp.obj(self.x)
        self.g = self.nlp.grad(self.x)
        self.gnorm = norms.norm2(self.g)
        self.f0 = self.f
        self.g0 = self.gnorm

        # Optional arguments
        self.maxiter = kwargs.get('maxiter', max(10 * self.nlp.n, 1000))
        self.tsolve = 0.0

    def solve(self):

        tstart = cputime()

        # Initial LBFGS matrix is the identity. In other words,
        # the initial search direction is the steepest descent direction.

        # This is the original L-BFGS stopping condition.
        #stoptol = self.nlp.stop_d * max(1.0, norms.norm2(self.x))
        stoptol = max(self.abstol, self.reltol * self.g0)

        while self.gnorm > stoptol and self.iter < self.maxiter:

            if not self.silent:
                print '%-5d  %-12g  %-12g' % (self.iter, self.f, self.gnorm)

            # Obtain search direction
            d = self.lbfgs * (-self.g)

            # Prepare for modified More-Thuente linesearch
            if self.iter == 0:
                stp0 = 1.0 / self.gnorm
            else:
                stp0 = 1.0
            SWLS = StrongWolfeLineSearch(self.f,
                                         self.x,
                                         self.g,
                                         d,
                                         lambda z: self.nlp.obj(z),
                                         lambda z: self.nlp.grad(z),
                                         stp=stp0)
            # Perform linesearch
            SWLS.search()

            # SWLS.x  contains the new iterate
            # SWLS.g  contains the objective gradient at the new iterate
            # SWLS.f  contains the objective value at the new iterate
            s = SWLS.x - self.x
            self.x = SWLS.x
            y = SWLS.g - self.g
            self.g = SWLS.g
            self.gnorm = norms.norm2(self.g)
            self.f = SWLS.f

            # Update inverse Hessian approximation using the most recent pair
            self.lbfgs.store(s, y)
            self.iter += 1

        self.tsolve = cputime() - tstart
        self.converged = (self.iter < self.maxiter)

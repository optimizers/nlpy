"""
 TRUNK
 Trust-Region Method for Unconstrained Programming.

 A first unconstrained optimization solver in Python
 The Python version of the celebrated  F90/95 solver
 D. Orban                        Montreal Sept. 2003
"""
from nlpy.optimize.solvers import lbfgs    # For preconditioning
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.tools import norms
from nlpy.tools.timing import cputime

import numpy
import sys
from math import sqrt

__docformat__ = 'restructuredtext'

class TrunkFramework:
    """
    An abstract framework for a trust-region-based algorithm for nonlinear
    unconstrained programming. Instantiate using

    `TRNK = TrunkFramework(nlp, TR, TrSolver)`

    :parameters:

        :nlp:   a :class:`NLPModel` object representing the problem. For
                instance, nlp may arise from an AMPL model
        :TR:    a :class:`TrustRegionFramework` object
        :TrSolver:   a :class:`TrustRegionSolver` object.


    :keywords:

        :x0:           starting point                    (default nlp.x0)
        :reltol:       relative stopping tolerance       (default `nlp.stop_d`)
        :abstol:       absolute stopping tolerance       (default 1.0e-6)
        :maxiter:      maximum number of iterations      (default max(1000,10n))
        :inexact:      use inexact Newton stopping tol   (default False)
        :ny:           apply Nocedal/Yuan linesearch     (default False)
        :nbk:          max number of backtracking steps in Nocedal/Yuan
                       linesearch                        (default 5)
        :monotone:     use monotone descent strategy     (default False)
        :nIterNonMono: number of iterations for which non-strict descent can
                       be tolerated if monotone=False    (default 25)
        :silent:       verbosity level                   (default False)

    Once a `TrunkFramework` object has been instantiated and the problem is
    set up, solve problem by issuing a call to `TRNK.solve()`. The algorithm
    stops as soon as the Euclidian norm of the gradient falls below

        ``max(abstol, reltol * g0)``

    where ``g0`` is the Euclidian norm of the gradient at the initial point.
    """

    def __init__(self, nlp, TR, TrSolver, **kwargs):

        self.nlp    = nlp
        self.TR     = TR
        self.TrSolver = TrSolver
        self.solver   = None    # Will point to solver data in Solve()
        self.iter   = 0         # Iteration counter
        self.cgiter = 0
        self.x      = kwargs.get('x0', self.nlp.x0)
        self.f      = self.nlp.obj(self.x)
        self.f0     = self.f
        self.g      = self.nlp.grad(self.x)  # Current  gradient
        self.g_old  = self.g                   # Previous gradient
        self.save_g = False              # For methods that need g_{k-1} and g_k
        self.gnorm  = norms.norm2(self.g)
        self.g0     = self.gnorm
        self.alpha  = 1.0       # For Nocedal-Yuan backtracking linesearch
        self.tsolve = 0.0

        self.reltol  = kwargs.get('reltol', self.nlp.stop_d)
        self.abstol  = kwargs.get('abstol', 1.0e-6)
        self.maxiter = kwargs.get('maxiter', max(1000, 10*self.nlp.n))
        self.silent  = kwargs.get('silent', False)
        self.ny      = kwargs.get('ny', False)
        self.nbk     = kwargs.get('nbk', 5)
        self.inexact = kwargs.get('inexact', False)
        self.monotone = kwargs.get('monotone', False)
        self.nIterNonMono = kwargs.get('nIterNonMono', 25)

        self.hformat = '%-5s  %8s  %7s  %5s  %8s  %8s  %4s\n'
        self.header  = self.hformat % ('Iter','f(x)','|g(x)|','cg','rho','Radius','Stat')
        self.hlen   = len(self.header)
        self.hline  = '-' * self.hlen + '\n'
        self.format = '%-5d  %8.1e  %7.1e  %5d  %8.1e  %8.1e  %4s\n'
        self.radii = [ TR.Delta ]

    def hprod(self, v, **kwargs):
        """
        Default hprod based on nlp's hprod. User should overload to
        provide a custom routine, e.g., a quasi-Newton approximation.
        """
        return self.nlp.hprod(self.nlp.pi0, v)

    def precon(self, v, **kwargs):
        """
        Generic preconditioning method---must be overridden
        """
        return v

    def PostIteration(self, **kwargs):
        """
        Override this method to perform work at the end of an iteration. For
        example, use this method for preconditioners that need updating,
        e.g., a limited-memory BFGS preconditioner.
        """
        return None

    def Solve(self, **kwargs):

        nlp = self.nlp
        rho = 1                  # Dummy initial value for rho
        niter = 0                # Dummy initial value for number of inners
        status = ''              # Dummy initial step status
        if self.inexact:
            cgtol = 1.0
        else:
            cgtol = -1.0
        stoptol = max(self.abstol, self.reltol * self.g0)

        # Initialize non-monotonicity parameters.
        if not self.monotone:
            fMin = fRef = fCan = self.f0
            l = 0
            sigRef = sigCan = 0

        t = cputime()

        while self.gnorm > stoptol and self.iter <= self.maxiter:

            # Print out header, say, every 20 iterations
            if self.iter % 20 == 0 and not self.silent:
                sys.stdout.write(self.hline)
                sys.stdout.write(self.header)
                sys.stdout.write(self.hline)

            if not self.silent:
                sys.stdout.write(self.format % (self.iter, self.f,
                          self.gnorm, niter, rho,
                          self.TR.Delta, status))

            # Save current gradient
            if self.save_g:
                self.g_old = self.g.copy()

            # Iteratively minimize the quadratic model in the trust region
            # m(s) = <g, s> + 1/2 <s, Hs>
            # Note that m(s) does not include f(x): m(0) = 0.

            if self.inexact:
                cgtol = max(1.0e-6, min(0.5 * cgtol, sqrt(self.gnorm)))

            H = SimpleLinearOperator(nlp.n, nlp.n,
                                     lambda v: self.hprod(v),
                                     symmetric=True)

            self.solver = self.TrSolver(self.g, H)
            self.solver.Solve(prec=self.precon,
                              radius=self.TR.Delta,
                              reltol=cgtol,
                              #debug=True
                              )

            step = self.solver.step
            snorm = self.solver.stepNorm
            niter = self.solver.niter

            # Obtain model value at next candidate
            m = self.solver.m
            if m is None:
                m = numpy.dot(self.g, step) + 0.5*numpy.dot(step, H * step)

            self.cgiter += niter
            x_trial = self.x + step
            f_trial = nlp.obj(x_trial)

            rho  = self.TR.Rho(self.f, f_trial, m)

            if not self.monotone:
                rhoHis = (fRef - f_trial)/(sigRef - m)
                rho = max(rho, rhoHis)

            status = 'Rej'
            if rho >= self.TR.eta1:

                # Trust-region step is accepted.

                self.TR.UpdateRadius(rho, snorm)
                self.x = x_trial
                self.f = f_trial
                self.g = nlp.grad(self.x)
                self.gnorm = norms.norm2(self.g)
                status = 'Acc'

                # Update non-monotonicity parameters.
                if not self.monotone:
                    sigRef = sigRef - m
                    sigCan = sigCan - m
                    if f_trial < fMin:
                        fCan = f_trial
                        fMin = f_trial
                        sigCan = 0
                        l = 0
                    else:
                        l = l + 1

                    if f_trial > fCan:
                        fCan = f_trial
                        sigCan = 0

                    if l == self.nIterNonMono:
                        fRef = fCan
                        sigRef = sigCan

            else:

                # Trust-region step is rejected.

                if self.ny: # Backtracking linesearch following "Nocedal & Yuan"
                    slope = numpy.dot(self.g, step)
                    bk = 0
                    while bk < self.nbk and \
                            f_trial >= self.f + 1.0e-4 * self.alpha * slope:
                        bk = bk + 1
                        self.alpha /= 1.2
                        x_trial = self.x + self.alpha * step
                        f_trial = nlp.obj(x_trial)
                    self.x = x_trial
                    self.f = f_trial
                    self.g = nlp.grad(self.x)
                    self.gnorm = norms.norm2(self.g)
                    self.TR.Delta = self.alpha * snorm
                    status = 'N-Y'
                else:
                    self.TR.UpdateRadius(rho, snorm)

            self.status = status
            self.radii.append(self.TR.Delta)
            self.PostIteration()
            self.iter += 1
            self.alpha = 1.0     # For the next iteration

        self.tsolve = cputime() - t    # Solve time

        # Set final solver status.
        if self.gnorm <= stoptol:
            self.status = 'opt'
        else: # self.iter > self.maxiter:
            self.status = 'itr'



class TrunkLbfgsFramework(TrunkFramework):
    """
    Class TrunkLbfgsFramework is a subclass of TrunkFramework. The method is
    based on the same trust-region algorithm with Nocedal-Yuan backtracking.
    The only difference is that a limited-memory BFGS preconditioner is used
    and maintained along the iterations. See class TrunkFramework for more
    information.
    """

    def __init__(self, nlp, TR, TrSolver, **kwargs):

        TrunkFramework.__init__(self, nlp, TR, TrSolver, **kwargs)
        self.npairs = kwargs.get('npairs', 5)
        self.lbfgs = lbfgs.InverseLBFGS(nlp.n, npairs=self.npairs)
        self.save_g = True

    def precon(self, v, **kwargs):
        """
        This method implements limited-memory BFGS preconditioning. It
        overrides the default precon() of class TrunkFramework.
        """
        return self.lbfgs.solve(v)

    def PostIteration(self, **kwargs):
        """
        This method updates the limited-memory BFGS preconditioner by appending
        the most rencet (s,y) pair to it and possibly discarding the oldest one
        if all the memory has been used.
        """
        s = self.alpha * self.solver.step
        y = self.g - self.g_old
        self.lbfgs.store(s, y)

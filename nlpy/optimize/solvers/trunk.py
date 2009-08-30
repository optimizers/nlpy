#!/usr/bin/env python
"""
                    T R U N K
 Trust-Region Method for Unconstrained Programming.

 A first unconstrained optimization solver in Python
 The Python version of the celebrated  F90/95 solver
 D. Orban                        Montreal Sept. 2003
"""
from nlpy import amplpy
from nlpy.optimize.tr import trustregion
from nlpy.optimize.solvers import lbfgs    # For preconditioning
from nlpy.tools import norms
from nlpy.tools.timing import cputime

import numpy
import sys
from math import sqrt

class TrunkFramework:
    """
    Class TrunkFramework is an abstract framework for a trust-region-based
    algorithm for nonlinear unconstrained programming. Instantiate using

        TRNK = TrunkFramework(nlp, TR, TrSolver)

    where nlp is a NLP object representing the problem---for instance, nlp
    may arise from an AMPL model---TR is a TrustRegionFramework object and
    TrSolver is a TrustRegionSolver object.
    The nlp object should have the following methods
    
        obj     to evaluate the objective function value
        grad    to evaluate the objective function gradient
        hprod   to compute matrix-vector products with the Hessian matrix
                of the objective at the most recent iterate.
        
    nlp should also possess a member named 'n' giving the number of variables.

    The TrustRegionSolver should have the Solve method and possess members named
    niter, step and stepNorm.

    Keyword arguments are:
        x0           starting point                     (default nlp.x0)
        reltol     relative stopping tolerance        (default 1.0e-12)
        abstol     absolute stopping tolerance        (default 1.0e-6)
        maxiter      maximum number of outer iterations (default max(100,2n))
        inexact      use inexact Newton stopping tol    (default False)
        ny           apply Nocedal/Yuan linesearch      (default False)
        silent       verbosity level                    (default False)

    Once a TrunkFramework object has been instantiated and the problem is
    set up, solve problem by issuing a call to TRNK.solve(). The algorithm
    stops as soon as the Euclidian norm of the gradient falls below

        max(abstol, reltol * g0)

    where g0 is the Euclidian norm of the gradient at the initial point.
    """

    def __init__(self, nlp, TR, TrSolver, **kwargs):

        self.nlp    = nlp
        self.TR     = TR
        self.TrSolver = TrSolver
        self.solver   = None    # Will point to solver data in Solve()
        self.iter   = 0
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

        self.reltol  = kwargs.get('reltol',  1.0e-12)
        self.abstol  = kwargs.get('abstol',  1.0e-6)
        self.maxiter = kwargs.get('maxiter', max(100, 2*self.nlp.n))
        self.silent  = kwargs.get('silent',  False)
        self.ny      = kwargs.get('ny',      False)
        self.inexact = kwargs.get('inexact', False)
    
        self.hformat = '%-5s  %8s  %7s  %5s  %8s  %8s  %4s\n'
        self.header  = self.hformat % ('Iter','f(x)','|g(x)|','cg','rho','Radius','Stat')
        self.hlen   = len(self.header)
        self.hline  = '-' * self.hlen + '\n'
        self.format = '%-5d  %8.1e  %7.1e  %5d  %8.1e  %8.1e  %4s\n'
        self.radii = [ TR.Delta ]

    def Precon(self, v, **kwargs):
        """
        Generic preconditioning method---must be overridden
        """
        return v

    def UpdatePrecon(self, **kwargs):
        """
        Override this method for preconditioners that need updating,
        e.g., a limited-memory BFGS preconditioner.
        """
        return None

    def Solve(self):

        nlp = self.nlp
        rho = 1                  # Dummy initial value for rho
        niter = 0                # Dummy initial value for number of inners
        status = ''              # Dummy initial step status
        if self.inexact:
            cgtol = 1.0
        else:
            cgtol = -1.0
        stoptol = max(self.abstol, self.reltol * self.g0)

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

            self.solver = self.TrSolver(self.g,
                                        matvec = lambda v: nlp.hprod(nlp.pi0,v),
                                        #H = nlp.hess(self.x,nlp.pi0),
                                        prec = self.Precon,
                                        radius = self.TR.Delta,
                                        reltol = cgtol)
            self.solver.Solve()

            step = self.solver.step
            snorm = self.solver.stepNorm
            niter = self.solver.niter

            # Obtain model value at next candidate
            m = self.solver.m
            if m is None:
                m = numpy.dot(self.g,step) + \
                    0.5*numpy.dot(step, nlp.hprod(nlp.pi0,step))

            self.cgiter += niter
            x_trial = self.x + step
            f_trial = nlp.obj(x_trial)
            rho  = self.TR.Rho(self.f, f_trial, m)

            status = 'Rej'
            if rho >= self.TR.eta1:
                self.TR.UpdateRadius(rho, snorm)
                self.x = x_trial
                self.f = f_trial
                self.g = nlp.grad(self.x)
                self.gnorm = norms.norm2(self.g)
                status = 'Acc'
            else:
                if self.ny: # Backtracking linesearch following "Nocedal & Yuan"
                    slope = numpy.dot(self.g, step)
                    while f_trial >= self.f + 1.0e-4 * self.alpha * slope:
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

            self.radii.append(TR.Delta)
            self.UpdatePrecon()
            self.iter += 1
            self.alpha = 1.0     # For the next iteration

        self.tsolve = cputime() - t    # Solve time


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
        self.lbfgs = lbfgs_class.LbfgsUpdate(nlp.n, npairs = self.npairs)
        self.save_g = True

    def Precon(self, v, **kwargs):
        """
        This method implements limited-memory BFGS preconditioning. It
        overrides the default Precon() of class TrunkFramework.
        """
        return self.lbfgs.solve(self.iter, v)

    def UpdatePrecon(self, **kwargs):
        """
        This method updates the limited-memory BFGS preconditioner by appending
        the most rencet (s,y) pair to it and possibly discarding the oldest one
        if all the memory has been used.
        """
        s = self.alpha * self.solver.step
        y = self.g - self.g_old
        self.lbfgs.store(self.iter, s, y)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        sys.stderr.write('Please specify model name\n')
        sys.exit(-1)

    # Set printing standards for arrays
    numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

    showbanner = True
    t = cputime()
    ProblemName = sys.argv[1:]
    nlp = amplpy.AmplModel(ProblemName)                       # Create a model

    if nlp.m != 0:                             # Check for unconstrained problem
        sys.stderr.write('Problem has general constraints\n')
        sys.exit(-1)

    TR = trustregion.TrustRegionFramework(Delta=1.0,
                                          eta1=0.05,
                                          eta2=0.9,
                                          gamma1=0.25,
                                          gamma2=2.5)

    # When instantiating TrunkFramework of TrunkLbfgsFramework,
    # we select a trust-region subproblem solver of our choice.
    #TRNK = TrunkLbfgsFramework(nlp,
    TRNK = TrunkFramework(nlp,
                          TR,
                          #trustregion.TrustRegionGLTR,
                          trustregion.TrustRegionCG,
                          silent = False,
                          ny = True,
                          inexact = True)
    TRNK.TR.Delta = 0.1 * TRNK.gnorm         # Reset initial trust-region radius
    t_setup = cputime() - t                  # Setup time

    if showbanner:
        print
        print '------------------------------------------'
        print 'Trunk: Solving problem %-s with parameters' % ProblemName
        print 'eta1 = %-g  eta2 = %-g  gamma1 = %-g  gamma2 = %-g Delta0 = %-g' % (TR.eta1, TR.eta2, TR.gamma1, TR.gamma2, TR.Delta)
        print '------------------------------------------'
        print

    TRNK.Solve()
    
    # Output final statistics
    print
    print 'Final variables:', TRNK.x
    print
    print '-------------------------------'
    print 'Trunk: End of Execution'
    print '  Problem                     : %-s' % ProblemName
    print '  Dimension                   : %-d' % nlp.n
    print '  Initial/Final Objective     : %-g/%-g' % (TRNK.f0, TRNK.f)
    print '  Initial/Final Gradient Norm : %-g/%-g' % (TRNK.g0, TRNK.gnorm)
    print '  Number of iterations        : %-d' % TRNK.iter
    print '  Number of function evals    : %-d' % TRNK.nlp.feval
    print '  Number of gradient evals    : %-d' % TRNK.nlp.geval
    print '  Number of Hessian  evals    : %-d' % TRNK.nlp.Heval
    print '  Number of matvec products   : %-d' % TRNK.nlp.Hprod
    print '  Total/Average Lanczos iter  : %-d/%-g' % (TRNK.cgiter, (float(TRNK.cgiter)/TRNK.iter))
    print '  Setup/Solve time            : %-gs/%-gs' % (t_setup, TRNK.tsolve)
    print '  Total time                  : %-gs' % (t_setup + TRNK.tsolve)
    print '-------------------------------'

    #nlp.writesol(TRNK.x, nlp.pi0, 'And the winner is')    # Output "solution"
    nlp.close()                                    # Close connection with model

    # Say we want to plot the evolution of the trust-region radius
    try:
        import pylab
    except:
        sys.stderr.write('If you had pylab installed, you would be looking ')
        sys.stderr.write('at a plot of the evolution of the trust-region ')
        sys.stderr.write('radius, right now.\n')
        sys.exit(0)
    radii = numpy.array(TRNK.radii, 'd')
    pylab.plot(numpy.where(radii < 100, radii, 100))
    pylab.title('Trust-region radius')
    pylab.show()

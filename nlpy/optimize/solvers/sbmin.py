# -*- coding: utf-8 -*-
"""
SBMIN
A Trust-Region Method for Bound-Constrained Optimization.
"""
from nlpy.tools.norms import norm_infty
from nlpy.tools.timing import cputime
from nlpy.tools.exceptions import UserExitRequest
import numpy as np
import logging
from math import sqrt
from nlpy.model import QPModel
from pykrylov import LinearOperator

__docformat__ = 'restructuredtext'


class SBMINFramework(object):
    """
    An abstract framework for a trust-region-based algorithm for the nonlinear
    bound-constrained optimization problem

        minimize    f(x)
        subject to  l <= x <= u

    where some components of l and/or u may be infinite.

    Instantiate using

        `SBMIN = SBMINFramework(nlp, TR, TrSolver)`

    :parameters:

        :nlp:       a :class:`NLPModel` object representing the problem. For
                      instance, nlp may arise from an AMPL model
        :TR:        a :class:`TrustRegionFramework` object
        :TrSolver:  a :class:`TrustRegionSolver` object.


    :keywords:

        :x0:           starting point                    (default nlp.x0)
        :reltol:       relative stopping tolerance       (default 1.0e-5)
        :maxiter:      maximum number of iterations      (default 10n)
        :logger_name:  name of a logger object that can be used in the post
                       iteration                         (default None)
        :verbose:      print some info if True                 (default True)

    Once a `SBMINFramework` object has been instantiated and the problem is
    set up, solve problem by issuing a call to `SBMIN.solve()`.


    The algorithm stops as soon as one of this criteria is statisfied:

        :exitOptimal:   the infinity norm of the projected gradient
                        into the feasible box falls below `reltol`
                        (default 1.0e-7)

        :exitIter:      maximum number of iterations is reached
                        (default max(100, 10n)

        :exitTR:        trust region radius is smaller than 10*eps_mach

    """

    def __init__(self, nlp, TR, TrSolver, **kwargs):

        self.nlp    = nlp
        self.TR     = TR
        self.TrSolver = TrSolver
        self.solver   = None    # Will point to solver data in Solve()
        self.iter   = 0         # Iteration counter
        self.x0      = kwargs.get('x0', self.nlp.x0.copy())
        self.x = None
        self.f      = None
        self.f0     = kwargs.get('f0', None)
        self.g      = None
        self.g_old  = kwargs.get('g0', None)
        self.lg     = None
        self.lg_old = kwargs.get('Lg0', None)
        self.save_g = False           # For methods that need g_{k-1} and g_k
        self.save_lg = False          # Similar to save_g
        self.pgnorm  = None
        self.tsolve = 0.0
        self.true_step = None
        self.update_on_rejected_step = kwargs.get('update_on_rejected_step',
                                                  False)

        # Option for resetting trust-region radius
        self.reset_radius = kwargs.get('reset_radius', True)

        # Options for Nocedal-Yuan backtracking
        self.ny      = kwargs.get('ny', False)
        self.nbk     = kwargs.get('nbk', 5)
        self.alpha   = 1.0

        # Use magical steps to update slack variables
        self.magic_steps_cons = kwargs.get('magic_steps_cons', False)
        self.magic_steps_agg = kwargs.get('magic_steps_agg', False)

        # If both options are set, use the aggressive type
        if self.magic_steps_agg and self.magic_steps_cons:
            self.magic_steps_cons = False

        # Options for non monotone descent strategy
        self.monotone = kwargs.get('monotone', False)
        self.nIterNonMono = kwargs.get('nIterNonMono', 10)

        self.abstol  = kwargs.get('abstol', 1.0e-7)
        self.reltol  = kwargs.get('reltol', 1.0e-7)
        self.maxiter = kwargs.get('maxiter', 10*self.nlp.n)
        self.verbose = kwargs.get('verbose', False)
        self.total_bqpiter = 0
        self.cgiter = 0
        self.total_cgiter = 0

        self.hformat = '%-5s  %9s  %7s %7s %5s  %8s  %8s  %4s'
        self.header  = self.hformat % ('Iter', 'f(x)', '|g(x)|', 'step', 'bqp',
                                       'rho', 'Radius', 'Stat')
        self.hlen = len(self.header)
        self.hline = '-' * self.hlen
        self.format = '%-5d  %9.2e  %7.1e %7.1e %5d  %8.1e  %8.1e  %4s'
        self.format0 = '%-5d  %9.2e  %7.1e %7s %5s  %8s  %8s  %4s'
        self.radii = None

        # Initialize some counters for BQP
        self.hprod_bqp_linesearch = 0
        self.hprod_bqp_linesearch_fail = 0
        self.nlinesearch = 0
        self.hprod_bqp_cg = 0

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlpy.sbmin')
        self.log = logging.getLogger(logger_name)
        self.log.addHandler(logging.NullHandler())
        if not self.verbose:
            self.log.propagate = False

    def project(self, x):
        "Project x into the bounds."
        return np.maximum(np.minimum(x, self.nlp.Uvar), self.nlp.Lvar)

    def projected_gradient(self, x, g):
        """
        Compute the projected gradient of f(x) into the feasible box

                   l <= x <= u
        """
        return x - self.project(x - g)

    def magical_step(self, x, g, **kwargs):
        """
        Compute a "magical step" to improve the convergence rate of the
        inner minimization algorithm. This step minimizes the augmented
        Lagrangian with respect to the slack variables only for a fixed set
        of decision variables.
        """
        on = self.nlp.nlp.original_n
        m_step = np.zeros(self.nlp.n)
        m_step[on:] = -g[on:]/self.nlp.rho
        # Assuming slack variables are restricted to [0,+inf) interval
        m_step[on:] = np.where(-m_step[on:] > x[on:], -x[on:], m_step[on:])
        return m_step

    def hprod(self, x, z, v, **kwargs):
        """
        Default hprod based on nlp's hprod. User should overload to
        provide a custom routine, e.g., a quasi-Newton approximation.
        """
        return self.nlp.hprod(x, z, v)

    def PostIteration(self, **kwargs):
        """
        Override this method to perform work at the end of an iteration. For
        example, use this method for updating a LBFGS Hessian
        """
        return None

    def Solve(self, **kwargs):

        nlp = self.nlp

        # Gather initial information.
        self.x = self.project(self.x0)
        self.x_old = self.x.copy()

        if self.f0 is None:
            self.f0 = nlp.obj(self.x)
        self.f = self.f0

        if self.g is None:
            self.g = nlp.grad(self.x)

        self.pgnorm = norm_infty(self.projected_gradient(self.x, self.g))
        self.pg0 = self.pgnorm

        if self.save_lg:
            if self.lg_old is None:
                self.lg_old = self.nlp.dual_feasibility(self.x)
            self.lg = self.lg_old.copy()

        self.f  = self.f0

        if self.reset_radius:
            # Reset initial trust-region radius.
            self.TR.Delta = np.maximum(0.1 * self.pgnorm, .2)
        self.radii = [self.TR.Delta]

        # Initialize non-monotonicity parameters.
        if not self.monotone:
            self.log.debug('Using Non monotone descent strategy')
            fMin = fRef = fCan = self.f0
            l = 0
            sigRef = sigCan = 0

        bqptol = 1.0
        stoptol = self.reltol * self.pg0 + self.abstol
        step_status = None
        exitIter = exitUser = exitTR = False
        exitOptimal = self.pgnorm <= stoptol
        status = ''

        # Print out header and initial log.
        if self.iter % 20 == 0:
            self.log.info(self.hline)
            self.log.info(self.header)
            self.log.info(self.hline)
            self.log.info(self.format0 % (self.iter, self.f,
                                          self.pgnorm, '', '', '',
                                          '', ''))

        t = cputime()

        while not (exitUser or exitOptimal or exitIter or exitTR):
            #dercheck = DerivativeChecker(nlp, self.x)
            #dercheck.check(verbose=True)

            self.iter += 1
            self.x_old = self.x.copy()
            self.f_old = self.f
            # Save current gradient for quasi-Newton approximation
            if self.save_g:
                self.g_old = self.g.copy()

            if self.save_lg:
                self.lg_old = self.lg.copy()

            # Iteratively minimize the quadratic model in the trust region
            #          m(d) = <g, d> + 1/2 <d, Hd>
            #     s.t.     ll <= d <= uu
            qp = TrustBQPModel(nlp, self.x, self.TR.Delta, self.hprod, gk=self.g)
            if step_status != 'Rej':
                bqptol = max(1e-6, min(0.1 * bqptol, sqrt(self.pgnorm)))

            self.solver = self.TrSolver(qp, qp.grad, **kwargs)
            self.solver.Solve(reltol=bqptol)

            step = self.solver.step
            stepnorm = self.solver.stepNorm
            bqpiter = self.solver.niter
            self.cgiter = self.solver.bqpSolver.cgiter
            self.hprod_bqp_linesearch += self.solver.bqpSolver.hprod_bqp_linesearch
            self.hprod_bqp_linesearch += self.solver.bqpSolver.hprod_bqp_linesearch_fail
            self.nlinesearch += self.solver.bqpSolver.nlinesearch
            self.hprod_bqp_cg += self.solver.bqpSolver.hprod_bqp_cg

            # Obtain model value at next candidate
            m = self.solver.m

            self.total_bqpiter += bqpiter
            self.total_cgiter += self.cgiter
            x_trial = self.x + step
            f_trial = nlp.obj(x_trial)

            # Aggressive magical steps
            # (i.e. the magical steps can influence the trust region size)
            if self.magic_steps_agg:
                x_inter = x_trial.copy()
                f_inter = f_trial
                g_inter = nlp.grad(x_inter)
                m_step = self.magical_step(x_inter, g_inter)
                x_trial = x_inter + m_step
                f_trial = nlp.obj(x_trial)
                if f_trial <= f_inter:
                    # Safety check for machine-precision errors in magical step
                    m = m - (f_inter - f_trial)

            rho = self.TR.Rho(self.f, f_trial, m)

            if not self.monotone:
                rhoHis = (fRef - f_trial) / (sigRef - m)
                rho = max(rho, rhoHis)

            step_status = 'Rej'

            if rho >= self.TR.eta1:

                # Trust-region step is successful
                if self.magic_steps_cons:
                    g_trial = nlp.grad(x_trial)
                    m_step = self.magical_step(x_trial, g_trial)
                    x_trial += m_step
                    f_trial = nlp.obj(x_trial)
                    stepnorm = np.linalg.norm(x_trial - self.x)
                    # Safety check for machine-precision errors in magical step
                    if f_trial <= self.f:
                        m = m - (self.f - f_trial)

                self.TR.UpdateRadius(rho, stepnorm)
                self.x = x_trial.copy()
                self.f = f_trial
                self.g = nlp.grad(self.x)
                self.pgnorm = norm_infty(self.projected_gradient(self.x, self.g))

                step_status = 'Acc'

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

                # Attempt Nocedal-Yuan backtracking if requested
                if self.ny:
                    alpha = self.alpha
                    slope = np.dot(self.g, step)
                    bk = 0

                    while bk < self.nbk and \
                            f_trial >= self.f + 1.0e-4 * alpha * slope:
                        bk = bk + 1
                        alpha /= 1.5
                        x_trial = self.x + alpha * step
                        f_trial = nlp.obj(x_trial)

                    if f_trial >= self.f + 1.0e-4 * alpha * slope:
                        # Backtrack failed to produce an improvement,
                        # keep the current x, f, and g.
                        # (Monotone strategy)
                        step_status = 'NY-'
                    else:
                        # Backtrack succeeded, update the current point
                        self.x = x_trial.copy()
                        self.f = f_trial
                        self.g = nlp.grad(self.x)

                        # Conservative magical step if backtracking succeeds
                        if self.magic_steps_cons:
                            m_step = self.magical_step(self.x, self.g)
                            self.x += m_step
                            self.f = nlp.obj(self.x)
                            self.g = nlp.grad(self.x)

                        self.pgnorm = norm_infty(self.projected_gradient(self.x, self.g))

                        step_status = 'NY+'

                    # Update the TR radius regardless of backtracking success
                    self.TR.Delta = alpha * stepnorm

                else:
                    # Trust-region step is unsuccessful
                    self.TR.UpdateRadius(rho, stepnorm)

            self.step_status = step_status
            self.radii.append(self.TR.Delta)
            status = ''

            if self.save_lg:
                self.lg = nlp.dual_feasibility(self.x)

            self.true_step = self.x - self.x_old
            self.pstatus = step_status if step_status != 'Acc' else ''
            self.radius = self.radii[-2]

            try:
                self.PostIteration()
            except UserExitRequest:
                status = 'usr'

            # Print out header, say, every 20 iterations
            if self.iter % 20 == 1 and self.verbose:
                self.log.info(self.hline)
                self.log.info(self.header)
                self.log.info(self.hline)

            self.log.info(self.format % (self.iter, self.f,
                          self.pgnorm, np.linalg.norm(self.true_step),
                          bqpiter, rho, self.radius, self.pstatus))

            exitOptimal = self.pgnorm <= stoptol
            exitIter    = self.iter > self.maxiter
            exitTR      = self.TR.Delta <= 10.0 * self.TR.eps
            exitUser    = status == 'usr'

        self.tsolve = cputime() - t

        # Set final solver status.
        if exitUser:
            pass
        elif exitOptimal:
            status = 'opt'
        elif exitTR:
            status = 'tr'
        else:  # self.iter > self.maxiter:
            status = 'itr'
        self.status = status


class SBMINLqnFramework(SBMINFramework):
    """
    Class SBMINLqnFramework is a subclass of SBMINFramework. The method is
    based on a trust-region-based algorithm for nonlinear box constrained
    programming.
    The only difference is that a limited-memory Quasi-Newton Hessian
    approximation is used and maintained along the iterations. See class
    SBMINFramework for more information.
    """
    def __init__(self, nlp, TR, TrSolver, **kwargs):

        qn = kwargs.get('quasi_newton', 'LBFGS')
        SBMINFramework.__init__(self, nlp, TR, TrSolver, **kwargs)
        self.lbfgs = eval(qn+'(nlp.n, npairs=100, scaling=False)')
        self.save_g = True

    def hprod(self, x, z, v, **kwargs):
        return self.lbfgs.matvec(v)

    def PostIteration(self, **kwargs):
        """
        Update the limited-memory quasi-Newton Hessian by appending
        the most recent (s,y) pair to it and possibly discarding the oldest one
        if all the memory has been used.
        """
        # Quasi-Newton approximation update on *successful* iterations
        if self.step_status == 'Acc' or self.step_status == 'N-Y Acc':
            s = self.true_step.copy()
            y = self.g - self.g_old
            yd = y.copy()
            self.lbfgs.store(s, y, yd)


class SBMINPartialLqnFramework(SBMINFramework):
    """
    Class SBMINPartialLqnFramework is a subclass of SBMINFramework. The method
    is based on a trust-region-based algorithm for nonlinear box constrained
    programming.
    The only difference is that a limited-memory Quasi Newton Hessian
    approximation is used and maintained along the iterations. Unlike the
    SBMINLqnFramework class, limited-memory matrix does not approximate the
    first order term in the Hessian, i.e. not the pJ'J term.
    """
    def __init__(self, nlp, TR, TrSolver, **kwargs):

        SBMINFramework.__init__(self, nlp, TR, TrSolver, **kwargs)
        self.save_lg = True

    def PostIteration(self, **kwargs):
        """
        This method updates the limited-memory quasi-Newton Hessian by
        appending the most recent (s,y) pair to it and possibly discarding the
        oldest one if all the memory has been used.
        """
        # Quasi-Newton approximation update on *successful* iterations
        if self.step_status == 'Acc' or self.step_status == 'N-Y Acc':
            s = self.x - self.x_old
            y = self.lg - self.lg_old
            self.nlp.hupdate(s, y)
        elif self.update_on_rejected_step:
            s = self.solver.step
            y = self.nlp.dual_feasibility(self.x_old + s) - self.lg_old
            self.nlp.hupdate(s, y)


class SBMINStructuredLqnFramework(SBMINFramework):
    """
    Class SBMINPartialLqnFramework is a subclass of SBMINFramework. The method
    is based on a trust-region-based algorithm for nonlinear box constrained
    programming.
    The only difference is that a limited-memory Quasi Newton Hessian
    approximation is used and maintained along the iterations. Unlike the
    SBMINLqnFramework class, limited-memory matrix does not approximate the
    first order term in the Hessian, i.e. not the pJ'J term.
    """
    def __init__(self, nlp, TR, TrSolver, **kwargs):

        SBMINFramework.__init__(self, nlp, TR, TrSolver, **kwargs)
        self.save_lg = True

    def PostIteration(self, **kwargs):
        """
        This method updates the limited-memory quasi-Newton Hessian by
        appending the most recent (s,y) pair to it and possibly discarding the
        oldest one if all the memory has been used.
        """
        # Quasi-Newton approximation update on *successful* iterations
        if self.step_status == 'Acc' or self.step_status == 'N-Y Acc':
            s = self.x - self.x_old
            Jx = self.nlp.nlp.jac(self.x)
            Jx_old = self.nlp.nlp.jac(self.x_old)
            consx = self.nlp.nlp.cons(self.x)
            gx = self.nlp.nlp.grad(self.x)
            gx_old = self.nlp.nlp.grad(self.x_old)
            pi = self.nlp.pi
            rho = self.nlp.rho
            dc = -pi + rho * consx
            ydB = gx - gx_old + Jx.T * dc - Jx_old.T * dc
            yB = self.g - self.g_old
            self.nlp.update(s, yB, ydB)
        elif self.update_on_rejected_step:
            s = self.solver.step
            Jx = self.nlp.nlp.jac(self.x_old + s)
            Jx_old = self.nlp.nlp.jac(self.x_old)
            consx = self.nlp.nlp.cons(self.x_old + s)
            gx = self.nlp.nlp.grad(self.x_old + s)
            gx_old = self.nlp.nlp.grad(self.x_old)
            pi = self.nlp.pi
            rho = self.nlp.rho
            dc = -pi + rho * consx
            ydB = gx - gx_old + Jx.T * dc - Jx_old.T * dc
            yB = self.nlp.grad(self.x_old+s) - self.g_old
            self.nlp.update(s, yB, ydB)


class TrustBQPModel(QPModel):
    """
    Class for defining a Model to pass to BQP solver:
                min     m(xk + s) = g's + 1/2 s'Hs
                s.t.       l <= xk + s <= u
                           || s ||_∞  <= delta

    where `g` is the gradient evaluated at xk
    and `H` is  the Hessian evaluated at xk.
    """

    def __init__(self, nlp, xk, delta, hprod, **kwargs):

        Lvar = np.maximum(nlp.Lvar - xk, -delta)
        Uvar = np.minimum(nlp.Uvar - xk, delta)

        self.nlp = nlp
        self.x0 = np.zeros(self.nlp.n)
        self.xk = xk.copy()
        self.delta = delta
        gk = kwargs.get('gk', None)
        if gk is None:
            gk = self.nlp.grad(self.xk)

        self._hprod = hprod

        Hk = LinearOperator(nlp.n, nlp.n, symmetric=True,
                            matvec=lambda u: self._hprod(self.xk, None, u))

        QPModel.__init__(self, gk, Hk, name='TrustRegionSubproblem',
                         Lvar=Lvar, Uvar=Uvar)

        self._x = -np.infty*np.ones(self.n)
        self._Hx = -np.infty*np.ones(self.n)

    def obj(self, x):
        if not (self._x == x).all():
            self._x = x.copy()
            self._Hx = None
        if self._Hx is None:
            self._Hx = self.H*x
            self.Hprod += 1
        cHx = self._Hx.copy()
        cHx *= 0.5
        cHx += self.c
        return np.dot(cHx, x)

    def grad(self, x):
        if not (self._x == x).all():
            self._x = x.copy()
            self._Hx = None
        if self._Hx is None:
            self._Hx = self.H*x
            self.Hprod += 1
        Hx = self._Hx.copy()
        Hx += self.c
        return Hx

    def hess(self, x, z):
        return self.H

    def hprod(self, x, z, p):
        if not (self._x == p).all():
            self._x = p.copy()
            self._Hx = None
        if self._Hx is None:
            self._Hx = self.H*p
            self.Hprod += 1
        return self._Hx

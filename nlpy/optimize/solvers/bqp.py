"""
This module implements a matrix-free active-set method for the
bound-constrained quadratic program

    minimize  g'x + 1/2 x'Hx  subject to l <= x <= u,

where l and u define a (possibly unbounded) box. The method
implemented is that of More and Toraldo described in

    J. J. More and G. Toraldo, On the solution of large
    quadratic programming problems with bound constraints,
    SIAM Journal on Optimization, 1(1), pp. 93-113, 1991.
"""

from nlpy.krylov.pcg import TruncatedCG
from pykrylov.linop import LinearOperator
from pykrylov.linop import SymmetricallyReducedLinearOperator as ReducedHessian
from nlpy.tools.utils import identical, where, NullHandler
from nlpy.tools.exceptions import InfeasibleError, \
                                  UserExitRequest, \
                                  GeneralConstraintsError
from numpy import setdiff1d
import numpy as np
import logging


__docformat__ = 'restructuredtext'


class SufficientDecreaseCG(TruncatedCG):
    """
    An implementation of the conjugate-gradient algorithm with
    a sufficient decrease stopping condition.

    :keywords:
        :cg_reltol: a relative stopping tolerance based on the decrease
                    of the quadratic objective function. The test is
                    triggered if, at iteration k,

                    q{k-1} - qk <= cg_reltol * max { q{j-1} - qj | j < k}

                    where qk is the value q(xk) of the quadratic objective
                    at the iterate xk.

    See the documentation of TruncatedCG for more information.

    Exit when either :
        - norm of gradient is less than eps_g times the initial gradient
        - stalling
        - sufficient decrease
    """

    def __init__(self, g, H, **kwargs):
        TruncatedCG.__init__(self, g, H, **kwargs)
        self.name = 'Suff-CG'
        self.qOld = 0.0   # Initial value of quadratic objective.
        self.best_decrease = 0
        self.decrease_old = 0
        self.cg_reltol = kwargs.get('cg_reltol', 0.1)
        self.detect_stalling = kwargs.get('detect_stalling', True)
        self.g0 = np.linalg.norm(g)

    def post_iteration(self):
        """
        Implement the sufficient decrease stopping condition. This test
        costs one dot product, five products between scalars and two
        additions of scalars.
        """
        decrease = self.qOld - self.qval
        self.log.debug('Current / best decrease : %7.1e / %7.1e' %
                       (decrease, self.best_decrease))

        if self.qval <= -1e+25:
            raise UserExitRequest

        if self.iter != 1:
            if decrease >= 100 * self.decrease_old:
                msg = 'CG stops because of sufficient decrease'
                self.log.debug(msg)
                raise UserExitRequest

        self.decrease_old = decrease
        self.qOld = self.qval

        # TODO : remove detect_stalling.
        if self.detect_stalling:
            if decrease <= self.cg_reltol * self.best_decrease:
                self.log.debug('CG stops for lack of substantial progress')
                raise UserExitRequest
            else:
                self.best_decrease = max(self.best_decrease, decrease)
        return None


class BQP(object):
    """
    A matrix-free active-set method for the bound-constrained quadratic
    program. May be used to solve trust-region subproblems in infinity
    norm.
    """

    def __init__(self, qp, **kwargs):

        if qp.m != 0:
            raise GeneralConstraintsError('QP has general constraints')

        self.qp = qp
        self.Lvar = qp.Lvar
        self.Uvar = qp.Uvar
        self.H = LinearOperator(qp.n, qp.n,
                                lambda u: self.qp.hprod(self.qp.x0, None, u),
                                symmetric=True)

        # Relative stopping tolerance in projected gradient iterations.
        self.pgrad_reltol = 0.25

        # Relative stopping tolerance in conjugate gradient iterations.
        self.cg_reltol = 0.1

        # Armijo-style linesearch parameter.
        self.armijo_factor = 1.0e-2

        self.cgiter = 0   # Total number of CG iterations.

        self.verbose = kwargs.get('verbose', True)
        self.hformat = '          %-5s  %9s  %8s  %5s'
        self.header  = self.hformat % ('Iter', 'q(x)', '|pg(x)|', 'cg')
        self.hlen    = len(self.header)
        self.hline   = '          ' + '-' * self.hlen
        self.format  = '          %-5d  %9.2e  %8.2e  %5d'
        self.format0 = '          %-5d  %9.2e  %8.2e  %5s'

        # Create a logger for solver.
        self.log = logging.getLogger('nlpy.bqp')
        try:
            self.log.addHandler(logging.NullHandler())  # For Python 2.7.x
        except:
            self.log.addHandler(NullHandler())  # For Python 2.6.x (and older?)

    def check_feasible(self, x):
        """
        Safety function. Check that x is feasible with respect to the
        bound constraints.
        """
        if np.any((x < self.Lvar) | (x > self.Uvar)):
            raise InfeasibleError
        return None

    def pgrad(self, x, g=None, active_set=None, check_feasible=True):
        """
        Compute the projected gradient of the quadratic at x.
        If the actual gradient is known, it should be passed using the
        `g` keyword.
        If the active set at x0 is known, it should be passed using the
        `active_set` keyword.
        Optionally, check that x is feasible.

        The projected gradient pg is defined componentwise as follows:

        pg[i] = min(g[i],0)  if x[i] is at its lower bound,
        pg[i] = max(g[i],0)  if x[i] is at its upper bound,
        pg[i] = g[i]         otherwise.
        """
        if check_feasible: self.check_feasible(x)
        if g is None: g = self.qp.grad(x)

        if active_set is None: active_set = self.get_active_set(x)
        lower, upper = active_set

        pg = g.copy()
        pg[lower] = np.minimum(g[lower], 0)
        pg[upper] = np.maximum(g[upper], 0)
        return pg

    def project(self, x):
        "Project a given x into the bounds in Euclidian norm."
        return np.minimum(self.qp.Uvar, np.maximum(self.qp.Lvar, x))

    def get_active_set(self, x, check_feasible=True):
        """
        Return the set of active constraints at x.
        Optionally, check that x is feasible.

        Returns the couple (lower,upper) containing the indices of variables
        that are at their lower and upper bound, respectively.
        """
        lower_active = where(x == self.Lvar)
        upper_active = where(x == self.Uvar)
        return(lower_active, upper_active)

    def breakpoints(self, x, d):
        """
        Find the smallest and largest breakpoints on the half line x + t*d.
        We assume that x is feasible. Return the smallest and largest t such
        that x + t*d lies on the boundary.
        """
        pos = where((d > 0) & (x < self.Uvar))  # Hit the upper bound.
        neg = where((d < 0) & (x > self.Lvar))  # Hit the lower bound.
        npos = len(pos)
        nneg = len(neg)
        if npos + nneg == 0:                    # No breakpoint.
            return (np.inf, np.inf)
        bk_min = np.inf
        bk_max = 0
        if npos > 0:
            pos_steps = (self.Uvar[pos] - x[pos]) / d[pos]
            bk_min = min(bk_min, np.min(pos_steps))
            bk_max = max(bk_max, np.max(pos_steps))
        if nneg > 0:
            neg_steps = (self.Lvar[neg] - x[neg]) / d[neg]
            bk_min = min(bk_min, np.min(neg_steps))
            bk_max = max(bk_max, np.max(neg_steps))
        bk_max = max(bk_max, bk_min)
        self.log.debug('Nearest  breakpoint: %7.1e' % bk_min)
        self.log.debug('Farthest breakpoint: %7.1e' % bk_max)
        return (bk_min, bk_max)

    def projected_linesearch(self, x, g, d, qval, step=1.0, **kwargs):
        """
        Perform an Armijo-like projected linesearch in the direction d.
        Here, x is the current iterate, g is the gradient at x,
        d is the search direction, qval is q(x) and
        step is the initial steplength.
        """

        check_feasible = kwargs.get('check_feasible', True)
        if check_feasible:
            self.check_feasible(x)

        # Check for local optimality.
        tol = 1.0e-6 * np.linalg.norm(x)

        if np.linalg.norm(self.project(x + step*d) - x) < tol:
            return (x, qval, step)

        if np.dot(g, d) >= 0:
            raise ValueError('Not a descent direction.')

        qp = self.qp
        factor = self.armijo_factor

        self.log.debug('Projected linesearch with initial q = %7.1e' % qval)

        # Obtain stepsize to nearest and farthest breakpoints.
        bk_min, bk_max = self.breakpoints(x, d)

        if kwargs.get('use_bk_min', False):
            step = bk_min

        xps = self.project(x + step * d)
        q_xps = qp.obj(xps)
        slope = np.dot(g, xps - x)

        if slope >= 0:   # May happen after a projection.
            step = min(step, bk_min)
            xps = self.project(x + step * d)
            q_xps = qp.obj(xps)
            slope = np.dot(g, xps - x)

        decrease = (q_xps < qval + factor * slope)

        if not decrease:
            # Perform projected Armijo linesearch in order to reduce the step
            # until a successful step is found.
            while not decrease and step >= 1.0e-8:
                step /= 6
                xps = self.project(x + step * d)
                q_xps = qp.obj(xps)
                self.log.debug('  Backtracking with step = %7.1e q = %7.1e' %
                               (step, q_xps))
                slope = np.dot(g, xps - x)
                decrease = (q_xps < qval + factor * slope)
        else:
            # The initial step yields sufficient decrease. See if we can
            # find a larger step with larger decrease.
            if step < bk_max:
                increase = True
                x_ok = xps.copy()  # Most recent iterate satisfying Armijo.
                q_ok = q_xps
                q_prev = q_xps
                while increase and step <= bk_max:
                    step *= 6
                    xps = self.project(x + step * d)
                    q_xps = qp.obj(xps)
                    self.log.debug('  Extrapolating with step = %7.1e' +
                                   ' q = %7.1e' % (step, q_xps))
                    slope = np.dot(g, xps - x)
                    increase = \
                        slope < 0 and \
                        (q_xps < qval + factor * slope) and \
                        q_xps <= q_prev
                    if increase:
                        x_ok = xps.copy()
                        q_ok = q_xps
                        q_prev = q_xps
                xps = x_ok.copy()
                q_xps = q_ok
        q_xps = qp.obj(xps)

        if q_xps > qval:
            xps = x.copy()
            q_xps = qval
        self.log.debug('Projected linesearch ends with q = %7.1e' % q_xps)
        return (xps, q_xps, step)

    def projected_gradient(self, x0,
                           g=None, active_set=None, qval=None, **kwargs):
        """
        Perform a sequence of projected gradient steps starting from x0.
        If the actual gradient at x is known, it should be passed using the
        `g` keyword.
        If the active set at x0 is known, it should be passed using the
        `active_set` keyword.
        If the value of the quadratic objective at x0 is known, it should
        be passed using the `qval` keyword.

        Return (x,(lower,upper)) where x is an updated iterate that satisfies
        a sufficient decrease condition or at which the active set, given by
        (lower,upper), settled down.
        """
        maxiter = kwargs.get('maxiter', 10)
        check_feasible = kwargs.get('check_feasible', True)

        if check_feasible:
            self.check_feasible(x0)

        if g is None:
            g = self.qp.grad(x0)

        if qval is None:
            qval = self.qp.obj(x0)

        if active_set is None:
            active_set = self.get_active_set(x0)
        lower, upper = active_set

        self.log.debug('Entering projected gradient with q = %7.1e' % qval)

        x = x0.copy()
        settled_down = False
        sufficient_decrease = False
        best_decrease = 0
        iter = 0

        while not settled_down and not sufficient_decrease and iter < maxiter:

            iter += 1
            qOld = qval
            # TODO: Use appropriate initial steplength.
            (x, qval, step) = self.projected_linesearch(x, g, -g, qval, step=1)
            g = self.qp.grad(x)

            # Check decrease in objective.
            decrease = qOld - qval
            msg  = 'Current / best decrease in projected gradient :'
            msg += ' %7.1e / %7.1e' % (decrease, best_decrease)
            self.log.debug(msg)

            sufficient_decrease = decrease <= self.pgrad_reltol * best_decrease
            best_decrease = max(best_decrease, decrease)

            # Check active set at updated iterate.
            lowerTrial, upperTrial = self.get_active_set(x)
            settled_down = identical(lower, lowerTrial) and \
                           identical(upper, upperTrial)
            lower, upper = lowerTrial, upperTrial

        return (x, (lower, upper))

    def to_boundary(self, x, d, free_vars, **kwargs):
        """
        Given vectors `x` and `d` and some bounds on x,
        return a positive alpha such that

          `x + alpha * d = boundary
        """
        check_feasible = kwargs.get('check_feasible', True)
        if check_feasible:
            self.check_feasible(x)

        # Obtain stepsize to nearest and farthest breakpoints.
        bk_min, _ = self.breakpoints(x, d)

        x = self.project(x + bk_min * d)  # To avoid tiny rounding errors.

        # Do another projected gradient update
        (x, (lower, upper)) = self.projected_gradient(x)

        return (x, (lower, upper))

    def solve(self, **kwargs):

        # Shortcuts for convenience.
        qp = self.qp
        n = qp.n
        maxiter = kwargs.get('maxiter', 20 * n)
        abstol = kwargs.get('abstol', 1.0e-7)
        reltol = kwargs.get('reltol', 1.0e-5)

        # Compute initial data.

        self.log.debug('q before initial x projection = %7.1e' % qp.obj(qp.x0))
        x = self.project(qp.x0)
        self.log.debug('q after  initial x projection = %7.1e' % qp.obj(x))
        lower, upper = self.get_active_set(x)
        iter = 0

        # Compute stopping tolerance.
        g = qp.grad(x)
        pg = self.pgrad(x, g=g, active_set=(lower, upper))
        pgNorm = np.linalg.norm(pg)

        stoptol = reltol * pgNorm + abstol
        self.log.debug('Main loop with iter=%d and pgNorm=%g' % (iter, pgNorm))

        exitStalling = exitOptimal = exitIter = False

        # Print out header and initial log.
        self.log.info(self.hline)
        self.log.info(self.header)
        self.log.info(self.hline)
        self.log.info(self.format0 % (iter, 0.0, pgNorm, ''))

        while not (exitOptimal or exitIter or exitStalling):

            cgiter_1 = 0
            cgiter_2 = 0

            x_old = x.copy()
            iter += 1

            if iter >= maxiter:
                exitIter = True
                continue

            # Projected-gradient phase: determine next working set.
            (x, (lower, upper)) = \
                self.projected_gradient(x, g=g, active_set=(lower, upper))
            g = qp.grad(x)
            qval = qp.obj(x)
            self.log.debug('q after projected gradient = %8.2g' % qval)
            pg = self.pgrad(x, g=g, active_set=(lower, upper))
            pgNorm = np.linalg.norm(pg)

            if pgNorm <= stoptol:
                exitOptimal = True
                self.log.debug('Exiting because residual is small')
                self.log.info(self.format % (iter, qval,
                              pgNorm, 0))
                continue

            # Conjugate gradient phase: explore current face.

            # 1. Obtain indices of the free variables.
            fixed_vars = np.concatenate((lower, upper))
            free_vars = setdiff1d(np.arange(n, dtype=np.int), fixed_vars)

            # 2. Construct reduced QP.
            self.log.debug('Starting CG on current face.')

            ZHZ = ReducedHessian(self.H, free_vars)
            Zg  = g[free_vars]

            cg = SufficientDecreaseCG(Zg, ZHZ, detect_stalling=True)
            try:
                cg.Solve(abstol=1.0e-5, reltol=1.0e-3)
            except UserExitRequest:
                msg  = 'CG is no longer making substantial progress'
                msg += ' (%d its)' % cg.niter
                self.log.debug(msg)

            # At this point, CG returned from a clean user exit or
            # because its original stopping test was triggered.
            msg = 'CG stops (%d its, status = %s)' % (cg.niter, cg.status)
            self.log.debug(msg)
            cgiter_1 = cg.niter

            # 3. Expand search direction.
            d = np.zeros(n)
            d[free_vars] = cg.step

            if cg.infDescent and cg.step.size != 0 and cg.dir.size != 0:
                msg  = 'iter :%d  Negative curvature detected' % iter
                msg += ' (%d its)' % cg.niter
                self.log.debug(msg)

                nc_dir = np.zeros(n)
                nc_dir[free_vars] = cg.dir
                (x, (lower, upper)) = self.to_boundary(x, nc_dir, free_vars)
                qval = qp.obj(x)
            else:
                # 4. Update x using projected linesearch with initial step=1.
                x, qval, step = self.projected_linesearch(x, g, d, qval)

                self.log.debug('q after first CG pass = %8.2g' % qval)

            g = qp.grad(x)
            pg = self.pgrad(x, g=g)
            pgNorm = np.linalg.norm(pg)

            if pgNorm <= stoptol:
                exitOptimal = True
                self.log.debug('Exiting because residual is small')
                self.log.info(self.format % (iter, qval,
                              pgNorm, cg.niter))
                self.cgiter += cg.niter
                continue

            # Compare active set to binding set.
            lower, upper = self.get_active_set(x)

            if np.all(g[lower] >= 0) and np.all(g[upper] <= 0):
                # The active set agrees with the binding set.
                # Continue CG iterations with tighter tolerance.
                # This currently incurs a little bit of extra work
                # by instantiating a new CG object.
                self.log.debug('Active set = binding set. Continuing CG.')

                fixed_vars = np.concatenate((lower, upper))
                free_vars = setdiff1d(np.arange(n, dtype=np.int), fixed_vars)
                ZHZ = ReducedHessian(self.H, free_vars)
                Zg  = g[free_vars]
                cg = SufficientDecreaseCG(Zg, ZHZ, detect_stalling=True)
                cg.Solve(absol=1.0e-6, reltol=1.0e-4)

                msg = 'CG stops (%d its, status = %s)' % (cg.niter, cg.status)
                self.log.debug(msg)
                cgiter_2 = cg.niter

                d = np.zeros(n)
                d[free_vars] = cg.step

                if cg.infDescent and cg.step.size != 0:
                    msg  = 'iter :%d  Negative curvature detected' % iter
                    msg += ' (%d its)' % cg.niter
                    self.log.debug(msg)

                    nc_dir = np.zeros(n)
                    nc_dir[free_vars] = cg.dir
                    (x, (lower, upper)) = self.to_boundary(x, nc_dir,
                                                           free_vars)
                    qval = qp.obj(x)
                else:
                    # 4. Update x using projected linesearch with step=1.
                    x, qval, step = self.projected_linesearch(x, g, d, qval)

                self.log.debug('q after second CG pass = %8.2g' % qval)

                g = qp.grad(x)
                pg = self.pgrad(x, g=g, active_set=(lower, upper))
                pgNorm = np.linalg.norm(pg)

                # Exit if second CG pass results in optimality
                if pgNorm <= stoptol:
                    self.log.debug('Exiting because residual is small')
                    exitOptimal = True

            cgiter = cgiter_1 + cgiter_2  # Total CG iters in this BQP iter.
            self.cgiter += cgiter         # Total CG iters so far.
            exitStalling = (np.linalg.norm(x-x_old)) <= 1e-18
            self.log.info(self.format % (iter, qval, pgNorm, cgiter))

        self.exitOptimal = exitOptimal
        self.exitIter = exitIter
        self.exitStalling = exitStalling

        if self.exitOptimal:
            self.status = 0
        elif self.exitIter:
            self.status = -1
        elif self.exitStalling:
            self.status = -2
        else:
            self.status = -5

        self.niter = iter
        self.x = x
        self.qval = qval
        self.lower = lower
        self.upper = upper
        return


if __name__ == '__main__':
    import sys
    from nlpy.model import AmplModel

    qp = AmplModel(sys.argv[1])
    bqp = BQP(qp)
    bqp.solve(maxiter=50, stoptol=1.0e-8)
    print 'optimal = ', repr(bqp.exitOptimal)
    print 'niter = ', bqp.niter
    print 'solution: ', bqp.x
    print 'objective value: ', bqp.qval
    print 'vars on lower bnd: ', bqp.lower
    print 'vars on upper bnd: ', bqp.upper

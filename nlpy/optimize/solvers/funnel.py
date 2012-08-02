# An implementation of the trust-funnel method.

from nlpy.model import NLPModel, QPModel
from nlpy.krylov.linop import PysparseLinearOperator, SimpleLinearOperator
from nlpy.krylov import ProjectedCG, LSTRFramework
from nlpy.optimize.solvers.lsqr import LSQRFramework
from nlpy.optimize.solvers.ldfp import LDFP, StructuredLDFP
from nlpy.tools.timing import cputime

from pysparse.sparse.pysparseMatrix import PysparseMatrix

from math import sqrt
import numpy as np
import logging

__docformat__ = 'restructuredtext'


class Funnel(object):
    """
    A trust-funnel framework for equality-constrained optimization.
    D. Orban and N. I. M. Gould, from N. I. M. Gould's original Matlab
    implementation.
    """

    def __init__(self, nlp, **kwargs):
        """
        Instantiate a trust-funnel framework for a given equality-constrained
        problem.

        :keywords:
            :nlp: `NLPModel` instance.
            :atol: Absolute stopping tolerance
            :Delta_f: Initial trust-region radius for the objective model
            :Delta_c: Initial trust-region radius for the constraints model
            :iterative_solver: solver used to find steps (0=gltr, 1=gmres)
            :maxit: maximum number of iterations allowed
            :stop_p: required accuracy for ||c||
            :stop_d: required accuracy for ||g+J'y||
            :maxit_refine: maximum number of iterative refinements per solve
        """

        # Bail out if nlp is not a NLPModel instance.
        if not isinstance(nlp, NLPModel):
            msg = 'Input problem must be a NLPModel instance.'
            raise ValueError(msg)

        # Bail out if problem has bounds or general inequality constraints.
        if nlp.nlowerB + nlp.nupperB + nlp.nrangeB:
            msg = 'Problem has bound constraints.'
            raise ValueError(msg)

        if nlp.nlowerC + nlp.nupperC + nlp.nrangeC:
            msg = 'Problem has general inequality constraints.'
            raise ValueError(msg)

        self.nlp = nlp
        self.x = nlp.x0.copy()
        self.f0 = self.f = nlp.obj(self.x)
        self.pResid = self.dResid = None
        self.optimal = False

        self.niter = 0
        self.tsolve = 0.0

        # Set default options.
        self.atol = kwargs.get('atol', 1.0e-8)
        self.Delta_f_0 = self.Delta_f = kwargs.get('Delta_f', 1.0)
        self.Delta_c_0 = self.Delta_c = kwargs.get('Delta_c', 1.0)
        self.iterative_solver = kwargs.get('iterative_solver', 0)
        self.maxit = kwargs.get('maxit', 200)
        self.stop_p = kwargs.get('stop_p', nlp.stop_p)
        self.stop_d = kwargs.get('stop_d', nlp.stop_d)
        self.maxit_refine = kwargs.get('maxit_refine', 0)

        self.hdr_fmt = '%4s   %2s %9s %8s %8s %8s %8s %6s %6s'
        self.hdr = self.hdr_fmt % ('iter', 'NT', 'f', '|c|', '|g+J''y|',
                                   'Delta_f', 'Delta_c', 'thetaMax', 'CG')
        self.linefmt1='%4d%c%c %c%c %9.2e %8.2e %8.2e %8.2e %8.2e %8.2e %6.0f'
        self.linefmt='%4d%c%c %c%c %9.2e %8.2e %8.2e %8.2e %8.2e %8.2e %6.0f'

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'funnel.solver')
        self.log = logging.getLogger(logger_name)
        #self.log.addHandler(logging.NullHandler())

        # Display initial information.
        self.display_basic_info()

        return

    def display_basic_info(self):
        "Display basic info about current problem."
        nlp = self.nlp
        self.log.info('Problem %s' % nlp.name)
        self.log.info('Linear  equality constraints: %d' % nlp.nlin)
        self.log.info('General equality constraints: %d' % (nlp.nequalC - nlp.nlin))
        self.log.info('Primal stopping tolerance: %6.1e' % self.stop_p)
        self.log.info('Dual   stopping tolerance: %6.1e' % self.stop_d)
        self.log.info('Key:')
        self.log.info('iter type   : [f] or [c]')
        self.log.info('overall step: [s]uccessful, [v]ery successful, [u]nsuccessful or (2)nd-order')
        self.log.info('normal  step: [0] (none), [r]esidual small, [b]oundary or [?] (other)')
        self.log.info('tangent step: [0] (none), [r]esidual small, [b]oundary, [-] neg. curvature, [>] max iter or [?] (other)')
        return

    def cons(self, x):
        """
        Return the value of the equality constraints evaluated at x and
        reformulated so that the right-hand side if zero, i.e., the original
        constraints c(x) = c0 are reformulated as c(x) - c0 = 0.
        """
        return (self.nlp.cons(x) - self.nlp.Lcon)

    def jac(self, x):
        """
        Return the Jacobian matrix of the equality constraints at x as a PysparseMatrix.
        """
        _J = self.nlp.jac(x, store_zeros=True)  # Keep explicit zeros.
        return PysparseMatrix(matrix=_J)

    def hprod(self, x, y, v):
        """
        Return the product of the Hessian of the Lagrangian evaluated at (x,y)
        with the vector v. By default, this method uses the `hprod` method of the `nlp`
        attribute.
        Subclass to implement different matrix-vector products.
        """
        return self.nlp.hprod(x,y,v)

    def forcing(self, k, val):
        "Return forcing term number `k`."
        if k == 1:
            return 1.0e-2 * min(1, min(abs(val), val*val))
        if k == 2:
            return 1.0e-2 * min(1, min(abs(val), val*val))
        return 1.0e-2 * min(1, min(abs(val), val*val))

    def post_iteration(self, **kwargs):
        pass

    def lsq(self, A, b, reg=0.0, radius=None, **kwargs):
        """
        Solve the linear least-squares problem in the variable x

            minimize |Ax - b| + reg * |x|

        in the Euclidian norm with LSQR. Optionally, x may be
        subject to a Euclidian-norm trust-region constraint
        |x| <= radius. This function returns (x, xNorm, status).
        """
        LSQR = LSQRFramework(A)
        LSQR.solve(b, radius=radius, damp=reg, show=False)
        return (LSQR.x, LSQR.xnorm, LSQR.status)

    def nyf(self, x, f, fTrial, g, step, bkmax=5, armijo=1.0-4):
        """
        Perform a simple backtracking linesearch on the objective function
        from `x` along `step`. Here, `f` and `fTrial` are the objective value
        at `x` and `x + step`, respectively, and `g` is the gradient of the
        objective at `x`.

        Return (x, f, steplength), where `x + steplength * step` satisfies
        the Armijo condition and `f` is the objective value at this new point.
        """
        slope = np.dot(g, step)
        bk = 0 ; alpha = 1.0 ; xTrial = x
        while bk < bkmax and \
                fTrial >= f + armijo * alpha * slope:
            bk = bk + 1
            alpha /= 1.2
            xTrial = x + alpha * step
            fTrial = self.nlp.obj(xTrial)
        return (xTrial, fTrial, alpha)

    def nyc(self, x, theta, thetaTrial, c,
            dtheta, step, bkmax=5, armijo=1.0-4):
        """
        Perform a simple backtracking linesearch on the infeasibility measure
        theta(x) = 1/2 * |c(x)|^2 from `x` along `step`. Here, `theta` and
        `thetaTrial` are the infeasibility at `x` and `x + step`, respectively,
        `c` is the vector of constraints at `x`,
        and `dtheta` is the gradient of the infeasibility measure at `x`.

        Note that grad theta(x) = J(x).T * c(x).

        Return (x, c, theta, steplength), where `x + steplength + step`
        satisifes the Armijo condition, and `c` and `theta` are the vector of
        constraints and the infeasibility at this new point, respectively.
        """
        slope = np.dot(dtheta, step)
        bk = 0 ; alpha = 1.0 ; xTrial = x ; cTrial = c
        while bk < bkmax and \
                thetaTrial >= theta + armijo * alpha * slope:
            bk = bk + 1
            alpha /= 1.2
            xTrial = x + alpha * step
            cTrial = self.nlp.cons(xTrial)
            thetaTrial = 0.5 * np.dot(cTrial, cTrial)
        return (xTrial, cTrial, thetaTrial, alpha)

    def solve(self, **kwargs):
        """
        Solve current problem with trust-funnel framework.

        :keywords:
            :ny: Enable Nocedal-Yuan backtracking linesearch.

        :returns:
            This method sets the following members of the instance:
            :f: Final objective value
            :optimal: Flag indicating whether normal stopping conditions were
                      attained
            :pResid: Final primal residual
            :dResid: Final dual residual
            :niter: Total number of iterations
            :tsolve: Solve time.

        """

        ny = kwargs.get('ny', True)
        reg = kwargs.get('reg', 0.0)
        #ny = False

        tsolve = cputime()

        # Set some shortcuts.
        nlp = self.nlp
        n = nlp.n
        m = nlp.m
        x = self.x
        f = self.f
        c = self.cons(x)
        y = nlp.pi0.copy()
        self.it = 0

        # Initialize some constants.
        kappa_n = 1.0e+2   # Factor of pNorm in normal step TR radius.
        kappa_b = 0.99     # Fraction of TR to compute tangential step.
        kappa_delta = 0.1  # Progress factor to compute tangential step.

        # Trust-region parameters.
        eta_1 = 1.0e-5
        eta_2 = 0.95
        eta_3 = 0.5
        gamma_1 = 0.25
        gamma_3 = 2.5

        kappa_tx1 = 0.9 # Factor of theta_max in max acceptable infeasibility.
        kappa_tx2 = 0.5 # Convex combination factor of theta and thetaTrial.

        # Compute constraint violation.
        theta = 0.5 * np.dot(c,c)

        # Set initial funnel radius.
        kappa_ca = 1.0e+3    # Max initial funnel radius.
        kappa_cr = 2.0       # Infeasibility tolerance factor.
        theta_max = max(kappa_ca, kappa_cr * theta)

        # Evaluate first-order derivatives.
        g = nlp.grad(x)
        J = self.jac(x)
        Jop = PysparseLinearOperator(J)

        # Initial radius for f- and c-iterations.
        Delta_f = max(self.Delta_f, .1 * np.linalg.norm(g))
        Delta_c = max(self.Delta_c, .1 * sqrt(2*theta))

        # Reset initial multipliers to least-squares estimates by
        # approximately solving:
        #   [ I   J' ] [ w ]   [ -g ]
        #   [ J   0  ] [ y ] = [  0 ].
        # This is equivalent to solving
        #   minimize |g + J'y|.
        if m > 0:
            y, _, _ = self.lsq(Jop.T, -g, reg=reg)

        pNorm = cNorm = 0
        if m > 0:
            pNorm = np.linalg.norm(c);
            cNorm = np.linalg.norm(c, np.inf)
            grad_lag = g + Jop.T * y
        else:
            grad_lag = g.copy()
        dNorm = np.linalg.norm(grad_lag)/(1 + np.linalg.norm(y))

        # Display current info if requested.
        self.log.info(self.hdr)
        self.log.info(self.linefmt1 % (0, ' ', ' ', ' ', ' ', f, pNorm,
                                       dNorm, Delta_f, Delta_c, theta_max, 0))

        # Compute primal stopping tolerance.
        stop_p = max(self.atol, self.stop_p * pNorm)
        self.log.debug('pNorm = %8.2e, cNorm = %8.2e, dNorm = %8.e2' % (pNorm,cNorm,dNorm))

        optimal = (pNorm <= stop_p) and (dNorm <= self.stop_d)
        self.log.debug('optimal: %s' % repr(optimal))

        # Start of main iteration.
        while not optimal and (self.it < self.maxit):

            self.it += 1
            Delta = min(Delta_f, Delta_c)
            cgiter = 0

            # 1. Compute normal step as an (approximate) solution to
            #    minimize |c + J n|  subject to  |n| <= min(Delta_c, kN |c|).

            if self.it > 1 and \
                    pNorm <= stop_p and \
                    dNorm >= 1.0e+4 * self.stop_d:

                self.log.debug('Setting nStep=0 b/c need to work on optimality')
                nStep = np.zeros(n)
                nStepNorm = 0.0
                n_end = '0'
                m_xpn = 0       # Model value at x+n.

            else:

                nStep_max = min(Delta_c, kappa_n * pNorm)
                nStep, nStepNorm, lsq_status = self.lsq(Jop, -c,
                                                        radius=nStep_max,
                                                        reg=reg)

                if lsq_status == 'residual small':
                    n_end = 'r'
                elif lsq_status == 'trust-region boundary active':
                    n_end = 'b'
                else:
                    n_end = '?'

                # Evaluate the model of the obective after the normal step.
                _Hv = self.hprod(x, y, nStep) # H*nStep
                #m_xpn = np.dot(g, nStep) + 0.5 * np.dot(nStep, _Hv)
                m_xpn = qp.obj(nStep)

            self.log.debug('Normal step norm = %8.2e' % nStepNorm)
            self.log.debug('Model value: %9.2e' % m_xpn)

            # 2. Compute tangential step if normal step is not too long.

            if nStepNorm <= kappa_b * Delta:

                # 2.1. Compute Lagrange multiplier estimates and dual residuals
                #      by minimizing |(g + H n) + J'y|

                if nStepNorm == 0.0:
                    gN = g   # Note: this is just a pointer ; g will not be modified below.
                else:
                    gN = g + _Hv

                y_new, y_norm, _ = self.lsq(Jop.T, -gN, reg=reg)
                r = gN + Jop.T * y_new

                # Here Nick does iterative refinement to improve r and y_new...

                # Compute dual optimality measure.
                residNorm = np.linalg.norm(r)
                norm_gN = np.linalg.norm(gN)
                pi = 0.0
                if residNorm > 0:
                    pi = abs(np.dot(gN, r))/residNorm

                # 2.2. If the dual residuals are large, compute a suitable
                #      tangential step as a solution to:
                #      minimize    g't + 1/2 t' H t
                #      subject to  Jt = 0, |n+t| <= Delta.

                if pi > self.forcing(3, theta):

                    self.log.debug('Computing tStep...')
                    Delta_within = Delta - nStepNorm

                    Hop = SimpleLinearOperator(n, n,
                                               lambda v: self.hprod(x,y_new,v),
                                               symmetric=True)
                    qp = QPModel(gN, Hop, A=J.matrix if m > 0 else None)
                    #PPCG = ProjectedCG(gN, Hop,
                    #                   A=J.matrix if m > 0 else None,
                    #                   radius=Delta_within, dreg=reg)
                    PPCG = ProjectedCG(qp, radius=Delta_within, dreg=reg)
                    PPCG.Solve()
                    tStep = PPCG.step
                    tStepNorm = PPCG.stepNorm
                    cgiter = PPCG.iter

                    self.log.debug('|t| = %8.2e' % tStepNorm)

                    if PPCG.status == 'residual small':
                        t_end = 'r'
                    elif PPCG.onBoundary and not PPCG.infDescent:
                        t_end = 'b'
                    elif PPCG.infDescent:
                        t_end = '-'
                    elif PPCG.status == 'max iter':
                        t_end = '>'
                    else:
                        t_end = '?'

                    # Compute total step and model decrease.
                    step = nStep + tStep
                    stepNorm = np.linalg.norm(step)
                    #_Hv = self.hprod(x,y,step)    # y or y_new?
                    #m_xps = np.dot(g, step) + 0.5 * np.dot(step, _Hv)
                    m_xps = qp.obj(step)

                else:

                    self.log.debug('Setting tStep=0 b/c pi is sufficiently small')
                    tStepNorm = 0
                    t_end = '0'
                    step = nStep
                    stepNorm = nStepNorm
                    m_xps = m_xpn

                y = y_new

            else:

                # No need to compute a tangential step.
                self.log.debug('Setting tStep=0 b/c the normal step is too large')
                t_end = '0'
                y = np.zeros(m)
                tStepNorm = 0.0
                step = nStep
                stepNorm = nStepNorm
                m_xps = m_xpn

            self.log.debug('Model decrease = %9.2e' % m_xps)

            # Compute trial point and evaluate local data.
            xTrial = x + step
            fTrial = nlp.obj(xTrial)
            cTrial = self.cons(xTrial)
            thetaTrial = 0.5 * np.dot(cTrial, cTrial)
            delta_f = -m_xps           # Overall improvement in the model.
            delta_ft = m_xpn - m_xps   # Improvement due to tangential step.
            Jspc = c + Jop * step

            # Compute improvement in linearized feasibility.
            delta_feas = theta - 0.5 * np.dot(Jspc, Jspc)

            # Decide whether to consider the current iteration
            # an f- or a c-iteration.

            if tStepNorm > 0 and (delta_f >= self.forcing(2, theta)) and \
                    delta_f >= kappa_delta * delta_ft and \
                    thetaTrial <= theta_max:

                # Step 3. Consider that this is an f-iteration.
                it_type = 'f'

                # Decide whether trial point is accepted.
                ratio = (f - fTrial)/delta_f

                self.log.debug('f-iter ratio = %9.2e' % ratio)

                if ratio >= eta_1:    # Successful step.
                    suc = 's'
                    x = xTrial
                    f = fTrial
                    c = cTrial
                    self.step = step.copy()
                    theta = thetaTrial

                    # Decide whether to update f-trust-region radius.
                    if ratio >= eta_2:
                        suc = 'v'
                        Delta_f = min(max(Delta_f, gamma_3 * stepNorm),
                                      1.0e+10)

                    # Decide whether to update c-trust-region radius.
                    if thetaTrial < eta_3 * theta_max:
                        ns = nStepNorm if nStepNorm > 0 else Delta_c
                        Delta_c = min(max(Delta_c, gamma_3 * ns),
                                      1.0e+10)

                    self.log.debug('New Delta_f = %8.2e' % Delta_f)
                    self.log.debug('New Delta_c = %8.2e' % Delta_c)

                else:                 # Unsuccessful step (ratio < eta_1).

                    attempt_SOC = True
                    suc = 'u'

                    if attempt_SOC:

                        self.log.debug('  Attempting second-order correction')

                        # Attempt a second-order correction by solving
                        # minimize |cTrial + J n|  subject to  |n| <= Delta_c.
                        socStep, socStepNorm, socStatus = self.lsq(Jop,
                                                                   -cTrial,
                                                                   radius=Delta_c,
                                                                   reg=reg)

                        if socStatus != 'trust-region boundary active':

                            # Consider SOC step as candidate step.
                            xSoc = xTrial + socStep
                            fSoc = nlp.obj(xSoc) ; cSoc = self.cons(xSoc)
                            thetaSoc = 0.5 * np.dot(cSoc, cSoc)
                            ratio = (f - fSoc)/delta_f

                            # Decide whether to accept SOC step.
                            if ratio >= eta_1 and thetaSoc <= theta_max:
                                suc = '2'
                                x = xSoc
                                f = fSoc
                                c = cSoc
                                theta = thetaSoc
                                self.step = step + socStep
                            else:

                                # Backtracking linesearch a la Nocedal & Yuan.
                                # Abandon SOC step. Backtrack from x+step.
                                if ny:
                                    (x, f, alpha) = self.nyf(x, f, fTrial,
                                                             g, step)
                                    #g = nlp.grad(x)
                                    c = self.cons(x)
                                    theta = 0.5 * np.dot(c,c)
                                    self.step = step + alpha * socStep
                                    Delta_f = min(alpha, .8) * stepNorm
                                    suc = 'y'

                                else:
                                    Delta_f = gamma_1 * Delta_f

                        else:  # SOC step lies on boundary of trust region.
                            Delta_f = gamma_1 * Delta_f

                    else:  # SOC step not attempted.

                        # Backtracking linesearch a la Nocedal & Yuan.
                        if ny:
                            (x, f, alpha) = self.nyf(x, f, fTrial, g, step)
                            #g = nlp.grad(x)
                            c = self.cons(x)
                            theta = 0.5 * np.dot(c,c)
                            self.step = alpha * step
                            Delta_f = min(alpha, .8) * stepNorm
                            suc = 'y'
                        else:
                            Delta_f = gamma_1 * Delta_f


            else:

                # Step 4. Consider that this is a c-iteration.
                it_type = 'c'

                # Display information.
                self.log.debug('c-iteration because ')
                if tStepNorm == 0.0:
                    self.log.debug('|t|=0')
                if delta_f < self.forcing(2, theta):
                    self.log.debug('delta_f=%8.2e < forcing=%8.2e'\
                                    % (delta_f, self.forcing(2, theta)))
                if delta_f < kappa_delta * delta_ft:
                    self.log.debug('delta_f=%8.2e < frac * delta_ft=%8.2e' \
                                    % (delta_f, delta_ft))
                if thetaTrial > theta_max:
                    self.log.debug('thetaTrial=%8.2e > theta_max=%8.2e' \
                                    % (thetaTrial, theta_max))

                # Step 4.1. Check trial point for acceptability.
                if delta_feas < 0:
                    self.log.debug(' !!! Warning: delta_feas is negative !!!')

                ratio = (theta - thetaTrial + 1.0e-16)/(delta_feas + 1.0e-16)
                self.log.debug('c-iter ratio = %9.2e' % ratio)

                if ratio >= eta_1:         # Successful step.
                    x = xTrial
                    f = fTrial
                    c = cTrial
                    self.step = step.copy()
                    suc = 's'

                    # Step 4.2. Update Delta_c.
                    if ratio >= eta_2:     # Very successful step.
                        ns = nStepNorm if nStepNorm > 0 else Delta_c
                        Delta_c = min(max(Delta_c, gamma_3 * ns),
                                      1.0e+10)
                        suc = 'v'

                    # Step 4.3. Update maximum acceptable infeasibility.
                    theta_max = max(kappa_tx1 * theta_max,
                                    kappa_tx2*theta + (1-kappa_tx2)*thetaTrial)
                    theta = thetaTrial

                else:                      # Unsuccessful step.

                    # Backtracking linesearch a la Nocedal & Yuan.
                    ns = nStepNorm if nStepNorm > 0 else Delta_c
                    if ny:
                        (x, c, theta, alpha) = self.nyc(x, theta, thetaTrial,
                                                        c, Jop.T*c, step)
                        f = nlp.obj(x)
                        #g = nlp.grad(x)
                        self.step = alpha * step
                        Delta_c = min(alpha, .8) * ns
                        suc = 'y'
                    else:
                        Delta_c = gamma_1 * ns #Delta_c
                        suc = 'u'

                self.log.debug('New Delta_c = %8.2e' % Delta_c)
                self.log.debug('New theta_max = %8.2e' % theta_max)

            # Step 5. Book keeping.
            if ratio >= eta_1 or ny:
                g = nlp.grad(x)
                J = self.jac(x)
                Jop = PysparseLinearOperator(J)
                self.post_iteration()

                pNorm = cNorm = 0
                if m > 0:
                    pNorm = np.linalg.norm(c)
                    cNorm = np.linalg.norm(c, np.inf)
                    grad_lag = g + Jop.T * y
                else:
                    grad_lag = g.copy()
                dNorm = np.linalg.norm(grad_lag)/(1 + np.linalg.norm(y))

            if self.it % 20 == 0:
                self.log.info(self.hdr)
            self.log.info(self.linefmt % (self.it, it_type, suc, n_end,
                                          t_end, f, pNorm, dNorm, Delta_f,
                                          Delta_c, theta_max, cgiter))

            optimal = (pNorm <= stop_p) and (dNorm <= self.stop_d)
        # End while.

        self.tsolve = cputime() - tsolve

        if optimal:
            self.status = 0  # Successful solve.
            self.log.info('Found an optimal solution! Yeah!')
        else:
            self.status = 1 # Refine this in the future.

        self.x = x
        self.f = f
        self.optimal = optimal
        self.pResid = pNorm
        self.dResid = dNorm
        self.niter = self.it

        return


# A Variant of the Funnel class using LSTR instead of LSQR.

class LSTRFunnel(Funnel):

    def lsq(self, A, b, radius=None, **kwargs):
        """
        Solve the linear least-squares problem in the variable x

            minimize |Ax - b|

        in the Euclidian norm with LSTR. Optionally, x may be
        subject to a Euclidian-norm trust-region constraint
        |x| <= radius. This function returns (x, xNorm, status).
        """
        LSTR = LSTRFramework(A, -b, radius=radius)
        LSTR.Solve()
        return (LSTR.step, LSTR.stepNorm, LSTR.status)


# Specialized Funnel class with limited-memory approximation to the
# Hessian of the Lagrangian.

class LDFPFunnel(Funnel):
    """
    A specialized Funnel in which we maintain limited-memory DFP approximations
    of the Hessian matrices of the constraints.
    """

    def __init__(self, nlp, **kwargs):
        """
        A version of the Funnel algorithm in which the contribution of the
        constraints to the Hessian of the Lagrangian is replaced with a
        limited-memory quasi-Newton approximation.
        """
        Funnel.__init__(self, nlp, **kwargs)

        # Initialize LDFP structure.
        self.ldfps = []
        npairs = kwargs.get('npairs',5)
        for j in range(nlp.m):
            self.ldfps.append(LDFP(nlp.n, npairs=npairs))

        # Members to memorize old and current Jacobian matrices.
        self.J = self.J_old = None

    def jac(self, x):
        """
        Return the Jacobian of the constraints as a PysparseMatrix and
        store the previous Jacobian for use in the limited-memory quasi-Newton
        approximation of the Hessian of the Lagrangian.
        """
        J = Funnel.jac(self, x)
        if self.J is not None:
            self.J_old = self.J.copy()
        self.J = J
        return J

    def hprod(self, x, y, v):
        """
        Return an approximation to the product of the Hessian of the Lagrangian

        |         H(x,y) := H0(x) + sum yj * Hj(x)

        with the vector v, where H0 is the Hessian of the objective function
        and Hj is the Hessian of the j-th constraint. The product with H0 is
        preserved while the products with the Hj's are approximated.
        """
        m = self.nlp.m
        Hv = Funnel.hprod(self, x, np.zeros(m), v)   # = H0 * v
        for j in range(m):
            Hv += y[j] * self.ldfps[j].matvec(v)   # approx. = sum_i yi Hi*v
        return Hv

    def post_iteration(self, **kwargs):
        """
        Store the most recent {s,y} pair and update the L-DFP approximation
        of each constraint Hessian.
        """
        m = self.nlp.m
        step = self.step
        J = self.J ; J_old = self.J_old
        for j in range(m):
            y = J[j,:].getNumpyArray() - J_old[j,:].getNumpyArray()
            self.ldfps[j].store(step, y)
        return


# Specialized Funnel class with structured limited-memory approximation to the
# Hessian of the Lagrangian.

class StructuredLDFPFunnel(Funnel):
    """
    A specialized Funnel in which we maintain structured limited-memory DFP
    approximations of the Hessian matrices of the constraints.
    """

    def __init__(self, nlp, **kwargs):
        """
        A version of the Funnel algorithm in which the contribution of the
        constraints to the Hessian of the Lagrangian is replaced with a
        limited-memory quasi-Newton approximation.
        """
        Funnel.__init__(self, nlp, **kwargs)

        # Members to memorize old and current Jacobian matrices.
        self.J = self.J_old = None

        # Get sparsity pattern of Jacobian.
        J = self.jac(nlp.x0)
        (val, irow, jcol) = J.find()

        # Initialize LDFP structure.
        self.ldfps = []
        npairs = kwargs.get('npairs',5)
        for j in range(nlp.m):

            #print 'Initializing ldfps[%d]' % j
            row_j = np.where(irow==j)[0]
            vars = np.sort(jcol[row_j])
            #print vars
            self.ldfps.append(StructuredLDFP(nlp.n, npairs=npairs, vars=vars))

    def jac(self, x):
        """
        Return the Jacobian of the constraints as a PysparseMatrix and
        store the previous Jacobian for use in the limited-memory quasi-Newton
        approximation of the Hessian of the Lagrangian.
        """
        J = Funnel.jac(self, x)
        if self.J is not None:
            self.J_old = self.J.copy()
        self.J = J
        return J

    def hprod(self, x, y, v):
        """
        Return an approximation to the product of the Hessian of the Lagrangian

        |         H(x,y) := H0(x) + sum yj * Hj(x)

        with the vector v, where H0 is the Hessian of the objective function
        and Hj is the Hessian of the j-th constraint. The product with H0 is
        preserved while the products with the Hj's are approximated.
        """
        m = self.nlp.m
        Hv = Funnel.hprod(self, x, np.zeros(m), v)   # = H0 * v
        for j in range(m):
            #print 'hprod with constraint %d' % j
            jvars = self.ldfps[j].vars  # Nonzero vars in j-th gradient.
            Hv[jvars] += y[j] * self.ldfps[j].matvec(v[jvars])   # approx. = sum_i yi Hi*v
        return Hv

    def post_iteration(self, **kwargs):
        """
        Store the most recent {s,y} pair and update the L-DFP approximation
        of each constraint Hessian.
        """
        m = self.nlp.m
        step = self.step
        J = self.J ; J_old = self.J_old
        for j in range(m):
            #print 'Storing %d-th pair' % j
            jvars = self.ldfps[j].vars  # Nonzero vars in j-th gradient.
            #y = J[j,jvars].getNumpyArray() - J_old[j,jvars].getNumpyArray()
            # Using take() seems faster than the above.
            # Should modifiy take() so it accepts take(integer,list).
            rowj = [j] * len(jvars)
            y = J.take(rowj,jvars) - J_old.take(rowj,jvars)
            self.ldfps[j].store(step[jvars], y)
        return

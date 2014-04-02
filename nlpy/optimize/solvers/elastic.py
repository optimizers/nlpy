# -*- coding: utf-8 -*-

from pysparse.sparse import spmatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
from pysparse.sparse.pysparseMatrix import PysparseIdentityMatrix as Identity

from nlpy.model import NLPModel, QPModel
from nlpy.optimize.tr import trustregion as T
from nlpy.krylov.linop import PysparseLinearOperator
from nlpy.optimize.tr.trustregion import TrustRegionCG as TRCG
try:      # To solve augmented systems
    from nlpy.linalg.pyma57 import PyMa57Context as LBLContext
except:
    from nlpy.linalg.pyma27 import PyMa27Context as LBLContext
from nlpy.precon import GenericPreconditioner
from nlpy.tools.utils import where
from nlpy.tools.timing import cputime
from nlpy.tools.norms import  norm2, norm_infty

from math import sqrt, log10
import numpy as np
import logging


__docformat__ = 'restructuredtext'


np.seterr(invalid='raise')  # Force invalid operations to raise an exception.
np.seterr(divide='raise')  # Force invalid operations to raise an exception.

###############################################################################


class ElasticInteriorFramework(object):
    """
    ipm = ElasticInteriorFramework(problem)
    Instantiate an InteriorFramework class embedding the nonlinear
    program in the L1-penalty/elastic framework of [GOT03]_.

    References
    ----------

    .. [GOT03]  N. I. M. Gould, D. Orban and Ph. L. Toint, *An Interior-Point
                l1-Penalty Method for Nonlinear Optimization*, Technical Report
                RAL-TR-2003-022, Rutherford Appleton Laboratory, Chilton,
                Oxfordshire, UK, 2003.

    .. [CO09] Z. Coulibaly and D. Orban, *An L1 Elastic Interior-Point Method
              for Mathematical Programs with Complementarity Constraints*,
              GERAD Technical Report G-2009-74, Montreal, Canada, 2009.
    """

    # ========================================================================

    def __init__(self, nlp, maxiter=100, mu_min=None, nu_max=1e+12, **kwargs):
        """
        An ``ElasticInteriorFramework`` object represents a nonlinear
        optimization problem in expanded elastic form. All options not listed
        below are passed directly to the ``L1BarrierMeritFunction``
        constructor.

        :parameters:
            :nlp: NLPModel instance.

        :keywords:
            :maxiter: Maximum allowed number of iterations
            :mu_min:  Smallest allowed barrier parameter
            :nu_max:  Largest allowed penalty parameter
        """

        self.l1bar = L1BarrierMeritFunction(nlp, **kwargs)
        mu = self.l1bar.get_barrier_parameter()
        self.l1bar.l1.set_stopping_tolerances(10*mu, 10*mu, 10*mu)

        # Constants pertaining to the update of various parameters.
        self.tau1 = 10.0              # Factor of nu in updating rule.
        self.tau2 = 10.0              # Constant added to nu in updating rule.
        self.mu_factor = 0.2          # Factor of mu in updating rule.
        self.mu_min = mu_min          # Smallest allowed barrier parameter.
        if mu_min is None:
            self.mu_min = min(nlp.stop_d, nlp.stop_p, nlp.stop_c)/1000
        self.nu_max = nu_max          # Largest allowed penalty parameter.
        self.etaE = min(mu**(1.1)*100, mu/10) # Threshold to increase nu.
        self.etaI = min(mu**(1.1)*100, mu/10) # Threshold to increase nu.
        self.etaX = min(mu**(1.1)*100, mu/10) # Threshold to increase nu.
        self.kappaL = 0.5             # Used when projecting multipliers.
        self.kappaU = 1.0e+20

        self.attempt_magical_steps_outer = True
        self.attempt_magical_steps_inner = True

        # Other algorithmic parameters.
        self.maxiter = maxiter        # Max number of overall inner iterations.
        self.non_monotone_maxiter = 5
        self.armijo_slope = 1.0e-4
        self.ny_backtrack_max = 5

        # Attributes to be filled in later.
        self.niter = 1
        self.x = None
        self.tsolve = 0
        self.optimal = False
        self.f = 0
        self.dResid = self.cResid = self.pResid = 0

        # Create a logger for solver.
        self.logger = logging.getLogger('elastic.solver')

        # Get logger for details of the interior-point data, if one exists.
        barrier_logger_name = kwargs.get('barrier_logger_name', 'elastic.bar')
        self.bar_logger = logging.getLogger(barrier_logger_name)
        self.bar_logger.addHandler(logging.NullHandler())

        return

    # ========================================================================

    def initialize_penalty_parameters(self, xst, **kwargs):
        """
        Initialize the penalty parameters. Subclass and override this method to
        implement more sophisticated initialization procedures.
        """
        l1 = self.l1bar.l1
        l1.set_penalty_parameters(1.0e+1, 1.0e+1, 1.0e+1)
        return

    # ========================================================================

    def update_penalty_parameters(self, xst, yzuv, **kwargs):
        """
        Update penalty parameters. Subclass to implement more elaborate rules.
        """

        # Shortcuts.
        l1 = self.l1bar.l1 ; nlp = l1.nlp ; eC = nlp.equalC
        lC = nlp.lowerC ; uC = nlp.upperC ; rC = nlp.rangeC
        nrC = nlp.nrangeC
        n = nlp.n ; m = nlp.m
        x = xst[:n]

        pFeas = nlp.primal_feasibility(x, **kwargs)
        eFeas = norm_infty(pFeas[eC])
        iFeas = norm_infty(pFeas[lC+uC+rC])
        bFeas = norm_infty(pFeas[m+nrC:])

        (y, z, u, v) = l1.get_yzuv(yzuv)
        (nuE, nuS, nuT) = l1.get_penalty_parameters()

        y_nlp = l1.shifted_multipliers(y)
        if norm_infty(y_nlp) > 0.9 * nuE:
            nuE = self.tau1 * nuE + self.tau2

        if eFeas > self.etaE:
            nuE = self.tau1 * nuE + self.tau2

        if iFeas > self.etaI:
            nuS = self.tau1 * nuS + self.tau2

        if bFeas > self.etaX:
            nuT = self.tau1 * nuT + self.tau2

        l1.set_penalty_parameters(nuE, nuS, nuT)
        self.logger.debug('Setting penalty parameters to (%7.1e,%7.1e,%7.1e)' \
                          % (nuE, nuS, nuT))

        return

    # ========================================================================

    def update_barrier_parameter(self, *args, **kwargs):
        """
        Update the barrier parameter and the inner iteration stopping
        tolerances. Subclass to implement more elaborate rules.
        """

        # Shortcuts.
        l1bar = self.l1bar ; l1 = l1bar.l1

        gamma = 0.05              # Restriction: gamma > 0.
        tau = 0.6 * 2/(1+gamma)   # Restriction 1 < tau < 2/(1+gamma).

        mu = l1bar.get_barrier_parameter()
        mu = min(mu**tau, self.mu_factor*mu)
        l1bar.set_barrier_parameter(mu)
        self.logger.debug('Setting barrier parameter to %7.1e' % mu)

        # Update stopping tolerance for inner iterations.
        stop_d = 10 * mu**(1+gamma)
        stop_p = 10 * mu    # Not used by the barrier method.
        stop_c = 10 * mu #min(mu**1.3, mu)
        l1.set_stopping_tolerances(stop_d, stop_p, stop_c)

        # Update feasibility thresholds for penalty parameter (must go to 0).
        self.etaE = min(mu**(1.1), mu/10)
        self.etaI = min(mu**(1.1), mu/10)
        self.etaX = min(mu**(1.1), mu/10)

        return

    # ========================================================================

    def norm_st(self, xst):
        """
        Return the infinity norm of the vector [s,t].

        :parameters:
            :xst:  Vector of primal variables.
        """

        nlp = self.l1bar.l1.nlp ; n = nlp.n ; m = nlp.m ; nB = nlp.nbounds
        st = xst[nlp.n:]
        stNorm = norm_infty(st)
        return stNorm

    # ========================================================================

    def inner_residuals(self, xst, yzuv, c=None, g=None, J=None, scale=False):
        """
        Compute the vector of residuals used in the inner iteration stopping
        criterion:

        |          [    g(x,y,z;nu)    ]  ^
        |          [   nu*e - y' - u   ]  |  = dFeas
        |          [   nu*e - z' - v   ]  v
        |          [ (C(x)+S)*y - mu*e ]  <- = complCSY
        |          [   (X+T)*z - mu*e  ]  <- = complXTZ
        |          [    S*u - mu*e     ]  <- = complSU
        |          [    T*v - mu*e     ]  <- = complTV

        where g(x,y,z;nu) is the gradient of the Lagrangian, and y' and z' are
        the `shifted` multipliers, i.e., multipliers corresponding to equality
        constraints are shifted by the value of the penalty parameter.

        :parameters:
            :xst:  Vector of primal variables
            :yzuv: Vector of multiplier estimates for perturbed general
                   constraints and bounds.

        :keywords:
            :c:  Constraints vector of l1 penalty problem, if available
            :g:  Gradient vector of the l1 objective function, if available
            :J:  Jacobian of the constraints of the l1 problem, if available.
            :scale:  If set to `True`, residuals will be scaled.
        """

        # Shortcuts.
        l1bar = self.l1bar ; l1 = l1bar.l1 ; nlp = l1.nlp
        st = xst[nlp.n:]
        yz = yzuv[:l1.m] ; uv = yzuv[l1.m:]

        if c is None: c = l1.cons(xst)

        # Dual feasibility.
        dFeas = l1.dual_feasibility(xst, yzuv, g=g, J=J)

        # Complementarity.
        (csyz, stuv) = l1.complementarity(xst, yzuv, c=c)
        mu = l1bar.get_barrier_parameter()
        csyz -= mu
        stuv -= mu

        # Possibly scale residuals using a KKTResiduals instance.

        return (dFeas, csyz, stuv)

    # ========================================================================

    def dual_step(self, xst, step, yzuv, c=None, J=None):
        """
        Compute the step in the dual variables given the step in the primal
        variables.

        :parameters:
            :xst:  Vector of primal variables
            :step: step in the primal variables
            :yzuv: current value of the dual variables

        :keywords:
            :c:  Constraints vector of l1 penalty problem, if available
            :J:  Jacobian of the constraints of the l1 problem, if available.
        """

        # Shortcuts.
        l1bar = self.l1bar ; l1 = l1bar.l1 ; nlp = l1.nlp

        if c is None: c = l1.cons(xst)
        if J is None: J = l1.jac(xst)
        Jop = PysparseLinearOperator(J, symmetric=False)

        yzuv_p = l1bar.primal_multipliers(xst, c=c)
        yzuv_step = np.empty_like(yzuv_p)

        yz_p = yzuv_p[:l1.m] ; yz = yzuv[:l1.m]
        yzuv_step[:l1.m] = -yz + yz_p - (Jop * step) * yz/c  # Step in (y,z).

        uv_p = yzuv_p[l1.m:] ; uv = yzuv[l1.m:]
        st = xst[nlp.n:] ; st_step = step[nlp.n:]
        yzuv_step[l1.m:] = uv_p - uv - uv/st * st_step       # Step in (u,v).

        return yzuv_step

    # ========================================================================

    def project_multipliers(self, xst, yzuv, alpha, dyzuv, c=None):
        "Project the primal-dual multiplier estimates into a safety box."

        # Shortcuts.
        l1bar = self.l1bar ; l1 = l1bar.l1
        kL = self.kappaL ; kU = self.kappaU

        # Projection box.
        mu = l1bar.get_barrier_parameter()
        yzuvP = l1bar.primal_multipliers(xst, c=c)
        L = kL * np.minimum(1.0, np.minimum(yzuv, yzuvP))
        U = np.maximum(kU, np.maximum(yzuv, np.maximum(kU/mu, kU*yzuvP)))

        yzuv_new = yzuv + alpha * dyzuv
        print 'Projecting ', yzuv_new
        yzuv_new = np.maximum(L, np.minimum(yzuv_new, U))
        print 'Projection:', yzuv_new

        return yzuv_new

    # ========================================================================

    def steplength(self, xst, step, yzuv, dstep, c=None):
        """
        Return largest feasible steplength along the primal and dual steps to
        satisfy the fraction to the boundary rule.
        """

        # Shortcuts.
        l1bar = self.l1bar ; l1 = l1bar.l1 ; nlp = l1.nlp
        n = nlp.n ; m = nlp.m ; nrC = nlp.nrangeC
        lB = nlp.lowerB ; uB = nlp.upperB ; rB = nlp.rangeB
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB
        Lvar = nlp.Lvar ; Uvar = nlp.Uvar ; nB = nlp.nbounds

        # Initialize step size and fraction to the boundary.
        alpha = 1.0
        mu = l1bar.get_barrier_parameter()
        frac = 1 - min(0.01, 10*mu)
        if c is None: c = l1.cons(xst)

        # Ensure multipliers stay positive.
        neg = where(dstep < 0)
        if len(neg) > 0: alpha = min(alpha, np.min(-frac*yzuv[neg]/dstep[neg]))

        # Ensure elastics stay positive.
        st = xst[n:] ; st_step = step[n:]
        neg = where(st_step < 0)
        if len(neg) > 0: alpha = min(alpha, np.min(-frac*st[neg]/st_step[neg]))

        # Ensure linear constraints "x+t ≥ 0" remain strictly satisfied.
        dx = step[:n] ; dt = step[n+m:] ; cxt = c[m+nrC:]

        # ... lower bounds
        dxtlB = dx[lB] + dt[:nlB]
        neg = where(dxtlB < 0)
        if len(neg) > 0:
            xtlB = cxt[:nlB]
            alpha = min(alpha, np.min(-frac*xtlB[neg]/dxtlB[neg]))

        # ... upper bounds
        dxtuB = -dx[uB] + dt[nlB:nlB+nuB]
        neg = where(dxtuB < 0)
        if len(neg) > 0:
            xtuB = cxt[nlB:nlB+nuB]
            alpha = min(alpha, np.min(-frac*xtuB[neg]/dxtuB[neg]))

        # ... lower side of 2-sided bounds
        dxtrB = dx[rB] + dt[nlB+nuB:nB]
        neg = where(dxtrB < 0)
        if len(neg) > 0:
            xtrB = cxt[nlB+nuB:nlB+nuB+nrB]
            alpha = min(alpha, np.min(-frac*xtrB[neg]/dxtrB[neg]))

        # ... upper side of 2-sided bounds.
        dxtrB = -dx[rB] + dt[nlB+nuB:nB]
        neg = where(dxtrB < 0)
        if len(neg) > 0:
            xtrB = cxt[nlB+nuB+nrB:]
            alpha = min(alpha, np.min(-frac*xtrB[neg]/dxtrB[neg]))

        # Process general constraints, if any.
        c_trial = l1.cons(xst + alpha * step)
        while np.any(c_trial < (1-frac) * c):
            alpha /= 2
            c_trial = l1.cons(xst + alpha * step)

        return alpha

    # ========================================================================

    def magical_step(self, xst, c=None):
        """
        Reset the elastics s and t so as to further decrease the barrier merit
        function for a given x. Modifies `xst` in place.

        :parameters:
            :xst:   vector of primal variables.

        :keywords:
            :c:     vector of constraints of original NLP. Must conform to
                    :meth:`consPos`.
        """
        # Shortcuts.
        l1bar = self.l1bar ; l1 = l1bar.l1 ; nlp = l1.nlp ; m = nlp.m
        eC = nlp.equalC ; lC = nlp.lowerC ; uC = nlp.upperC ; rC = nlp.rangeC
        nrC = nlp.nrangeC
        rB = nlp.rangeB
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB
        Sqrt = np.sqrt
        logger = self.logger

        (x,s,t) = l1.get_xst(xst)
        nuE, nuS, nuT = l1.get_penalty_parameters()
        mu = l1bar.get_barrier_parameter()
        iC = lC + uC
        stride = 0.95

        # Process elastics associated to general constraints.
        if c is None: c = nlp.consPos(x)
        seC = (mu - nuE * c[eC] + Sqrt((nuE * c[eC])**2 +   mu*mu))/(2*nuE)
        s[eC] += stride * (seC - s[eC])
        logger.debug('Resetting s[Ec] to %s' % str(s[eC]))
        siC = (2*mu - nuS * c[iC] + Sqrt((nuS * c[iC])**2 + 4*mu*mu))/(2*nuS)
        s[iC] += stride * (siC - s[iC])
        logger.debug('Resetting s[iC] to %s' % str(s[iC]))

        # For range constraints, need the largest positive real root of a cubic.
        if nrC > 0:
            # Construct p(s) = q3 s^3 + q2 s^2 + q1 s + q0.
            cm = c[rC] * c[m:]
            cp = c[rC] + c[m:]
            q3 = nuS
            q2 = nuS * cp - 3*mu
            q1 = nuS * cm - 2*mu * cp
            q0 = -mu * cm
            for i in rC:
                k = rC.index(i)
                p = np.poly1d([q3, q2[k], q1[k], q0[k]])
                si = max([float(r) for r in p.r if r == float(r)])
                s[i] += stride * (si - s[i])
            logger.debug('Resetting s[rC] to %s' % str(s[rC]))

        # Process elastics associated to bound constraints.
        b = nlp.get_bounds(x)
        t_new = (2*mu - nuT * b[:nlB+nuB] + \
                Sqrt((nuT * b[:nlB+nuB])**2 + 4*mu*mu))/(2*nuT)
        t[:nlB+nuB] += stride * (t_new - t[:nlB+nuB])
        logger.debug('Resetting t[lB+uB] to %s' % str(t[:nlB+nuB]))

        # For two-sided bounds, need the largest positive real root of a cubic.
        if nrB > 0:
            bm = b[nlB+nuB:nlB+nuB+nrB] * b[nlB+nuB+nrB:]
            bp = b[nlB+nuB:nlB+nuB+nrB] + b[nlB+nuB+nrB:]
            q3 = nuT
            q2 = nuT * bp - 3*mu
            q1 = nuT * bm - 2*mu * bp
            q0 = -mu * bm
            for i in range(nrB):
                p = np.poly1d([q3,q2[i],q1[i],q0[i]])
                t_new = max([float(r) for r in p.r if r == float(r)])
                t[nlB+nuB+i] += stride * (t_new - t[nlB+nuB+i])
            logger.debug('Resetting t[rB] to %s' % str(t[nlB+nuB:nlB+nuB+nrB]))

        # Safety checks. Should disappear.
        c = l1.cons(xst)
        logger.debug('s = %s' % str(s))
        logger.debug('t = %s' % str(t))
        logger.debug('c = %s' % str(c))
        if np.any(s < 0):
            raise ValueError('Some elastics are negative.')
        if np.any(t < 0):
            raise ValueError('Some elastics are negative.')
        if np.any(c < 0):
            raise ValueError('Infeasible elastic reset.')

        return

    # ========================================================================

    def solve(self, **kwargs):
        """
        Solve problem using the interior/exterior elastic method.

        :keywords:
            :monotone:  Use monotone descent strategy (default: `True`)
        """

        # TODO
        # 1. Check algorithm in depth.
        # 2. Scale stopping conditions.
        # 3. Fix multiplier projection.

        monotone = kwargs.get('monotone', True)

        # Shortcuts.
        l1bar = self.l1bar ; l1 = l1bar.l1 ; nlp = l1.nlp
        logger = self.logger
        bar_logger = self.bar_logger

        # Formats and headers for printing.
        heads = ('It','Obj','DFeas','Compl','pFeas','Elast','Delta',
                'Alpha','Rho','LogMu','LogNu','Cg','Stat')
        header_fmt = '%5s %8s %7s %7s %7s %7s %7s %7s %8s %5s %5s %3s %4s'
        header = header_fmt % heads
        len_header = len(header)
        oneline_fmt1 = ' %4d %8.1e %7.1e %7.1e %7.1e %7.1e %7.1e'
        oneline_fmt2 = ' %7.1e %8.1e %5.1f %5.1f %3d %3s'

        # Various initializations.
        TR = T.TrustRegionFramework(eta1=1.0e-4, eta2=0.9,
                                    gamma1=0.25, gamma2=2.5)
        xst = l1bar.x0
        x, s, t = l1.get_xst(xst)
        yzuv = l1bar.primal_multipliers(xst)
        y, z, u, v = l1.get_yzuv(yzuv)
        f = nlp.obj(x)
        gf = nlp.grad(x)                      # For scaling KKT conditions.
        bar = l1bar.obj(xst)                  # Barrier function value.
        g = l1bar.grad(xst)                   #, y, z, u, v, g=g, J=J)
        H = l1bar.hess(xst)                   #, y, z, u, v, c=c, J=J)
        gNorm = norm2(g)
        self.initialize_penalty_parameters(xst, g=g) #, c=c)
        nuE, nuS, nuT = l1.get_penalty_parameters()
        max_penalty = max(nuE, nuS, nuT)
        mu = l1bar.get_barrier_parameter()
        niter = 0

        logger.info('-' * len_header)
        logger.info(header)

        bar_logger.info('-' * len_header)
        bar_logger.info(header)

        time0 = cputime()

        # Form KKT residuals of original NLP.
        y_nlp = l1.shifted_multipliers(y)
        kktRes = nlp.kkt_residuals(x, y_nlp, z)
        nlp_pResid = max(norm_infty(kktRes.pFeas), norm_infty(kktRes.bFeas))
        nlp_dResid = norm_infty(kktRes.dFeas)
        nlp_cResid = max(norm_infty(kktRes.gComp), norm_infty(kktRes.bComp))

        # Scale stopping conditions.
        d_scale0 = max(1, norm_infty(gf))
        nlp_pResid0 = max(1, nlp_pResid)
        nlp_stop_d = nlp.stop_d * d_scale0
        nlp_stop_p = nlp.stop_p * nlp_pResid0
        nlp_stop_c = nlp.stop_c * d_scale0

        nlp_optimal = nlp_pResid <= nlp_stop_p and \
                      nlp_dResid <= nlp_stop_d and \
                      nlp_cResid <= nlp_stop_c

        # Compute primal-dual conditions of l1 barrier problem.
        dFeas, csyz, stuv = self.inner_residuals(xst, yzuv) #, g=g, c=c, J=J)
        dResid = norm_infty(dFeas)
        cResid = max(norm_infty(csyz), norm_infty(stuv))

        bar_optimal = dResid <= l1.stop_d * d_scale0 and \
                      cResid <= l1.stop_c * d_scale0

        max_st = self.norm_st(xst)
        status_line = oneline_fmt1 % (niter, f,
                                      nlp_dResid, nlp_cResid, nlp_pResid,
                                      max_st, TR.Delta)

        bar_status_line = oneline_fmt1 % (niter, bar, dResid, cResid, 0,
                                          max_st, TR.Delta)

        # Set initial trust-region radius.
        TR.Delta = max(min(1.0e+10, 0.1 * gNorm), 1.0)

        stat = 'Opt'   # In case inner iterations are bypassed.

        # Outer loop.
        while not nlp_optimal and niter < self.maxiter \
              and max_penalty <= self.nu_max and mu >= self.mu_min:

            # Reset truncated CG tolerance at each new outer iteration.
            cgtol = 1.0e-1

            # Reinitialize non-monotonicity parameters at each outer iteration.
            if not monotone:
                bar_min = bar_ref = bar_can = bar
                l = 0
                sig_ref = sig_can = 0

            # Enter inner iteration.
            while not bar_optimal and not nlp_optimal and niter < self.maxiter:

                niter += 1

                # Solve trust-region subproblem with truncated CG.
                cgtol = max(1.0e-6, min(cgtol/2, sqrt(gNorm)))
                Hop = PysparseLinearOperator(H)

                # Factorize preconditioner.
                P = ElasticPreconditioner(self, H, xst, yzuv) #, c=c, J=J)
                if not P.posdef:
                    msg = 'Cannot make preconditioner positive definite.'
                    msg += ' Using identity matrix.'
                    logger.debug(msg)
                    P = lambda v: v

                #trcg = TRCG(g, Hop)
                qp = QPModel(g, Hop)
                trcg = TRCG(qp)
                trcg.Solve(radius=TR.Delta, reltol=cgtol, prec=P)
                step = trcg.step
                stepNorm = trcg.stepNorm
                cgiter = trcg.niter
                m = trcg.m
                if m is None:
                    #m = np.dot(g,step) + 0.5 * np.dot(step, Hop * step)
                    m = qp.obj(step)

                dyzuv = self.dual_step(xst, step, yzuv) #, c=c, J=J)

                # get_alpha could also return c(xTrial)?
                alpha = self.steplength(xst, step, yzuv, dyzuv)

                xst_trial = xst + alpha * step
                bar_trial = l1bar.obj(xst_trial)

                stat = 'Rej'
                # Sometimes, pred < 0. Why?
                rho = TR.Rho(bar, bar_trial, m, check_positive=False)

                if not monotone:
                    rho_his = (bar_ref - bar_rial)/(sig_ref - m)
                    rho = max(rho, rho_his)

                # Step acceptance test.
                if rho >= TR.eta1:

                    stat = 'Acc'
                    TR.UpdateRadius(rho, stepNorm)

                    # Update non-monotonicity parameters.
                    if not monotone:
                        sigRef = sigRef - m
                        sigCan = sigCan - m
                        if bar_trial < bar_min:
                            bar_can = bar_trial
                            bar_min = bar_trial
                            sig_can = 0
                            l = 0
                        else:
                            l = l + 1

                        if bar_trial > bar_can:
                            bar_cn = bar_trial
                            sig_can = 0

                        if l == self.non_monotone_maxiter:
                            bar_ref = bar_can
                            sig_ref = sig_can

                else:

                    # Perform backtracking along step.
                    slope = np.dot(g, step)
                    if slope >= 0:
                        msg = 'Oops! Slope=%g is uphill!!!' % slope
                        logger.debug(msg)
                        TR.UpdateRadius(rho, stepNorm)

                    else:

                        bk = 0
                        a = self.armijo_slope
                        found_feasible = True
                        while bk < self.ny_backtrack_max:
                            if bar_trial > bar + a * alpha * slope and \
                               found_feasible:
                                break
                            found_feasible = False
                            bk += 1
                            alpha /= 1.2
                            xst_trial = xst + alpha * step
                            try:
                                bar_trial = l1bar.obj(xst_trial)
                                found_feasible = True
                            except FloatingPointError:
                                # It could be that for some alphas, xst_trial
                                # is not strictly feasible. Cut some more.
                                pass

                        if bk < self.ny_backtrack_max:
                            stat = 'NY+'
                            TR.Delta = min(1.0e+10, 2 * alpha * stepNorm)
                        else:
                            if found_feasible:
                                stat = 'NY-'
                            else:
                                xst_trial = xst
                                bar_trial = bar
                            TR.UpdateRadius(rho, stepNorm)

                # Update primal variables.
                xst = xst_trial
                (x, s, t) = l1.get_xst(xst)

                if self.attempt_magical_steps_inner:
                    self.magical_step(xst_trial)

                # Update dual variables.
                yzuv += alpha * dyzuv
                #yzuv = self.project_multipliers(xst, yzuv, alpha, dyzuv)
                (y, z, u, v) = l1.get_yzuv(yzuv)

                f = nlp.obj(x)
                gf = nlp.grad(x) ; d_scale = max(1, norm_infty(gf))
                bar = bar_trial
                g = l1bar.grad(xst)
                gNorm = norm2(g)
                H = l1bar.hess(xst, yzuv)

                # Compute primal-dual conditions of l1 barrier problem.
                dFeas, csyz, stuv = self.inner_residuals(xst, yzuv)
                dResid = norm_infty(dFeas)
                cResid = max(norm_infty(csyz), norm_infty(stuv))
                max_st = self.norm_st(xst)
                bar_optimal = dResid <= l1.stop_d * d_scale and \
                              cResid <= l1.stop_c * d_scale

                # Form KKT residuals of original NLP.
                y_nlp = l1.shifted_multipliers(y)
                kktRes = nlp.kkt_residuals(x, y_nlp, z)
                nlp_pResid = max(norm_infty(kktRes.pFeas),
                                 norm_infty(kktRes.bFeas))
                nlp_dResid = norm_infty(kktRes.dFeas)
                nlp_cResid = max(norm_infty(kktRes.gComp),
                                 norm_infty(kktRes.bComp))

                nlp_stop_d = nlp.stop_d * d_scale
                nlp_stop_c = nlp.stop_c * d_scale
                nlp_optimal = \
                        nlp_pResid <= nlp_stop_p and \
                        nlp_dResid <= nlp_stop_d and \
                        nlp_cResid <= nlp_stop_c

                # Display main log.
                status_line += oneline_fmt2 % (alpha, rho, log10(mu),
                                               log10(max_penalty),
                                               cgiter, stat)
                logger.info(status_line)

                status_line = oneline_fmt1 % (niter, f,
                                              nlp_dResid, nlp_cResid,
                                              nlp_pResid, max_st, TR.Delta)

                # Write interior-point data to file if requested.
                bar_status_line += oneline_fmt2 % (alpha, rho, log10(mu),
                                                   log10(max_penalty),
                                                   cgiter, stat)
                bar_logger.info(bar_status_line)

                bar_status_line = oneline_fmt1 % (niter, bar, dResid, cResid,
                                                  0, max_st, TR.Delta)

            # End of inner iteration.

            # Perform magical step.
            if self.attempt_magical_steps_outer:
                self.magical_step(xst)

            # Update penalty parameters if necessary.
            self.update_penalty_parameters(xst, yzuv)
            nuE, nuS, nuT = l1.get_penalty_parameters()
            max_penalty = max(nuE, nuS, nuT)

            # Update barrier parameter.
            self.update_barrier_parameter()
            mu = l1bar.get_barrier_parameter()

            # Update barrier value and derivatives to reflect new parameters.
            bar = l1bar.obj(xst)
            g = l1bar.grad(xst)
            gNorm = norm2(g)
            H = l1bar.hess(xst, yzuv)
            bar_optimal = False

            # Reset initial trust-region radius at each new outer iteration.
            if stat == 'NY+':
                TR.Delta = max(1, min(1.0e+10, 2 * alpha * stepNorm))
            else:
                TR.Delta = max(1, min(1.0e+10, 0.1 * gNorm))

        # End of outer iteration.

        # Display final status.
        status_line += oneline_fmt2 % (alpha, rho, log10(mu),
                                       log10(max_penalty),
                                       cgiter, stat)
        logger.info(status_line)

        bar_status_line += oneline_fmt2 % (alpha, rho, log10(mu),
                                           log10(max_penalty),
                                           cgiter, stat)
        bar_logger.info(bar_status_line)

        tsolve = cputime() - time0

        if niter >= self.maxiter:
            self.status = 'Maximum number of iterations'
        elif nlp_optimal:
            self.status = 'Optimal'
        elif max_penalty > self.nu_max:
            self.status = 'Penalty parameter too large'
        elif mu < self.mu_min:
            self.status = 'Barrier parameter too small'
        else:
            self.status = 'Unknown'

        logger.info('Reason for stopping: ' + self.status)

        self.x = x ; self.s = s ; self.t = t
        self.y = y ; self.z = z ; self.u = u ; self.v = v
        self.y_nlp = y_nlp
        self.f = f
        self.dResid = nlp_dResid
        self.cResid = nlp_cResid
        self.pResid = nlp_pResid
        self.niter = niter
        self.tsolve = tsolve
        self.optimal = nlp_optimal

        return


###############################################################################

# The following derived classes implement various penalty parameter
# intializations.


class ElasticInteriorFramework2(ElasticInteriorFramework):

    # Heuristic 1: nu = |grad f| (SNOPT rule).

    def initialize_penalty_parameters(self, xst, **kwargs):
        """
        Initialize the penalty parameters using the so-called SNOPT rule:

        nuE = nuS = nuT = max(1, |∇f|).

        :keywords:
            :g:     gradient of original objective function at initial point,
                    if available.
        """
        l1 = self.l1bar.l1 ; nlp = l1.nlp
        (x, s, t) = l1.get_xst(xst)
        g = kwargs.get('g', nlp.grad(x))
        nu = max(1.0, norm_infty(g))
        l1.set_penalty_parameters(nu, nu, nu)
        return


###############################################################################


class ElasticInteriorFramework3(ElasticInteriorFramework):

    def initialize_penalty_parameters(self, xst, **kwargs):
        """
        Initialize the penalty parameters so as to approximately satisfy part
        of the dual feasiblity condition.

        Heuristic: nu = |y + u| (i ∈ I), nu = |y + u|/2 (i ∈ E).
        """
        l1bar = self.l1bar ; l1 = l1bar.l1 ; nlp = l1.nlp ; m = nlp.m
        eC = nlp.equalC ; neC = nlp.nequalC
        rC = nlp.rangeC ; nrC = nlp.nrangeC
        iC = nlp.lowerC + nlp.upperC + rC
        niC = nlp.nlowerC + nlp.nupperC + nrC
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB
        nB = nlp.nbounds ; rB = nlp.rangeB
        yzuvP = l1bar.primal_multipliers(xst, **kwargs)
        (yP, zP, uP, vP) = l1.get_yzuv(yzuvP)

        nuE = max(1.0, 0.5 * norm_infty(yP[eC] + uP[eC]))

        nuS = max(1.0, norm_infty(yP[iC] + uP[iC]))
        nuS = max(nuS, norm_infty(yP[m:] + uP[rC]))

        nuT = max(1.0, norm_infty(zP[:nB] + vP))
        nuT = max(nuT, norm_infty(zP[nB:] + vP[nlB+nuB:]))

        l1.set_penalty_parameters(nuE, nuS, nuT)
        return


###############################################################################


class ElasticInteriorFramework4(ElasticInteriorFramework):

    def initialize_penalty_parameters(self, xst, **kwargs):
        """
        Initialize the penalty parameters so as to approximately satisfy part
        of the dual feasiblity condition.

        Heuristic: nu = |u + y| (i ∈ I), nu = max |∇ ci| (i ∈ E).
        """
        l1bar = self.l1bar ; l1 = l1bar.l1 ; nlp = l1.nlp ; m = nlp.m
        eC = nlp.equalC ; neC = nlp.nequalC
        rC = nlp.rangeC
        iC = nlp.lowerC + nlp.upperC + rC
        nlB = nlp.nlowerB ; nuB = nlp.nupperB
        nB = nlp.nbounds ; rB = nlp.rangeB

        (x, s, t) = l1.get_xst(xst)
        yzuvP = l1bar.primal_multipliers(xst, **kwargs)
        (yP, zP, uP, vP) = l1.get_yzuv(yzuvP)

        nuE = 1.0
        for i in eC:
            nuE = max(nuE, norm_infty(nlp.igrad(i,x)))

        nuS = max(1.0, norm_infty(yP[iC] + uP[iC]))
        nuS = max(nuS, norm_infty(yP[m:] + uP[rC]))

        nuT = max(1.0, norm_infty(zP[:nB] + vP))
        nuT = max(nuT, norm_infty(zP[nB:] + vP[nlB+nuB:]))

        l1.set_penalty_parameters(nuE, nuS, nuT)
        return


###############################################################################


class ElasticPreconditioner(GenericPreconditioner):
    """
    Construct a preconditioner with the same structure as HessPhi, but in which
    the Hessian of the Lagrangian is replaced by some approximation. This
    approximation is repeatedly modified if necessary until the preconditioner
    is positive definite. For instance, Happrox might be diag(H) or band(H),
    or even H itself.
    """

    def __init__(self, eif, Happrox, xst, yzuv, **kwargs):

        l1bar = eif.l1bar ; l1 = l1bar.l1 ; nlp = l1.nlp
        n = nlp.n
        K = Happrox.copy()

        # Modify Happrox until preconditioner is positive definite.
        posdef = False
        e = np.ones(n) ; rn = range(n)
        nMods = 0
        while not posdef and nMods < 5:
            LBL = LBLContext(K)
            if LBL.neig > 0:
                nMods = nMods + 1
                K.addAt(100 * e, rn, rn)
            else:
                posdef = True

        # If we can't modify H sufficiently, try identity matrix.
        if not posdef:
            I = 100 * Identity(size=l1.n)
            K = l1bar.hess(xst, yzuv, H=I, **kwargs)
            LBL = LBLContext(K)
            posdef = (LBL.neig == 0)

        GenericPreconditioner.__init__(self, K)
        self.LBL = LBL
        self.posdef = posdef  # Indicates whether precond is positive definite.
        return

    def precon(self, x):
        self.LBL.solve(x)
        return self.LBL.x.copy()


###############################################################################

# Check derivatives of L1MeritFunction and L1BarrierMeritFunction.

if __name__ == "__main__":

    from nlpy.model import AmplModel
    from nlpy.tools.dercheck import DerivativeChecker
    import sys

    if len(sys.argv) < 2:
        print 'Use %s <problem-name>' % sys.argv[0]
        sys.exit()
    # End if...
    np.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

    nprobs = len(sys.argv[1:])
    for prob in sys.argv[1:]:

        nlp = AmplModel(prob)
        l1bar = L1BarrierMeritFunction(nlp)
        l1 = l1bar.l1
        if nprobs == 1:
            nlp.display_basic_info()
            l1.display_basic_info()
            l1bar.display_basic_info()

        # Check derivatives.
        #print 'Checking derivatives of original problem.'
        #derchk = DerivativeChecker(nlp, nlp.x0, step=1.0e-7, tol=1.0e-4)
        #derchk.check(grad=True, jac=True, hess=True, chess=True,
        #             verbose=False)

        xst = l1.x0
        print 'Checking derivatives of l1 merit function problem.'
        derchk = DerivativeChecker(l1, xst, step=1.0e-7, tol=1.0e-4)
        derchk.check(grad=True, jac=True, hess=True, chess=True,
                     verbose=False)

        print 'Checking derivatives of l1 barrier problem.'
        derchkbar = DerivativeChecker(l1bar, xst, step=1.0e-7, tol=1.0e-4)
        derchkbar.check(grad=True, hess=True, verbose=False)

        nlp.close()

# End if __name__ == ...

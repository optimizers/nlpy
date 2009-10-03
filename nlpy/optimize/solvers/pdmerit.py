"""
This module implements a purely primal-dual interior-point methods for
bound-constrained optimization. The method uses the primal-dual merit function
of Forsgren and Gill and solves subproblems by means of a truncated conjugate
gradient method with trust region.

References:

 [1] A. Forsgren and Ph. E. Gill, Primal-Dual Interior Methods for Nonconvex
     Nonlinear Programming, SIAM Journal on Optimization, 8(4):1132-1152, 1998

 [2] Ph. E. Gill and E. M. Gertz, A primal-dual trust region algorithm for
     nonlinear optimization, Mathematical Programming, 100(1):49-94, 2004

 [3] P. Armand and D. Orban, A Trust-Region Interior-Point Method for
     Bound-Constrained Programming Based on a Primal-Dual Merit Function,
     Cahier du GERAD G-xxxx, GERAD, Montreal, Quebec, Canada, 2008.

D. Orban, Montreal
"""

from pysparse          import spmatrix
from nlpy.tools        import List
from nlpy.tools.timing import cputime
from nlpy.tools.norms  import norm2, norm_infty
from math              import sqrt


import numpy as np
import sys

import pdb

class PrimalDualInteriorPointFramework:

    def __init__(self, nlp, TR, TrSolver, **kwargs):
        """
        Solve the bound-constrained problem

          minimize f(x)  subject to  x >= 0.

        Implements a framework based on the primal-dual merit function of
        Forsgren and Gill (1998). For now, only bound-constrained problems are
        supported.
        """
        self.nlp = nlp

        if nlp.nlowerC + nlp.nupperC + nlp.nrangeC > 0:
            raise ValueError, 'Only bound-constrained problems are supported.'

        if nlp.nfixedB > 0:
            raise ValueError, 'Fixed variables are currently not supported.'

        self.TR = TR
        self.TrSolver = TrSolver

        self.explicit = kwargs.get('explicit', False)  # Form Hessian or not
        self.mu = kwargs.get('mu', 1.0)
        self.mu_min = 1.0e-12
        self.bound_rel_factor = 0.1
        self.bound_abs_factor = 0.1

        self.maxiter       = kwargs.get('maxiter', max(100, 2*self.nlp.n))
        self.silent        = kwargs.get('silent',  False)
        self.ny            = kwargs.get('ny',      False)
        self.inexact       = kwargs.get('inexact', False)
        self.nyMax         = kwargs.get('nyMax',   5)
        self.opportunistic = kwargs.get('opportunistic', True)

        # Shortcuts for convenience
        self.lowerB  = np.array(self.nlp.lowerB, dtype=np.int)
        self.upperB  = np.array(self.nlp.upperB, dtype=np.int)
        self.rangeB  = np.array(self.nlp.rangeB, dtype=np.int)
        self.nlowerB = nlp.nlowerB
        self.nupperB = nlp.nupperB
        self.nrangeB = nlp.nrangeB

        # Number of dual variables
        self.ndual = self.nlowerB + self.nupperB + 2 * self.nrangeB

        # The multipliers z associated to bound constraints are ordered as
        # follows:
        #
        #  xi >= Li (i in lowerB) : z[:nlowerB]
        #  xi <= Ui (i in upperB) : z[nlowerB:nlowerB+nupperB]
        #  xi >= Li (i in rangeB) : z[nlowerB+nupperB:nlowerB+nupperB+nrangeB]
        #  xi <= Ui (i in rangeB) : z[nlowerB+nupper+nrangeB:]

        # Set appropriate primal-dual starting point
        (self.x, self.z) = self.StartingPoint()
        assert np.all(self.x > 0)
        assert np.all(self.z > 0)

        # Assemble the part of the primal-dual Hessian matrix that is constant.
        n = self.nlp.n ; ndual = self.ndual
        self.B = self.PDHessTemplate()
        self.diagB = np.empty(ndual)  # For diagonal preconditioning.

        self.iter = 0
        self.cgiter = 0
        self.f = None      # Used to record final objective function value
        self.psi = None    # Used to record final merit function value
        self.gf  = None    # Used to record final objective function gradient
        self.g   = None    # Used to record final merit function gradient
        self.g_old = None  # A previous gradient we may wish to keep around
        self.gNorm = None  # Used to record final merit function gradient norm
        self.save_g = False
        self.ord = 2       # Norm used throughout

        self.hformat = ' %-5s  %8s  %7s  %7s  %7s  %5s  %8s  %8s  %4s\n'
        head = ('Iter','f(x)','Resid','mu','alpha','cg','rho','Delta','Stat')
        self.header  = self.hformat % head
        self.hlen    = len(self.header)
        self.hline   = '-' * self.hlen + '\n'
        self.itFormat = '%-5d  '
        self.format='%8.1e  %7.1e  %7.1e  %7.1e  %5d  %8.1e  %8.1e  %4s\n'
        self.format0='%8.1e  %7.1e  %7.1e  %7.1e  %5d  %8.1e  %8.1e  %4s\n'
        self.printFrequency = 50

        # Optimality residuals, updated along the iterations
        self.dRes = None
        self.cRes = None
        self.pRes = None

        self.optimal = False
        self.debug = kwargs.get('debug', False)

        #self.path = []

        return


    def AtOptimality(self, x, z, **kwargs):
        """
        Shortcut.
        """
        return self.nlp.AtOptimality(x, np.array([]), z, **kwargs)


    def StartingPoint(self, **kwargs):
        """
        Compute a strictly feasible initial primal-dual estimate (x0, z0).
        """
        n = self.nlp.n
        Lvar = self.nlp.Lvar ; lowerB = self.lowerB
        Uvar = self.nlp.Uvar ; upperB = self.upperB ; rangeB = self.rangeB
        bnd_rel = self.bound_rel_factor ; bnd_abs = self.bound_abs_factor

        x0 = self.nlp.x0
        x = np.ones(n)
        z = np.ones(self.ndual)

        # Compute primal starting guess.

        x[lowerB] = np.maximum(x0[lowerB],
                               (1+np.sign(Lvar[lowerB])*bnd_rel)*Lvar[lowerB]+ \
                                   bnd_abs)
        x[upperB] = np.maximum(x0[upperB],
                               (1-np.sign(Uvar[upperB])*bnd_rel)*Uvar[upperB]+ \
                                   bnd_abs)
        for i in rangeB:
            sgn = np.sign(Lvar[i])
            sl = (1+sgn*bnd_rel) * Lvar[i] + bnd_abs
            sgn = np.sign(Uvar[i])
            su = (1-sgn*bnd_rel) * Uvar[i] + bnd_abs
            x[i] = max(min(x0[i], su), sl)
            #x[i] = 0.5 * (Lvar[i] + Uvar[i])

        # Compute dual starting guess.

        #g = self.nlp.grad(x)
        #self.mu = np.linalg.norm(np.core.multiply(x, g), ord=np.inf)
        #self.mu = max(1.0, self.mu)
        #z = np.empty(self.ndual, 'd')
        #for i in self.lowerB:
        #    k = self.lowerB.index(i)
        #    z[k] = self.mu/(x[i]-Lvar[i])
        #for i in self.upperB:
        #    k = self.nlowerB + self.upperB.index(i)
        #    z[k] = self.mu/(Uvar[i]-x[i])
        #for i in self.rangeB:
        #    k = self.nlowerB + self.nupperB + self.rangeB.index(i)
        #    z[i] = self.mu/(x[i]-Lvar[i])
        #    k += self.nrangeB
        #    z[k] = self.mu/(Uvar[i]-x[i])

        return (x,z)


    def PrimalMultipliers(self, x, **kwargs):
        """
        Return the vector of primal multipliers at `x`. The value of the barrier
        parameter used can either be supplied using the keyword argument `mu` or
        the current value of the instance is used.
        """
        mu = kwargs.get('mu', self.mu)
        Lvar = self.nlp.Lvar ; Uvar = self.nlp.Uvar
        lowerB = self.lowerB ; upperB = self.upperB ; rangeB = self.rangeB
        nlowerB = self.nlowerB ; nupperB = self.nupperB ; nrangeB = self.nrangeB

        z = np.empty(self.ndual)
        z[:nlowerB] = mu/(x[lowerB]-Lvar[lowerB])
        z[nlowerB:nlowerB+nupperB] = mu/(Uvar[upperB]-x[upperB])
        z[nlowerB+nupperB:nlowerB+nupperB+nrangeB] = mu/(x[rangeB]-Lvar[rangeB])
        z[nlowerB+nupperB+nrangeB:] = mu/(Uvar[rangeB]-x[rangeB])

        return z

    def PDMerit(self, x, z, **kwargs):
        """
        Evaluate the primal-dual merit function at (x,z). If there are b >= 0
        one-sided bound constraints and q >= 0 two-sided bound constraints, the
        vector z should have length b+2q. The
        components z[i] (0 <= i < b+q) correspond, in increasing order of
        variable indices to
            - variables with a lower bound only,
            - variables with an upper bound only,
            - the lower bound on variables that are bounded below and above.

        The components z[i] (b+q <= i < b+2q) correspond to the upper bounds on
        variables that are bounded below and above.

        This function returns the value of the primal-dual merit function. The
        current value of the objective function can be supplied via the keyword
        argument `f`.
        """
        mu = kwargs.get('mu', self.mu)
        f = kwargs.get('f', None)
        n = self.nlp.n
        Lvar = self.nlp.Lvar ; Uvar = self.nlp.Uvar
        lowerB = self.lowerB ; upperB = self.upperB ; rangeB = self.rangeB
        nlowerB = self.nlowerB ; nupperB = self.nupperB ; nrangeB = self.nrangeB

        # Include contribution of objective function.
        if f is None:
            f = self.nlp.obj(x)
        merit = f

        # Include contribution of bound constraints.
        slow = x[lowerB] - Lvar[lowerB]
        merit -= mu * np.sum(np.log(slow))
        merit += np.dot(slow, z[:nlowerB])
        merit -= mu * np.sum(np.log(slow * z[:nlowerB]))

        supp = Uvar[upperB] - x[upperB]
        merit -= mu * np.sum(np.log(supp))
        merit += np.dot(supp, z[nlowerB:nlowerB+nupperB])
        merit -= mu * np.sum(np.log(supp * z[nlowerB:nlowerB+nupperB]))

        slo2 = x[rangeB] - Lvar[rangeB]
        merit -= mu * np.sum(np.log(slo2))
        merit += np.dot(slo2, z[nlowerB+nupperB:nlowerB+nupperB+nrangeB])
        merit -= mu * np.sum(np.log(slo2 * z[nlowerB+nupperB:nlowerB+nupperB+nrangeB]))

        sup2 = Uvar[rangeB] - x[rangeB]
        merit -= mu * np.sum(np.log(sup2))
        merit += np.dot(sup2, z[nlowerB+nupperB+nrangeB:])
        merit -= mu * np.sum(np.log(sup2 * z[nlowerB+nupperB+nrangeB:]))

        return merit

    def GradPDMerit(self, x, z, **kwargs):
        """
        Evaluate the gradient of the primal-dual merit function at (x,z).
        See :meth:`PDMerit` for a description of `z`.
        """
        mu = kwargs.get('mu', self.mu)
        check_optimal = kwargs.get('check_optimal', False)
        gf = kwargs.get('g', None)

        n = self.nlp.n
        Lvar = self.nlp.Lvar ; Uvar = self.nlp.Uvar
        lowerB = self.lowerB ; upperB = self.upperB ; rangeB = self.rangeB
        nlowerB = self.nlowerB ; nupperB = self.nupperB ; nrangeB = self.nrangeB

        g = np.empty(n + self.ndual)

        # Gradient of the objective function.
        if gf is None:
            g[:n] = self.nlp.grad(x)
        else:
            g[:n] = gf.copy()

        # Check optimality conditions at this point if requested.
        if check_optimal: 
            res, self.optimal = self.AtOptimality(x, z, g=g[:n])
            self.dRes = res[0] ; self.cRes = res[2] ; self.pRes = res[4]

        # Segement z for conciseness
        n1 = nlowerB ; n2 = n1 + nupperB ; n3 = n2 + nrangeB

        # Assemble the gradient with respect to x.
        g[lowerB] += -2 * mu / (x[lowerB] - Lvar[lowerB]) + z[:n1]
        g[upperB] +=  2 * mu / (Uvar[upperB] - x[upperB]) - z[n1:n2]
        g[rangeB] += -2 * mu / (x[rangeB] - Lvar[rangeB]) + z[n2:n3]
        g[rangeB] +=  2 * mu / (Uvar[rangeB] - x[rangeB]) - z[n3:]

        # Assemble the gradient with respect to z.
        g[n:n+n1]    = x[lowerB] - Lvar[lowerB] - mu/z[:n1]
        g[n+n1:n+n2] = Uvar[upperB] - x[upperB] - mu/z[n1:n2]
        g[n+n2:n+n3] = x[rangeB] - Lvar[rangeB] - mu/z[n2:n3]
        g[n+n3:]     = Uvar[rangeB] - x[rangeB] - mu/z[n3:]

        return g

    def HessProd(self, x, z, p, **kwargs):
        """
        Compute the matrix-vector product between the Hessian matrix of the
        primal-dual merit function at (x,z) and the vector p. See
        help(PDMerit) for a description of z. If there are b bounded variables
        and q two-sided bounds, the vector p should have length n+b+2q. The
        Hessian matrix has the general form

            [ H + 2 mu X^{-2}      I     ]
            [      I           mu Z^{-2} ].
        """
        mu = kwargs.get('mu', self.mu)
        n = self.nlp.n
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar
        Hp = np.zeros(n + self.ndual, 'd')

        Hp[:n] = self.nlp.hprod(self.nlp.pi0, p[:n])
        for i in self.lowerB:
            k = self.lowerB.index(i)
            Hp[i] += 2.0 * mu/(x[i] - Lvar[i])**2 * p[i] + p[n+k]
            Hp[n+k] += p[i] + mu/z[k]**2 * p[n+k]
        for i in self.upperB:
            k = self.nlowerB + self.upperB.index(i)
            Hp[i] += 2.0 * mu/(Uvar[i] - x[i])**2 * p[i] - p[n+k]
            Hp[n+k] += -p[i] - mu/z[k]**2 * p[n+k]
        for i in self.rangeB:
            k = self.nlowerB + self.nupperB + self.rangeB.index(i)
            Hp[i] += 2.0 * mu/(x[i] - Lvar[i])**2 * p[i] + p[n+k]
            Hp[n+k] += p[i] + mu/z[k]**2 * p[n+k]
            k += self.nrangeB
            Hp[i] += 2.0 * mu/(Uvar[i] - x[i])**2 * p[i] - p[n+k]
            Hp[n+k] += -p[i] + mu/z[k]**2 * p[n+k]

        return Hp
        
    def PDHessProd(self, x, z, p, **kwargs):
        """
        Compute the matrix-vector product between the modified Hessian matrix of
        the primal-dual merit function at (x,z) and the vector p. See
        help(PDMerit) for a description of z. If there are b bounded variables
        and q two-sided bounds, the vector p should have length n+b+2q.
        The Hessian matrix has the general form

            [ H + 2 X^{-1} Z      I     ]
            [      I           Z^{-1} X ].
        """
        mu = kwargs.get('mu', self.mu)
        n = self.nlp.n
        ndual = self.ndual
        N = self.nlowerB + self.nupperB + self.nrangeB
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar
        Hp = np.zeros(n + self.ndual, 'd')

        Hp[:n] = self.nlp.hprod(self.nlp.pi0, p[:n])
        for i in self.lowerB:
            k = self.lowerB.index(i)
            Hp[i] += 2.0 * z[k]/(x[i]-Lvar[i]) * p[i] + p[n+k]
            Hp[n+k] += p[i] + (x[i]-Lvar[i])/z[k] * p[n+k]
        for i in self.upperB:
            k = self.nlowerB + self.upperB.index(i)
            Hp[i] += 2.0 * z[k]/(Uvar[i]-x[i]) * p[i] - p[n+k]
            Hp[n+k] += -p[i] - (Uvar[i]-x[i])/z[k] * p[n+k]
        for i in self.rangeB:
            k = self.nlowerB + self.nupperB + self.rangeB.index(i)
            Hp[i] += 2.0 * z[k]/(x[i]-Lvar[i]) * p[i] + p[n+k]
            Hp[n+k] += p[i] + (x[i]-Lvar[i])/z[k] * p[n+k]
            k += self.nrangeB
            Hp[i] += 2.0 * z[k]/(Uvar[i]-x[i]) * p[i] - p[n+k]
            Hp[n+k] += -p[i] + (Uvar[i]-x[i])/z[k] * p[n+k]

        return Hp

    def PDHessTemplate(self, **kwargs):
        """
        Assemble the part of the modified Hessian matrix of the primal-dual
        merit function that is iteration independent. The function PDHess()
        fills in the blanks by updating the rest of the matrix.
        """
        n = self.nlp.n ; ndual = self.ndual
        Lvar = self.nlp.Lvar ; Uvar = self.nlp.Uvar
        lowerB = self.lowerB ; upperB = self.upperB ; rangeB = self.rangeB
        nlowerB = self.nlowerB ; rlowerB = List(range(nlowerB))
        nupperB = self.nupperB ; rupperB = List(range(nupperB))
        nrangeB = self.nrangeB ; rrangeB = List(range(nrangeB))

        B = spmatrix.ll_mat_sym(n+ndual, self.nlp.nnzh + 3 * ndual)

        B.put( 1.0, n + rlowerB, lowerB)
        B.put(-1.0, n + nlowerB + rupperB, upperB)
        B.put( 1.0, n + nlowerB + nupperB + rrangeB, rangeB)
        B.put(-1.0, n + nlowerB + nupperB + nrangeB + rrangeB, rangeB)

        return B

    def PDHess(self, x, z, **kwargs):
        """
        Assemble the modified Hessian matrix of the primal-dual merit function
        at (x,z). See :meth:`PDMerit` for a description of `z`.
        The Hessian matrix has the general form

            [ H + 2 X^{-1} Z      I     ]
            [      I           Z^{-1} X ].
        """
        mu = kwargs.get('mu', self.mu)
        n = self.nlp.n
        ndual = self.ndual
        Lvar = self.nlp.Lvar ; Uvar = self.nlp.Uvar
        lowerB = self.lowerB ; upperB = self.upperB ; rangeB = self.rangeB
        nlowerB = self.nlowerB ; rlowerB = List(range(nlowerB))
        nupperB = self.nupperB ; rupperB = List(range(nupperB))
        nrangeB = self.nrangeB ; rrangeB = List(range(nrangeB))

        # Segement z for conciseness
        n1 = nlowerB ; n2 = n1 + nupperB ; n3 = n2 + nrangeB

        B = self.B

        # Update (1,1) block.
        B[:n,:n] = self.nlp.hess(x, self.nlp.pi0)
        B.update_add_at(2*z[:n1]/(x[lowerB]-Lvar[lowerB]),   lowerB, lowerB)
        B.update_add_at(2*z[n1:n2]/(Uvar[upperB]-x[upperB]), upperB, upperB)
        B.update_add_at(2*z[n2:n3]/(x[rangeB]-Lvar[rangeB]), rangeB, rangeB)
        B.update_add_at(2*z[n3:]/(Uvar[rangeB]-x[rangeB]),   rangeB, rangeB)
        
        # Update (2,2) block.
        B.put((x[lowerB]-Lvar[lowerB])/z[:n1], n+rlowerB)
        B.put((Uvar[upperB]-x[upperB])/z[n1:n2], n+nlowerB+rupperB)
        B.put((x[rangeB]-Lvar[rangeB])/z[n2:n3], n+nlowerB+nupperB+rrangeB)
        B.put((Uvar[rangeB]-x[rangeB])/z[n3:], n+nlowerB+nupperB+nrangeB+rrangeB)

        # Store diagonal of B for diagonal preconditioning.
        B.take(self.diagB, range(ndual))
        self.diagB = np.maximum(1, self.diagB)

        return None

    def ftb(self, x, z, step, **kwargs):
        """
        Compute the largest alpha in ]0,1] such that
            (x,z) + alpha * step >= (1 - tau) * (x,z)
        where 0 < tau < 1. By default, tau = 0.9.
        """
        tau = kwargs.get('tau', 0.9)
        n = self.nlp.n
        Lvar = self.nlp.Lvar ; Uvar = self.nlp.Uvar
        lowerB = self.lowerB ; upperB = self.upperB ; rangeB = self.rangeB

        #pdb.set_trace()

        dx = step[:n]  # Step in x.
        lowerneg = np.where(dx[lowerB] < 0)[0]
        upperneg = np.where(dx[upperB] < 0)[0]
        rangeneg = np.where(dx[rangeB] < 0)[0]
        stpmax = 1.0
        if len(lowerneg) > 0:
            idx = lowerB[lowerneg]
            stpmax = min(stpmax, np.min(-tau*(x[idx]-Lvar[idx])/dx[idx]))
        if len(upperneg) > 0:
            idx = upperB[upperneg]
            stpmax = min(stpmax, np.min(-tau*(Uvar[idx]-x[idx])/dx[idx]))
        if len(rangeneg) > 0:
            idx = rangeB[rangeneg]
            stpmax = min(stpmax, np.min(-tau*(x[idx]-Lvar[idx])/dx[idx]))
            stpmax = min(stpmax, np.min(-tau*(Uvar[idx]-x[idx])/dx[idx]))

        stpx = stpmax

        dz = step[n:]  # Step in z.
        stpmax = 1.0
        whereneg = np.where(dz < 0)[0]
        if len(whereneg) > 0:
            stpmax = min(stpmax, np.min(-tau * z[whereneg]/dz[whereneg]))

        return stpx, stpmax


    def Precon(self, v, **kwargs):
        """
        Generic preconditioning method---must be overridden
        """
        return v #/self.diagB

    def UpdatePrecon(self, **kwargs):
        """
        Override this method for preconditioners that need updating,
        e.g., a limited-memory BFGS preconditioner.
        """
        return None

    def SolveInner(self, **kwargs):
        """
        Perform a series of inner iterations so as to minimize the primal-dual
        merit function with the current value of the barrier parameter to within
        some given tolerance. The only optional argument recognized is

            stopTol     stopping tolerance (default: 10 mu).
        """

        nlp = self.nlp
        n = nlp.n
        rho = 1                  # Dummy initial value for rho
        niter = 0                # Dummy initial value for number of inners
        status = ''              # Dummy initial step status
        alpha = 0.0              # Fraction-to-the-boundary step size
        if self.inexact:
            cgtol = 1.0
        else:
            cgtol = 1.0e-6
        inner_iter = 0           # Inner iteration counter
            
        # Obtain starting point
        (x,z) = (self.x, self.z)

        # Obtain first-order data at starting point
        if self.iter == 0:
            f = nlp.obj(x) ; gf = nlp.grad(x)
        else:
            f = self.f ; gf = self.gf

        psi = self.PDMerit(x, z, f=f)
        g = self.GradPDMerit(x, z, g=gf, check_optimal=True)
        gNorm = norm2(g) #np.linalg.norm(g, ord=self.ord)
        if self.optimal: return
        
        # Reset initial trust-region radius
        self.TR.Delta = 0.1 * gNorm #max(10.0, gNorm)

        # Set inner iteration stopping tolerance
        stopTol = kwargs.get('stopTol', 10*self.mu) #1.0e-3 * gNorm)
        finished = (gNorm <= stopTol) or (self.iter > self.maxiter)

        while not finished:

            # Print out header every so many iterations
            if not self.silent:
                if self.iter % self.printFrequency == 0:
                    sys.stdout.write(self.hline)
                    sys.stdout.write(self.header)
                    sys.stdout.write(self.hline)
    
                if inner_iter == 0:
                    sys.stdout.write(('*' + self.itFormat) % self.iter)
                else:
                    sys.stdout.write((' ' + self.itFormat) % self.iter)

                sys.stdout.write(self.format % (f,
                                 max(norm_infty(self.dRes),
                                     norm_infty(self.cRes),
                                     norm_infty(self.pRes)),
                                 self.mu, alpha, niter, rho,
                                 self.TR.Delta, status))

            # Set stopping tolerance for trust-region subproblem.
            if self.inexact:
                cgtol = max(1.0e-8, min(0.1 * cgtol, sqrt(gNorm)))
                if self.debug: self._debugMsg('cgtol = ' + str(cgtol))

            # Update Hessian matrix with current iteration information.
            self.PDHess(x,z)

            if self.debug:
                self._debugMsg('H = ') ; print self.B
                self._debugMsg('g = ' + np.str(g))
                self._debugMsg('gNorm = ' + str(gNorm))
                self._debugMsg('stopTol = ' + str(stopTol))
                self._debugMsg('dRes = ' + np.str(self.dRes))
                self._debugMsg('cRes = ' + np.str(self.cRes))
                self._debugMsg('pRes = ' + np.str(self.pRes))
                self._debugMsg('optimal = ' + str(self.optimal))

            # Iteratively minimize the quadratic model in the trust region
            # m(s) = <g, s> + 1/2 <s, Hs>
            # Note that m(s) does not include f(x): m(0) = 0.            
            solver = self.TrSolver(g,
                                   #matvec=lambda v: self.PDHessProd(x,z,v),
                                   H = self.B,
                                   prec=self.Precon,
                                   radius=self.TR.Delta,
                                   reltol=cgtol,
                                   #debug=True,
                                   #btol=.9,
                                   #cur_iter=np.concatenate((x,z))
                                   )
            solver.Solve()

            if self.debug:
                self._debugMsg('x = ' + np.str(x))
                self._debugMsg('z = ' + np.str(z))
                self._debugMsg('step = ' + np.str(solver.step))
                self._debugMsg('step norm = ' + str(solver.stepNorm))

            # Record total number of CG iterations.
            niter = solver.niter
            self.cgiter += solver.niter

            # Compute maximal step to the boundary and next candidate.
            alphax, alphaz = self.ftb(x, z, solver.step)
            alpha = min(alphax, alphaz)
            dx = solver.step[:n] ; dz = solver.step[n:]
            x_trial = x + alphax * dx
            z_trial = z + alphaz * dz
            f_trial = nlp.obj(x_trial)
            psi_trial = self.PDMerit(x_trial, z_trial, f=f_trial)

            # Compute ratio of predicted versus achieved reduction.
            rho  = self.TR.Rho(psi, psi_trial, solver.m)

            if self.debug:
                self._debugMsg('m = ' + str(solver.m))
                self._debugMsg('x_trial = ' + np.str(x_trial))
                self._debugMsg('z_trial = ' + np.str(z_trial))
                self._debugMsg('psi_trial = ' + str(psi_trial))
                self._debugMsg('rho = ' + str(rho))

            # Accept or reject next candidate
            status = 'Rej'
            if rho >= self.TR.eta1:
                self.TR.UpdateRadius(rho, solver.stepNorm)
                x = x_trial
                z = z_trial
                f = f_trial
                psi = psi_trial
                gf = nlp.grad(x)
                g = self.GradPDMerit(x, z, g=gf, check_optimal=True)
                gNorm = norm2(g) #np.linalg.norm(g, ord=self.ord)
                if self.optimal:
                    finished = True
                    continue
                status = 'Acc'
            else:
                if self.ny: # Backtracking linesearch a la "Nocedal & Yuan"
                    slope = np.dot(g, solver.step)
                    target = psi + 1.0e-4 * alpha * slope
                    j = 0

                    while (psi_trial >= target) and (j < self.nyMax):
                        alphax /= 1.2 ; alphaz /= 1.2
                        alpha = min(alphax, alphaz)
                        target = psi + 1.0e-4 * alpha * slope
                        x_trial = x + alphax * dx
                        z_trial = z + alphaz * dz
                        f_trial = nlp.obj(x_trial)
                        psi_trial = self.PDMerit(x_trial, z_trial, f=f_trial)
                        j += 1

                    if self.opportunistic or (j < self.nyMax):
                        x = x_trial
                        z = z_trial
                        f = f_trial
                        psi = psi_trial
                        gf = nlp.grad(x)
                        g = self.GradPDMerit(x, z, g=gf, check_optimal=True)
                        gNorm = norm2(g) #np.linalg.norm(g, ord=self.ord)
                        if self.optimal:
                            finished = True
                            continue
                        self.TR.Delta = alpha * solver.stepNorm
                        status = 'N-Y'

                    else:
                        self.TR.UpdateRadius(rho, solver.stepNorm)

                else:
                    self.TR.UpdateRadius(rho, solver.stepNorm)

            self.UpdatePrecon()
            self.iter += 1
            inner_iter += 1
            finished = (gNorm <= stopTol) or (self.iter > self.maxiter)
            if self.debug: sys.stderr.write('\n')

            #(dRes, cRes, pRes) = self.OptimalityResidual(x, z, mu = self.mu)
            #maxRes = max(np.linalg.norm(dRes, ord=np.inf), cRes, pRes)

        # Store final iterate
        (self.x, self.z) = (x, z)
        self.f = f
        self.gf = gf
        self.g = g
        self.gNorm = gNorm
        self.psi = psi
        return

    def SolveOuter(self, **kwargs):

        nlp = self.nlp
        n = nlp.n
        print self.x
        
        # Measure solve time
        t = cputime()

        # Solve sequence of inner iterations
        while (not self.optimal) and (self.mu >= self.mu_min) and \
                (self.iter < self.maxiter):
            self.SolveInner(stopTol=max(1.0e-7, 5*self.mu))
            #self.z = self.PrimalMultipliers(self.x)
            res, self.optimal = self.AtOptimality(self.x, self.z)
            self.UpdateMu()

        self.tsolve = cputime() - t    # Solve time
        if self.optimal:
            print 'First-order optimal solution found'
        elif self.iter >= self.maxiter:
            print 'Maximum number of iterations reached'
        else:
            print 'Reached smallest allowed value of barrier parameter'
        return

    def UpdateMu(self, **kwargs):
        """
        Update the barrier parameter before the next round of inner iterations.
        """
        self.mu /= 5
        #res = max(self.dRes, self.cRes)
        #guard = min(res/5.0, res**(1.5))
        #if guard <= self.mu/5.0:
        #    self.mu = guard
        #else:
        #    self.mu = min(self.mu/5.0, self.mu**(1.5))
        #self.mu = min(self.mu/5.0, self.mu**(1.5))
        #self.mu = min(self.mu/2.0, max(self.mu/5.0, res/5.0))
        return None


    def _debugMsg(self, msg):
        sys.stderr.write('Debug:: ' + msg + '\n')
        return None


if __name__ == '__main__':

    import amplpy
    import trustregion
    import sys
    import pylab

    # Set printing standards for arrays
    np.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

    prob = sys.argv[1]

    # Initialize problem
    nlp = amplpy.AmplModel(prob)
    #nlp.stop_p = 1.0e-12
    #nlp.stop_d = 1.0e-12
    #nlp.stop_c = 1.0e-12

    # Initialize trust region framework
    TR = trustregion.TrustRegionFramework(Delta = 1.0,
                                           eta1 = 0.0001,
                                           eta2 = 0.95,
                                           gamma1 = 1.0/3,
                                           gamma2 = 2.5)

    # Set up interior-point framework
    TRIP = PrimalDualInteriorPointFramework(
                nlp,
                TR,
                #trustregion.TrustRegionGLTR,
                trustregion.TrustRegionCG,
                silent = False,
                ny = True,
                inexact = True,
                debug = False,
                maxiter = 150,
                mu = 1.0
          )

    # Scale stopping conditions
    g = nlp.grad(TRIP.x)
    (dRes, cRes, pRes) = nlp.OptimalityResidual(TRIP.x, TRIP.z, gradf = g)
    dRes_norm = np.linalg.norm(dRes, ord=TRIP.ord)
    TRIP.nlp.stop_d = max(TRIP.nlp.stop_d, 1.0e-8 * max(1.0, dRes_norm))
    TRIP.nlp.stop_c = max(TRIP.nlp.stop_c, 1.0e-6 * max(1.0, cRes))
    print 'Target tolerances: (%7.1e, %7.1e)' % \
        (TRIP.nlp.stop_d, TRIP.nlp.stop_c)

    # Reset initial value of mu to a more sensible value
    TRIP.mu = 10.0 #max(dRes_norm, cRes) #* 100
    #TRIP.mu = np.linalg.norm(np.core.multiply(TRIP.x, g), ord=np.inf)
    
    # Solve problem
    TRIP.SolveOuter()

    # Display final statistics
    print 'Final variables:'; print TRIP.x
    print 'Final multipliers:'; print TRIP.z
    print
    print 'Optimal: ', TRIP.optimal
    print 'Variables: ', TRIP.nlp.n
    print '# lower, upper, 2-sided bounds: %-d, %-d, %-d' % \
        (TRIP.nlowerB, TRIP.nupperB, TRIP.nrangeB)
    print 'Primal feasibility error  = %15.7e' % TRIP.pRes
    print 'Dual   feasibility error  = %15.7e' % TRIP.dRes
    print 'Complementarity error     = %15.7e' % TRIP.cRes
    print 'Number of function evals  = %d' % TRIP.nlp.feval
    print 'Number of gradient evals  = %d' % TRIP.nlp.geval
    print 'Number of Hessian  evals  = %d' % TRIP.nlp.Heval
    print 'Number of matvec products = %d' % TRIP.nlp.Hprod
    print 'Final objective value     = %15.7e' % TRIP.f
    print 'Solution time: ', TRIP.tsolve

    nlp.close()

    #print TRIP.path[0], TRIP.path[-1]
    #pylab.plot([TRIP.path[i][0] for i in range(len(TRIP.path))],
    #            [TRIP.path[i][1] for i in range(len(TRIP.path))],
    #            '.-', linewidth=2)
    #pylab.show()

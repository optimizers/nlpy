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
from nlpy.tools.timing import cputime
from math              import log, sqrt

import numpy
import sys

class PrimalDualInteriorPointFramework:

    def __init__( self, nlp, TR, TrSolver, **kwargs ):
        """
        Implements a framework based on the primal-dual merit function of
        Forsgren and Gill (1998). For now, only bound-constrained problems are
        supported.
        """
        self.nlp = nlp

        if nlp.nlowerC + nlp.nupperC + nlp.nrangeC > 0:
            raise ValueError, 'Only bound-constrained problems are supported.'

        self.TR = TR
        self.TrSolver = TrSolver
        self.solver = None

        self.explicit = kwargs.get('explicit', False)  # Form Hessian or not
        self.mu = kwargs.get('mu', 1.0)
        self.mu_min = 1.0e-12
        self.bound_rel_factor = 0.01
        self.bound_abs_factor = 0.1

        self.maxiter       = kwargs.get('maxiter', max(100, 2*self.nlp.n))
        self.silent        = kwargs.get('silent',  False)
        self.ny            = kwargs.get('ny',      False)
        self.inexact       = kwargs.get('inexact', False)
        self.nyMax         = kwargs.get('nyMax',   5)
        self.opportunistic = kwargs.get('opportunistic', True)

        # Shortcuts for convenience
        self.lowerB  = numpy.array(self.nlp.lowerB, dtype=numpy.int32)
        self.upperB  = numpy.array(self.nlp.upperB, dtype=numpy.int32)
        self.rangeB  = numpy.array(self.nlp.rangeB, dtype=numpy.int32)

        # Number of dual variables
        self.ndual = self.nlowerB + self.nupperB + 2 * self.nrangeB
        
        # Set appropriate primal-dual starting point
        (self.x, self.z) = self.StartingPoint()

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
        self.hlen    = len( self.header )
        self.hline   = '-' * self.hlen + '\n'
        self.itFormat = '%-5d  '
        self.format='%8.1e  %7.1e  %7.1e  %7.1e  %5d  %8.1e  %8.1e  %4s\n'
        self.format0='%8.1e  %7.1e  %7.1e  %7.1e  %5d  %8.1e  %8.1e  %4s\n'
        self.printFrequency = 50

        # Optimality residuals, updated along the iterations
        self.d_res = None
        self.c_res = None
        self.p_res = None

        self.optimal = False
        self.debug = kwargs.get('debug', False)

        self.path = []

        return

    def StartingPoint( self, **kwargs ):
        """
        Compute a strictly feasible initial primal-dual estimate (x0, z0).
        """
        n = self.nlp.n
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar
        x0 = self.nlp.x0
        x = numpy.ones(n)
        z = numpy.ones(self.ndual)
        x[self.lowerB] = numpy.maximum(x0[self.lowerB],
                               (1 + numpy.sign(Lvar[self.lowerB]) * \
                               self.bound_rel_factor) * Lvar[self.lowerB] + \
                               self.bound_abs_factor )
        x[self.upperB] = numpy.maximum(x0[self.upperB],
                               (1 - numpy.sign(Uvar[self.upperB]) * \
                               self.bound_rel_factor) * Uvar[self.upperB] + \
                               self.bound_abs_factor )
        #for i in self.upperB:
        #    sgn = numpy.sign(Uvar[i])
        #    su = (1-sng*self.bound_rel_factor) * Uvar[i] + self.bound_abs_factor
        #    x[i] = min( x0[i], su )
        for i in self.rangeB:
            sgn = numpy.sign(Lvar[i])
            sl = (1+sgn*self.bound_rel_factor) * Lvar[i] + self.bound_abs_factor
            sgn = numpy.sign(Uvar[i])
            su = (1-sgn*self.bound_rel_factor) * Uvar[i] + self.bound_abs_factor
            x[i] = max( min( x0[i], su ), sl )
            #x[i] = 0.5 * (Lvar[i] + Uvar[i])
        #g = self.nlp.grad(x)
        #self.mu = numpy.linalg.norm( numpy.core.multiply(x, g), ord=numpy.inf )
        #self.mu = max( 1.0, self.mu )
        #z = numpy.empty( self.ndual, 'd' )
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
        mu = kwargs.get('mu', self.mu)
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar
        z = numpy.empty( self.ndual, 'd' )
        z[:self.nlowerB] = self.mu/(x[self.lowerB]-Lvar[self.lowerB])
        for i in self.lowerB:
            k = self.lowerB.index(i)
            z[k] = self.mu/(x[i]-Lvar[i])
        for i in self.upperB:
            k = self.nlowerB + self.upperB.index(i)
            z[k] = self.mu/(Uvar[i]-x[i])
        for i in self.rangeB:
            k = self.nlowerB + self.nupperB + self.rangeB.index(i)
            z[k] = self.mu/(x[i]-Lvar[i])
            k += self.nrangeB
            z[k] = self.mu/(Uvar[i]-x[i])
        return z

    def PDMerit( self, x, z, **kwargs ):
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

        This function returns (f, merit) where f is the value of the objective
        function at x and merit is the value of the primal-dual merit function.
        """
        mu = kwargs.get( 'mu', self.mu )
        eval_f = kwargs.get( 'eval_f', True )
        n = self.nlp.n
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar

        # Include contribution of objective function
        if eval_f:
            self.f = self.nlp.obj( x )
        merit = self.f

        # Include contribution of bound constraints
        for i in self.lowerB:
            k = self.lowerB.index(i)
            si = x[i] - Lvar[i]
            merit -= mu * log( si )
            merit += si * z[k]
            merit -= mu * log( si * z[k] )
        for i in self.upperB:
            k = self.nlowerB + self.upperB.index(i)
            si = Uvar[i] - x[i]
            merit -= mu * log( si )
            merit += si * z[k]
            merit -= mu * log( si * z[k] )
        for i in self.rangeB:   # Process two-sided bounds
            k = self.nlowerB + self.nupperB + self.rangeB.index(i) # Lower bound
            si = x[i] - Lvar[i]
            merit -= mu * log( si )
            merit += si * z[k]
            merit -= mu * log( si * z[k] )
            k += self.nrangeB                                      # Upper bound
            si = Uvar[i] - x[i]
            try:
                merit -= mu * log( si )
            except:
                print 'li, xi, ui, si = ', Lvar[i], x[i], Uvar[i], si
                sys.exit(1)
            merit += si * z[k]
            merit -= mu * log( si * z[k] )
        return merit

    def GradPDMerit( self, x, z, **kwargs ):
        """
        Evaluate the gradient of the primal-dual merit function at (x,z).
        See help(PDMerit) for a description of z.
        """
        mu = kwargs.get( 'mu', self.mu )
        check_optimal = kwargs.get( 'check_optimal', False )
        eval_g = kwargs.get( 'eval_g', True )

        n = self.nlp.n
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar
        g = numpy.empty( n + self.ndual, 'd' )

        # Gradient of the objective function
        if eval_g:
            self.gf = self.nlp.grad(x)
        g[:n] = self.gf

        # Check optimality conditions at this point if requested
        if check_optimal: self.optimal = self.AtOptimality(x, z)

        # Assemble the gradient with respect to x and z
        for i in self.lowerB:
            k = self.lowerB.index(i)
            g[i] -= 2.0 * mu/(x[i] - Lvar[i]) - z[k]
            g[n+k] = x[i] - Lvar[i] - mu/z[k]
        for i in self.upperB:
            k = self.nlowerB + self.upperB.index(i)
            g[i] += 2.0 * mu/(Uvar[i] - x[i]) - z[k]
            g[n+k] = Uvar[i] - x[i] - mu/z[k]
        for i in self.rangeB:
            k = self.nlowerB + self.nupperB + self.rangeB.index(i)
            g[i] -= 2.0 * mu/(x[i] - Lvar[i]) - z[k]
            g[n+k] = x[i] - Lvar[i] - mu/z[k]
            k += self.nrangeB
            g[i] += 2.0 * mu/(Uvar[i] - x[i]) - z[k]
            g[n+k] = Uvar[i] - x[i] - mu/z[k]

        return g

    def HessProd( self, x, z, p, **kwargs ):
        """
        Compute the matrix-vector product between the Hessian matrix of the
        primal-dual merit function at (x,z) and the vector p. See
        help(PDMerit) for a description of z. If there are b bounded variables
        and q two-sided bounds, the vector p should have length n+b+2q. The
        Hessian matrix has the general form

            [ H + 2 mu X^{-2}      I     ]
            [      I           mu Z^{-2} ].
        """
        mu = kwargs.get( 'mu', self.mu )
        n = self.nlp.n
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar
        Hp = numpy.zeros( n + self.ndual, 'd' )

        Hp[:n] = self.nlp.hprod( self.nlp.pi0, p[:n] )
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
        
    def PDHessProd( self, x, z, p, **kwargs ):
        """
        Compute the matrix-vector product between the modified Hessian matrix of
        the primal-dual merit function at (x,z) and the vector p. See
        help(PDMerit) for a description of z. If there are b bounded variables
        and q two-sided bounds, the vector p should have length n+b+2q.
        The Hessian matrix has the general form

            [ H + 2 X^{-1} Z      I     ]
            [      I           Z^{-1} X ].
        """
        mu = kwargs.get( 'mu', self.mu )
        n = self.nlp.n
        ndual = self.ndual
        N = self.nlowerB + self.nupperB + self.nrangeB
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar
        Hp = numpy.zeros( n + self.ndual, 'd' )

        Hp[:n] = self.nlp.hprod( self.nlp.pi0, p[:n] )
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

    def PDHessTemplate( self, **kwargs ):
        """
        Assemble the part of the modified Hessian matrix of the primal-dual
        merit function that is iteration independent. The function PDHess()
        fills in the blanks by updating the rest of the matrix.
        """
        n = self.nlp.n
        ndual = self.ndual
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar

        B = spmatrix.ll_mat_sym( n+ndual, n+2*ndual )

        for i in self.lowerB:
            k = self.lowerB.index(i)
            B[n+k,i] = 1.0
        for i in self.upperB:
            k = self.nlowerB + self.upperB.index(i)
            B[n+k,i] = -1.0
        for i in self.rangeB:
            k = self.nlowerB + self.nupperB + self.rangeB.index(i)
            B[n+k,i] = 1.0
            k += self.nrangeB
            B[n+k,i] = -1.0

        return B

    def PDHess( self, B, x, z, **kwargs ):
        """
        Assemble the modified Hessian matrix of the primal-dual merit function
        at (x,z). See help(PDMerit) for a description of z.
        The Hessian matrix has the general form

            [ H + 2 X^{-1} Z      I     ]
            [      I           Z^{-1} X ].
        """
        mu = kwargs.get( 'mu', self.mu )
        n = self.nlp.n
        ndual = self.ndual
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar

        B[:n,:n] = self.nlp.hess( x, self.nlp.pi0 )

        for i in self.lowerB:
            k = self.lowerB.index(i)
            B[i,i] +=  2.0 * z[k]/(x[i]-Lvar[i])
            B[n+k,n+k] = (x[i]-Lvar[i])/z[k]
        for i in self.upperB:
            k = self.nlowerB + self.upperB.index(i)
            B[i,i] += 2.0 * z[k]/(Uvar[i]-x[i])
            B[n+k,n+k] = (Uvar[i]-x[i])/z[k]
        for i in self.rangeB:
            k1 = self.nlowerB + self.nupperB + self.rangeB.index(i)
            k2 = k1 + self.nrangeB
            B[i,i] += 2.0 * ( z[k1]/(x[i]-Lvar[i]) + z[k2]/(Uvar[i]-x[i]) )
            B[n+k1,n+k1] = (x[i]-Lvar[i])/z[k1]
            B[n+k2,n+k2] = (Uvar[i]-x[i])/z[k2]

        return None

    def ftb( self, x, z, step, **kwargs ):
        """
        Compute the largest alpha in ]0,1] such that
            (x,z) + alpha * step >= (1 - tau) * (x,z)
        where 0 < tau < 1. By default, tau = 0.99.
        """
        tau = kwargs.get( 'tau', 0.9 )
        alpha = 1.0
        n = self.nlp.n
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar

        for i in self.lowerB:
            k = self.lowerB.index(i)
            if step[i] < 0.0: alpha = min(alpha, -tau * (x[i]-Lvar[i])/step[i])
            if step[n+k] < 0.0: alpha = min(alpha, -tau * z[k]/step[n+k])

        for i in self.upperB:
            k = self.nlowerB + self.upperB.index(i)
            if step[i] > 0.0: alpha = min(alpha, tau * (Uvar[i]-x[i])/step[i])
            if step[n+k] < 0.0: alpha = min(alpha, -tau * z[k]/step[n+k])

        for i in self.rangeB:
            k = self.nlowerB + self.nupperB + self.rangeB.index(i)
            if step[i] < 0.0:
                alpha = min(alpha, -tau * (x[i]-Lvar[i])/step[i])
            if step[i] > 0.0:
                alpha = min(alpha,  tau * (Uvar[i]-x[i])/step[i])
            if step[n+k] < 0.0: alpha = min(alpha, -tau * z[k]/step[n+k])
            k += self.nrangeB
            if step[n+k] < 0.0: alpha = min(alpha, -tau * z[k]/step[n+k])

        return alpha

    def Precon( self, v, **kwargs ):
        """
        Generic preconditioning method---must be overridden
        """
        return v

    def UpdatePrecon( self, **kwargs ):
        """
        Override this method for preconditioners that need updating,
        e.g., a limited-memory BFGS preconditioner.
        """
        return None

    def SolveInner( self, **kwargs ):
        """
        Perform a series of inner iterations so as to minimize the primal-dual
        merit function with the current value of the barrier parameter to within
        some given tolerance. The only optional argument recognized is

            stoptol     stopping tolerance (default: 10 mu).
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
            psi = self.PDMerit(x, z)
            self.g = self.GradPDMerit(x, z, check_optimal=True, eval_g=True)
        else:
            psi = self.PDMerit(x, z, eval_f=False)
            self.g = self.GradPDMerit(x, z, check_optimal=True, eval_g=True)
        self.gNorm = numpy.linalg.norm(self.g, ord = self.ord)
        if self.optimal: return
        
        # Reset initial trust-region radius
        self.TR.Delta = 0.1 * self.gNorm #max( 10.0, self.gNorm )

        # Set inner iteration stopping tolerance
        stopTol = kwargs.get( 'stopTol', 1.0e-3 * self.gNorm )

        # Initialize Hessian matrix
        B = self.PDHessTemplate()

        #(dRes, cRes, pRes) = self.OptimalityResidual(x, z, mu = self.mu)
        #maxRes = max(numpy.linalg.norm(dRes, ord=numpy.inf), cRes, pRes)
        #while (maxRes > stopTol) and (self.iter <= self.maxiter):

        finished = (self.gNorm <= stopTol) or (self.iter > self.maxiter)

        while not finished:

            #self.path.append( list(self.x) )

            # Print out header every so many iterations
            if self.iter % self.printFrequency == 0 and not self.silent:
                sys.stdout.write( self.hline )
                sys.stdout.write( self.header )
                sys.stdout.write( self.hline )
    
            if not self.silent:
                if inner_iter == 0:
                    sys.stdout.write(('*' + self.itFormat) % self.iter)
                else:
                    sys.stdout.write((' ' + self.itFormat) % self.iter)
                sys.stdout.write(self.format % (self.f,
                                 max(self.d_res, self.c_res, self.p_res),
                                 self.mu, alpha, niter, rho,
                                 self.TR.Delta, status ) )

            if self.debug:
                self._debugMsg('g = ' + numpy.str(g))
                self._debugMsg('gNorm = ' + str(self.gNorm))
                self._debugMsg('stopTol = ' + str(stopTol))

            # Save current gradient
            if self.save_g:
                self.g_old = self.g.copy()

            # Set stopping tolerance for trust-region subproblem
            if self.inexact:
                cgtol = max( 1.0e-8, min( 0.1 * cgtol, sqrt(self.gNorm) ))
                if self.debug: self._debugMsg('cgtol = ' + str(cgtol))


            # Update Hessian matrix with current iteration information
            self.PDHess(B,x,z)

            # Iteratively minimize the quadratic model in the trust region
            # m(s) = <g, s> + 1/2 <s, Hs>
            # Note that m(s) does not include f(x): m(0) = 0.            
            self.solver = self.TrSolver(
                               self.g,
                               #matvec = lambda v: self.PDHessProd(x,z,v),
                               H = B,
                               prec = self.Precon,
                               radius = self.TR.Delta,
                               reltol = cgtol #,
                               #btol = .9,
                               #cur_iter = numpy.concatenate( (x,z) )
                               )
            self.solver.Solve()

            step  = self.solver.step
            snorm = self.solver.stepNorm
            niter = self.solver.niter

            if self.debug:
                self._debugMsg('x = ' + numpy.str(x))
                self._debugMsg('z = ' + numpy.str(z))
                self._debugMsg('step = ' + numpy.str(step))

            # Obtain model value at next candidate
            m = self.solver.m
            self.cgiter += niter

            # Compute maximal step to the boundary and next candidate
            alpha = self.ftb(x, z, step)
            x_trial = x + alpha * step[:n]
            z_trial = z + alpha * step[n:]
            #x_trial = x + step[:n]
            #z_trial = z + step[n:]
            psi_trial = self.PDMerit( x_trial, z_trial )
            rho  = self.TR.Rho( psi, psi_trial, m )

            # Accept or reject next candidate
            status = 'Rej'
            if rho >= self.TR.eta1:
                self.TR.UpdateRadius( rho, snorm )
                x = x_trial
                z = z_trial
                psi = psi_trial
                self.g = self.GradPDMerit(x, z, check_optimal=True, eval_g=True)
                try:
                    self.gNorm = numpy.linalg.norm(self.g, ord=self.ord)
                except:
                    print 'Offending g = ', self.g
                    sys.exit(1)
                if self.optimal:
                    finished = True
                    continue
                status = 'Acc'
            else:
                if self.ny: # Backtracking linesearch a la "Nocedal & Yuan"
                    slope = numpy.dot(self.g, step)
                    target = psi + 1.0e-4 * alpha * slope
                    j = 0

                    while (psi_trial >= target) and (j < self.nyMax):
                        alpha /= 1.2
                        target = psi + 1.0e-4 * alpha * slope
                        x_trial = x + alpha * step[:n]
                        z_trial = z + alpha * step[n:]
                        psi_trial = self.PDMerit( x_trial, z_trial )
                        j += 1

                    if self.opportunistic or (j < self.nyMax):
                        x = x_trial
                        z = z_trial
                        psi = psi_trial
                        self.g = self.GradPDMerit(x, z, check_optimal=True, eval_g=True)
                        self.gNorm = numpy.linalg.norm(self.g, ord=self.ord)
                        if self.optimal:
                            finished = True
                            continue
                        self.TR.Delta = alpha * snorm
                        status = 'N-Y'

                    else:
                        self.TR.UpdateRadius(rho, snorm)

                else:
                    self.TR.UpdateRadius(rho, snorm)

            self.UpdatePrecon()
            self.iter += 1
            inner_iter += 1
            finished = (self.gNorm <= stopTol) or (self.iter > self.maxiter)
            if self.debug: sys.stderr.write('\n')

            #(dRes, cRes, pRes) = self.OptimalityResidual(x, z, mu = self.mu)
            #maxRes = max(numpy.linalg.norm(dRes, ord=numpy.inf), cRes, pRes)

        # Store final iterate
        (self.x, self.z) = (x, z)
        self.psi = psi
        return

    def SolveOuter( self, **kwargs ):

        nlp = self.nlp
        n = nlp.n

        print self.x
        
        # Measure solve time
        t = cputime()

        # Solve sequence of inner iterations
        while (not self.optimal) and (self.mu >= self.mu_min) and \
                (self.iter <= self.maxiter):
            self.SolveInner( stopTol = max(1.0e-7, 5*self.mu) )
            self.z = self.PrimalMultipliers(self.x)
            self.optimal = self.AtOptimality(self.x, self.z)
            self.UpdateMu()

        self.tsolve = cputime() - t    # Solve time
        if self.optimal:
            print 'First-order optimal solution found'
        elif self.iter >= self.maxiter:
            print 'Maximum number of iterations reached'
        else:
            print 'Reached smallest allowed value of barrier parameter'
        return

    def UpdateMu( self, **kwargs ):
        """
        Update the barrier parameter before the next round of inner iterations.
        """
        res = max(self.d_res, self.c_res)
        #guard = min( res/5.0, res**(1.5) )
        #if guard <= self.mu/5.0:
        #    self.mu = guard
        #else:
        #    self.mu = min( self.mu/5.0, self.mu**(1.5) )
        #self.mu = min( self.mu/5.0, self.mu**(1.5) )
        self.mu = min(self.mu/2.0, max(self.mu/5.0, res/5.0))
        return None
        
    def OptimalityResidual( self, x, z, **kwargs ):
        """
        Compute optimality residual for bound-constrained problem
           [ g - z ]
           [  Xz   ],
        where g is the gradient of the objective function.
        """
        gradf = kwargs.get('gradf', self.gf)
        mu = kwargs.get('mu', 0.0)
        n = self.nlp.n
        Lvar = self.nlp.Lvar
        Uvar = self.nlp.Uvar
        
        # Compute dual feasibility and complementarity residuals
        d_res = gradf.copy()
        c_res = 0.0
        p_res = 0.0
        for i in self.lowerB:
            k = self.lowerB.index(i)
            d_res[i] -= z[k]
            c_res = max( c_res, abs((x[i] - Lvar[i]) * z[k] - mu) )
            p_res = max( p_res, max(0.0, Lvar[i]-x[i]) )
        for i in self.upperB:
            k = self.nlowerB + self.upperB.index(i)
            d_res[i] += z[k]
            c_res = max( c_res, abs((Uvar[i] - x[i]) * z[k] - mu) )
            p_res = max( p_res, max(0.0, x[i]-Uvar[i]) )
        for i in self.rangeB:
            k = self.nlowerB + self.nupperB + self.rangeB.index(i)
            d_res[i] -= z[k]
            c_res = max( c_res, abs((x[i] - Lvar[i]) * z[k] - mu) )
            p_res = max( p_res, max(0.0, Lvar[i]-x[i]) )
            k += self.nrangeB
            d_res[i] += z[k]
            c_res = max( c_res, abs((Uvar[i] - x[i]) * z[k] - mu) )
            p_res = max( p_res, max(0.0, x[i]-Uvar[i]) )
        
        return (d_res, c_res, p_res)

    def AtOptimality( self, x, z, **kwargs ):
        """
        Compute the infinity norm of the optimality residual at (x,z). See
        help(OptimalityResidual) for more information.
        """
        # We could save computation by computing the infinity norm directly
        # without forming the residual vector.
        nlp = self.nlp
        n = nlp.n
        (dual_res, self.c_res, self.p_res) = self.OptimalityResidual(x,z)
        self.d_res = numpy.linalg.norm( dual_res, ord = self.ord )

        if self.d_res <= nlp.stop_d and self.c_res <= nlp.stop_c and \
           self.p_res <= nlp.stop_p:
            return True
        return False

    def _debugMsg(self, msg):
        sys.stderr.write('Debug:: ' + msg + '\n')
        return None


if __name__ == '__main__':

    import amplpy
    import trustregion
    import sys
    import pylab

    # Set printing standards for arrays
    numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

    prob = sys.argv[1]

    # Initialize problem
    nlp = amplpy.AmplModel( prob )
    #nlp.stop_p = 1.0e-12
    #nlp.stop_d = 1.0e-12
    #nlp.stop_c = 1.0e-12

    # Initialize trust region framework
    TR = trustregion.TrustRegionFramework( Delta = 1.0,
                                           eta1 = 0.0001,
                                           eta2 = 0.95,
                                           gamma1 = 1.0/3,
                                           gamma2 = 2.5 )

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
    g = nlp.grad( TRIP.x )
    (d_res, c_res, p_res) = TRIP.OptimalityResidual(TRIP.x, TRIP.z, gradf = g)
    d_res_norm = numpy.linalg.norm(d_res, ord=TRIP.ord)
    TRIP.nlp.stop_d = max(TRIP.nlp.stop_d, 1.0e-8 * max(1.0, d_res_norm))
    TRIP.nlp.stop_c = max(TRIP.nlp.stop_c, 1.0e-6 * max(1.0, c_res))
    print 'Target tolerances: (%7.1e, %7.1e)' % \
        (TRIP.nlp.stop_d, TRIP.nlp.stop_c)

    # Reset initial value of mu to a more sensible value
    TRIP.mu = 10.0 #max(d_res_norm, c_res) #* 100
    #TRIP.mu = numpy.linalg.norm( numpy.core.multiply(TRIP.x, g), ord=numpy.inf)
    
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
    print 'Primal feasibility error  = %15.7e' % TRIP.p_res
    print 'Dual   feasibility error  = %15.7e' % TRIP.d_res
    print 'Complementarity error     = %15.7e' % TRIP.c_res
    print 'Number of function evals  = %d' % TRIP.nlp.feval
    print 'Number of gradient evals  = %d' % TRIP.nlp.geval
    print 'Number of Hessian  evals  = %d' % TRIP.nlp.Heval
    print 'Number of matvec products = %d' % TRIP.nlp.Hprod
    print 'Final objective value     = %15.7e' % TRIP.f
    print 'Solution time: ', TRIP.tsolve

    nlp.close()

    #print TRIP.path[0], TRIP.path[-1]
    #pylab.plot( [TRIP.path[i][0] for i in range(len(TRIP.path))],
    #            [TRIP.path[i][1] for i in range(len(TRIP.path))],
    #            '.-', linewidth=2 )
    #pylab.show()

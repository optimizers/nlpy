# Long-step primal-dual interior-point method for linear programming
# From Algorithm IPF on p.110 of Stephen J. Wright's book
# "Primal-Dual Interior-Point Methods", SIAM ed., 1997.
# The method uses the augmented system formulation. These systems
# are solved using PyMa27 or PyMa57.
#
# D. Orban, Montreal 2004. Revised September 2009.

try:                            # To solve augmented systems
    from nlpy.linalg.pyma57 import PyMa57Context as LBLContext
except:
    from nlpy.linalg.pyma27 import PyMa27Context as LBLContext
from nlpy.tools import norms
from nlpy.tools import sparse_vector_class as sv
from nlpy.tools.timing import cputime

from pysparse import spmatrix
import numpy as np
import sys

class LPInteriorPointSolver:

    def __init__(self, lp, **kwargs):
        """
        Solve the linear program

           minimize     c'x
           subject to   Ax = b, x >= 0,

        where c is a sparse cost vector, A is a sparse constraint matrix and b is
        a dense right-hand side.

        The problem MUST be in standard form. No prior conversion is made.

        The values returned are

         x..............the final iterate
         y..............the final value of the Lagrange multipliers associated
                        to Ax=b
         s..............the final value of the Lagrange multipliers associated
                        to x>=0
         obj_value......the final cost
         iter...........the total number of iterations
         residual.......the final relative residual
         solve_time.....the time to solve the LP
         status.........a string describing the exit status.

         Keyword arguments may be passed to change options. They are as follows:

         debug                True/False (default: False)
              Will output some debugging info during the run.
              
         PredictorCorrector   True/False (default: True)
              Uses Mehrotra's predictor-corrector method. If set
              to False, a classical long-step method will be used.
              In both cases, the augmented system formulation is
              used and systems are solved with MA27.

         itermax              integer (default: 100)
              Sets the maximum number of iterations.

         tolerance            float (default: 1.0e-6)
              The algorithm terminates successfully as soon as the
              relative residual of the KKT conditions is smaller
              than 'tolerance' in infinity norm.          
        """
        self.lp = lp

        self.debug = kwargs.get('debug', False)

        self.A = lp.A()               # Constraint matrix
        m, n = self.A.shape
        
        # Residuals
        self.ry = np.zeros(n)
        self.rb = np.zeros(m)
        self.rc = np.zeros(n)

        self.b = lp.cons(self.ry)     # Right-hand side
        self.c = lp.cost()            # Cost vector
        self.c0 = 0 #lp.obj(self.ry)     # Constant term in objective
        self.normbc = 1 + max(norms.norm_infty(self.b), sv.norm_infty(self.c))

        # Initialize augmented matrix
        self.H = spmatrix.ll_mat_sym(n+m, n + self.A.nnz)
        self.H[n:,:n] = self.A

        fmt_hdr = '%-4s  %9s  %-8s  %-7s  %-4s  %-4s  %-8s'
        self.header = fmt_hdr % ('Iter', 'Cost', 'Residual', 'Mu', 'AlPr', 'AlDu', 'IR Resid')
        self.format1 = '%-4d  %9.2e  %-8.2e'
        self.format2 = '  %-7.1e  %-4.2f  %-4.2f  %-8.2e'

        return

    def solve(self, x=None, **kwargs):

        lp = self.lp
        debug = self.debug
        itermax = kwargs.get('itermax', 10*lp.n)
        tolerance = kwargs.get('tolerance', 1.0e-6)
        PredictorCorrector = kwargs.get('PredictorCorrector', True)

        # Transfer pointers for convenience
        m, n = self.A.shape
        A = self.A ; b = self.b ; c = self.c ; H = self.H
        ry = self.ry ; rb = self.rb ; rc = self.rc

        # Initialize
        if x is None: x = self.set_initial_guess(self.lp, **kwargs)
        z = x.copy()
        y = np.zeros(m)
        dz = np.zeros(n)
        rhs = np.zeros(n+m)
        finished = False
        iter = 0

        sys.stdout.write(self.header + '\n')
        sys.stdout.write('-' * len(self.header) + '\n')

        setup_time = cputime()

        # Main loop
        while not finished:

            if debug:
                sys.stderr.write(' z = ', z)
                sys.stderr.write(' x = ', x)

            # Compute residuals
            #  rb = b - Ax
            A.matvec(x, rb)
            rb = b - rb
            #  rc = Xs
            rc = x*z
            #  ry = A^T y + s - c
            A.matvec_transp(y, ry)
            ry += z
            for k in c.keys(): ry[k] -= c[k]
    
            nrb = norms.norm_infty(rb)
            nrc = norms.norm_infty(rc)
            nry = norms.norm_infty(ry)
            if debug:
                sys.stderr.write(' nrb, nrc, nry = ', nrb, nrc, nry)
            residual = max(nrb, nrc, nry) / self.normbc
            obj_value = sv.dot(c, x)

            sys.stdout.write(self.format1 % (iter, obj_value, residual))
    
            if residual <= tolerance:
                status = 'Optimal solution found'
                finished = True
            elif  iter > itermax:
                status = 'Max number of iterations reached'
                finished = True
            else:
                # Solve the linear system
                #
                # [ -X^{-1} Z    A^T ]  [ dx ] = - [ ry - X^{-1} rc ]
                # [   A          0   ]  [ dy ]     [ -rb             ]
                #
                # and recover  dz = -X^{-1} (rc + Z dx)
                # with rc = Xs - sigma * mu e.

                # Compute augmented matrix and factorize it
                H.put(-z/x)  # In places (1,1), (2,2), ..., (n,n) by default
                LBL = LBLContext(H)

                # Compute mu
                mu = sum(rc)/n
                tau = max(.9995, 1.0-mu)

                if PredictorCorrector:
                    # Use Mehrotra predictor-corrector method
                    # Compute affine-scaling step, i.e. with sigma = 0
                    rhs[:n] = -(ry - z)
                    rhs[n:] = rb
                    (step, nres, neig) = self.solveSystem(LBL, rhs)
                    # Recover dx and dz
                    dx = step[:n]
                    dz = -z * (1 + dx/x)
                    # Compute primal and dual stepsizes
                    alpha_p = self.maxStepLength(x, dx)
                    alpha_d = self.maxStepLength(z, dz)
                    # Estimate duality gap after affine-scaling step
                    muAff = np.dot(x + alpha_p * dx, z + alpha_d * dz)/n
                    sigma = (muAff/mu)**3

                    if debug:
                        sys.stderr.write(' alpha_pAFF, alpha_dAFF, muAFF, sigma =', (alpha_p, alpha_d, muAff, sigma))

                    # Incorporate predictor information for corrector step
                    rc += dx*dz
                else:
                    # Use long-step method
                    # Compute right-hand side
                    sigma = min(0.1, 100*mu)

                # Assemble right-hand side
                rc -= sigma * mu
                ry -= rc/x
                rhs[:n] = -ry
                rhs[n:] = rb

                # Solve augmented system
                (step, nres, neig) = self.solveSystem(LBL, rhs)

                # Recover step
                dx = step[:n]
                dy = step[n:]
                dz = -(rc + z*dx)/x

                # Compute primal and dual stepsizes
                alpha_p = tau * self.maxStepLength(x, dx)
                alpha_d = tau * self.maxStepLength(z, dz)

                sys.stdout.write(self.format2 % (mu, alpha_p, alpha_d, nres))
                sys.stdout.write('\n')

                # Update iterates
                x += alpha_p * dx
                y += alpha_d * dy
                z += alpha_d * dz

                iter += 1

        solve_time = cputime() - setup_time
        sys.stdout.write('\n')
        sys.stdout.write('-' * len(self.header) + '\n')

        self.x = x
        self.y = y
        self.z = z
        self.obj_value = obj_value + self.c0
        self.iter = iter
        self.residual = residual
        self.solve_time = solve_time
        self.status = status
        return

    def set_initial_guess(self, lp, **kwargs):
        # Compute initial guess
        bigM = max(self.A.norm('inf'), self.normbc)
        x = 100 * bigM * np.ones(lp.n)
        return x

    def maxStepLength(self, x, d):
        """
        Returns the max step length from x to the boundary
        of the nonnegative orthant in the direction d
        alpha_max = min [ 1, min { -x[i]/d[i] | d[i] < 0 } ].
        Note that 0 < alpha_max <= 1.
        """
        whereneg = np.nonzero(np.where(d < 0, d, 0))[0]
        dxneg = [-x[i]/d[i] for i in whereneg]
        dxneg.append(1)
        return min(dxneg)

    def solveSystem(self, LBL, rhs, itref_threshold=1.0e-5, nitrefmax=5):
        LBL.solve(rhs)
        nr = norms.norm2(LBL.residual)
        # If residual not small, perform iterative refinement
        LBL.refine(rhs, tol=itref_threshold, nitref=nitrefmax)
        nr1 = norms.norm2(LBL.residual)
        return (LBL.x, nr1, LBL.neig)


class RegLPInteriorPointSolver(LPInteriorPointSolver):

    def __init__(self, lp, **kwargs):
        LPInteriorPointSolver.__init__(self, lp, **kwargs)
        self.regpr = kwargs.get('regpr', 1.0)
        self.regdu = kwargs.get('regdu', 1.0)
        return

    def solve(self, x=None, **kwargs):

        lp = self.lp
        debug = self.debug
        itermax = kwargs.get('itermax', 10*lp.n)
        tolerance = kwargs.get('tolerance', 1.0e-6)
        PredictorCorrector = kwargs.get('PredictorCorrector', True)

        # Transfer pointers for convenience
        m, n = self.A.shape
        A = self.A ; b = self.b ; c = self.c ; H = self.H
        ry = self.ry ; rb = self.rb ; rc = self.rc
        regpr = self.regpr ; regdu = self.regdu

        # Initialize
        if x is None: x = self.set_initial_guess(self.lp, **kwargs)
        z = x.copy()
        y = np.zeros(m)
        dz = np.zeros(n)
        dr = np.zeros(m)
        ds = np.zeros(n)
        rhs = np.zeros(n+m)
        finished = False
        iter = 0

        sys.stdout.write(self.header + '\n')
        sys.stdout.write('-' * len(self.header) + '\n')

        setup_time = cputime()

        # Main loop
        while not finished:

            if debug:
                sys.stderr.write(' z = ', z)
                sys.stderr.write(' x = ', x)

            # Compute residuals
            #  rb = b - Ax
            A.matvec(x, rb)
            rb = b - rb
            #  rc = Xs
            rc = x*z
            #  ry = A^T y + s - c
            A.matvec_transp(y, ry)
            ry += z
            for k in c.keys(): ry[k] -= c[k]
    
            nrb = norms.norm_infty(rb)
            nrc = norms.norm_infty(rc)
            nry = norms.norm_infty(ry)
            if debug:
                sys.stderr.write(' nrb, nrc, nry = ', nrb, nrc, nry)
            residual = max(nrb, nrc, nry) / self.normbc
            obj_value = sv.dot(c, x)

            sys.stdout.write(self.format1 % (iter, obj_value, residual))
    
            if residual <= tolerance:
                status = 'Optimal solution found'
                finished = True
            elif  iter > itermax:
                status = 'Max number of iterations reached'
                finished = True
            else:
                # Solve the linear system
                #
                # [ -(X^{-1} Z + rho I)    A'    ] [ dx ] = - [ ry - X^{-1} rc ]
                # [   A                  delta I ] [ dy ]     [ -rb             ]
                #
                # and recover  dz = -X^{-1} (rc + Z dx)
                # with rc = Xs - sigma * mu e.

                # Compute augmented matrix and factorize it
                H.put(-z/x - regpr)  # In places (1,1), ..., (n,n) by default
                H.put(regdu, range(n,n+m))
                LBL = LBLContext(H)

                # Compute mu
                mu = sum(rc)/n
                tau = max(.9995, 1.0-mu)

                if PredictorCorrector:
                    # Use Mehrotra predictor-corrector method
                    # Compute affine-scaling step, i.e. with sigma = 0
                    rhs[:n] = -(ry - z)
                    rhs[n:] = rb
                    (step, nres, neig) = self.solveSystem(LBL, rhs)
                    # Recover dx and dz
                    dx = step[:n]
                    dz = -z * (1 + dx/x)
                    # Compute primal and dual stepsizes
                    alpha_p = self.maxStepLength(x, dx)
                    alpha_d = self.maxStepLength(z, dz)
                    # Estimate duality gap after affine-scaling step
                    muAff = np.dot(x + alpha_p * dx, z + alpha_d * dz)/n
                    sigma = (muAff/mu)**3

                    if debug:
                        sys.stderr.write(' alpha_pAFF, alpha_dAFF, muAFF, sigma =', (alpha_p, alpha_d, muAff, sigma))

                    # Incorporate predictor information for corrector step
                    rc += dx*dz
                else:
                    # Use long-step method
                    # Compute right-hand side
                    sigma = min(0.1, 100*mu)

                # Assemble right-hand side
                rc -= sigma * mu
                ry -= rc/x
                rhs[:n] = -ry
                rhs[n:] = rb

                # Solve augmented system
                (step, nres, neig) = self.solveSystem(LBL, rhs)

                # Recover step
                dx = step[:n]
                dy = step[n:]
                dz = -(rc + z*dx)/x

                # Compute primal and dual stepsizes
                alpha_p = tau * self.maxStepLength(x, dx)
                alpha_d = tau * self.maxStepLength(z, dz)

                sys.stdout.write(self.format2 % (mu, alpha_p, alpha_d, nres))
                sys.stdout.write('\n')

                # Update iterates
                x += alpha_p * dx
                y += alpha_d * dy
                z += alpha_d * dz

                regpr /= 2
                regdu /= 2

                iter += 1

        solve_time = cputime() - setup_time
        sys.stdout.write('\n')
        sys.stdout.write('-' * len(self.header) + '\n')

        self.x = x
        self.y = y
        self.z = z
        self.obj_value = obj_value + self.c0
        self.iter = iter
        self.residual = residual
        self.solve_time = solve_time
        self.status = status
        return



############################################################

def usage():
    sys.stderr.write('Use: %-s problem_name\n' % sys.argv[0])
    sys.stderr.write(' where problem_name represents a linear program\n')


if __name__ == '__main__':

    from nlpy.model import SlackFramework

    if len(sys.argv) < 2:
        usage()
        sys.exit(-1)

    probname = sys.argv[1]

    lp = SlackFramework(probname)
    if not lp.islp():
        sys.stderr.write('Input problem must be a linear program\n')
        sys.exit(1)

    A = lp.A()
    print 'm, n = ', (lp.m, lp.n)
    print 'A = '
    print A
    print 'A.shape = ', A.shape

    #lpSolver = LPInteriorPointSolver(lp)
    lpSolver = RegLPInteriorPointSolver(lp)
    lpSolver.solve()

    print 'Final x: ', lpSolver.x
    print 'Final y: ', lpSolver.y
    print 'Final z: ', lpSolver.z

    sys.stdout.write('\n' + lpSolver.status + '\n')
    sys.stdout.write(' #Iterations: %-d\n' % lpSolver.iter)
    sys.stdout.write(' RelResidual: %7.1e\n' % lpSolver.residual)
    sys.stdout.write(' Final cost : %7.1e\n' % lpSolver.obj_value)
    sys.stdout.write(' Solve time : %6.2fs\n' % lpSolver.solve_time)

    # End
    lp.close()

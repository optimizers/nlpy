# -*- coding: utf-8 -*-
# Long-step primal-dual interior-point method for linear programming
# From Algorithm IPF on p.110 of Stephen J. Wright's book
# "Primal-Dual Interior-Point Methods", SIAM ed., 1997.
# The method uses the augmented system formulation. These systems
# are solved using PyMa27 or PyMa57.
#
# D. Orban, Montreal 2004. Revised September 2009.

from nlpy.model import SlackFramework
try:                            # To solve augmented systems
    from nlpy.linalg.pyma57 import PyMa57Context as LBLContext
except:
    from nlpy.linalg.pyma27 import PyMa27Context as LBLContext
from nlpy.linalg.scaling import mc29ad
from nlpy.tools.norms import norm2, norm_infty
from nlpy.tools import sparse_vector_class as sv
from nlpy.tools.timing import cputime

from pysparse.sparse import spmatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix
import numpy as np
from math import sqrt
import sys
import pdb


class RegLPInteriorPointSolver:

    def __init__(self, lp, **kwargs):
        """
        Solve a linear program of the form

           minimize    c' x
           subject to  A1 x + A2 s = b,                                 (LP)
                       s >= 0,

        where the variables x are the original problem variables and s are
        slack variables. Any linear program may be converted to the above form
        by instantiation of the `SlackFramework` class. The conversion to the
        slack formulation is mandatory in this implementation.

        The method is a variant of Mehrotra's predictor-corrector method where
        steps are computed by solving the primal-dual system in augmented form.

        Primal and dual regularization parameters may be specified by the user
        via the opional keyword arguments `regpr` and `regdu`. Both should be
        positive real numbers and should not be "too large". By default they are
        set to 1.0 and updated at each iteration.

        If `scale` is set to `True`, (LP) is scaled automatically prior to
        solution so as to equilibrate the rows and columns of the constraint
        matrix [A1 A2].

        Advantages of this method are that it is not sensitive to dense columns
        in A, no special treatment of the unbounded variables x is required, and
        a sparse symmetric quasi-definite system of equations is solved at each
        iteration. The latter, although indefinite, possesses a Cholesky-like
        factorization. Those properties makes the method typically more robust
        that a standard predictor-corrector implementation and the linear system
        solves are often much faster than in a traditional interior-point method
        in augmented form.
        """

        if not isinstance(lp, SlackFramework):
            msg = 'Input problem must be an instance of SlackFramework'
            raise ValueError, msg

        scale = kwargs.get('scale', True)
        self.verbose = kwargs.get('verbose', True)
        self.stabilize = kwargs.get('stabilize', False)

        self.lp = lp
        self.A = lp.A()               # Constraint matrix
        if not isinstance(self.A, PysparseMatrix):
            self.A = PysparseMatrix(matrix=self.A)

        m, n = self.A.shape
        # Record number of slack variables in LP
        self.nSlacks  = lp.n - lp.original_n

        # Constant vectors
        zero = np.zeros(n)
        self.b = -lp.cons(zero)     # Right-hand side
        self.c0 = lp.obj(zero)      # Constant term in objective
        self.c =  lp.grad(zero[:lp.original_n]) #lp.cost()  # Cost vector

        # Apply in-place problem scaling if requested.
        self.prob_scaled = False
        if scale:
            self.t_scale = cputime()
            self.scale()
            self.t_scale = cputime() - self.t_scale

        self.normb  = norm2(self.b)
        self.normc  = norm2(self.c)
        self.normbc = 1 + max(self.normb, self.normc)

        # Initialize augmented matrix
        self.H = PysparseMatrix(size=n+m,
                                sizeHint=n+m+self.A.nnz,
                                symmetric=True)

        # We perform the analyze phase on the augmented system only once.
        # self.LBL will be initialized in set_initial_guess().
        self.LBL = None

        self.regpr = kwargs.get('regpr', 1.0) ; self.regpr_min = 1.0e-8
        self.regdu = kwargs.get('regdu', 1.0) ; self.regdu_min = 1.0e-8

        # Check input parameters.
        if self.regpr < 0.0: self.regpr = 0.0
        if self.regdu < 0.0: self.regdu = 0.0

        # Dual regularization is necessary for stabilization.
        if self.regdu == 0.0:
            sys.stderr.write('Warning: No dual regularization in effect\n')
            sys.stderr.write('         Stabilization has been turned off\n')
            self.stabilize = False

        # Initialize format strings for display
        fmt_hdr = '%-4s  %9s' + '  %-8s'*6 + '  %-7s  %-4s  %-4s' + '  %-8s'*8
        self.header = fmt_hdr % ('Iter', 'Cost', 'pResid', 'dResid', 'cResid',
                                 'rGap', 'qNorm', 'rNorm', 'Mu', 'AlPr', 'AlDu',
                                 'LS Resid', 'RegPr', 'RegDu', 'Rho q', 'Del r',
                                 'Min(s)', 'Min(z)', 'Max(s)')
        self.format1  = '%-4d  %9.2e'
        self.format1 += '  %-8.2e' * 6
        self.format2  = '  %-7.1e  %-4.2f  %-4.2f'
        self.format2 += '  %-8.2e' * 8 + '\n'

        if self.verbose: self.display_stats()

        return

    def display_stats(self):
        """
        Display vital statistics about the input problem.
        """
        import os
        lp = self.lp
        w = sys.stdout.write
        w('\n')
        w('Problem Path: %s\n' % lp.name)
        w('Problem Name: %s\n' % os.path.basename(lp.name))
        w('Number of problem variables: %d\n' % lp.original_n)
        w('Number of free variables: %d\n' % lp.nfreeB)
        w('Number of problem constraints excluding bounds: %d\n' %lp.original_m)
        w('Number of slack variables: %d\n' % (lp.n - lp.original_n))
        w('Adjusted number of variables: %d\n' % lp.n)
        w('Adjusted number of constraints excluding bounds: %d\n' % lp.m)
        w('Number of nonzeros in constraint matrix: %d\n' % self.A.nnz)
        w('Constant term in objective: %8.2e\n' % self.c0)
        w('Cost vector norm: %8.2e\n' % self.normc)
        w('Right-hand side norm: %8.2e\n' % self.normb)
        w('Initial primal regularization: %8.2e\n' % self.regpr)
        w('Initial dual   regularization: %8.2e\n' % self.regdu)
        if self.prob_scaled:
            w('Time for scaling: %6.2fs\n' % self.t_scale)
        w('\n')
        return

    def scale(self, **kwargs):
        """
        Equilibrate the constraint matrix of the linear program. Equilibration
        is done by first dividing every row by its largest element in absolute
        value and then by dividing every column by its largest element in
        absolute value. In effect the original problem

          minimize c'x  subject to  A1 x + A2 s = b, x >= 0

        is converted to

          minimize (Cc)'x  subject to  R A1 C x + R A2 C s = Rb, x >= 0,

        where the diagonal matrices R and C operate row and column scaling
        respectively.

        Upon return, the matrix A and the right-hand side b are scaled and the
        members `row_scale` and `col_scale` are set to the row and column
        scaling factors.

        The scaling may be undone by subsequently calling :meth:`unscale`. It is
        necessary to unscale the problem in order to unscale the final dual
        variables. Normally, the :meth:`solve` method takes care of unscaling
        the problem upon termination.
        """
        w = sys.stdout.write
        m, n = self.A.shape
        row_scale = np.zeros(m)
        col_scale = np.zeros(n)
        (values,irow,jcol) = self.A.find()

        if self.verbose:
            w('Smallest and largest elements of A prior to scaling: ')
            w('%8.2e %8.2e\n' % (np.min(np.abs(values)),np.max(np.abs(values))))

        # Find row scaling.
        for k in range(len(values)):
            row = irow[k]
            val = abs(values[k])
            row_scale[row] = max(row_scale[row], val)
        row_scale[row_scale == 0.0] = 1.0

        if self.verbose:
            w('Largest row scaling factor = %8.2e\n' % np.max(row_scale))

        # Apply row scaling to A and b.
        values /= row_scale[irow]
        self.b /= row_scale

        # Find column scaling.
        for k in range(len(values)):
            col = jcol[k]
            val = abs(values[k])
            col_scale[col] = max(col_scale[col], val)
        col_scale[col_scale == 0.0] = 1.0

        if self.verbose:
            w('Largest column scaling factor = %8.2e\n' % np.max(col_scale))

        # Apply column scaling to A and c.
        values /= col_scale[jcol]
        self.c[:self.lp.original_n] /= col_scale[:self.lp.original_n]

        if self.verbose:
            w('Smallest and largest elements of A after scaling: ')
            w('%8.2e %8.2e\n' % (np.min(np.abs(values)),np.max(np.abs(values))))

        # Overwrite A with scaled values.
        self.A.put(values,irow,jcol)

        # Save row and column scaling.
        self.row_scale = row_scale
        self.col_scale = col_scale

        self.prob_scaled = True

        return

    def unscale(self, **kwargs):
        """
        Restore the constraint matrix A, the right-hand side b and the cost
        vector c to their original value by undoing the row and column
        equilibration scaling.
        """
        row_scale = self.row_scale
        col_scale = self.col_scale
        on = self.lp.original_n

        # Unscale constraint matrix A.
        self.A.row_scale(row_scale)
        self.A.col_scale(col_scale)

        # Unscale right-hand side and cost vectors.
        self.b *= row_scale
        self.c[:on] *= col_scale[:on]

        # Recover unscaled multipliers y and z.
        self.y *= self.row_scale
        self.z /= self.col_scale[on:]

        self.prob_scaled = False

        return

    def solve(self, **kwargs):
        """
        Solve the input problem with the primal-dual-regularized
        interior-point method. Accepted input keyword arguments are

        :keywords:
          :itermax:  The maximum allowed number of iterations (default: 10n)

          :tolerance:  Stopping tolerance (default: 1.0e-6)

          :PredictorCorrector:  Use the predictor-corrector method
                                (default: `True`). If set to `False`, a variant
                                of the long-step method is used. The long-step
                                method is generally slower and less robust.

        Upon exit, the following members of the class instance are set:

         x..............final iterate
         y..............final value of the Lagrange multipliers associated
                        to A1 x + A2 s = b
         z..............final value of the Lagrange multipliers associated
                        to s>=0
         obj_value......final cost
         iter...........total number of iterations
         kktResid.......final relative residual
         solve_time.....time to solve the LP
         status.........string describing the exit status
         short_status...short version of status, used for printing.
        """
        lp = self.lp
        itermax = kwargs.get('itermax', 10*lp.n)
        tolerance = kwargs.get('tolerance', 1.0e-6)
        PredictorCorrector = kwargs.get('PredictorCorrector', True)
        check_infeasible = kwargs.get('check_infeasible', True)

        # Transfer pointers for convenience.
        m, n = self.A.shape ; on = lp.original_n
        A = self.A ; b = self.b ; c = self.c ; H = self.H
        regpr = self.regpr ; regdu = self.regdu
        regpr_min = self.regpr_min ; regdu_min = self.regdu_min

        # Obtain initial point from Mehrotra's heuristic.
        # set_initial_guess() initializes self.LBL which is reused below.
        (x,y,z) = self.set_initial_guess(self.lp, **kwargs)

        # Slack variables are the trailing variables in x.
        s = x[on:] ; ns = self.nSlacks

        # Initialize steps in dual variables.
        dz = np.zeros(ns)

        col_scale = np.empty(n)

        # Allocate room for right-hand side of linear systems.
        rhs = np.zeros(n+m)
        finished = False
        iter = 0

        # Acceptance thresholds for primal and dual reg parameters.
        t1 = t2 = 0.99

        solve_time = cputime()

        # Main loop.
        while not finished:

            # Display initial header every so often.
            if self.verbose and iter % 20 == 0:
                sys.stdout.write('\n' + self.header + '\n')
                sys.stdout.write('-' * len(self.header) + '\n')

            # Compute residuals.
            pFeas = A*x - b
            comp = s*z ; sz = sum(comp)                     # comp   = S z
            dFeas = y*A ; dFeas[:on] -= self.c              # dFeas1 = A1'y - c
            dFeas[on:] += z                                 # dFeas2 = A2'y + z
            mu = sz/ns

            # Adjust regularization parameters
            mu = sum(comp)/ns
            if mu < 1:
                regpr = sqrt(mu)
                regdu = sqrt(mu)

            # At the first iteration, initialize perturbation vectors
            # (q=primal, r=dual).
            if iter == 0:
                regpr = self.regpr ; regdu = self.regdu
                if regpr > 0:
                    q = dFeas/regpr ; qNorm = norm2(q) ; rho_q = regpr * qNorm
                else:
                    q = dFeas ; qNorm = norm2(q) ; rho_q = 0.0
                if regdu > 0:
                    r = -pFeas/regdu ; rNorm = norm2(r) ; del_r = regdu * rNorm
                else:
                    r = -pFeas ; rNorm = norm2(r) ; del_r = 0.0
                #q = np.zeros(n) ; qNorm = 0 ; rho_q = 0
                #r = np.zeros(m) ; rNorm = 0 ; del_r = 0
                pr_infeas_count = 0  # Used to detect primal infeasibility.
                du_infeas_count = 0  # Used to detect dual infeasibility.
                pr_last_iter = 0
                du_last_iter = 0
                mu0 = mu
            else:
                # Adjust regularization parameters.
                #regpr = max(min(regpr/10, regpr**(1.1)), regpr_min)
                #regdu = max(min(regdu/10, regdu**(1.1)), regdu_min)
                # 1) rho+ |dx| <= const * s'z
                # 2) del+ |dy| <= const * s'z
                if regdu > 0:
                    regdu = min(regdu/10, sz/normdy/100, (sz/normdy)**(1.1))
                    regdu = max(regdu, regdu_min)
                if regpr > 0:
                    regpr = min(regpr/10, sz/normdx/100, (sz/normdx)**(1.1))
                    regpr = max(regpr, regpr_min)

                # Check for infeasible problem.
                if check_infeasible:
                    if mu < 1.0e-6 * mu0 and rho_q > 1000 * mu: # tolerance:
                        pr_infeas_count += 1
                        if pr_infeas_count > 1 and pr_last_iter == iter-1:
                            if pr_infeas_count > 3:
                                status = 'Problem is (locally) dual infeasible'
                                short_status = 'dInf'
                                finished = True
                                continue
                        pr_last_iter = iter

                    if mu < 1.0e-6 * mu0 and del_r > 1000 * mu: # tolerance:
                        du_infeas_count += 1
                        if du_infeas_count > 1 and du_last_iter == iter-1:
                            if du_infeas_count > 3:
                                status='Problem is (locally) primal infeasible'
                                short_status = 'pInf'
                                finished = True
                                continue
                        du_last_iter = iter

            # Compute residual norms and scaled residual norms.
            #pResid = norm_infty(pFeas + regdu * r)/(1+self.normc)
            #dResid = norm_infty(dFeas - regpr * q)/(1+self.normb)
            pResid = norm2(pFeas) ; spResid = pResid/(1+self.normc)
            cResid = norm2(comp)  ; scResid = cResid/self.normbc
            dResid = norm2(dFeas) ; sdResid = dResid/(1+self.normb)

            # Compute relative duality gap.
            cx = np.dot(c,x[:on])
            by = np.dot(b,y)
            rgap  = cx - by
            #rgap += regdu * (rNorm**2 + np.dot(r,y))
            rgap  = abs(rgap) / (1 + abs(cx))

            # Compute overall residual for stopping condition.
            kktResid = max(spResid, sdResid, rgap)
            #kktResid = max(pResid, cResid, dResid)

            # Display objective and residual data.
            if self.verbose:
                sys.stdout.write(self.format1 % (iter, cx, pResid, dResid,
                                                 cResid, rgap, qNorm, rNorm))

            if kktResid <= tolerance:
                status = 'Optimal solution found'
                short_status = 'opt'
                finished = True
                continue

            if iter >= itermax:
                status = 'Maximum number of iterations reached'
                short_status= 'iter'
                finished = True
                continue

            # Record some quantities for display
            mins = np.min(s)
            minz = np.min(z)
            maxs = np.max(s)

            # Repeatedly assemble system and compute step until primal and
            # dual regularization parameters have appropriate values.

            # Reset primal and dual regularization parameters to best guess
            #if iter > 0:
            #    regpr = max(regpr_min, 0.5*sigma*dResid/normds)
            #    regdu = max(regdu_min, 0.5*sigma*pResid/normdy)

            step_acceptable = False

            while not step_acceptable:

                # Solve the linear system
                # 
                # [-pI          0          A1'] [∆x]   [c - A1' y             ]
                # [ 0   -(S^{-1} Z + pI)   A2'] [∆s] = [  - A2' y - µ S^{-1} e]
                # [ A1          A2         dI ] [∆y]   [b - A1 x - A2 s       ]
                #
                # where s are the slack variables, p is the primal
                # regularization parameter, d is the dual regularization
                # parameter, and  A = [ A1  A2 ]  where the columns of A1
                # correspond to the original problem variables and those of A2
                # correspond to slack variables.
                #
                # We recover ∆z = -z - S^{-1} (Z ∆s + µ e).
                # Compute augmented matrix and factorize it.
                factorized = False
                nb_bump = 0
                while not factorized and nb_bump < 5:

                    if self.stabilize:
                        col_scale[:on] = sqrt(regpr)
                        col_scale[on:] = np.sqrt(z/s + regpr)
                        H.put(-sqrt(regdu), range(n))
                        H.put( sqrt(regdu), range(n,n+m))
                        AA = self.A.copy()
                        AA.col_scale(1/col_scale)
                        H[n:,:n] = AA
                    else:
                        if regpr > 0: H.put(-regpr,       range(on))
                        H.put(-z/s - regpr, range(on,n))
                        if regdu > 0: H.put(regdu,        range(n,n+m))

                    #if iter == 5:
                    #    # Export current matrix to file for futher inspection.
                    #    import os
                    #    name = os.path.basename(self.lp.name)
                    #    fname = '.'.join(name.split('.')[:-1]) + '.mtx'
                    #    H.exportMmf(fname)

                    self.LBL.factorize(H)
                    factorized = True

                    # If the augmented matrix does not have full rank, bump up
                    # regularization parameters.
                    if not self.LBL.isFullRank:
                        if self.verbose:
                            sys.stderr.write('Primal-Dual Matrix ')
                            sys.stderr.write('Rank Deficient')
                        if regdu == 0.0:
                            sys.stderr.write('... No regularization in effect')
                            sys.stderr.write('... bailing out\n')
                            factorized = False
                            nb_bump = 5
                            continue
                        else:
                            sys.stderr.write('... bumping up reg parameters\n')
                        regpr *= 10 ; regdu *= 10
                        nb_bump += 1
                        factorized = False

                # Abandon if regularization is unsuccessful.
                if not self.LBL.isFullRank and nb_bump >= 5:
                    status = 'Unable to regularize sufficiently.'
                    short_status = 'degn'
                    finished = True
                    continue  # Does this get us out of the outer while?

                # Compute duality measure.
                mu = sz/ns

                if PredictorCorrector:
                    # Use Mehrotra predictor-corrector method.
                    # Compute affine-scaling step, i.e. with centering = 0.
                    rhs[:n]    = -dFeas
                    rhs[on:n] += z
                    rhs[n:]    = -pFeas

                    # if 'stabilize' is on, must scale right-hand side.
                    if self.stabilize:
                        rhs[:n] /= col_scale
                        rhs[n:] /= sqrt(regdu)

                    (step, nres, neig) = self.solveSystem(rhs)

                    # Unscale step if 'stabilize' is on.
                    if self.stabilize:
                        step[:n] *= sqrt(regdu) / col_scale

                    # Recover dx and dz.
                    dx = step[:n]
                    ds = dx[on:]
                    dz = -z * (1 + ds/s)

                    # Compute largest allowed primal and dual stepsizes.
                    (alpha_p, ip) = self.maxStepLength(s, ds)
                    (alpha_d, ip) = self.maxStepLength(z, dz)

                    # Estimate duality gap after affine-scaling step.
                    muAff = np.dot(s + alpha_p * ds, z + alpha_d * dz)/ns
                    sigma = (muAff/mu)**3

                    # Incorporate predictor information for corrector step.
                    comp += ds*dz
                else:
                    # Use long-step method: Compute centering parameter.
                    sigma = min(0.1, 100*mu)

                # Assemble right-hand side with centering information.
                comp -= sigma * mu

                if PredictorCorrector:
                    # Only update rhs[on:n]; the rest of rhs did not change.
                    if self.stabilize:
                        rhs[on:n] += (comp/s - z)/col_scale[on:n]
                    else:
                        rhs[on:n] += comp/s - z
                else:
                    rhs[:n]    = -dFeas
                    rhs[on:n] += comp/s
                    rhs[n:]    = -pFeas

                    # If 'stabilize' is on, must scale right-hand side.
                    # In the predictor-corrector method, this has already been
                    # done.
                    if self.stabilize:
                        rhs[:n] /= col_scale
                        rhs[n:] /= sqrt(regdu)

                # Solve augmented system.
                (step, nres, neig) = self.solveSystem(rhs)

                # Unscale step if 'stabilize' is on.
                if self.stabilize:
                    step[:n] *= sqrt(regdu) / col_scale

                # Recover step.
                dx = step[:n]
                ds = dx[on:]
                dy = step[n:]

                normds = norm2(ds) ; normdy = norm2(dy) ; normdx = norm2(dx)
                step_acceptable = True  # Must get rid of this

            # End while not step_acceptable

            # Recover step in z.
            dz = -(comp + z*ds)/s

            # Compute largest allowed primal and dual stepsizes.
            (alpha_p, ip) = self.maxStepLength(s, ds)
            (alpha_d, id) = self.maxStepLength(z, dz)

            # Compute fraction-to-the-boundary factor.
            tau = max(.9995, 1.0-mu)

            if PredictorCorrector:
                # Compute actual stepsize using Mehrotra's heuristic
                mult = 0.1

                # ip=-1 if ds ≥ 0, and id=-1 if dz ≥ 0
                if (ip != -1 or id != -1) and ip != id:
                    mu_tmp = np.dot(s + alpha_p * ds, z + alpha_d * dz)/ns

                if ip != -1 and ip != id:
                    zip = z[ip] + alpha_d * dz[ip]
                    gamma_p = (mult*mu_tmp - s[ip]*zip)/(alpha_p*ds[ip]*zip)
                    alpha_p *= max(1-mult, gamma_p)

                if id != -1 and ip != id:
                    sid = s[id] + alpha_p * ds[id]
                    gamma_d = (mult*mu_tmp - z[id]*sid)/(alpha_d*dz[id]*sid)
                    alpha_d *= max(1-mult, gamma_d)

                if ip==id and ip != -1:
                    # There is a division by zero in Mehrotra's heuristic
                    # Fall back on classical rule.
                    alpha_p *= tau
                    alpha_d *= tau

            else:
                alpha_p *= tau
                alpha_d *= tau

            # Display data.
            if self.verbose:
                sys.stdout.write(self.format2 % (mu, alpha_p, alpha_d,
                                                 nres, regpr, regdu, rho_q,
                                                 del_r, mins, minz, maxs))

            # Update primal variables and slacks.
            x += alpha_p * dx

            # Update dual variables.
            y += alpha_d * dy
            z += alpha_d * dz

            # Update perturbation vectors.
            q *= (1-alpha_p) ; q += alpha_p * dx
            r *= (1-alpha_d) ; r += alpha_d * dy
            qNorm = norm2(q) ; rNorm = norm2(r)
            rho_q = regpr * qNorm/(1+self.normc)
            del_r = regdu * rNorm/(1+self.normb)

            iter += 1

        solve_time = cputime() - solve_time

        if self.verbose:
            sys.stdout.write('\n')
            sys.stdout.write('-' * len(self.header) + '\n')

        # Transfer final values to class members.
        self.x = x
        self.y = y
        self.z = z
        self.iter = iter
        self.pResid = pResid ; self.cResid = cResid ; self.dResid = dResid
        self.rgap = rgap
        self.kktResid = kktResid
        self.solve_time = solve_time
        self.status = status
        self.short_status = short_status

        # Unscale problem if applicable.
        if self.prob_scaled: self.unscale()

        # Recompute final objective value.
        self.obj_value = np.dot(self.c, x[:on]) + self.c0
        return

    def set_initial_guess(self, lp, **kwargs):
        """
        Compute initial guess according the Mehrotra's heuristic. Initial values
        of x are computed as the solution to the least-squares problem

          minimize ||s||  subject to  A1 x + A2 s = b

        which is also the solution to the augmented system

          [ 0   0   A1' ] [x]   [0]
          [ 0   I   A2' ] [s] = [0]
          [ A1  A2   0  ] [w]   [b].

        Initial values for (y,z) are chosen as the solution to the least-squares
        problem

          minimize ||z||  subject to  A1' y = c,  A2' y + z = 0

        which can be computed as the solution to the augmented system

          [ 0   0   A1' ] [w]   [c]
          [ 0   I   A2' ] [z] = [0]
          [ A1  A2   0  ] [y]   [0].

        To ensure stability and nonsingularity when A does not have full row
        rank, the (1,1) block is perturbed to 1.0e-4 * I and the (3,3) block is
        perturbed to -1.0e-4 * I.

        The values of s and z are subsequently adjusted to ensure they are
        positive. See [Methrotra, 1992] for details.
        """
        n = lp.n ; m = lp.m ; ns = self.nSlacks ; on = lp.original_n

        # Set up augmented system matrix and factorize it.
        self.H.put(1.0e-4, range(on))
        self.H.put(1.0, range(on,n))
        self.H.put(-1.0e-4, range(n,n+m))
        self.H[n:,:n] = self.A
        self.LBL = LBLContext(self.H, sqd=True)  # Perform analyze and factorize

        # Assemble first right-hand side and solve.
        rhs = np.zeros(n+m)
        rhs[n:] = self.b
        (step, nres, neig) = self.solveSystem(rhs)
        x = step[:n].copy()
        s = x[on:]  # Slack variables. Must be positive.

        # Assemble second right-hand side and solve.
        rhs[:on] = self.c
        rhs[on:] = 0.0

        (step, nres, neig) = self.solveSystem(rhs)
        y = step[n:].copy()
        z = step[on:n].copy()

        # Use Mehrotra's heuristic to ensure (s,z) > 0.
        if np.all(s >= 0):
            dp = 0.0
        else:
            dp = -1.5 * min(s[s < 0])
        if np.all(z >= 0):
            dd = 0.0
        else:
            dd = -1.5 * min(z[z < 0])

        if dp == 0.0: dp = 1.5
        if dd == 0.0: dd = 1.5

        es = sum(s+dp)
        ez = sum(z+dd)
        xs = sum((s+dp) * (z+dd))

        dp += 0.5 * xs/ez
        dd += 0.5 * xs/es
        s += dp
        z += dd

        if not np.all(s>0) or not np.all(z>0):
            raise ValueError, 'Initial point not strictly feasible'

        return (x,y,z)

    def maxStepLength(self, x, d):
        """
        Returns the max step length from x to the boundary
        of the nonnegative orthant in the direction d
        alpha_max = min [ 1, min { -x[i]/d[i] | d[i] < 0 } ].
        Note that 0 < alpha_max <= 1.
        """
        whereneg = np.where(d < 0)[0]
        if len(whereneg) > 0:
            dxneg = -x[whereneg]/d[whereneg]
            kmin = np.argmin(dxneg)
            stepmax = min(1.0, dxneg[kmin])
            if stepmax == 1.0:
                kmin = -1
            else:
                kmin = whereneg[kmin]
        else:
            stepmax = 1.0
            kmin = -1
        return (stepmax, kmin)

    def solveSystem(self, rhs, itref_threshold=1.0e-5, nitrefmax=3):
        self.LBL.solve(rhs)
        #nr = norm2(self.LBL.residual)
        self.LBL.refine(rhs, tol=itref_threshold, nitref=nitrefmax)
        nr = norm2(self.LBL.residual)
        return (self.LBL.x, nr, self.LBL.neig)


class RegLPInteriorPointSolver29(RegLPInteriorPointSolver):

    def scale(self, **kwargs):
        """
        Scale the constraint matrix of the linear program. The scaling is done
        so that the scaled matrix has all its entries near 1.0 in the sense that
        the square of the sum of the logarithms of the entries is minimized.

        In effect the original problem

          minimize c'x  subject to  A1 x + A2 s = b, x >= 0

        is converted to

          minimize (Cc)'x  subject to  R A1 C x + R A2 C s = Rb, x >= 0,

        where the diagonal matrices R and C operate row and column scaling
        respectively.

        Upon return, the matrix A and the right-hand side b are scaled and the
        members `row_scale` and `col_scale` are set to the row and column
        scaling factors.

        The scaling may be undone by subsequently calling :meth:`unscale`. It is
        necessary to unscale the problem in order to unscale the final dual
        variables. Normally, the :meth:`solve` method takes care of unscaling
        the problem upon termination.
        """
        (values, irow, jcol) = self.A.find()
        m, n = self.A.shape

        # Obtain row and column scaling
        row_scale, col_scale, ifail = mc29ad(m, n, values, irow, jcol)

        # row_scale and col_scale contain in fact the logarithms of the
        # scaling factors.
        row_scale = np.exp(row_scale)
        col_scale = np.exp(col_scale)

        # Apply row and column scaling to constraint matrix A.
        values *= row_scale[irow]
        values *= col_scale[jcol]

        # Overwrite A with scaled matrix.
        self.A.put(values,irow,jcol)
        
        # Apply row scaling to right-hand side b.
        self.b *= row_scale

        # Apply column scaling to cost vector c.
        self.c[:self.lp.original_n] *= col_scale[:self.lp.original_n]

        # Save row and column scaling.
        self.row_scale = row_scale
        self.col_scale = col_scale

        self.prob_scaled = True
        
        return

    def unscale(self, **kwargs):
        """
        Restore the constraint matrix A, the right-hand side b and the cost
        vector c to their original value by undoing the row and column
        equilibration scaling.
        """
        row_scale = self.row_scale
        col_scale = self.col_scale
        on = self.lp.original_n

        # Unscale constraint matrix A.
        self.A.row_scale(1/row_scale)
        self.A.col_scale(1/col_scale)

        # Unscale right-hand side b.
        self.b /= row_scale

        # Unscale cost vector c.
        self.c[:on] /= col_scale[:on]

        # Recover unscaled multipliers y and z.
        self.y /= row_scale
        self.z *= col_scale[on:]

        self.prob_scaled = False

        return

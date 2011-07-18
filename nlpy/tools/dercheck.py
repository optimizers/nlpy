"""
A simple derivative checker.
"""

import numpy as np
from math import sqrt
import sys

macheps = np.finfo(np.double).eps  # Machine epsilon.

class DerivativeChecker:

    def __init__(self, nlp, x, **kwargs):
        """
        The `DerivativeChecker` class provides facilities for verifying
        numerically the accuracy of first and second-order derivatives
        implemented in an optimization model.

        `nlp` should be a `NLPModel` and `x` is the point about which we are
        checking the derivative.
        """

        self.step = kwargs.get('step', sqrt(macheps))
        self.tol  = kwargs.get('tol', 100*sqrt(macheps))

        self.nlp = nlp
        self.x = x
        self.grad_errs = []
        self.jac_errs = []
        self.hess_errs = []
        self.chess_errs = []

        headfmt = '%4s  %4s        %22s  %22s  %7s\n'
        self.head = headfmt % ('Fun','Comp','Expected','Finite Diff','Rel.Err')
        self.d1fmt = '%4d  %4d        %22.15e  %22.15e  %7.1e\n'
        head2fmt = '%4s  %4s  %4s  %22s  %22s  %7s\n'
        self.head2 = head2fmt % ('Fun','Comp','Comp','Expected',
                                 'Finite Diff','Rel.Err')
        self.d2fmt = '%4d  %4d  %4d  %22.15e  %22.15e  %7.1e\n'

        return


    def check(self, **kwargs):

        grad = kwargs.get('grad', True)            # Check objective  gradient.
        hess = kwargs.get('hess', True)            # Check objective  Hessian.
        jac  = kwargs.get('jac', True)             # Check constraint Jacobian.
        chess = kwargs.get('chess', True)          # Check constraint Hessians.
        verbose = kwargs.get('verbose', True)

        # Skip constraints if problem is unconstrained.
        jac = (jac and self.nlp.m > 0)
        chess = (chess and self.nlp.m > 0)

        if grad:
            self.grad_errs = self.check_obj_gradient(verbose)
            self.display(self.grad_errs, self.head)
        if jac:
            self.jac_errs = self.check_con_jacobian(verbose)
            self.display(self.jac_errs, self.head)
        if hess:
            self.hess_errs = self.check_obj_hessian(verbose)
            self.display(self.hess_errs, self.head2)
        if chess:
            self.chess_errs = self.check_con_hessians(verbose)
            self.display(self.chess_errs, self.head2)

        return


    def display(self, errs, header):
        nerrs = len(errs)
        sys.stderr.write('Found %d errors.\n' % nerrs)
        if nerrs > 0:
            sys.stderr.write(header)
            sys.stderr.write('-' * len(header) + '\n')

            for err in errs:
                sys.stderr.write(err)

        return


    def check_obj_gradient(self, verbose=False):
        nlp = self.nlp
        n = nlp.n
        fx = nlp.obj(self.x)
        gx = nlp.grad(self.x)
        h = self.step
        errs = []

        if verbose:
            sys.stderr.write('Objective gradient\n')

        # Check partial derivatives in turn.
        for i in range(n):
            xph = self.x.copy() ; xph[i] += h
            xmh = self.x.copy() ; xmh[i] -= h
            dfdxi = (nlp.obj(xph) - nlp.obj(xmh))/(2*h)
            err = abs(gx[i] - dfdxi)/(1 + abs(gx[i]))

            line = self.d1fmt % (0, i, gx[i], dfdxi, err)
            if verbose:
                sys.stderr.write(line)

            if err > self.tol:
                errs.append(line)

        return errs


    def check_obj_hessian(self, verbose=False):
        nlp = self.nlp
        n = nlp.n
        Hx = nlp.hess(self.x, np.zeros(nlp.m))
        h = self.step
        errs = []

        if verbose:
            sys.stderr.write('Objective Hessian\n')

        # Check second partial derivatives in turn.
        for i in range(n):
            xph = self.x.copy() ; xph[i] += h
            xmh = self.x.copy() ; xmh[i] -= h
            dgdx = (nlp.grad(xph) - nlp.grad(xmh))/(2*h)
            for j in range(i+1):
                dgjdxi = dgdx[j]
                err = abs(Hx[i,j] - dgjdxi)/(1 + abs(Hx[i,j]))

                line = self.d2fmt % (0, i, j, Hx[i,j], dgjdxi, err)
                if verbose:
                    sys.stderr.write(line)

                if err > self.tol:
                    errs.append(line)

        return errs


    def check_con_jacobian(self, verbose=False):
        nlp = self.nlp
        n = nlp.n ; m = nlp.m
        if m == 0: return []   # Problem is unconstrained.

        Jx = nlp.jac(self.x)
        h = self.step
        errs = []

        if verbose:
            sys.stderr.write('Constraints Jacobian\n')

        # Check partial derivatives of each constraint in turn.
        for i in range(n):
            xph = self.x.copy() ; xph[i] += h
            xmh = self.x.copy() ; xmh[i] -= h
            dcdx = (nlp.cons(xph) - nlp.cons(xmh))/(2*h)
            for j in range(m):
                dcjdxi = dcdx[j] #(nlp.icons(j, xph) - nlp.icons(j, xmh))/(2*h)
                err = abs(Jx[j,i] - dcjdxi)/(1 + abs(Jx[j,i]))

                line = self.d1fmt % (j+1, i, Jx[j,i], dcjdxi, err)
                if verbose:
                    sys.stderr.write(line)

                if err > self.tol:
                    errs.append(line)

        return errs


    def check_con_hessians(self, verbose=False):
        nlp = self.nlp
        n = nlp.n ; m = nlp.m
        h = self.step
        errs = []

        if verbose:
            sys.stderr.write('Constraints Hessians\n')

        # Check each Hessian in turn.
        for k in range(m):
            y = np.zeros(m) ; y[k] = 1
            Hk = nlp.hess(self.x, y, obj_weight=0)

            # Check second partial derivatives in turn.
            for i in range(n):
                xph = self.x.copy() ; xph[i] += h
                xmh = self.x.copy() ; xmh[i] -= h
                dgdx = (nlp.igrad(k, xph) - nlp.igrad(k, xmh))/(2*h)
                for j in range(i+1):
                    dgjdxi = dgdx[j]
                    err = abs(Hk[i,j] - dgjdxi)/(1 + abs(Hk[i,j]))

                    line = self.d2fmt % (k+1, i, j, Hk[i,j], dgjdxi, err)
                    if verbose:
                        sys.stderr.write(line)

                    if err > self.tol:
                        errs.append(line)

        return errs


if __name__ == '__main__':

    import sys
    from nlpy.model import AmplModel
    nlp = AmplModel(sys.argv[1])
    print 'Checking at x = ', nlp.x0
    derchk = DerivativeChecker(nlp, nlp.x0)
    derchk.check(verbose=False)
    nlp.close()

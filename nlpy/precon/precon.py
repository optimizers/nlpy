"""
A first draft of a generic class to build preconditioners

Each preconditioner object K has:
   - a 'shape'  member, giving its size
   - a 'precon' method which takes a vector x as argument
     and returns the vector y satisfying Ky = x.
     Equivalently, a preconditioner may be called directly
     as in
     K = DiagonalPreconditioner(A)
     y = K(x)

D. Orban                             Montreal, March 2007
"""

import numpy as np
from pysparse.sparse import spmatrix
try:     # To solve symmetric linear systems
        from nlpy.linalg.pyma57 import PyMa57Context as LBLContext
except:
        from nlpy.linalg.pyma27 import PyMa27Context as LBLContext


class GenericPreconditioner(object):

    def __init__(self, A, **kwargs):
        self.shape = A.shape

    def precon(self, x):
        raise NotImplementedError('This method must be overridden')
        return x

    def __call__(self, x):
        return self.precon(x)


class DiagonalPreconditioner(GenericPreconditioner):

    def __init__(self, A, **kwargs):
        GenericPreconditioner.__init__(self, A, **kwargs)
        self.threshold = kwargs.get('threshold', 1.0e-3)
        if len(A.shape) == 1:
            self.Dinv = 1.0 / np.maximum(np.abs(A), self.threshold)
        else:
            self.Dinv = 1.0 / np.maximum(np.abs(A.takeDiagonal()),
                                         self.threshold)

    def __setitem__(self, key, value):
        self.Dinv[key] = 1.0 / max(abs(value), self.threshold)
        return

    def __getitem__(self, key):
        return self.Dinv[key]

    def __str__(self):
        n = self.shape[0]
        return 'Diagonal preconditioner of size (%d,%d)' % (n, n)

    def __repr__(self):
        return repr(self.Dinv)

    def precon(self, x):
        return self.Dinv * x


class BandedPreconditioner(GenericPreconditioner):

    def __init__(self, A, **kwargs):
        GenericPreconditioner.__init__(self, A, **kwargs)
        n = self.shape[0]
        self.bandwidth = kwargs.get('bandwidth', 5)
        self.threshold = kwargs.get('threshold', 1.0e-3)

        # See how many nonzeros are in the requested band of A
        nnz = 0
        sumrowelm = np.zeros(n, 'd')
        for j in range(n):
            for i in range(min(self.bandwidth, n - j - 1)):
                if A[j + i + 1, j] != 0.0:
                    nnz += 1
                    sumrowelm[j] += abs(A[j + i + 1, j])

        M = spmatrix.ll_mat_sym(n, nnz + n)

        # Assemble banded matrix --- ensure it is positive definite
        for j in range(n):
            M[j, j] = max(A[j, j], 2 * sumrowelm[j] + self.threshold)
            for i in range(min(self.bandwidth, n - j - 1)):
                if A[j + i + 1, j] != 0.0:
                    M[j + i + 1, j] = A[j + i + 1, j]

        # Factorize preconditioner
        self.lbl = LBLContext(M)

        # Only need factors of M --- can discard M itself
        del M

    def precon(self, x):
        self.lbl.solve(x)
        return self.lbl.x.copy()

# For a LBFGS preconditioner, see class LbfgsUpdate

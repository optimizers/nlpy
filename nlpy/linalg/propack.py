"""
Compute k-leading singular values/vectors of a linear operator using
PROPACK (http://soi.stanford.edu/~rmunk/PROPACK/).

Michael P. Friedlander, University of British Columbia
Dominique Orban, Ecole Polytechnique de Montreal
"""

import numpy as np
from nlpy.linalg._pypropack import dlansvd

__docformat__ = 'restructuredtext'

class propack:

    def __init__(self, A):
        """
        Create an instance of a SVD solver for a linear operator.

        :parameters:

           :A: is an object that represents an m-by-n linear
               operator. The object must support matrix-vector
               multiplication, ie,

               v = A*x, where x is n-by-1;
               w = A.T*y, where y is m-by-1.
        """
        self.A = A
        self.itn = 0
        self.U = None # left singular vectors
        self.V = None # right singular vectors
        self.s = None # singular values
        self.its = 0
        # status =
        # 1: The k largest singular triplets were computed succesfully
        # 0<j<k: An invariant subspace of dimension j was found.
        # -1 : k singular triplets did not converge within kmax iterations.
        self.status = None

    def solve(self, k=6, **kwargs):
        """
        Compute k-leading singular values of a symmetric linear operator.

        :parameters:

           :k:        number of singular values/vectors to compute (default 6)

        :keywords:

           :kmax:     maximum dimension of Krylov subspace (default 10*`k`)
           :tol_orth: level of orthogonality to maintain among Lanczos vectors
                      (default 1.0e-8)
           :tol_opt:  convergence tolerace of singular values (default 1e-12)
           :eta:      During reorthogonalization, all vectors with components
                      larger than `eta` along the latest Lanczos vector will be
                      purged (default 1.0e-12)
           :Anorm:    estimate of the norm of the operator (default 1.0)
           :cgs:      reorthogonalization option.
                      1: classical Gram-Schmidt
                      2  modified Gram-Schmidt (default)
           :getu:     if `True`, compute left singular vectors (default `True`)
           :getv:     if `True`, compute right singular vectors (default `True`)
        """
        kmax = kwargs.get('kmax',1+10*k)       # guarantees kmax >= 1
        tol_orth = kwargs.get('tol_orth',1e-8)
        tol_opt = kwargs.get('tol_opt',1e-12)
        eta = kwargs.get('eta',1e-12)
        Anorm_est = kwargs.get('Anorm',1.0)
        cgs = kwargs.get('cgs',2)
        getu = kwargs.get('getu',True)
        getv = kwargs.get('getv',True)

        A = self.A
        m, n = A.shape
        k = max(1,k)
        k = min(k, kmax)
        k = min(k,min(m, n))

        jobu = 'Y' if getu else 'N'
        jobv = 'Y' if getv else 'N'

        doption = np.array([tol_orth, eta, Anorm_est])
        ioption = np.array([cgs, 1],dtype=np.int) # Not sure what 2nd arg means

        U, self.s, self.bnd, V, self.info = \
                dlansvd(jobv, jobu, m, n, k, kmax, \
                        self._Aprod, doption, ioption, tol_opt)

        # U and V have kmax columns, but only the first k are useful.
        if getu: self.U = U[:m,:k]
        if getv: self.V = V[:n,:k]

    def _Aprod(self, transa, x):
        if transa[0] == 'n':
            self.its += 1
            return self.A*x
        else:
            return self.A.T*x

if __name__ == '__main__':

    np.random.seed(123)

    m = 42
    n = 21
    k = 4; kmax = 50

    A = np.asmatrix(np.random.random((m,n)))

    svd = propack(A)
    svd.solve()

    U,sigma,V = np.linalg.svd(A)

    print 'sigma (propack) = ', svd.s[:k]
    print 'sigma (linpack) = ', sigma[:k]

    print '%2s  %12s  %12s' % ('k','norm(A-USV'')','norm(s[k:])')
    for k in range(1,min(m,n)+1):

        svd.solve(k)
        U = svd.U; V = svd.V; s = svd.s
        B = np.dot(np.dot(U, np.diag(s)), V.T)
        E = A - B
        print '%2i  %12.7e  %12.7e'% (k,np.linalg.norm(E),np.linalg.norm(sigma[k:]))

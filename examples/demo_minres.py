"""
Minres demo.
"""

from nlpy.krylov import Minres
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.precon import DiagonalPreconditioner
import numpy as np

if __name__ == '__main__':

    # Set printing standards for arrays
    np.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

    # Create A = diag(1,2,...,n) and b = [1, 1, ..., 1].
    n = 10  # Must have n > 4.
    A = SimpleLinearOperator(n, n, lambda v: np.arange(1,n+1)*v, symmetric=True)
    b = np.ones(n)
    
    M = DiagonalPreconditioner(np.arange(1,n+1))
    for i in range(1,4): M[i] = 1.0

    K = Minres(A)
    K.solve(b, precon=M, show=True) # Solves Ax = b with preconditioner M.
    print 'Solution: ' ; print K.x

    K.solve(b, shift=2.0, precon=M, show=True) # Solves (A-2I)x = b.
    print 'Solution: ' ; print K.x

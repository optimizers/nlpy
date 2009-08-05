"""
Pycfs: Incomplete Cholesky factorization and preconditioned conjugate gradient
"""

import numpy
import _pycfs

class PycfsContext:

    def __init__( self, A, mem = 5 ):
        """
        PycfsContext( A, mem ) instantiates a new abstract object for the
        computation of an incomplete Cholesky factorization of the n x n sparse
        matrix A stored in LL or CSR format. The optional parameter mem
        represents the amount of additional memory to allow for the
        factorization. By default, mem = 5, setting the additional memory to
        5n. The members of the PycfsContext object are

            shape       the shape of the matrix A
            n           the order of the matrix A
            iter        the number of iterations after solution of a system
                         by means of a preconditioned conjugate gradient method
            info        the exit code of the solve method
            relres      the relative residual after solution
            nc          the most recent negative curvature encountered
            shift       the final shift which led to the factorization
            x           the current solution
            ncd         a direction of negative curvature if nc == True.
        """
        #if not A.issym:
        #    raise ValueError, 'Input matrix must be symmetric'
        self.shape = A.shape
        self.n = A.shape[0]
        self.mem = mem
        self.iter = 0
        self.info = 0
        self.relres = 0.0
        self.nc = 0.0
        self.tsolve = None
        self.x   = numpy.zeros( self.n, 'd' )
        self.ncd = numpy.zeros( self.n, 'd' )
        self.context = _pycfs.icfs( A, self.mem )
        self.shift = self.context.get_shift()

    def solve( self, b, maxiter = 0, rtol = 1.0e-6 ):
        """
        Solve the system of linear equations Ax = b using the preconditioned
        conjugate gradient algorithm. The preconditioner has been computed
        upon instantiation of a PycfsContext class as a limited-memory Cholesky
        factorization of A.
        
        The method updates the members self.iter, self.info,
        self.relres and self.nc of the object containting respectively the
        number of iterations, the exit code, the relative residual and a flag
        indicating whether or not negative curvature was detected.

        Optional arguments are

            maxiter      maximum number of iterations (default: max(n,100))
            rtol         relative stopping tolerance  (default: 1.0e-6)
        """
        (self.iter, self.info, self.relres, self.nc, self.tsolve) = self.context.pcg( b, self.x, self.ncd, maxiter, rtol )

    def fetch( self ):
        """
        Fetch limited-memory Cholesky factor of original matrix.
        Returns a lower triangular matrix in LL format.
        """
        return self.context.fetch()

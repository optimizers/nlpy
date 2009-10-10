"""
Ma27: Direct multifrontal solution of symmetric systems
"""

import numpy
from pysparse.pysparseMatrix import PysparseMatrix
from pysparse import spmatrix
from sils import Sils
from nlpy.linalg import _pyma27

class PyMa27Context( Sils ):

    def __init__( self, A, **kwargs ):
        """
        Create a PyMa27Context object representing a context to solve
        the square symmetric linear system of equations

            A x = b.
    
        A should be given in ll_mat format and should be symmetric.
        The system will first be analyzed and factorized, for later
        solution. Residuals will be computed dynamically if requested.

        The factorization is a multi-frontal variant of the Bunch-Parlett
        factorization, i.e.

            A = L B Lt

        where L is unit lower triangular, and B is symmetric and block diagonal
        with either 1x1 or 2x2 blocks.
        
        A special option is available is the matrix A is known to be symmetric
        (positive or negative) definite, or symmetric quasi-definite (sqd).
        SQD matrices have the general form

            [ E  Gt ]
            [ G  -F ]

        where both E and F are positive definite. As a special case, positive
        definite matrices and negative definite matrices are sqd. SQD matrices
        can be factorized with potentially much sparser factors. Moreover, the
        matrix B reduces to a diagonal matrix.

        Currently accepted keyword arguments are:

           sqd  Flag indicating symmetric quasi-definite matrix (default: False)

        Example:
            import pyma27
            import norms
            P = pyma27.PyMa27Context( A )
            P.solve( rhs, get_resid = True )
            print norms.norm2( P.residual )

        Pyma27 relies on the sparse direct multifrontal code MA27
        from the Harwell Subroutine Library archive.
        """

        if isinstance(A, PysparseMatrix):
            thisA = A.matrix
        else:
            thisA = A

        Sils.__init__( self, thisA, **kwargs )

        # Statistics on A
        self.rwords = 0      # Number of real words used during factorization
        self.iwords = 0      #           int
        self.ncomp  = 0      #           data compresses performed in analysis
        self.nrcomp = 0      #           real
        self.nicomp = 0      #           int
        self.n2x2pivots = 0  #           2x2 pivots used
        self.neig   = 0      #           negative eigenvalues detected

        # Factors
        self.L = spmatrix.ll_mat( self.n, self.n, 0 )
        self.B = spmatrix.ll_mat_sym( self.n, 0 )

        # Analyze and factorize matrix
        self.context = _pyma27.factor( thisA, self.sqd )
        (self.rwords, self.iwords, self.ncomp, self.nrcomp, self.nicomp,
         self.n2x2pivots, self.neig, self.rank) = self.context.stats()

        self.isFullRank = (self.rank == self.n)

    def solve( self, b, get_resid = True ):
        """
        solve(b) solves the linear system of equations Ax = b.
        The solution will be found in self.x and residual in
        self.residual.
        """
        self.context.ma27( b, self.x, self.residual, get_resid )
        return None

    def refine( self, b, nitref = 3, tol = 1.0e-8, **kwargs):
        """
        refine( b, tol, nitref ) performs iterative refinement if necessary
        until the scaled residual norm ||b-Ax||/(1+||b||) falls below the
        threshold 'tol' or until nitref steps are taken.
        Make sure you have called solve() with the same right-hand
        side b before calling refine().
        The residual vector self.residual will be updated to reflect
        the updated approximate solution. 

        By default, tol = 1.0e-8 and nitref = 3.
        """
        self.context.refine( self.x, self.residual, b, tol, nitref )
        return None

    def fetch_perm( self ):
        """
        fetch_perm() returns the permutation vector p used
        to compute the factorization of A. Rows and columns
        were permuted so that
        
              P^T  A P = L  B  L^T

        where i-th row of P is the p(i)-th row of the
        identity matrix, L is unit upper triangular and
        B is block diagonal with 1x1 and 2x2 blocks.
        """
        return self.context.fetchperm()

    def fetch_lb( self ):
        """
        fetch_lb() returns the factors L and B of A such that
        
              P^T  A P = L  B  L^T

        where P is as in fetch_perm(), L is unit upper
        triangular and B is block diagonal with 1x1 and 2x2
        blocks. Access to the factors is available as soon
        as a PyMa27Context has been instantiated.
        """
        self.context.fetchlb( self.L, self.B )
        return None
        

if __name__ == '__main__':

    import sys
    from pysparse import spmatrix
    import numpy
    import norms

    M = spmatrix.ll_mat_from_mtx( sys.argv[1] )
    (m,n) = M.shape
    if m != n:
        sys.stderr( 'Matrix must be square' )
        sys.exit(1)
    if not M.issym:
        sys.stderr( 'Matrix must be symmetric' )
        sys.exit(2)
    e = numpy.ones( n, 'd' )
    rhs = numpy.zeros( n, 'd' )
    M.matvec( e, rhs )
    sys.stderr.write( ' Factorizing matrix... ' )
    G = PyMa27Context( M )
    sys.stderr.write( ' done\n' )
    sys.stderr.write( ' Solving system... ' )
    G.solve( rhs )
    sys.stderr.write( ' done\n' )
    sys.stderr.write( ' Residual = %-g\n' % norms.norm_infty( G.residual ) )
    sys.stderr.write( ' Relative error = %-g\n' % norms.norm_infty( G.x - e ) )
    sys.stderr.write( ' Performing iterative refinement if necessary... ' )
    nr1 = nr = norms.norm_infty( G.residual )
    nitref = 0
    while nr1 > 1.0e-6 and nitref < 5 and nr1 <= nr:
        nitref += 1
        G.refine( rhs )
        nr1 = norms.norm_infty( G.residual )
    sys.stderr.write( ' done\n' )
    sys.stderr.write( ' After %-d refinements:\n' % nitref )
    sys.stderr.write( ' Residual = %-g\n' % norms.norm_infty( G.residual ) )
    sys.stderr.write( ' Relative error = %-g\n' % norms.norm_infty( G.x - e ) )

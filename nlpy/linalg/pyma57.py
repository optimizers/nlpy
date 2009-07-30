"""
Ma57: Direct multifrontal solution of symmetric systems

$Id:$
"""


import numpy
from pysparse import spmatrix
from sils import Sils

try:
    import _pyma57
except:
    import nlpy_error
    nlpy_error.ReportNlpyModuleFatalError( "PyMa57", "MA57" )

class PyMa57Context( Sils ):

    def __init__( self, A, **kwargs ):
        """
        Create a PyMa57Context object representing a context to solve
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
            import pyma57
            import norms
            P = pyma57.PyMa57Context( A )
            P.solve( rhs, get_resid = True )
            print norms.norm2( P.residual )

        Pyma57 relies on the sparse direct multifrontal code MA57
        from the Harwell Subroutine Library (free for academic use).

        From the MA57 spec sheet, 'In addition to being more efficient largely
        through its use of the Level 3 BLAS, it has many added features. Among
        these are: a fast mapping of data prior to a numerical factorization,
        the ability to modify pivots if the matrix is not definite, the
        efficient solution of several right-hand sides, a routine implementing
        iterative refinement, and the possibility of restarting the
        factorization should it run out of space.'


        References
        ----------

        1. I. S. Duff, "MA57 -- A New Code for the Solution of Sparse Symmetric
           Indefinite Systems", ACM Transactions on Mathematical Software (30),
           p. 118-144, 2004.
        2. I. S. Duff and J. K. Reid, "The Multifrontal Solution of Indefinite
           Sparse Symmetric Linear Systems", Transactions on Mathematical
           Software (9), p. 302--325, 1983.
        3. http://hsl.rl.ac.uk/hsl2007/hsl20074researchers.html
        """
 
        Sils.__init__( self, A, **kwargs )

        # Statistics on A
        self.nzFact = 0      # Number of nonzeros in factors
        self.nRealFact = 0   # Storage for real data of factors
        self.nIntFact = 0    # Storage for int  data of factors
        self.front = 0       # Largest front size
        self.n2x2pivots = 0  # Number of 2x2 pivots used
        self.neig = 0        # Number of negative eigenvalues detected
        self.rank = 0        # Matrix rank

        # Factors
        #self.L = spmatrix.ll_mat( self.n, self.n, 0 )
        #self.B = spmatrix.ll_mat_sym( self.n, 0 )

        # Analyze and factorize matrix
        self.context = _pyma57.factor( A, self.sqd )
        (self.nzFact, self.nRealFact, self.nIntFact, self.front,
         self.n2x2pivots, self.neig, self.rank) = self.context.stats()

        self.isFullRank = (self.rank == self.n)

    def solve( self, b, get_resid = True ):
        """
        solve(b) solves the linear system of equations Ax = b.
        The solution will be found in self.x and residual in
        self.residual.
        """
        self.context.ma57( b, self.x, self.residual, get_resid )
        return None

    def refine( self, b, nitref = 3, **kwargs ):
        """
        refine( b, nitref ) performs iterative refinement if necessary
        until the scaled residual norm ||b-Ax||/(1+||b||) falls below the
        threshold 'tol' or until nitref steps are taken.
        Make sure you have called solve() with the same right-hand
        side b before calling refine().
        The residual vector self.residual will be updated to reflect
        the updated approximate solution. 

        By default, nitref = 3.
        """
        (self.cond, self.cond2, self.berr,
         self.berr2, self.dirError,
         self.matNorm, self.xNorm,
         self.relRes) = self.context.refine(self.x, self.residual, b, nitref)
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

#     def fetch_lb( self ):
#         """
#         fetch_lb() returns the factors L and B of A such that
        
#               P^T  A P = L  B  L^T

#         where P is as in fetch_perm(), L is unit upper
#         triangular and B is block diagonal with 1x1 and 2x2
#         blocks. Access to the factors is available as soon
#         as a PyMa27Context has been instantiated.
#         """
#         self.context.fetchlb( self.L, self.B )
#         return None
        

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
    G = PyMa57Context( M )
    w = sys.stderr.write
    w( ' done\n' )
    w( ' Matrix order = %d\n' % G.n )
    w( ' Number of 2x2 pivots = %d\n' % G.n2x2pivots )
    w( ' Number of negative eigenvalues = %d\n' % G.neig )
    w( ' Matrix rank = %d\n' % G.rank )
    w( ' Matrix is rank deficient : %s\n' % repr(G.isFullRank) )
    w( ' Solving system... ' )
    G.solve( rhs )
    w( ' done\n' )
    w( ' Residual = %-g\n' % norms.norm_infty( G.residual ) )
    w( ' Relative error = %-g\n' % norms.norm_infty( G.x - e ) )
    w( ' Performing iterative refinement if necessary... ' )
    G.refine( rhs )
    w( ' done\n' )
    w( ' Computed estimates:\n' )
    w( '   Condition number estimate: %8.1e\n' % G.cond )
    w( '   First backward error estimate: %8.1e\n' % G.berr )
    w( '   Second backward error estimate: %8.1e\n' % G.berr2 )
    w( '   Direct error estimate: %8.1e\n' % G.dirError )
    w( '   Infinity-norm of input matrix: %8.1e\n' % G.matNorm )
    w( '   Infinity-norm of computed solution: %8.1e\n' % G.xNorm )
    w( '   Relative residual: %8.1e\n' % G.relRes )
    #w( ' Residual = %-g\n' % norms.norm_infty( G.residual ) )
    w( ' Relative error = %-g\n' % norms.norm_infty( G.x - e ) )

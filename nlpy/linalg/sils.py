"""
SILS: An abstract framework for the factorization of symmetric indefinite
matrices. The factorizations currently implemented are those of MA27 and MA57
from the Harwell Subroutine Library (http://hsl.rl.ac.uk).

$Id:$
"""

import numpy

class Sils:
    """
    Abstract class for the factorization and solution of symmetric indefinite
    systems of linear equations. The methods of this class must be overridden.
    """

    def __init__( self, A, **kwargs ):
        if not A.issym: self = None
        self.n = A.shape[0]
        self.sqd = 'sqd' in kwargs and kwargs['sqd']

        # Solution and residual vectors
        self.x = numpy.zeros( self.n, 'd' )
        self.residual = numpy.zeros( self.n, 'd' )

        self.context = None

    def solve( self, b, get_resid = True ):
        # Must be subclassed
        raise NotImplementedError

    def refine( self, b, nitref = 3 ):
        # Must be subclassed
        raise NotImplementedError

    def fetch_perm( self ):
        # Must be subclassed
        raise NotImplementedError

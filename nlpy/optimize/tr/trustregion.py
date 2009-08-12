"""
Class definition for Trust-Region Algorithm
"""

from nlpy.krylov import ppcg
from nlpy.krylov import pygltr
import numpy
from math import sqrt

class TrustRegionFramework:

    # Constructor
    def __init__( self, **kwargs ):
        """
        Initializes an object allowing management of a trust region.
        Valid optional keywords include
        
          Delta         Initial trust-region radius (default: 1.0)
          eta1          Step acceptance threshold   (default: 0.01)
          eta2          Radius increase threshold   (default: 0.99)
          gamma1        Radius decrease factor      (default: 1/3)
          gamma2        Radius increase factor      (default: 2.5)

        Subclass and override UpdateRadius() to implement custom trust-region
        management rules.

        See, e.g.,
         A. R. Conn, N. I. M. Gould and Ph. L. Toint, Trust-Region Methods,
         MP01 MPS-SIAM Series on Optimization, 2000.
        """
        self.Delta0 = kwargs.get( 'Delta', 1.0 ) # Initial trust-region radius
        self.Delta  = self.Delta0 # Current trust-region radius
        self.DeltaMax = 1.0e+10 # Largest trust-region radius
        self.eta1   = kwargs.get( 'eta1', 0.01 )     # Step acceptance threshold
        self.eta2   = kwargs.get( 'eta2', 0.99 )     # Radius increase threshold
        self.gamma1 = kwargs.get( 'gamma1', 1.0/3 )  # Radius decrease factor
        self.gamma2 = kwargs.get( 'gamma2', 2.5 )    # Radius increase factor
        self.eps    = self._Epsilon() # Machine epsilon

    def _Epsilon( self ):
        """
        Return approximate value of machine epsilon
        """
        one = 1.0
        eps = 1.0
        while (one + eps) > one:
            eps = eps / 2.0
        return eps*2.0

    def Rho( self, f, f_trial, m ):
        """
        Compute the ratio of actual versus predicted reduction
        rho = (f - f_trial)/(-m)
        """
        pred = -m + max( 1.0, abs( f ) ) * 10.0 * self.eps
        if pred > 0:
            ared = f - f_trial + max( 1.0, abs( f ) ) * 10.0 * self.eps
            return ared/pred
        else:
            # Error: Negative predicted reduction
            raise ValueError, 'TrustRegion:: Nonpositive predicted reduction!'
            return None

    def UpdateRadius( self, rho, stepNorm ):
        """
        Update the trust-region radius. The rule implemented by this method is:

          Delta = gamma1 * stepNorm          if ared/pred <  eta1
          Delta = gamma2 * Delta          if ared/pred >= eta2
          Delta unchanged otherwise,

        where ared/pred is the quotient computed by self.Rho().
        """
        if rho < self.eta1:
            self.Delta = self.gamma1 * stepNorm
        elif rho >= self.eta2:
            self.Delta = min( max(self.Delta, self.gamma2 * stepNorm),
                              self.DeltaMax )

    # Reset radius to original value
    def ResetRadius( self ):
        self.Delta = self.Delta0


class TrustRegionSolver:

    def __init__( self, g, **kwargs ):
        self.g = g

    def Solve( self ):
        # Must override
        return None

class TrustRegionCG( TrustRegionSolver ):

    def __init__( self, g, **kwargs ):
        """
        Instantiate a trust-region subproblem solver based on the truncated
        conjugate gradient of Steihaug and Toint.
        See module ppcg for more information.
        """
        TrustRegionSolver.__init__(self, g, **kwargs)
        self.cgSolver = ppcg.Ppcg(g, **kwargs)
        self.niter = 0
        self.stepNorm = 0.0
        self.step = None
        self.hprod = kwargs.get( 'matvec', None )
        self.H = kwargs.get( 'H', None )
        if self.hprod is None and self.H is None:
            raise ValueError, 'Specify one of hprod and H'
        self.m = None # Model value at candidate solution

    def Solve( self ):
        """
        Solve trust-region subproblem using the truncated conjugate gradient
        algorithm.
        """
        self.cgSolver.Solve()
        self.niter = self.cgSolver.iter
        self.stepNorm = sqrt( self.cgSolver.xNorm2 )
        self.step= self.cgSolver.x
        # Compute model reduction
        m = numpy.dot(self.g, self.step)
        if self.hprod:
            m += 0.5 * numpy.dot( self.step, self.hprod(self.step) )
        else:
            v = numpy.empty(self.H.shape[0], 'd')
            self.H.matvec(self.step, v)
            m += 0.5 * numpy.dot( self.step, v )
        self.m = m
        return

class TrustRegionGLTR( TrustRegionSolver ):

    def __init__( self, g, **kwargs ):
        """
        Instantiate a trust-region subproblem solver based on the Generalized
        Lanczos iterative method of Gould, Lucidi, Roma and Toint.
        See module pygltr for more information.
        """
        TrustRegionSolver.__init__(self, g, **kwargs)
        self.gltrSolver = pygltr.PyGltrContext(g, **kwargs)
        self.niter = 0
        self.stepNorm = 0.0
        self.step = None
        self.hprod = kwargs.get( 'matvec', None )
        self.H = kwargs.get( 'H', None )
        if self.hprod is None and self.H is None:
            raise ValueError, 'Specify one of hprod and H'
        self.m = None

    def Solve( self ):
        """
        Solve the trust-region subproblem using the generalized Lanczos method.
        """
        if self.hprod is not None:
            self.gltrSolver.implicit_solve( self.hprod )
        else:
            self.gltrSolver.explicit_solve( self.H )

        self.niter = self.gltrSolver.niter
        self.stepNorm = self.gltrSolver.snorm
        self.step = self.gltrSolver.step
        self.m = self.gltrSolver.m
        return

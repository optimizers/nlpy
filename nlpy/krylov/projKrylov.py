"""
A general framework for implementing projected Krylov methods. Such methods are
variations on all the well-known Krylov methods to solve block augmented linear
systems, i.e., linear systems of the form

          [ H    A^T ] [ x ] = [ c ]
          [ A     0  ] [ y ]   [ b ],

where H and A are matrices and A^T is the transpose of A. Here, H may or may
not be symmetric and may or may not have stronger properties, such as positive
definiteness. It may be available as an operator only or explicitly. However,
all projected Krylov methods currently require that B be available explicitly.

Such matrices arise, for example, in the solution of partial differential
equations (e.g., Maxwell or Navier-Stokes) by the finite-element method. For
more information on general Krylov methods and projected Krylov methods, see
the references below.

This module defines the `ProjectedKrylov` generic class. Other modules subclass
`ProjectedKrylov` to implement specific algorithms. Currently, the following
methods are implemented

          +------------------------------+----------+----------+
          | Method                       | Module   | Class    |
          +==============================+==========+==========+
          | Projected Conjugate Gradient | ppcg     | Ppcg     |
          +------------------------------+----------+----------+
          | Projected Bi-CGSTAB          | pbcgstab | Pbcgstab |
          +------------------------------+----------+----------+

Other projected iterative methods may be found as part of the PyKrylov
package. See http://github.com/dpo/pykrylov.

References
----------

.. [Kel95] C. T. Kelley, *Iterative Methods for Linear and Nonlinear
           Equations*, SIAM, Philadelphia, PA, 1995.
.. [Orb08] D. Orban, *Projected Krylov Methods for Unsymmetric Augmented
           Systems*, Cahiers du GERAD G-2008-46, GERAD, Montreal, Canada, 2008.

"""

__docformat__ = 'restructuredtext'

import numpy
from pysparse.sparse import spmatrix   # To assemble the projection matrix

try:                            # To compute projections
    from nlpy.linalg.pyma57 import PyMa57Context as LBLContext
except:
    from nlpy.linalg.pyma27 import PyMa27Context as LBLContext

from nlpy.tools import norms
from nlpy.tools.timing import cputime
import sys

class ProjectedKrylov:
    def __init__(self, c, **kwargs):

        self.prefix = 'Generic PK: '   # Should be overridden in subclass
        self.name = 'Generic Projected Krylov Method (should be subclassed)'

        self.debug = kwargs.get('debug', False)
        self.abstol = kwargs.get('abstol', 1.0e-8)
        self.reltol = kwargs.get('reltol', 1.0e-6)
        self.max_itref = kwargs.get('max_itref', 3)
        self.itref_tol = kwargs.get('itref_tol', 1.0e-6)
        self.factorize = kwargs.get('factorize', True)
        self.precon = kwargs.get('precon', None)

        # Optional keyword arguments
        self.A = kwargs.get('A', None)
        self.b = kwargs.get('rhs', None)
        self.n = c.shape[0]            # Number of variables
        if self.A is None:
            self.m = 0
            self.nnzA = 0
        else:
            self.m = self.A.shape[0]  # Number of constraints
            self.nnzA = self.A.nnz    # Number of nonzeros in constraint matrix
        self.nnzP = 0                 # Number of nonzeros in projection matrix
        self.c = c
        self.maxiter = kwargs.get('maxiter', 2 * self.n)

        self.matvec = kwargs.get('matvec', None)
        self._matvec_found = (self.matvec != None)
        if self.matvec is None:
            self.H = kwargs.get('H', None)
            if self.H is None:
                raise ValueError, 'Either matvec or H must be supplied'

        self.Proj = kwargs.get('Proj', None)
        self.factorized = (self.Proj != None) # Factorization already performed

        # Initializations
        self.t_fact     = 0.0     # Timing of factorization phase
        self.t_feasible = 0.0     # Timing of feasibility phase
        self.t_solve    = 0.0     # Timing of iterative solution phase
        self.x_feasible = None
        self.converged = False


    def _write(self, msg):
        sys.stderr.write(self.prefix + msg)


    def Factorize(self):
        """
        Assemble projection matrix and factorize it

           P = [ G   A^T ]
               [ A    0  ],

        where G is the preconditioner, or the identity matrix if no
        preconditioner was given.
        """
        if self.A is None:
            raise ValueError, 'No linear equality constraints were specified'

        # Form projection matrix
        P = spmatrix.ll_mat_sym(self.n + self.m, self.nnzA + self.n)
        if self.precon is not None:
            P[:self.n,:self.n] = self.precon
        else:
            for i in range(self.n):
                P[i,i] = 1
        P[self.n:,:self.n] = self.A

        if self.debug:
                msg = 'Factorizing projection matrix '
                msg += '(size %-d, nnz = %-d)...\n' %  (P.shape[0],P.nnz)
                self._write(msg)
        self.t_fact = cputime()
        self.Proj = LBLContext(P)
        self.t_fact = cputime() - self.t_fact
        if self.debug:
                msg = ' done (%-5.2fs)\n' % self.t_fact
                self._write(msg)
        self.factorized = True
        return


    def CheckAccurate(self):
        """
        Make sure constraints are consistent and residual is satisfactory
        """
        scale_factor = norms.norm_infty(self.Proj.x[:self.n])
        if self.b is not None:
            scale_factor = max(scale_factor, norms.norm_infty(self.b))
        max_res = max(1.0e-6 * scale_factor, self.abstol)
        res = norms.norm_infty(self.Proj.residual)
        if res > max_res:
            if self.Proj.isFullRank:
                self._write(' Large residual. ' +
                             'Factorization likely inaccurate...\n')
            else:
                self._write(' Large residual. ' +
                             'Constraints likely inconsistent...\n')
        if self.debug:
            self._write(' accurate to within %8.1e...\n' % res)
        return


    def FindFeasible(self):
        """
        If rhs was specified, obtain x_feasible satisfying the constraints
        """
        n = self.n
        if self.debug: self._write('Obtaining feasible solution...\n')
        self.t_feasible = cputime()
        self.rhs[n:] = self.b
        self.Proj.solve(self.rhs)
        self.x_feasible = self.Proj.x[:n].copy()
        self.t_feasible = cputime() - self.t_feasible
        self.CheckAccurate()
        if self.debug:
            self._write(' done (%-5.2fs)\n' % self.t_feasible)
        return


    def Solve(self):
        """
        This is the Solve method of the abstract projectedKrylov class. The
        class must be specialized and this method overridden.
        """
        raise NotImplementedError, 'This method must be subclassed'

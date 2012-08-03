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
from pysparse.sparse.pysparseMatrix import PysparseMatrix

try:                            # To compute projections
    from nlpy.linalg.pyma57 import PyMa57Context as LBLContext
except:
    from nlpy.linalg.pyma27 import PyMa27Context as LBLContext

from nlpy.tools import norms
from nlpy.tools.timing import cputime
import logging


class ProjectedKrylov(object):
    """
    :keywords:
        :A:  the `constraint` matrix. Must be given as an explicit matrix.
        :H:  the operator in the leading block. Only matrix-vector products
             with ``H`` are required in projected Krylov methods. ``H`` can be
             given as a linear operator.
        :c:  the first part of the right-hand side vector.
        :b:  the second part of the right-hand side vector (default: ``None``,
             meaning the vector of zeros).
        :abstol:  absolute stopping tolerance (default: 1.0e-8).
        :reltol:  relative stopping tolerance (default: 1.0e-6).
        :maxiter:  maximum number of iterations of the Krylov method.
        :max_itref:  maximum number of iterative refinement steps after a
                     projection (default: 3).
        :itref_tol:  acceptable residual tolerance during iterative refinement
                     (default: 1.0e-6).
        :factorize: if set to ``True``, the projector will be factorized (this
                    is the default). If set to ``False``, an existing
                    factorization should be given in ``Proj``.
        :Proj:    an existing factorization of the projector. If not ``None``,
                  ``factorize`` will be set to ``False``.
        :precon:  preconditioner. Normally this is a cheap approximation to
                  ``H``. It must be specified as an explicit matrix.
        :logger_name:  Name of a logger (Default: `None`).
    """

    def __init__(self, c, H, **kwargs):

        self.prefix = 'Generic PK: '   # Should be overridden in subclass
        self.name = 'Generic Projected Krylov Method (should be subclassed)'

        self.debug = kwargs.get('debug', False)
        self.abstol = kwargs.get('abstol', 1.0e-8)
        self.reltol = kwargs.get('reltol', 1.0e-6)
        self.max_itref = kwargs.get('max_itref', 3)
        self.itref_tol = kwargs.get('itref_tol', 1.0e-6)
        self.factorize = kwargs.get('factorize', True)
        self.precon = kwargs.get('precon', None)

        logger_name = kwargs.get('logger_name', None)
        self.log = logging.getLogger(logger_name)
        self.log.propagate = False

        # Optional keyword arguments
        self.A = kwargs.get('A', None)
        if self.A is not None:
            self.log.debug('Constraint matrix has shape (%d,%d)' % \
                    (self.A.shape[0], self.A.shape[1]))
            if isinstance(self.A, PysparseMatrix):
                self.A = self.A.matrix
        else:
            self.log.debug('No constraint matrix specified')
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
        self.H = H
        self.maxiter = kwargs.get('maxiter', 2 * self.n)

        self.Proj = kwargs.get('Proj', None)
        self.factorized = (self.Proj != None) # Factorization already performed

        self.dreg = kwargs.get('dreg', 0.0)   # Dual regularization.

        # Initializations
        self.t_fact     = 0.0     # Timing of factorization phase
        self.t_feasible = 0.0     # Timing of feasibility phase
        self.t_solve    = 0.0     # Timing of iterative solution phase
        self.x_feasible = None
        self.converged = False


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
            r = range(self.n)
            P.put(1, r, r)
            #for i in range(self.n):
            #    P[i,i] = 1
        P[self.n:,:self.n] = self.A

        # Add regularization if requested.
        if self.dreg > 0.0:
            r = range(self.n, self.n + self.m)
            P.put(-self.dreg, r, r)

        msg = 'Factorizing projection matrix '
        msg += '(size %-d, nnz = %-d)...' %  (P.shape[0],P.nnz)
        self.log.debug(msg)
        self.t_fact = cputime()
        self.Proj = LBLContext(P)
        self.t_fact = cputime() - self.t_fact
        self.log.debug('... done (%-5.2fs)' % self.t_fact)
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
                self.log.info(' Large residual. ' +
                              'Factorization likely inaccurate...')
            else:
                self.log.info(' Large residual. ' +
                              'Constraints likely inconsistent...')
        self.log.debug(' accurate to within %8.1e...' % res)
        return


    def FindFeasible(self):
        """
        If rhs was specified, obtain x_feasible satisfying the constraints
        """
        n = self.n
        self.log.debug('Obtaining feasible solution...')
        self.t_feasible = cputime()
        self.rhs[n:] = self.b
        self.Proj.solve(self.rhs)
        self.x_feasible = self.Proj.x[:n].copy()
        self.t_feasible = cputime() - self.t_feasible
        self.CheckAccurate()
        self.log.debug('... done (%-5.2fs)' % self.t_feasible)
        return


    def Solve(self):
        """
        This is the Solve method of the abstract projectedKrylov class. The
        class must be specialized and this method overridden.
        """
        raise NotImplementedError, 'This method must be overridden.'

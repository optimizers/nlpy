# nlp.py
# Define an abstract class to represent a general
# nonlinear optimization problem.
# D. Orban, 2004.
from nlpy.model import KKTresidual
from pykrylov.linop import LinearOperator, DiagonalOperator
from pykrylov.linop import BlockLinearOperator, ReducedLinearOperator
from pykrylov.linop import linop_from_ndarray
from nlpy.tools.decorators import deprecated
from nlpy.tools.utils import where
from pysparse.sparse import PysparseMatrix as psp
from scipy import sparse as sp
import logging
import numpy as np
import sys


class NLPModel(object):
    """
    An abstract nonlinear optimization problem. It features methods to
    evaluate the objective and constraint functions, and their derivatives.
    Instances of the general class do not do anything interesting; they must
    be subclassed and specialized.

    :parameters:

        :n:       number of variables (default: 0)
        :m:       number of general (non bound) constraints (default: 0)
        :name:    model name (default: 'Generic')

    :keywords:

        :x0:      initial point (default: all 0)
        :pi0:     vector of initial multipliers (default: all 0)
        :Lvar:    vector of lower bounds on the variables
                  (default: all -Infinity)
        :Uvar:    vector of upper bounds on the variables
                  (default: all +Infinity)
        :Lcon:    vector of lower bounds on the constraints
                  (default: all -Infinity)
        :Ucon:    vector of upper bounds on the constraints
                  (default: all +Infinity)
    """

    _id = -1

    def __init__(self, n=0, m=0, name='Generic', **kwargs):

      self._nvar = self._n = n   # Number of variables
      self._ncon = self._m = m   # Number of general constraints
      self._name = name         # Problem name

      # Set initial point
      if 'x0' in kwargs.keys():
        self.x0 = np.ascontiguousarray(kwargs['x0'], dtype=float)
      else:
        self.x0 = np.zeros(self.n, dtype=np.float)

      # Set initial multipliers
      if 'pi0' in kwargs.keys():
        self.pi0 = np.ascontiguousarray(kwargs['pi0'], dtype=np.float)
      else:
        self.pi0 = np.zeros(self.m, dtype=np.float)

      # Set lower bounds on variables    Lvar[i] <= x[i]  i = 1,...,n
      if 'Lvar' in kwargs.keys():
        self.Lvar = np.ascontiguousarray(kwargs['Lvar'], dtype=np.float)
      else:
        self.Lvar = -np.inf * np.ones(self.n, dtype=np.float)

      # Set upper bounds on variables    x[i] <= Uvar[i]  i = 1,...,n
      if 'Uvar' in kwargs.keys():
        self.Uvar = np.ascontiguousarray(kwargs['Uvar'], dtype=np.float)
      else:
        self.Uvar = np.inf * np.ones(self.n, dtype=np.float)

      # Set lower bounds on constraints  Lcon[i] <= c[i]  i = 1,...,m
      if 'Lcon' in kwargs.keys():
        self.Lcon = np.ascontiguousarray(kwargs['Lcon'], dtype=np.float)
      else:
        self.Lcon = -np.inf * np.ones(self.m, dtype=np.float)

      # Set upper bounds on constraints  c[i] <= Ucon[i]  i = 1,...,m
      if 'Ucon' in kwargs.keys():
        self.Ucon = np.ascontiguousarray(kwargs['Ucon'], dtype=np.float)
      else:
        self.Ucon = np.inf * np.ones(self.m, dtype=np.float)

      # Default classification of constraints
      self._lin = []                        # Linear    constraints
      self._nln = range(self.m)             # Nonlinear constraints
      self._net = []                        # Network   constraints
      self._nlin = len(self.lin)            # Number of linear constraints
      self._nnln = len(self.nln)            # Number of nonlinear constraints
      self._nnet = len(self.net)            # Number of network constraints

      # Maintain lists of indices for each type of constraints:
      self.rangeC = []    # Range constraints:       cL <= c(x) <= cU
      self.lowerC = []    # Lower bound constraints: cL <= c(x)
      self.upperC = []    # Upper bound constraints:       c(x) <= cU
      self.equalC = []    # Equality constraints:    cL  = c(x)  = cU

      for i in range(self.m):
        if self.Lcon[i] > -np.inf and self.Ucon[i] < np.inf:
          if self.Lcon[i] == self.Ucon[i]:
            self.equalC.append(i)
          else:
            self.rangeC.append(i)
        elif self.Lcon[i] > -np.inf:
          self.lowerC.append(i)
        elif self.Ucon[i] < np.inf:
          self.upperC.append(i)

      self.nlowerC = len(self.lowerC)   # Number of lower bound constraints
      self.nrangeC = len(self.rangeC)   # Number of range constraints
      self.nupperC = len(self.upperC)   # Number of upper bound constraints
      self.nequalC = len(self.equalC)   # Number of equality constraints

      # Define permutations to order constraints / multipliers.
      self.permC = self.equalC + self.lowerC + self.upperC + self.rangeC

      # Proceed similarly with bound constraints
      self.rangeB = []
      self.lowerB = []
      self.upperB = []
      self.fixedB = []
      self.freeB  = []

      for i in range(self.n):
        if self.Lvar[i] > -np.inf and self.Uvar[i] < np.inf:
          if self.Lvar[i] == self.Uvar[i]:
            self.fixedB.append(i)
          else:
            self.rangeB.append(i)
        elif self.Lvar[i] > -np.inf:
          self.lowerB.append(i)
        elif self.Uvar[i] < np.inf:
          self.upperB.append(i)
        else:
          self.freeB.append(i)

      self.nlowerB = len(self.lowerB)
      self.nrangeB = len(self.rangeB)
      self.nupperB = len(self.upperB)
      self.nfixedB = len(self.fixedB)
      self.nfreeB  = len(self.freeB )
      self.nbounds = self.n - self.nfreeB

      # Define permutations to order bound constraints / multipliers.
      self.permB = self.fixedB + self.lowerB + self.upperB + \
                   self.rangeB + self.freeB

      # Define default stopping tolerances
      self._stop_d = 1.0e-6    # Dual feasibility
      self._stop_c = 1.0e-6    # Complementarty
      self._stop_p = 1.0e-6    # Primal feasibility

      # Define scaling attributes.
      self.g_max = 1.0e2      # max gradient entry (constant)
      self.scale_obj = None   # Objective scaling
      self.scale_con = None   # Constraint scaling

      # Problem-specific logger.
      self.__class__._id += 1
      self._id = self.__class__._id
      self.logger = logging.getLogger(name=self.name + '_' + str(self._id))
      self.logger.setLevel(logging.INFO)
      fmt = logging.Formatter('%(name)-10s %(levelname)-8s %(message)s')
      hndlr = logging.StreamHandler(sys.stdout)
      hndlr.setFormatter(fmt)
      self.logger.addHandler(hndlr)

    @property
    def nvar(self):
      "Number of variables."
      return self._nvar

    @property
    def n(self):
      "Number of variables."
      return self._n

    @property
    def ncon(self):
      "Number of constraints (excluding bounds)."
      return self._ncon

    @property
    def m(self):
      "Number of constraints (excluding bounds)."
      return self._m

    @property
    def name(self):
      "Problem name."
      return self._name

    @property
    def lin(self):
      "Indices of linear constraints."
      return self._lin

    @property
    def nlin(self):
      "Number of linear constraints."
      return self._nlin

    @property
    def nln(self):
      "Indices of nonlinear constraints."
      return self._nln

    @property
    def nnln(self):
      "Number of nonlinear constraints."
      return self._nnln

    @property
    def nnet(self):
      "Number of network constraints."
      return self._nnet

    @property
    def net(self):
      "Inidices of network constraints."
      return self._net

    @property
    def stop_d(self):
      "Tolerance on dual feasibility"
      return self._stop_d

    @stop_d.setter
    def stop_d(self, value):
      self._stop_d = max(0, value)

    @property
    def stop_c(self):
      "Tolerance on complementarity"
      return self._stop_c

    @stop_c.setter
    def stop_c(self, value):
      self._stop_c = max(0, value)

    @property
    def stop_p(self):
      "Tolerance on primal feasibility"
      return self._stop_p

    @stop_p.setter
    def stop_p(self, value):
      self._stop_p = max(0, value)

    @deprecated
    def get_stopping_tolerances(self):
      "Return current stopping tolerances"
      return (self.stop_d, self.stop_p, self.stop_c)

    @deprecated
    def set_stopping_tolerances(self, stop_d, stop_p, stop_c):
      self.stop_d = stop_d
      self.stop_p = stop_p
      self.stop_c = stop_c
      return

    def compute_scaling_obj(self, x=None, g_max=1.0e2, reset=False):
        """
        Compute objective scaling.

        :parameters:

            :x: Determine scaling by evaluating functions at this
                point. Default is to use :attr:`self.x0`.
            :g_max: Maximum allowed gradient. Default: :attr:`g_max = 1e2`.
            :reset: Set to `True` to unscale the problem.

        The procedure used here closely
        follows IPOPT's behavior; see Section 3.8 of

          Waecther and Biegler, 'On the implementation of an
          interior-point filter line-search algorithm for large-scale
          nonlinear programming', Math. Prog. A (106), pp.25-57, 2006

        which is a scalar rescaling that ensures the inf-norm of the
        gradient (at x) isn't larger than 'g_max'.
        """
        # Remove scaling if requested
        if reset:
            self.scale_obj = None
            self.pi0 = self.model.get_pi0()  # get original multipliers
            return

        # Quick return if the problem is already scaled
        if self.scale_obj is not None:
            return

        if x is None: x = self.x0
        g = self.grad(x)
        gNorm = np.linalg.norm(g, np.inf)
        self.scale_obj = g_max / max(g_max, gNorm)  # <= 1 always

        # Rescale the Lagrange multiplier
        self.pi0 *= self.scale_obj

        return gNorm

    def compute_scaling_cons(self, x=None, g_max=1.0e2, reset=False):
        """
        Compute constraint scaling.

        :parameters:

            :x: Determine scaling by evaluating functions at this
                point. Default is to use :attr:`self.x0`.
            :g_max: Maximum allowed gradient. Default: :attr:`g_max = 1e2`.
            :reset: Set to `True` to unscale the problem.
        """
        # Remove scaling if requested
        if reset:
            self.scale_con = None
            self.Lcon = self.model.get_Lcon()  # lower bounds on constraints
            self.Ucon = self.model.get_Ucon()  # upper bounds on constraints
            return

        # Quick return if the problem is already scaled
        if self.scale_con is not None:
            return

        m = self.m
        if x is None: x = self.x0
        d_c = np.empty(m)
        J = self.jop(x)

        # Find inf-norm of each row of J
        gmaxNorm = 0            # holds the maxiumum row-norm of J
        imaxNorm = 0            # holds the corresponding index
        e = np.zeros(self.ncon)
        for i in xrange(m):
            e[i] = 1
            giNorm = np.linalg.norm(J.T * e, 1)  # Matrix 1-norm (max abs col)
            e[i] = 0
            d_c[i] = g_max / max(g_max, giNorm)  # <= 1 always
            if giNorm > gmaxNorm:
                gmaxNorm = giNorm
                imaxNorm = i
            gmaxNorm = max(gmaxNorm, giNorm)

        self.scale_con = d_c

        # Scale constraint bounds: componentwise multiplications
        self.Lcon *= d_c        # lower bounds on constraints
        self.Ucon *= d_c        # upper bounds on constraints

        # Return largest row norm and its index
        return (imaxNorm, gmaxNorm)

    def primal_feasibility(self, x, c=None):
        """
        Evaluate the primal feasibility residual at x. If `c` is given, it
        should conform to :meth:`cons_pos`.
        """
        # Shortcuts.
        eC = self.equalC ; m = self.m ; nrC = self.nrangeC
        nB = self.nbounds ; nrB = self.nrangeB

        pFeas = np.empty(m+nrC+nB+nrB)
        pFeas[:m+nrC] = -self.cons_pos(x) if c is None else -c
        not_eC = [i for i in range(m+nrC) if i not in eC]
        pFeas[eC] = np.abs(pFeas[eC])
        pFeas[not_eC] = np.maximum(0, pFeas[not_eC])
        pFeas[m:m+nrC] = np.maximum(0, pFeas[m:m+nrC])
        pFeas[m+nrC:] = -self.get_bounds(x)
        pFeas[m+nrC:] = np.maximum(0, pFeas[m+nrC:])

        return pFeas

    def dual_feasibility(self, x, y, z, g=None, J=None, **kwargs):
        """
        Evaluate the dual feasibility residual at (x,y,z). The argument `J`,
        if supplied, should be a linear operator representing the constraints
        Jacobian. It should conform to either :meth:`jac` or :meth:`jac_pos`
        depending on the value of `all_pos` (see below).

        The multipliers `z` should conform to :meth:`get_bounds`.

        :keywords:
            :obj_weight: weight of the objective gradient in dual feasibility.
                         Set to zero to check Fritz-John conditions instead
                         of KKT conditions. (default: 1.0)
            :all_pos:    if `True`, indicates that the multipliers `y` conform
                         to :meth:`jac_pos`. If `False`, `y` conforms to
                         :meth:`jac`. In all cases, `y` should be appropriately
                         ordered. If the positional argument `J` is specified,
                         it must be consistent with the layout of `y`.
                         (default: `True`)
        """
        # Shortcuts.
        lB  = self.lowerB  ; uB  = self.upperB  ; rB  = self.rangeB
        nlB = self.nlowerB ; nuB = self.nupperB ; nrB = self.nrangeB

        obj_weight = kwargs.get('obj_weight', 1.0)
        all_pos = kwargs.get('all_pos', True)

        if J is None:
            J = self.jop_pos(x) if all_pos else self.jop(x)

        if obj_weight == 0.0:   # Checking Fritz-John conditions.
            dFeas = -J.T * y
        else:
            dFeas = self.grad(x) if g is None else g[:]
            if obj_weight != 1.0:
                dFeas *= obj_weight
            dFeas -= J.T * y
        dFeas[lB] -= z[:nlB]
        dFeas[uB] -= z[nlB:nlB+nuB]
        dFeas[rB] -= z[nlB+nuB:nlB+nuB+nrB] - z[nlB+nuB+nrB:]

        return dFeas

    def complementarity(self, x, y, z, c=None):
        """
        Evaluate the complementarity residuals at (x,y,z). If `c` is specified,
        it should conform to :meth:`cons_pos` and the multipliers `y` should
        appear in the same order. The multipliers `z` should conform to
        :meth:`get_bounds`.

        :returns:
            :cy:  complementarity residual for general constraints
            :xz:  complementarity residual for bound constraints.
        """
        # Shortcuts.
        lC  = self.lowerC  ; uC  = self.upperC  ; rC  = self.rangeC
        nlC = self.nlowerC ; nuC = self.nupperC ; nrC = self.nrangeC

        not_eC = lC+uC+rC + range(nlC+nuC+nrC, nlC+nuC+nrC+nrC)
        if c is None: c = self.cons_pos(x)

        cy = c[not_eC] * y[not_eC]
        xz = self.get_bounds(x) * z

        return (cy, xz)

    def kkt_residuals(self, x, y, z, c=None, g=None, J=None, **kwargs):
        """
        Return the first-order residuals. There is no check on the sign of the
        multipliers unless `check` is set to `True`. Keyword arguments not
        specified below are passed directly to :meth:`primal_feasibility`,
        :meth:`dual_feasibility` and :meth:`complementarity`.

        If `J` is specified, it should conform to :meth:`jac_pos` and the
        multipliers `y` should be consistent with the Jacobian.

        :keywords:
            :check:  check sign of multipliers.

        :returns:
            :kkt:  KKT residuals as a KKTresidual instance.
        """
        # Shortcuts.
        m = self.m ; nrC = self.nrangeC ; eC = self.equalC
        check = kwargs.get('check', True)

        if check:
            not_eC = [i for i in range(m+nrC) if i not in eC]
            if len(where(y[not_eC] < 0)) > 0:
                raise ValueError('Multipliers for inequalities must be >= 0.')
            if not np.all(z >= 0):
                raise ValueError('Multipliers for bounds must be >= 0.')

        pFeas = self.primal_feasibility(x, c=c)
        dFeas = self.dual_feasibility(x, y, z, g=g, J=J)
        cy, xz = self.complementarity(x, y, z, c=c)

        return KKTresidual(dFeas, pFeas[:m+nrC], pFeas[m+nrC:], cy, xz)

    def at_optimality(self, x, z, **kwargs):
      """
      Checks whether the KKT residuals meet the stopping conditions.
      """
      kkt = self.optimality_residuals(x, z, **kwargs)
      return kkt.dFeas <= self.stop_d and \
             kkt.comp <= self.stop_c and  \
             kkt.feas <= self.stop_p

    def bounds(self, x):
        """
        Return the vector with components x[i]-Lvar[i] or Uvar[i]-x[i] in such
        a way that the bound constraints on the problem variables are
        equivalent to bounds(x) >= 0. The bounds are odered as follows:

        [lowerB | upperB | rangeB (lower) | rangeB (upper) ].
        """
        lB  = self.lowerB  ; uB   = self.upperB  ; rB   = self.rangeB
        nlB = self.nlowerB ; nuB  = self.nupperB ; nrB  = self.nrangeB
        nB  = self.nbounds ; Lvar = self.Lvar    ; Uvar = self.Uvar

        b = np.empty(nB+nrB, dtype=x.dtype)
        b[:nlB] = x[lB] - Lvar[lB]
        b[nlB:nlB+nuB] = Uvar[uB] - x[uB]
        b[nlB+nuB:nlB+nuB+nrB] = x[rB] - Lvar[rB]
        b[nlB+nuB+nrB:] = Uvar[rB] - x[rB]
        return b

    def obj(self, x, **kwargs):
      "Evaluate the objective function at x."
      raise NotImplementedError('This method must be subclassed.')

    def grad(self, x, **kwargs):
      "Evaluate the objective gradient at x."
      raise NotImplementedError('This method must be subclassed.')

    def cons(self, x, **kwargs):
      "Evaluate vector of constraints at x."
      raise NotImplementedError('This method must be subclassed.')

    def cons_pos(self, x):
      """
      Convenience function to return the vector of constraints
      reformulated as

          ci(x) - ai  = 0  for i in equalC
          ci(x) - Li >= 0  for i in lowerC + rangeC
          Ui - ci(x) >= 0  for i in upperC + rangeC.

      The constraints appear in natural order, except for the fact that the
      'upper side' of range constraints is appended to the list.

      Scaling should be applied in cons().
      """
      m = self.m
      equalC = self.equalC
      lowerC = self.lowerC
      upperC = self.upperC
      rangeC = self.rangeC ; nrangeC = self.nrangeC

      # Set the type of c to the type of x to allow for object arrays.
      # This is useful to AD packages.
      c = np.empty(m + nrangeC, dtype=x.dtype)
      c[:m] = self.cons(x)
      c[m:] = c[rangeC]

      c[equalC] -= self.Lcon[equalC]
      c[lowerC] -= self.Lcon[lowerC]
      c[upperC] -= self.Ucon[upperC] ; c[upperC] *= -1
      c[rangeC] -= self.Lcon[rangeC]
      c[m:]     -= self.Ucon[rangeC] ; c[m:] *= -1

      return c

    def icons(self, i, x, **kwargs):
      "Evaluate i-th constraint at x."
      raise NotImplementedError('This method must be subclassed.')

    def igrad(self, i, x, **kwargs):
      "Evalutate i-th dense constraint gradient at x."
      raise NotImplementedError('This method must be subclassed.')

    def sigrad(self, i, x, **kwargs):
      "Evaluate i-th sparse constraint gradient at x."
      raise NotImplementedError('This method must be subclassed.')

    def jac(self, x, **kwargs):
      "Evaluate constraints Jacobian at x."
      raise NotImplementedError('This method must be subclassed.')

    def jac_pos(self, x, **kwargs):
      "Evaluate the Jacobian of :meth:`cons_pos` at x."
      raise NotImplementedError('This method must be subclassed.')

    def jprod(self, x, p, **kwargs):
      "Evaluate Jacobian-vector product at x with p."
      raise NotImplementedError('This method must be subclassed')

    def jtprod(self, x, p, **kwargs):
      "Evaluate transposed-Jacobian-vector product at x with p."
      raise NotImplementedError('This method must be subclassed')

    def jop(self, x):
      "Obtain Jacobian at xas a linear operator."
      return LinearOperator(self.n, self.m,
                            lambda v: self.jprod(x, v),
                            matvec_transp=lambda u: self.jtprod(x, u),
                            symmetric=False,
                            dtype=np.float)

    def jop_pos(self, x):
      "Jacobian of :meth:`cons_pos` at x as a linear operator."
      J = self.jop(x)
      e = np.ones(self.ncon + self.nrangeC)
      e[self.upperC] = -1
      e[self.ncon:] = -1
      JR = ReducedLinearOperator(J, self.rangeC, range(self.nvar))
      Jpos = BlockLinearOperator([[J], [JR]], dtype=np.float)
      D = DiagonalOperator(e)
      return D * Jpos  # Flip sign of 'upper' constraints.

    def lag(self, x, z, **kwargs):
      """
      Evaluate Lagrangian at (x, z). The constraints and bounds are
      assumed to be ordered as in :meth:`cons_pos` and :meth:`bounds`.
      """
      m = self.m ; nrC = self.nrangeC
      l = self.obj(x)
      # The following ifs are necessary because np.dot returns None
      # when passed empty arrays of objects (i.e., dtype = np.object).
      # This causes AD tools to error out.
      if self.m > 0:
        l -= np.dot(z[:m+nrC], self.cons_pos(x))
      if self.nbounds > 0:
        l -= np.dot(z[m+nrC:], self.bounds(x))
      return l

    def hess(self, x, z=None, **kwargs):
      "Evaluate Lagrangian Hessian at (x, z)."
      raise NotImplementedError('This method must be subclassed.')

    def hprod(self, x, z, p, **kwargs):
      """
      Evaluate matrix-vector product between
      the Hessian of the Lagrangian at (x, z) and p.
      """
      raise NotImplementedError('This method must be subclassed.')

    def hiprod(self, i, x, p, **kwargs):
      """
      Evaluate matrix-vector product between
      the Hessian of the i-th constraint at x and p.
      """
      raise NotImplementedError('This method must be subclassed.')

    def ghivprod(self, g, v, **kwargs):
      """
      Evaluate the vector of dot products (g, Hi*v) where Hi is the Hessian
      of the i-th constraint at x, i = 1, ..., ncon.
      """
      raise NotImplementedError('This method must be subclassed.')

    def hop(self, x, z=None, **kwargs):
      "Obtain Lagrangian Hessian at (x, z) as a linear operator."
      return LinearOperator(self.n, self.n,
                            lambda v: self.hprod(x, z, v, **kwargs),
                            symmetric=True,
                            dtype=np.float)

    def display_basic_info(self):
      """
      Display vital statistics about the current model.
      """
      write = self.logger.info
      write('Problem Name: %s' % self.name)
      write('Number of Variables: %d' % self.n)
      write('Number of Bound Constraints: %d' % self.nbounds)
      write(' (%d lower, %d upper, %d two-sided)' %
            (self.nlowerB, self.nupperB, self.nrangeB))
      if self.nlowerB > 0: write('Lower bounds: %s' % repr(self.lowerB))
      if self.nupperB > 0: write('Upper bounds: %s' % repr(self.upperB))
      if self.nrangeB > 0: write('Two-Sided bounds: %s' % repr(self.rangeB))
      if self.nlowerB + self.nupperB + self.nrangeB > 0:
        write('Vector of lower bounds: %s' % repr(self.Lvar))
        write('Vectof of upper bounds: %s' % repr(self.Uvar))
      write('Number of General Constraints: %d' % self.m)
      write(' (%d equality, %d lower, %d upper, %d range)' %
            (self.nequalC, self.nlowerC, self.nupperC, self.nrangeC))
      if self.nequalC > 0: write('Equality: %s' % repr(self.equalC))
      if self.nlowerC > 0: write('Lower   : %s' % repr(self.lowerC))
      if self.nupperC > 0: write('Upper   : %s' % repr(self.upperC))
      if self.nrangeC > 0: write('Range   : %s' % repr(self.rangeC))
      if self.nequalC + self.nlowerC + self.nupperC + self.nrangeC > 0:
        write('Vector of constraint lower bounds: %s' % repr(self.Lcon))
        write('Vector of constraint upper bounds: %s' % repr(self.Ucon))
        write('Number of Linear Constraints: %d' % self.nlin)
        write('Number of Nonlinear Constraints: %d' % self.nnln)
        write('Number of Network Constraints: %d' % self.nnet)
      write('Initial Guess: %s' % repr(self.x0))

      return

    def __repr__(self):
      dat = (self.__class__.__name__, self.name, self.n, self.m)
      return '%s %s with %d variables and %d constraints' % dat


class PySparseNLPModel(NLPModel):
  """
  An `NLPModel` where sparse matrices are returned in PySparse format.
  The `NLPModel`'s `jac` and `hess` methods should return that sparse
  Jacobian and Hessian in coordinate format: (vals, rows, cols).
  """

  def hess(self, *args, **kwargs):
    vals, rows, cols = super(PySparseNLPModel, self).hess(*args, **kwargs)
    H = psp(size=self.nvar, sizeHint=vals.size, symmetric=True)
    H.put(vals, rows, cols)
    return H

  def jac(self, *args, **kwargs):
    vals, rows, cols = super(PySparseNLPModel,
                             self).jac(*args, **kwargs)
    J = psp(nrow=self.ncon, ncol=self.nvar,
            sizeHint=vals.size, symmetric=False)
    J.put(vals, rows, cols)
    return J


class SciPyNLPModel(NLPModel):
  """
  An `NLPModel` where sparse matrices are returned in SciPy
  coordinate (COO) format. The `NLPModel`'s `jac` and `hess` methods
  should return that sparse Jacobian and Hessian in coordinate format:
  (vals, rows, cols).
  """

  def hess(self, *args, **kwargs):
    vals, rows, cols = super(SciPyNLPModel, self).hess(*args, **kwargs)
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(self.nvar, self.nvar))

  def jac(self, *args, **kwargs):
    if self.ncon == 0:  # SciPy cannot create sparse matrix of size 0.
      return linop_from_ndarray(np.empty((0, self.nvar), dtype=np.float))
    vals, rows, cols = super(SciPyNLPModel, self).jac(*args, **kwargs)
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(self.ncon, self.nvar))


class QPModel(NLPModel):
    """
    A generic class to represent a quadratic programming problem

    minimize    c'x + 1/2 x'Hx
    subject to  L <= A*x <= U
                l <=  x  <= u.
    """

    def __init__(self, c, H, A=None, name='GenericQP', **kwargs):
        """
        :parameters:
            :c:   Numpy array to represent the linear objective
            :A:   linear operator to represent the constraint matrix.
                  It must be possible to perform the operations `A*x`
                  and `A.T*y` for Numpy arrays `x` and `y` of appropriate size.
                  If `A` is `None`, it will be replaced with an empty linear
                  operator.
            :H:   linear operator to represent the objective Hessian.
                  It must be possible to perform the operation `H*x`
                  for a Numpy array `x` of appropriate size. The operator `H`
                  should be symmetric.

        See the documentation of `NLPModel` for futher information.
        """

        # Basic checks.
        n = c.shape[0]
        if A is None:
            m = 0
            self.A = LinearOperator(n, 0,
                                    lambda x: np.empty((0, 1)),
                                    matvec_transp=lambda y: np.empty((n, 0)),
                                    dtype=np.float)
        else:
            if A.shape[1] != n or H.shape[0] != n or H.shape[1] != n:
                raise ValueError('Shapes are inconsistent')
            m = A.shape[0]
            self.A = A

        super(QPModel, self).__init__(n=n, m=m, name=name, **kwargs)
        self.c = c
        self.H = H

        # Default classification of constraints
        self._lin = range(self.m)             # Linear    constraints
        self._nln = []                        # Nonlinear constraints
        self._net = []                        # Network   constraints
        self._nlin = len(self.lin)            # Number of linear constraints
        self._nnln = len(self.nln)            # Number of nonlinear constraints
        self._nnet = len(self.net)            # Number of network constraints

    def obj(self, x):
        cHx = self.hprod(x, 0, x)
        cHx *= 0.5
        cHx += self.c
        return np.dot(cHx, x)

    def grad(self, x):
        Hx = self.hprod(x, 0, x)
        Hx += self.c
        return Hx

    def cons(self, x):
        if isinstance(self.A, np.ndarray):
          return np.dot(self.A, x)
        return self.A * x

    def A(self, x):
        return self.A

    def jac(self, x):
        return self.A

    def jprod(self, x, p):
        return self.cons(p)

    def jtprod(self, x, p):
        if isinstance(self.A, np.ndarray):
          return np.dot(self.A.T, p)
        return self.A.T * p

    def hess(self, x, z):
        return self.H

    def hprod(self, x, z, p):
        if isinstance(self.H, np.ndarray):
          return np.dot(self.H, p)
        return self.H * p


class LPModel(QPModel):
    """
    A generic class to represent a linear programming problem

    minimize    c'x
    subject to  L <= A*x <= U
                l <=  x  <= u.
    """

    def __init__(self, c, A=None, name='GenericLP', **kwargs):
        """
        :parameters:
            :c:   Numpy array to represent the linear objective
            :A:   linear operator to represent the constraint matrix.
                  It must be possible to perform the operations `A*x`
                  and `A.T*y` for Numpy arrays `x` and `y` of appropriate size.

        See the documentation of `NLPModel` for futher information.
        """

        n = c.shape[0]
        H = LinearOperator(n, n,
                           lambda x: np.zeros(n),
                           symmetric=True,
                           dtype=np.float)
        super(LPModel, self).__init__(c, H, A, name=name, **kwargs)

    def obj(self, x):
        return np.dot(self.c, x)

    def grad(self, x):
        return self.c

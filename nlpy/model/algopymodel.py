from nlpy.model import NLPModel
import algopy
import numpy as np


class AlgopyModel(NLPModel):
  """
  A class to represent optimization problems in which derivatives
  are computed via algorithmic differentiation through Algopy.
  Algopy only supplies dense derivatives.
  See the documentation of `NLPModel` for further information.
  """

  def __init__(self, n=0, m=0, name='Algopy-Generic', **kwargs):
    super(AlgopyModel, self).__init__(n, m, name, **kwargs)

    self._cg_obj = None
    self._cg_cons = None
    self._cg_lag = None

    self._trace_obj(self.x0)
    self._trace_lag(self.x0, self.pi0)
    if m > 0:
      self._trace_cons(self.x0)

  @property
  def cg_obj(self):
    "Objective function call graph."
    return self._cg_obj

  @property
  def cg_cons(self):
    "Constraints call graph."
    return self._cg_cons

  @property
  def cg_lag(self):
    "Lagrangian call graph."
    return self._cg_lag

  def _trace_obj(self, x):
    "Trace the objective function evaluation."

    if self._cg_obj is not None: return
    cg = algopy.CGraph()
    x = algopy.Function(x)
    y = self.obj(x)
    cg.trace_off()
    cg.independentFunctionList = [x]
    cg.dependentFunctionList = [y]
    self._cg_obj = cg

  def _trace_cons(self, x):
    "Trace the constraints evaluation."

    if self._cg_cons is not None or self.m == 0: return
    cg = algopy.CGraph()
    x = algopy.Function(x)
    y = self.cons(x)
    cg.trace_off()
    cg.independentFunctionList = [x]
    cg.dependentFunctionList = [y]
    self._cg_cons = cg

  def _trace_lag(self, x, z):
    "Trace the Lagrangian evaluation."

    if self._cg_lag is not None: return
    self._trace_obj(x)
    self._trace_cons(x)
    unconstrained = self.m == 0 and self.nbounds == 0

    if unconstrained:
      self._cg_lag = self._cg_obj
      return

    cg = algopy.CGraph()
    xz = np.concatenate((x, z))
    xz = algopy.Function(xz)
    l = self.lag(xz[:self.nvar], xz[self.nvar:])
    cg.independentFunctionList = [xz]
    cg.dependentFunctionList = [l]
    self._cg_lag = cg

  def grad(self, x, **kwargs):
    "Evaluate the objective gradient at x."
    return self._cg_obj.gradient(x)

  # Override lag because Algopy won't apply numpy.dot() between a
  # Numpy array and an array of Functions.
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
      l -= algopy.dot(z[:m+nrC], self.cons(x))
    if self.nbounds > 0:
      l -= algopy.dot(z[m+nrC:], self.bounds(x))
    return l

  def hess(self, x, z, **kwargs):
    "Return the Hessian of the objective at x."
    xz = np.concatenate((x, z))
    return self._cg_lag.hessian(xz)[:self.nvar, :self.nvar]

  def hprod(self, x, z, v, **kwargs):
    "Return the Hessian-vector product at x with v."
    xz = np.concatenate((x, z))
    v0 = np.concatenate((v, np.zeros(self.ncon)))
    return self._cg_lag.hess_vec(xz, v0)[:self.nvar]

  def cons_pos(self, x):
    """
    Because AlgoPy does not support fancy indexing, it is necessary
    to formulate constraints in the form

        ci(x)  = 0  for i in equalC
        ci(x) >= 0  for i in lowerC.
    """
    return self.cons(x)

  def jac(self, x, **kwargs):
    "Return constraints Jacobian at x."
    return self._cg_cons.jacobian(x)

  def jac_pos(self, x, **kwargs):
    "Return reformulated constraints Jacobian at x."
    return self.jac(x, **kwargs)

  def jprod(self, x, v, **kwargs):
    "Return the Jacobian-vector product at x with v."
    return self._cg_cons.jac_vec(x, v)

  def jtprod(self, x, v, **kwargs):
    "Return the transpose-Jacobian-vector product at x with v."
    return self._cg_cons.vec_jac(v, x)

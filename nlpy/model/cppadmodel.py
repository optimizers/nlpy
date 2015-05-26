from nlpy.model import NLPModel
import pycppad
import numpy as np


class CppADModel(NLPModel):
    """
    A class to represent optimization problems in which derivatives
    are computed via algorithmic differentiation through CPPAD.
    See the documentation of `NLPModel` for further information.
    """

    def __init__(self, n=0, m=0, name='CppAD-Generic', **kwargs):
        super(CppADModel, self).__init__(n, m, name, **kwargs)

        self._cppad_adfun_obj = None
        self._cppad_adfun_cons = None
        self._cppad_adfun_cons_pos = None
        self._cppad_adfun_lag = None

        # Trace objective and constraint functions.
        self._trace_obj(self.x0)
        self._trace_lag(self.x0, self.pi0)
        if self.m > 0:
          self._trace_cons(self.x0)
          self._trace_cons_pos(self.x0)

    def _trace_obj(self, x):
        ax = pycppad.independent(x)
        ay = self.obj(ax)
        self._cppad_adfun_obj = pycppad.adfun(ax, np.array([ay]))

    def _trace_cons(self, x):
        ax = pycppad.independent(x)
        ay = self.cons(ax)

        if not isinstance(ay, np.ndarray):
            ay = np.array([ay])

        self._cppad_adfun_cons = pycppad.adfun(ax, ay)

    def _trace_cons_pos(self, x):
        ax = pycppad.independent(x)
        ay = self.cons_pos(ax)

        if not isinstance(ay, np.ndarray):
            ay = np.array([ay])

        self._cppad_adfun_cons_pos = pycppad.adfun(ax, ay)

    def _trace_lag(self, x, z):
        if self.m == 0 and self.nbounds == 0:
          self._cppad_adfun_lag = self._cppad_adfun_obj
          return
        axz = pycppad.independent(np.concatenate((x, z)))
        ax = axz[:self.nvar]
        az = axz[self.nvar:]
        ay = self.lag(ax, az)
        self._cppad_adfun_lag = pycppad.adfun(axz, np.array([ay]))

    def _cppad_obj(self, x):
        "Return the objective function from the CppAD tape."
        return self._cppad_adfun_obj.function(x)

    def grad(self, x, **kwargs):
        "Return the objective gradient at x."
        self._cppad_adfun_obj.forward(0, x)
        return self._cppad_adfun_obj.reverse(1, np.array([1.]))

    def hess(self, x, z, **kwargs):
        "Return the Hessian of the Lagrangian at (x,z)."
        xz = np.concatenate((x, z))
        H = self._cppad_adfun_lag.hessian(xz, np.array([1.]))
        return H[:self.nvar, :self.nvar]

    def hprod(self, x, z, v, **kwargs):
        "Return the Hessian-vector product at x with v."
        # forward: order zero (computes function value)
        xz = np.concatenate((x, z))
        v0 = np.concatenate((v, np.zeros(self.ncon)))
        self._cppad_adfun_lag.forward(0, xz)

        # forward: order one (computes directional derivative)
        self._cppad_adfun_lag.forward(1, v0)

        # reverse: order one (computes gradient of directional derivative)
        return self._cppad_adfun_lag.reverse(2, np.array([1.]))[:self.nvar]

    def _cppad_cons(self, x, **kwargs):
        "Return the constraints from the CppAD tape."
        return self._cppad_adfun_cons.function(x)

    def jac(self, x, **kwargs):
        "Return constraints Jacobian at x."
        return self._cppad_adfun_cons.jacobian(x)

    def jprod(self, x, v, **kwargs):
        "Return the product of v with the Jacobian at x."
        # forward: order zero (computes function value)
        self._cppad_adfun_cons.forward(0, x)

        # forward: order one (computes directional derivative)
        return self._cppad_adfun_cons.forward(1, v)

    def jtprod(self, x, v, **kwargs):
        "Return the product of v with the transpose Jacobian at x."
        # forward: order zero (computes function value)
        self._cppad_adfun_cons.forward(0, x)

        # forward: order one (computes directional derivative)
        return self._cppad_adfun_cons.reverse(1, v)

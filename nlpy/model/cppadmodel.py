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

        # Trace objective and constraint functions.
        self._trace_obj(self.x0)
        if self.m > 0: self._trace_cons(self.x0)

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

    def _cppad_obj(self, x):
        "Return the objective function from the CppAD tape."
        return self._cppad_adfun_obj.function(x)

    def grad(self, x, **kwargs):
        "Return the objective gradient at x."
        return self._cppad_grad(x, **kwargs)

    def _cppad_grad(self, x, **kwargs):
        "Return the objective gradient from the CppAD tape."
        self._cppad_adfun_obj.forward(0, x)
        return self._cppad_adfun_obj.reverse(1, np.array([1.]))

    def hess(self, x, z, **kwargs):
        "Return the Hessian of the objective at x."
        return self._cppad_hess(x, z, **kwargs)

    def _cppad_hess(self, x, z, **kwargs):
        "Return the objective hessian from the CppAD tape."
        return self._cppad_adfun_obj.hessian(x, np.array([1.]))

    def hess_vec(self, x, z, v, **kwargs):
        "Return the Hessian-vector product at x with v."
        return self._cppad_hess_vec(x, z, v, **kwargs)

    def _cppad_hess_vec(self, x, z, v, **kwargs):
        "Return the objective hessian vector product from the CppAD tape."

        # forward: order zero (computes function value)
        self._cppad_adfun_obj.forward(0, x)

        # forward: order one (computes directional derivative)
        self._cppad_adfun_obj.forward(1, v)

        # reverse: order one (computes gradient of directional derivative)
        return self._cppad_adfun_obj.reverse(2, np.array([1.]))

    def _cppad_cons(self, x, **kwargs):
        "Return the constraints from the CppAD tape."
        return self._cppad_adfun_cons.function(x)

    def jac(self, x, **kwargs):
        "Return constraints Jacobian at x."
        return self._cppad_jac(x, **kwargs)

    def _cppad_jac(self, x, **kwargs):
        "Return the constraints Jacobian from the CppAD tape."
        return self._cppad_adfun_cons.jacobian(x)

    def jprod(self, x, v, **kwargs):
        "Return the product of v with the Jacobian at x."
        return self._cppad_jac_vec(x, v, **kwargs)

    def _cppad_jac_vec(self, x, v, **kwargs):
        "Return the product of v with the Jacobian at x from CppAD tape."

        # forward: order zero (computes function value)
        self._cppad_adfun_cons.forward(0, x)

        # forward: order one (computes directional derivative)
        return self._cppad_adfun_cons.forward(1, v)

    def jtprod(self, x, v, **kwargs):
        "Return the product of v with the transpose Jacobian at x."
        return self._cppad_vec_jac(x, v, **kwargs)

    def _cppad_vec_jac(self, x, v, **kwargs):
        """
        Return the product of v with the transpose Jacobian at x
        from CppAD tape.
        """
        # forward: order zero (computes function value)
        self._cppad_adfun_cons.forward(0, x)

        # forward: order one (computes directional derivative)
        return self._cppad_adfun_cons.reverse(1, v)

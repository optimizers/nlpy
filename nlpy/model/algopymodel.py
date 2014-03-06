from nlpy.model import NLPModel
import algopy


class AlgopyModel(NLPModel):
    """
    A class to represent optimization problems in which derivatives
    are computed via algorithmic differentiation through Algopy.
    See the documentation of `NLPModel` for further information.
    """

    def __init__(self, n=0, m=0, name='Algopy-Generic', **kwargs):
        super(AlgopyModel, self).__init__(n, m, name, **kwargs)

        try:
            self._trace_obj(self.x0)
        except:
            pass

        if m > 0:
          try:
              self._trace_cons(self.x0)
          except:
              pass

    def _trace_obj(self, x):
        "Trace the objective function evaluation."

        cg = algopy.CGraph()
        x = algopy.Function(x)
        y = self.obj(x)
        cg.trace_off()
        cg.independentFunctionList = [x]
        cg.dependentFunctionList = [y]
        self._cg_obj = cg

    def _trace_cons(self, x):
        "Trace the constraint function evaluation."

        cg = algopy.CGraph()
        x = algopy.Function(x)
        y = self.cons(x)
        cg.trace_off()
        cg.independentFunctionList = [x]
        cg.dependentFunctionList = [y]
        self._cg_cons = cg

    def grad(self, x, **kwargs):
        "Evaluate the objective gradient at x."
        return self._cg_obj.gradient(x)

    def hess(self, x, z, **kwargs):
        "Return the Hessian of the objective at x."
        return self.dense_hess(x, z, **kwargs)

    def dense_hess(self, x, z, **kwargs):
        "Return the Hessian of the objective at x in dense format."
        return self._cg_obj.hessian(x)

    def hess_vec(self, x, z, v, **kwargs):
        "Return the Hessian-vector product at x with v."
        return self._cg_obj.hess_vec([x], [v])

    def jac(self, x, **kwargs):
        "Return constraints Jacobian at x."
        return self.dense_jac(x, **kwargs)

    def dense_jac(self, x, **kwargs):
        "Return constraints Jacobian at x in dense format."
        return self._cg_cons.jacobian(x)

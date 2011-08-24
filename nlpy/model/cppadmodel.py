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
        NLPModel.__init__(self, n, m, name, **kwargs)

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
        return self._cppad_adfun_obj.reverse(1, np.array( [ 1. ] ))


    def hess(self, x, z, **kwargs):
        "Return the Hessian of the objective at x."
        return self._cppad_hess(x, z, **kwargs)


    def _cppad_hess(self, x, z, **kwargs):
        "Return the objective hessian from the CppAD tape."
        return self._cppad_adfun_obj.hessian(x, np.array( [ 1. ] ))


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


    def jac_vec(self, x, v, **kwargs):
        "Return the product of v with the Jacobian at x."
        return self._cppad_jac_vec(x, v, **kwargs)


    def _cppad_jac_vec(self, x, v, **kwargs):
        "Return the product of v with the Jacobian at x from CppAD tape."

        # forward: order zero (computes function value)
        self._cppad_adfun_cons.forward(0, x)

        # forward: order one (computes directional derivative)
        return self._cppad_adfun_cons.forward(1, v)


    def vec_jac(self, x, v, **kwargs):
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


    def get_jac_linop(self, x, **kwargs):
        "Return the Jacobian at x as a linear operator."
        J = SimpleLinearOperator(self.n, self.m,
                                 lambda v: self.jac_vec(x,v),
                                 matvec_transp=lambda v: self.vec_jac(x,v),
                                 symmetric=False)
        return J


if __name__ == '__main__':

    from nlpy.optimize.solvers.lbfgs import LBFGSFramework
    from nlpy.optimize.solvers.ldfp  import LDFPTrunkFramework
    from nlpy.optimize.solvers.trunk import TrunkFramework
    from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
    from nlpy.optimize.tr.trustregion import TrustRegionCG as TRSolver
    import nlpy.tools.logs
    import logging, sys

    # Define a few problems.

    class Rosenbrock(CppADModel):

        def obj(self, x, **kwargs):
            return np.sum( 100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2 )


    class HS7(CppADModel):

        def obj(self, x, **kwargs):
            return np.log(1 + x[0]**2) - x[1]

        def cons(self, x, **kwargs):
            return (1 + x[0]**2)**2 + x[1]**2 - 4


    nvar = 2
    rosenbrock = Rosenbrock(n=nvar, name='Rosenbrock', x0=-np.ones(nvar))
    #nlp = rosenbrock
    hs7 = HS7(n=2, m=1, name='HS7', x0=2*np.ones(2))
    nlp = hs7

    v = np.ones(nvar)
    w = np.array([-2.])

    g = nlp.grad(nlp.x0)
    H = nlp.hess(nlp.x0, nlp.x0)
    Hv = nlp.hess_vec(nlp.x0, nlp.x0, v)
    #H_sparse = nlp.sparse_hess(nlp.x0, nlp.x0)

    print 'number of variables: ', nlp.n
    print 'initial guess: ', nlp.x0
    print 'f(x0) = ', nlp.obj(nlp.x0)
    print 'g(x0) = ', g
    print 'H(x0) = ', H
    print 'H(x0)*v = ', Hv
    # #print 'H_sparse(x0) = ', H_sparse
    if nlp.m > 0:
        c = nlp.cons(nlp.x0)
        J = nlp.jac(nlp.x0)
        print 'c(x0) = ', c
        print 'J(x0) = ', J
        print 'J(x0) * v = ', nlp.jac_vec(nlp.x0, v)
        print 'J(x0).T * [-2] = ', nlp.vec_jac(nlp.x0, w)

#     # Solve with linesearch-based L-BFGS method.
#     lbfgs = LBFGSFramework(nlp, npairs=5, scaling=True, silent=False)
#     lbfgs.solve()

#     # Create root logger.
#     log = logging.getLogger('adolcmodel')
#     log.setLevel(logging.INFO)
#     fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
#     hndlr = logging.StreamHandler(sys.stdout)
#     hndlr.setFormatter(fmt)
#     log.addHandler(hndlr)

#     # Configure the subproblem solver logger
#     nlpy.tools.logs.config_logger('adolcmodel.ldfp',
#                                   filemode='w',
#                                   stream=sys.stdout)

#     tr = TR(Delta=1.0, eta1=0.05, eta2=0.9, gamma1=0.25, gamma2=2.5)

#     # Solve with trust-region-based L-DFP method.
# #    ldfp = LDFPTrunkFramework(nlp, tr, TRSolver,
# #                              ny=True, monotone=False,
# #                              logger_name='adolcmodel.ldfp')
# #    ldfp.TR.Delta = 0.1 * np.linalg.norm(g)         # Reset initial trust-region radius
# #    ldfp.Solve()

#     # Solve with trust-region-based method.
#     trnk = TrunkFramework(nlp, tr, TRSolver,
#                           ny=True, monotone=False,
#                           logger_name='adolcmodel.ldfp')
#     trnk.TR.Delta = 0.1 * np.linalg.norm(g)         # Reset initial trust-region radius
#     trnk.Solve()


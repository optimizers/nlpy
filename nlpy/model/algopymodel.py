from nlpy.model import NLPModel
import algopy
import numpy as np

class AlgopyModel(NLPModel):

    def __init__(self, n=0, m=0, name='Generic', **kwargs):
        NLPModel.__init__(self, n, m, name, **kwargs)

        try:
            self._trace_obj(self.x0)
        except:
            pass

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
        return self._cg_obj.gradient([x])[0]


    def hess(self, x, z, **kwargs):
        "Return the Hessian of the objective at x."
        return self.dense_hess(x,z,**kwargs)


    def dense_hess(self, x, z, **kwargs):
        "Return the Hessian of the objective at x in dense format."
        return self._cg_obj.hessian([x])[0]


    def hess_vec(self, x, z, v, **kwargs):
        "Return the Hessian-vector product at x with v."
        return self._cg_obj.hess_vec([x], [v])


    def jac(self, x, **kwargs):
        "Return constraints Jacobian at x."
        return self.dense_jac(x,**kwargs)


    def dense_jac(self, x, **kwargs):
        "Return constraints Jacobian at x in dense format."
        return self._cg_cons.jacobian([x])[0]




# if __name__ == '__main__':

#     from nlpy.optimize.solvers.lbfgs import LBFGSFramework
#     from nlpy.optimize.solvers.ldfp  import LDFPTrunkFramework
#     from nlpy.optimize.solvers.trunk import TrunkFramework
#     from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
#     from nlpy.optimize.tr.trustregion import TrustRegionCG as TRSolver
#     import nlpy.tools.logs
#     import logging, sys

#     class AlgopyRosenbrock(AlgopyModel):

#         def obj(self, x, **kwargs):
#             return np.sum( 100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2 )


#     nvar = 10
#     nlp = AlgopyRosenbrock(n=nvar, name='Rosenbrock', x0=-np.ones(nvar))


#     class hs7(AlgopyModel):

#         def obj(self, x, **kwargs):
#             return np.log(1 + x[0]**2) - x[1]

#         def cons(self, x, **kwargs):
#             return (1 + x[0]**2)**2 + x[1]**2 - 4

#     g = nlp.grad(nlp.x0)
#     # print 'initial guess: ', nlp.x0
#     # print 'number of variables: ', nlp.n
#     # print 'f(x0) = ', nlp.obj(nlp.x0)
#     # print 'g(x0) = ', g

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


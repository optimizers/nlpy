from nlpy.model import NLPModel
import adolc
import numpy as np

has_colpack = False

class AdolcModel(NLPModel):

    # Count the number of instances of this class to generate
    # non-conflicting tape ids. Must be a mutable type.
    __NUM_INSTANCES = [-1]

    def __init__(self, n=0, m=0, name='Generic', **kwargs):
        NLPModel.__init__(self, n, m, name, **kwargs)
        self.__NUM_INSTANCES[0] += 1

        # Trace objective and constraint functions.
        self._obj_trace_id = None
        self._trace_obj(self.x0)

        self._con_trace_id = None
        if self.m > 0: self._trace_con(self.x0)

        self.first_sparse_hess_eval = True
        self.first_sparse_jac_eval  = True


    def _get_trace_id(self):
        "Return an available trace id."
        return 100*self.__NUM_INSTANCES[0]


    def get_obj_trace_id(self):
        "Return the trace id for the objective function."
        return self._obj_trace_id


    def get_con_trace_id(self):
        "Return the trace id for the constraints."
        return self._con_trace_id


    def _trace_obj(self, x):

        if self._obj_trace_id is None:

            #print 'Tracing objective...'
            self._obj_trace_id = self._get_trace_id()
            adolc.trace_on(self._obj_trace_id)
            x = adolc.adouble(x)
            adolc.independent(x)
            y = self.obj(x)
            adolc.dependent(y)
            adolc.trace_off()


    def _trace_con(self, x):

        if self._con_trace_id is None and self.m > 0:

            #print 'Tracing constraints...'
            self._con_trace_id = self._get_trace_id() + 1
            adolc.trace_on(self._con_trace_id)
            x = adolc.adouble(x)
            adolc.independent(x)
            y = self.cons(x)
            adolc.dependent(y)
            adolc.trace_off()


    def _adolc_obj(self, x):
        "Evaluate the objective function from the ADOL-C tape."
        return adolc.function(self._obj_trace_id, x)


    def grad(self, x, **kwargs):
        "Evaluate the objective gradient at x."
        return self._adolc_grad(x, **kwargs)


    def _adolc_grad(self, x, **kwargs):
        "Evaluate the objective gradient from the ADOL-C tape."
        return adolc.gradient(self._obj_trace_id, x)


    def hess(self, x, z, **kwargs):
        "Return the Hessian of the objective at x."
        if has_colpack:
            return self.sparse_hess(x, z, **kwargs)
        return self.dense_hess(x, z, **kwargs)


    def dense_hess(self, x, z, **kwargs):
        "Return the Hessian of the objective at x in dense format."
        return adolc.hessian(self._obj_trace_id, x)


    def hprod(self, x, z, v, **kwargs):
        "Return the Hessian-vector product at x with v."
        return adolc.hess_vec(self._obj_trace_id, x, v)


    def sparse_hess(self, x, z, **kwargs):
        "Return the Hessian of the objective at x in sparse format."
        options = np.zeros(2,dtype=int)
        if self.first_sparse_hess_eval:
            nnz, rind, cind, values =  \
                    adolc.colpack.sparse_hess_no_repeat(self._obj_trace_id,
                                                        x,
                                                        options=options)
            self.nnzH  = nnz
            self.hess_rind = rind
            self.hess_cind = cind
            self.hess_values = values
            self.first_sparse_hess_eval = False
            return rind, cind, values

        else:
            return adolc.colpack.sparse_hess_repeat(self._obj_trace_id,
                                                    x,
                                                    self.hess_rind,
                                                    self.hess_cind,
                                                    self.hess_values)


    def _adolc_cons(self, x, **kwargs):
        "Evaluate the constraints from the ADOL-C tape."
        return adolc.function(self._con_trace_id, x)


    def jac(self, x, **kwargs):
        "Return constraints Jacobian at x."
        if has_colpack:
            return self.sparse_jac(x, **kwargs)
        return self.dense_jac(x, **kwargs)


    def dense_jac(self, x, **kwargs):
        "Return constraints Jacobian at x in dense format."
        return self._adolc_jac(x, **kwargs)


    def _adolc_jac(self, x, **kwargs):
        "Evaluate the constraints Jacobian from the ADOL-C tape."
        return adolc.jacobian(self._con_trace_id, x)


    def sparse_jac(self, x, **kwargs):
        "Return constraints Jacobian at x in sparse format."
        [nnz, rind, cind, values] =sparse_jac_no_repeat(tape_tag, x, options)
        options = np.zeros(4,dtype=int)
        if self.first_sparse_jac_eval:
            nnz, rind, cind, values =  \
                    adolc.colpack.sparse_jac_no_repeat(self._con_trace_id,
                                                       x,
                                                       options=options)
            self.nnzJ  = nnz
            self.jac_rind = rind
            self.jac_cind = cind
            self.jac_values = values
            self.first_sparse_jac_eval = False
            return rind, cind, values

        else:
            return adolc.colpack.sparse_jac_repeat(self._jac_trace_id,
                                                   x,
                                                   self.jac_rind,
                                                   self.jac_cind,
                                                   self.jac_values)



    def jac_vec(self, x, v, **kwargs):
        "Return the product of v with the Jacobian at x."
        return adolc.jac_vec(self._con_trace_id, x, v)


    def vec_jac(self, x, v, **kwargs):
        "Return the product of v with the transpose Jacobian at x."
        return adolc.vec_jac(self._con_trace_id, x, v)



if __name__ == '__main__':

    from nlpy.optimize.solvers.lbfgs import LBFGSFramework
    from nlpy.optimize.solvers.ldfp  import LDFPTrunkFramework
    from nlpy.optimize.solvers.trunk import TrunkFramework
    from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
    from nlpy.optimize.tr.trustregion import TrustRegionCG as TRSolver
    import nlpy.tools.logs
    import logging, sys

    # Define a few problems.

    class AdolcRosenbrock(AdolcModel):

        def obj(self, x, **kwargs):
            return np.sum( 100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2 )


    class AdolcHS7(AdolcModel):

        def obj(self, x, **kwargs):
            return np.log(1 + x[0]**2) - x[1]

        def cons(self, x, **kwargs):
            return (1 + x[0]**2)**2 + x[1]**2 - 4


    nvar = 5
    rosenbrock = AdolcRosenbrock(n=nvar, name='Rosenbrock', x0=-np.ones(nvar))
    hs7 = AdolcHS7(n=2, m=1, name='HS7', x0=2*np.ones(2))

    nlp = hs7

    g = nlp.grad(nlp.x0)
    H = nlp.hess(nlp.x0, nlp.x0)
    #H_sparse = nlp.sparse_hess(nlp.x0, nlp.x0)
    print 'number of variables: ', nlp.n
    print 'initial guess: ', nlp.x0
    print 'f(x0) = ', nlp.obj(nlp.x0)
    print 'g(x0) = ', g
    print 'H(x0) = ', H
    #print 'H_sparse(x0) = ', H_sparse
    if nlp.m > 0 :
        print 'number of constraints: ', nlp.m
        c = nlp.cons(nlp.x0)
        J = nlp.jac(nlp.x0)
        v = np.array([-1.,-1.])
        w = np.array([2])
        print 'c(x0) = ', c
        print 'J(x0) = ', J
        print 'J(x0) * [-1,1] = ', nlp.jac_vec(nlp.x0, v)
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


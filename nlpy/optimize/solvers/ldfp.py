"""
A limited-memory DFP method for unconstrained minimization. A symmetric and
positive definite approximation of the Hessian matrix is built and updated at
each iteration following the Davidon-Fletcher-Powell formula. For efficiency,
only the recent observed curvature is incorporated into the approximation,
resulting in a *limited-memory* scheme.

The main idea of this method is that the DFP formula is dual to the BFGS
formula. Therefore, by swapping s and y in the (s,y) pairs, the InverseLBFGS
class updates a limited-memory DFP approximation to the Hessian, rather than
a limited-memory BFGS approximation to its inverse.
"""

from nlpy.model.amplpy import AmplModel
from nlpy.optimize.solvers.lbfgs import InverseLBFGS
from nlpy.optimize.solvers.trunk import TrunkFramework
import numpy

__docformat__ = 'restructuredtext'

# Subclass InverseLBFGS to update a LDFP approximation to the Hessian
# (as opposed to a LBFGS approximation to its inverse).
class LDFP(InverseLBFGS):

    def __init__(self, n, npairs=5, **kwargs):
        InverseLBFGS.__init__(self, n, npairs, **kwargs)

    def store(self, iter, new_s, new_y):
        # Simply swap s and y.
        InverseLBFGS.store(self, iter, new_y, new_s)


# Subclass AmplModel to define a problem in which the Hessian matrix
# is a limited-memory DFP approximation.
class LDFPModel(AmplModel):

    def __init__(self, model, **kwargs):
        AmplModel.__init__(self, model, **kwargs)
        self.ldfp = LDFP(self.n, **kwargs)

    def hess(self, x, z):
        raise NotImplementedError, 'Only matrix-vector products are available.'

    def hprod(self, z, v, **kwargs):
        """
        Compute the matrix-vector product between the limited-memory DFP
        approximation kept in storage and the vector `v`. The argument `z`
        is ignored and is only present for compatibility with the original
        `hprod`.
        """
        self.Hprod += 1
        iter = kwargs.get('iter', None)
        if iter is None:
            raise ValueError, 'Please specify iteration number.'
        Hv = self.ldfp.matvec(iter, v)
        return Hv

# Subclass solver TRUNK so the LDFP matrix update is performed at the
# end of each iteration.
class LDFPTrunkFramework(TrunkFramework):
    
    def __init__(self, nlp, TR, TrSolver, **kwargs):
        TrunkFramework.__init__(self, nlp, TR, TrSolver, **kwargs)
        self.save_g = True

    def PostIteration(self, **kwargs):
        """
        This method updates the limited-memory DFP approximation by appending
        the most recent (s,y) pair to it and possibly discarding the oldest one
        if all the memory has been used.
        """
        if self.status != 'Rej':
            s = self.alpha * self.solver.step
            y = self.g - self.g_old
            self.nlp.ldfp.store(self.iter, s, y)
        return None


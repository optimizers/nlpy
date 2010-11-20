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
    """
    A limited-memory DFP framework for quasi-Newton methods. See the
    documentation of `InverseLBFGS`.
    """

    def __init__(self, n, npairs=5, **kwargs):
        InverseLBFGS.__init__(self, n, npairs, **kwargs)

    def store(self, new_s, new_y):
        # Simply swap s and y.
        InverseLBFGS.store(self, new_y, new_s)


# Subclass solver TRUNK so the LDFP matrix update is performed at the
# end of each iteration.
class LDFPTrunkFramework(TrunkFramework):

    def __init__(self, nlp, TR, TrSolver, **kwargs):
        TrunkFramework.__init__(self, nlp, TR, TrSolver, **kwargs)
        self.ldfp = LDFP(self.nlp.n, **kwargs)
        self.save_g = True

    def hprod(self, v, **kwargs):
        """
        Compute the matrix-vector product between the limited-memory DFP
        approximation kept in storage and the vector `v`.
        """
        return self.ldfp.matvec(v)

    def PostIteration(self, **kwargs):
        """
        This method updates the limited-memory DFP approximation by appending
        the most recent (s,y) pair to it and possibly discarding the oldest one
        if all the memory has been used.
        """
        if self.status != 'Rej':
            s = self.alpha * self.solver.step
            y = self.g - self.g_old
            self.nlp.ldfp.store(s, y)
        return None


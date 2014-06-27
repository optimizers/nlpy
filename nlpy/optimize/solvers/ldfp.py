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

from nlpy.optimize.solvers.lbfgs import InverseLBFGS
from nlpy.optimize.solvers.trunk import Trunk

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


class StructuredLDFP(InverseLBFGS):
    """
    A limited-memory DFP framework for quasi-Newton methods that only
    memorizes updates corresponding to certain variables. This is useful
    when approximating the Hessian of a constraint with a sparse Jacobian.
    """

    def __init__(self, n, npairs=5, **kwargs):
        """
        See the documentation of `InverseLBFGS` for complete information.

        :keywords:
            :vars: List of variables participating in the quasi-Newton
                   update. If `None`, all variables participate.
        """
        self.on = n   # Original value of n.
        self.vars = kwargs.get('vars', None)  # None means all variables.
        if self.vars is None:
            nvars = n
        else:
            nvars = len(self.vars)

        # This next initialization will set self.n to nvars.
        # The original value of n was saved in self.on.
        InverseLBFGS.__init__(self, nvars, npairs, **kwargs)

    def store(self, new_s, new_y):
        """
        Store a new (s,y) pair. This method takes "small" vectors as
        input, i.e., corresponding to the variables participating in
        the quasi-Newton update.
        """
        InverseLBFGS.store(self, new_y, new_s)

    def matvec(self, v):
        """
        Take a small vector and return a small vector giving the
        contribution of the Hessian approximation to the
        matrix-vector product.
        """
        return InverseLBFGS.matvec(self, v)


# Subclass solver TRUNK to maintain an LDFP approximation to the Hessian and
# perform the LDFP matrix update at the end of each iteration.

class LDFPTrunk(Trunk):

    def __init__(self, nlp, TR, TrSolver, **kwargs):
        super(Trunk, self).__init__(self, nlp, TR, TrSolver, **kwargs)
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
        if self.step_status != 'Rej':
            s = self.alpha * self.solver.step
            y = self.g - self.g_old
            self.ldfp.store(s, y)
        return None

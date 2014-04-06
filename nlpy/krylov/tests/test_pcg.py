# Tests relative to the truncated conjugate gradient method.
from numpy.testing import *
import unittest
from nlpy.model import QPModel
from nlpy.krylov import TruncatedCG
from pykrylov.linop import LinearOperator
import numpy as np


class Test_PCG(unittest.TestCase):

    def setUp(self):
        self.n = 10
        diag = np.random.random(self.n) + 2
        sub  = np.random.random(self.n - 1)

        def hprod(x):
            Hx = diag*x
            Hx[1:]  += sub*x[1:]
            Hx[:-1] += sub*x[:-1]
            return Hx

        self.H = LinearOperator(self.n, self.n, hprod, symmetric=True)

    def test_noregion(self):
        # Convex model.

        expected_x = np.ones(self.n)
        c = -(self.H * expected_x)
        qp = QPModel(c, self.H, A=None)
        trcg = TruncatedCG(qp)
        trcg.solve(radius=None)
        assert(np.allclose(trcg.step, expected_x))

    def test_convex_interior(self):
        # Convex model with an interior solution.

        expected_x = np.ones(self.n)
        c = -(self.H * expected_x)
        qp = QPModel(c, self.H)
        trcg = TruncatedCG(qp)
        trcg.solve(radius=1.2 * np.linalg.norm(expected_x))
        assert(np.allclose(trcg.step, expected_x))

    def test_convex_boundary(self):
        # Create a convex model with a solution on the boundary.
        # The solution coincides with the unconstrained minimizer

        expected_x = np.ones(self.n)
        c = -(self.H * expected_x)
        qp = QPModel(c, self.H)
        trcg = TruncatedCG(qp)
        trcg.solve(radius=np.linalg.norm(expected_x))
        assert(np.allclose(trcg.step, expected_x))

    def test_convex_boundary2(self):
        # Create a convex model with a solution on the boundary.
        # The unconstrained minimizer lies outside the trust region.
        # This test is constructed from the necessary and sufficient
        # optimality conditions for the trust-region subproblem.

        expected_x = np.ones(self.n)
        c = -(self.H * expected_x + expected_x)  # lambda := 1.
        qp = QPModel(c, self.H)
        trcg = TruncatedCG(qp)
        trcg.solve(radius=np.linalg.norm(expected_x))

        # Here we can't demand too much accuracy since we could hit the border
        # close to the expected solution.
        assert(np.allclose(trcg.step, expected_x, atol=1.0e-1, rtol=1.0e-1))

    def test_nonconvex(self):
        # Create a nonconvex model with a solution on the boundary.
        # Example from the trust-region book, Section 7.3.1.2 page 179.

        diag = np.array([-2, -1, 0, 1], dtype=np.float)
        H = LinearOperator(4, 4, lambda x: diag*x, symmetric=True)
        c = np.ones(4)

        # -c is a direction of negative curvature.
        radius = 2
        expected_x = -c / np.linalg.norm(c) * radius
        qp = QPModel(c, H)
        trcg = TruncatedCG(qp)
        trcg.solve(radius=np.linalg.norm(expected_x))


if __name__ == '__main__':
    unittest.main()

from numpy.testing import *
import algopy
from nlpy.model.algopymodel import AlgopyModel

import numpy as np


class AlgopyRosenbrock(AlgopyModel):
    "The standard Rosenbrock function."

    def obj(self, x, **kwargs):
        return algopy.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class AlgopyHS7(AlgopyModel):
    "Problem #7 in the Hock and Schittkowski collection."

    def obj(self, x, **kwargs):
        return algopy.log(1 + x[0]**2) - x[1]

    def cons(self, x, **kwargs):
        return (1 + x[0]**2)**2 + x[1]**2 - 4


def get_derivatives(nlp):
    g = nlp.grad(nlp.x0)
    H = nlp.dense_hess(nlp.x0, nlp.x0)
    if nlp.m > 0:
        J = nlp.dense_jac(nlp.x0)
        return (g, H, J)
    else:
        return (g, H)


class Test_AlgopyRosenbrock(TestCase):

    def setUp(self):
        self.rosenbrock = AlgopyRosenbrock(n=5, name='Rosenbrock', x0=-np.ones(5))

    def test_gradient(self):
        (g, H) = get_derivatives(self.rosenbrock)
        expected_g = np.array([-804., -1204., -1204., -1204., -400.])
        expected_H = np.array([[1602.,  400.,    0.,    0.,   0.],
                               [ 400., 1802.,  400.,    0.,   0.],
                               [   0.,  400., 1802.,  400.,   0.],
                               [   0.,    0.,  400., 1802., 400.],
                               [   0.,    0.,    0.,  400., 200.]])
        assert(np.allclose(g, expected_g))
        assert(np.allclose(H, expected_H))


class Test_AlgopyHS7(TestCase):

    def setUp(self):
        self.hs7 = AlgopyHS7(n=2, m=1, name='HS7', x0=2*np.ones(2))

    def test_hs7(self):
        hs7 = self.hs7
        (g, H, J) = get_derivatives(hs7)
        expected_g = np.array([0.8, -1.])
        expected_H = np.array([[-0.24, 0.], [0., 0.]])
        expected_J = np.array([[40., 4.]])
        assert(np.allclose(g, expected_g))
        assert(np.allclose(H, expected_H))
        assert(np.allclose(J, expected_J))

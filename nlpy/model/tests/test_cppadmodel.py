# Tests relative to algorithmic differentiation with CppAD.
from numpy.testing import *
from nlpy.model import CppADModel
import numpy as np


class CppADRosenbrock(CppADModel):
    "The standard Rosenbrock function."

    def obj(self, x, **kwargs):
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class CppADHS7(CppADModel):
    "Problem #7 in the Hock and Schittkowski collection."

    def obj(self, x, **kwargs):
        return np.log(1 + x[0]**2) - x[1]

    def cons(self, x, **kwargs):
        return (1 + x[0]**2)**2 + x[1]**2 - 4


def get_derivatives(nlp):
    g = nlp.grad(nlp.x0)
    H = nlp.hess(nlp.x0, nlp.x0)
    if nlp.m > 0:
        J = nlp.jac(nlp.x0)
        v = -np.ones(nlp.n)
        w = 2*np.ones(nlp.m)
        Jv = nlp.jac_vec(nlp.x0, v)
        JTw = nlp.vec_jac(nlp.x0, w)
        return (g, H, J, Jv, JTw)
    else:
        return (g, H)


class Test_CppADRosenbrock(TestCase):

    def setUp(self):
        self.nlp = CppADRosenbrock(n=5, name='Rosenbrock', x0=-np.ones(5))

    def test_rosenbrock(self):
        (g, H) = get_derivatives(self.nlp)
        expected_g = np.array([-804., -1204., -1204., -1204., -400.])
        expected_H = np.array([[1602.,  400.,    0.,    0.,   0.],
                               [ 400., 1802.,  400.,    0.,   0.],
                               [   0.,  400., 1802.,  400.,   0.],
                               [   0.,    0.,  400., 1802., 400.],
                               [   0.,    0.,    0.,  400., 200.]])
        assert(np.allclose(g, expected_g))
        assert(np.allclose(H, expected_H))


class Test_CppADHS7(TestCase):

    def setUp(self):
        self.nlp = CppADHS7(n=2, m=1, name='HS7', x0=2*np.ones(2))

    def test_hs7(self):
        hs7 = self.nlp
        (g, H, J, Jv, JTw) = get_derivatives(hs7)
        expected_g = np.array([0.8, -1.])
        expected_H = np.array([[-0.24, 0.], [0., 0.]])
        expected_J = np.array([[40., 4.]])
        expected_Jv = np.array([-44.])
        expected_JTw = np.array([80., 8.])
        assert(np.allclose(g, expected_g))
        assert(np.allclose(H, expected_H))
        assert(np.allclose(J, expected_J))
        assert(np.allclose(Jv, expected_Jv))
        assert(np.allclose(JTw, expected_JTw))
        Jop = hs7.get_jac_linop(hs7.x0)
        Jopv = Jop * (-np.ones(hs7.n))
        JopTw = Jop.T * (2*np.ones(hs7.m))
        assert(np.allclose(Jopv, expected_Jv))
        assert(np.allclose(JopTw, expected_JTw))

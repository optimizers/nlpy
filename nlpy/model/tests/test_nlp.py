# Tests relative to pure Python models.
from numpy.testing import *
from nlpy.model import QPModel, LPModel
from pykrylov.linop import LinearOperator, linop_from_ndarray
import numpy as np


class Test_LPModel(TestCase):

  def setUp(self):
    self.n = n = 5; self.m = m = 3
    self.c = np.random.random(n)
    self.A = np.random.random((m, n))
    self.lp1 = LPModel(self.c,
                       A=self.A,
                       Lvar=-np.random.random(n),
                       Uvar=np.random.random(n),
                       Lcon=-2*np.random.random(n),
                       Ucon=2*np.random.random(n))

    self.lp2 = LPModel(self.c,
                       Lvar=-np.random.random(n),
                       Uvar=np.random.random(n),
                       Lcon=-2*np.random.random(n),
                       Ucon=2*np.random.random(n))

  def test_constructor(self):
    lp1 = self.lp1
    assert(isinstance(lp1, LPModel))
    assert(lp1.n == self.n)
    assert(lp1.m == self.m)
    assert(lp1.nlin == self.m)
    assert(lp1.nnln == 0)
    assert(lp1.nnet == 0)

    lp2 = self.lp2
    J = lp2.jac(lp2.x0)
    assert(isinstance(J, LinearOperator))
    assert(J.shape == (0, self.n))

  def test_lp(self):
    lp = self.lp1
    x = np.random.random(self.n)
    assert(np.allclose(lp.obj(x), np.dot(self.c, x)))
    assert(lp.grad(x) is self.c)

    p = np.random.random(self.n)
    assert(np.allclose(lp.jprod(x, p), np.dot(self.A, p)))
    q = np.random.random(self.m)
    assert(np.allclose(lp.jtprod(x, q), np.dot(self.A.T, q)))

    H = lp.hess(x, 0)
    assert(H.symmetric is True)
    assert(np.allclose(H * x, np.zeros(self.n)))


class Test_QPModel(TestCase):

  def setUp(self):
    self.n = n = 5; self.m = m = 3
    self.c = np.random.random(n)
    self.A = np.random.random((m, n))
    H = np.random.random((n, n))
    self.H = H + H.T
    self.qp = QPModel(self.c,
                      linop_from_ndarray(self.H),
                      A=self.A,
                      Lvar=-np.random.random(n),
                      Uvar=np.random.random(n),
                      Lcon=-2*np.random.random(n),
                      Ucon=2*np.random.random(n))

  def test_constructor(self):
    qp = self.qp
    assert(isinstance(qp, QPModel))
    assert(qp.n == self.n)
    assert(qp.m == self.m)
    assert(qp.nlin == self.m)
    assert(qp.nnln == 0)
    assert(qp.nnet == 0)

  def test_qp(self):
    qp = self.qp
    x = np.random.random(self.n)
    assert(np.allclose(qp.obj(x), np.dot(self.c + 0.5 * np.dot(self.H, x), x)))
    assert(np.allclose(qp.grad(x), self.c + np.dot(self.H, x)))

    J = qp.jac(x)
    assert(J.shape == (self.m, self.n))
    p = np.random.random(self.n)
    assert(np.allclose(qp.jprod(x, p), np.dot(self.A, p)))
    q = np.random.random(self.m)
    assert(np.allclose(qp.jtprod(x, q), np.dot(self.A.T, q)))

    H = qp.hess(x, 0)
    assert(np.allclose(H * x, np.dot(self.H, x)))

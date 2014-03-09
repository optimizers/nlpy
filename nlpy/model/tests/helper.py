# Helper for nlpy.model tests
from numpy.testing import *
import numpy as np


class RosenbrockData(object):

  def __init__(self):
    self.expected_f = 1616.0
    self.expected_g = np.array([-804., -1204., -1204., -1204., -400.])
    self.expected_H = np.array([[1602.,  400.,    0.,    0.,   0.],
                                [ 400., 1802.,  400.,    0.,   0.],
                                [   0.,  400., 1802.,  400.,   0.],
                                [   0.,    0.,  400., 1802., 400.],
                                [   0.,    0.,    0.,  400., 200.]])
    v = np.arange(1, self.expected_H.shape[0] + 1, dtype=np.float)
    self.expected_Hv = np.dot(self.expected_H, v)


class Hs7Data(object):

  def __init__(self):
    self.expected_f = -0.39056208756589972
    self.expected_c = np.array([29.0])
    self.expected_l = -25.3905620876   # uses cons_pos().
    self.expected_g = np.array([0.8, -1.])
    self.expected_H = np.array([[-52.24,  0.],
                                [  0.  , -2.]])
    v = np.arange(1, self.expected_H.shape[0] + 1, dtype=np.float)
    self.expected_Hv = np.dot(self.expected_H, v)
    self.expected_J = np.array([[40., 4.]])
    self.expected_Jv = np.dot(self.expected_J, v)
    w = 2 * np.ones(self.expected_J.shape[0])
    self.expected_JTw = np.dot(self.expected_J.T, w)


rosenbrock_data = RosenbrockData()
hs7_data = Hs7Data()


class Rosenbrock(object):

  def get_derivatives(self, nlp):
    return get_derivatives_coord(nlp)

  def test_rosenbrock(self):
    f = get_values(self.nlp)
    assert_almost_equal(f, rosenbrock_data.expected_f)

    (g, H, Hv) = self.get_derivatives(self.nlp)
    assert(np.allclose(g, rosenbrock_data.expected_g))
    assert(np.allclose(H, rosenbrock_data.expected_H))
    assert(np.allclose(Hv, rosenbrock_data.expected_Hv))


class Hs7(object):

  def get_derivatives(self, nlp):
    return get_derivatives_coord(nlp)

  def test_hs7(self):
    (f, c, l) = get_values(self.nlp)
    assert_almost_equal(f, hs7_data.expected_f)
    assert_allclose(c, hs7_data.expected_c)
    assert_almost_equal(l, hs7_data.expected_l)

    (g, H, Hv, J, Jv, JTw) = self.get_derivatives(self.nlp)

    assert(np.allclose(g, hs7_data.expected_g))
    assert(np.allclose(H, hs7_data.expected_H))
    assert(np.allclose(Hv, hs7_data.expected_Hv))
    assert(np.allclose(J, hs7_data.expected_J))
    assert(np.allclose(Jv, hs7_data.expected_Jv))
    assert(np.allclose(JTw, hs7_data.expected_JTw))


def get_values(nlp):
  f = nlp.obj(nlp.x0)
  if nlp.m > 0:
    c = nlp.cons(nlp.x0)
    l = nlp.lag(nlp.x0, nlp.pi0)
    return (f, c, l)
  else:
    return f


def get_derivatives_plain(nlp):
  g = nlp.grad(nlp.x0)
  H = nlp.hess(nlp.x0, nlp.pi0)
  v = np.arange(1, nlp.nvar + 1, dtype=np.float)
  Hv = nlp.hprod(nlp.x0, nlp.pi0, v)
  if nlp.m > 0:
      J = nlp.jac(nlp.x0)
      Jop = nlp.jop(nlp.x0)
      Jv = Jop * v
      w = 2 * np.ones(nlp.ncon)
      JTw = Jop.T * w
      return (g, H, Hv, J, Jv, JTw)
  else:
      return (g, H, Hv)


def get_derivatives_coord(nlp):
  g = nlp.grad(nlp.x0)
  H = ndarray_from_coord(nlp.nvar, nlp.nvar,
                         *nlp.hess(nlp.x0, nlp.pi0), symmetric=True)
  v = np.arange(1, nlp.nvar + 1, dtype=np.float)
  Hv = nlp.hprod(nlp.x0, nlp.pi0, v)
  if nlp.m > 0:
      J = ndarray_from_coord(nlp.ncon, nlp.nvar,
                             *nlp.jac(nlp.x0), symmetric=False)
      Jop = nlp.jop(nlp.x0)
      Jv = Jop * v
      w = 2 * np.ones(nlp.ncon)
      JTw = Jop.T * w
      return (g, H, Hv, J, Jv, JTw)
  else:
      return (g, H, Hv)


def get_derivatives_llmat(nlp):
  g = nlp.grad(nlp.x0)
  H = ndarray_from_ll_mat_sym(nlp.hess(nlp.x0, nlp.pi0))
  v = np.arange(1, nlp.nvar + 1, dtype=np.float)
  Hv = nlp.hprod(nlp.x0, nlp.pi0, v)
  if nlp.m > 0:
      J = ndarray_from_ll_mat(nlp.jac(nlp.x0))
      Jop = nlp.jop(nlp.x0)
      Jv = Jop * v
      w = 2 * np.ones(nlp.ncon)
      JTw = Jop.T * w
      return (g, H, Hv, J, Jv, JTw)
  else:
      return (g, H, Hv)


def get_derivatives_scipy(nlp):
  g = nlp.grad(nlp.x0)
  H = nlp.hess(nlp.x0, nlp.pi0).todense()
  H = H + np.triu(H, 1).T
  v = np.arange(1, nlp.nvar + 1, dtype=np.float)
  Hv = nlp.hprod(nlp.x0, nlp.pi0, v)
  if nlp.m > 0:
      J = nlp.jac(nlp.x0).todense()
      Jop = nlp.jop(nlp.x0)
      Jv = Jop * v
      w = 2 * np.ones(nlp.ncon)
      JTw = Jop.T * w
      return (g, H, Hv, J, Jv, JTw)
  else:
      return (g, H, Hv)


def ndarray_from_ll_mat_sym(spA):
  n = spA.shape[0]
  A = np.zeros((n, n), dtype=np.float)
  for i in range(n):
    for j in range(i+1):
      A[i, j] = spA[i, j]
      A[j, i] = A[i, j]
  return A


def ndarray_from_ll_mat(spA):
  m, n = spA.shape
  A = np.zeros((m, n), dtype=np.float)
  for i in range(m):
    for j in range(n):
      A[i, j] = spA[i, j]
  return A


def ndarray_from_coord(nrow, ncol, vals, rows, cols, symmetric=False):
  A = np.zeros((nrow, ncol), dtype=np.float)
  for k in range(len(vals)):
    row = rows[k]
    col = cols[k]
    A[row, col] += vals[k]
    if symmetric and row != col:
      A[col, row] = vals[k]
  return A

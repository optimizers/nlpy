# Tests relative to algorithmic differentiation with AMPL.
from numpy.testing import *
from nlpy.model import AmplModel
from pysparse.sparse.spmatrix import ll_mat, ll_mat_sym
import numpy as np
import os

this_path = os.path.dirname(os.path.realpath(__file__))


def get_values(nlp):
  f = nlp.obj(nlp.x0)
  if nlp.m > 0:
    c = nlp.cons(nlp.x0)
    return (f, c)
  else:
    return f


def get_derivatives(nlp):
  g = nlp.grad(nlp.x0)
  H = ndarray_from_ll_mat_sym(nlp.hess(nlp.x0, nlp.pi0))
  if nlp.m > 0:
      J = ndarray_from_ll_mat(nlp.jac(nlp.x0))
      return (g, H, J)
  else:
      return (g, H)


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


class Test_AmplRosenbrock(TestCase):

  def setUp(self):
    model = os.path.join(this_path, 'rosenbrock.mod')
    self.rosenbrock = AmplModel(model)  # x0 = (-1, ..., -1)

  def test_rosenbrock(self):
    f = get_values(self.rosenbrock)
    expected_f = 1616.0
    assert_almost_equal(f, expected_f)

    (g, H) = get_derivatives(self.rosenbrock)
    expected_g = np.array([-804., -1204., -1204., -1204., -400.])
    expected_H = np.array([[1602.,  400.,    0.,    0.,   0.],
                           [ 400., 1802.,  400.,    0.,   0.],
                           [   0.,  400., 1802.,  400.,   0.],
                           [   0.,    0.,  400., 1802., 400.],
                           [   0.,    0.,    0.,  400., 200.]])
    assert(np.allclose(g, expected_g))
    assert(np.allclose(H, expected_H))


class Test_AmplHS7(TestCase):

    def setUp(self):
      model = os.path.join(this_path, 'hs007.mod')
      self.hs7 = AmplModel(model)  # x0 = (2, 2)

    def test_hs7(self):
      hs7 = self.hs7
      (f, c) = get_values(hs7)
      expected_f = -0.39056208756589972
      expected_c = np.array([29.0])  # AMPL leaves the constant out.
      assert_almost_equal(f, expected_f)
      assert_allclose(c, expected_c)

      (g, H, J) = get_derivatives(hs7)
      expected_g = np.array([0.8, -1.])
      expected_H = np.array([[-0.24, 0.], [0., 0.]])
      expected_J = np.array([[40., 4.]])
      assert(np.allclose(g, expected_g))
      assert(np.allclose(H, expected_H))
      assert(np.allclose(J, expected_J))

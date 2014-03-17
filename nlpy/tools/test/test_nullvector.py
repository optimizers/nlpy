# Tests relative to null vectors.
from numpy.testing import *
from nlpy.tools.nullvector import NullVector
import numpy as np


class TestNullVector(TestCase):

  def setUp(self):
    self.n = 10
    self.val = np.random.random()
    self.v = NullVector(self.n, self.val, dtype=np.float)

  def test_constructor(self):
    assert(isinstance(self.v, NullVector))
    assert(self.v.dtype == np.float)
    assert(self.v.value == np.float(self.val))
    assert(len(self.v) == self.n)
    assert(self.v.shape == (self.n,))

  def test_getitem(self):
    for i in range(self.n):
      assert(self.v[i] == np.float(self.val))

    assert(isinstance(self.v[1:self.n-1], NullVector))
    assert(isinstance(self.v[:-1], NullVector))
    assert(self.val in self.v)
    assert_raises(IndexError, self.v.__getitem__, 10)
    for v in self.v:
      assert(v == np.float(self.val))

  def test_add(self):
    uval = np.random.random()
    u = NullVector(self.n, uval, dtype=np.float)
    w = u + self.v
    assert(isinstance(w, NullVector))
    assert(len(w) == self.n)
    assert(w.value == np.float(self.val) + np.float(uval))
    assert(w.dtype == np.result_type(self.v.dtype, u.dtype))

    u = NullVector(self.n + 1, uval, dtype=np.float)
    assert_raises(ValueError, lambda y: y + self.v, u)

    w = u + 2
    assert(isinstance(w, NullVector))
    assert(w.value == uval + 2)

    u += 2
    assert(u.value == np.float(uval) + 2)

    u = np.random.random(self.n)
    w = u + self.v
    assert(isinstance(w, np.ndarray))
    assert(len(w) == self.n)
    for i in range(self.n):
      assert(w[i] == u[i] + self.val)

    u = np.random.random(self.n + 1)
    assert_raises(ValueError, lambda y: y + self.v, u)

  def test_sub(self):
    uval = np.random.random()
    u = NullVector(self.n, uval, dtype=np.float)
    w = u - self.v
    assert(isinstance(w, NullVector))
    assert(len(w) == self.n)
    assert(w.value == np.float(uval) - np.float(self.val))
    assert(w.dtype == np.result_type(self.v.dtype, u.dtype))

    u = NullVector(self.n + 1, uval, dtype=np.float)
    assert_raises(ValueError, lambda y: y - self.v, u)

    w = u - 2
    assert(isinstance(w, NullVector))
    assert(w.value == uval - 2)

    u -= 2
    assert(u.value == np.float(uval) - 2)

    u = np.random.random(self.n)
    w = u - self.v
    assert(isinstance(w, np.ndarray))
    assert(len(w) == self.n)
    for i in range(self.n):
      assert(w[i] == u[i] - self.val)

    u = np.random.random(self.n + 1)
    assert_raises(ValueError, lambda y: y - self.v, u)

  def test_mul(self):
    uval = np.random.random()
    u = NullVector(self.n, uval, dtype=np.float)
    w = u * self.v
    assert(isinstance(w, NullVector))
    assert(len(w) == self.n)
    assert(w.value == np.float(self.val) * np.float(uval))
    assert(w.dtype == np.result_type(self.v.dtype, u.dtype))

    u = NullVector(self.n + 1, uval, dtype=np.float)
    assert_raises(ValueError, lambda y: y * self.v, u)

    w = u * 2
    assert(isinstance(w, NullVector))
    assert(w.value == uval * 2)

    u *= 2
    assert(u.value == np.float(uval) * 2)

    u = np.random.random(self.n)
    w = u * self.v
    assert(isinstance(w, np.ndarray))
    assert(len(w) == self.n)
    for i in range(self.n):
      assert(w[i] == u[i] * self.val)

    u = np.random.random(self.n + 1)
    assert_raises(ValueError, lambda y: y * self.v, u)

  def test_div(self):
    uval = np.random.random() + 1  # Ensure nonzero.
    u = NullVector(self.n, uval, dtype=np.float)
    w = self.v / u
    assert(isinstance(w, NullVector))
    assert(len(w) == self.n)
    assert(w.value == np.float(self.val) / np.float(uval))
    assert(w.dtype == np.result_type(self.v.dtype, u.dtype))

    u = NullVector(self.n + 1, uval, dtype=np.float)
    assert_raises(ValueError, lambda y: self.v / y, u)

    w = u / 2
    assert(isinstance(w, NullVector))
    assert(w.value == uval / 2)

    u /= 2
    assert(u.value == np.float(uval) / 2)

    u = np.random.random(self.n) + 1
    w = self.v / u
    assert(isinstance(w, np.ndarray))
    assert(len(w) == self.n)
    for i in range(self.n):
      assert(w[i] == self.val / u[i])

    u = np.random.random(self.n + 1)
    assert_raises(ValueError, lambda y: self.v / y, u)

  def test_pow(self):
    power = 2
    u = self.v ** power
    assert(isinstance(u, NullVector))
    assert(u.value == np.float(self.val) ** power)
    assert(len(u) == self.n)
    assert(u.dtype == np.result_type(np.float, np.int))

    u **= power
    assert(u.value == np.float(self.val) ** (2*power))

  def test_unary(self):
    u = -self.v
    assert(isinstance(u, NullVector))
    assert(len(u) == self.n)
    assert(u.value == -self.v.value)

    assert(+self.v is self.v)

    w = abs(u)
    assert(isinstance(w, NullVector))
    assert(len(w) == self.n)
    assert(w.value == self.val)

  def test_order(self):
    u = NullVector(self.n, self.val, dtype=np.float)
    assert_equal(u == self.v, range(self.n))

    u = NullVector(self.n, self.val + 1, dtype=np.float)
    assert_equal(u == self.v, [])

    u = NullVector(self.n + 1, self.val, dtype=np.float)
    assert_equal(u == self.v, [])

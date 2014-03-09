from numpy.testing import *
import algopy
from nlpy.model.algopymodel import AlgopyModel
from nlpy.model.tests.helper import *
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
    c = algopy.zeros(1, dtype=x)
    c[0] = (1 + x[0]**2)**2 + x[1]**2 - 4  # AlgoPy doesn't support cons_pos()
    return c


class Test_AlgopyRosenbrock(TestCase, Rosenbrock):  # Test def'd in Rosenbrock

  def get_derivatives(self, nlp):
    return get_derivatives_plain(nlp)

  def setUp(self):
    self.nlp = AlgopyRosenbrock(n=5, name='Rosenbrock', x0=-np.ones(5))


class Test_AlgopyHS7(TestCase, Hs7):  # Test def'd in Hs7

  def get_expected(self):
    hs7_data = Hs7Data()
    hs7_data.expected_c = np.array([25.])  # AlgoPy doesn't support cons_pos()
    return hs7_data

  def get_derivatives(self, nlp):
    return get_derivatives_plain(nlp)

  def setUp(self):
    self.nlp = AlgopyHS7(n=2, m=1, name='HS7', x0=2*np.ones(2), pi0=np.ones(1),
                         Lcon=np.array([0.]), Ucon=np.array([0.]))

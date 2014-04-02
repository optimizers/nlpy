# -*- coding: utf-8 -*-
from nlpy.model import NLPModel, QuasiNewtonModel, SlackModel

import numpy as np


class AugmentedLagrangian(NLPModel):
  """
  A bound-constrained augmented Lagrangian. In case the original NLP has
  general inequalities, slack variables are introduced.

  The augmented Lagrangian is defined as:

    L(x, π; ρ) := f(x) - π'c(x) + ½ ρ |c(x)|².

  where π are the current Lagrange multiplier estimates and ρ is the
  current penalty parameter.

  :parameters:

    :nlp:   original NLPModel.

  :keywords:

    :rho:  initial value for the penalty parameter (default: 10.)
    :pi:   vector of initial multipliers (default: all zero.)
  """

  def __init__(self, nlp, **kwargs):

    if nlp.m == nlp.nequalC:
      self.nlp = nlp
    else:
      self.nlp = SlackModel(nlp, keep_variable_bounds=True, **kwargs)

    NLPModel.__init__(self, n=self.nlp.n, m=0, name='Al-'+self.nlp.name,
                      Lvar=self.nlp.Lvar, Uvar=self.nlp.Uvar)

    self.rho_init = kwargs.get('rho', 10.)
    self._rho = self.rho_init

    self.pi0 = np.zeros(self.nlp.m)
    self.pi = self.pi0.copy()
    self.x0 = self.nlp.x0

  @property
  def rho(self):
    return self._rho

  @rho.setter
  def rho(self, value):
    self._rho = max(0, value)

  def obj(self, x, **kwargs):
    """
    Evaluate augmented Lagrangian function.
    """
    cons = self.nlp.cons(x)

    alfunc = self.nlp.obj(x)
    alfunc -= np.dot(self.pi, cons)
    alfunc += 0.5 * self.rho * np.dot(cons, cons)
    return alfunc

  def grad(self, x, **kwargs):
    """
    Evaluate augmented Lagrangian gradient.
    """
    nlp = self.nlp
    J = nlp.jop(x)
    cons = nlp.cons(x)
    algrad = nlp.grad(x) + J.T * (self.rho * cons - self.pi)
    return algrad

  def dual_feasibility(self, x, **kwargs):
    """
    Evaluate Lagrangian gradient.
    """
    nlp = self.nlp
    J = nlp.jop(x)
    lgrad = nlp.grad(x) - J.T * self.pi
    return lgrad

  def hprod(self, x, z, v, **kwargs):
    """
    Compute the Hessian-vector product of the Hessian of the augmented
    Lagrangian with arbitrary vector v.
    """
    nlp = self.nlp
    cons = nlp.cons(x)

    w = nlp.hprod(x, self.rho * cons - self.pi, v)
    J = nlp.jop(x)
    return w + self.rho * J.T * J * v

  def hess(self, *args, **kwargs):
    return self.hop(*args, **kwargs)


class QuasiNewtonAugmentedLagrangian(QuasiNewtonModel, AugmentedLagrangian):
  """
  A bound-constrained augmented Lagrangian model with a quasi-Newton
  approximate Hessian. In instances of this class, the quasi-Newton
  Hessian approximates the Hessian of the augmented Lagrangian as a whole.

  If the quasi-Newton Hessian should approximate only the Hessian of the
  Lagrangian, consider an initialization of the form

      AugmentedLagrangian(QuasiNewtonModel(...))
  """
  pass  # All the work is done by the parent classes.

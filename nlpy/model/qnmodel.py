from nlpy.model import NLPModel

__docformat__ = 'restructuredtext'


class QuasiNewtonModel(NLPModel):
  """
  An ``NLPModel`` in which the Hessian is given by a quasi-Newton
  approximation.

  :keywords:
    :H: the `class` of a quasi-Newton linear operator.
        This keyword is mandatory.

  Keywords accepted by the quasi-Newton class will be passed
  directly to its constructor.
  """

  def __init__(self, *args, **kwargs):
    super(QuasiNewtonModel, self).__init__(*args, **kwargs)
    qn_cls = kwargs.pop('H')
    self._H = qn_cls(self.nvar, **kwargs)

  @property
  def H(self):
      return self._H

  def hess(self, *args, **kwargs):
    return self.H

  def hop(self, *args, **kwargs):
    return self.H

  def hprod(self, x, z, v, **kwargs):
    return self.H * v

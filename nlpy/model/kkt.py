class KKTresidual(object):
  """
  A generic class to package KKT residuals and corresponding scalings.
  """
  def __init__(self, dFeas, pFeas, bFeas, gComp, bComp, **kwargs):
    """
    :parameters:
        :dFeas: dual feasibility residual
        :pFeas: primal feasibility residual, taking into account
                constraints that are not bound constraints,
        :bFeas: primal feasibility with respect to bounds,
        :gComp: complementarity residual with respect to constraints
                that are not bound constraints,
        :bComp: complementarity residual with respect to bounds.
    """
    self._dFeas = dFeas
    self._pFeas = pFeas
    self._bFeas = bFeas
    self._gComp = gComp
    self._bComp = bComp
    self._is_scaling = kwargs.get('is_scaling', False)
    if self._is_scaling:
      self._scaling = None
    else:
      if 'scaling' in kwargs:
        self.set_scaling(kwargs['scaling'])
      else:
        self._scaling = KKTresidual(1.0, 1.0, 1.0, 1.0, 1.0,
                                    is_scaling=True)
    return

  @property
  def scaling(self):
      return self._scaling

  @scaling.setter
  def scaling(self, *args, **kwargs):
      self.set_scaling(*args, **kwargs)

  @property
  def is_scaling(self):
    "Indicates whether instance is a scaling or residual instance."
    return self._is_scaling

  @property
  def dFeas(self):
    "Scaled dual feasibility residual."
    df = self._dFeas
    return df if self.is_scaling else df / self._scaling._dFeas

  @dFeas.setter
  def dFeas(self, value):
    self._dFeas = max(0, value)

  @property
  def pFeas(self):
    "Scaled primal feasibility with respect to general constraints."
    pf = self._pFeas
    return pf if self.is_scaling else pf / self._scaling._pFeas

  @pFeas.setter
  def pFeas(self, value):
    self._pFeas = max(0, value)

  @property
  def bFeas(self):
    "Scaled primal feasibility with respect to bounds."
    bf = self._bFeas
    return bf if self.is_scaling else bf / self._scaling._bFeas

  @bFeas.setter
  def bFeas(self, value):
    self._bFeas = max(0, value)

  @property
  def feas(self):
    "Scaled primal feasibility with respect to constraints and bounds."
    return max(self.pFeas, self.bFeas)

  @property
  def gComp(self):
    "Scaled complementarity with respect to general constraints."
    gc = self._gComp
    return gc if self.is_scaling else gc / self._scaling._gComp

  @gComp.setter
  def gComp(self, value):
    self._gComp = max(0, value)

  @property
  def bComp(self):
    "Scaled complementarity with respect to bounds."
    bc = self._bComp
    return bc if self.is_scaling else bc / self._scaling._bComp

  @bComp.setter
  def bComp(self, value):
    self._bComp = max(0, value)

  @property
  def comp(self):
    "Scaled complementarity with respect to constraints and bounds."
    return max(self.gComp, self.bComp)

  def set_scaling(self, scaling, **kwargs):
      "Assign scaling values. `scaling` must be a `KKTresidual` instance."
      if self._is_scaling:
          raise ValueError('instance represents scaling factors.')
      if not isinstance(scaling, KKTresidual):
          raise ValueError('scaling must be a KKTresidual instance.')
      self._scaling = scaling
      self._scaling._is_scaling = True
      return

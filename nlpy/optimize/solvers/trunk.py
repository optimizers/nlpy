"""
 TRUNK
 Trust-Region Method for Unconstrained Programming.

 D. Orban            Montreal Sept. 2003
"""
from nlpy.model import QPModel
from nlpy.optimize.tr.trustregion import TrustRegionSolver
from nlpy.tools import norms
from nlpy.tools.timing import cputime
from nlpy.tools.exceptions import UserExitRequest
import numpy
import logging
from math import sqrt

__docformat__ = 'restructuredtext'


class Trunk(object):
  """
  An abstract framework for a trust-region-based algorithm for nonlinear
  unconstrained programming. Instantiate using

  `TRNK = TrunkFramework(nlp, TR, TrSolver)`

  :parameters:

    :nlp:       a :class:`NLPModel` object representing the problem. For
                instance, nlp may arise from an AMPL model.
    :TR:        a :class:`TrustRegion` instance.
    :TrSolver:  a trust-region solver to be passed as argument to
                the :class:`TrustRegionSolver` constructor.


  :keywords:

    :x0:           starting point                     (default nlp.x0)
    :reltol:       relative stopping tolerance        (default `nlp.stop_d`)
    :abstol:       absolute stopping tolerance        (default 1.0e-6)
    :maxiter:      maximum number of iterations       (default max(1000,10n))
    :inexact:      use inexact Newton stopping tol    (default False)
    :ny:           apply Nocedal/Yuan linesearch      (default False)
    :nbk:          max number of backtracking steps in Nocedal/Yuan
                   linesearch                         (default 5)
    :monotone:     use monotone descent strategy      (default False)
    :nIterNonMono: number of iterations for which non-strict descent can
                   be tolerated if ``monotone=False`` (default 25)
    :logger_name:  name of a logger object that can be used in the post
                   iteration                          (default None)

  Once a `TrunkFramework` object has been instantiated and the problem is
  set up, solve problem by issuing a call to `TRNK.solve()`. The algorithm
  stops as soon as the Euclidian norm of the gradient falls below

    ``max(abstol, reltol * g0)``

  where ``g0`` is the Euclidian norm of the gradient at the initial point.
  """

  def __init__(self, nlp, TR, TrSolver, **kwargs):

    self.nlp = nlp
    self.TR = TR
    self.TrSolver = TrSolver
    self.solver = None  # Will point to solver data in Solve()
    self.iter = 0     # Iteration counter
    self.total_cgiter = 0
    self.x    = kwargs.get('x0', self.nlp.x0.copy())
    self.f    = None
    self.f0   = None
    self.g    = None
    self.g_old  = None
    self.save_g = False
    self.gnorm  = None
    self.g0   = None
    self.alpha  = 1.0     # For Nocedal-Yuan backtracking linesearch
    self.tsolve = 0.0

    self.step_accepted = False
    self.dvars = None
    self.dgrad = None

    self.reltol  = kwargs.get('reltol', self.nlp.stop_d)
    self.abstol  = kwargs.get('abstol', 1.0e-6)

    self.ny = kwargs.get('ny', False)
    self.nbk = kwargs.get('nbk', 5)
    self.inexact = kwargs.get('inexact', False)
    self.monotone = kwargs.get('monotone', False)
    self.nIterNonMono = kwargs.get('nIterNonMono', 25)
    self.logger = kwargs.get('logger', None)

    self.hformat = '%-5s  %8s  %7s  %5s  %8s  %8s  %8s  %4s'
    self.header  = self.hformat % ('Iter','f(x)','|g(x)|','cg','rho','Step','Radius','Stat')
    self.hlen    = len(self.header)
    self.hline   = '-' * self.hlen
    self.format  = '%-5d  %8.1e  %7.1e  %5d  %8.1e  %8.1e  %8.1e  %4s'
    self.format0 = '%-5d  %8.1e  %7.1e  %5s  %8s  %8s  %8.1e  %4s'
    self.radii = [TR.radius]

    # Setup the logger. Install a NullHandler if no output needed.
    logger_name = kwargs.get('logger_name', 'nlpy.trunk')
    self.log = logging.getLogger(logger_name)
    self.log.addHandler(logging.NullHandler())
    self.log.propagate = False

  def precon(self, v, **kwargs):
    """
    Generic preconditioning method---must be overridden
    """
    return v

  def post_iteration(self, **kwargs):
    """
    Override this method to perform work at the end of an iteration. For
    example, use this method for preconditioners that need updating,
    e.g., a limited-memory BFGS preconditioner.
    """
    return None

  def solve(self, **kwargs):
    """
    :keywords:
      :maxiter:  maximum number of iterations.

    All other keyword arguments are passed directly to the constructor of
    the trust-region solver.
    """

    self.maxiter = kwargs.get('maxiter', max(1000, 10 * self.nlp.n))
    if 'maxiter' in kwargs:
      kwargs.pop('maxiter')

    nlp = self.nlp

    # Gather initial information.
    self.f    = self.nlp.obj(self.x)
    self.f0   = self.f
    self.g    = self.nlp.grad(self.x)
    self.g_old  = self.g
    self.gnorm  = norms.norm2(self.g)
    self.g0   = self.gnorm

    cgtol = 1.0 if self.inexact else -1.0
    stoptol = max(self.abstol, self.reltol * self.g0)
    step_status = None
    exitUser = False
    exitOptimal = self.gnorm <= stoptol
    exitIter = self.iter >= self.maxiter
    status = ''

    # Initialize non-monotonicity parameters.
    if not self.monotone:
      fMin = fRef = fCan = self.f0
      l = 0
      sigRef = sigCan = 0

    t = cputime()

    # Print out header and initial log.
    if self.iter % 20 == 0:
      self.log.info(self.hline)
      self.log.info(self.header)
      self.log.info(self.hline)
      self.log.info(self.format0 % (self.iter, self.f,
                                    self.gnorm, '', '', '',
                                    self.TR.radius, ''))

    while not (exitUser or exitOptimal or exitIter):

      self.iter += 1
      self.alpha = 1.0

      if self.save_g:
        self.g_old = self.g.copy()

      # Iteratively minimize the quadratic model in the trust region
      # m(s) = <g, s> + 1/2 <s, Hs>
      # Note that m(s) does not include f(x): m(0) = 0.

      if self.inexact:
        cgtol = max(1.0e-6, min(0.5 * cgtol, sqrt(self.gnorm)))

      qp = QPModel(self.g, self.nlp.hop(self.x, self.nlp.pi0))
      self.solver = TrustRegionSolver(qp, self.TrSolver)
      self.solver.solve(prec=self.precon, radius=self.TR.radius, reltol=cgtol)

      step = self.solver.step
      snorm = self.solver.step_norm
      cgiter = self.solver.niter

      # Obtain model value at next candidate
      m = self.solver.m
      if m is None:
        m = qp.obj(step)

      self.total_cgiter += cgiter
      x_trial = self.x + step
      f_trial = nlp.obj(x_trial)

      rho = self.TR.ratio(self.f, f_trial, m)

      if not self.monotone:
        rhoHis = (fRef - f_trial) / (sigRef - m)
        rho = max(rho, rhoHis)

      step_status = 'Rej'
      self.step_accepted = False

      if rho >= self.TR.eta1:

        # Trust-region step is accepted.

        self.TR.update_radius(rho, snorm)
        self.x = x_trial
        self.f = f_trial
        self.g = nlp.grad(self.x)
        self.gnorm = norms.norm2(self.g)
        self.dvars = step
        if self.save_g:
          self.dgrad = self.g - self.g_old
        step_status = 'Acc'
        self.step_accepted = True

        # Update non-monotonicity parameters.
        if not self.monotone:
          sigRef = sigRef - m
          sigCan = sigCan - m
          if f_trial < fMin:
            fCan = f_trial
            fMin = f_trial
            sigCan = 0
            l = 0
          else:
            l = l + 1

          if f_trial > fCan:
            fCan = f_trial
            sigCan = 0

          if l == self.nIterNonMono:
            fRef = fCan
            sigRef = sigCan

      else:

        # Trust-region step is rejected.
        if self.ny:  # Backtracking linesearch a la Nocedal & Yuan.

          slope = numpy.dot(self.g, step)
          bk = 0
          while bk < self.nbk and f_trial >= self.f + 1.0e-4 * self.alpha * slope:
            bk = bk + 1
            self.alpha /= 1.2
            x_trial = self.x + self.alpha * step
            f_trial = nlp.obj(x_trial)
          self.x = x_trial
          self.f = f_trial
          self.g = nlp.grad(self.x)
          self.gnorm = norms.norm2(self.g)
          self.TR.radius = self.alpha * snorm
          snorm /= self.alpha
          step_status = 'N-Y'
          self.step_accepted = True
          self.dvars = self.alpha * step
          if self.save_g:
            self.dgrad = self.g - self.g_old

        else:
          self.TR.update_radius(rho, snorm)

      self.step_status = step_status
      self.radii.append(self.TR.radius)
      status = ''
      try:
        self.post_iteration()
      except UserExitRequest:
        status = 'usr'

      # Print out header, say, every 20 iterations
      if self.iter % 20 == 0:
        self.log.info(self.hline)
        self.log.info(self.header)
        self.log.info(self.hline)

      pstatus = step_status if step_status != 'Acc' else ''
      self.log.info(self.format % (self.iter, self.f,
                    self.gnorm, cgiter, rho, snorm,
                    self.TR.radius, pstatus))

      exitOptimal = self.gnorm <= stoptol
      exitIter = self.iter > self.maxiter
      exitUser = status == 'usr'

    self.tsolve = cputime() - t  # Solve time

    # Set final solver status.
    if status == 'usr':
      pass
    elif self.gnorm <= stoptol:
      status = 'opt'
    else:  # self.iter > self.maxiter:
      status = 'itr'
    self.status = status


class QNTrunk(Trunk):

  def __init__(self, *args, **kwargs):
    super(QNTrunk, self).__init__(*args, **kwargs)
    self.save_g = True

  def post_iteration(self, **kwargs):
    # Update quasi-Newton approximation.
    if self.step_accepted:
      self.nlp.H.store(self.dvars, self.dgrad)

"""
Linesearch methods guaranteeing satisfaction of the strong Wolfe conditions.
"""

import numpy
from nlpy.optimize.ls import _pycsrch

class StrongWolfeLineSearch:
    """
    A general-purpose linesearch procedure enforcing the strong
    Wolfe conditions

    f(x+td) <= f(x) + ftol * t * <g(x),d>    (Armijo condition)

    | <g(x+td),d> | <= gtol * | <g(x),d> |   (curvature condition)

    This is a Python interface to the More and Thuente linesearch.

    Instantiate as follows

    SWLS = StrongWolfeLineSearch(f, g, d, obj, grad)

    where

    * f     is the objective value    at the current iterate x
    * g     is the objective gradient at the current iterate x
    * d     is the current search direction
    * obj   is a scalar function used to evaluate the value of
            the objective at x + t d, given t.
    * grad  is a scalar function used to evaluate the gradient
            of the objective at x + t d, given t.

    :keywords:

        :ftol:    the constant used in the Armijo condition (1e-3)
        :gtol:    the constant used in the curvature condition (0.9)
        :xtol:    a minimal relative step bracket length (1e-10)
        :stp:     an initial step value (1.0)
        :stpmin:  the initial lower bound of the bracket
        :stpmax:  the initial upper bound of the bracket

    To ensure existence of a step satisfying the strong Wolfe
    conditions, d should be a descent direction for f at x and
    ftol <= gtol.

    The final value of the step will be held in SWLS.stp

    In case an error happens, the return value SWLS.stp will be set
    to None and SWLS.message will describe what happened.

    After the search, SWLS.armijo will be set to True if the step
    computed satisfies the Armijo condition and SWLS.curvature will
    be set to True if the step satisfies the curvature condition.
    """
    def __init__(self, f, g, d, obj, grad, **kwargs):

        # Mandatory arguments
        self.f = f   # Function value f(xk)
        self.g = g.copy()   # Gradient of f at xk
        self.d = d   # Direction along which to search

        self.obj  = obj   # To evaluate function value
        self.grad = grad  # To evaluate function gradient

        # Optional arguments
        self.ftol   = kwargs.get('ftol', 1.0e-4)
        self.gtol   = kwargs.get('gtol', 0.9)
        self.xtol   = kwargs.get('xtol', 1.0e-1)
        self.stp    = kwargs.get('stp', 1.0)
        self.stpmin = kwargs.get('stpmin', 0.0)

        self.slope  = numpy.dot(self.g, self.d)
        self.stpmax = kwargs.get('stpmax',
                                  max(4*min(self.stp,1.0),
                                       0.1*self.f/(- self.slope*self.ftol)))

        # Initialize context object
        self.context = _pycsrch.Init(self.ftol,
                                      self.gtol,
                                      self.xtol,
                                      self.stp,
                                      self.stpmin,
                                      self.stpmax)

        self.armijo = False
        self.curvature = False
        self.message = None

    def search(self):

        if self.slope >= 0.0:
            self.stp = None
            self.message  = 'Direction is not a descent direction. '
            self.message += 'Slope = %-g' % self.slope
            return

        self.stp, task = self.context.csrch(self.f, self.slope)

        while task[:2] == 'FG':
            #print '  ls trying step = ', self.stp
            self.f = self.obj(self.stp)
            self.g = self.grad(self.stp)
            self.slope = numpy.dot(self.g, self.d)
            self.stp, task = self.context.csrch(self.f, self.slope)

        if task[:4] == 'CONV':     # Strong Wolfe conditions satisfied
            self.armijo = True
            self.curvature = True
        elif task[:4] == 'WARN':   # Armijo condition only is satisfied
            self.armijo = True
        self.message = task
        return


if __name__ == '__main__':

    import amplpy
    import numpy
    from math import sqrt
    import sys

    nlp = amplpy.AmplModel(sys.argv[1])
    f = nlp.obj(nlp.x0)
    g = nlp.grad(nlp.x0)
    d = -g
    SWLS = StrongWolfeLineSearch(f,
                                  g,
                                  d,
                                  lambda t: nlp.obj(nlp.x0 + t * d),
                                  lambda t: nlp.grad(nlp.x0 + t * d))
    print ' Before search'
    print '   f = ', f
    print '   initial slope = ', SWLS.slope
    print '   stpmax = ', SWLS.stpmax
    SWLS.search()
    print ' After search'
    print '   f = ', SWLS.f
    print '   step length = ', SWLS.stp
    print '   final slope = ', SWLS.slope
    print '   armijo = ', SWLS.armijo
    print '   curvature = ', SWLS.curvature
    print '   message = ', SWLS.message

    import pylab
    npoints = 50
    stepmax = 2*SWLS.stp
    stepmin = SWLS.stpmin
    delta = (stepmax - stepmin)/npoints
    t = numpy.arange(stepmin, stepmax, delta)
    y = map(lambda u: nlp.obj(nlp.x0 + u * d)/f, t)
    pylab.plot(t, y)
    pylab.show()
    nlp.close()


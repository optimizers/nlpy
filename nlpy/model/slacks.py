"""
A framework for converting a general nonlinear program into a farm with
(possibly nonlinear) equality constraints and bounds only, by adding slack
variables.

D. Orban, Montreal, 2008.
"""

__docformat__ = 'restructuredtext'

import numpy
from nlpy.model import AmplModel
from nlpy.tools import List
from pysparse.pysparseMatrix import PysparseMatrix as sp

class SlackFramework( AmplModel ):

    def __init__(self, model, **kwargs):
        """
        General framework for converting a nonlinear optimization problem to a
        form using slack variables.

        The initial problem has the form

        minimize    f(x)
        subject to  ci(x) = ai,          i = 1, ..., m,
                    gjL <= gj(x) <= gjU, j = 1, ..., p,
                    xkL <= xk <= xkU,    k = 1, ..., n,

        where some or all lower bounds gjL and xkL may be equal to -Infinity,
        and some or all upper bounds gjU and xkU may be equal to +Infinity.

        The transformed problem is

        minimize    f(x)
        subject to  ci(x) - ai = 0,        i = 1, ..., m,
                    gj(x) - gjL - sjL = 0, j = 1, ..., p, for which gjL > -Inf,
                    sjL >= 0,              j = 1, ..., p, for which gjL > -Inf,
                    gjU - gj(x) - sjU = 0, j = 1, ..., p, for which gjU < +Inf,
                    sjU >= 0,              j = 1, ..., p, for which gjU < +Inf,
                    xk - xkL - tkL = 0,    k = 1, ..., n, for which xkL > -Inf,
                    tkL >= 0,              k = 1, ..., n, for which xkL > -Inf,
                    xkU - xk - tkU = 0,    k = 1, ..., n, for which xkU < +Inf,
                    tkU >= 0,              k = 1, ..., n, for which xkU < +Inf.

        In the latter problem, the only inequality constraints are bounds on
        the slack variables. The other constraints are (typically) nonlinear
        equalities.

        This framework does not initialize the slack variables
        sL, sU, tL, and tU by default since for different purposes, they will be
        initialized differently. However, calling self.InitializeSlacks() will
        initialize them to zero or a value supplied as argument. Users may want
        to override this method.
        """
        AmplModel.__init__(self, model, **kwargs)
        
        # Define slacks for inequality constraints with a lower bound
        self.n_con_low = self.nlowerC + self.nrangeC
        self.s_low = numpy.empty(self.n_con_low)

        # Define slacks for inequality constraints with an upper bound
        self.n_con_up = self.nupperC + self.nrangeC
        self.s_up = numpy.empty(self.n_con_up)

        # Define slacks for variables with a lower bound
        self.n_var_low = self.nlowerB + self.nrangeB
        self.t_low = numpy.empty(self.n_var_low)

        # Define slacks for variables with an upper bound
        self.n_var_up = self.nupperB + self.nrangeB
        self.t_up = numpy.empty(self.n_var_up)

    def InitializeSlacks(self, val=0.0, **kwargs):
        """
        Initialize slacks to the value val. This may need to be overridden.
        """
        self.s_low[:] = val
        self.s_up[:]  = val
        self.t_low[:] = val
        self.t_up[:]  = val
        return None

    def Cons(self, x):
        """
        Evaluate the vector of general constraints for the modified problem.
        Constraints are stored in the order in which they appear in the
        original problem. If constraint i is a range constraint, c[i] will
        be the constraint that has the slack on the lower bound on c[i].
        The constraint with the slack on the upper bound on c[i] will be stored
        in position m + k, where k is the position of index i in
        rangeC, i.e., k=0 iff constraint i is the range constraint that
        appears first, k=1 iff it appears second, etc.
        """
        m = self.m
        equalC = self.equalC
        lowerC = self.lowerC ; nlowerC = self.nlowerC
        upperC = self.upperC ; nupperC = self.nupperC
        rangeC = self.rangeC ; nrangeC = self.nrangeC
        s_low  = self.s_low  ; s_up    = self.s_up

        c = numpy.empty(m + nrangeC)
        c[:m] = self.cons(x)
        c[m:] = c[rangeC]

        c[equalC] -= self.Lcon[equalC]
        c[lowerC] -= self.Lcon[lowerC] ; c[lowerC] -= s_low[:nlowerC]

        c[upperC] -= self.Ucon[upperC] ; c[upperC] *= -1
        c[upperC] -= s_up[:nupperC]

        c[rangeC] -= self.Lcon[rangeC] ; c[rangeC] -= s_low[nlowerC:]
        c[m:]     -= self.Ucon[rangeC] ; c[m:] *= -1
        c[m:]     -= s_up[nupperC:]

        return c

    def Bounds(self, x):
        """
        Evaluate the vector of equality constraints corresponding to bounds
        on the variables in the original problem.
        """
        lowerB = self.lowerB ; nlowerB = self.nlowerB
        upperB = self.upperB ; nupperB = self.nupperB
        rangeB = self.rangeB ; nrangeB = self.nrangeB
        n  = self.n
        t_low  = self.t_low  ; t_up    = self.t_up

        b = numpy.empty(n + nrangeB)
        b[:n] = x[:]
        b[n:] = x[rangeB]

        b[lowerB] -= self.Lvar[lowerB] ; b[lowerB] -= t_low[:nlowerB]

        b[upperB] -= self.Uvar[upperB] ; b[upperB] *= -1
        b[upperB] -= t_up[:nupperB]

        b[rangeB] -= self.Lvar[rangeB] ; b[rangeB] -= t_low[nlowerB:]
        b[n:]     -= self.Uvar[rangeB] ; b[n:] *= -1
        b[n:]     -= t_up[nupperB:]

        return b

    def Jac(self, x):
        """
        Evaluate the Jacobian matrix of all equality constraints of the
        transformed problem. The gradients of the general constraints appear in
        'natural' order, i.e., in the order in which they appear in the problem.
        The gradients of range constraints appear in two places: first in the
        'natural' location and again after all other general constraints, with a
        flipped sign to account for the upper bound on those constraints.

        The gradients of the linear equalities corresponding to bounds on the
        original variables appear in the following order:

        1. variables with a lower bound only
        2. lower bound on variables with two-sided bounds
        3. variables with an upper bound only
        4. upper bound on variables with two-sided bounds

        The overall Jacobian of the new constraints thus has the form

           n     s   sU    t   tU
        +-----+----+----+----+----+
        |  J  | -I |    |    |    |   <-- general constraints (natural order)
        +-----+----+----+----+----+
        | -JR |    | -I |    |    |   <-- 'upper' side of range constraints
        +-----+----+----+----+----+
        |  I  |    |    | -I |    |   <-- bounds, ordered as explained above
        +-----+----+----+----+----+
        | -I  |    |    |    | -I |   <-- 'upper' side of two-sided bounds
        +-----+----+----+----+----+

        where the signs corresponding to 'upper' constraints and upper bounds
        should be flipped in the (1,1) and (3,1) blocks.
        """
        n = self.n
        m = self.m

        # List() simply allows operations such as 1 + [2,3] -> [3,4]
        lowerC = List(self.lowerC) ; nlowerC = self.nlowerC
        upperC = List(self.upperC) ; nupperC = self.nupperC
        rangeC = List(self.rangeC) ; nrangeC = self.nrangeC
        lowerB = List(self.lowerB) ; nlowerB = self.nlowerB
        upperB = List(self.upperB) ; nupperB = self.nupperB
        rangeB = List(self.rangeB) ; nrangeB = self.nrangeB
        nbnds  = nlowerB + nupperB + nrangeB
        nSlacks = m+nrangeC

        # Initialize sparse Jacobian
        nJ = n + m + nrangeC + nbnds + nrangeB
        mJ =     m + nrangeC + nbnds + nrangeB
        nnzJ = 2 * self.nnzj + m + nrangeC + nbnds + nrangeB  # Overestimate
        J = sp(nrow=mJ, ncol=nJ, sizeHint=nnzJ)

        # Insert contribution of general constraints
        J[:m,:n] = self.jac(x)
        J[upperC,:n] *= -1.0               # Flip sign of 'upper' gradients
        J[m:nSlacks,:n] = -J[rangeC,:n]    # Append 'upper' side of range const.

        # Insert contribution of slacks on the general constraints
        J.put(-1.0, lowerC, n+lowerC)
        J.put(-1.0, upperC, n+upperC)
        J.put(-1.0, rangeC, n+rangeC)
        J.put(-1.0, m+rangeC, n+m+rangeC)

        # Insert contribution of bound constraints on the original problem
        bot  = nSlacks ; J.put( 1.0, range(bot,bot+nlowerB), lowerB)
        bot += nlowerB ; J.put( 1.0, range(bot,bot+nrangeB), rangeB)
        bot += nrangeB ; J.put(-1.0, range(bot,bot+nupperB), upperB)
        bot += nupperB ; J.put(-1.0, range(bot,bot+nrangeB), rangeB)

        # Insert contribution of slacks on the bound constraints
        bot = m+nrangeC ; J.put(-1.0, range(bot,bot+nlowerB), n+nSlacks+lowerB)
        bot += nlowerB  ; J.put(-1.0, range(bot,bot+nrangeB), n+nSlacks+rangeB)
        bot += nrangeB  ; J.put(-1.0, range(bot,bot+nupperB), n+nSlacks+upperB)
        bot += nupperB  ; J.put(-1.0, range(bot,bot+nrangeB), n+nSlacks+rangeB)

        return J

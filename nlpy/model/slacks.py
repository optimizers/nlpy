"""
A framework for converting a general nonlinear program into a farm with
(possibly nonlinear) equality constraints and bounds only, by adding slack
variables.

.. moduleauthor:: D. Orban <dominique.orban@gerad.ca>
"""

__docformat__ = 'restructuredtext'

import numpy
from nlpy.model import AmplModel
from nlpy.tools import List
from pysparse.pysparseMatrix import PysparseMatrix as sp
from pysparse import spmatrix

class SlackFramework( AmplModel ):
    r"""
    General framework for converting a nonlinear optimization problem to a
    form using slack variables.

    The initial problem has the form

    .. math::

       \min        & f(x) & \\
       \text{s.t.} & c_i(x) = a_i,                 & i = 1, \ldots, m, \\
                   & g_j^L \leq g_j(x) \leq g_j^U, & j = 1, \ldots, p, \\
                   & x_k^L \leq x_k \leq x_k^U,    & k = 1, \ldots, n,

    where some or all lower bounds :math:`g_j^L` and :math:`x_k^L` may be equal
    to :math:`- \infty`, and some or all upper bounds :math:`g_j^U`
    and :math:`x_k^U` may be equal to :math:`+ \infty`.

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

    The order of variables in the transformed problem is as follows:

    [  x  |  sL  |  sU  |  tL  |  tU  ]

    where:

    - sL = [ sLL | sLR ], sLL being the slack variables corresponding to
      general constraints with a lower bound only, and sLR being the slack
      variables corresponding to the 'lower' side of range constraints.

    - sU = [ sUU | sUR ], sUU being the slack variables corresponding to
      general constraints with an upper bound only, and sUR being the slack
      variables corresponding to the 'upper' side of range constraints.

    - tL = [ tLL | tLR ], tLL being the slack variables corresponding to
      variables with a lower bound only, and tLR being the slack variables
      corresponding to the 'lower' side of two-sided bounds.

    - tU = [ tUU | tUR ], tUU being the slack variables corresponding to
      variables with an upper bound only, and tLR being the slack variables
      corresponding to the 'upper' side of two-sided bounds.

    This framework initializes the slack variables sL, sU, tL, and tU to
    zero by default.

    Note that the slack framework does not update all members of AmplModel,
    such as the index set of constraints with an upper bound, etc., but
    rather performs the evaluations of the constraints for the updated
    model implicitly.
    """

    def __init__(self, model, **kwargs):

        AmplModel.__init__(self, model, **kwargs)

        # Save number of variables and constraints prior to transformation
        self.original_n = self.n
        self.original_m = self.m
        self.original_nbounds = self.nbounds
        
        # Number of slacks for inequality constraints with a lower bound
        n_con_low = self.nlowerC + self.nrangeC ; self.n_con_low = n_con_low

        # Number of slacks for inequality constraints with an upper bound
        n_con_up = self.nupperC + self.nrangeC ; self.n_con_up = n_con_up

        # Number of slacks for variables with a lower bound
        n_var_low = self.nlowerB + self.nrangeB ; self.n_var_low = n_var_low

        # Number of slacks for variables with an upper bound
        n_var_up = self.nupperB + self.nrangeB ; self.n_var_up = n_var_up

        # Update effective number of variables and constraints
        self.n  = self.original_n + n_con_low + n_con_up + n_var_low + n_var_up
        self.m  = self.original_m + self.nrangeC + n_var_low + n_var_up

        # Redefine primal and dual initial guesses
        self.original_x0 = self.x0[:]
        self.x0 = numpy.zeros(self.n)
        self.x0[:self.original_n] = self.original_x0[:]

        self.original_pi0 = self.pi0[:]
        self.pi0 = numpy.zeros(self.m)
        self.pi0[:self.original_m] = self.original_pi0[:]

        return

    def InitializeSlacks(self, val=0.0, **kwargs):
        """
        Initialize all slack variables to given value. This method may need to
        be overridden.
        """
        self.x0[self.original_n:] = val
        return


    def obj(self, x):
        return AmplModel.obj(self, x[:self.original_n])


    def cons(self, x):
        """
        Evaluate the vector of general constraints for the modified problem.
        Constraints are stored in the order in which they appear in the
        original problem. If constraint i is a range constraint, c[i] will
        be the constraint that has the slack on the lower bound on c[i].
        The constraint with the slack on the upper bound on c[i] will be stored
        in position m + k, where k is the position of index i in
        rangeC, i.e., k=0 iff constraint i is the range constraint that
        appears first, k=1 iff it appears second, etc.

        Constraints appear in the following order:

        [ c  ]   general constraints in origninal order
        [ cR ]   'upper' side of range constraints
        [ b  ]   linear constraints corresponding to bounds on original problem
        [ bR ]   linear constraints corresponding to 'upper' side of two-sided
                 bounds
        """
        m = self.m
        equalC = self.equalC
        lowerC = self.lowerC ; nlowerC = self.nlowerC
        upperC = self.upperC ; nupperC = self.nupperC
        rangeC = self.rangeC ; nrangeC = self.nrangeC

        mslow = self.original_n + self.nrangeC + self.n_con_low
        msup  = mslow + self.n_con_up
        s_low = x[self.original_n + self.nrangeC:mslow]
        s_up  = x[mslow:msup]

        c = numpy.empty(m)
        c[:self.original_m] = AmplModel.cons(self, x[:self.original_n])
        c[self.original_m:self.original_m + self.nrangeC] = c[rangeC]

        c[equalC] -= self.Lcon[equalC]
        c[lowerC] -= self.Lcon[lowerC] ; c[lowerC] -= s_low[:nlowerC]

        c[upperC] -= self.Ucon[upperC] ; c[upperC] *= -1
        c[upperC] -= s_up[:nupperC]

        c[rangeC] -= self.Lcon[rangeC] ; c[rangeC] -= s_low[nlowerC:]

        c[self.original_m:self.original_m+self.nrangeC] -= self.Ucon[rangeC]
        c[self.original_m:self.original_m+self.nrangeC] *= -1
        c[self.original_m:self.original_m+self.nrangeC] -= s_up[nupperC:]

        # Add linear constraints corresponding to bounds on original problem
        lowerB = self.lowerB ; nlowerB = self.nlowerB ; Lvar = self.Lvar
        upperB = self.upperB ; nupperB = self.nupperB ; Uvar = self.Uvar
        rangeB = self.rangeB ; nrangeB = self.nrangeB

        nt = self.original_n + self.n_con_low + self.n_con_up
        ntlow = nt + self.n_var_low
        t_low = x[nt:ntlow]
        t_up  = x[ntlow:]

        b = c[self.original_m+self.nrangeC:]

        b[:nlowerB] = x[lowerB] - Lvar[lowerB] - t_low[:nlowerB]
        b[nlowerB:nlowerB+nrangeB] = x[rangeB] - Lvar[rangeB] - t_low[nlowerB:]
        b[nlowerB+nrangeB:nlowerB+nrangeB+nupperB] = Uvar[upperB] - x[upperB] - t_up[:nupperB]
        b[nlowerB+nrangeB+nupperB:] = Uvar[rangeB] - x[rangeB] - t_up[nupperB:]

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

        t_low  = x[msup:mtlow]
        t_up   = x[mtlow:]

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

    def _jac(self, x, lp=False):
        """
        Helper method to assemble the Jacobian matrix of the constraints of the
        transformed problems. See the documentation of :meth:`jac` for more
        information.

        The positional argument `lp` should be set to `True` only if the problem
        is known to be a linear program. In this case, the evaluation of the
        constraint matrix is cheaper and the argument `x` is ignored.
        """
        n = self.original_n
        m = self.original_m

        # List() simply allows operations such as 1 + [2,3] -> [3,4]
        lowerC = List(self.lowerC) ; nlowerC = self.nlowerC
        upperC = List(self.upperC) ; nupperC = self.nupperC
        rangeC = List(self.rangeC) ; nrangeC = self.nrangeC
        lowerB = List(self.lowerB) ; nlowerB = self.nlowerB
        upperB = List(self.upperB) ; nupperB = self.nupperB
        rangeB = List(self.rangeB) ; nrangeB = self.nrangeB
        nbnds  = nlowerB + nupperB + 2*nrangeB
        nSlacks = nlowerC + nupperC + 2*nrangeC

        #print '%d variables and %d constraints' % (self.n, self.m)
        #print '%d slack variables' % nSlacks
        #print 'nlowerC = ', nlowerC
        #print 'nupperC = ', nupperC
        #print 'nrangeC = ', nrangeC
        #print 'nlowerB = ', nlowerB
        #print 'nupperB = ', nupperB
        #print 'nrangeB = ', nrangeB

        # Initialize sparse Jacobian
        nnzJ = 2 * self.nnzj + m + nrangeC + nbnds + nrangeB  # Overestimate
        J = sp(nrow=self.m, ncol=self.n, sizeHint=nnzJ)
        #J = spmatrix.ll_mat(self.m, self.n, nnzJ)

        # Insert contribution of general constraints
        if lp:
            J[:m,:n] = AmplModel.A(self)
        else:
            J[:m,:n] = AmplModel.jac(self,x[:n])
        J[upperC,:n] *= -1.0               # Flip sign of 'upper' gradients
        #J[upperC,:n].scale(-1.0)
        J[m:m+nrangeC,:n] = J[rangeC,:n]  # Append 'upper' side of range const.
        #J[m:m+nrangeC,:n].scale(-1.0)
        J[m:m+nrangeC,:n] *= -1.0        # Flip sign of 'upper' range gradients.

        # Create a few index lists
        rlowerC = List(range(nlowerC)) ; rlowerB = List(range(nlowerB))
        rupperC = List(range(nupperC)) ; rupperB = List(range(nupperB))
        rrangeC = List(range(nrangeC)) ; rrangeB = List(range(nrangeB))

        # Insert contribution of slacks on general constraints
        J.put(-1.0,      lowerC, n + rlowerC)
        J.put(-1.0,      upperC, n + nlowerC + rupperC)
        J.put(-1.0,      rangeC, n + nlowerC + nupperC + rrangeC)
        J.put(-1.0, m + rrangeC, n + nlowerC + nupperC + nrangeC + rrangeC)

        # Insert contribution of bound constraints on the original problem
        bot  = m+nrangeC ; J.put( 1.0, bot + rlowerB, lowerB)
        bot += nlowerB   ; J.put( 1.0, bot + rrangeB, rangeB)
        bot += nrangeB   ; J.put(-1.0, bot + rupperB, upperB)
        bot += nupperB   ; J.put(-1.0, bot + rrangeB, rangeB)

        # Insert contribution of slacks on the bound constraints
        bot  = m+nrangeC
        J.put(-1.0, bot + rlowerB, n + nSlacks + rlowerB)
        bot += nlowerB
        J.put(-1.0, bot + rrangeB, n + nSlacks + nlowerB + rrangeB)
        bot += nrangeB
        J.put(-1.0, bot + rupperB, n + nSlacks + nlowerB + nrangeB + rupperB)
        bot += nupperB
        J.put(-1.0, bot + rrangeB, n+nSlacks+nlowerB+nrangeB+nupperB+rrangeB)

        return J

    def jac(self, x):
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

        .. math::

           \begin{bmatrix}
              J   & -I &    &    &    \\
             -J_R &    & -I &    &    \\
              I   &    &    & -I &    \\
             -I   &    &    &    & -I
           \end{bmatrix}

        where the columns correspond, in order, to the variables `x`, `s`, `sU`,
        `t`, and `tU`, and the rows correspond, in order, to

        1. general constraints (in natural order)
        2. 'upper' side of range constraints
        3. bounds, ordered as explained above
        4. 'upper' side of two-sided bounds,

        and where the signs corresponding to 'upper' constraints and upper
        bounds are flipped in the (1,1) and (3,1) blocks.
        """
        return self._jac(x, lp=False)

    def A(self):
        """
        Return the constraint matrix if the problem is a linear program. See the
        documentation of :meth:`jac` for more information.
        """
        return self._jac(0, lp=True)

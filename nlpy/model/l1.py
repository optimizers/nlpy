from nlpy.model import NLPModel
from pysparse.sparse import spmatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp


class L1MeritFunction(NLPModel):
    u"""
    l1merit = L1MeritFunction(problem)
    Instantiate an L1-penalty/elastic framework as described in [GOT03]_.

    As part of the embedding, the bound constraints are reformulated as follows

        xL ≤ x ≤ xU      becomes       x  - xL + tR ≥ 0,
                                       xU - x  + tR ≥ 0  and  tR ≥ 0
        xL ≤ x           becomes       x  - xL + tL ≥ 0  and  tL ≥ 0
             x ≤ xU      becomes       xU - x  + tU ≥ 0  and  tU ≥ 0.

    The counters nrangeB, nlowerB, nupperB countain the number of bounds
    of the first, second and third kind respectively. To those are
    appended the number of fixed and free variables: nfixedB and nfreeB.

    Similarly, constraints are reformulated as follows

        cL ≤ c(x) ≤ cU   becomes       c(x) - cL   + sR ≥ 0,
                                       cU   - c(x) + sR ≥ 0  and  sR ≥ 0
        cL ≤ c(x)        becomes       c(x) - cL   + sL ≥ 0  and  sL ≥ 0
             c(x) ≤ cU   becomes       cU   - c(x) + sU ≥ 0  and  sU ≥ 0.

    The counters nrangeC, nlowerC and nupperC contain the number of
    constraints of the first, second and third kind respectively. To those
    are appended the number of equality and 'free' constraints (if any):
    nequalC and nfreeC.

    References
    ----------

    .. [GOT03]  N. I. M. Gould, D. Orban and Ph. L. Toint, *An Interior-Point
                l1-Penalty Method for Nonlinear Optimization*, Technical Report
                RAL-TR-2003-022, Rutherford Appleton Laboratory, Chilton,
                Oxfordshire, UK, 2003.

    .. [CO09] Z. Coulibaly and D. Orban, *An L1 Elastic Interior-Point Method
              for Mathematical Programs with Complementarity Constraints*,
              GERAD Technical Report G-2009-74, Montreal, Canada, 2009.
    """

    def __init__(self, nlp, nuE=1.0, nuS=1.0, nuT=1.0, **kwargs):
        """
        An ``L1MeritFunction`` object represents a nonlinear optimization
        problem in expanded elastic form.

        :parameters:
           :nlp:  Original NLP

        :keywords:
           :nuE: Initial penalty parameter for equality constraints
           :nuS:  Initial penalty parameter for elastic variables corresponding
                  to general constraints
           :nuT:  Initial penalty parameter for elastic variables corresponding
                  to bounds.
        """

        nB   = nlp.nlowerB + nlp.nupperB + nlp.nrangeB
        nB2  = nlp.nlowerB + nlp.nupperB + 2*nlp.nrangeB
        nvar = nlp.n + nlp.m + nB
        ncon = nlp.m + nlp.nrangeC + nB2

        # Lower bounds on the variables of the l1 problem.
        Lvar = np.empty(nvar)
        Lvar[:nlp.n] = -np.inf
        Lvar[nlp.n:] = 0

        NLPModel.__init__(self, n=nvar, m=ncon,
                          name=nlp.name + ' (l1)', Lvar=Lvar,
                          Lcon=np.zeros(ncon), **kwargs)

        self.nlp = nlp

        # Indices of bounded variables.
        self.Bounds = nlp.lowerB + nlp.upperB + nlp.rangeB

        # Maintain counters for effective number of bounds.
        self.nBounds  = nB
        self.nBounds2 = nB2

        eqC  = nlp.equalC  ; lC   = nlp.lowerC
        uC   = nlp.upperC  ; rC   = nlp.rangeC
        neqC = nlp.nequalC ; nlC  = nlp.nlowerC
        nuC  = nlp.nupperC ; nrC  = nlp.nrangeC
        Lvar = nlp.Lvar    ; Uvar = nlp.Uvar
        Lcon = nlp.Lcon    ; Ucon = nlp.Ucon

        # Initial point.
        n = nlp.n ; m = nlp.m ; nB = self.nBounds
        self.x0 = np.zeros(n + m + nB)
        self.x0[:n] = nlp.x0[:]
        x0 = self.x0[:n]

        # Stopping tolerances.
        self.stop_p = 1.0e-6      # Primal feasibility.
        self.stop_d = 1.0e-6      # Dual feasibility.
        self.stop_c = 1.0e-6      # Complementarity.

        # Total number of constraints (range constraints count as two).
        self.nConst = neqC + nlC + nuC + 2 * nrC

        # Initial penalty parameters.
        self._nuE = nuE
        self._nuS = nuS
        self._nuT = nuT

        # Constant that determines initial elastics: s = max(0,-c) + ethresh
        self.ethresh = 0.1

        # Set initial elastics so (x0, s0) is strictly feasible.
        # Note that a single elastic suffices for a range constraint.
        # s is ordered exactly like c.
        c = self.cons_pos(x0)
        self.s = self.x0[n:n+m]
        self.s[eqC] = np.maximum(0.0, -c[eqC])
        self.s[lC + uC] = np.maximum(0.0, -c[lC + uC])
        self.s[rC] = np.maximum(0.0, -c[m:])
        self.s[rC] = np.maximum(self.s[rC], -c[rC])
        self.s += self.ethresh

        # Shortcuts
        lB  = nlp.lowerB  ; uB  = nlp.upperB  ; rB  = nlp.rangeB
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB

        # Set initial elastics for the bound constraints so (x0, t0)
        # strictly satisfies the bounds. Single elastic for 2-sided bounds.
        # t = [ lowerB | upperB | rangeB ].
        self.t = self.x0[n+m:]
        self.t[:nlB] = np.maximum(0.0, Lvar[lB]-x0[lB])
        self.t[nlB:nlB+nuB] = np.maximum(0.0, x0[uB] - Uvar[uB])

        self.t[nlB+nuB:nB] = np.maximum(0.0, Lvar[rB]-x0[rB])
        self.t[nlB+nuB:nB] = np.maximum(self.t[nlB+nuB:nB], x0[rB]-Uvar[rB])
        self.t += self.ethresh

        return

    @property
    def nuE(self):
        return self._nuE

    @nuE.setter
    def nuE(self, value):
        self._nuE = value

    @property
    def nuS(self):
        return self._nuS

    @nuS.setter
    def nuS(self, value):
        self._nuS = value

    @property
    def nuT(self):
        return self._nuT

    @nuT.setter
    def nuT(self, value):
        self._nuT = value

    def get_penalty_parameters(self):
        return (self.nuE, self.nuS, self.nuT)

    def set_penalty_parameters(self, nuE, nuS, nuT):
        self.nuE = nuE
        self.nuS = nuS
        self.nuT = nuT
        return

    def get_xst(self, xst):
        "Split vector xst into x, s and t subvectors."
        nlp = self.nlp ; n = nlp.n ; m = nlp.m
        x = xst[:n] ; s = xst[n:n+m] ; t = xst[n+m:]
        return (x, s, t)

    def get_yzuv(self, yzuv):
        "Split vector of multipliers yzuv into y, z, u and v subvectors."
        nlp = self.nlp ; n = nlp.n ; m = nlp.m
        nrC = nlp.nrangeC ; nB2 = self.nBounds2
        y = yzuv[:m+nrC] ; z = yzuv[m+nrC:m+nrC+nB2]
        u = yzuv[m+nrC+nB2:m+nrC+nB2+m] ; v = yzuv[m+nrC+nB2+m:]
        return (y, z, u, v)

    def shifted_multipliers(self, y):
        "(yE, yL, yU, yRL, yRU) -> (yE-nuE, yL, yU, yRL, yRU)"

        nlp = self.nlp ; eC = nlp.equalC
        (nuE, nuS, nuT) = self.get_penalty_parameters()

        y_nlp = y.copy()
        y_nlp[eC] -= nuE

        return y_nlp

    def nlp_multipliers(self, y, shift=True):
        "(yE, yL, yU, yRL, yRU) -> (yE-nuE, yL, -yU, yRL-yRU)"

        nlp = self.nlp ; m = nlp.m
        eC = nlp.equalC ; uC = nlp.upperC ; rC = nlp.rangeC
        (nuE, nuS, nuT) = self.get_penalty_parameters()

        y_nlp = y[:m].copy()
        if shift: y_nlp[eC] -= nuE
        y_nlp[uC] *= -1
        y_nlp[rC] -= y[m:]

        return y_nlp

    def obj(self, xst, f=None, c=None):
        """
        Evaluate the elastic L1 penalty function at (x,s,t).

        :parameters:
            :xst:  vector of primal variables [x, s, t]

        :keywords:
            :f: Objective function value if available.
            :c: Vector of constraints if available.
        """

        # Shortcuts.
        nlp = self.nlp
        eqC = nlp.equalC ; lC = nlp.lowerC
        uC = nlp.upperC ; rC = nlp.rangeC

        (x,s,t) = self.get_xst(xst)

        p = nlp.obj(x) if f is None else f
        if c is None: c = self.cons_pos(x)

        # Add contribution from ...
        p += self.nuE * np.sum(c[eqC] + 2*s[eqC])  # ... equalities
        p += self.nuS * np.sum(s[lC + uC + rC])    # ... inequalities
        p += self.nuT * np.sum(t)                  # ... bounds

        return p

    def cons_pos(self, x, c=None):
        """
        Convenience function to return the vector of constraints of the
        original NLP reformulated as

            ci(x) - ai  = 0  for i in equalC
            ci(x) - Li >= 0  for i in lowerC + rangeC
            Ui - ci(x) >= 0  for i in upperC + rangeC.

        The constraints appear in natural order, except for the fact that the
        'upper side' of range constraints is appended to the end of the list.

        No need to apply scaling, since this is already done in cons().

        :parameters:
            :x:  vector of primal variables x (variables of original problem)

        :keywords:
            :c:  constraint vector of original problem, if available.
        """

        # Shortcuts.
        nlp = self.nlp
        n = nlp.n ; m = nlp.m ; nB = self.nBounds ; nB2 = self.nBounds2
        eqC = nlp.equalC ; lC = nlp.lowerC ; uC = nlp.upperC
        rC = nlp.rangeC ; nrC = nlp.nrangeC
        Lcon = nlp.Lcon ; Ucon = nlp.Ucon

        ec = np.empty(self.m)  # Reformulated constraints.

        # General constraints.
        ec[:m] = nlp.cons(x) if c is None else c[:]
        ec[m:m+nrC] = ec[rC]
        ec[eqC] -= Lcon[eqC]
        ec[lC]  -= Lcon[lC]
        ec[uC]  -= Ucon[uC] ; ec[uC] *= -1
        ec[rC]  -= Lcon[rC]
        ec[m:m+nrC]  -= Ucon[rC] ; ec[m:m+nrC] *= -1

        return ec

    def cons(self, xst, c=None):
        u"""
        Return the vector of constraints of the L1 elastic problem based on
        the constraints of the original NLP:

             c(x) = cE   becomes       c(x) - cE   + sE ≥ 0  and  sE ≥ 0
        cL ≤ c(x)        becomes       c(x) - cL   + sL ≥ 0  and  sL ≥ 0
             c(x) ≤ cU   becomes       cU   - c(x) + sU ≥ 0  and  sU ≥ 0
        cL ≤ c(x) ≤ cU   becomes       c(x) - cL   + sR ≥ 0,
                                       cU   - c(x) + sR ≥ 0  and  sR ≥ 0,

        xL ≤ x           becomes       x  - xL + tL ≥ 0  and  tL ≥ 0
             x ≤ xU      becomes       xU - x  + tU ≥ 0  and  tU ≥ 0
        xL ≤ x ≤ xU      becomes       x  - xL + tR ≥ 0,
                                       xU - x  + tR ≥ 0  and  tR ≥ 0.

        The vector of constraints does not include the bounds (s,t) ≥ 0.

        :parameters:
            :xst:  vector of primal variables [x, s, t]

        :keywords:
            :c:  constraint vector of original problem, if available.
        """

        # Shortcuts.
        nlp = self.nlp
        n = nlp.n ; m = nlp.m ; nB = self.nBounds ; nB2 = self.nBounds2
        eqC = nlp.equalC ; lC = nlp.lowerC ; uC = nlp.upperC
        rC = nlp.rangeC ; nrC = nlp.nrangeC
        Lcon = nlp.Lcon ; Ucon = nlp.Ucon
        lB = nlp.lowerB ; nlB = nlp.nlowerB
        uB = nlp.upperB ; nuB = nlp.nupperB
        rB = nlp.rangeB ; nrB = nlp.nrangeB
        Lvar = nlp.Lvar ; Uvar = nlp.Uvar

        (x,s,t) = self.get_xst(xst)

        ec = np.empty(m + nrC + nB2)  # Elastic constraints.

        # General constraints.
        ec[:m] = nlp.cons(x) if c is None else c[:]
        ec[m:m+nrC] = ec[rC]
        ec[eqC] -= Lcon[eqC]
        ec[lC]  -= Lcon[lC]
        ec[uC]  -= Ucon[uC] ; ec[uC] *= -1
        ec[rC]  -= Lcon[rC]
        ec[m:m+nrC]  -= Ucon[rC] ; ec[m:m+nrC] *= -1
        ec[:m] += s
        ec[m:m+nrC] += s[rC]

        # Former bounds that became linear inequalities.
        ec2 = ec[m+nrC:]
        ec2[:nlB] = x[lB] - Lvar[lB]
        ec2[nlB:nlB+nuB] = Uvar[uB] - x[uB]
        ec2[nlB+nuB:nB] = x[rB] - Lvar[rB]
        ec2[nB:] = Uvar[rB] - x[rB]
        ec2[:nB] += t
        ec2[nB:] += t[nlB+nuB:]

        return ec

    def grad(self, xst, g=None):
        """
        Return the gradient vector of the L1 merit function.

        :parameters:
            :xst:  vector of primal variables [x, s, t].

        :keywords:
            :g:  gradient vector of the objective function, if available.
        """

        # Shortcuts.
        nlp = self.nlp ; n = nlp.n ; m = nlp.m
        eqC = nlp.equalC ; neqC = nlp.nequalC
        lC = nlp.lowerC ; uC = nlp.upperC ; rC = nlp.rangeC
        nlB = nlp.nlowerB ; nuB = nlp.nupperB ; nrB = nlp.nrangeB
        nB = self.nBounds
        (x,s,t) = self.get_xst(xst)

        grad = np.empty(self.n)

        # Assemble x-part of gradient.
        grad[:n] = nlp.grad(x) if g is None else g[:]
        if neqC > 0:
            _JE = nlp.jac(x)[eqC,:]
            JE = PysparseLinearOperator(_JE, symmetric=False)
            eE = np.ones(neqC)
            grad[:n] += self.nuE * (JE.T * eE)

        # Assemble s-part of gradient.
        grads = grad[n:n+m]
        grads[lC+uC+rC] = self.nuS
        grads[eqC] = 2*self.nuE

        # Assemble t-part of gradient.
        gradt = grad[n+m:]
        gradt[:] = self.nuT * np.ones(nB)

        return grad

    def jac_pos(self, x, **kwargs):
        u"""
        Convenience function to evaluate the Jacobian matrix of the constraints
        of the original NLP reformulated as

            ci(x) = ai      for i in equalC
            ci(x) - Li ≥ 0  for i in lowerC
            ci(x) - Li ≥ 0  for i in rangeC
            Ui - ci(x) ≥ 0  for i in upperC
            Ui - ci(x) ≥ 0  for i in rangeC.

        The gradients of the general constraints appear in
        'natural' order, i.e., in the order in which they appear in the problem.
        The gradients of range constraints appear in two places: first in the
        'natural' location and again after all other general constraints, with a
        flipped sign to account for the upper bound on those constraints.

        The overall Jacobian of the new constraints thus has the form

        [ J ]
        [-JR]

        This is a `m + nrangeC` by `n` matrix, where `J` is the Jacobian of the
        general constraints in the order above in which the sign of the 'less
        than' constraints is flipped, and `JR` is the Jacobian of the 'less
        than' side of range constraints.
        """
        store_zeros = kwargs.get('store_zeros', False)
        store_zeros = 1 if store_zeros else 0
        nlp = self.nlp
        n = nlp.n ; m = nlp.m ; nrC = nlp.nrangeC
        uC = nlp.upperC ; rC = nlp.rangeC

        # Initialize sparse Jacobian
        J = sp(nrow=m + nrC, ncol=n, sizeHint=nlp.nnzj+10*nrC,
               storeZeros=store_zeros)

        # Insert contribution of general constraints
        J[:m,:n]  = nlp.jac(x, store_zeros=store_zeros)
        J[uC,:n] *= -1                 # Flip sign of 'upper' gradients
        J[m:,:n]  = -J[rC,:n]          # Append 'upper' side of range const.
        return J

    def jac(self, xst, J=None, **kwargs):
        """
        Return the constraint Jacobian of the L1 merit function problem.

        :parameters:
            :xst:  vector of primal variables [x, s, t].

        :keywords:
            :J:  constraint Jacobian of original problem, if available.
                 If supplied, `J` must conform to the output of the `jac_pos()`
                 method.
        """

        def Range(*args):
            return np.arange(*args, dtype=np.int)

        # Shortcuts.
        nlp = self.nlp ; n = nlp.n ; m = nlp.m
        eqC = nlp.equalC   ; neqC = nlp.nequalC
        lC  = nlp.lowerC   ; nlC  = nlp.nlowerC
        uC  = nlp.upperC   ; nuC  = nlp.nupperC
        rC  = nlp.rangeC   ; nrC  = nlp.nrangeC
        lB  = nlp.lowerB   ; nlB  = nlp.nlowerB
        uB  = nlp.upperB   ; nuB  = nlp.nupperB
        rB  = nlp.rangeB   ; nrB  = nlp.nrangeB
        nB  = self.nBounds ; nB2  = self.nBounds2
        (x,s,t) = self.get_xst(xst)

        # We order constraints and variables as follows:
        #
        #                     x   s   t
        #  l ≤  c(x) ≤ u  [  J   I     ]   } m
        # (l ≤) c(x) ≤ u  [ -JR  I     ]   } nrC
        #  l ≤  x         [  I       I ]   ^
        #       x  ≤ u    [ -I       I ]   | nB2
        #  l ≤  x (≤ u)   [  I       I ]   |
        # (l ≤) x  ≤ u    [ -I       I ]   v
        #
        #                    n   m   nB

        Jp = sp(nrow=m+nrC+nB2, ncol=n+m+nB, symmetric=False,
                sizeHint=nlp.nnzj+10*nrC+2*m+2*nB+nB2)

        # Contributions from original problem variables.
        r_lB = Range(nlB) ; r_uB = Range(nuB) ; r_rB = Range(nrB)
        Jp[:m+nrC,:n] = self.jac_pos(x) if J is None else J[:,:]
        Jp.put( 1, m + nrC + r_lB, lB)              #  l ≤  x.
        Jp.put(-1, m + nrC + nlB + r_uB, uB)        #       x  ≤ u.
        Jp.put( 1, m + nrC + nlB + nuB + r_rB, rB)  #  l ≤  x (≤ u).
        Jp.put(-1, m + nrC + nB + r_rB, rB)         # (l ≤) x  ≤ u.

        # Contributions from elastics on original bound constraints.
        Jp.put(1, m + nrC + r_lB, n + m + r_lB)             #  xL + tL ≥  l.
        Jp.put(1, m + nrC + nlB + r_uB, n + m + nlB + r_uB) # -xU + tU ≥ -u.
        # xR + tR ≥ l  and  -xR + tR ≥ -u.
        Jp.put(1, m + nrC + nlB + nuB + r_rB, n + m + nlB + nuB + r_rB)
        Jp.put(1, m + nrC + nlB + nuB + nrB + r_rB, n + m + nlB + nuB + r_rB)

        # Note that in the Jacobian, the elastics are ordered exactly like the
        # constraints of the original problem.

        # Contributions from elastics for equality constraints.
        a_eqC = np.array(eqC)
        Jp.put(1, eqC, n + a_eqC)                     # cE(x) + sE ≥ 0.

        # Contributions from elastics for lower inequality constraints.
        #r_lC = Range(nlC)
        a_lC = np.array(lC)
        Jp.put(1, lC, n + a_lC)

        # Contributions from elastics for upper inequality constraints.
        a_uC = np.array(uC)
        Jp.put(1, uC, n + a_uC)

        # Contributions from elastics for range constraints.
        r_rC = Range(nrC)
        a_rC = np.array(rC)
        Jp.put(1, rC, n + a_rC)
        Jp.put(1, m + r_rC, n + a_rC)

        return Jp

    def igrad(self, i, xst):
        "Do not call this function. For derivative checking purposes only."
        J = self.jac(xst)
        gi = J[i,:].getNumpyArray()[0]
        return gi

    def hess(self, xst, yzuv=None, *args, **kwargs):
        """
        Evaluate the Hessian matrix of the Lagrangian associated to the L1
        merit function problem.

        :parameters:
            :xst:  vector of primal variables [x, s, t].
            :yzuv: vector of dual variables [y, z, u, v]. If `None`, the
                   Hessian of the objective will be evaluated.
        """

        # Shortcuts.
        nlp = self.nlp ; m = nlp.m ; nrC = nlp.nrangeC

        (x,s,t) = self.get_xst(xst)
        obj_weight = kwargs.get('obj_weight', 1.0)
        shift = (obj_weight != 0.0)
        if yzuv is not None:
            (y,z,u,v) = self.get_yzuv(yzuv)
        else:
            y = np.zeros(m+nrC)
        y2 = self.nlp_multipliers(y, shift=shift)

        H = sp(nrow=self.n, ncol=self.n, symmetric=True, sizeHint=nlp.nnzh)
        H[:nlp.n,:nlp.n] = nlp.hess(x, y2, **kwargs)
        return H

    def bounds(self, xst):
        "Return the vector of bound constraints."
        n = self.nlp.n
        return xst[n:].copy()

    def primal_feasibility(self, xst, c=None):
        """
        Evaluate the primal feasibility residual at xst.

        :parameters:
            :xst:  vector of primal variables.

        :keywords:
            :c:    vector of constraints of l1 penalty problem, if available.
        """
        # Shortcuts.
        m = self.m ; nB = self.nBounds

        pFeas = np.empty(m+m+nB)
        pFeas[:m] = -self.cons(xst) if c is None else -c[:]
        pFeas[:m] = np.maximum(0, pFeas[:m])
        pFeas[m:] = -self.get_bounds(xst)
        pFeas[m:] = np.maximum(0, pFeas[m:])
        return pFeas

    def dual_feasibility(self, xst, yzuv, g=None, J=None):
        """
        Return the vector of dual feasibility for the interior/exterior merit
        function with specific values of the Lagrange multiplier estimates.

        :parameters:
            :xst:  Vector of primal variables
            :yzuv: Vector of multiplier estimates for perturbed general
                   constraints and bounds.

        :keywords:
            :g:  Gradient vector of the l1 objective function, if available
            :J:  Jacobian of the constraints of the l1 penalty problem as a
                 linear operator, if available.
        """

        # Shortcuts.
        nlp = self.nlp ; n = nlp.n

        if g is None: g = self.grad(xst)
        if J is None: J = self.jop(xst)

        yz = yzuv[:self.m]
        uv = yzuv[self.m:]

        # Contributions from...
        dFeas = g - J.T * yz  # ... general and linear constraints.
        dFeas[n:] -= uv       # ... bounds on s and t.
        return dFeas

    def complementarity(self, xst, yzuv, c=None):
        """
        Evaluate the complementarity residuals at (xst,yzuv). If `c` is
        specified, it should conform to :meth:`cons_pos` and the multipliers `y`
        should appear in the same order. The multipliers `z` should conform to
        :meth:`get_bounds`.

        :parameters:
            :xst:  vector of primal variables
            :yzuv: vector of Lagrange multiplier estimates for general
                   constraints and bounds.

        :keywords:
            :c:    vector of constraints of l1 penalty problem, if available.

        :returns:
            :csyz:  complementarity residual for general constraints
            :stuv:  complementarity residual for bound constraints.
        """
        # Shortcuts.
        nlp = self.nlp
        st = xst[nlp.n:]
        yz = yzuv[:self.m] ; uv = yzuv[self.m:]

        if c is None: c = self.cons(xst)
        csyz =  c*yz
        stuv = st*uv

        return (csyz, stuv)


class L1BarrierMeritFunction(NLPModel):
    u"""
    l1interior = L1BarrierMeritFunction(problem)
    Instantiate an L1BarrierMeritFunction object embedding the nonlinear
    program in the L1-penalty/elastic framework of [GOT03]_.

    The l1 barrier merit function problem with barrier parameter µ is:

        minimize φ(x;µ)
           x

    where

        φ(x;µ) := ψ(x,s;ν) - µ * ∑ log(ci(x) + si)
                           - µ * ∑ log(si)
                           - µ * ∑ log(xi + ti)
                           - µ * ∑ log(ti),

    and where ψ(x,s;ν) is the l1 penalty merit function.

    This is an unconstrained problem.

    References
    ----------

    .. [GOT03]  N. I. M. Gould, D. Orban and Ph. L. Toint, *An Interior-Point
                l1-Penalty Method for Nonlinear Optimization*, Technical Report
                RAL-TR-2003-022, Rutherford Appleton Laboratory, Chilton,
                Oxfordshire, UK, 2003.

    .. [CO09] Z. Coulibaly and D. Orban, *An L1 Elastic Interior-Point Method
              for Mathematical Programs with Complementarity Constraints*,
              GERAD Technical Report G-2009-74, Montreal, Canada, 2009.
    """

    def __init__(self, nlp, mu=5.0, **kwargs):


        nB  = nlp.nlowerB + nlp.nupperB + nlp.nrangeB
        nvar = nlp.n + nlp.m + nB

        NLPModel.__init__(self, n=nvar, m=0, name=nlp.name + ' (l1barrier)',
                          **kwargs)

        self.l1 = L1MeritFunction(nlp, **kwargs)
        self.x0 = self.l1.x0
        self._mu = mu
        return

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = max(0, value)

    def obj(self, xst, p=None, f=None, c=None):
        """
        Evaluate the barrier merit function.

        :parameters:
            :xst:  vector of primal variables [x, s, t].

        :keywords:
            :p: L1 merit function value if available
            :f: Original objective function value if available
            :c: Constraint vector if available.
        """

        # Shortcuts.
        l1 = self.l1
        Sum = np.sum ; Log = np.log

        if p is None: p = l1.obj(xst, f=f, c=c)
        if c is None: c = l1.cons(xst)
        (x, s, t) = l1.get_xst(xst)

        # The vector c includes constraints of the form c(x)+s and x+t.
        bar = Sum(Log(c)) + Sum(Log(s)) + Sum(Log(t))
        return p - self.mu * bar

    def primal_multipliers(self, xst, c=None, **kwargs):
        """
        Compute vector of primal Lagrange multiplier estimates at (x,s,t) with
        the current value of the barrier parameter.

        :parameters:
            :xst:  vector of primal variables [x, s, t].

        :keywords:
            :c:  Current constraints vector of l1 problem, if available.
        """

        # Shortcuts.
        l1 = self.l1
        mu = self.get_barrier_parameter()
        (x,s,t) = l1.get_xst(xst)

        if c is None: c = l1.cons(xst)
        return mu / np.concatenate((c, s, t))

    def grad(self, xst, g=None, J=None, **kwargs):
        """
        Evaluate the gradient of the barrier merit function.

        :parameters:
            :xst:  vector of primal variables [x, s, t].

        :keywords:
            :g:  Gradient vector of the objective function, if available
            :J:  Jacobian of the constraints, if available. Must conform to
                 ``jac_pos()``.
        """

        # The expression of the gradient is the same as that of dual
        # feasibility with the exception that primal multipliers are used.
        yzuv = self.primal_multipliers(xst)
        return self.l1.dual_feasibility(xst, yzuv, g=g, J=J, **kwargs)

    def hess(self, xst, yzuv=None, c=None, J=None, H=None):
        """
        Return the primal-dual Hessian matrix of the interior/exterior merit
        function with specific values of the Lagrange multiplier estimates. If
        you require the exact (primal) Hessian, setting yzuv=None will result
        in using  the primal multipliers.

        :parameters:
            :xst:  Vector of primal variables
            :yzuv: Vector of multiplier estimates. If set to `None`, the
                   primal multipliers will be used, which effectively yields
                   the Hessian of the barrier objective.

        :keywords:
            :c:  Constraint vector of the l1 problem, if available.
            :J:  Jacobian of the constraints of the l1 problem, if available.
            :H:  Hessian of the Lagrangian of the l1 problem, or an estimate.
        """

        # Shortcuts
        l1 = self.l1 ; nlp = l1.nlp ; n = nlp.n

        if c is None: c = l1.cons(xst)
        if J is None: J = l1.jac(xst)

        st = xst[n:]
        if yzuv is None: yzuv = self.primal_multipliers(xst, c=c)
        yz = yzuv[:l1.m]
        uv = yzuv[l1.m:]

        # Hbar = H(xst) + J(xst)' C(xst)^{-1} YZ J(xst) + bits with u and v.
        Hbar = l1.hess(xst, yzuv) if H is None else H.copy()
        _JCYJ = spmatrix.symdot(J.matrix, yz / c)
        JCYJ = sp(matrix=_JCYJ)
        Hbar += JCYJ
        r1 = range(n, self.n)
        Hbar.addAt(uv / st, r1, r1)
        return Hbar

# nlp.py
# Define an abstract class to represent a general
# nonlinear optimization problem.
# D. Orban, 2004.
import numpy as np


class KKTresidual:
    """
    A generic class to package KKT residuals and corresponding scalings.
    """
    def __init__(self, dFeas, pFeas, bFeas, gComp, bComp, **kwargs):
        """
        :parameters:
            dFeas: dual feasibility residual
            pFeas: primal feasibility residual, taking into account
                   constraints that are not bound constraints,
            bFeas: primal feasibility with respect to bounds,
            gComp: complementarity residual with respect to constraints
                   that are not bound constraints,
            bComp: complementarity residual with respect to bounds.
        """
        self.dFeas = dFeas
        self.pFeas = pFeas
        self.bFeas = bFeas
        self.gComp = gComp
        self.bComp = bComp
        self._is_scaling = kwargs.get('is_scaling', False)
        if self._is_scaling:
            self.scaling = None
        else:
            if 'scaling' in kwargs:
                self.set_scaling(kwargs['scaling'])
            else:
                self.scaling = KKTresidual(1.0, 1.0, 1.0, 1.0, 1.0,
                                           is_scaling=True)
        return

    def set_scaling(self, scaling, **kwargs):
        "Assign scaling values. `scaling` must be a `KKTresidual` instance."
        if self._is_scaling:
            raise ValueError, 'instance represents scaling factors.'
        if not isinstance(scaling, KKTresidual):
            raise ValueError, 'scaling must be a KKTresidual instance.'
        self.scaling = scaling
        self.scaling._is_scaling = True
        return


class NLPModel:
    """
    Instances of class NLPModel represent an abstract nonlinear optimization
    problem. It features methods to evaluate the objective and constraint
    functions, and their derivatives. Instances of the general class do not
    do anything interesting; they must be subclassed and specialized.

    :parameters:

        :n:       number of variables (default: 0)
        :m:       number of general (non bound) constraints (default: 0)
        :name:    model name (default: 'Generic')

    :keywords:

        :x0:      initial point (default: all 0)
        :pi0:     vector of initial multipliers (default: all 0)
        :Lvar:    vector of lower bounds on the variables
                  (default: all -Infinity)
        :Uvar:    vector of upper bounds on the variables
                  (default: all +Infinity)
        :Lcon:    vector of lower bounds on the constraints
                  (default: all -Infinity)
        :Ucon:    vector of upper bounds on the constraints
                  (default: all +Infinity)

    Constraints are classified into 3 classes: linear, nonlinear and network.

    Indices of linear constraints are found in member :attr:`lin`
    (default: empty).

    Indices of nonlinear constraints are found in member :attr:`nln`
    (default: all).

    Indices of network constraints are found in member :attr:`net`
    (default: empty).

    If necessary, additional arguments may be passed in kwargs.
    """

    def __init__(self, n=0, m=0, name='Generic', **kwargs):

        self.n = n          # Number of variables
        self.m = m          # Number of general constraints
        self.name = name    # Problem name

        # Initialize local value for Infinity
        self.Infinity = 1e+20
        self.negInfinity = - self.Infinity
        self.zero = 0.0
        self.one = 1.0

        # Set initial point
        if 'x0' in kwargs.keys():
            self.x0 = kwargs['x0']
        else:
            self.x0 = np.zeros(self.n, 'd')

        # Set initial multipliers
        if 'pi0' in kwargs.keys():
            self.pi0 = kwargs['pi0']
        else:
            self.pi0 = np.zeros(self.m, 'd')

        # Set lower bounds on variables    Lvar[i] <= x[i]  i = 1,...,n
        if 'Lvar' in kwargs.keys():
            self.Lvar = kwargs['Lvar']
        else:
            self.Lvar = self.negInfinity * np.ones(self.n, 'd')

        # Set upper bounds on variables    x[i] <= Uvar[i]  i = 1,...,n
        if 'Uvar' in kwargs.keys():
            self.Uvar = kwargs['Uvar']
        else:
            self.Uvar = self.Infinity * np.ones(self.n, 'd')

        # Set lower bounds on constraints  Lcon[i] <= c[i]  i = 1,...,m
        if 'Lcon' in kwargs.keys():
            self.Lcon = kwargs['Lcon']
        else:
            self.Lcon = self.negInfinity * np.ones(self.m, 'd')

        # Set upper bounds on constraints  c[i] <= Ucon[i]  i = 1,...,m
        if 'Ucon' in kwargs.keys():
            self.Ucon = kwargs['Ucon']
        else:
            self.Ucon = self.Infinity * np.ones(self.m, 'd')

        # Default classification of constraints
        self.lin = []                        # Linear    constraints
        self.nln = range(self.m)             # Nonlinear constraints
        self.net = []                        # Network   constraints
        self.nlin = len(self.lin)            # Number of linear constraints
        self.nnln = len(self.nln)            # Number of nonlinear constraints
        self.nnet = len(self.net)            # Number of network constraints

        # Maintain lists of indices for each type of constraints:
        self.rangeC = []    # Range constraints:       cL <= c(x) <= cU
        self.lowerC = []    # Lower bound constraints: cL <= c(x)
        self.upperC = []    # Upper bound constraints:       c(x) <= cU
        self.equalC = []    # Equality constraints:    cL  = c(x)  = cU
        self.freeC  = []    # "Free" constraints:    -inf <= c(x) <= inf

        for i in range(self.m):
            if self.Lcon[i] > self.negInfinity and self.Ucon[i] < self.Infinity:
                if self.Lcon[i] == self.Ucon[i]:
                    self.equalC.append(i)
                else:
                    self.rangeC.append(i)
            elif self.Lcon[i] > self.negInfinity:
                self.lowerC.append(i)
            elif self.Ucon[i] < self.Infinity:
                self.upperC.append(i)
            else:
                self.freeC.append(i)

        self.nlowerC = len(self.lowerC)   # Number of lower bound constraints
        self.nrangeC = len(self.rangeC)   # Number of range constraints
        self.nupperC = len(self.upperC)   # Number of upper bound constraints
        self.nequalC = len(self.equalC)   # Number of equality constraints
        self.nfreeC  = len(self.freeC )   # The rest: should be 0

        # Proceed similarly with bound constraints
        self.rangeB = []
        self.lowerB = []
        self.upperB = []
        self.fixedB = []
        self.freeB  = []

        for i in range(self.n):
            if self.Lvar[i] > self.negInfinity and self.Uvar[i] < self.Infinity:
                if self.Lvar[i] == self.Uvar[i]:
                    self.fixedB.append(i)
                else:
                    self.rangeB.append(i)
            elif self.Lvar[i] > self.negInfinity:
                self.lowerB.append(i)
            elif self.Uvar[i] < self.Infinity:
                self.upperB.append(i)
            else:
                self.freeB.append(i)

        self.nlowerB = len(self.lowerB)
        self.nrangeB = len(self.rangeB)
        self.nupperB = len(self.upperB)
        self.nfixedB = len(self.fixedB)
        self.nfreeB  = len(self.freeB )
        self.nbounds = self.n - self.nfreeB

        # Define default stopping tolerances
        self.stop_d = 1.0e-6    # Dual feasibility
        self.stop_c = 1.0e-6    # Complementarty
        self.stop_p = 1.0e-6    # Primal feasibility

        # Initialize some counters
        self.feval = 0    # evaluations of objective  function
        self.geval = 0    #                           gradient
        self.Heval = 0    #                Lagrangian Hessian
        self.Hprod = 0    #                matrix-vector products with Hessian
        self.ceval = 0    #                constraint functions
        self.Jeval = 0    #                           gradients
        self.Jprod = 0    #                matrix-vector products with Jacobian

    def ResetCounters(self):
        """
        Reset the `feval`, `geval`, `Heval`, `Hprod`, `ceval`, `Jeval` and
        `Jprod` counters of the current instance to zero.
        """
        self.feval = 0
        self.geval = 0
        self.Heval = 0
        self.Hprod = 0
        self.ceval = 0
        self.Jeval = 0
        self.Jprod = 0
        return None

    # Evaluate optimality residuals
    def OptimalityResiduals(self, x, z, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Decide whether optimality is attained
    def AtOptimality(self, x, z, **kwargs):
        (d, c, p) = self.OptimalityResiduals(x, z, **kwargs)
        if d <= self.stop_d and c <= self.stop_c and p <= self.stop_p:
            return True
        return False

    # Evaluate objective function at x
    def obj(self, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate objective gradient at x
    def grad(self, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate vector of constraints at x
    def cons(self, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate i-th constraint at x
    def icons(self, i, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evalutate i-th constraint gradient at x
    # Gradient is returned as a dense vector
    def igrad(self, i, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate i-th constraint gradient at x
    # Gradient is returned as a sparse vector
    def sigrad(self, i, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate constraints Jacobian at x
    def jac(self, x, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate Lagrangian Hessian at (x,z)
    def hess(self, x, z, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate matrix-vector product between
    # the Hessian of the Lagrangian and a vector
    def hprod(self, x, z, p, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

    # Evaluate matrix-vector product between
    # the Hessian of the i-th constraint and a vector
    def hiprod(self, i, x, p, **kwargs):
        raise NotImplementedError, 'This method must be subclassed.'

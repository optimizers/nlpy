# nlp.py
# Define an abstract class to represent a general
# nonlinear optimization problem.
# D. Orban, 2004.
from pykrylov.linop import LinearOperator
import numpy as np


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
            raise ValueError('instance represents scaling factors.')
        if not isinstance(scaling, KKTresidual):
            raise ValueError('scaling must be a KKTresidual instance.')
        self.scaling = scaling
        self.scaling._is_scaling = True
        return


class NLPModel(object):
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

        self.nvar = self.n = n   # Number of variables
        self.ncon = self.m = m   # Number of general constraints
        self.name = name         # Problem name

        # Set initial point
        if 'x0' in kwargs.keys():
            self.x0 = np.ascontiguousarray(kwargs['x0'], dtype=float)
        else:
            self.x0 = np.zeros(self.n, 'd')

        # Set initial multipliers
        if 'pi0' in kwargs.keys():
            self.pi0 = np.ascontiguousarray(kwargs['pi0'], dtype=float)
        else:
            self.pi0 = np.zeros(self.m, 'd')

        # Set lower bounds on variables    Lvar[i] <= x[i]  i = 1,...,n
        if 'Lvar' in kwargs.keys():
            self.Lvar = np.ascontiguousarray(kwargs['Lvar'], dtype=float)
        else:
            self.Lvar = -np.inf * np.ones(self.n, 'd')

        # Set upper bounds on variables    x[i] <= Uvar[i]  i = 1,...,n
        if 'Uvar' in kwargs.keys():
            self.Uvar = np.ascontiguousarray(kwargs['Uvar'], dtype=float)
        else:
            self.Uvar = np.inf * np.ones(self.n, 'd')

        # Set lower bounds on constraints  Lcon[i] <= c[i]  i = 1,...,m
        if 'Lcon' in kwargs.keys():
            self.Lcon = np.ascontiguousarray(kwargs['Lcon'], dtype=float)
        else:
            self.Lcon = -np.inf * np.ones(self.m, 'd')

        # Set upper bounds on constraints  c[i] <= Ucon[i]  i = 1,...,m
        if 'Ucon' in kwargs.keys():
            self.Ucon = np.ascontiguousarray(kwargs['Ucon'], dtype=float)
        else:
            self.Ucon = np.inf * np.ones(self.m, 'd')

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
            if self.Lcon[i] > -np.inf and self.Ucon[i] < np.inf:
                if self.Lcon[i] == self.Ucon[i]:
                    self.equalC.append(i)
                else:
                    self.rangeC.append(i)
            elif self.Lcon[i] > -np.inf:
                self.lowerC.append(i)
            elif self.Ucon[i] < np.inf:
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
            if self.Lvar[i] > -np.inf and self.Uvar[i] < np.inf:
                if self.Lvar[i] == self.Uvar[i]:
                    self.fixedB.append(i)
                else:
                    self.rangeB.append(i)
            elif self.Lvar[i] > -np.inf:
                self.lowerB.append(i)
            elif self.Uvar[i] < np.inf:
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

        # Define scaling attributes.
        self.g_max = 1.0e2      # max gradient entry (constant)
        self.scale_obj = None   # Objective scaling
        self.scale_con = None   # Constraint scaling

        # Initialize some counters
        self.feval = 0    # evaluations of objective  function
        self.geval = 0    #                           gradient
        self.Heval = 0    #                Lagrangian Hessian
        self.Hprod = 0    #                matrix-vector products with Hessian
        self.ceval = 0    #                constraint functions
        self.Jeval = 0    #                           gradients
        self.Jprod = 0    #                Jacobian-vector products
        self.JTprod = 0   #                transpose Jacobian-vector products

    def ResetCounters(self):
        """
        Reset the eval counters to zero.
        """
        self.feval = 0
        self.geval = 0
        self.Heval = 0
        self.Hprod = 0
        self.ceval = 0
        self.Jeval = 0
        self.Jprod = 0
        self.JTprod = 0
        return None

    def get_stopping_tolerances(self):
        return (self.stop_d, self.stop_p, self.stop_c)

    def set_stopping_tolerances(self, stop_d, stop_p, stop_c):
        self.stop_d = stop_d
        self.stop_p = stop_p
        self.stop_c = stop_c
        return

    # Evaluate optimality residuals
    def OptimalityResiduals(self, x, z, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    # Decide whether optimality is attained
    def AtOptimality(self, x, z, **kwargs):
        (d, c, p) = self.OptimalityResiduals(x, z, **kwargs)
        if d <= self.stop_d and c <= self.stop_c and p <= self.stop_p:
            return True
        return False

    def compute_scaling_obj(self, x=None, g_max=1.0e2, reset=False):
        """Compute objective scaling."""
        raise NotImplementedError('This method must be subclassed.')

    def compute_scaling_cons(self, x=None, g_max=1.0e2, reset=False):
        """Compute constraint scaling."""
        raise NotImplementedError('This method must be subclassed.')

    # Evaluate objective function at x
    def obj(self, x, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    # Evaluate objective gradient at x
    def grad(self, x, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    # Evaluate vector of constraints at x
    def cons(self, x, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    # Evaluate i-th constraint at x
    def icons(self, i, x, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    # Evalutate i-th constraint gradient at x
    # Gradient is returned as a dense vector
    def igrad(self, i, x, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    # Evaluate i-th constraint gradient at x
    # Gradient is returned as a sparse vector
    def sigrad(self, i, x, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    # Evaluate constraints Jacobian at x
    def jac(self, x, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    # Evaluate Jacobian-vector product
    def jprod(self, x, p, **kwargs):
        raise NotImplementedError('This method must be subclassed')

    # Evaluate transposed-Jacobian-vector product
    def jtprod(self, x, p, **kwargs):
        raise NotImplementedError('This method must be subclassed')

    # Evaluate Lagrangian Hessian at (x,z)
    def hess(self, x, z=None, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    # Evaluate matrix-vector product between
    # the Hessian of the Lagrangian and a vector
    def hprod(self, x, z, p, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    # Evaluate matrix-vector product between
    # the Hessian of the i-th constraint and a vector
    def hiprod(self, i, x, p, **kwargs):
        raise NotImplementedError('This method must be subclassed.')

    def display_basic_info(self):
        """
        Display vital statistics about the current model.
        """
        import sys
        write = sys.stderr.write
        write('Problem Name: %s\n' % self.name)
        write('Number of Variables: %d\n' % self.n)
        write('Number of Bound Constraints: %d' % self.nbounds)
        write(' (%d lower, %d upper, %d two-sided)\n' %
              (self.nlowerB, self.nupperB, self.nrangeB))
        if self.nlowerB > 0: write('Lower bounds: %s\n' % self.lowerB)
        if self.nupperB > 0: write('Upper bounds: %s\n' % self.upperB)
        if self.nrangeB > 0: write('Two-Sided bounds: %s\n' % self.rangeB)
        write('Vector of lower bounds: %s\n' % self.Lvar)
        write('Vectof of upper bounds: %s\n' % self.Uvar)
        write('Number of General Constraints: %d' % self.m)
        write(' (%d equality, %d lower, %d upper, %d range)\n' %
              (self.nequalC, self.nlowerC, self.nupperC, self.nrangeC))
        if self.nequalC > 0: write('Equality: %s\n' % self.equalC)
        if self.nlowerC > 0: write('Lower   : %s\n' % self.lowerC)
        if self.nupperC > 0: write('Upper   : %s\n' % self.upperC)
        if self.nrangeC > 0: write('Range   : %s\n' % self.rangeC)
        write('Vector of constraint lower bounds: %s\n' % self.Lcon)
        write('Vector of constraint upper bounds: %s\n' % self.Ucon)
        write('Initial Guess: %s\n' % self.x0)

        return


class QPModel(NLPModel):
    """
    A generic class to represent a quadratic programming problem

    minimize    c'x + 1/2 x'Hx
    subject to  L <= A*x <= U
                l <=  x  <= u.
    """

    def __init__(self, c, H, A=None, name='GenericQP', **kwargs):
        """
        :parameters:
            :c:   Numpy array to represent the linear objective
            :A:   linear operator to represent the constraint matrix.
                  It must be possible to perform the operations `A*x`
                  and `A.T*y` for Numpy arrays `x` and `y` of appropriate size.
                  If `A` is `None`, it will be replaced with an empty linear
                  operator.
            :H:   linear operator to represent the objective Hessian.
                  It must be possible to perform the operation `H*x`
                  for a Numpy array `x` of appropriate size. The operator `H`
                  should be symmetric.

        See the documentation of `NLPModel` for futher information.
        """

        # Basic checks.
        n = c.shape[0]
        if A is None:
            m = 0
            self.A = LinearOperator(n, 0,
                                    lambda x: np.empty((0, 1)),
                                    matvec_transp=lambda y: np.empty((n, 0)),
                                    dtype=np.float)
        else:
            if A.shape[1] != n or H.shape[0] != n or H.shape[1] != n:
                raise ValueError('Shapes are inconsistent')
            m = A.shape[0]
            self.A = A

        super(QPModel, self).__init__(n=n, m=m, name=name, **kwargs)
        self.c = c
        self.H = H

        # Default classification of constraints
        self.lin = range(self.m)             # Linear    constraints
        self.nln = []                        # Nonlinear constraints
        self.net = []                        # Network   constraints
        self.nlin = len(self.lin)            # Number of linear constraints
        self.nnln = len(self.nln)            # Number of nonlinear constraints
        self.nnet = len(self.net)            # Number of network constraints

    def obj(self, x):
        cHx = self.hprod(x, 0, x)
        cHx *= 0.5
        cHx += self.c
        return np.dot(cHx, x)

    def grad(self, x):
        Hx = self.hprod(x, 0, x)
        Hx += self.c
        return Hx

    def cons(self, x):
        if isinstance(self.A, np.ndarray):
          return np.dot(self.A, x)
        return self.A * x

    def A(self, x):
        return self.A

    def jac(self, x):
        return self.A

    def jprod(self, x, p):
        return self.cons(p)

    def jtprod(self, x, p):
        if isinstance(self.A, np.ndarray):
          return np.dot(self.A.T, p)
        return self.A.T * p

    def hess(self, x, z):
        return self.H

    def hprod(self, x, z, p):
        if isinstance(self.H, np.ndarray):
          return np.dot(self.H, p)
        return self.H * p


class LPModel(QPModel):
    """
    A generic class to represent a linear programming problem

    minimize    c'x
    subject to  L <= A*x <= U
                l <=  x  <= u.
    """

    def __init__(self, c, A=None, name='GenericLP', **kwargs):
        """
        :parameters:
            :c:   Numpy array to represent the linear objective
            :A:   linear operator to represent the constraint matrix.
                  It must be possible to perform the operations `A*x`
                  and `A.T*y` for Numpy arrays `x` and `y` of appropriate size.

        See the documentation of `NLPModel` for futher information.
        """

        n = c.shape[0]
        H = LinearOperator(n, n,
                           lambda x: np.zeros(n),
                           symmetric=True,
                           dtype=np.float)
        super(LPModel, self).__init__(c, H, A, name=name, **kwargs)

    def obj(self, x):
        return np.dot(self.c, x)

    def grad(self, x):
        return self.c

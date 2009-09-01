from nlpy.optimize.ls.pymswolfe import StrongWolfeLineSearch
from nlpy.tools import norms
from nlpy.tools.timing import cputime
import numpy

class InverseLbfgs:
    """
    Class InverseLbfgs is a container used to store and manipulate
    limited-memory BFGS matrices. It may be used, e.g., in a LBFGS solver for
    unconstrained minimization or as a preconditioner. The limited-memory
    matrix that is implicitly stored is a positive definite approximation to
    the inverse Hessian. Therefore, search directions may be obtained by
    computing matrix-vector products only. Such products are efficiently
    computed by means of a two-loop recursion.

    Instantiation is as follows

        lbfgsupdate = LbfgsUpdate(n)

    where n is the number of variables of the problem.
    
    Optional keywords include

        npairs        the number of (s,y) pairs stored (default: 5)
        scaling       enable scaling of the 'initial matrix'. Scaling is
                      done as 'method M3' in the LBFGS paper by Zhou and
                      Nocedal; the scaling factor is <sk,yk>/<yk,yk>
                      (default: False).
        
    Member functions are

        store         to store a new (s,y) pair and discard the oldest one
                      in case the maximum storage has been reached,
        matvec        to compute a matrix-vector product between the current
                      positive-definite approximation to the inverse Hessian
                      and a given vector.
    """

    def __init__(self, n, npairs=5, **kwargs):

        # Mandatory arguments
        self.n = n
        self.npairs = npairs

        # Optional arguments
        self.scaling = kwargs.get('scaling', False)

        # Storage of the (s,y) pairs
        self.s = numpy.empty((self.n, self.npairs), 'd')
        self.y = numpy.empty((self.n, self.npairs), 'd')

        # Allocate two arrays once and for all:
        #  alpha contains the multipliers alpha[i]
        #  ys    contains the dot products <si,yi>
        # Only the relevant portion of each array is used
        # in the two-loop recursion.
        self.alpha = numpy.empty(self.npairs, 'd')
        self.ys = numpy.empty(self.npairs, 'd')
        self.gamma = 1.0

    def store(self, iter, new_s, new_y):
        """
        Store the new pair (new_s,new_y) computed at iteration iter.
        """
        position = iter % self.npairs  # Indices are zero-based
        self.s[:,position] = new_s.copy()
        self.y[:,position] = new_y.copy()
        return

    def matvec(self, iter, v):
        """
        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the inverse Hessian matrix and the
        vector v using the LBFGS two-loop recursion formula. The 'iter'
        argument is the current iteration number.
        
        When the inner product <y,s> of one of the pairs is nearly zero, the
        function returns the input vector v, i.e., no preconditioning occurs.
        In this case, a safeguarding step should probably be taken.
        """
        q = v
        for i in range(min(self.npairs, iter)):
            k = (iter-1-i) % self.npairs
            self.ys[k] = numpy.dot(self.y[:,k], self.s[:,k])
            if abs(self.ys[k]) < 1.0e-12: return v
            self.alpha[k] = numpy.dot(self.s[:,k], q)/self.ys[k]
            q -= self.alpha[k] * self.y[:,k]
            
        r = q
        if self.scaling and iter > 0:
            last = (iter-1) % self.npairs
            self.gamma = self.ys[last]/numpy.dot(self.y[:,last],self.y[:,last])
            r *= self.gamma
            
        for i in range(min(self.npairs-1,iter-1), -1, -1):
            k = (iter-1-i) % self.npairs
            beta = numpy.dot(self.y[:,k], r)/self.ys[k]
            r += (self.alpha[k] - beta) * self.s[:,k]
        return r

    def solve(self, iter, v):
        """
        This is an alias for matvec used for preconditioning.
        """
        return self.matvec(iter, v)


class LbfgsFramework:
    """
    Class LbfgsFramework provides a framework for solving unconstrained
    optimization problems by means of the limited-memory BFGS method.

    Instantiation is done by

        lbfgs = LbfgsFramework(nlp)

    where nlp is an instance of a nonlinear problem. A solution of the
    problem is obtained by called the solve member function, as in

        lbfgs.solve().

    Optional keyword arguments include

        npairs      the number of (s,y) pairs to store (default: 5)
        x0          the starting point (default: nlp.x0)
        maxiter     the maximum number of iterations (default: max(10n,1000))

    Other keyword arguments will be passed to LbfgsUpdate.
        
    The linesearch used in this version is Jorge Nocedal's modified More and
    Thuente linesearch, attempting to ensure satisfaction of the strong Wolfe
    conditions. The modifications attempt to limit the effects of rounding
    error inherent to the More and Thuente linesearch.
    """
    def __init__(self, nlp, **kwargs):

        self.nlp = nlp
        self.npairs = kwargs.get('npairs', 5)
        self.silent = kwargs.get('silent', False)
        self.iter   = 0
        self.nresets = 0
        self.converged = False
        
        self.lbfgs = InverseLbfgs(self.nlp.n, **kwargs)

        self.x = kwargs.get('x0', self.nlp.x0)
        self.f = self.nlp.obj(self.x)
        self.g = self.nlp.grad(self.x)
        self.gnorm = norms.norm2(self.g)
        self.f0 = self.f
        self.g0 = self.gnorm

        # Optional arguments
        self.maxiter = kwargs.get('maxiter', max(10*self.nlp.n, 1000))
        self.tsolve = 0.0

    def solve(self):

        tstart = cputime()

        # Initial LBFGS matrix is the identity. In other words,
        # the initial search direction is the steepest descent direction
        stoptol = self.nlp.stop_d * max(1.0, norms.norm2(self.x))

        while self.gnorm > stoptol and self.iter < self.maxiter:

            if not self.silent:
                print '%-5d  %-12g  %-12g' % (self.iter, self.f, self.gnorm)

            # Obtain search direction
            d = self.lbfgs.matvec(self.iter, -self.g)

            # Prepare for modified More-Thuente linesearch
            if self.iter == 0:
                stp0 = 1.0/self.gnorm
            else:
                stp0 = 1.0
            SWLS = StrongWolfeLineSearch(self.f,
                                         self.x,
                                         self.g,
                                         d,
                                         lambda z: self.nlp.obj(z),
                                         lambda z: self.nlp.grad(z),
                                         stp = stp0)
            # Perform linesearch
            SWLS.search()

            # SWLS.x  contains the new iterate
            # SWLS.g  contains the objective gradient at the new iterate
            # SWLS.f  contains the objective value at the new iterate
            s = SWLS.x - self.x
            self.x = SWLS.x
            y = SWLS.g - self.g
            self.g = SWLS.g
            self.gnorm = norms.norm2(self.g)
            self.f = SWLS.f
            stoptol = self.nlp.stop_d * max(1.0, norms.norm2(self.x))

            # Update inverse Hessian approximation using the most recent pair
            self.lbfgs.store(self.iter, s, y)
            self.iter += 1

        self.tsolve = cputime() - tstart
        self.converged = (self.iter < self.maxiter)

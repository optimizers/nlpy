# A General module that implements various linesearch schemes
# D. Orban, Montreal, Jan. 2005.

class LineSearch:
    """
    A generic linesearch class. Most methods of this
    class should be overridden by subclassing.
    """

    def __init__(self, **kwargs):
        self._id = 'Generic Linesearch'
        return

    def _test(self, func, x, d, slope, f = None, t = 1.0, **kwargs):
        """
        Given a descent direction d for function func at the
        current iterate x, see if the steplength t satisfies
        a specific linesearch condition. 

        Must be overridden.
        """
        return True # Must override

    def search(self, func, x, d, slope, f = None, **kwargs):
        """
        Given a descent direction d for function func at the
        current iterate x, compute a steplength t such that
        func(x + t * d) satisfies a linesearch condition
        when compared to func(x). The value of the argument
        slope should be the directional derivative of func in
        the direction d: slope = f'(x;d) < 0. If given, f should
        be the value of func(x). If not given, it will be evaluated.

        func can point to a defined function or be a lambda function.
        For example, in the univariate case:
            test(lambda x: x**2, 2.0, -1, 4.0)
        """
        # Must override
        if slope >= 0.0:
            raise ValueError, "Direction must be a descent direction"
            return None
        t = 1.0
        while not self._test(func, x, d, f = f, t = t, **kwargs):
            pass
        return t


class ArmijoLineSearch(LineSearch):
    """
    An Armijo linesearch with backtracking. This class implements the simple
    Armijo test

    f(x + t * d) <= f(x) + t * beta * f'(x;d)

    where 0 < beta < 1/2 and f'(x;d) is the directional derivative of f in the
    direction d. Note that f'(x;d) < 0 must be true.

    :keywords:

        :beta:      Value of beta. Default: 0.001
        :tfactor:   Amount by which to reduce the steplength
                    during the backtracking. Default: 0.5.
    """
    
    def __init__(self, **kwargs):
        LineSearch.__init__(self, **kwargs)
        self.beta = max(min(kwargs.get('beta', 1.0e-4), 0.5), 1.0e-10)
        self.tfactor = max(min(kwargs.get('tfactor', 0.1), 0.999), 1.0e-3)
        return

    def _test(self, func, x, d, slope, f = None, t = 1.0, **kwargs):
        """
        Given a descent direction d for function func at the
        current iterate x, see if the steplength t satisfies
        the Armijo linesearch condition. 
        """
        if f is None:
            f = func(x)
        f_plus = func(x + t * d)
        return (f_plus <= f + t * self.beta * slope)

    def search(self, func, x, d, slope, f = None, **kwargs):
        """
        Given a descent direction d for function func at the
        current iterate x, compute a steplength t such that
        func(x + t * d) satisfies the Armijo linesearch condition
        when compared to func(x). The value of the argument
        slope should be the directional derivative of func in
        the direction d: slope = f'(x;d) < 0. If given, f should
        be the value of func(x). If not given, it will be evaluated.

        func can point to a defined function or be a lambda function.
        For example, in the univariate case:

            `test(lambda x: x**2, 2.0, -1, 4.0)`
        """
        if f is None:
            f = func(x)
        if slope >= 0.0:
            raise ValueError, "Direction must be a descent direction"
            return None
        t = 1.0
        while not self._test(func, x, d, slope, f = f, t = t, **kwargs):
            t *= self.tfactor
        return t




# Test
if __name__ == '__main__':
    
    # Simple example:
    #    steepest descent method
    #    with Armijo backtracking
    from numpy import array, dot
    from nlpy.tools.norms import norm_infty
    
    def rosenbrock(x):
        return 10.0 * (x[1]-x[0]**2)**2 + (1-x[0])**2

    def rosenbrockxy(x,y):
        return rosenbrock((x,y))

    def rosengrad(x):
        return array([ -40.0 * (x[1] - x[0]**2) * x[0] - 2.0 * (1-x[0]),
                        20.0 * (x[1] - x[0]**2) ], 'd')

    ALS = ArmijoLineSearch(tfactor = 0.2)
    x = array([-0.5, 1.0], 'd')
    xmin = xmax = x[0]
    ymin = ymax = x[1]
    f = rosenbrock(x)
    grad = rosengrad(x)
    d = -grad
    slope = dot(grad, d)
    t = 0.0
    tlist = []
    xlist = [x[0]]
    ylist = [x[1]]
    iter = 0
    print '%-d\t%-g\t%-g\t%-g\t%-g\t%-g\t%-g' % (iter, f, norm_infty(grad), x[0], x[1], t, slope)
    while norm_infty(grad) > 1.0e-5:

        iter += 1

        # Perform linesearch
        t = ALS.search(rosenbrock, x, d, slope, f = f)
        tlist.append(t)

        # Move ahead
        x += t * d
        xlist.append(x[0])
        ylist.append(x[1])
        xmin = min(xmin, x[0])
        xmax = max(xmax, x[0])
        ymin = min(ymin, x[1])
        ymax = max(ymax, x[1])
        f = rosenbrock(x)
        grad = rosengrad(x)
        d = -grad
        slope = dot(grad, d)
        print '%-d\t%-g\t%-g\t%-g\t%-g\t%-g\t%-g' % (iter, f, norm_infty(grad), x[0], x[1], t, slope)

    try:
        from pylab import *
    except:
        import sys
        sys.stderr.write('If you had Matplotlib, you would be looking\n')
        sys.stderr.write('at a contour plot right now...\n')
        sys.exit(0)
    xx = arange(-1.5, 1.5, 0.01)
    yy = arange(-0.5, 1.5, 0.01)
    XX, YY = meshgrid(xx, yy)
    ZZ = rosenbrockxy(XX, YY)
    plot(xlist, ylist, 'r-', lw = 1.5)
    plot([xlist[0]], [ylist[0]], 'go', [xlist[-1]], [ylist[-1]], 'go')
    contour(XX, YY, ZZ, 30, linewidths = 1.5, alpha = 0.75, origin = 'lower')
    title('Steepest descent with Armijo linesearch on the Rosenbrock function')
    show()



"""
A pure Python/numpy implementation of the Steihaug-Toint
truncated preconditioned conjugate gradient algorithm.

$Id: trpcg.py 88 2008-09-29 04:43:15Z d-orban $
"""

import numpy
from math import sqrt

def to_boundary(s, p, Delta, ss = None ):
    """
    Given vectors s and p and a trust-region radius Delta > 0, this function
    returns the positive scalar sigma such that
        || s + sigma * p || = Delta
    in Euclidian norm. If known, supply optional argument ss whose value
    should be the squared Euclidian norm of argument s.
    """
    sp = numpy.dot(s,p)
    pp = numpy.dot(p,p)
    if ss is None: ss = numpy.dot(s,s)
    sigma = (-sp + sqrt( sp*sp + pp * ( Delta*Delta - ss ) ) )
    sigma /= pp
    return sigma

def trpcg( g, **kwargs ):
    """
    Solve the quadratic trust-region subproblem

        minimize    <g,s> + 1/2 <s,Hs>
        subject to  <s,s> <= Delta

    by means of the truncated conjugate gradient algorithm
    (aka the Steihaug-Toint algorithm). The notation <x,y>
    indicates the dot product of vectors x and y. H must
    be a symmetric matrix of appropriate size, but not
    necessarily positive definite. By default, Delta=||g||.

    Use:

        (s,it,snorm) = trpcg(H,g,Delta)

    The return values are

        s          the final step
        it         the number of iterations
        snorm      Euclidian norm of the step

    Accepted keywords are

        Delta      the trust-region radius (default: ||g||)
        hprod      a function to compute matrix-vector products
        H          the explicit matrix H
        tol        the relative tolerance (default: 1.0e-6) in
                   case the solution is interior
        maxit      the maximum number of iterations (default: 2n)
        prec       a user-defined preconditioner.

    If both hprod and H are specified, hprod takes precedence. If
    hprod is specified, it is supposed to receive a vector p as input
    and return the matrix-vector product H*p. If H is given, it must
    have a method named 'matvec' to compute matrix-vector products.
    """
    n = len(g)

    # Grab optional arguments
    hprod = kwargs.get( 'hprod', None )
    if hprod is None:
        H = kwargs.get( 'H', None )
        if H is None:
            raise ValueError, 'Specify one of hprod or H'
            return None
    tol = kwargs.get( 'tol', 1.0e-6 )   # Default relative tolerance
    maxit = kwargs.get( 'maxit', 2*n )  # Default max # iterations
    prec = kwargs.get( 'prec', lambda v: v ) # Default preconditioner = identity

    # Initialization
    y = prec(g)
    res = sqrt( numpy.dot( g, y ) )
    Delta = kwargs.get( 'Delta', res )

    s = numpy.zeros( n, 'd' )  # s0 = 0
    snorm = 0.0
    # Initialize r as a copy of g not to alter the original g
    r = g.copy()                 # r = g + H s0 = g
    p = -y                       # p = - preconditioned residual
    k = 0

    res0 = res                   # Initial residual
    rtol = tol*res0              # Relative tolerance
    Hp = numpy.empty( n, 'd' )

    while res > rtol and k < maxit:

        # Compute matrix-vector product H*p
        if hprod is not None:
            Hp = hprod( p )
        else:
            H.matvec( p, Hp )

        pHp = numpy.dot( p, Hp )

        # Compute steplength to the boundary
        sigma = to_boundary(s,p,Delta,ss=snorm*snorm)

        # Compute CG steplength
        alpha = res*res/pHp

        if pHp <= 0.0 or alpha > sigma:
            # p is direction of singularity or negative curvature
            # Follow this direction to the boundary of the region
            # Or: next iterate will be past the boundary
            # Follow direction p to the boundary of the region
            s += sigma * p
            snorm = Delta
            return (s,k,snorm)

        s += alpha * p
        snorm = sqrt(numpy.dot(s,s))
        r += alpha * Hp
        y = prec(r)
        res1 = sqrt( numpy.dot( r, y ) )
        res1overres = res1/res
        beta = res1overres * res1overres
        p = -y + beta * p
        res = res1
        k += 1

    return (s,k,snorm)

def model_value( H, g, s ):
    # Return <g,s> + 1/2 <s,Hs>
    n = len(g)
    Hs = numpy.zeros( n, 'd' )
    H.matvec( s, Hs )
    q = 0.5 * numpy.dot( s, Hs )
    q = numpy.dot(g,s) + q
    return q

def model_grad( H, g, s ):
    # Return g + Hs
    n = len(g)
    Hs = numpy.zeros( n, 'd' )
    H.matvec( s, Hs )
    return g+Hs

def H_prod( H, v ):
    # For simulation purposes only
    n = H.shape[0]
    Hv = numpy.zeros( n, 'd' )
    H.matvec( v, Hv )
    return Hv

if __name__ == '__main__':

    from pysparse import spmatrix
    import precon
    from nlpy_timing import cputime
    #from demo_pygltr import SpecSheet
    #(H,g) = SpecSheet() # The GLTR spec sheet example
    
    t_setup = cputime()
    H = spmatrix.ll_mat_from_mtx( '1138bus.mtx' )
    n = H.shape[0]
    e = numpy.ones( n, 'd' )
    g = numpy.empty( n, 'd' )
    H.matvec( e, g )
    K = precon.DiagonalPreconditioner(H)
    #K = precon.BandedPreconditioner(H)
    t_setup = cputime() - t_setup

    t_solve = cputime()
    #(s,it,snorm) = trpcg(g, Delta=40.0, hprod = lambda p: H_prod(H,p) )
    (s,it,snorm) = trpcg(g,
                         Delta = 33.5,
                         H = H,
                         prec = K.precon
                         )
    t_solve = cputime() - t_solve

    print ' #it = ', it
    print ' snorm = ', snorm
    print ' q(s) = ', model_value( H, g, s )
    print ' setup time: ', t_setup
    print ' solve time: ', t_solve

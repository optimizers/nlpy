import numpy as np
from numpy import infty
from numpy.linalg import norm

def norm1(x):
    return norm(x,ord=1)

def norm2(x):
    return norm(x)

def normp(x,p):
    return norm(x,ord=p)

def norm_infty(x):
    return norm(x,ord=infty)

def normest(A, tol=1.0e-6, maxits=100):
    """
    Estimate the spectral norm of the matrix A.
    The matrix should behave like a linear operator, i.e. allow for
    matrix-vector products and transposed-matrix-vector products to
    be performed via A*x and A.T*y.
    """
    m, n = A.shape
    itn = 0

    # Compute an estimate of the abs-val column sums.
    v = np.ones(m)
    v[np.random.randn(m) < 0] = -1
    x = abs(A.T*v)

    # Normalize the starting vector.
    e = norm(x)
    if e == 0:
        return e, itn
    x = x/e
    e0 = 0
    while abs(e-e0) > tol*e:
        e0 = e
        Ax = A*x
        normAx = norm(Ax)
        if normAx == 0:
            Ax = np.random.rand(m)
            normAx = norm(Ax)
        x = A.T*Ax
        normx = norm(x)
        e = normx / normAx
        x = x / normx
        itn += 1
        if itn > maxits:
            print "Warning: normest didn't converge!"
            break
    return e, itn


if __name__ == '__main__':
    from nlpy.krylov.linop import SimpleLinearOperator as aslinop

    tol = 1e-6 ; maxits = 100
    
    print "Unsymmetric matrices"
    for n in xrange(1,100):
        m = n/2 + 1
        A = np.random.randn(n,m)
        Aop = aslinop(A.shape[1], A.shape[0],
                      lambda v: np.dot(A,v),
                      matvec_transp=lambda v:np.dot(A.T,v))

        normA = np.linalg.norm(A,2)
        normAop,_ = normest(Aop, tol=tol, maxits=maxits)
        error = abs(normA - normAop) / max(1,normA)
        if error > tol*100:
            print "Error in normest = %8.1e" % error

    print
    print "Symmetric matrices"
    for n in xrange(1,100):
        A = np.random.rand(n,n)
        A = .5*(A.T + A)
        Aop = aslinop(A.shape[1], A.shape[0],
                      lambda v: np.dot(A,v),
                      symmetric=True)

        if not Aop.check_symmetric(): print "Oops!"

        normA = np.linalg.norm(A,2)
        normAop,_ = normest(Aop, tol=tol, maxits=maxits)
        error = abs(normA - normAop) / max(1,normA)
        if error > tol*100:
            print "Error in normest = %8.1e" % error

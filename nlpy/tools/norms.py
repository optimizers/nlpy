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
    """
    m, n = A.shape
    itn = 0

    # Compute an estimate of the abs-val column sums.
    v = np.ones(m)
    v[np.random.randn(m) < 0] = -1
    x = abs(A.T*v)

    e = norm(x)
    if e == 0:
        return e, itn
    x = x/e
    e0 = 0
    while abs(e-e0) > tol*e:
        e0 = e
        Ax = A*x
        if norm(Ax) == 0:
            Ax = np.random.rand(m)
        x = A.T*Ax
        normx = norm(x)
        e = normx / norm(Ax)
        x = x / normx
        itn += 1
        if itn > maxits:
            print "Warning: normest didn't converge!"
            break
    return e, itn

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

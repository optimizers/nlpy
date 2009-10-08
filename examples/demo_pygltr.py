"""
Simple demo of the pygltr module.
A number of matrices in MatrixMarket format are read on
the command line, and two versions of gltr are called,
with the same trust-region radius. A comparison of the
running times for the two versions is illustrated on a
simple plot.
"""
from nlpy.krylov import pygltr
from nlpy.tools import norms
from nlpy.tools.timing import cputime
from pysparse import spmatrix
from string import strip, atof
import numpy as np
import sys
import os

def SpecSheet(n=10000):
    """
    Implement the example from the GLTR spec sheet
    """
    g = np.ones(n)
    H = spmatrix.ll_mat_sym(n, 2*n-1)
    for i in range(n): H[i,i] = -2
    for i in range(1, n): H[i,i-1] = 1
    return (H, g)

def SpecSheet_explicit(radius=10.0):
    (H, g) = SpecSheet()
    return gltr_explicit(H, g, radius=radius, prec=lambda v: 0.5*v)

def SpecSheet_implicit(radius=10.0):
    (H, g) = SpecSheet()
    return gltr_implicit(H, g, radius=radius, prec=lambda v: 0.5*v)

def ReadMatrix(fname):
    """
    Read matrix from file fname in MatrixMarket format.
    Alternatively, could read from Ampl nl file.
    Returns a pointer to the matrix, or None if an error occured.
    """
    H = spmatrix.ll_mat_from_mtx(fname)
    (n, m) = H.shape
    if n != m:
        sys.stderr.write('Hessian matrix must be square')
        return None
    if not H.issym:
        sys.stderr.write('Hessian matrix must be symmetric')
        return None
    return H

def MatVecProd(H, v):
    """
    Compute a matrix-vector product and return a vector with the result.
    """
    Hv = np.empty(H.shape[0])
    H.matvec(v, Hv)
    return Hv

def gltr_explicit_fromfile(fname, g, **kwargs):
    H = ReadMatrix(fname)
    if H is None: return None
    return gltr_explicit(H, g, **kwargs)

def gltr_explicit(H, g, **kwargs):
    G = pygltr.PyGltrContext(g, **kwargs)
    t = cputime()
    G.explicit_solve(H.to_csr())
    t = cputime() - t
    return (G.m, G.mult, G.snorm, G.niter, G.nc, G.ierr, t)

def gltr_implicit_fromfile(fname, g, **kwargs):
    H = ReadMatrix(fname)
    if H is None: return None
    return gltr_implicit(H, g, **kwargs)

def gltr_implicit(H, g, **kwargs):
    G = pygltr.PyGltrContext(g, **kwargs)
    H.to_csr()
    t = cputime()
    G.implicit_solve(lambda v: MatVecProd(H,v))
    t = cputime() - t
    return (G.m, G.mult, G.snorm, G.niter, G.nc, G.ierr, t)
    

# Test the module
if __name__ == '__main__':
    # Problems are specified on the command line
    ProbList = sys.argv[1:]
    nprobs = len(ProbList)

    radius = 10.0
    t_list_I  = []
    t_list_II = []

    header = '       %8s  %5s  %6s  %6s  %6s  %4s' % ('Problem', 'Size', 'Nnz', 'Iter', 'Time', 'Exit')
    head = '%8s  %5d  %6d  %6d  %6.2f  %4d\n'
    header_expl = '%-5s  ' % 'Expl'
    header_impl = '%-5s  ' % 'Impl'
    lhead = len(header)
    sys.stderr.write(header + '\n')
    sys.stderr.write('-' * lhead + '\n')
    
    # Run example from spec sheet
    (f, m, sn, nit, nc, ierr, t1) = SpecSheet_explicit()
    sys.stdout.write(header_expl)
    sys.stdout.write(head % ('SpcSheet', 10000, 19999, nit, t1, ierr))

    (f, m, sn, nit, nc, ierr, t2) = SpecSheet_implicit()
    sys.stdout.write(header_impl)
    sys.stdout.write(head % ('SpcSheet', 10000, 19999, nit, t2, ierr))
    sys.stderr.write('-' * lhead + '\n')

    # Run problems given on the command line
    for p in range(len(ProbList)):
        problem = os.path.basename(ProbList[p])
        H = ReadMatrix(ProbList[p])
        if problem[-4:] == '.mtx':
            ProbList[p] = problem[:-4]
            problem = problem[:-4]
        ncol = H.shape[1]
        if H is not None:
            g = np.ones(H.shape[0])
            (f, m, sn, nit, nc, ierr, t1) = gltr_explicit(H, g, radius=radius, ST=False, itmax=2*ncol, litmax=ncol)
            sys.stdout.write(header_expl)
            sys.stdout.write(head % (problem[-8:], ncol, H.nnz, nit, t1, ierr))

            (f, m, sn, nit, nc, ierr, t2) = gltr_implicit(H, g, radius=radius, ST=False, itmax=2*ncol, litmax=ncol)
            sys.stdout.write(header_impl)
            sys.stdout.write(head % (problem[-8:], ncol, H.nnz, nit, t2, ierr))

            sys.stdout.write('-' * lhead + '\n')
        else:
            # Remove problem from list
            ProbList[p] = []
            nprobs -= 1



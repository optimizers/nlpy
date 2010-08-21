# An illustration of how to fetch the incomplete Cholesky factor from Pycfs.
# We compute the minimum limited-memory factor that yields the exact factor.

from __future__ import division
from pysparse import spmatrix
from nlpy.precon import pycfs
import sys

def usage():
    progname = sys.argv[0]
    sys.stderr.write('Usage\n');
    sys.stderr.write('  %-s problem [problem ...]\n' % progname)
    sys.stderr.write('  where each problem is the file name of a matrix\n')
    sys.stderr.write('  in MatrixMarket sparse format (*.mtx).\n')

def pycfs_fact(ProblemList, bigp=500):
    import os

    if len(ProblemList) == 0:
        usage()
        sys.exit(1)
    
    header = '%-12s  %-6s  %-6s  %-7s  %-6s  %-6s  %-8s  %-10s  %-3s' % ('Name', 'n', 'nnz(A)', 'nnz(L)', 'den(A)', 'den(L)', 'shift', 'mem', 'p*')
    lhead = len(header)
    print header
    print '-' * lhead
    fmt = '%-12s  %-6d  %-6d  %-7d  %-6.2f  %-6.2f  %-8g  %-10d  %-3d'

    for problem in ProblemList:

        A = spmatrix.ll_mat_from_mtx(problem)
        (m, n) = A.shape
        if m != n: break
        (aval,arow,acol) = A.find()

        prob = os.path.basename(problem)
        if prob[-4:] == '.mtx': prob = prob[:-4]

        P = pycfs.PycfsContext(A, mem=bigp)
        L = P.fetch()  # Retrieve Cholesky factor
        nnzA = A.nnz
        nnzL = L.nnz
        densityA = 100.0 * nnzA/(n*(n+1)/2)
        densityL = 100.0 * nnzL/(n*(n+1)/2)
        shift = P.shift
        memory_available = n*bigp
        p_nnz = (nnzL-nnzA)/n
        (lval,lrow,lcol) = L.find()

        print fmt % (prob, n, nnzA, nnzL, densityA, densityL, shift, memory_available, p_nnz)

        try:
            import matplotlib.pyplot as plt
            from pyorder.tools.spy import FastSpy
            # Plot sparsity patterns
            left = plt.subplot(121)
            right = plt.subplot(122)
            FastSpy(m,n,arow,acol,ax=left)
            FastSpy(m,n,lrow,lcol,ax=right)
            plt.show()
        except:
            sys.stderr.write('Not plotting sparsity patterns.')
            sys.stderr.write(' Did you install Matplotlib?\n')
        

    print '-' * lhead
    return None

if __name__ == '__main__':

    pycfs_fact(sys.argv[1:])    

"""
Illustrates usage of PyMa27 for factorization of Symmetric Quasi-Definite (sqd)
matrices. For a description and properties of sqd matrices, see

 R. J. Vanderbei, Symmetric Quasi-Definite Matrices,
 SIAM Journal on Optimization, 5(1), 100-113, 1995.

Example usage: python demo_sqd.py bcsstk18.mtx bcsstk11.mtx

D. Orban, Montreal, December 2007.
"""

try:
    from nlpy.linalg.pyma57 import PyMa57Context as LBLContext
except:
    from nlpy.linalg.pyma27 import PyMa27Context as LBLContext

from pysparse.sparse import spmatrix
import numpy as np
from nlpy.tools.timing import cputime
import sys

if len(sys.argv) < 3:
    sys.stderr.write('Please supply two positive definite matrices as input')
    sys.stderr.write(' in MatrixMarket format.\n')
    sys.exit(1)

# Create symmetric quasi-definite matrix K
A = spmatrix.ll_mat_from_mtx(sys.argv[1])
C = spmatrix.ll_mat_from_mtx(sys.argv[2])

nA = A.shape[0]
nC = C.shape[0]
K = spmatrix.ll_mat_sym(nA + nC, A.nnz + C.nnz + min(nA,nC))
K[:nA,:nA] = A
K[nA:,nA:] = C
K[nA:,nA:].scale(-1.0)
idx = np.arange(min(nA,nC), dtype=np.int)
K.put(1, nA+idx, idx)

# Create right-hand side rhs=K*e
e = np.ones(nA+nC)
rhs = np.empty(nA+nC)
K.matvec(e,rhs)

# Factorize and solve Kx = rhs, knowing K is sqd
t = cputime()
LDL = LBLContext(K, sqd=True)
t = cputime() - t
sys.stderr.write('Factorization time with sqd=True : %5.2fs   ' % t )
LDL.solve(rhs, get_resid=False)
sys.stderr.write('Error: %7.1e\n' % np.linalg.norm(LDL.x - e, ord=np.Inf))

# Do it all over again, pretending we don't know K is sqd
t = cputime()
LBL = LBLContext(K)
t = cputime() - t
sys.stderr.write('Factorization time with sqd=False: %5.2fs   ' % t )
LBL.solve(rhs, get_resid=False)
sys.stderr.write('Error: %7.1e\n' % np.linalg.norm(LBL.x - e, ord=np.Inf))

try:
    import pylab
    from pyorder.tools import FastSpy
    # Plot sparsity pattern
    fig = pylab.figure()
    cur_ax = fig.gca()
    cur_ax.plot( [0,nA+nC-1], [nA,nA], 'b--', linewidth=1 )
    cur_ax.plot( [nA,nA], [0,nA+nC-1], 'b--', linewidth=1 )
    (val,irow,jcol) = K.find()
    FastSpy(nA+nC, nA+nC, irow, jcol, sym=True, ax=cur_ax)
    #pylab.savefig('sqd.png', bbox_inches='tight')
    pylab.show()
except:
    sys.stderr.write('Not plotting sparsity pattern.')
    sys.stderr.write(' Did you install Matplotlib?\n')

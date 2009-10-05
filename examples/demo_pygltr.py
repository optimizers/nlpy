"""
Simple demo of the pygltr module.
A number of matrices in MatrixMarket format are read on
the command line, and two versions of gltr are called,
with the same trust-region radius. A comparison of the
running times for the two versions is illustrated on a
simple plot.
"""
from nlpy.krylov import pygltr
from pysparse import spmatrix
import numpy
from nlpy.tools import norms
from nlpy.tools.timing import cputime
import sys
import os
from string import strip, atof

def SpecSheet():
    """
    Implement the example from the GLTR spec sheet
    """
    n = 10000
    g = numpy.ones( n, 'd' )
    H = spmatrix.ll_mat_sym( n, 2*n - 1 )
    for i in range( n ): H[i,i] = -2
    for i in range( 1, n ): H[i,i-1] = 1
    return (H, g )

def SpecSheet_explicit():
    radius = 10.0
    (H, g) = SpecSheet()
    return gltr_explicit( H, g, radius = radius, prec = lambda v: 0.5*v )

def SpecSheet_implicit():
    radius = 10.0
    (H, g) = SpecSheet()
    return gltr_implicit( H, g, radius = radius, prec = lambda v: 0.5*v )

def ReadMatrix( fname ):
    """
    Read matrix from file fname in MatrixMarket format.
    Alternatively, could read from Ampl nl file.
    Returns a pointer to the matrix, or None if an error occured.
    """
    H = spmatrix.ll_mat_from_mtx( fname )
    (n, m) = H.shape
    if n != m:
        sys.stderr.write( 'Hessian matrix must be square' )
        return None
    if not H.issym:
        sys.stderr.write( 'Hessian matrix must be symmetric' )
        return None
    return H

def MatVecProd( H, v ):
    """
    Compute a matrix-vector product and return a vector with the result.
    If reading a matrix from nl file instead, the MatVecProd function
    should be replaced by the built-in amplpy.hprod.
    """
    Hv = numpy.zeros( H.shape[0], 'd' )
    H.matvec( v, Hv )
    return Hv

def gltr_explicit_fromfile( fname, g, **kwargs ):
    H = ReadMatrix( fname )
    if H is None: return None
    return gltr_explicit( H, g, **kwargs )

def gltr_explicit( H, g, **kwargs ):
    G = pygltr.PyGltrContext( g, **kwargs )
    #G.debug = True
    t = cputime()
    G.explicit_solve( H.to_csr() )
    t = cputime() - t
    return (G.m, G.mult, G.snorm, G.niter, G.nc, G.ierr, t)

def gltr_implicit_fromfile( fname, g, **kwargs ):
    H = ReadMatrix( fname )
    if H is None: return None
    return gltr_implicit( H, g, **kwargs )

def gltr_implicit( H, g, **kwargs ):
    G = pygltr.PyGltrContext( g, **kwargs )
    #G.debug = True
    H.to_csr()
    t = cputime()
    G.implicit_solve( lambda v: MatVecProd(H,v) )
    t = cputime() - t
    return (G.m, G.mult, G.snorm, G.niter, G.nc, G.ierr, t)
    

# Test the module
if __name__ == '__main__':
    # Problems are specified on the command line
    ProbList = sys.argv[1:]
    nprobs = len( ProbList )

    radius = 10.0
    t_list_I  = []
    t_list_II = []

    header = '       %8s  %5s  %6s  %6s  %6s  %4s' % ('Problem', 'Size', 'Nnz', 'Iter', 'Time', 'Exit')
    head = '%8s  %5d  %6d  %6d  %6.2f  %4d\n'
    header_expl = '%-5s  ' % 'Expl'
    header_impl = '%-5s  ' % 'Impl'
    lhead = len( header )
    sys.stdout.write( header + '\n' )
    sys.stdout.write( '-' * lhead + '\n' )
    
    # Run example from spec sheet
    (f, m, sn, nit, nc, ierr, t1) = SpecSheet_explicit()
    sys.stdout.write(header_expl)
    sys.stdout.write(head % ('SpcSheet', 10000, 19999, nit, t1, ierr))
    #t_list_I.append( t1 )
    (f, m, sn, nit, nc, ierr, t2) = SpecSheet_implicit()
    sys.stdout.write(header_impl)
    sys.stdout.write(head % ('SpcSheet', 10000, 19999, nit, t2, ierr))
    #t_list_II.append( t2 )
    sys.stdout.write( '-' * lhead + '\n' )

    # Run problems given on the command line
    for p in range( len( ProbList ) ):
        problem = os.path.basename(ProbList[p])
        H = ReadMatrix(ProbList[p])
        if problem[-4:] == '.mtx':
            ProbList[p] = problem[:-4]
            problem = problem[:-4]
        ncol = H.shape[1]
        if H is not None:
            g = numpy.ones( H.shape[0], 'd' )
            (f, m, sn, nit, nc, ierr, t1) = gltr_explicit(H, g, radius=radius, ST=False, itmax=2*ncol, litmax=ncol)
            sys.stdout.write(header_expl)
            sys.stdout.write(head % (problem[-8:], ncol, H.nnz, nit, t1, ierr))
            t_list_I.append( t1 )
            (f, m, sn, nit, nc, ierr, t2) = gltr_implicit(H, g, radius=radius, ST=False, itmax=2*ncol, litmax=ncol)
            sys.stdout.write(header_impl)
            sys.stdout.write(head % (problem[-8:], ncol, H.nnz, nit, t2, ierr))
            t_list_II.append( t2 )
            sys.stdout.write( '-' * lhead + '\n' )
        else:
            # Remove problem from list
            ProbList[p] = []
            nprobs -= 1
        #sys.stdout.write( '-' * lhead + '\n\n' )

    # Obtain timings from pure Fortran GLTR
    # These timings were obtained using the built-in dtime()
    # with all defaults GLTR options and a radius = 1.0D+0
    try:
        fp = open( '../.benchmark/Gltr/test_problems', 'r' )
    except:
        sys.stderr.write( 'Cannot open ../.benchmark/Gltr/test_problems\n' )
        sys.stderr.write( 'Make sure you have run "make bmark"\n' )
        sys.exit(1)
        
    f90probs = fp.readlines()
    fp.close()
    try:
        ft = open( '../.benchmark/Gltr/gltr_timings', 'r' )
    except:
        sys.stderr.write( 'Cannot open ../.benchmark/Gltr/gltr_timings\n' )
        sys.stderr.write( 'Make sure you have run "make bmark"\n' )
        sys.exit(1)

    f90tmgs = ft.readlines()
    ft.close()
    t_Fortran_ref = {}
    for i in range( len( f90probs ) ):
        k = strip( f90probs[i] )
        if k[-4:] == '.mtx':
            k = k[:-4]
        t = atof( f90tmgs[i] )
        t_Fortran_ref[k] = t

    #try:
    #    ft = open( '../.benchmark/Gltr/spec_sheet_timing', 'r' )
    #except:
    #    sys.stderr.write('Cannot open ../.benchmark/Gltr/spec_sheet_timings\n')
    #    sys.stderr.write( 'Make sure you have run "make bmark"\n' )
    #    sys.exit(1)

    #t = atof( ft.readline() )
    #ft.close()
    #t_Fortran_ref['Spec sheet'] = t

    #t_Fortran = [t_Fortran_ref['Spec sheet']] + [ t_Fortran_ref[k] for k in ProbList ]
    t_Fortran = [ t_Fortran_ref[k] for k in ProbList ]

    # Plot the timings
    try:
        import matplotlib
        if matplotlib.__version__ < '0.65':
            import matplotlib.matlab as MM
        else:
            import matplotlib.pylab as MM
    except:
        sys.stderr.write( ' If you had Matplotlib installed, you\n' )
        sys.stderr.write( ' would be looking at timing plots right now.\n' )
        sys.exit( 0 )

    darkblue = '#2c11cf'
    lightblue = '#8f84e0'
    steelblue = '#5d82ef'
    x = range( nprobs )
    ax = MM.axes()
    ax.plot( x, t_list_I,  color = darkblue,  linewidth = 3 )
    ax.plot( x, t_list_II, color = lightblue, linewidth = 3 )
    ax.plot( x, t_Fortran, color = steelblue, linewidth = 3 )
    ax.set_xticks( x )
    #ax.set_xticklabels( ['SpecSheet'] + ProbList, rotation = 45, horizontalalignment = 'right' )
    ax.set_xticklabels( ProbList, rotation=45, horizontalalignment='right' )
    ax.set_ylabel( 'Time (s)' )
    ax.set_title( 'Comparing Explicit and Implicit versions of PyGLTR' )
    ax.legend( ['Explicit', 'Implicit', 'Fortran'], 'lower right' )
    MM.show()

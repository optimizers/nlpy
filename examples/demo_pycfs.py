import numpy
from pysparse import spmatrix
from nlpy.precon import pycfs
from nlpy.tools.timing import cputime
import os
import sys

def usage():
    progname = sys.argv[0]
    sys.stderr.write('Usage\n');
    sys.stderr.write('  %-s problem [problem ...]\n' % progname)
    sys.stderr.write('  where each problem is the file name of a matrix\n')
    sys.stderr.write('  in MatrixMarket sparse format (*.mtx).\n')

def test_pycfs(ProblemList, plist=[0, 2, 5, 10], latex=False):

    if len(ProblemList) == 0:
        usage()
        sys.exit(1)
    
    if latex:
        # For LaTeX output
        fmt = '%-12s & %-5s & %-6s & %-2s & %-4s & %-8s & %-4s & %-8s & %-6s & %-6s\\\\\n'
        fmt1 = '%-12s & %-5d & %-6d & '
        fmt2 = '%-2d & %-4d & %-8.1e & %-4d & %-8.1e & %-6.2f & %-6.2f\\\\\n'
        hline = '\\hline\n'
        skip = '&&&'
    else:
        # For ASCII output
        fmt = '%-12s %-5s %-6s %-2s %-4s %-8s %-4s %-8s %-6s %-6s\n'
        fmt1 = '%-12s %-5d %-6d '
        fmt2 = '%-2d %-4d %-8.1e %-4d %-8.1e %-6.2f %-6.2f\n'
        skip = ' ' * 26


    header = fmt % ('Name','Size','nnz','p','info','shift','iter','relResid','fact','solve')
    lhead = len(header)
    if not latex: hline = '-' * lhead + '\n'
    sys.stderr.write(hline + header + hline)

    time_list = {}        # Record timings
    iter_list = {}        # Record number of iterations

    for problem in ProblemList:

        A = spmatrix.ll_mat_from_mtx(problem)
        (m, n) = A.shape
        if m != n: break

        prob = os.path.basename(problem)
        if prob[-4:] == '.mtx': prob = prob[:-4]

        # Right-hand side is Ae, as in Icfs.
        e = numpy.ones(n, 'd');
        b = numpy.ones(n, 'd');
        A.matvec(e, b)

        sys.stdout.write(fmt1 % (prob, n, A.nnz))
        advance = False

        # Call icfs and pcg
        tlist = []
        ilist = []
        for pval in plist:
            t0 = cputime()
            P = pycfs.PycfsContext(A, mem=pval)
            t_fact = cputime() - t0
            P.solve(b)
            t_solve = P.tsolve
            tlist.append(t_fact + t_solve)
            ilist.append(P.iter)

            if advance: sys.stdout.write(skip)
            sys.stdout.write(fmt2 % (pval, P.info, P.shift, P.iter, P.relres, t_fact, t_solve))
            advance = True
            
        time_list[prob] = tlist
        iter_list[prob] = ilist
        sys.stderr.write(hline)

    return (time_list,iter_list)


if __name__ == '__main__':
    # Obtain timings for the specified problems
    plist = [0, 2, 5, 10]
    (t_lst, i_lst) = test_pycfs(sys.argv[1:], plist=plist)

    # I obtained time_ref by running the default icf driver on my machine
    # for the same values of p as in plist.
    t_ref = { '1138bus'  : [0.959e-02, 0.522e-02, 0.338e-02, 0.316e-02],
              'bcsstk08' : [0.443e-01, 0.380e-01, 0.760e-02, 0.887e-02],
              'bcsstk09' : [0.709e-02, 0.551e-02, 0.697e-02, 0.286e-01],
              'bcsstk10' : [0.833e-02, 0.626e-02, 0.544e-02, 0.438e-02],
              'bcsstk11' : [0.350e-01, 0.432e-01, 0.359e-01, 0.360e-01],
              'bcsstk18' : [0.253e+00, 0.144e+00, 0.139e+00, 0.123e+00],
              'bcsstk19' : [0.968e-02, 0.228e-02, 0.247e-02, 0.316e-02] }

    K = t_lst.keys()

    if len(K) > 1:
        try:
            import matplotlib
            if matplotlib.__version__ < '0.65':
                import matplotlib.matlab as MM
            else:
                import matplotlib.pylab as MM
        except:
            print ' If you had Matplotlib installed, you would be looking'
            print ' at timing plots right now...'
            sys.exit(0)

        darkblue = '#2c11cf'
        lightblue = '#8f84e0'
        steelblue = '#5d82ef'
        x = range(len(t_lst.keys()))
        ax = []
        ax.append(MM.axes([ .05, .05, .40, .40 ])) # lower left
        ax.append(MM.axes([ .05, .55, .40, .40 ])) # upper left
        ax.append(MM.axes([ .55, .05, .40, .40 ])) # lower right
        ax.append(MM.axes([ .55, .55, .40, .40 ])) # upper right
        for i in range(4):
            ax[i].plot(x, [ t_lst[k][i] for k in K ], color=darkblue,  linewidth=3)
            ax[i].plot(x, [ t_ref[k][i] for k in K ], color=lightblue, linewidth=3)
            ax[i].set_xticks([])
            ax[i].set_title('Limited memory p = %-d' % plist[i], fontsize='small')
            ax[i].legend(['Python', 'Fortran'], 'upper left')
        for i in [2,3]:
            ax[i].set_ylabel('Time (s)', fontsize='small')
        MM.show()

        # For the number of iterations, use first value of p as reference
        x = range(len(i_lst.keys()))
        ax = MM.subplot(111)
        lgnd = []
        for i in range(len(plist)):
            lgnd.append('p = %-d' % plist[i])
        ax.plot(x, [ (1.0*i_lst[k][0])/i_lst[k][0] for k in K ], 'k-')
        ax.plot(x, [ (1.0*i_lst[k][1])/i_lst[k][0] for k in K ], 'k:')
        ax.plot(x, [ (1.0*i_lst[k][2])/i_lst[k][0] for k in K ], 'k-.')
        ax.plot(x, [ (1.0*i_lst[k][3])/i_lst[k][0] for k in K ], 'k--')
        ax.legend(lgnd, 'upper right')
        ax.set_title('Number of iterations(p)/Number of iterations(0)')
        ax.set_xticklabels(K, rotation = 45, horizontalalignment = 'right', fontsize='small')
        MM.show()
        

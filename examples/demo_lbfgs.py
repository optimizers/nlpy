"""
Demo of the limited-memory BFGS method. Each problem given on the command line
is solved for several values of the limited-memory parameter.
"""

from nlpy.model import amplpy
from nlpy.optimize.solvers.lbfgs import LBFGSFramework
import os
import sys

headerfmt = '%-15s  %-6s  %-5s  %-12s  %-12s  %-6s  %-7s\n'
header = headerfmt % ('Problem', 'n', 'pairs', 'Obj', 'Grad', 'Iter', 'Time')
hlen = len( header )
format = '%-15s  %-6d  %-5d  %-12g  %-12g  %-6d  %-7g\n'
sys.stdout.write(header)
sys.stdout.write('-' * hlen + '\n')

for ProblemName in sys.argv[1:]:
    nlp = amplpy.AmplModel( ProblemName )

    for m in [1, 2, 3, 4, 5, 10, 15, 20]:
        lbfgs = LBFGSFramework(nlp, npairs=m, scaling=True, silent=True)
        lbfgs.solve()

        # Output final statistics
        probname = os.path.basename(ProblemName)
        if probname[-3:] == '.nl': probname = probname[:-3]
        sys.stdout.write(format % (probname, nlp.n, lbfgs.npairs, lbfgs.f,
                                   lbfgs.gnorm, lbfgs.iter, lbfgs.tsolve))

#!/usr/bin/env python
from nlpy.model import amplpy
from nlpy.optimize.solvers.lbfgs import LbfgsFramework
from nlpy.tools.timing import cputime
import numpy
import sys

if len(sys.argv) < 2:
    sys.stderr.write('Please specify model name\n')
    sys.exit(-1)

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

for ProblemName in sys.argv[1:]:

    t_setup = cputime()
    nlp = amplpy.AmplModel(ProblemName)         # Create a model

    if nlp.nbounds > 0 or nlp.m > 0:            # Check for unconstrained problem
        sys.stderr.write('Problem has bounds or general constraints\n')
        nlp.close()
        continue

    nlp.stop_d = 1.0e-12    
    lbfgs = LbfgsFramework(nlp, npairs=5, scaling=True, silent=False)
    t_setup = cputime() - t_setup

    lbfgs.solve()

    # Output final statistics
    print
    print 'Final variables:', lbfgs.x
    print
    print '-------------------------------'
    print 'LBFGS: End of Execution'
    print '  Problem                     : %-s' % ProblemName
    print '  Dimension                   : %-d' % nlp.n
    print '  Number of (s,y) pairs stored: %-d' % lbfgs.npairs
    print '  Converged to optimality     : %-s' % repr(lbfgs.converged)
    print '  Initial/Final Objective     : %-g/%-g' % (lbfgs.f0, lbfgs.f)
    print '  Initial/Final Gradient Norm : %-g/%-g' % (lbfgs.g0,lbfgs.gnorm)
    print '  Number of iterations        : %-d' % lbfgs.iter
    print '  Scaling                     : %-s' % repr(lbfgs.lbfgs.scaling)
    print '  Setup/Solve time            : %-gs/%-gs' % (t_setup,lbfgs.tsolve)
    print '  Total time                  : %-gs' % (t_setup + lbfgs.tsolve)
    print '-------------------------------'

    #nlp.writesol( lbfgs.x, nlp.pi0, 'And the winner is' ) # Output  solution
    nlp.close()                                # Close connection with model


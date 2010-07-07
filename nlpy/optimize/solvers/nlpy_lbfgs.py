#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import amplpy
from nlpy.optimize.solvers.lbfgs import LBFGSFramework
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
import sys

usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent unconstrained nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)
    
parser.add_option("-a", "--abstol", action="store", type="float",
                  default=1.0e-6, dest="abstol",
                  help="Absolute stopping tolerance")
parser.add_option("-m", "--npairs", action="store", type="int",
                  default=5, dest="npairs",
                  help="Number of (s,y) pairs to store")
parser.add_option("-I", "--noscale", action="store_true", default=False,
                  dest="noscale",
                  help="Do not scale initial matrix (use identity matrix I)")
parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxiter",  help="Specify maximum number of iterations")
parser.add_option("-r", "--reltol", action="store", type="float",
                  default=1.0e-12, dest="reltol",
                  help="Relative stopping tolerance")
parser.add_option("-v", "--verbose", action="store_true", default=False,
                  dest="verbose", help="Print iterations detail")

# Parse command-line options
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxiter is not None:
    opts['maxiter'] = options.maxiter
opts['reltol'] = options.reltol
opts['abstol'] = options.abstol

if options.verbose:
    print 'Using options:'
    print opts

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

# Define formats for output table.
hdrfmt = '%-15s  %5s  %15s  %7s  %6s  %6s  %5s'
hdr = hdrfmt % ('Name','Iter','Objective','dResid','Setup','Solve','Optim')
lhdr = len(hdr)
fmt = '%-15s  %5d  %15.8e  %7.1e  %6.2f  %6.2f  %5s'
print hdr
print '-' * lhdr

for ProblemName in args:

    t_setup = cputime()
    nlp = amplpy.AmplModel(ProblemName)       # Create a model

    if nlp.nbounds > 0 or nlp.m > 0:          # Check for unconstrained problem
        sys.stderr.write('Problem has bounds or general constraints\n')
        nlp.close()
        continue

    nlp.stop_d = 1.0e-12    
    lbfgs = LBFGSFramework(nlp,
                           npairs=options.npairs,
                           scaling=not options.noscale,
                           silent=not options.verbose)
    t_setup = cputime() - t_setup

    lbfgs.solve()

    # Output final statistics
    print fmt % (ProblemName, lbfgs.iter, lbfgs.f, lbfgs.gnorm, t_setup, lbfgs.tsolve, lbfgs.converged)

    if options.verbose:
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


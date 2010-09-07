#!/usr/bin/env python

from nlpy import __version__
from ldfp import LDFPModel
#from noisyldfp import LDFPNoisyModel
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionCG as TRSolver
from ldfp import LDFPTrunkFramework as solver
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
import sys

def pass_to_trunk(nlp, **kwargs):

    if nlp.nbounds > 0 or nlp.m > 0:         # Check for unconstrained problem
        sys.stderr.write('%s has %d bounds and %d general constraints\n' % (ProblemName, nlp.nbounds, nlp.m))
        return None

    verbose = kwargs.get('verbose', False)

    t = cputime()
    tr = TR(Delta=1.0, eta1=0.05, eta2=0.9, gamma1=0.25, gamma2=2.5)

    # When instantiating TrunkFramework of TrunkLbfgsFramework,
    # we select a trust-region subproblem solver of our choice.
    TRNK = solver(nlp, tr, TRSolver, **kwargs)
    TRNK.TR.Delta = 0.1 * TRNK.gnorm         # Reset initial trust-region radius
    t_setup = cputime() - t                  # Setup time

    if verbose:
        print
        print '------------------------------------------'
        print 'LDFP: Solving problem %-s with parameters' % ProblemName
        hdr = 'eta1 = %-g  eta2 = %-g  gamma1 = %-g  gamma2 = %-g Delta0 = %-g'
        print hdr % (tr.eta1, tr.eta2, tr.gamma1, tr.gamma2, tr.Delta)
        print '------------------------------------------'
        print

    TRNK.Solve()
    
    # Output final statistics
    if verbose:
        print
        print 'Final variables:', TRNK.x
        print
        print '-------------------------------'
        print 'Trunk: End of Execution'
        print '  Problem                     : %-s' % ProblemName
        print '  Dimension                   : %-d' % nlp.n
        print '  Initial/Final Objective     : %-g/%-g' % (TRNK.f0, TRNK.f)
        print '  Initial/Final Gradient Norm : %-g/%-g' % (TRNK.g0, TRNK.gnorm)
        print '  Number of iterations        : %-d' % TRNK.iter
        print '  Number of function evals    : %-d' % TRNK.nlp.feval
        print '  Number of gradient evals    : %-d' % TRNK.nlp.geval
        print '  Number of Hessian  evals    : %-d' % TRNK.nlp.Heval
        print '  Number of matvec products   : %-d' % TRNK.nlp.Hprod
        print '  Total/Average Lanczos iter  : %-d/%-g' % (TRNK.cgiter, (float(TRNK.cgiter)/TRNK.iter))
        print '  Setup/Solve time            : %-gs/%-gs' % (t_setup, TRNK.tsolve)
        print '  Total time                  : %-gs' % (t_setup + TRNK.tsolve)
        print '-------------------------------'

    return (t_setup, TRNK)

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
parser.add_option("-c", "--classic", action="store_true", default=False,
                  dest="classic", help="Do not use backtracking linesearch instead of shrinking trust region")
parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxiter",  help="Specify maximum number of iterations")
parser.add_option("-k", "--nBacktrack", action="store", type="int",
                  default=5, dest="nbk", help="Max number of backtracking linesearch steps")
parser.add_option("-M", "--monotone", action="store_true", default=False,
                  dest="monotone", help="Use monotone descent strategy")
parser.add_option("-n", "--nNonMono", action="store", type="int", default=25,
                  dest="nNonMono", help="Number of iterations for which increase is tolerable")
parser.add_option("-r", "--reltol", action="store", type="float",
                  default=1.0e-12, dest="reltol",
                  help="Relative stopping tolerance")
parser.add_option("-v", "--verbose", action="store_true", default=False,
                  dest="verbose", help="Print iterations detail")
parser.add_option("-x", "--exact", action="store_true", default=False,
                  dest="exact", help="Do not use inexact Newton framework")
parser.add_option("-N", "--noisy", action="store_true", default=False,
                  dest="noisy", help="Simulate a noisy problem (watch out!)")

# Parse command-line options
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxiter is not None:
    opts['maxiter'] = options.maxiter
opts['ny'] = not options.classic
opts['bk'] = options.nbk
opts['monotone'] = options.monotone
opts['nIterMono'] = options.nNonMono
opts['silent'] = not options.verbose
opts['inexact'] = not options.exact
opts['reltol'] = options.reltol
opts['abstol'] = options.abstol

if options.verbose:
    print 'Using options:'
    print opts

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

# Define formats for output table.
hdrfmt = '%-15s  %5s  %15s  %7s  %6s  %6s  %4s'
hdr = hdrfmt % ('Name','Iter','Objective','dResid','Setup','Solve','Stat')
lhdr = len(hdr)
fmt = '%-15s  %5d  %15.8e  %7.1e  %6.2f  %6.2f  %4s'
print hdr
print '-' * lhdr

for ProblemName in args:
    # Create model
    if options.noisy:
        nlp = LDFPNoisyModel(ProblemName, npairs=options.npairs)
    else:
        nlp = LDFPModel(ProblemName, npairs=options.npairs)
    if ProblemName[-3:] == '.nl':
        ProblemName = ProblemName[:-3]
    t_setup, TRNK = pass_to_trunk(nlp, **opts)
    print fmt % (ProblemName, TRNK.iter, TRNK.f, TRNK.gnorm, t_setup, TRNK.tsolve, TRNK.status)
    nlp.close()                               # Close connection with model


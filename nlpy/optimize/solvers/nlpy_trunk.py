#!/usr/bin/env python

from nlpy.model import amplpy
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionCG, TrustRegionCGLBFGS
from nlpy.optimize.solvers.trunk import TrunkFramework as solver
from nlpy.tools.timing import cputime
from argparse import ArgumentParser
import logging
import numpy
import sys


def pass_to_trunk(nlp, options, showbanner=True):

    if nlp.nbounds > 0 or nlp.m > 0:         # Check for unconstrained problem
        sys.stderr.write('%s has %d bounds and %d general constraints\n' % (ProblemName, nlp.nbounds, nlp.m))
        return None

    t = cputime()
    tr = TR(Delta=1.0, eta1=0.05, eta2=0.9, gamma1=0.25, gamma2=2.5)

    # When instantiating TrunkFramework,
    # we select a trust-region subproblem solver of our choice.
    TRSolver = TrustRegionCGLBFGS if options.lbfgs else TrustRegionCG
    TRNK = solver(nlp, tr, TRSolver, ny=not options.nony,
                  inexact=not options.exact,
                  logger_name='nlpy.trunk', reltol=options.rtol,
                  abstol=options.atol, maxiter=options.maxiter)
    t_setup = cputime() - t                  # Setup time

    if showbanner:
        print
        print '------------------------------------------'
        print 'Trunk: Solving problem %-s with parameters' % ProblemName
        hdr = 'eta1 = %-g  eta2 = %-g  gamma1 = %-g  gamma2 = %-g Delta0 = %-g'
        print hdr % (tr.eta1, tr.eta2, tr.gamma1, tr.gamma2, tr.Delta)
        print '------------------------------------------'
        print

    TRNK.Solve()
    return TRNK

# Declare command-line arguments.
parser = ArgumentParser(description='A Newton/CG trust-region solver for' + \
                                    ' unconstrained optimization')
parser.add_argument('--plot', action='store_true',
                    help='Plot evolution of trust-region radius')
parser.add_argument('--nony', action='store_true',
                    help='Deactivate Nocedal-Yuan backtracking')
parser.add_argument('--exact', action='store_true',
                    help='Use exact Newton strategy')
parser.add_argument('--atol', action='store', type=float, default=1.0e-8,
                    dest='atol', help='Absolute stopping tolerance')
parser.add_argument('--rtol', action='store', type=float, default=1.0e-6,
                    dest='rtol', help='Relative stopping tolerance')
parser.add_argument('--maxiter', action='store', type=int, default=100,
                    dest='maxiter', help='Maximum number of iterations')
parser.add_argument('--monotone', action='store_true',
                    help='Use monotone descent strategy')
options, args = parser.parse_known_args()

# Create root logger.
log = logging.getLogger('nlpy.trunk')
level = logging.INFO
log.setLevel(level)
fmt = logging.Formatter('%(name)-10s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)
multiple_probs = len(args)

for ProblemName in args:
    nlp = amplpy.AmplModel(ProblemName)         # Create a model
    TRNK = pass_to_trunk(nlp, options, showbanner=not multiple_probs)
    #nlp.writesol(TRNK.x, nlp.pi0, 'And the winner is')    # Output "solution"
    nlp.close()                                 # Close connection with model

# Output final statistics
if not multiple_probs:
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
    print '  Total/Average Lanczos iter  : %-d/%-g' % (TRNK.total_cgiter, (float(TRNK.total_cgiter)/TRNK.iter))
    print '  Setup/Solve time            : %-gs/%-gs' % (t_setup, TRNK.tsolve)
    print '  Total time                  : %-gs' % (t_setup + TRNK.tsolve)
    print '-------------------------------'

    # Plot the evolution of the trust-region radius on the last problem
    if TRNK is not None and args.plot:
        try:
            import pylab
        except:
            sys.stderr.write('If you had pylab installed, you would be looking ')
            sys.stderr.write('at a plot of the evolution of the trust-region ')
            sys.stderr.write('radius, right now.\n')
            sys.exit(0)
        radii = numpy.array(TRNK.radii, 'd')
        pylab.plot(numpy.where(radii < 100, radii, 100))
        pylab.title('Trust-region radius')
        pylab.show()

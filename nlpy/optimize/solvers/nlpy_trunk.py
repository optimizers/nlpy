#!/usr/bin/env python

from nlpy.model import amplpy
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionCG as TRSolver
from nlpy.optimize.solvers import TrunkFramework as solver
#from nlpy.optimize.solvers import TrunkLbfgsFramework as solver
from nlpy.tools.timing import cputime
import numpy
import sys

def pass_to_trunk(nlp, showbanner=True):

    if nlp.nbounds > 0 or nlp.m > 0:         # Check for unconstrained problem
        sys.stderr.write('%s has %d bounds and %d general constraints\n' % (ProblemName, nlp.nbounds, nlp.m))
        return None

    t = cputime()
    tr = TR(Delta=1.0, eta1=0.05, eta2=0.9, gamma1=0.25, gamma2=2.5)

    # When instantiating TrunkFramework of TrunkLbfgsFramework,
    # we select a trust-region subproblem solver of our choice.
    TRNK = solver(nlp, tr, TRSolver, silent=False, ny=True, inexact=True)
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

    # Output final statistics
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
    return TRNK

if len(sys.argv) < 2:
    sys.stderr.write('Please specify model name\n')
    sys.exit(-1)

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

for ProblemName in sys.argv[1:]:
    nlp = amplpy.AmplModel(ProblemName)         # Create a model
    TRNK = pass_to_trunk(nlp, showbanner=True)
    #nlp.writesol(TRNK.x, nlp.pi0, 'And the winner is')    # Output "solution"
    nlp.close()                                 # Close connection with model

# Plot the evolution of the trust-region radius on the last problem
if TRNK is not None:
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

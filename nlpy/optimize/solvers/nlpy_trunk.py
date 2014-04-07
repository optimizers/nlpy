#!/usr/bin/env python

from nlpy.model import QuasiNewtonModel, AmplModel, LBFGS
from nlpy.optimize.tr.trustregion import TrustRegion
from nlpy.krylov.pcg import TruncatedCG
from nlpy.optimize.solvers.trunk import Trunk, QNTrunk
from nlpy.tools.timing import cputime
from argparse import ArgumentParser
import logging
import numpy
import sys


class QuasiNewtonAmplModel(QuasiNewtonModel, AmplModel):
  # All the work is done by the parent classes.
  pass


def pass_to_trunk(nlp, options, showbanner=True):

    if nlp.nbounds + nlp.m > 0:         # Check for unconstrained problem
        msg = '%s has %d bounds and %d constraints' % (nlp.name,
                                                       nlp.nbounds, nlp.m)
        raise ValueError(msg)

    t = cputime()
    tr = TrustRegion(radius=1.0, eta1=0.05, eta2=0.9, gamma1=0.25, gamma2=2.5)

    TrunkClass = QNTrunk if options.lbfgs else Trunk
    trunk = TrunkClass(nlp, tr, TruncatedCG,
                       ny=not options.nony,
                       inexact=not options.exact,
                       logger_name='nlpy.trunk',
                       reltol=options.rtol, abstol=options.atol,
                       maxiter=options.maxiter)
    t_setup = cputime() - t

    if showbanner:
        print
        print '------------------------------------------'
        print 'Trunk: Solving problem %-s with parameters' % nlp.name
        hdr = 'eta1 = %-g  eta2 = %-g  gamma1 = %-g  gamma2 = %-g Delta0 = %-g'
        print hdr % (tr.eta1, tr.eta2, tr.gamma1, tr.gamma2, tr.Delta)
        print 'Set up problem in %f seconds' % t_setup
        print '------------------------------------------'
        print

    trunk.solve()
    return trunk

# Declare command-line arguments.
parser = ArgumentParser(description='A Newton/CG trust-region solver for' +
                                    ' unconstrained optimization')
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
parser.add_argument('--lbfgs', action='store_true',
                    help='Use L-BFGS approximation')
parser.add_argument('--npairs', action='store', type=int, default=5,
                    dest='npairs', help='Number of L-BFGS pairs')
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
multiple_probs = len(args) > 0

for name in args:
    if options.lbfgs:
      nlp = QuasiNewtonAmplModel(name, H=LBFGS, npairs=options.npairs)
    else:
      nlp = AmplModel(name)
    trunk = pass_to_trunk(nlp, options, showbanner=not multiple_probs)

# Output final statistics
if not multiple_probs:
    log.info('Final variables:' + repr(trunk.x))
    log.info('Trunk: End of Execution')
    log.info('  Problem                     : %-s' % name)
    log.info('  Dimension                   : %-d' % nlp.n)
    log.info('  Initial/Final Objective     : %-g/%-g' % (trunk.f0, trunk.f))
    log.info('  Initial/Final Gradient Norm : %-g/%-g' % (trunk.g0, trunk.gnorm))
    log.info('  Number of iterations        : %-d' % trunk.iter)
    log.info('  Number of function evals    : %-d' % trunk.nlp.feval)
    log.info('  Number of gradient evals    : %-d' % trunk.nlp.geval)
    log.info('  Number of Hessian  evals    : %-d' % trunk.nlp.Heval)
    log.info('  Number of matvec products   : %-d' % trunk.nlp.Hprod)
    log.info('  Total/Average CG iter       : %-d/%-g' % (trunk.total_cgiter,
                                                          (float(trunk.total_cgiter)/trunk.iter)))
    log.info('  Solve time                  : %-gs/%-gs' % trunk.tsolve)

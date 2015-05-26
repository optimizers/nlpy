#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import amplpy
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionBQP as TRSolver
from nlpy.optimize.solvers.sbmin import SBMINFramework, SBMINLqnFramework
from nlpy.tools.exceptions import GeneralConstraintsError
from nlpy.tools.timing import cputime
from nlpy.tools.logs import config_logger
import logging
from optparse import OptionParser
import numpy as np
import sys
import os


def pass_to_sbmin(nlp, **kwargs):

    if nlp.m > 0:         # Check for constrained problem
        msg = '%s has %d general constraints\n' % (ProblemName, nlp.m)
        raise GeneralConstraintsError(msg)

    qn = kwargs.get('quasi_newton', None)

    t = cputime()
    tr = TR(eta1=1.0e-4, eta2=0.9, gamma1=0.3, gamma2=2.5)
    if qn is None:
        sbmin = SBMINFramework(nlp, tr, TRSolver, **kwargs)
    else:
        sbmin = SBMINLqnFramework(nlp, tr, TRSolver, **kwargs)

    t_setup = cputime() - t
    sbmin.Solve()
    return (t_setup, sbmin)


if len(sys.argv) < 2:
    sys.stderr.write('Please specify model name\n')
    sys.exit(-1)

usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

parser.add_option("-m", "--monotone", action="store_true",
                  default=False, dest="monotone",
                  help="Enable monotone strategy")
parser.add_option("-y", "--nocedal_yuan", action="store_true",
                  default=False, dest="ny",
                  help="Enable Nocedal-Yuan backtracking strategy")
parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxit",  help="Specify maximum number of iterations")
parser.add_option("-p", "--print_level", action="store", type="int",
                  default=0, dest="print_level",
                  help="Print iterations detail (0, 1 or 2)")
parser.add_option("-r", "--plot_radi", action="store_true",
                  default=False, dest="plot_radi",
                  help="Plot the evolution of the trust-region radius")
parser.add_option("-q", "--quasi_newton", action="store", type="string",
                  default=None, dest="quasi_newton",
                  help="LBFGS or LSR1")


# Parse command-line options:
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxit is not None:
    opts['maxiter'] = options.maxit
opts['ny'] = options.ny
opts['monotone'] = options.monotone
opts['print_level'] = options.print_level
opts['plot_radi'] = options.plot_radi
opts['quasi_newton'] = options.quasi_newton

# Set printing standards for arrays
np.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

multiple_problems = len(args) > 1
error = False
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')

# Create root logger.
nlpylogger = logging.getLogger('root')
nlpylogger.setLevel(logging.INFO)
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
nlpylogger.addHandler(hndlr)

if multiple_problems:

    nlpylogger.propagate = False

    # Configure subproblem logger.
    config_logger('nlpy.sbmin',
                  filename='sbmin.log',
                  filemode='w',
                  stream=None,
                  propagate=False)

    # Define formats for output table.
    hdrfmt = '%-10s %5s %5s %5s %15s %6s %6s'

    hdr = hdrfmt % ('Name', 'n', 'Iter', 'Hprod', 'Objective', 'Solve', 'Time')
    lhdr = len(hdr)
    fmt = '%-10s %5d %5d %5d %15.8e %6s %6.2f'
    nlpylogger.info(hdr)
    nlpylogger.info('-' * lhdr)

else:

    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)

    # Configure sbmin logger.
    sbminlogger = logging.getLogger('nlpy.sbmin')
    sbminlogger.addHandler(hndlr)
    sbminlogger.propagate = False
    sbminlogger.setLevel(logging.INFO)
    if options.print_level >= 4:
        sbminlogger.setLevel(logging.DEBUG)

    # Configure bqp logger.
    if options.print_level >= 2:
        bqplogger = logging.getLogger('nlpy.bqp')
        bqplogger.setLevel(logging.INFO)
        bqplogger.addHandler(hndlr)
        bqplogger.propagate = False
        if options.print_level >= 4:
            bqplogger.setLevel(logging.DEBUG)

    # Configure pcg logger
    if options.print_level >= 3:
        pcglogger = logging.getLogger('nlpy.pcg')
        pcglogger.setLevel(logging.INFO)
        pcglogger.addHandler(hndlr)
        pcglogger.propagate = False
        if options.print_level >= 4:
            pcglogger.setLevel(logging.DEBUG)


def apply_scaling(nlp):
    "Apply scaling to the NLP and print something if asked."
    gNorm = nlp.compute_scaling_obj()

# Solve each problem in turn.
for ProblemName in args:
    nlp = amplpy.MFAmplModel(ProblemName)         # Create a model
    apply_scaling(nlp)

    t = cputime()
    t_setup, SBMIN = pass_to_sbmin(nlp, **opts)
    total_time = cputime() - t

    if multiple_problems:
        problemName = os.path.splitext(os.path.basename(ProblemName))[0]
        nlpylogger.info(fmt % (problemName, nlp.n, SBMIN.iter, SBMIN.nlp.Hprod,
                               SBMIN.f, repr(SBMIN.status), total_time))
    nlp.close()

if not multiple_problems and not error:
    # Output final statistics
    sbminlogger.info('')
    sbminlogger.info('Final variables: %-s' % repr(SBMIN.x))
    sbminlogger.info('')
    sbminlogger.info('-------------------------------')
    sbminlogger.info('SBMIN: End of Execution')
    sbminlogger.info('  Problem                               : %-s' % ProblemName)
    sbminlogger.info('  Dimension                             : %-d' % nlp.n)
    sbminlogger.info('  Initial/Final Objective               : %-g/%-g' % (SBMIN.f0, SBMIN.f))
    sbminlogger.info('  Initial/Final Projected Gradient Norm : %-g/%-g' % (SBMIN.pg0, SBMIN.pgnorm))
    sbminlogger.info('  Number of iterations        : %-d' % SBMIN.iter)
    sbminlogger.info('  Number of function evals    : %-d' % SBMIN.nlp.feval)
    sbminlogger.info('  Number of gradient evals    : %-d' % SBMIN.nlp.geval)
    sbminlogger.info('  Number of Hessian  evals    : %-d' % SBMIN.nlp.Heval)
    sbminlogger.info('  Number of Hessian matvecs   : %-d' % SBMIN.nlp.Hprod)
    sbminlogger.info('  Total/Average BQP iter      : %-d/%-g' % (SBMIN.total_bqpiter, (float(SBMIN.total_bqpiter)/SBMIN.iter)))
    sbminlogger.info('  Setup/Solve time            : %-gs/%-gs' % (t_setup, SBMIN.tsolve))
    sbminlogger.info('  Total time                  : %-gs' % (total_time))
    sbminlogger.info('  Status                      : %-s', SBMIN.status)
    sbminlogger.info('-------------------------------')
    sbminlogger.info('  Number of Hprod in BQP')
    sbminlogger.info('     - linesearch              : %-d' % SBMIN.hprod_bqp_linesearch)
    sbminlogger.info('     - # linesearch            : %-d' % SBMIN.nlinesearch)
    sbminlogger.info('     - linesearch mean         : %-g' % (float(SBMIN.hprod_bqp_linesearch)/SBMIN.nlinesearch))
    sbminlogger.info('     - cg                      : %-d' % SBMIN.hprod_bqp_cg)

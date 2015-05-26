#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import amplpy
from nlpy.optimize.solvers.bqp2 import BQP
from nlpy.tools.timing import cputime
from nlpy.tools.logs import config_logger
from nlpy.tools.exceptions import GeneralConstraintsError

import logging
from optparse import OptionParser
import numpy as np
import sys
import os


def pass_to_bqp(nlp, **kwargs):

    if nlp.m > 0:         # Check for constrained problem
        msg = '%s has %d general constraints\n' % (ProblemName, nlp.m)
        raise GeneralConstraintsError(msg)

    t = cputime()
    bqp = BQP(nlp, **kwargs)

    t_setup = cputime() - t                  # Setup time
    t = cputime()
    bqp.solve()
    tsolve = cputime() - t
    return (t_setup, tsolve, bqp)


if len(sys.argv) < 2:
    sys.stderr.write('Please specify model name\n')
    sys.exit(-1)

usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxit",  help="Specify maximum number of iterations")
parser.add_option("-p", "--print_level", action="store", type="int",
                  default=0, dest="print_level",
                  help="Print iterations detail (0, 1 or 2)")

# Parse command-line options:
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxit is not None:
    opts['maxiter'] = options.maxit
opts['print_level'] = options.print_level

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
    config_logger('nlpy.bqp',
                  filename='bqp.log',
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

    # Configure bqp logger.
    bqplogger = logging.getLogger('nlpy.bqp')
    bqplogger.addHandler(hndlr)
    bqplogger.propagate = False
    bqplogger.setLevel(logging.INFO)

    # Configure pcg logger
    if options.print_level >= 2:
        pcglogger = logging.getLogger('nlpy.pcg')
        pcglogger.setLevel(logging.INFO)
        pcglogger.addHandler(hndlr)
        pcglogger.propagate = False
        if options.print_level >= 4:
            pcglogger.setLevel(logging.DEBUG)


# Solve each problem in turn.
for ProblemName in args:
    nlp = amplpy.MFAmplModel(ProblemName)         # Create a model

    t = cputime()
    t_setup, tsolve, bqp = pass_to_bqp(nlp, **opts)
    total_time = cputime() - t

    if multiple_problems:
        problemName = os.path.splitext(os.path.basename(ProblemName))[0]
        nlpylogger.info(fmt % (problemName, nlp.n, bqp.iter, bqp.nlp.Hprod,
                               bqp.f, repr(bqp.status), total_time))
    nlp.close()

if not multiple_problems and not error:
    # Output final statistics
    bqplogger.info('')
    bqplogger.info('Final variables: %-s' % repr(bqp.x))
    bqplogger.info('')
    bqplogger.info('-------------------------------')
    bqplogger.info('bqp: End of Execution')
    bqplogger.info('  Problem                               : %-s' % ProblemName)
    bqplogger.info('  Dimension                             : %-d' % nlp.n)
    bqplogger.info('  Initial/Final Objective               : %-g/%-g' % (bqp.qval0, bqp.qval))
    bqplogger.info('  Initial/Final Projected Gradient Norm : %-g/%-g' % (bqp.pg0, bqp.pgNorm))
    bqplogger.info('  Number of iterations        : %-d' % bqp.niter)
    bqplogger.info('  Number of function evals    : %-d' % bqp.qp.feval)
    bqplogger.info('  Number of gradient evals    : %-d' % bqp.qp.geval)
    bqplogger.info('  Number of Hessian  evals    : %-d' % bqp.qp.Heval)
    bqplogger.info('  Number of Hessian matvecs   : %-d' % bqp.qp.Hprod)
    bqplogger.info('  Setup/Solve time            : %-gs/%-gs' % (t_setup, tsolve))
    bqplogger.info('  Total time                  : %-gs' % (total_time))
    bqplogger.info('  Status                      : %-s', bqp.status)
    bqplogger.info('-------------------------------')
    bqplogger.info('  Number of Hprod in BQP')
    bqplogger.info('     - linesearch              : %-d' % bqp.hprod_bqp_linesearch)
    bqplogger.info('     - # linesearch            : %-d' % bqp.nlinesearch)
    bqplogger.info('     - linesearch mean         : %-g' % (float(bqp.hprod_bqp_linesearch)/bqp.nlinesearch))
    bqplogger.info('     - cg                      : %-d' % bqp.hprod_bqp_cg)

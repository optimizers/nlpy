#!/usr/bin/env python
# Main driver for Elastic method.

from nlpy import __version__
from nlpy.model import AmplModel
from nlpy.optimize.solvers.elastic import ElasticInteriorFramework2 as EIF
from nlpy.tools.timing import cputime
from nlpy.tools.logs import config_logger
from math import log10
from optparse import OptionParser
import numpy as np
import logging
import sys
import os

def pass_to_elastic(nlp, **kwargs):

    verbose = (kwargs.get('print_level') >= 2)
    kwargs.pop('print_level')
    maxiter = kwargs.get('maxit',200)

    t = cputime()
    eif = EIF(nlp, maxiter=maxiter, **kwargs)
    t_setup = cputime() - t    # Setup time.

    if verbose:
        nlp.display_basic_info()

    try:
        eif.solve()
    except:
        pass

    return (t_setup, eif)


usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxit",  help="Specify maximum number of iterations")
parser.add_option("-s", "--solution", action="store_true",
                  dest="solution_requested", default=False,
                  help="Output solution file")
parser.add_option("-o", "--print_level", action="store", type="int",
                  default=0, dest="print_level",
                  help="Print iterations detail (0, 1, 2 or 3)")

# Parse command-line options
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxit is not None:
    opts['maxit'] = options.maxit
opts['print_level'] = options.print_level

# Create root logger.
log = logging.getLogger('elastic')
level = logging.DEBUG if options.print_level == 3 else logging.INFO
log.setLevel(level)
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

# Define a logger for the details of the interior-point data. Optional.
config_logger('elastic.barrier',
              filename='elastic-barrier.log',
              filemode='w',
              stream=None)
opts['barrier_logger_name'] = 'elastic.barrier'

#np.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

multiple_problems = len(args) > 1

if multiple_problems:
    # Define formats for output table.
    hdrfmt = '%-15s %5s %5s %15s %7s %7s %7s %5s %6s %6s %5s'
    hdr = hdrfmt % ('Name','Iter','Feval','Objective','dResid','pResid','Comp',
                    'LogPen', 'Setup','Solve','Opt')
    lhdr = len(hdr)
    fmt = '%-15s %5d %5d %15.8e %7.1e %7.1e %7.1e %5.1f %6.2f %6.2f %5s'
    log.info(hdr)
    log.info('-' * lhdr)

for problemName in args:

    nlp = AmplModel(problemName)
    problemName = os.path.splitext(os.path.basename(problemName))[0]

    # Create solution logger (if requested).
    if options.solution_requested:
        solution_logger = config_logger('elastic.solution',
                                        filename=problemName+'.sol',
                                        filemode='w', stream=None)

    t_setup, eif = pass_to_elastic(nlp, **opts)
    l1 = eif.l1bar.l1
    max_penalty = max(l1.get_penalty_parameters())

    if not multiple_problems:  # Output final statistics
        print
        print 'Final variables:', eif.x
        print
        print '-------------------------------'
        print 'Funnel: End of Execution'
        print '  Problem                     : %-s' % nlp.name
        print '  Dimension                   : %-d' % nlp.n
        print '  Final Objective             : %-g' % eif.f
        print '  Number of iterations        : %-d' % eif.niter
        print '  Number of function evals    : %-d' % nlp.feval
        print '  Number of gradient evals    : %-d' % nlp.geval
        print '  Number of Hessian  evals    : %-d' % nlp.Heval
        print '  Number of matvec products   : %-d' % nlp.Hprod
        print '  Number of constraints evals : %-d' % nlp.ceval
        print '  Number of Jacobian evals    : %-d' % nlp.Jeval
        print '  Setup/Solve time            : %-gs/%-gs' % (t_setup,eif.tsolve)
        print '  Total time                  : %-gs' % (t_setup + eif.tsolve)
        print '-------------------------------'

    if multiple_problems:
        niter = eif.niter ; feval = nlp.feval ; tsolve = eif.tsolve
        if not eif.optimal:  # Indicate failures.
            niter = -niter ; feval = -feval
            t_setup = -t_setup ; tsolve = -tsolve
        log.info(fmt % (problemName, niter, feval, eif.f,
                        eif.dResid, eif.pResid, eif.cResid,
                        log10(max_penalty),
                        t_setup, tsolve, eif.optimal))

    if options.solution_requested:
        # Output solution.
        solution_logger.info('# Problem %s' % problemName)
        solution_logger.info('# Primal variables.')
        solution_logger.info(eif.x)
        solution_logger.info('# Lagrange multipliers for general constraints.')
        solution_logger.info(eif.y_nlp)
        solution_logger.info('# Lagrange multipliers for bound constraints.')
        solution_logger.info(eif.z)

    # Terminate.
    nlp.close()

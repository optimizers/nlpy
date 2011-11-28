#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import AmplModel
from nlpy.optimize.solvers.funnel import Funnel, LSTRFunnel, LDFPFunnel, \
                                         StructuredLDFPFunnel
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
import nlpy.tools.logs
import sys, logging, os

# Create root logger.
log = logging.getLogger('funnel')
log.setLevel(logging.INFO)
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

# Configure the solver logger.
sublogger = logging.getLogger('funnel.solver')
sublogger.setLevel(logging.INFO)
sublogger.addHandler(hndlr)
sublogger.propagate = False


def pass_to_funnel(nlp, **kwargs):

    verbose = (kwargs['print_level'] == 2)
    qn = kwargs.pop('quasi_newton')

    t = cputime()
    if qn:
        funn = StructuredLDFPFunnel(nlp, logger_name='funnel.solver', **kwargs)
        # funn = LDFPFunnel(nlp, logger_name='funnel.solver', **kwargs)
    else:
        funn = Funnel(nlp, logger_name='funnel.solver', **kwargs)
    t_setup = cputime() - t    # Setup time.

    funn.solve() #reg=1.0e-8)

    return (t_setup, funn)


usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent unconstrained nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

parser.add_option("-p", "--stop_p", action="store", type="float",
                  default=1.0e-5, dest="stop_p",
                  help="Primal stopping tolerance")
parser.add_option("-d", "--stop_d", action="store", type="float",
                  default=1.0e-5, dest="stop_d",
                  help="Dual stopping tolerance")
parser.add_option("-q", "--quasi_newton", action="store_true",
                  default=False, dest="quasi_newton",
                  help="Use LDFP approximation of Hessian")
parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxit",  help="Specify maximum number of iterations")
parser.add_option("-o", "--print_level", action="store", type="int",
                  default=0, dest="print_level",
                  help="Print iterations detail (0, 1 or 2)")

# Parse command-line options
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxit is not None:
    opts['maxit'] = options.maxit
opts['stop_p'] = options.stop_p
opts['stop_d'] = options.stop_d
opts['quasi_newton'] = options.quasi_newton
opts['print_level'] = options.print_level

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

multiple_problems = len(args) > 1
error = False

if multiple_problems:
    # Define formats for output table.
    hdrfmt = '%-10s %5s %5s %15s %7s %7s %6s %6s %5s'
    hdr = hdrfmt % ('Name','Iter','Feval','Objective','dResid','pResid',
                    'Setup','Solve','Opt')
    lhdr = len(hdr)
    fmt = '%-10s %5d %5d %15.8e %7.1e %7.1e %6.2f %6.2f %5s'
    log.info(hdr)
    log.info('-' * lhdr)

# Solve each problem in turn.
for ProblemName in args:

    nlp = AmplModel(ProblemName)

    # Check for equality-constrained problem.
    n_ineq = nlp.nlowerC + nlp.nupperC + nlp.nrangeC
    if nlp.nbounds > 0 or n_ineq > 0:
        msg = '%s has %d bounds and %d inequality constraints\n'
        log.error(msg % (nlp.name, nlp.nbounds, n_ineq))
        error = True
    else:
        ProblemName = os.path.basename(ProblemName)
        if ProblemName[-3:] == '.nl':
            ProblemName = ProblemName[:-3]
        t_setup, funn = pass_to_funnel(nlp, **opts)
        if multiple_problems:
            log.info(fmt % (ProblemName, funn.niter, funn.nlp.feval, funn.f,
                            funn.dResid, funn.pResid,
                            t_setup, funn.tsolve, funn.optimal))
    nlp.close()  # Close model.

if not multiple_problems and not error:
    # Output final statistics
    log.info('--------------------------------')
    log.info('Funnel: End of Execution')
    log.info('  Problem                      : %-s' % ProblemName)
    log.info('  Number of variables          : %-d' % nlp.n)
    log.info('  Number of linear constraints : %-d' % nlp.nlin)
    log.info('  Number of general constraints: %-d' % (nlp.m - nlp.nlin))
    log.info('  Initial/Final Objective      : %-g/%-g' % (funn.f0, funn.f))
    log.info('  Number of iterations         : %-d' % funn.niter)
    log.info('  Number of function evals     : %-d' % funn.nlp.feval)
    log.info('  Number of gradient evals     : %-d' % funn.nlp.geval)
    #log.info('  Number of Hessian  evals     : %-d' % funn.nlp.Heval)
    log.info('  Number of Hessian matvecs    : %-d' % funn.nlp.Hprod)
    log.info('  Setup/Solve time             : %-gs/%-gs' % (t_setup, funn.tsolve))
    log.info('  Total time                   : %-gs' % (t_setup + funn.tsolve))
    log.info('--------------------------------')

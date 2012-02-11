#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import SlackFramework
from nlpy.optimize.solvers.cqp import RegQPInteriorPointSolver
from nlpy.optimize.solvers.cqp import RegQPInteriorPointSolver3x3
from nlpy.tools.norms import norm2
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
import os
import sys
import logging

# Create root logger.
log = logging.getLogger('cqp')
log.setLevel(logging.INFO)
fmt = logging.Formatter('%(name)-10s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

# Configure the solver logger.
sublogger = logging.getLogger('cqp.solver')
sublogger.setLevel(logging.DEBUG)
sublogger.addHandler(hndlr)
sublogger.propagate = False

usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent convex quadratic programs."""

# Define formats for output table.
hdrfmt = '%-15s  %5s  %15s  %7s  %7s  %7s  %6s  %6s  %4s'
hdr = hdrfmt % ('Name', 'Iter', 'Objective', 'pResid', 'dResid',
                'Gap', 'Setup', 'Solve', 'Stat')
fmt = '%-15s  %5d  %15.8e  %7.1e  %7.1e  %7.1e  %6.2f  %6.2f  %4s'

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

# File name options
parser.add_option("-i", "--iter", action="store", type="int", default=None,
        dest="maxiter",  help="Specify maximum number of iterations")
parser.add_option("-t", "--tol", action="store", type="float", default=None,
        dest="tol", help="Specify relative stopping tolerance")
parser.add_option("-p", "--regpr", action="store", type="float", default=None,
        dest="regpr", help="Specify initial primal regularization parameter")
parser.add_option("-d", "--regdu", action="store", type="float", default=None,
        dest="regdu", help="Specify initial dual regularization parameter")
parser.add_option("-S", "--no-scale", action="store_true",
        dest="no_scale", default=False, help="Turn off problem scaling")
parser.add_option("-3", "--3x3", action="store_true",
        dest="sys3x3", default=False, help="Use 3x3 block linear system")
parser.add_option("-l", "--long-step", action="store_true", default=False,
        dest="longstep", help="Use long-step method")
parser.add_option("-f", "--assume-feasible", action="store_true",
        default=False, dest="assume_feasible",
        help="Deactivate infeasibility check")
parser.add_option("-V", "--verbose", action="store_true", default=False,
        dest="verbose", help="Set verbose mode")

# Parse command-line options
(options, args) = parser.parse_args()

# Decide which class to instantiate.
if options.sys3x3:
    print 'Brave man! Using 3x3 block system!'
    Solver = RegQPInteriorPointSolver3x3
else:
    Solver = RegQPInteriorPointSolver

opts_init = {}
if options.regpr is not None:
    opts_init['regpr'] = options.regpr
if options.regdu is not None:
    opts_init['regdu'] = options.regdu

opts_solve = {}
if options.maxiter is not None:
    opts_solve['itermax'] = options.maxiter
if options.tol is not None:
    opts_solve['tolerance'] = options.tol

# Set printing standards for arrays.
numpy.set_printoptions(precision=3, linewidth=70, threshold=10, edgeitems=2)

multiple_problems = len(args) > 1

if not options.verbose:
    log.info(hdr)
    log.info('-'*len(hdr))

for probname in args:

    t_setup = cputime()
    qp = SlackFramework(probname)
    t_setup = cputime() - t_setup

    # isqp() should be implemented in the near future.
    #if not qp.isqp():
    #    log.info('Problem %s is not a quadratic program\n' % probname)
    #    qp.close()
    #    continue

    # Pass problem to RegQP.
    regqp = Solver(qp,
                   scale=not options.no_scale,
                   verbose=options.verbose,
                   **opts_init)

    regqp.solve(PredictorCorrector=not options.longstep,
                check_infeasible=not options.assume_feasible,
                **opts_solve)

    # Display summary line.
    probname=os.path.basename(probname)
    if probname[-3:] == '.nl': probname = probname[:-3]

    if not options.verbose:
        log.info(fmt % (probname, regqp.iter, regqp.obj_value,
                        regqp.pResid, regqp.dResid, regqp.rgap,
                        t_setup, regqp.solve_time, regqp.short_status))
        if regqp.short_status == 'degn':
            log.info(' F')  # Could not regularize sufficiently.

    qp.close()

log.info('-'*len(hdr))

if not multiple_problems:
    x = regqp.x[:qp.original_n]
    log.info('Final x: %s, |x| = %7.1e' % (repr(x),norm2(x)))
    log.info('Final y: %s, |y| = %7.1e' % (repr(regqp.y),norm2(regqp.y)))
    log.info('Final z: %s, |z| = %7.1e' % (repr(regqp.z),norm2(regqp.z)))

    log.info(regqp.status)
    log.info('#Iterations: %-d' % regqp.iter)
    log.info('RelResidual: %7.1e' % regqp.kktResid)
    log.info('Final cost : %21.15e' % regqp.obj_value)
    log.info('Setup time : %6.2fs' % t_setup)
    log.info('Solve time : %6.2fs' % regqp.solve_time)

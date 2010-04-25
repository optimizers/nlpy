#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import SlackFramework
from nlpy.optimize.solvers.lp import RegLPInteriorPointSolver
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
import os
import sys

usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent linear programs."""

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
parser.add_option("-s", "--no-stabilize", action="store_true",
        dest="no_stabilize", default=False, help="Turn off stabilization")
parser.add_option("-S", "--no-scale", action="store_true",
        dest="no_scale", default=False, help="Turn off problem scaling")
parser.add_option("-l", "--long-step", action="store_false", default=False,
        dest="longstep", help="Use long-step method")
parser.add_option("-V", "--verbose", action="store_true", default=False,
        dest="verbose", help="Set verbose mode")

# Parse command-line options
(options, args) = parser.parse_args()

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
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

if not options.verbose:
    sys.stderr.write(hdr + '\n' + '-'*len(hdr) + '\n')

for probname in args:

    t_setup = cputime()
    lp = SlackFramework(probname)
    t_setup = cputime() - t_setup

    islp = True
    if not lp.islp():
        sys.stderr.write('Problem %s is not a linear program\n' % probname)
        islp = False
        lp.close()
        continue

    # Pass problem to RegLP.
    reglp = RegLPInteriorPointSolver(lp,
                                     scale=not options.no_scale,
                                     stabilize=not options.no_stabilize,
                                     verbose=options.verbose,
                                     **opts_init)
    reglp.solve(PredictorCorrector=not options.longstep,
                **opts_solve)

    # Display summary line.
    probname=os.path.basename(probname)
    if probname[-3:] == '.nl': probname = probname[:-3]

    if not options.verbose:
        sys.stdout.write(fmt % (probname, reglp.iter, reglp.obj_value,
                                reglp.pResid, reglp.dResid, reglp.rgap,
                                t_setup, reglp.solve_time, reglp.short_status))
        if reglp.short_status == 'degn':
            sys.stdout.write(' F')  # Could not regularize sufficiently.
        sys.stdout.write('\n')

    lp.close()

if islp:
    if not options.verbose:
        sys.stderr.write('-'*len(hdr) + '\n')
    else:
        print 'Final x: ', reglp.x[:lp.original_n]
        print 'Final y: ', reglp.y
        print 'Final z: ', reglp.z

        sys.stdout.write('\n' + reglp.status + '\n')
        sys.stdout.write(' #Iterations: %-d\n' % reglp.iter)
        sys.stdout.write(' RelResidual: %7.1e\n' % reglp.kktResid)
        sys.stdout.write(' Final cost : %21.15e\n' % reglp.obj_value)
        sys.stdout.write(' Setup time : %6.2fs\n' % t_setup)
        sys.stdout.write(' Solve time : %6.2fs\n' % reglp.solve_time)

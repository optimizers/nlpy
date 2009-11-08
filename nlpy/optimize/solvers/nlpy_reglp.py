#!/usr/bin/env python

from nlpy.model import SlackFramework
from nlpy.optimize.solvers.lp import RegLPInteriorPointSolver
from nlpy.tools.timing import cputime
import numpy
import os
import sys

if len(sys.argv) < 2:
    sys.stderr.write('Use: %-s problem_name\n' % sys.argv[0])
    sys.stderr.write(' where problem_name represents a linear program\n')
    sys.exit(-1)

# Set printing standards for arrays.
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

# Define formats for output table.
hdrfmt = '%-15s  %5s  %15s  %7s  %7s  %7s  %6s  %6s'
hdr = hdrfmt % ('Name', 'Iter', 'Objective', 'pResid', 'dResid',
                'Gap', 'Setup', 'Solve')
fmt = '%-15s  %5d  %15.8e  %7.1e  %7.1e  %7.1e  %6.2f  %6.2f'

oneproblem = True
if len(sys.argv[1:]) > 1:
    oneproblem=False
    sys.stderr.write(hdr + '\n' + '-'*len(hdr) + '\n')

for probname in sys.argv[1:]:

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
    reglp = RegLPInteriorPointSolver(lp, stabilize=True, verbose=oneproblem)
    reglp.solve()

    # Display summary line.
    probname=os.path.basename(probname)
    if probname[-3:] == '.nl': probname = probname[:-3]

    if not oneproblem:
        sys.stdout.write(fmt % (probname, reglp.iter, reglp.obj_value,
                                reglp.pResid, reglp.dResid, reglp.rgap,
                                t_setup, reglp.solve_time))
        if reglp.status != 'Optimal solution found':
            sys.stdout.write(' F')  # Problem was not solved to optimality.
        sys.stdout.write('\n')

    lp.close()

if islp:
    if not oneproblem:
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

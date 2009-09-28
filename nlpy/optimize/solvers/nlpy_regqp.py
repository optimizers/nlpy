#!/usr/bin/env python

from nlpy.model import SlackFramework
from nlpy.optimize.solvers.cqp import RegQPInteriorPointSolver
import numpy
import os
import sys

if len(sys.argv) < 2:
    sys.stderr.write('Use: %-s problem_name\n' % sys.argv[0])
    sys.stderr.write(' where problem_name represents a convex')
    sys.stderr.write(' quadratic program\n')
    sys.exit(-1)

# Set printing standards for arrays.
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

# Define formats for output table.
hdrfmt = '%-15s  %5s  %15s  %7s  %7s  %7s  %6s'
hdr = hdrfmt % ('Name', 'Iter', 'Objective', 'pResid', 'dResid', 'Gap', 'Time')
fmt = '%-15s  %5d  %15.8e  %7.1e  %7.1e  %7.1e  %6.2f'

oneproblem = True
if len(sys.argv[1:]) > 1:
    oneproblem=False
    sys.stderr.write(hdr + '\n' + '-'*len(hdr) + '\n')

for probname in sys.argv[1:]:

    qp = SlackFramework(probname)

    # isqp() should be implemented in the near future.
    #if not qp.isqp():
    #    sys.stderr.write('Problem %s is not a linear program\n' % probname)
    #    qp.close()
    #    continue

    # Pass problem to RegQP.
    regqp = RegQPInteriorPointSolver(qp, scale=False, verbose=oneproblem)
    regqp.solve()

    # Display summary line.
    probname=os.path.basename(probname)
    if probname[-3:] == '.nl': probname = probname[:-3]

    if not oneproblem:
        sys.stdout.write(fmt % (probname, regqp.iter, regqp.obj_value,
                                regqp.pResid, regqp.dResid, regqp.rgap,
                                regqp.solve_time))
        if regqp.status != 'Optimal solution found':
            sys.stdout.write(' F')  # Problem was not solved to optimality.
        sys.stdout.write('\n')

    qp.close()

if not oneproblem:
    sys.stderr.write('-'*len(hdr) + '\n')
else:
    print 'Final x: ', regqp.x[:qp.original_n]
    print 'Final y: ', regqp.y
    print 'Final z: ', regqp.z

    sys.stdout.write('\n' + regqp.status + '\n')
    sys.stdout.write(' #Iterations: %-d\n' % regqp.iter)
    sys.stdout.write(' RelResidual: %7.1e\n' % regqp.kktResid)
    sys.stdout.write(' Final cost : %21.15e\n' % regqp.obj_value)
    sys.stdout.write(' Solve time : %6.2fs\n' % regqp.solve_time)

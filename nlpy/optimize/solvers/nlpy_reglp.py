#!/usr/bin/env python

from nlpy.model import SlackFramework
from nlpy.optimize.solvers.lp import RegLPInteriorPointSolver
import numpy
import sys

if len(sys.argv) < 2:
    sys.stderr.write('Use: %-s problem_name\n' % sys.argv[0])
    sys.stderr.write(' where problem_name represents a linear program\n')
    sys.exit(-1)

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

probname = sys.argv[1]

lp = SlackFramework(probname)
if not lp.islp():
    sys.stderr.write('Input problem must be a linear program\n')
    sys.exit(1)

reglp = RegLPInteriorPointSolver(lp)
reglp.solve(itermax=100, tolerance=1.0e-6, PredictorCorrector=True)

print 'Final x: ', reglp.x
print 'Final y: ', reglp.y
print 'Final z: ', reglp.z

sys.stdout.write('\n' + reglp.status + '\n')
sys.stdout.write(' #Iterations: %-d\n' % reglp.iter)
sys.stdout.write(' RelResidual: %7.1e\n' % reglp.kktResid)
sys.stdout.write(' Final cost : %21.15e\n' % reglp.obj_value)
sys.stdout.write(' Solve time : %6.2fs\n' % reglp.solve_time)

# End
lp.close()

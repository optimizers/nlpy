from nlpy.model import AmplModel, QPModel
from nlpy.krylov import SimpleLinearOperator
from nlpy.krylov import ProjectedCG as Ppcg
from nlpy.tools import norms
import numpy
import logging
import sys

if len(sys.argv) < 2:
        print 'This demo reads the nl file of an AMPL-decoded model and'
        print 'demonstrates the projected conjugate gradient by solving the'
        print 'following quadratic program with trust-region constraint:'
        print '     min   <g, x> + 1/2 <x, H x>'
        print '     s.t.  J d = 0,  |d| <= radius,'
        print
        print ' where g = grad(x0), H = Hess(x0), J = jac(x0), radius = |g|/10'
        print 'Please supply a constrained problem as input'
        sys.exit(0)

problem = sys.argv[1]
nlp = AmplModel(problem)

if nlp.m == 0:
    sys.stderr.write('Problem should have general constraints\n')
    nlp.close()
    sys.exit(1)

x0 = numpy.ones(nlp.n) #nlp.x0            # Initial point
g  = numpy.random.random(nlp.n) #nlp.grad(x0)      # Objective gradient
J  = nlp.jac(x0)       # Obtain constraints Jacobian
H  = SimpleLinearOperator(nlp.n, nlp.n,
                          lambda p: nlp.hprod(nlp.x0,nlp.pi0,p),
                          symmetric=True)
delta = 1000 #max(1.0, 100 * norms.norm_infty(g))

# You can test constraints with a nonzero rhs by uncommenting the following,
# adding the 'rhs' and removing the 'radius' keywords in Ppcg below.
# Also make sure to adjust the computation of 'feas' at the bottom of the file.
#e = numpy.ones(nlp.n)
#c = numpy.empty(nlp.m, 'd')
#J.matvec(e, c)

# Create root logger.
rootlog = logging.getLogger('PPCG')
rootlog.setLevel(logging.INFO)
fmt = logging.Formatter('%(name)-10s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
rootlog.addHandler(hndlr)

# Create solver logger.
log = logging.getLogger('PPCG.ppcg')
log.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(name)-10s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

# Call projected CG to solve problem
#   min  <g,d> + 1/2 <d, Hd>
#   s.t  Jd = 0,  |d| <= delta
qp = QPModel(g, H, A=J)
CG = Ppcg(qp, radius=delta, logger_name='PPCG.ppcg')
#CG = Ppcg(g, A=J, rhs=-c, matvec=hprod, debug=True)
CG.Solve()

Jd = numpy.empty(nlp.m, 'd')
J.matvec(CG.x, Jd)
feas = norms.norm_infty(Jd)
#feas = norms.norm_infty(c + Jd)  # for nonzero rhs constraint
snorm = norms.norm2(CG.x) # The trust-region is in the l2 norm

rootlog.info('Number of variables      : %-d' % nlp.n)
rootlog.info('Number of constraints    : %-d' % nlp.m)
rootlog.info('Converged                : %-s' % repr(CG.converged))
rootlog.info('Trust-region radius      : %7.1e' % delta)
rootlog.info('Solution norm            : %7.1e' % snorm)
rootlog.info('Final/Initial Residual   : ' + \
             '%7.1e / %7.1e' % (CG.residNorm,CG.residNorm0))
rootlog.info('Feasibility error        : %7.1e' % feas)
rootlog.info('Number of iterations     : %-d' % CG.iter)
